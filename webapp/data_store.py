"""Two-year warehouse store with incremental, overlap-safe refresh.

WHY NOT JUST RE-PULL

Two years across six servers is roughly eight times the 90-day extract -- far
too much to re-query whenever the dashboard wants fresh numbers, both in time
and in BigQuery bytes. So history is kept locally, partitioned by month, and
each refresh queries only what is new.

THE OVERLAP, AND WHY IT IS NOT OPTIONAL

A naive "fetch everything after the last watermark" loses rows. Trades are
written to the warehouse when they CLOSE, so a position opened last week and
closed today appears with a recent partition but an old open time; worse,
records get amended after the fact (corrections, late settlement). Re-reading a
few days behind the watermark and de-duplicating on the trade's identity keeps
those, at a cost proportional to the overlap rather than the history.

De-duplication keeps the LAST occurrence of each id, so an amended row replaces
the original rather than sitting beside it.

PARTITION LAYOUT

    warehouse/<database>/<YYYY-MM>.parquet

Monthly files let a query touch only the months it needs, keep any single
rewrite small, and mean a refresh rewrites at most the current month and the
one before it.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
WAREHOUSE = ROOT / "warehouse"
WAREHOUSE.mkdir(exist_ok=True)

#: Default history for every model and screen.
DEFAULT_HISTORY_DAYS = 730

#: THE SIX LIVE REAL-MONEY SERVERS, and where each one's data comes from.
#:
#: Five are in the MySQL instance behind `ld4-dbproxy` and are the system of
#: record. The sixth, `mt5_dubai_live01`, is NOT behind the proxy: it is read
#: from its own MariaDB host (`server.yaml` carries its host and port). It is
#: also replicated into BigQuery by the Traze backoffice pipeline
#: (`dubai_backfill`), which remains the fallback path.
#:
#: This list exists because the warehouse previously WAS the definition of
#: "all servers" -- code iterated the directories on disk. A server with no
#: directory therefore vanished from every model, every P&L total and every
#: exposure figure without appearing anywhere as missing. Firm P&L was
#: understated by one whole live server and nothing on the site said so.
#: Coverage is now measured against this list, so a gap is reported rather
#: than silently absorbed.
LIVE_SERVERS = {
    "mt4_live01": "mysql",
    "mt4_live02": "mysql",
    "mt4_live03": "mysql",
    "mt4_live04": "mysql",
    "mt5_live01": "mysql",
    "mt5_dubai_live01": "mysql",
}


def coverage(history_days: int = DEFAULT_HISTORY_DAYS) -> dict:
    """Which live servers the warehouse actually holds, and which it does not.

    Reads parquet footer metadata only -- row counts and file names -- so it is
    cheap enough to render on every page load.
    """
    import pyarrow.parquet as pq

    # Months the window should contain. A server is not "covered" merely because
    # it has some files: an interrupted backfill leaves a hole in the middle of
    # the range, and a presence-only check reports that as healthy. One such
    # interruption removed nine of mt5_live01's twenty-five months -- including
    # the most recent and busiest -- while coverage still read "present".
    today = pd.Timestamp.utcnow().tz_localize(None).to_period("M")
    earliest = (pd.Timestamp.utcnow().tz_localize(None)
                - pd.Timedelta(days=history_days)).to_period("M")
    wanted = pd.period_range(earliest, today, freq="M")

    servers, missing, incomplete, total = [], [], [], 0
    for name, source in LIVE_SERVERS.items():
        directory = WAREHOUSE / name
        files = sorted(directory.glob("*.parquet")) if directory.is_dir() else []
        rows = 0
        for path in files:
            try:
                rows += pq.ParquetFile(path).metadata.num_rows
            except Exception:
                continue
        total += rows

        held = set()
        for path in files:
            try:
                held.add(pd.Period(path.stem, freq="M"))
            except (ValueError, TypeError):
                continue
        # A server that genuinely started mid-window (dubai began 2025-02) is not
        # missing the months before it existed, so gaps are only counted from
        # the first month it actually holds.
        start = min(held) if held else None
        expected = [m for m in wanted if start is None or m >= start]
        gaps = sorted(str(m) for m in expected if m not in held)

        entry = {
            "server": name,
            "source": source,
            "present": bool(files),
            "files": len(files),
            "rows": rows,
            "first": files[0].stem if files else None,
            "last": files[-1].stem if files else None,
            "gaps": gaps,
            "gap_count": len(gaps),
            "complete": bool(files) and not gaps,
        }
        servers.append(entry)
        if not files:
            missing.append(name)
        elif gaps:
            incomplete.append(name)

    return {
        "servers": servers,
        "missing": missing,
        "incomplete": incomplete,
        "expected": len(LIVE_SERVERS),
        "present": len(LIVE_SERVERS) - len(missing),
        "complete": len(LIVE_SERVERS) - len(missing) - len(incomplete),
        "rows": total,
    }

#: How far behind the watermark each refresh re-reads. Three days covers
#: same-week amendments and late-closing positions; longer costs more bytes for
#: rapidly diminishing returns.
OVERLAP_DAYS = 3

#: Identity used for de-duplication, per platform. MT4's ticket is the position;
#: MT5's deal id is unique per deal.
IDENTITY_COLUMNS = ("database", "order")


@dataclass
class RefreshPlan:
    database: str
    start: datetime
    end: datetime
    reason: str
    estimated_bytes: int = 0

    @property
    def estimated_cost_usd(self) -> float:
        return self.estimated_bytes / 1e12 * 6.25


def _db() -> sqlite3.Connection:
    connection = sqlite3.connect(ROOT / "app.db", check_same_thread=False, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute(
        "CREATE TABLE IF NOT EXISTS warehouse_state ("
        " database TEXT PRIMARY KEY, watermark REAL, rows INTEGER,"
        " last_refresh REAL, detail TEXT)")
    return connection


def watermark(database: str) -> datetime | None:
    """Newest record already stored for this database."""
    with _db() as connection:
        row = connection.execute(
            "SELECT watermark FROM warehouse_state WHERE database = ?", (database,)).fetchone()
    if row is None or row["watermark"] is None:
        return None
    return datetime.fromtimestamp(row["watermark"], tz=timezone.utc)


def set_watermark(database: str, value: datetime, rows: int, detail: str = "") -> None:
    with _db() as connection:
        connection.execute(
            "INSERT INTO warehouse_state (database, watermark, rows, last_refresh, detail)"
            " VALUES (?,?,?,?,?) ON CONFLICT(database) DO UPDATE SET"
            " watermark=excluded.watermark, rows=excluded.rows,"
            " last_refresh=excluded.last_refresh, detail=excluded.detail",
            (database, value.timestamp(), rows, time.time(), detail))
        connection.commit()


def store_state() -> list[dict]:
    with _db() as connection:
        rows = connection.execute("SELECT * FROM warehouse_state ORDER BY database").fetchall()
    return [{
        "database": r["database"],
        "watermark": datetime.fromtimestamp(r["watermark"], tz=timezone.utc).isoformat()
        if r["watermark"] else None,
        "rows": r["rows"],
        "last_refresh": datetime.fromtimestamp(r["last_refresh"], tz=timezone.utc).isoformat()
        if r["last_refresh"] else None,
        "age_days": round((time.time() - r["last_refresh"]) / 86400, 2) if r["last_refresh"] else None,
        "detail": r["detail"],
    } for r in rows]


def plan_refresh(database: str, history_days: int = DEFAULT_HISTORY_DAYS) -> RefreshPlan:
    """Decide what range to fetch: a full backfill, or the incremental tail."""
    now = datetime.now(timezone.utc)
    mark = watermark(database)
    if mark is None:
        return RefreshPlan(database, now - timedelta(days=history_days), now,
                           f"initial backfill of {history_days} days")
    start = mark - timedelta(days=OVERLAP_DAYS)
    return RefreshPlan(database, start, now,
                       f"incremental from {start:%Y-%m-%d} "
                       f"({OVERLAP_DAYS}-day overlap for amendments)")


def partition_path(database: str, period: str) -> Path:
    directory = WAREHOUSE / database
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{period}.parquet"


def write_partitions(database: str, frame: pd.DataFrame,
                     time_column: str = "close_time") -> dict:
    """Merge a fetched slice into the monthly partitions.

    Only the months the new data touches are rewritten, and each is merged with
    what is already there before de-duplication -- so an overlap re-read
    replaces amended rows instead of duplicating them.
    """
    if frame.empty:
        return {"written": 0, "months": []}

    stamps = pd.to_datetime(frame[time_column], errors="coerce")
    frame = frame.loc[stamps.notna()].copy()
    frame["_period"] = stamps.loc[stamps.notna()].dt.strftime("%Y-%m")

    written, months, repaired = 0, [], []
    for period, chunk in frame.groupby("_period", observed=True):
        path = partition_path(database, str(period))
        chunk = chunk.drop(columns=["_period"])
        combined = chunk
        if path.exists():
            try:
                existing = pd.read_parquet(path)
                combined = pd.concat([existing, chunk], ignore_index=True)
            except Exception as error:
                # A PARTITION THAT WILL NOT READ IS A DEAD END, not a crash.
                # Before this, an unreadable file raised here and the whole
                # refresh for the month died -- which meant the one operation
                # that could replace the bad file was the one thing the bad
                # file prevented. mt4_live02/2026-09.parquet sat truncated from
                # 17 Sep 04:01 (valid PAR1 header, no footer, 7.3 MB against
                # 25-36 MB for its neighbours) and every September realised P&L
                # on that server silently read as zero.
                # Keep the corpse for inspection, then rebuild the month from
                # what we just fetched.
                quarantine = path.with_suffix(f".parquet.corrupt-{int(time.time())}")
                try:
                    path.replace(quarantine)
                except Exception:
                    pass
                repaired.append({"month": str(period), "moved_to": quarantine.name,
                                 "error": f"{type(error).__name__}: {error}"})
                combined = chunk
        keys = [c for c in IDENTITY_COLUMNS if c in combined.columns]
        if keys:
            # keep="last": the freshly fetched row wins over the stored one, so
            # an amendment supersedes the original.
            combined = combined.drop_duplicates(subset=keys, keep="last")
        # ATOMIC REPLACE. `to_parquet(path)` wrote straight over the live file,
        # so anything that interrupted the write -- the process dying, OneDrive
        # taking the handle, a stalled disk -- left a half-written partition AND
        # destroyed the good copy, because there was no other copy. Write beside
        # it and rename: os.replace is atomic on Windows and POSIX, so a reader
        # sees either the whole old file or the whole new one, never a partial.
        temp = path.with_suffix(".parquet.tmp")
        combined.to_parquet(temp, index=False)
        temp.replace(path)
        written += len(chunk)
        months.append(str(period))
    out = {"written": written, "months": sorted(months)}
    if repaired:
        out["repaired"] = repaired
    return out


def read_history(databases: tuple[str, ...] | None = None,
                 start: datetime | None = None, end: datetime | None = None,
                 columns: list[str] | None = None) -> pd.DataFrame:
    """Read the stored history, touching only the months in range.

    Month-level pruning is the point of the layout: a 90-day view of a two-year
    store reads four files, not seven hundred days of rows.
    """
    wanted_periods = None
    if start is not None and end is not None:
        wanted_periods = {
            stamp.strftime("%Y-%m")
            for stamp in pd.date_range(start, end, freq="MS", inclusive="both")
        }
        wanted_periods.add(start.strftime("%Y-%m"))
        wanted_periods.add(end.strftime("%Y-%m"))

    parts = []
    for directory in sorted(WAREHOUSE.iterdir()):
        if not directory.is_dir():
            continue
        if databases and directory.name not in databases:
            continue
        for path in sorted(directory.glob("*.parquet")):
            if wanted_periods is not None and path.stem not in wanted_periods:
                continue
            try:
                parts.append(pd.read_parquet(path, columns=columns))
            except Exception:
                continue
    if not parts:
        return pd.DataFrame()

    frame = pd.concat(parts, ignore_index=True)
    if start is not None and "close_time" in frame.columns:
        stamps = pd.to_datetime(frame["close_time"], errors="coerce")
        naive_start = pd.Timestamp(start).tz_localize(None)
        naive_end = pd.Timestamp(end).tz_localize(None) if end else None
        mask = stamps >= naive_start
        if naive_end is not None:
            mask &= stamps <= naive_end
        frame = frame.loc[mask]
    return frame.reset_index(drop=True)


def store_summary() -> dict:
    """Size and span of what is on disk, for the admin screen."""
    databases = {}
    total_bytes, total_files = 0, 0
    for directory in sorted(WAREHOUSE.iterdir()):
        if not directory.is_dir():
            continue
        files = sorted(directory.glob("*.parquet"))
        if not files:
            continue
        size = sum(f.stat().st_size for f in files)
        total_bytes += size
        total_files += len(files)
        databases[directory.name] = {
            "months": len(files),
            "first_month": files[0].stem,
            "last_month": files[-1].stem,
            "size_mb": round(size / 1e6, 1),
        }
    return {"databases": databases, "total_mb": round(total_bytes / 1e6, 1),
            "files": total_files, "state": store_state()}
