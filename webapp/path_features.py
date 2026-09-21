"""Per-trade price-path excursions (MAE / MFE) for the training frame.

The entry, exit and drawdown models all need the same thing entry and exit
prices cannot give: how far price moved AGAINST and FOR a trade while it was
open. Minute bars come from mt4_live01's tick table -- a full year, and one
shared feed for every server (the XAUUSD tick counts are identical across the
MT4 servers) -- and excursions are computed for EVERY trade keyed on the
CANONICAL symbol, so MT4 / MT5 / cent trades all get a path from one feed.

SPEED: the bars are pulled with a HIGH/LOW-only query (the shared OHLC query in
tick_bars GROUP_CONCATs every tick to derive open/close, which excursions never
use), and cached INCREMENTALLY by day: the dataset is partitioned day/symbol
with a per-day marker recording the symbols it holds, so a nightly retrain
fetches only the new day instead of the whole window.

MEMORY: fetch in small day batches straight into the partitioned parquet
dataset; excursions are computed one symbol at a time, so the peak is one
batch during the fetch and one symbol's bars during the sweep.

Results are in basis points of entry price and direction-signed the way
tick_bars reports them: mae_bps <= 0 is adverse, mfe_bps >= 0 favourable. Bps
puts cent accounts and every instrument on one scale. Minute resolution is the
documented compromise (see tick_bars).
"""

from __future__ import annotations

import json
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PATH_DATABASE = "mt4_live01"      # the full-year feed shared by all servers
PATH_COLUMNS = ("mae_bps", "mfe_bps", "bars_seen")
BATCH_DAYS = 4                    # 4 days x two 12h chunks = one full 8-worker wave
WORKERS = 8

# High/low only: excursions need the bar's extremes, nothing else.
HL_SQL = """
SELECT symbol_name AS symbol,
       DATE_FORMAT(tm, '%%Y-%%m-%%d %%H:%%i:00') AS minute,
       MIN((bid + ask) / 2) AS low,
       MAX((bid + ask) / 2) AS high
FROM ticks
WHERE tm >= %s AND tm < %s
  AND symbol_name IN ({placeholders})
  AND bid > 0 AND ask > 0
GROUP BY symbol_name, minute
"""


def _range_extrema(values: np.ndarray, starts: np.ndarray, ends: np.ndarray,
                   reduce) -> np.ndarray:
    """Vectorised range-reduce (np.maximum / np.minimum) of values[s:e) for
    arrays of (s, e) with e > s, via a sparse table: O(n log n) build, O(1)
    per query, no Python loop over trades."""
    n = len(values)
    levels = [np.asarray(values, dtype=float)]
    j = 1
    while (1 << j) <= n:
        prev, half, width = levels[-1], 1 << (j - 1), 1 << j
        levels.append(reduce(prev[: n - width + 1], prev[half: half + n - width + 1]))
        j += 1
    length = ends - starts
    k = np.floor(np.log2(length)).astype(int)
    out = np.empty(len(starts), dtype=float)
    for level in np.unique(k):
        sel = k == level
        table = levels[level]
        out[sel] = reduce(table[starts[sel]], table[ends[sel] - (1 << level)])
    return out


# --------------------------------------------------------------- bar cache
def bars_root(scratch: Path, database: str = PATH_DATABASE) -> Path:
    return Path(scratch) / f"path_bars_{database}"


def _day_marker(root: Path, day: date) -> Path:
    return root / "_days" / f"{day.isoformat()}.json"


def _marker_symbols(marker: Path) -> set[str]:
    try:
        return set(json.loads(marker.read_text(encoding="utf-8")).get("symbols", []))
    except Exception:
        return set()


def _fetch_days(database: str, symbols: tuple[str, ...], days: list[date]) -> pd.DataFrame:
    """High/low minute bars for the given days: 12h chunks, in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from webapp import tick_bars

    sql = HL_SQL.format(placeholders=", ".join(["%s"] * len(symbols)))
    windows = []
    for day in days:
        d0 = datetime(day.year, day.month, day.day)
        windows.append((d0, d0 + timedelta(hours=12)))
        windows.append((d0 + timedelta(hours=12), d0 + timedelta(days=1)))
    frames = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(tick_bars._fetch_chunk, database, sql, symbols, w)
                   for w in windows]
        for future in as_completed(futures):
            chunk = future.result()
            if chunk is not None and not chunk.empty:
                frames.append(chunk)
    if not frames:
        return pd.DataFrame(columns=["symbol", "minute", "high", "low"])
    bars = pd.concat(frames, ignore_index=True)
    bars["minute"] = pd.to_datetime(bars["minute"])
    bars["high"] = pd.to_numeric(bars["high"], errors="coerce")
    bars["low"] = pd.to_numeric(bars["low"], errors="coerce")
    return bars.dropna(subset=["high", "low"]).drop_duplicates(["symbol", "minute"])


def _write_days(root: Path, bars: pd.DataFrame, days: list[date], symbols: set[str],
                replace: bool = True) -> None:
    """Write the given days into the partitioned dataset and stamp markers.
    replace=True rebuilds those days (fresh fetch of every symbol); replace=False
    appends only NEW symbol partitions (a delta fetch for symbols the marker
    had not seen), leaving existing partitions untouched."""
    if replace:
        for day in days:                   # replace, never append duplicates
            shutil.rmtree(root / f"day={day.isoformat()}", ignore_errors=True)
    if not bars.empty:
        bars = bars.assign(day=bars["minute"].dt.strftime("%Y-%m-%d"))
        bars[["day", "symbol", "minute", "high", "low"]].to_parquet(
            root, partition_cols=["day", "symbol"], engine="pyarrow", index=False)
    (root / "_days").mkdir(parents=True, exist_ok=True)
    for day in days:
        marker = _day_marker(root, day)
        merged = _marker_symbols(marker) | symbols if marker.exists() else set(symbols)
        marker.write_text(json.dumps({"symbols": sorted(merged)}), encoding="utf-8")


def ensure_window_bars(symbols: tuple[str, ...], start: datetime, end: datetime,
                       scratch: Path, log=None, database: str = PATH_DATABASE,
                       batch_days: int = BATCH_DAYS) -> Path | None:
    """Guarantee every day in [start, end) is cached for these symbols,
    fetching ONLY the missing days. Returns the dataset root."""
    root = bars_root(scratch, database)
    (root / "_days").mkdir(parents=True, exist_ok=True)
    want = set(symbols)
    days = [start.date() + timedelta(days=i)
            for i in range(max(0, (end.date() - start.date()).days))]
    # Two kinds of gap: a day never fetched (fetch every symbol), and a cached
    # day whose marker has not seen some REQUESTED symbols (fetch only that
    # delta and append). Markers record requested symbols, so an instrument
    # with no ticks on the feed can never make a cached day look missing.
    missing: list[date] = []
    delta: dict[frozenset, list[date]] = {}
    for d in days:
        marker = _day_marker(root, d)
        if not marker.exists():
            missing.append(d)
            continue
        unseen = want - _marker_symbols(marker)
        if unseen:
            delta.setdefault(frozenset(unseen), []).append(d)
    if log:
        n_delta = sum(len(v) for v in delta.values())
        log(f"{len(days) - len(missing) - n_delta}/{len(days)} days cached; "
            f"fetching {len(missing)} missing days"
            + (f" + a symbol delta on {n_delta} days" if n_delta else ""))
    for i in range(0, len(missing), batch_days):
        batch = missing[i:i + batch_days]
        bars = _fetch_days(database, symbols, batch)
        _write_days(root, bars, batch, want, replace=True)
        if log:
            log(f"fetched {min(i + batch_days, len(missing))}/{len(missing)} missing days "
                f"(+{len(bars):,} bars)")
        del bars
    for unseen, ddays in delta.items():
        sub = tuple(sorted(unseen))
        for i in range(0, len(ddays), batch_days):
            batch = ddays[i:i + batch_days]
            bars = _fetch_days(database, sub, batch)
            _write_days(root, bars, batch, want, replace=False)
            del bars
        if log:
            log(f"symbol delta: {len(sub)} new symbols over {len(ddays)} cached days")
    return root if any((root / "_days").glob("*.json")) else None


def migrate_window_dataset(old_root: Path, scratch: Path,
                           database: str = PATH_DATABASE, log=None) -> int:
    """One-off: fold a completed window-keyed dataset (symbol-partitioned,
    from the first run) into the incremental day/symbol cache so the next
    retrain fetches only new days. Returns rows migrated."""
    old_root = Path(old_root)
    if not (old_root / "_complete").exists():
        return 0
    root = bars_root(scratch, database)
    (root / "_days").mkdir(parents=True, exist_ok=True)
    total = 0
    symbols: set[str] = set()
    for part in sorted(old_root.glob("symbol=*")):
        symbol = part.name.split("=", 1)[1]
        symbols.add(symbol)
    # process day by day to bound memory: read everything once per symbol
    # would be simpler but holds a symbol-year in memory; per symbol is fine.
    per_day: dict[str, list[pd.DataFrame]] = {}
    for symbol in sorted(symbols):
        try:
            frame = pd.read_parquet(old_root, filters=[("symbol", "==", symbol)],
                                    columns=["minute", "high", "low"])
        except Exception:
            continue
        if frame.empty:
            continue
        frame["minute"] = pd.to_datetime(frame["minute"])
        frame["symbol"] = symbol
        frame["day"] = frame["minute"].dt.strftime("%Y-%m-%d")
        frame[["day", "symbol", "minute", "high", "low"]].to_parquet(
            root, partition_cols=["day", "symbol"], engine="pyarrow", index=False)
        for day in frame["day"].unique():
            per_day.setdefault(day, [])
        total += len(frame)
        if log:
            log(f"migrated {symbol}: {len(frame):,} bars")
        del frame
    for day in per_day:
        marker = _day_marker(root, date.fromisoformat(day))
        merged = _marker_symbols(marker) | symbols if marker.exists() else set(symbols)
        marker.write_text(json.dumps({"symbols": sorted(merged)}), encoding="utf-8")
    return total


def _open_dataset(root: Path):
    """Discover the day/symbol-partitioned dataset ONCE. With ~200 symbols x
    ~180 days the tree holds tens of thousands of part files; a per-symbol
    pd.read_parquet rediscovers all of them every call (12+ minutes for one
    sweep), whereas one discovery plus per-symbol filters is seconds. Marker
    files live under `_days/`, which pyarrow's default ignore-prefixes skip."""
    import pyarrow.dataset as pads
    return pads.dataset(str(root), format="parquet", partitioning="hive")


def _symbol_bars(dataset, symbol: str) -> pd.DataFrame:
    """One symbol's bars from the discovered dataset, minute-sorted."""
    import pyarrow.dataset as pads
    try:
        bars = dataset.to_table(filter=pads.field("symbol") == symbol,
                                columns=["minute", "high", "low"]).to_pandas()
    except Exception:
        return pd.DataFrame()
    if bars.empty:
        return bars
    bars["minute"] = pd.to_datetime(bars["minute"])
    return bars.sort_values("minute").drop_duplicates("minute").reset_index(drop=True)


# --------------------------------------------------------------- excursions
def attach_excursions(trades: pd.DataFrame, scratch: Path, log=None,
                      database: str = PATH_DATABASE) -> pd.DataFrame:
    """Return `trades` with mae_bps / mfe_bps / bars_seen columns attached.

    Needs symbol (canonical), open_time, close_time, direction, open_price.
    Trades whose window has no bars (instrument not on the feed, or a hold
    shorter than the bar grid) get NaN and bars_seen 0 -- callers exclude them
    from path-model training rather than imputing.
    """
    n = len(trades)
    mae = np.full(n, np.nan)
    mfe = np.full(n, np.nan)
    seen = np.zeros(n, dtype=np.int64)
    if n == 0:
        return trades.assign(mae_bps=mae, mfe_bps=mfe, bars_seen=seen)

    open_time = pd.to_datetime(trades["open_time"])
    close_time = pd.to_datetime(trades["close_time"])
    start = open_time.min().floor("D").to_pydatetime()
    end = (close_time.max().floor("D") + pd.Timedelta(days=1)).to_pydatetime()
    symbol_str = trades["symbol"].astype(str)
    symbols = tuple(sorted(symbol_str.unique()))
    root = ensure_window_bars(symbols, start, end, scratch, log=log, database=database)
    if root is None:
        return trades.assign(mae_bps=mae, mfe_bps=mfe, bars_seen=seen)

    # Floor the open to its minute so the bar CONTAINING the entry is included;
    # side="right" on the close includes the bar containing the exit.
    open_np = open_time.dt.floor("min").to_numpy()
    close_np = close_time.to_numpy()
    direction = pd.to_numeric(trades["direction"], errors="coerce").to_numpy(dtype=float)
    entry = pd.to_numeric(trades["open_price"], errors="coerce").to_numpy(dtype=float)
    groups = trades.groupby(symbol_str, observed=True).indices
    # ONE parallel read of every needed symbol (pyarrow reads the thousands of
    # small day-partition files across threads), then split in pandas: far
    # faster than a serial per-symbol read, which pays a file open per
    # symbol-day. ~18M bars x 3 columns is well under a gigabyte.
    try:
        import pyarrow.dataset as pads
        dataset = _open_dataset(root)
        bars_all = dataset.to_table(
            filter=pads.field("symbol").isin(list(groups.keys())),
            columns=["symbol", "minute", "high", "low"]).to_pandas()
    except Exception:
        return trades.assign(mae_bps=mae, mfe_bps=mfe, bars_seen=seen)
    if bars_all.empty:
        return trades.assign(mae_bps=mae, mfe_bps=mfe, bars_seen=seen)
    bars_all["minute"] = pd.to_datetime(bars_all["minute"])
    bars_all["symbol"] = bars_all["symbol"].astype(str)
    bars_all = (bars_all.sort_values(["symbol", "minute"])
                .drop_duplicates(["symbol", "minute"]).reset_index(drop=True))
    bar_groups = bars_all.groupby("symbol", observed=True).indices

    for symbol, idx in groups.items():
        rows_b = bar_groups.get(str(symbol))
        if rows_b is None or len(rows_b) == 0 or len(idx) == 0:
            continue
        series = bars_all.iloc[rows_b]
        minutes = series["minute"].to_numpy()
        highs = series["high"].to_numpy(dtype=float)
        lows = series["low"].to_numpy(dtype=float)
        starts = minutes.searchsorted(open_np[idx], side="left")
        ends = minutes.searchsorted(close_np[idx], side="right")
        ok = (ends > starts) & np.isfinite(entry[idx]) & (entry[idx] > 0)
        if ok.any():
            rows, s_ok, e_ok = idx[ok], starts[ok], ends[ok]
            window_high = _range_extrema(highs, s_ok, e_ok, np.maximum)
            window_low = _range_extrema(lows, s_ok, e_ok, np.minimum)
            d, e = direction[rows], entry[rows]
            adverse = np.where(d > 0, window_low - e, e - window_high)     # <= 0
            favourable = np.where(d > 0, window_high - e, e - window_low)  # >= 0
            mae[rows] = adverse / e * 1e4
            mfe[rows] = favourable / e * 1e4
            seen[rows] = e_ok - s_ok
        del series, minutes, highs, lows

    return trades.assign(mae_bps=mae, mfe_bps=mfe, bars_seen=seen)
