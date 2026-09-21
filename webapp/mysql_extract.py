"""Pull trade history from MySQL -- the primary source -- into the local store.

MySQL is the system of record and costs nothing per query, so it is the default
and BigQuery is the fallback. That ordering matters at two-year scale: the same
history through BigQuery is billed per byte scanned, every time.

PLATFORM DIFFERENCES, HANDLED HERE SO NOTHING DOWNSTREAM HAS TO CARE

* **MT4 `orders`** is already one row per round-trip: `open_ts`/`close_ts` as
  unix integers, entry and exit prices side by side.
* **MT5 `deals`** is deal-based -- an entry deal and an exit deal per position.
  They are joined on `position_id`, which is a genuine position identifier,
  rather than on `order` as the BigQuery path had to do. `entry` distinguishes
  the two sides (0 = in, 1 = out), so the pairing is exact instead of inferred.

Both are emitted in the same shape, so `trade_features` and the models see one
schema regardless of platform.

CHUNKING

Queries run a month at a time. `mt4_live01` alone holds 147.8M orders; asking
for two years in one statement would hold an enormous result set server-side
and risk a read timeout. A month is small enough to stream and to retry.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
SERVER_CONFIG = ROOT.parent / "server.yaml"

#: MySQL databases per logical server. `mt5_dubai_live01` is not behind the
#: ld4-dbproxy instance; it has its own MariaDB host and port, configured on its
#: `server.yaml` entry (see `connect_settings`).
MYSQL_DATABASES = {
    "mt4_live01": "mt4_live01",
    "mt4_live02": "mt4_live02",
    "mt4_live03": "mt4_live03",
    "mt4_live04": "mt4_live04",
    "mt5_live01": "mt5_live01",
    "mt5_dubai_live01": "mt5_dubai_live01",
}

#: The canonical output schema. Identical for MT4 and MT5.
OUTPUT_COLUMNS = ["database", "order", "login", "symbol", "cmd", "volume_lots",
                  "open_time", "close_time", "open_price", "close_price",
                  "sl", "tp", "net_profit", "commission", "storage", "state", "reason"]

MT4_SQL = """
SELECT `order`, login, symbol_name AS symbol, cmd, volume, open_ts, close_ts,
       open_price, close_price, sl, tp, profit, commission, storage, state, reason
FROM orders
WHERE close_ts >= %s AND close_ts < %s
  AND cmd IN (0, 1) AND open_ts > 0 AND open_price > 0
"""

# Entry and exit halves of each MT5 position, joined on position_id. `entry`
# marks the side: 0 opens, 1 closes. The exit carries the realised P&L; the
# entry carries the price and time the decision was made at.
MT5_SQL = """
SELECT x.deal AS `order`, x.login, x.symbol, e.action AS cmd,
       x.volume AS volume, e.time AS open_ts, x.time AS close_ts,
       e.time_msc AS open_msc, x.time_msc AS close_msc,
       e.price AS open_price, x.price AS close_price,
       x.profit, x.commission, x.storage
FROM deals x
JOIN deals e ON e.position_id = x.position_id AND e.entry = 0 AND e.time <= x.time
WHERE x.entry = 1 AND x.time >= %s AND x.time < %s
  AND x.volume > 0 AND e.price > 0
"""
# `e.time <= x.time` is not cosmetic. A position id can be reused across a
# reversal (`entry = 3`), so without it the join can pair an exit with a LATER
# entry and produce a trade that closes before it opens -- 44,953 such rows in a
# single month before this was added.
#
# The row identity is the EXIT DEAL (`x.deal`), not the position. MT5 closes a
# position in as many deals as it takes, and `data_store` de-duplicates on
# (database, `order`) -- so keying that on position_id collapsed every partial
# close into a single row and threw the rest away. Measured on the dubai server,
# where the same query shape could be checked against its source: 19.3% of rows
# and $5.05M of client P&L discarded, enough to flip that server from a $4.18M
# firm gain to an $816k firm loss. Each partial close is a separate realised
# event and is now stored as one.
#
# NOTE: mt5_live01 was backfilled under the old identity and its stored history
# is still missing those rows. It needs a full re-backfill; until then its firm
# P&L is understated. The warehouse-to-artefact reconciliation cannot detect
# this, because the loss happens upstream of the warehouse.


def connect_settings(database: str) -> dict:
    """pymysql connect kwargs for one server: its `server.yaml` entry's host,
    port, user and password where set, else `defaults`. Servers do not all sit
    behind one proxy -- dubai has its own host and port."""
    with open(SERVER_CONFIG) as stream:
        config = yaml.safe_load(stream)
    defaults = config["defaults"]
    entry = config["servers"].get(database, {})
    return {
        "host": entry.get("host") or config["servers"]["mt4_live01"]["host"],
        "port": int(entry.get("port") or defaults["port"]),
        "user": entry.get("user") or defaults["user"],
        "password": entry.get("password") or defaults["password"],
        "database": MYSQL_DATABASES[database],
    }


def _connection(database: str, timeout: int = 600):
    import pymysql

    return pymysql.connect(**connect_settings(database), connect_timeout=30,
                           read_timeout=timeout, charset="utf8mb4")


def _normalise_mt4(frame: pd.DataFrame, database: str) -> pd.DataFrame:
    frame = frame.rename(columns={"profit": "net_profit"})
    frame["database"] = database
    # MT4 volume is in hundredths of a lot.
    frame["volume_lots"] = pd.to_numeric(frame["volume"], errors="coerce") / 100.0
    frame["open_time"] = pd.to_datetime(frame["open_ts"], unit="s", errors="coerce")
    frame["close_time"] = pd.to_datetime(frame["close_ts"], unit="s", errors="coerce")
    frame["cmd"] = frame["cmd"].map({0: "buy", 1: "sell"}).astype("string")
    frame["state"] = "closed"
    return frame


def _normalise_mt5(frame: pd.DataFrame, database: str) -> pd.DataFrame:
    frame = frame.rename(columns={"profit": "net_profit"})
    frame["database"] = database
    # MT5 volume is in ten-thousandths of a lot.
    frame["volume_lots"] = pd.to_numeric(frame["volume"], errors="coerce") / 10000.0
    # Already DATETIME on this platform -- no unit conversion. `time_msc` is
    # the same instant to the millisecond; sub-second markouts (latency spec
    # MO100-MO500) are only meaningful on it, so it wins where present.
    for msc, sec, target in (("open_msc", "open_ts", "open_time"),
                             ("close_msc", "close_ts", "close_time")):
        stamps = pd.to_datetime(frame[sec], errors="coerce")
        if msc in frame.columns:
            stamps = pd.to_datetime(frame[msc], errors="coerce").fillna(stamps)
        frame[target] = stamps
    # Direction comes from the ENTRY deal's action. Taking it from the exit
    # would invert every trade -- a long is opened with a buy and closed with a
    # sell -- which is the same trap the BigQuery pairing had to avoid.
    frame["cmd"] = frame["cmd"].map({0: "buy", 1: "sell"}).astype("string")
    for column in ("sl", "tp", "reason"):
        frame[column] = pd.NA
    frame["state"] = "closed"
    return frame


def fetch_month(database: str, start: datetime, end: datetime) -> pd.DataFrame:
    """One month of closed trades from one server, in the canonical schema."""
    is_mt5 = database.startswith("mt5")
    sql = MT5_SQL if is_mt5 else MT4_SQL
    # The two platforms store time differently and the mismatch is silent: MT4
    # uses unix integers, MT5 a DATETIME. Passing timestamps to MT5 compares an
    # int against a datetime and returns zero rows rather than erroring.
    params = ((start.replace(tzinfo=None), end.replace(tzinfo=None)) if is_mt5
              else (int(start.timestamp()), int(end.timestamp())))
    connection = _connection(database)
    try:
        frame = pd.read_sql(sql, connection, params=params)
    finally:
        connection.close()
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if is_mt5:
        # One row per EXIT deal. A position scaled into with several entry deals
        # makes the join emit one row per (exit, entry) pair; the warehouse
        # de-duplication would keep an arbitrary entry. Keep the entry that
        # immediately precedes the exit -- the `QUALIFY ROW_NUMBER()` the
        # BigQuery path uses, which MariaDB does not support.
        frame = (frame.sort_values(["order", "open_ts"], kind="stable")
                 .drop_duplicates(subset="order", keep="last"))
    frame = _normalise_mt5(frame, database) if is_mt5 else _normalise_mt4(frame, database)
    # CENT DEFLATION at the SOURCE: every trade written to the warehouse -- and
    # therefore every model feature, P&L sum and exposure built from it -- is in
    # real dollars / real lots. A cent account's raw money and lots are 100x, so
    # without this the training corpus and every historical view are inflated on
    # ~86k accounts. Money: net_profit, commission, storage(swap). Lots:
    # volume_lots (prices are never cent-scaled).
    from webapp import trade_feed
    trade_feed.deflate_cent(frame, database, "login",
                            money_cols=("net_profit", "commission", "storage"),
                            lot_cols=("volume_lots",))
    for column in OUTPUT_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[OUTPUT_COLUMNS]


def month_ranges(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    ranges, cursor = [], datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    while cursor < end:
        following = (cursor + timedelta(days=32)).replace(day=1)
        ranges.append((max(cursor, start), min(following, end)))
        cursor = following
    return ranges


def refresh_recent(days: float = 2.0, databases=None, progress=None) -> dict:
    """Top up the warehouse with the LAST FEW DAYS of closed trades.

    `backfill` works a whole month at a time, which is right for history and
    far too heavy to run often. Nothing was topping the warehouse up between
    backfills, so on 16 Sep 2026 the latency scan was reading data that ended
    22 hours earlier while reporting itself as fresh. This fetches only the
    recent slice per server and merges it into the monthly partitions
    (write_partitions de-duplicates, so overlap is safe).
    """
    from webapp import data_store

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=float(days))
    out = {"servers": {}, "rows": 0, "failures": {}}
    for database in (databases or MYSQL_DATABASES):
        t0 = time.time()
        try:
            frame = fetch_month(database, start, end)
            written = 0
            if not frame.empty:
                written = data_store.write_partitions(database, frame, time_column="close_time")["written"]
                newest = pd.to_datetime(frame["close_time"]).max()
                data_store.set_watermark(database, end, written, f"top-up {days:g}d (newest close {newest})")
            out["servers"][database] = {"rows": int(len(frame)), "written": int(written),
                                        "seconds": round(time.time() - t0, 1)}
            out["rows"] += int(len(frame))
        except Exception as error:
            out["failures"][database] = f"{type(error).__name__}: {str(error)[:100]}"
        if progress:
            progress(f"{database}: {out['servers'].get(database) or out['failures'].get(database)}")
    return out


def backfill(database: str, days: int = 730, progress=None) -> dict:
    """Fill the local store for one server, a month at a time."""
    from webapp import data_store

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    written, months, failures = 0, [], []

    for month_start, month_end in month_ranges(start, end):
        label = month_start.strftime("%Y-%m")
        t0 = time.time()
        frame = None
        # Retry with backoff. A backfill runs for hours over a VPN, and a
        # transient DNS or connection failure previously abandoned every
        # remaining month -- 20 months of work lost to one dropped packet.
        for attempt in range(4):
            try:
                frame = fetch_month(database, month_start, month_end)
                break
            except Exception as error:
                if attempt == 3:
                    failures.append(f"{label}: {type(error).__name__}: {str(error)[:80]}")
                    if progress:
                        progress(f"{database} {label}: FAILED after 4 attempts "
                                 f"({type(error).__name__})")
                else:
                    wait = 15 * (attempt + 1)
                    if progress:
                        progress(f"{database} {label}: {type(error).__name__}, "
                                 f"retrying in {wait}s")
                    time.sleep(wait)
        if frame is None:
            continue
        if not frame.empty:
            result = data_store.write_partitions(database, frame, time_column="close_time")
            written += result["written"]
            months.append(label)
        if progress:
            progress(f"{database} {label}: {len(frame):,} rows in {time.time() - t0:.0f}s")

    if written:
        data_store.set_watermark(
            database, end, written,
            f"mysql backfill {len(months)} months" + (f", {len(failures)} failed" if failures else ""))
    return {"database": database, "rows": written, "months": months, "failures": failures}


def open_positions(database: str) -> pd.DataFrame:
    """Live open positions straight from the server.

    MT5 keeps a real `positions` table; MT4 represents an open position as an
    order with `close_ts = 0`. Both are the broker's own view, which is what a
    risk screen needs -- inferring open interest from a trade stream can only
    ever see positions opened inside the retained window.
    """
    if database.startswith("mt5"):
        sql = ("SELECT login, symbol, action AS cmd, volume, price_open, price_current,"
               " price_sl AS sl, price_tp AS tp, profit, storage, time_create"
               " FROM positions")
        divisor = 10000.0
    else:
        sql = ("SELECT login, symbol_name AS symbol, cmd, volume, open_price,"
               " sl, tp, profit, storage, open_ts AS time_create"
               " FROM orders WHERE close_ts = 0 AND cmd IN (0,1)")
        divisor = 100.0

    connection = _connection(database, timeout=180)
    try:
        frame = pd.read_sql(sql, connection)
    finally:
        connection.close()
    if frame.empty:
        return frame
    frame["database"] = database
    frame["volume_lots"] = pd.to_numeric(frame["volume"], errors="coerce") / divisor
    frame["direction"] = frame["cmd"].map({0: 1, 1: -1})
    frame["account_key"] = database + ":" + frame["login"].astype("int64").astype(str)
    # CENT DEFLATION: a cent account's lots and money are 100x their real value.
    # Without this the live exposure book is inflated ~100x on ~86k accounts --
    # the "$3.1bn gross on one symbol" absurdity. profit/storage are money;
    # volume_lots is the lot count that drives every notional downstream.
    from webapp import trade_feed
    trade_feed.deflate_cent(frame, database, "login",
                            money_cols=("profit", "storage"),
                            lot_cols=("volume_lots",))
    return frame
