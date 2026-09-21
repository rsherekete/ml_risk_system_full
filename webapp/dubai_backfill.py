"""Backfill `mt5_dubai_live01` into the warehouse from BigQuery.

WHY THIS EXISTS SEPARATELY FROM `mysql_extract`

The desk runs six live real-money servers. Five are in the MySQL instance behind
`ld4-dbproxy` and `mysql_extract` backfills them for free. The sixth is not
there at all -- the proxy exposes 31 databases and none of them is dubai, which
was verified by listing them rather than inferred from the naming. Its data is
replicated instead by the Traze backoffice pipeline into BigQuery, under
`operational_data_store` with a `dubai_traze_` prefix rather than a dataset of
its own, which is why a name-matching search reports it as absent.

Because nothing local held it, every model, every P&L total and every exposure
figure on the site silently excluded one entire live server.

WHAT IT COSTS

BigQuery bills by bytes scanned, so this reads month by month, selects only the
columns the warehouse schema needs, and reports the scan size of every query. A
dry run is available to price the whole backfill before spending anything.

CORRECTNESS

MT5 stores an entry deal and an exit deal separately; a trade is the pair. The
join carries the same `e.time <= x.time` guard as the MySQL path: a position id
can be reused across a reversal, and without the guard an exit pairs with a
LATER entry, producing trades that close before they open -- 44,953 such rows
in a single month when the MySQL version was missing it. Volume is in
ten-thousandths of a lot and direction comes from the ENTRY deal, since taking
it from the exit inverts every trade.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd

#: The logical server name, as it appears everywhere else in the app.
DATABASE = "mt5_dubai_live01"

BQ_PROJECT = "zfx-dwh-prod"
BQ_DATASET = "operational_data_store"
DEALS_TABLE = "dubai_traze_mt5_dubai_live01_deals"

#: Paired deals in the warehouse's canonical trade schema. Mirrors `MT5_SQL` in
#: `mysql_extract`, including the `e.time <= x.time` reversal guard.
PAIRED_SQL = """
SELECT
  -- Identity is the EXIT DEAL, not the position. MT5 closes a position in as
  -- many deals as it takes, and the warehouse de-duplicates on
  -- (database, `order`): keying that on position_id collapsed every partial
  -- close into one row. For dubai that silently discarded 19.3% of rows and
  -- $5.05M of client P&L, which flipped the server from a $4.18M firm GAIN to
  -- an $816k firm LOSS. The deal id is unique per closing event, so each
  -- partial close survives as the separate realised trade it actually is.
  x.deal                  AS `order`,
  x.position_id           AS position_id,
  x.login                 AS login,
  x.symbol                AS symbol,
  e.action                AS cmd,
  x.volume                AS volume,
  e.time                  AS open_ts,
  x.time                  AS close_ts,
  e.price                 AS open_price,
  x.price                 AS close_price,
  x.profit                AS profit,
  x.commission            AS commission,
  x.storage               AS storage,
  e.price_sl              AS sl,
  e.price_tp              AS tp,
  x.reason                AS reason
FROM `{table}` AS x
JOIN `{table}` AS e
  ON e.position_id = x.position_id
 AND e.entry = 0
 AND e.time <= x.time
WHERE x.entry = 1
  AND x.time >= @start AND x.time < @end
  AND x.volume > 0
  AND e.price > 0
-- One row per EXIT deal. A position can be scaled into with several entry
-- deals, and the join then emits one row per (exit, entry) pair -- 99,680 rows
-- for 86,416 real exits, each duplicating that exit's profit. De-duplicating
-- downstream happens to collapse them, but only by accident and with an
-- arbitrary entry surviving; picking the entry that immediately precedes the
-- exit makes the open price and direction correct by construction.
QUALIFY ROW_NUMBER() OVER (PARTITION BY x.deal ORDER BY e.time DESC) = 1
"""


def _client():
    from google.cloud import bigquery
    return bigquery.Client(project=BQ_PROJECT)


def _table_id() -> str:
    return f"{BQ_PROJECT}.{BQ_DATASET}.{DEALS_TABLE}"


def _job_config(start: datetime, end: datetime, dry_run: bool = False):
    from google.cloud import bigquery
    return bigquery.QueryJobConfig(
        dry_run=dry_run,
        use_query_cache=not dry_run,
        query_parameters=[
            bigquery.ScalarQueryParameter("start", "TIMESTAMP", start),
            bigquery.ScalarQueryParameter("end", "TIMESTAMP", end),
        ],
    )


def normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """BigQuery rows into the warehouse's canonical trade schema."""
    frame = frame.rename(columns={"profit": "net_profit"})
    frame["database"] = DATABASE

    # BigQuery returns NUMERIC as Python `Decimal`, which lands in the parquet
    # as an object column. Every downstream aggregate then dies on
    # "unsupported operand type(s) for +: 'decimal.Decimal' and 'float'" -- and
    # it dies deep inside a groupby during training, not at ingestion, so the
    # cause is a long way from the symptom. Coerced to float at the boundary.
    for column in ("net_profit", "open_price", "close_price", "commission",
                   "storage", "volume", "price_sl", "price_tp", "sl", "tp"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # MT5 volume is in ten-thousandths of a lot.
    frame["volume_lots"] = pd.to_numeric(frame["volume"], errors="coerce") / 10000.0
    # Timestamps arrive tz-aware from BigQuery; the warehouse stores naive UTC,
    # and mixing the two makes every later comparison raise.
    for source, target in (("open_ts", "open_time"), ("close_ts", "close_time")):
        stamps = pd.to_datetime(frame[source], errors="coerce", utc=True)
        frame[target] = stamps.dt.tz_localize(None)
    # Direction from the ENTRY deal: a long is opened with a buy and closed with
    # a sell, so reading the exit would invert every trade.
    frame["cmd"] = pd.to_numeric(frame["cmd"], errors="coerce").map(
        {0: "buy", 1: "sell"}).astype("string")
    # Unlike the MySQL MT5 path, this source does carry stops and the close
    # reason, so they are kept rather than filled with nulls -- the exit-policy
    # work needs exactly these.
    for column in ("sl", "tp"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    frame["state"] = "closed"
    return frame


def estimate(days: int = 730) -> dict:
    """Price the whole backfill without running it."""
    from webapp.mysql_extract import month_ranges

    client = _client()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    total_bytes = 0
    for month_start, month_end in month_ranges(start, end):
        job = client.query(PAIRED_SQL.format(table=_table_id()),
                           job_config=_job_config(month_start, month_end, dry_run=True))
        total_bytes += job.total_bytes_processed
    terabytes = total_bytes / 1e12
    return {"bytes": total_bytes, "terabytes": terabytes,
            "estimated_usd": terabytes * 6.25}


def fetch_month(start: datetime, end: datetime) -> tuple[pd.DataFrame, int]:
    """One month of paired, closed trades. Returns the frame and bytes billed."""
    client = _client()
    job = client.query(PAIRED_SQL.format(table=_table_id()),
                       job_config=_job_config(start, end))
    frame = job.result().to_dataframe()
    billed = job.total_bytes_billed or 0
    if frame.empty:
        return frame, billed
    return normalise(frame), billed


def backfill(days: int = 730, progress=None) -> dict:
    """Fill the warehouse for dubai, a month at a time.

    Deliberately mirrors `mysql_extract.backfill`: same month-at-a-time shape,
    same retry-with-backoff, same partition writer -- so once the files land,
    every screen and model picks the server up with no further change. They all
    read the warehouse directory, which is exactly why its absence was invisible.
    """
    from webapp import data_store
    from webapp.mysql_extract import month_ranges

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    written, billed_bytes, months, failures = 0, 0, [], []

    for month_start, month_end in month_ranges(start, end):
        label = month_start.strftime("%Y-%m")
        began = time.time()
        frame = None
        for attempt in range(4):
            try:
                frame, billed = fetch_month(month_start, month_end)
                billed_bytes += billed
                break
            except Exception as error:
                if attempt == 3:
                    failures.append(f"{label}: {type(error).__name__}: {str(error)[:90]}")
                    if progress:
                        progress(f"{DATABASE} {label}: FAILED after 4 attempts "
                                 f"({type(error).__name__})")
                else:
                    wait = 15 * (attempt + 1)
                    if progress:
                        progress(f"{DATABASE} {label}: {type(error).__name__}, "
                                 f"retrying in {wait}s")
                    time.sleep(wait)
        if frame is None:
            continue
        if not frame.empty:
            data_store.write_partitions(DATABASE, frame, time_column="close_time")
            written += len(frame)
        months.append(label)
        if progress:
            progress(f"{DATABASE} {label}: {len(frame):,} trades "
                     f"({billed_bytes / 1e9:.1f} GB billed, {time.time() - began:.0f}s)")

    if written:
        data_store.set_watermark(DATABASE, end, written,
                                 detail=f"bigquery backfill, {len(months)} months")

    return {"database": DATABASE, "rows": written, "months": months,
            "failures": failures, "billed_bytes": billed_bytes,
            "estimated_usd": billed_bytes / 1e12 * 6.25}
