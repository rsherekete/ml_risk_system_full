"""Tick-derived markouts for ALL SIX live servers, not just the two with tick tables.

Only mt4_live01 and mt5_live01 carry their own `ticks` table, so a per-server
build left 47% of accounts (mt4_live02/03/04, mt5_dubai_live01) with no market
context at all. But a price is a property of the SYMBOL, not of the server:
EURUSD's mid at 14:32 is the same market fact whichever broker server the
client happened to trade on. So the two tick tables are unioned into one
shared bar reference and joined to every server's trades.

Symbol names carry per-server decoration (EURUSD.raw, EURUSDm, XAUUSD.i), so
both sides are normalised to a base symbol before joining -- otherwise the
suffixed servers would silently match nothing, which is exactly the kind of
quiet zero-coverage failure this rebuild exists to remove.
"""
import sys, time, warnings
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
warnings.filterwarnings("ignore")
import pandas as pd
from google.cloud import bigquery

client = bigquery.Client(project="zfx-dwh-prod")
START, END = "2026-05-29", "2026-08-28"
OUT = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\markout_all_servers.parquet"

HORIZONS = [(1, "1m"), (5, "5m"), (30, "30m"), (60, "1h"), (240, "4h"), (1440, "1d"), (4320, "3d")]

# Strip the decoration brokers add per group/server so both sides join on the
# same base instrument. Mirrors `research.canonical_symbol`'s intent, in SQL.
NORMALISE = r"UPPER(REGEXP_REPLACE({col}, r'(?i)([._-]?(RAW|ECN|PRO|MICRO|MINI|CENT|STP|C|M|E|I|Z|\\+)+)$', ''))"

MT4_TRADES = """
  SELECT '{db}' AS database, login,
         {sym} AS symbol,
         TIMESTAMP_SECONDS(open_ts) AS open_time,
         DATE(TIMESTAMP_SECONDS(COALESCE(NULLIF(close_ts,0), open_ts))) AS day,
         IF(cmd = 0, 1, -1) AS direction
  FROM `zfx-dwh-prod.{db}.orders`
  WHERE _close_ts_partition >= TIMESTAMP('{start}') AND _close_ts_partition < TIMESTAMP('{end}')
    AND cmd IN (0,1) AND open_ts > 0 AND symbol_name IS NOT NULL
"""

MT5_TRADES = """
  SELECT '{db}' AS database, CAST(login AS INT64) AS login,
         {sym} AS symbol,
         time AS open_time, DATE(time) AS day,
         IF(action = 0, 1, -1) AS direction
  FROM `{table}`
  WHERE time >= TIMESTAMP('{start}') AND time < TIMESTAMP('{end}')
    AND action IN (0,1) AND symbol IS NOT NULL
"""

trade_blocks = [
    MT4_TRADES.format(db=db, start=START, end=END, sym=NORMALISE.format(col="symbol_name"))
    for db in ("mt4_live01", "mt4_live02", "mt4_live03", "mt4_live04")
]
trade_blocks.append(MT5_TRADES.format(
    db="mt5_live01", table="zfx-dwh-prod.mt5_live01.deals",
    start=START, end=END, sym=NORMALISE.format(col="symbol")))
trade_blocks.append(MT5_TRADES.format(
    db="mt5_dubai_live01", table="zfx-dwh-prod.operational_data_store.dubai_traze_mt5_dubai_live01_deals",
    start=START, end=END, sym=NORMALISE.format(col="symbol")))

bar_joins, selects = [], []
for minutes, label in HORIZONS:
    bar_joins.append(f"""
LEFT JOIN bars p_{label}
  ON p_{label}.symbol = t.symbol
 AND p_{label}.minute = TIMESTAMP_TRUNC(TIMESTAMP_ADD(t.open_time, INTERVAL {minutes} MINUTE), MINUTE)""")
    selects.append(f"AVG(SAFE_DIVIDE((p_{label}.mid - e.mid) * t.direction, NULLIF(e.mid,0))) AS markout_{label}")
bar_joins.append("""
LEFT JOIN bars pre_5m
  ON pre_5m.symbol = t.symbol
 AND pre_5m.minute = TIMESTAMP_TRUNC(TIMESTAMP_SUB(t.open_time, INTERVAL 5 MINUTE), MINUTE)""")
selects.append("AVG(SAFE_DIVIDE((e.mid - pre_5m.mid) * t.direction, NULLIF(e.mid,0))) AS runup_5m")

SQL = f"""
WITH bars AS (
  SELECT symbol, minute, AVG(mid) AS mid, AVG(spread) AS spread FROM (
    SELECT {NORMALISE.format(col='symbol_name')} AS symbol,
           TIMESTAMP_TRUNC(tm, MINUTE) AS minute,
           (bid + ask)/2 AS mid, ask - bid AS spread
    FROM `zfx-dwh-prod.mt4_live01.ticks`
    WHERE tm >= TIMESTAMP('{START}') AND tm < TIMESTAMP_ADD(TIMESTAMP('{END}'), INTERVAL 4 DAY)
      AND bid > 0 AND ask > 0
    UNION ALL
    SELECT {NORMALISE.format(col='symbol')} AS symbol,
           TIMESTAMP_TRUNC(ts, MINUTE) AS minute,
           (bid + ask)/2 AS mid, ask - bid AS spread
    FROM `zfx-dwh-prod.mt5_live01.ticks`
    WHERE ts >= TIMESTAMP('{START}') AND ts < TIMESTAMP_ADD(TIMESTAMP('{END}'), INTERVAL 4 DAY)
      AND bid > 0 AND ask > 0
  )
  GROUP BY symbol, minute
),
trades AS (
{" UNION ALL ".join(trade_blocks)}
)
SELECT t.database, t.login, t.day,
       COUNT(*) AS context_trades,
       AVG(SAFE_DIVIDE(e.spread, NULLIF(e.mid,0))) AS avg_relative_spread,
       {", ".join(selects)}
FROM trades t
JOIN bars e ON e.symbol = t.symbol AND e.minute = TIMESTAMP_TRUNC(t.open_time, MINUTE)
{"".join(bar_joins)}
GROUP BY database, login, day
"""

dry = client.query(SQL, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
gb = dry.total_bytes_processed / 1e9
print(f"dry-run: {gb:,.1f} GB (~${gb/1000*6.25:.2f})", flush=True)
if gb > 900:
    raise SystemExit(f"aborting: {gb:.0f} GB exceeds the 900 GB guard")

t0 = time.time()
job = client.query(SQL)
markouts = job.to_dataframe()
print(f"-> {len(markouts):,} account-days, scanned {job.total_bytes_processed/1e9:,.1f} GB [{time.time()-t0:.0f}s]", flush=True)

markouts["account_key"] = markouts["database"] + ":" + markouts["login"].astype("int64").astype(str)
markouts["day"] = pd.to_datetime(markouts["day"])
if {"markout_1m", "markout_1d"} <= set(markouts.columns):
    markouts["markout_persistence"] = markouts["markout_1d"] - markouts["markout_1m"]
if {"markout_5m", "runup_5m"} <= set(markouts.columns):
    markouts["anticipation_5m"] = markouts["markout_5m"] - markouts["runup_5m"]
horizons = [f"markout_{label}" for _, label in HORIZONS if f"markout_{label}" in markouts.columns]
markouts["markout_mean"] = markouts[horizons].mean(axis=1)
markouts["markout_positive_share"] = (markouts[horizons] > 0).sum(axis=1) / len(horizons)
markouts.to_parquet(OUT, index=False)

print(f"\nsaved -> {OUT}")
print("\ncoverage by server:")
print(markouts.groupby("database").agg(account_days=("account_key", "size"), accounts=("account_key", "nunique")).to_string())
