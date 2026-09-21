"""Tick-derived market-context (markout) features, computed entirely in BigQuery.

The join is done in SQL, never in pandas: ticks (2-6 TB) and trades both already
live in the warehouse, so only small per-account-day aggregates travel back.

Markout = how the market moved after a fill, signed by trade direction and
normalised by entry price so magnitudes compare across instruments. Positive
means the market moved the CLIENT'S way -- sustained positive markout is the
adverse-selection/toxic-flow signature the behavioural features cannot see.
"""
import sys, time, warnings
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
warnings.filterwarnings("ignore")
import pandas as pd
from google.cloud import bigquery

client = bigquery.Client(project="zfx-dwh-prod")
START, END = "2026-05-29", "2026-08-28"
OUT = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\markout_by_account_day.parquet"

# (minutes after fill, label). Short horizons = execution/latency effects;
# long ones = whether the client was directionally right on the timescale the
# routing decision actually spans.
HORIZONS = [(1, "1m"), (5, "5m"), (30, "30m"), (60, "1h"), (240, "4h"), (1440, "1d"), (4320, "3d")]

SPECS = {
    "mt4_live01": dict(
        ticks="zfx-dwh-prod.mt4_live01.ticks", tick_time="tm", tick_symbol="symbol_name",
        trades="zfx-dwh-prod.mt4_live01.orders", database="mt4_live01",
        trade_sql=f"""
          SELECT login, symbol_name AS symbol,
                 TIMESTAMP_SECONDS(open_ts) AS open_time,
                 DATE(TIMESTAMP_SECONDS(COALESCE(NULLIF(close_ts,0), open_ts))) AS day,
                 IF(cmd = 0, 1, -1) AS direction
          FROM `zfx-dwh-prod.mt4_live01.orders`
          WHERE _close_ts_partition >= TIMESTAMP('{START}') AND _close_ts_partition < TIMESTAMP('{END}')
            AND cmd IN (0, 1) AND open_ts > 0 AND symbol_name IS NOT NULL
        """),
    "mt5_live01": dict(
        ticks="zfx-dwh-prod.mt5_live01.ticks", tick_time="ts", tick_symbol="symbol",
        trades="zfx-dwh-prod.mt5_live01.deals", database="mt5_live01",
        trade_sql=f"""
          SELECT CAST(login AS INT64) AS login, symbol,
                 time AS open_time, DATE(time) AS day,
                 IF(action = 0, 1, -1) AS direction
          FROM `zfx-dwh-prod.mt5_live01.deals`
          WHERE time >= TIMESTAMP('{START}') AND time < TIMESTAMP('{END}')
            AND action IN (0, 1) AND symbol IS NOT NULL
        """),
}


def build_sql(spec):
    bar_joins, selects = [], []
    for minutes, label in HORIZONS:
        bar_joins.append(f"""
LEFT JOIN bars p_{label}
  ON p_{label}.symbol = t.symbol
 AND p_{label}.minute = TIMESTAMP_TRUNC(TIMESTAMP_ADD(t.open_time, INTERVAL {minutes} MINUTE), MINUTE)""")
        selects.append(
            f"AVG(SAFE_DIVIDE((p_{label}.mid - e.mid) * t.direction, NULLIF(e.mid, 0))) AS markout_{label}")
    # One pre-fill horizon: distinguishes anticipation (flat before, favourable
    # after -- latency signature) from momentum chasing (already moving before).
    bar_joins.append("""
LEFT JOIN bars pre_5m
  ON pre_5m.symbol = t.symbol
 AND pre_5m.minute = TIMESTAMP_TRUNC(TIMESTAMP_SUB(t.open_time, INTERVAL 5 MINUTE), MINUTE)""")
    selects.append("AVG(SAFE_DIVIDE((e.mid - pre_5m.mid) * t.direction, NULLIF(e.mid, 0))) AS runup_5m")

    return f"""
WITH bars AS (
  SELECT {spec['tick_symbol']} AS symbol,
         TIMESTAMP_TRUNC({spec['tick_time']}, MINUTE) AS minute,
         AVG((bid + ask) / 2) AS mid,
         AVG(ask - bid) AS spread
  FROM `{spec['ticks']}`
  WHERE {spec['tick_time']} >= TIMESTAMP('{START}')
    AND {spec['tick_time']} <  TIMESTAMP_ADD(TIMESTAMP('{END}'), INTERVAL 4 DAY)
    AND bid > 0 AND ask > 0
  GROUP BY symbol, minute
),
trades AS ({spec['trade_sql']})
SELECT
    '{spec['database']}' AS database,
    t.login,
    t.day,
    COUNT(*) AS context_trades,
    AVG(SAFE_DIVIDE(e.spread, NULLIF(e.mid, 0))) AS avg_relative_spread,
    {", ".join(selects)}
FROM trades t
JOIN bars e ON e.symbol = t.symbol AND e.minute = TIMESTAMP_TRUNC(t.open_time, MINUTE)
{"".join(bar_joins)}
GROUP BY database, login, day
"""


frames = []
for name, spec in SPECS.items():
    sql = build_sql(spec)
    dry = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
    gb = dry.total_bytes_processed / 1e9
    print(f"{name}: dry-run {gb:,.1f} GB  (~${gb/1000*6.25:.2f})", flush=True)
    if gb > 800:
        print(f"  SKIPPED -- {gb:.0f} GB exceeds the 800 GB guard", flush=True)
        continue
    t0 = time.time()
    job = client.query(sql)
    frame = job.to_dataframe()
    print(f"  -> {len(frame):,} account-days, actually scanned "
          f"{job.total_bytes_processed/1e9:,.1f} GB [{time.time()-t0:.0f}s]", flush=True)
    frames.append(frame)

if frames:
    markouts = pd.concat(frames, ignore_index=True)
    markouts["account_key"] = markouts["database"] + ":" + markouts["login"].astype("int64").astype(str)
    markouts["day"] = pd.to_datetime(markouts["day"])
    # Structure of the reaction matters more than any single horizon.
    if {"markout_1m", "markout_1d"} <= set(markouts.columns):
        markouts["markout_persistence"] = markouts["markout_1d"] - markouts["markout_1m"]
    if {"markout_5m", "runup_5m"} <= set(markouts.columns):
        markouts["anticipation_5m"] = markouts["markout_5m"] - markouts["runup_5m"]
    horizon_columns = [f"markout_{label}" for _, label in HORIZONS if f"markout_{label}" in markouts.columns]
    if len(horizon_columns) >= 3:
        markouts["markout_mean"] = markouts[horizon_columns].mean(axis=1)
        markouts["markout_positive_share"] = (markouts[horizon_columns] > 0).sum(axis=1) / len(horizon_columns)
    markouts.to_parquet(OUT, index=False)
    print(f"\nsaved {len(markouts):,} rows, {markouts['account_key'].nunique():,} accounts -> {OUT}")
    print(markouts[horizon_columns].describe().T[["mean", "50%", "std"]].to_string())
