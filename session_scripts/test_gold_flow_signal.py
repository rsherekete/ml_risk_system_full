"""XAUUSD 1h: does aggregate client positioning predict the next hour?

Pulls hourly client flow and hourly price bars for gold from BigQuery, builds
contrarian positioning features, walk-forward fits, and backtests a 1-lot
strategy with realised spread costs.

Includes two honesty checks that most backtests omit:
  * a random-signal control at the same trade rate -- if the strategy does not
    clearly beat it, the "edge" is trade selection luck;
  * always-long and always-short baselines -- gold trended over this window, so
    a directional strategy can look good purely by accident of drift.
"""
import sys, time, warnings
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from google.cloud import bigquery
from trading_data.flow_signal import add_target, backtest_signal, flow_features, walk_forward_signal

client = bigquery.Client(project="zfx-dwh-prod")
START, END = "2026-05-29", "2026-08-28"
BASE = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad"

# Gold trades under several decorated names per server; match the family.
FLOW_SQL = f"""
WITH mt4 AS (
  SELECT TIMESTAMP_TRUNC(TIMESTAMP_SECONDS(open_ts), HOUR) AS hour,
         login, cmd, volume / 100.0 AS lots
  FROM `zfx-dwh-prod.mt4_live01.orders`
  WHERE _close_ts_partition >= TIMESTAMP('{START}') AND _close_ts_partition < TIMESTAMP('{END}')
    AND cmd IN (0,1) AND open_ts > 0
    AND UPPER(symbol_name) LIKE 'XAUUSD%'
  UNION ALL
  SELECT TIMESTAMP_TRUNC(TIMESTAMP_SECONDS(open_ts), HOUR), login, cmd, volume / 100.0
  FROM `zfx-dwh-prod.mt4_live02.orders`
  WHERE _close_ts_partition >= TIMESTAMP('{START}') AND _close_ts_partition < TIMESTAMP('{END}')
    AND cmd IN (0,1) AND open_ts > 0 AND UPPER(symbol_name) LIKE 'XAUUSD%'
  UNION ALL
  SELECT TIMESTAMP_TRUNC(TIMESTAMP_SECONDS(open_ts), HOUR), login, cmd, volume / 100.0
  FROM `zfx-dwh-prod.mt4_live04.orders`
  WHERE _close_ts_partition >= TIMESTAMP('{START}') AND _close_ts_partition < TIMESTAMP('{END}')
    AND cmd IN (0,1) AND open_ts > 0 AND UPPER(symbol_name) LIKE 'XAUUSD%'
),
mt5 AS (
  SELECT TIMESTAMP_TRUNC(time, HOUR) AS hour, CAST(login AS INT64) AS login,
         action AS cmd, CAST(volume AS FLOAT64) / 10000.0 AS lots
  FROM `zfx-dwh-prod.mt5_live01.deals`
  WHERE time >= TIMESTAMP('{START}') AND time < TIMESTAMP('{END}')
    AND action IN (0,1) AND UPPER(symbol) LIKE 'XAUUSD%'
),
combined AS (SELECT * FROM mt4 UNION ALL SELECT * FROM mt5)
SELECT hour,
       SUM(IF(cmd = 0, lots, 0)) AS buy_volume,
       SUM(IF(cmd = 1, lots, 0)) AS sell_volume,
       COUNTIF(cmd = 0) AS buy_trades,
       COUNTIF(cmd = 1) AS sell_trades,
       COUNT(DISTINCT login) AS accounts
FROM combined
GROUP BY hour
"""

BARS_SQL = f"""
SELECT TIMESTAMP_TRUNC(tm, HOUR) AS hour,
       ARRAY_AGG(mid ORDER BY tm ASC LIMIT 1)[OFFSET(0)] AS open,
       MAX(mid) AS high, MIN(mid) AS low,
       ARRAY_AGG(mid ORDER BY tm DESC LIMIT 1)[OFFSET(0)] AS close,
       AVG(sp) AS spread, COUNT(*) AS ticks
FROM (
  SELECT tm, (bid + ask)/2 AS mid, ask - bid AS sp
  FROM `zfx-dwh-prod.mt4_live01.ticks`
  WHERE tm >= TIMESTAMP('{START}') AND tm < TIMESTAMP('{END}')
    AND UPPER(symbol_name) LIKE 'XAUUSD%' AND bid > 0 AND ask > 0
)
GROUP BY hour
"""

for name, sql in (("flow", FLOW_SQL), ("bars", BARS_SQL)):
    dry = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
    print(f"{name}: dry-run {dry.total_bytes_processed/1e9:.2f} GB (~${dry.total_bytes_processed/1e12*6.25:.2f})", flush=True)

t0 = time.time()
flow = client.query(FLOW_SQL).to_dataframe()
bars = client.query(BARS_SQL).to_dataframe()
print(f"flow {len(flow):,} hours | bars {len(bars):,} hours [{time.time()-t0:.0f}s]", flush=True)

for column in ("open", "high", "low", "close", "spread"):
    bars[column] = pd.to_numeric(bars[column], errors="coerce").astype("float64")
for column in ("buy_volume", "sell_volume"):
    flow[column] = pd.to_numeric(flow[column], errors="coerce").astype("float64")

merged = bars.merge(flow, on="hour", how="inner").sort_values("hour").reset_index(drop=True)
merged["hour"] = pd.to_datetime(merged["hour"])
print(f"merged {len(merged):,} hourly bars, {merged['hour'].min()} -> {merged['hour'].max()}")
print(f"mean spread ${merged['spread'].mean():.3f}, mean close ${merged['close'].mean():,.2f}, "
      f"mean accounts/hour {merged['accounts'].mean():.0f}\n", flush=True)

frame = add_target(flow_features(merged))
print(f"net_position: mean {frame['net_position'].mean():+.3f}, std {frame['net_position'].std():.3f} "
      f"(+1 = book fully long)")
usable = frame.loc[frame["next_return"].notna()]
correlation = usable["net_position"].corr(usable["next_return"])
print(f"raw correlation(net_position, next hour return) = {correlation:+.4f}  "
      f"({'CONTRARIAN' if correlation < 0 else 'TREND-FOLLOWING'} if significant)\n", flush=True)

t0 = time.time()
predictions = walk_forward_signal(frame, min_train_bars=400, refit_every=100)
print(f"walk-forward complete, {predictions.notna().sum():,} predicted bars [{time.time()-t0:.0f}s]\n", flush=True)

print("=== 1-lot XAUUSD strategy, spread costs charged ===")
for quantile in (0.5, 0.7, 0.85, 0.95):
    result = backtest_signal(frame, predictions, threshold_quantile=quantile)
    m = result.metrics
    if not m.get("trades"):
        print(f"  q={quantile}: no trades")
        continue
    print(f"  q={quantile}  trades {m['trades']:>4}  win {m['win_rate']:>5.1%}  "
          f"net ${m['total_net_pnl']:>10,.0f}  (gross ${m['total_gross_pnl']:>10,.0f} - costs ${m['total_costs']:>8,.0f})  "
          f"maxDD ${m['max_drawdown']:>9,.0f}  worst day ${m['max_daily_loss']:>8,.0f}  "
          f"Sharpe {m['daily_sharpe_annualised']:>5.2f}", flush=True)

best = backtest_signal(frame, predictions, threshold_quantile=0.85)
print("\n=== detail at q=0.85 ===")
for key, value in best.metrics.items():
    print(f"  {key:<26} {value:,.4f}" if isinstance(value, float) else f"  {key:<26} {value:,}")

print("\n=== controls: is this signal, or is it drift/luck? ===")
rng = np.random.default_rng(0)
random_results = []
for _ in range(20):
    fake = pd.Series(rng.normal(0, predictions.std(), len(frame)), index=frame.index)
    fake[predictions.isna()] = np.nan
    random_results.append(backtest_signal(frame, fake, threshold_quantile=0.85).metrics.get("total_net_pnl", 0))
print(f"  random signal, same trade rate: mean net ${np.mean(random_results):,.0f} "
      f"(std ${np.std(random_results):,.0f}, best ${np.max(random_results):,.0f})")

usable_bars = frame.loc[predictions.notna() & frame["next_return"].notna()]
for direction, label in ((1, "always long"), (-1, "always short")):
    move = usable_bars["close"] * (np.exp(usable_bars["next_return"]) - 1)
    pnl = (direction * move * 100).sum() - (usable_bars["spread"] * 100).sum()
    print(f"  {label:<24} net ${pnl:,.0f} over the same bars")

best.trades.to_parquet(f"{BASE}\\gold_signal_trades.parquet", index=False)
best.equity.to_parquet(f"{BASE}\\gold_signal_daily.parquet", index=False)
print(f"\nsaved trades and daily equity to {BASE}")
