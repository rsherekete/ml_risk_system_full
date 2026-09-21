"""XAUUSD 1h: type-segmented client positioning + market context -> next-hour move.

The previous attempt used ONE aggregate net-position number and scored a
correlation of -0.013, because informed and uninformed flow cancel when summed.
This keeps them separate: positioning is aggregated within behavioural class,
and the model can learn an opposite sign for smart money and dumb money.

Target: next-hour return, rank-normalised to [-1, +1], sign preserved.
"""
import sys, gc, time, warnings
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from google.cloud import bigquery
from trading_data.behaviour_features import build_active_day_frame
from trading_data.bigquery_data_client import compact_memory
from trading_data.flow_signal import backtest_signal
from trading_data.segmented_flow import (
    CLIENT_CLASSES, add_technical_context, classify_accounts, rank_normalised_target, segmented_positioning,
)

BASE = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad"
START, END = "2026-05-29", "2026-08-28"
client = bigquery.Client(project="zfx-dwh-prod")

# Per-TRADE gold flow, so it can be joined to per-account classes.
TRADES_SQL = f"""
SELECT * FROM (
  SELECT 'mt4_live01' AS database, login, TIMESTAMP_SECONDS(open_ts) AS ts,
         IF(cmd = 0, 1, -1) AS direction, volume/100.0 AS lots
  FROM `zfx-dwh-prod.mt4_live01.orders`
  WHERE _close_ts_partition >= TIMESTAMP('{START}') AND _close_ts_partition < TIMESTAMP('{END}')
    AND cmd IN (0,1) AND open_ts > 0 AND UPPER(symbol_name) LIKE 'XAUUSD%'
  UNION ALL
  SELECT 'mt4_live02', login, TIMESTAMP_SECONDS(open_ts), IF(cmd = 0, 1, -1), volume/100.0
  FROM `zfx-dwh-prod.mt4_live02.orders`
  WHERE _close_ts_partition >= TIMESTAMP('{START}') AND _close_ts_partition < TIMESTAMP('{END}')
    AND cmd IN (0,1) AND open_ts > 0 AND UPPER(symbol_name) LIKE 'XAUUSD%'
  UNION ALL
  SELECT 'mt4_live04', login, TIMESTAMP_SECONDS(open_ts), IF(cmd = 0, 1, -1), volume/100.0
  FROM `zfx-dwh-prod.mt4_live04.orders`
  WHERE _close_ts_partition >= TIMESTAMP('{START}') AND _close_ts_partition < TIMESTAMP('{END}')
    AND cmd IN (0,1) AND open_ts > 0 AND UPPER(symbol_name) LIKE 'XAUUSD%'
  UNION ALL
  SELECT 'mt5_live01', CAST(login AS INT64), time, IF(action = 0, 1, -1), CAST(volume AS FLOAT64)/10000.0
  FROM `zfx-dwh-prod.mt5_live01.deals`
  WHERE time >= TIMESTAMP('{START}') AND time < TIMESTAMP('{END}')
    AND action IN (0,1) AND UPPER(symbol) LIKE 'XAUUSD%'
)
"""
BARS_SQL = f"""
SELECT TIMESTAMP_TRUNC(tm, HOUR) AS hour,
       ARRAY_AGG(mid ORDER BY tm ASC LIMIT 1)[OFFSET(0)] AS open,
       MAX(mid) AS high, MIN(mid) AS low,
       ARRAY_AGG(mid ORDER BY tm DESC LIMIT 1)[OFFSET(0)] AS close,
       AVG(sp) AS spread, COUNT(*) AS ticks
FROM (SELECT tm, (bid+ask)/2 AS mid, ask-bid AS sp FROM `zfx-dwh-prod.mt4_live01.ticks`
      WHERE tm >= TIMESTAMP('{START}') AND tm < TIMESTAMP('{END}')
        AND UPPER(symbol_name) LIKE 'XAUUSD%' AND bid > 0 AND ask > 0)
GROUP BY hour
"""

t0 = time.time()
trades = client.query(TRADES_SQL).to_dataframe()
bars = client.query(BARS_SQL).to_dataframe()
print(f"gold trades {len(trades):,} | bars {len(bars):,} [{time.time()-t0:.0f}s]", flush=True)
trades["account_key"] = trades["database"] + ":" + trades["login"].astype("int64").astype(str)
# BigQuery hands back tz-aware UTC; the behavioural frame is tz-naive. Strip the
# zone so the day/hour keys join -- the same trap `compact_memory`'s sibling
# `_strip_timezones` exists to close on the main BQ client path.
trades["ts"] = pd.to_datetime(trades["ts"], utc=True).dt.tz_localize(None)
trades["hour"] = trades["ts"].dt.floor("h")
trades["day"] = trades["ts"].dt.floor("D")
trades["lots"] = pd.to_numeric(trades["lots"], errors="coerce").astype("float64")
for column in ("open", "high", "low", "close", "spread"):
    bars[column] = pd.to_numeric(bars[column], errors="coerce").astype("float64")
bars["hour"] = pd.to_datetime(bars["hour"], utc=True).dt.tz_localize(None)

# Point-in-time behavioural classes from each account's own prior record.
parts = []
for database in sorted(pd.read_parquet(f"{BASE}\\bq_90d_records.parquet", columns=["database"])["database"].unique()):
    part = compact_memory(pd.read_parquet(f"{BASE}\\bq_90d_records.parquet", filters=[("database", "==", database)]))
    parts.append(build_active_day_frame(part, max_gap_days=None))
    del part; gc.collect()
behaviour = pd.concat(parts, ignore_index=True)
del parts; gc.collect()
behaviour["day"] = pd.to_datetime(behaviour["day"])
classes = classify_accounts(behaviour)
print(f"classified {len(classes):,} account-days:")
print(classes["client_class"].value_counts().to_string(), flush=True)

positioning = segmented_positioning(trades, classes)
frame = bars.merge(positioning, on="hour", how="inner").sort_values("hour").reset_index(drop=True)
frame = add_technical_context(frame)
frame["next_return"] = np.log(frame["close"].shift(-1) / frame["close"])
frame["target"] = rank_normalised_target(frame)
usable = frame.loc[frame["next_return"].notna()]
print(f"\n{len(frame):,} hourly bars with segmented positioning", flush=True)

print("\ncorrelation of each class's positioning with the NEXT hour's return:")
for cls in CLIENT_CLASSES:
    column = f"net_position_{cls}"
    if column in usable and usable[column].notna().sum() > 50:
        correlation = usable[column].corr(usable["next_return"])
        share = usable.get(f"share_{cls}", pd.Series(np.nan)).mean()
        print(f"  {cls:<20} corr {correlation:+.4f}   mean share of volume {share:.1%}")
for column in ("smart_dumb_divergence", "smart_money_net", "dumb_money_net", "size_money_net"):
    if column in usable:
        print(f"  {column:<20} corr {usable[column].corr(usable['next_return']):+.4f}")

feature_columns = [c for c in frame.columns if c not in {
    "hour", "open", "high", "low", "close", "spread", "ticks",
    "next_return", "target", "log_return",
} and pd.api.types.is_numeric_dtype(frame[c])]
print(f"\n{len(feature_columns)} features", flush=True)

data = frame.loc[frame["target"].notna()].reset_index(drop=True)
predictions = pd.Series(np.nan, index=data.index, dtype="float64")
model = lgb.LGBMRegressor(n_estimators=150, verbose=-1, random_state=0)
fitted = False
MIN_TRAIN, REFIT = 400, 50
t0 = time.time()
for position in range(MIN_TRAIN, len(data)):
    if (position - MIN_TRAIN) % REFIT == 0 or not fitted:
        train = data.iloc[:position]
        model.fit(train[feature_columns], train["target"]); fitted = True
    if fitted:
        predictions.iloc[position] = model.predict(data.iloc[[position]][feature_columns])[0]
print(f"walk-forward: {predictions.notna().sum():,} predicted bars [{time.time()-t0:.0f}s]", flush=True)

scored = data.loc[predictions.notna()]
sp = float(pd.Series(predictions[predictions.notna()].to_numpy()).rank().corr(scored["target"].rank()))
hit = float((np.sign(predictions[predictions.notna()]) == np.sign(scored["next_return"])).mean())
print(f"\nSpearman(prediction, rank target) = {sp:+.4f}   directional hit rate = {hit:.1%}\n", flush=True)

print("=== 1-lot XAUUSD backtest, spread charged ===")
for quantile in (0.5, 0.7, 0.85, 0.95):
    result = backtest_signal(data, predictions, threshold_quantile=quantile)
    m = result.metrics
    if not m.get("trades"):
        continue
    print(f"  q={quantile}  trades {m['trades']:>4}  win {m['win_rate']:>5.1%}  net ${m['total_net_pnl']:>9,.0f}  "
          f"(gross ${m['total_gross_pnl']:>9,.0f} - cost ${m['total_costs']:>7,.0f})  maxDD ${m['max_drawdown']:>8,.0f}  "
          f"worst day ${m['max_daily_loss']:>7,.0f}  Sharpe {m['daily_sharpe_annualised']:>5.2f}", flush=True)

rng = np.random.default_rng(0)
controls = []
for _ in range(30):
    fake = pd.Series(rng.normal(0, float(predictions.std()), len(data)), index=data.index)
    fake[predictions.isna()] = np.nan
    controls.append(backtest_signal(data, fake, threshold_quantile=0.85).metrics.get("total_net_pnl", 0.0))
actual = backtest_signal(data, predictions, threshold_quantile=0.85).metrics.get("total_net_pnl", 0.0)
z = (actual - np.mean(controls)) / (np.std(controls) or 1)
print(f"\ncontrol: random signal mean ${np.mean(controls):,.0f} (std ${np.std(controls):,.0f})")
print(f"actual q=0.85 ${actual:,.0f}  ->  {z:+.2f} sigma vs random")
print("(needs roughly +2 sigma before it is worth taking seriously)")

importance = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
print("\ntop 15 features:")
print(importance.head(15).to_string())
