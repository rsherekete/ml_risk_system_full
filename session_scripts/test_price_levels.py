"""XAUUSD 1h: does the LIQUIDATION MAP predict the next hour's move?

Prior attempts fed the model what clients DID (net position, flow imbalance,
technicals) and scored ~0. This feeds it WHERE THEIR ORDERS SIT relative to
spot -- stop clusters, target clusters, underwater share, entry distribution --
segmented by client class.

The mechanism is not behavioural inference, it is order flow: a dense band of
stops 50bp below spot is queued market selling that fires if price gets there.
That is a real, structural reason to expect predictability where "what did
clients do yesterday" had none.

Cost control: the interval join (which positions were open during hour t) is
the expensive part, so it is capped at 30 days and dry-run before execution.
"""
import sys, time, warnings
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from google.cloud import bigquery
from trading_data.flow_signal import backtest_signal
from trading_data.price_levels import (
    add_exposure_dynamics, align_to_next_bar, build_flow_timeseries, build_level_timeseries,
)
from trading_data.segmented_flow import add_technical_context, rank_normalised_target

START, END = "2026-05-29", "2026-08-28"
MAX_OPEN_HOURS = 168  # a week; caps the UNNEST fan-out
client = bigquery.Client(project="zfx-dwh-prod")

# Each open position is expanded into one row per hour it was live. The
# GENERATE_TIMESTAMP_ARRAY fan-out is why MAX_OPEN_HOURS exists -- without a cap
# a single position left open for months would emit thousands of rows.
POSITIONS_SQL = f"""
WITH raw AS (
  SELECT login, symbol_name AS symbol,
         TIMESTAMP_SECONDS(open_ts) AS open_time,
         TIMESTAMP_SECONDS(IF(close_ts > 0, close_ts, {int(pd.Timestamp(END).timestamp())})) AS close_time,
         open_price, sl, tp, volume/100.0 AS volume_lots,
         IF(cmd = 0, 1, -1) AS direction
  FROM `zfx-dwh-prod.mt4_live01.orders`
  WHERE _close_ts_partition >= TIMESTAMP('{START}') AND _close_ts_partition < TIMESTAMP('{END}')
    AND cmd IN (0,1) AND open_ts > 0 AND open_price > 0
    AND UPPER(symbol_name) LIKE 'XAUUSD%'
), bounded AS (
  SELECT *, LEAST(close_time, TIMESTAMP_ADD(open_time, INTERVAL {MAX_OPEN_HOURS} HOUR)) AS capped_close
  FROM raw
  WHERE close_time > open_time
)
SELECT login, open_price, sl, tp, volume_lots, direction, open_time, hour
FROM bounded, UNNEST(GENERATE_TIMESTAMP_ARRAY(
       TIMESTAMP_TRUNC(open_time, HOUR), TIMESTAMP_TRUNC(capped_close, HOUR),
       INTERVAL 1 HOUR)) AS hour
WHERE hour >= TIMESTAMP('{START}') AND hour < TIMESTAMP('{END}')
"""

# Every open and close as a separate execution, so flow can be measured at the
# price it actually happened rather than inferred from a position snapshot.
EXECUTIONS_SQL = f"""
SELECT login, TIMESTAMP_TRUNC(TIMESTAMP_SECONDS(open_ts), HOUR) AS hour,
       open_price AS price, IF(cmd = 0, 1, -1) AS direction,
       volume/100.0 * 100 * open_price AS notional_usd, TRUE AS is_open
FROM `zfx-dwh-prod.mt4_live01.orders`
WHERE _close_ts_partition >= TIMESTAMP('{START}') AND _close_ts_partition < TIMESTAMP('{END}')
  AND cmd IN (0,1) AND open_ts > 0 AND open_price > 0 AND UPPER(symbol_name) LIKE 'XAUUSD%'
  AND TIMESTAMP_SECONDS(open_ts) >= TIMESTAMP('{START}')
UNION ALL
SELECT login, TIMESTAMP_TRUNC(TIMESTAMP_SECONDS(close_ts), HOUR), close_price,
       IF(cmd = 0, 1, -1), volume/100.0 * 100 * close_price, FALSE
FROM `zfx-dwh-prod.mt4_live01.orders`
WHERE _close_ts_partition >= TIMESTAMP('{START}') AND _close_ts_partition < TIMESTAMP('{END}')
  AND cmd IN (0,1) AND close_ts > 0 AND close_price > 0 AND UPPER(symbol_name) LIKE 'XAUUSD%'
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

dry = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
for name, sql in (("positions", POSITIONS_SQL), ("executions", EXECUTIONS_SQL), ("bars", BARS_SQL)):
    scanned = client.query(sql, job_config=dry).total_bytes_processed
    print(f"dry run {name}: {scanned/1e9:.2f} GB (~${scanned/1e12*6.25:.2f})", flush=True)

t0 = time.time()
positions = client.query(POSITIONS_SQL).to_dataframe()
bars = client.query(BARS_SQL).to_dataframe()
print(f"\nposition-hours {len(positions):,} | bars {len(bars):,} [{time.time()-t0:.0f}s]", flush=True)

for frame in (positions, bars):
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            frame[column] = pd.to_datetime(frame[column], utc=True).dt.tz_localize(None)
for column in ("open_price", "sl", "tp", "volume_lots"):
    positions[column] = pd.to_numeric(positions[column], errors="coerce").astype("float64")
for column in ("open", "high", "low", "close", "spread"):
    bars[column] = pd.to_numeric(bars[column], errors="coerce").astype("float64")

stop_rate = float((positions["sl"].fillna(0) != 0).mean())
target_rate = float((positions["tp"].fillna(0) != 0).mean())
print(f"stop-loss set on {stop_rate:.1%} of position-hours | take-profit on {target_rate:.1%}")
if stop_rate < 0.05:
    print("WARNING: almost no stops are set -- the stop-cluster features cannot carry signal")
print(f"unique positions {positions['open_time'].count():,}, "
      f"median hours open {positions.groupby(['login','open_time']).size().median():.0f}\n", flush=True)

executions = client.query(EXECUTIONS_SQL).to_dataframe()
executions["hour"] = pd.to_datetime(executions["hour"], utc=True).dt.tz_localize(None)
for column in ("price", "notional_usd"):
    executions[column] = pd.to_numeric(executions[column], errors="coerce").astype("float64")
print(f"executions {len(executions):,} ({executions['is_open'].mean():.0%} opens)\n", flush=True)

levels = build_level_timeseries(positions, bars, class_column=None)
flows = build_flow_timeseries(executions, bars, class_column=None)
# Stocks describe where the book sits; deltas describe the pressure it is
# applying. The earlier run fed only stocks and the walk-forward found almost
# nothing, so the dynamics are the substantive addition here.
levels = add_exposure_dynamics(levels)
combined = levels.merge(flows, on="hour", how="outer").sort_values("hour")
# Everything is measured over a COMPLETED hour, so it belongs to the next bar.
combined = align_to_next_bar(combined)

frame = bars.merge(combined, on="hour", how="inner").sort_values("hour").reset_index(drop=True)
frame = add_technical_context(frame)
frame["next_return"] = np.log(frame["close"].shift(-1) / frame["close"])
frame["target"] = rank_normalised_target(frame)
print(f"{len(frame):,} bars with level context, {len(levels.columns)} level features", flush=True)

usable = frame.loc[frame["next_return"].notna()]
level_columns = [c for c in combined.columns if c != "hour" and c in frame.columns
                 and pd.api.types.is_numeric_dtype(frame[c])]
print("\ncorrelation of each level/flow feature with the NEXT hour's return:")
correlations = {c: usable[c].corr(usable["next_return"]) for c in level_columns if usable[c].notna().sum() > 50}
for name, value in sorted(correlations.items(), key=lambda kv: -abs(kv[1] if pd.notna(kv[1]) else 0))[:20]:
    print(f"  {name:<36} {value:+.4f}")

# DIAGNOSTIC that decides what the band correlations actually mean. In the
# previous run they rose monotonically with band width (5pct > 2pct > 1pct >
# 50bp), which is the signature of a plain directional-positioning proxy rather
# than a genuine stop-cluster effect: a real magnet would be STRONGEST at the
# tight bands, where price can actually reach the stops. If partialling out
# net_direction kills the band correlations, the "liquidation map" is just net
# positioning wearing a costume.
if "net_direction" in usable and usable["net_direction"].notna().sum() > 50:
    base = usable["net_direction"]
    print("\npartial correlation with next return, controlling for net_direction:")
    for name in ("stop_below_5pct", "stop_below_1pct", "stop_below_50bp", "stop_below_025bp",
                 "stop_concentration", "net_flow_ratio", "flow_below_50bp"):
        if name not in usable or usable[name].notna().sum() < 50:
            continue
        pair = usable[[name, "next_return"]].join(base.rename("ctrl")).dropna()
        if len(pair) < 50 or pair["ctrl"].std() == 0:
            continue
        residual_x = pair[name] - np.polyval(np.polyfit(pair["ctrl"], pair[name], 1), pair["ctrl"])
        residual_y = pair["next_return"] - np.polyval(
            np.polyfit(pair["ctrl"], pair["next_return"], 1), pair["ctrl"])
        raw = pair[name].corr(pair["next_return"])
        print(f"  {name:<26} raw {raw:+.4f} -> partial {residual_x.corr(residual_y):+.4f}")

technical_columns = [c for c in frame.columns if c not in set(level_columns) | {
    "hour", "open", "high", "low", "close", "spread", "ticks", "next_return", "target", "log_return",
} and pd.api.types.is_numeric_dtype(frame[c])]
data = frame.loc[frame["target"].notna()].reset_index(drop=True)
MIN_TRAIN, REFIT = 400, 50


def walk_forward(columns, label):
    predictions = pd.Series(np.nan, index=data.index, dtype="float64")
    model = lgb.LGBMRegressor(n_estimators=150, verbose=-1, random_state=0)
    fitted = False
    for position in range(MIN_TRAIN, len(data)):
        if (position - MIN_TRAIN) % REFIT == 0 or not fitted:
            train = data.iloc[:position]
            model.fit(train[columns], train["target"]); fitted = True
        predictions.iloc[position] = model.predict(data.iloc[[position]][columns])[0]
    scored = data.loc[predictions.notna()]
    sp = float(pd.Series(predictions[predictions.notna()].to_numpy()).rank().corr(scored["target"].rank()))
    hit = float((np.sign(predictions[predictions.notna()]) == np.sign(scored["next_return"])).mean())
    print(f"  {label:<30} {len(columns):>3} features | Spearman {sp:+.4f} | hit {hit:.1%}", flush=True)
    return predictions, model


print(f"\n=== walk-forward, {len(data):,} bars ===")
# Technicals alone are the benchmark. Level features only matter if they beat
# it -- a good absolute score from the combined model proves nothing on its own.
_, _ = walk_forward(technical_columns, "technicals only")
predictions, model = walk_forward(technical_columns + level_columns, "technicals + levels/flow")
_, _ = walk_forward(level_columns, "levels/flow ALONE")

feature_columns = technical_columns + level_columns
print("\n=== 1-lot XAUUSD backtest, spread charged (technicals + levels/flow) ===")
for quantile in (0.5, 0.7, 0.85, 0.95):
    m = backtest_signal(data, predictions, threshold_quantile=quantile).metrics
    if m.get("trades"):
        print(f"  q={quantile}  trades {m['trades']:>4}  win {m['win_rate']:>5.1%}  net ${m['total_net_pnl']:>9,.0f}  "
              f"maxDD ${m['max_drawdown']:>8,.0f}  worst day ${m['max_daily_loss']:>7,.0f}  "
              f"Sharpe {m['daily_sharpe_annualised']:>5.2f}", flush=True)

# Random control: without it, a positive backtest number is unreadable.
rng = np.random.default_rng(0)
controls = []
for _ in range(30):
    fake = pd.Series(rng.normal(0, float(predictions.std()), len(data)), index=data.index)
    fake[predictions.isna()] = np.nan
    controls.append(backtest_signal(data, fake, threshold_quantile=0.85).metrics.get("total_net_pnl", 0.0))
actual = backtest_signal(data, predictions, threshold_quantile=0.85).metrics.get("total_net_pnl", 0.0)
z = (actual - np.mean(controls)) / (np.std(controls) or 1)
print(f"\ncontrol: random ${np.mean(controls):,.0f} (std ${np.std(controls):,.0f}) | "
      f"actual ${actual:,.0f} -> {z:+.2f} sigma")
print("(below ~+2 sigma this is noise, whatever the headline P&L says)")

importance = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
print("\ntop 15 features (level features marked *):")
for name, value in importance.head(15).items():
    print(f"  {name:<34} {value:>6}{' *' if name in level_columns else ''}")
