"""Active-hour features + per-class move attribution -> next-hour XAUUSD move.

Design under test:
  * features built on ACTIVE hours only (thresholded), targets on ALL hours
  * per-class flow with long-history horizons (2h..168h)
  * expanding-window betas attributing the move to each client type
  * risk posture, price levels, and technicals folded in

Controls are attached from the start, because the failure mode here is a
beautiful equity curve on 1,700 bars. Reported for every arm: an incremental
comparison against technicals alone, a random-signal sigma, and a sensitivity
sweep over the activity threshold (a result that only exists at one threshold
is a threshold, not a signal).
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
from trading_data.hourly_flow import build_hourly_frame, class_contribution_betas
from trading_data.price_levels import add_exposure_dynamics, align_to_next_bar, build_flow_timeseries
from trading_data.segmented_flow import add_technical_context, classify_accounts, rank_normalised_target

BASE = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad"
START, END = "2026-05-29", "2026-08-28"
client = bigquery.Client(project="zfx-dwh-prod")

TRADES_SQL = f"""
SELECT * FROM (
  SELECT 'mt4_live01' AS database, login, TIMESTAMP_SECONDS(open_ts) AS ts,
         IF(cmd = 0, 1, -1) AS direction, volume/100.0 AS lots,
         volume/100.0 * 100 * open_price AS notional_usd, open_price AS price
  FROM `zfx-dwh-prod.mt4_live01.orders`
  WHERE _close_ts_partition >= TIMESTAMP('{START}') AND _close_ts_partition < TIMESTAMP('{END}')
    AND cmd IN (0,1) AND open_ts > 0 AND open_price > 0 AND UPPER(symbol_name) LIKE 'XAUUSD%'
  UNION ALL
  SELECT 'mt4_live02', login, TIMESTAMP_SECONDS(open_ts), IF(cmd = 0, 1, -1), volume/100.0,
         volume/100.0 * 100 * open_price, open_price
  FROM `zfx-dwh-prod.mt4_live02.orders`
  WHERE _close_ts_partition >= TIMESTAMP('{START}') AND _close_ts_partition < TIMESTAMP('{END}')
    AND cmd IN (0,1) AND open_ts > 0 AND open_price > 0 AND UPPER(symbol_name) LIKE 'XAUUSD%'
  UNION ALL
  SELECT 'mt4_live04', login, TIMESTAMP_SECONDS(open_ts), IF(cmd = 0, 1, -1), volume/100.0,
         volume/100.0 * 100 * open_price, open_price
  FROM `zfx-dwh-prod.mt4_live04.orders`
  WHERE _close_ts_partition >= TIMESTAMP('{START}') AND _close_ts_partition < TIMESTAMP('{END}')
    AND cmd IN (0,1) AND open_ts > 0 AND open_price > 0 AND UPPER(symbol_name) LIKE 'XAUUSD%'
  UNION ALL
  SELECT 'mt5_live01', CAST(login AS INT64), time, IF(action = 0, 1, -1),
         CAST(volume AS FLOAT64)/10000.0, CAST(volume AS FLOAT64)/10000.0 * 100 * price, price
  FROM `zfx-dwh-prod.mt5_live01.deals`
  WHERE time >= TIMESTAMP('{START}') AND time < TIMESTAMP('{END}')
    AND action IN (0,1) AND UPPER(symbol) LIKE 'XAUUSD%' AND price > 0
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
print(f"trades {len(trades):,} | bars {len(bars):,} [{time.time()-t0:.0f}s]", flush=True)

trades["account_key"] = trades["database"] + ":" + trades["login"].astype("int64").astype(str)
trades["ts"] = pd.to_datetime(trades["ts"], utc=True).dt.tz_localize(None)
trades["hour"] = trades["ts"].dt.floor("h")
trades["day"] = trades["ts"].dt.floor("D")
for column in ("lots", "notional_usd", "price"):
    trades[column] = pd.to_numeric(trades[column], errors="coerce").astype("float64")
bars["hour"] = pd.to_datetime(bars["hour"], utc=True).dt.tz_localize(None)
for column in ("open", "high", "low", "close", "spread"):
    bars[column] = pd.to_numeric(bars[column], errors="coerce").astype("float64")

parts = []
for database in sorted(pd.read_parquet(f"{BASE}\\bq_90d_records.parquet", columns=["database"])["database"].unique()):
    part = compact_memory(pd.read_parquet(f"{BASE}\\bq_90d_records.parquet", filters=[("database", "==", database)]))
    parts.append(build_active_day_frame(part, max_gap_days=None))
    del part; gc.collect()
behaviour = pd.concat(parts, ignore_index=True)
del parts; gc.collect()
behaviour["day"] = pd.to_datetime(behaviour["day"])
classes = classify_accounts(behaviour)
del behaviour; gc.collect()
print(f"classified {len(classes):,} account-days", flush=True)

executions = trades.assign(is_open=True)
flows = add_exposure_dynamics(build_flow_timeseries(executions, bars, class_column=None),
                              columns=("net_flow_ratio", "vw_exec_distance", "flow_notional"))
flows = align_to_next_bar(flows)


def evaluate(min_trades, min_accounts, verbose=False):
    frame = build_hourly_frame(trades, classes, bars, min_trades, min_accounts)
    frame = add_technical_context(frame)
    frame = frame.merge(flows, on="hour", how="left", suffixes=("", "_lvl"))
    frame["target"] = rank_normalised_target(frame)

    imbalance = [c for c in frame.columns if c.startswith("flow_imbalance_") and "_" not in c.removeprefix("flow_imbalance_")]
    attribution = class_contribution_betas(frame, imbalance)
    frame = frame.merge(attribution, on="hour", how="left")

    technical = [c for c in frame.columns if c in {
        "log_return", "return_4h", "return_12h", "return_24h", "return_72h", "vol_4h", "vol_12h",
        "ma_gap_4h", "ma_gap_12h", "rsi14", "bar_range", "range_position", "relative_spread"}]
    contribution = [c for c in frame.columns if c.startswith("contribution_")]
    flow = [c for c in frame.columns if c not in set(technical) | set(contribution) | {
        "hour", "open", "high", "low", "close", "spread", "ticks", "next_return", "target",
        "is_active_hour"} and pd.api.types.is_numeric_dtype(frame[c])]

    data = frame.loc[frame["target"].notna()].reset_index(drop=True)
    active_share = data["is_active_hour"].mean()

    def walk_forward(columns, label, train_active_only=True):
        predictions = pd.Series(np.nan, index=data.index, dtype="float64")
        model = lgb.LGBMRegressor(n_estimators=150, num_leaves=15, min_child_samples=40,
                                  verbose=-1, random_state=0)
        fitted = False
        MIN_TRAIN, REFIT = 400, 50
        for position in range(MIN_TRAIN, len(data)):
            if (position - MIN_TRAIN) % REFIT == 0 or not fitted:
                history = data.iloc[:position]
                # Train on legible hours only; predict on every hour. Filtering
                # the TARGET side too would quietly select an easier problem
                # than the one actually traded.
                if train_active_only:
                    history = history.loc[history["is_active_hour"]]
                if len(history) > 150:
                    model.fit(history[columns], history["target"]); fitted = True
            if fitted:
                predictions.iloc[position] = model.predict(data.iloc[[position]][columns])[0]
        scored = data.loc[predictions.notna()]
        sp = float(pd.Series(predictions[predictions.notna()].to_numpy()).rank().corr(scored["target"].rank()))
        hit = float((np.sign(predictions[predictions.notna()]) == np.sign(scored["next_return"])).mean())
        if verbose:
            print(f"  {label:<38} {len(columns):>3}f | Spearman {sp:+.4f} | hit {hit:.1%}", flush=True)
        return predictions, sp, hit

    if verbose:
        print(f"\n=== thresholds: >={min_trades} trades, >={min_accounts} accounts "
              f"| {active_share:.1%} of {len(data):,} bars active ===")
        print("correlation of each class contribution with the next move:")
        for column in contribution:
            series = data[column]
            if series.notna().sum() > 100:
                print(f"  {column:<40} {series.corr(data['next_return']):+.4f}")
        walk_forward(technical, "technicals only")
        walk_forward(technical + flow, "technicals + hourly class flow")
        walk_forward(contribution, "class ATTRIBUTION alone")
    predictions, sp, hit = walk_forward(technical + flow + contribution, "FULL (flow + attribution)", True)
    if verbose:
        print(f"  {'FULL (flow + attribution)':<38} {len(technical+flow+contribution):>3}f | "
              f"Spearman {sp:+.4f} | hit {hit:.1%}", flush=True)
        print("\n1-lot backtest, spread charged:")
        for quantile in (0.7, 0.85, 0.95):
            m = backtest_signal(data, predictions, threshold_quantile=quantile).metrics
            if m.get("trades"):
                print(f"  q={quantile}  trades {m['trades']:>4}  win {m['win_rate']:>5.1%}  "
                      f"net ${m['total_net_pnl']:>9,.0f}  maxDD ${m['max_drawdown']:>8,.0f}  "
                      f"Sharpe {m['daily_sharpe_annualised']:>5.2f}", flush=True)
        rng = np.random.default_rng(0)
        controls = []
        for _ in range(30):
            fake = pd.Series(rng.normal(0, float(predictions.std()), len(data)), index=data.index)
            fake[predictions.isna()] = np.nan
            controls.append(backtest_signal(data, fake, threshold_quantile=0.85).metrics.get("total_net_pnl", 0.0))
        actual = backtest_signal(data, predictions, threshold_quantile=0.85).metrics.get("total_net_pnl", 0.0)
        z = (actual - np.mean(controls)) / (np.std(controls) or 1)
        print(f"  control: random ${np.mean(controls):,.0f} (std ${np.std(controls):,.0f}) | "
              f"actual ${actual:,.0f} -> {z:+.2f} sigma")
    return sp, hit, backtest_signal(data, predictions, threshold_quantile=0.85).metrics


evaluate(25, 10, verbose=True)

print("\n=== activity-threshold sensitivity ===")
print("(an edge that exists at only one threshold is a threshold, not a signal)")
for min_trades, min_accounts in ((5, 3), (10, 5), (25, 10), (50, 20), (100, 40)):
    sp, hit, m = evaluate(min_trades, min_accounts)
    print(f"  >={min_trades:>3} trades/>={min_accounts:>2} accts | Spearman {sp:+.4f} | hit {hit:.1%} | "
          f"q0.85 net ${m.get('total_net_pnl', 0):>9,.0f} Sharpe {m.get('daily_sharpe_annualised', 0):>5.2f}",
          flush=True)
