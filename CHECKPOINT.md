# System Checkpoint — 2026-08-31 (rev 2, late evening)

## REV 2 SUMMARY — what changed since rev 1

- **Strategy pivot to 90 days** (user-directed): the 2-year retrains are
  parked; both models retrained on 90d with all of tonight's fixes. The 2yr
  scripts are ready (`scratchpad/iterate_2yr.py`) but must not run yet.
- **Training is now ~40x faster**: `max_bin=63`, per-fit cap 3M rows
  (`max_fit_rows`), 150 trees @ lr 0.1. Trading 90d trains in 3.2 min; quant
  90d (18M rows, FULL density) in ~15 min. AUC held: quant 0.7002 at 90d.
- **90d results** (walk-forward): TRADING weighted+cashflow — hedge 2%
  +$4.78M / DD +$1.27M, hedge 5% +$3.8M/+$2.0M, both dominate (uniform model
  managed +$0.46M on this window). QUANT full-density — flat $90.1M, hedge 10%
  **+$47.3M (+53%) with DD −$4.85M → −$2.83M**; every fraction dominates.
  Flat B-books now ALIGNED across views (sample_fraction recorded in quant
  meta; metrics and display curves scaled to full-book units with an explicit
  note on the Overview).
- **Cashflow features now feed BOTH models** (quant: 6 of 9 in top-25
  importances immediately).
- **Final quant booster persisted** (`artifacts/quant_model.txt` +
  `_features.txt`) — live scoring works (verified: 0.729 on a stance-copy
  trade). Artifact now carries `close_time`/`open_price`/`close_price`.
- **VANTAGE COPYTRADER (admin-only, `/vantage`)**: `webapp/vantage.py` +
  `templates/vantage.html`. Credentials in `vantage.yaml` (repo root; example
  auto-created). COPY accounts ≥0.80 mean score / INVERT ≤0.30 (873 accounts:
  692 copy, 181 invert). 35% rule: k = 0.35×balance / strategy-DD@1x
  ($1,096,130); lots = round(client×k, 2). Brackets from expected $ via venue
  tick economics, on OUR direction. Kafka duckdb polling (zero-profit deals =
  openings), paper/live modes, kill switch, expected-vs-actual panel, sqlite
  order log. Stance/DD cached per artifact vintage (63s → 2ms).
- **Path-aware exit study**: bars fetched for 95%-coverage symbols; MAE/MFE
  for 965k of 1.8M selected trades (53.7% — mt5 servers have no tick tables);
  dollar conversion via each trade's own pnl/price-move. Output
  `scratch/quant_mae_mfe.parquet`; the Exit Policy tab flips path-aware
  automatically when it exists. Bars cached (`exit_study_bars.parquet`).
- Ultracode note: verification workflow for tonight's edits still owed.

---

# System Checkpoint — 2026-08-31 (rev 1)

The single source of truth for where the broker risk-analytics system stands:
what is proven, what is running, what changed and why. Written so that any
session — human or machine — can resume from here without re-deriving anything.

---

## 1. Headline results (all walk-forward, out-of-sample)

### Trading model (account-day routing) — artifact of 2026-08-31
- 4,323,494 exposure-day rows, 670 days, 6 servers, AUC **0.6991** on the true
  forward self-relative target (rises to **0.7396** over the last 90 days).
- Flat B-book earns **$1,095,790,141** over the window; reconciled to the
  warehouse to **$383 (0.000035%)**.
- **Live routing policy: hedge only `score >= 0.80`** (quota fallback 1%).
  Beats flat B-book on BOTH axes in every window tested:

  | window | uplift vs flat | drawdown vs flat |
  |---|---|---|
  | full 670d | **+$11.17M** | +$0.70M |
  | last 365d | +$5.20M | +$0.70M |
  | last 180d | +$7.77M | +$0.69M |
  | last 90d  | +$3.53M | +$0.69M |

  The old 5% quota **lost $43.8M** — hedging is insurance paid from a positive
  edge, and a wide quota spends premium on marginal cases. Threshold tuning is
  exhausted: hindsight-perfect selection on current scores caps at ~$13.8M
  (global) / ~$15.7M (per-decile).

### Quant model (trade-level copy routing) — completed 2026-08-31, 428 min
- 23,857,208 trades (17.2% day-stratified sample of 145M), 730 days, 6 servers,
  AUC **0.7433**.
- **Every hedge fraction dominates flat B-book** (sampled baseline $151.2M):

  | fraction | profit vs flat | DD vs flat | Sharpe |
  |---|---|---|---|
  | 2%  | +$55.3M | +$1.0M | 3.00 |
  | 5%  | +$81.7M | +$1.5M | 3.37 |
  | 10% | +$92.4M | +$1.6M | 3.64 |
  | 20% | **+$93.6M** | +$1.1M | 3.97 |

  First run on the free 2-year MySQL warehouse (previous artifact was 71 days
  of paid BigQuery, missing mt5_live01 entirely). **Live quant policy set to
  hedge_fraction = 0.10** — captures +$92.4M of the +$93.6M maximum with the
  best drawdown improvement and half the copy-trader execution load of 20%.

### The drawdown diagnosis (the structural finding)
The firm's worst drawdown ($31.3M, 16 days, Jan 2026) was **one crowded
position, not client skill**: 84.8% XAUUSD across 12,879 accounts; net gold
went −556 lots (Jan 2) → 93,759 lots (Jan 19); the worst day landed after the
book already exceeded 65k lots. The per-account model is structurally blind to
it (winners scored 0.5164 vs 0.4513; 0.5% cleared the hedge rule). Hence the
two-layer architecture:
- **Edge layer** (account/trade models): clients lose ~$250/account-day; keep
  the edge by hedging narrowly.
- **Risk layer** (`exposure_policy.py` + `book_risk.py` + Book Risk tab).
  **Backtest verdict: a STATIC cap fails economically** — every fixed limit
  from $5M to $500M destroys $621M–$1.1bn of profit because it binds on
  374–706 of 730 days. The book is structurally directional (clients sit
  permanently net long gold; the firm's core edge IS being short that crowd),
  so a cap that fires daily passes the edge to market along with the risk. It
  even made the January window worse (forfeited the firm-winning days inside
  it). What distinguished January was that exposure was ABNORMAL, not large —
  hence `simulate_anomaly`/`sweep_anomaly`: hedge only the share above each
  instrument's own trailing |net| percentile (120d window, prior days only).
  Sweep across p90/p95/p98/p99.5 runs after the trading retrain (`finish.py`);
  results land in `webapp/artifacts/book_risk.json` and the Book Risk tab
  shows both tables with the static-cap verdict stated.

Loss shape: 290 losing days totalling −$246M; the worst 20 days alone cost
$81M (7.4% of profit). Ceiling for perfect account selection: +$2.06bn — the
account model captures 0.54% of it, hence the risk layer matters more.

---

## 2. Data layer — 6/6 servers, complete

| server | months | rows | source |
|---|---|---|---|
| mt4_live01 | 25 | 37,980,546 | MySQL |
| mt4_live02 | 25 | 38,927,134 | MySQL |
| mt4_live03 | 25 | 2,420,016 | MySQL |
| mt4_live04 | 25 | 46,494,026 | MySQL |
| mt5_live01 | 25 | 19,613,079 | MySQL (re-backfilled — see §3) |
| mt5_dubai_live01 | 19 | 86,476 | BigQuery ($0.006 total) |

Also cached locally (VPN-independent):
- **Cash movements**: 4,899,732 rows, `webapp/cashflow_store.py`, 9
  point-in-time features feeding the trading model (leak-tested).
- **Contract specs**: 2,813 rows, `webapp/spec_cache/`, with asset-class
  contract-size fallback AND FX rates inferable from trade prices
  (`infer_rates_from_prices`) — exposure survives with no DB at all.

---

## 3. Critical bugs found and fixed (each verified)

1. **MT5 partial closes discarded** — both MT5 extracts keyed rows on
   `position_id` while the store dedups on `(database, order)`. Every partial
   close after the first was silently dropped: 19.3% of dubai rows; ~2.2% of
   mt5_live01 (July 2026 alone: 79,136 rows / $696k understated). Identity is
   now the **exit deal id**; dubai also QUALIFYs to the entry immediately
   preceding each exit (multi-entry positions fanned the join, duplicating
   profit). mt5_live01 fully re-backfilled: 0 dup deals, 0 closes-before-opens.
2. **Quant loader read 0 rows from the warehouse** — requested `account_key`,
   a derived column; parquet returned empty WITHOUT error, silently sending the
   whole view to the paid 90-day BigQuery fallback (which also skips
   mt5_live01). Fixed; fallback now logs loudly with the reason.
3. **equity_curves O(n²)** — full-frame scan per day + Python dict pool rebuild
   (~700M iterations). Vectorised (searchsorted + code-indexed arrays);
   verified bit-identical on 3 configs; >10min → 2.3s. `/trading/overview`
   cold: timeout → 0.4s.
4. **BigQuery NUMERIC → Decimal** — object dtype survives ingestion, kills
   training hours later inside a groupby. Coerced at both boundaries.
5. **Spec loader swallowed VPN errors** → empty specs → notional = 0 (“no
   risk” instead of “no data”). Now cached + shaped-empty + price-inferred
   rates; verified: XAUUSD $400k, USDJPY $100k, GBPJPY $133k with zero DB.
6. **Artifact schema drift** — exposure-day artifact lacked columns the
   templates referenced → 500s. `load_scores` now guarantees every expected
   column (NaN renders “--”); manifests fall back `gross_notional → pnl_20d →
   live_positions` for ranking.
7. **Coverage was presence-only** — an interrupted backfill left mt5_live01
   with 16/25 months while coverage said “present”. Now month-gap-aware
   (`data_store.coverage`), surfaced on Validation; `backfill_repair.py`
   fetches only missing months and **waits out VPN drops** (the tunnel dropped
   three times during this work; the repair resumed itself each time).
8. **In-process job polling** — chaining retrains via `job_state()` saw a
   separate process as “idle”, ran two 12GB jobs concurrently, OOM’d. The
   orchestrator now watches actual processes (psutil).
9. **Same-day-P&L validation label** — scoring the forward model on a
   contemporaneous label returns ~0.43 and looks inverted; the stored target
   (`sigma`) is the real label. Documented on the Validation tab.

---

## 4. What changed in the models (this retraining cycle)

- **Economic sample weights** (`economic_weights: bool = True` in
  TrainingConfig): rows weighted by the account's own `pnl_vol_20d`
  (log-scaled, clipped [0.1, 10]) so the loss reflects dollars, not row counts.
  Motivation (measured): the top size decile holds **81.5% of firm P&L** and
  had the WORST AUC (0.670 vs ~0.714 for deciles 5–9). Size-RANKING was tested
  and is catastrophic (−$411M) — weighting the objective is the correct use of
  size, ranking by it is not.
- **9 cashflow features** (deposits/withdrawals/net funding/churn/recency),
  cumulative-to-date, attached by backward merge_asof; leak test: a day-5 row
  sees a day-1 deposit, is blind to a day-10 withdrawal.
- **Six servers + corrected MT5 P&L** for the first time.

---

## 5. Web app state (all pages verified 200)

Tabs: Trading — Summary, Overview, A-Book, Clients, Account, Exposure,
Regions, Cash Flow, Surveillance, **Book Risk (new)**, Risk, Performance,
**Validation (new)**. Quant — Summary, Overview, Signals, Copy Trader,
Exposure, Regions, Cash Flow, **Exit Policy (new)**, Research, Risk,
Performance, Validation.

- **Overview** now leads with the *cumulative outperformance* chart (model −
  flat on its own axis, zero line marked) — on raw curves an $11M edge on
  $1.1bn is a line width.
- **Validation** shows: 6-server coverage w/ month gaps, warehouse
  reconciliation ($383 MATCH, cached, background recompute), AUC on the true
  target by window, policy sweep (quota vs threshold), dominance tables,
  target sanity.
- **Book Risk** documents the January diagnosis and will show the cap sweep
  when the backtest lands (`book_risk.report()`, cached to
  `webapp/artifacts/book_risk.json`).
- **Exit Policy** leads with an honest red panel: stop/target results on
  realised P&L are structurally biased (a stop applied to outcomes can only
  ever add profit) — path-aware evaluation activates automatically when trades
  carry `mae`/`mfe` from the minute-bar replay.

## 6. Running / next

- RUNNING: trading retrain (weighted + cashflow features + 6 servers, ~9h,
  started 20:59). Then `scratchpad/finish.py` (process-watch chained) builds
  the daily book + BOTH exposure sweeps, prints the weighted-vs-uniform
  threshold ladder, restarts the server and sweeps all routes.
- NEXT session: read `finish.py` output — the two open results are (a) does
  economic weighting lift the ladder above the uniform run's +$11–14M, and
  (b) does the anomaly exposure policy dominate where the static cap failed.
  Then: crowding features (per-symbol net-exposure percentile from
  `daily_book.parquet` as account-day model inputs — connects the two layers);
  per-decile thresholds on the manifest; dubai spec cache (KeyError — specs
  live in the BQ symbols table, not MySQL); exit-policy excursions via
  `tick_bars.excursions`; quant full-density validation (results are on a
  17.2% sample).

## 7. Honest caveats

- The 0.80 threshold was chosen on the same windows it is evaluated on; the
  DIRECTION (few, confident hedges) is robust across all four windows, the
  exact figure is fitted.
- Quant results are on a 17.2% sample; absolute dollars scale ~5.8x but
  selection effects at full density are unverified.
- Exposure-cap backtest assumes frictionless hedging; real spread/slippage
  reduces the benefit.
- Copy trader remains paper-mode; never live-tested.
