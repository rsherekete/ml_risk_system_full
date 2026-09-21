# ZFX Risk Intelligence

This repository holds two things that grew out of each other. Most of it is now
the **web application**; the notebooks it started as are still here and still
used for ad-hoc work.

| Start here | For |
|---|---|
| **[docs/MODULES.md](docs/MODULES.md)** | **What the system does, module by module** — the engines, both halves of the product, every tab, and where the data comes from |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Installing it, the config files, and the access-control traps |
| [BETA.md](BETA.md) | What the cut-down beta profile exposes, and why |
| The table below | The notebooks and the `trading_data` client library |

Run the app with `start_dev.ps1` (full application, port 3302) or
`start_beta.ps1` (beta profile, port 3310).

---

# Notebooks

Ad-hoc data analysis.

| Path | What it is |
|---|---|
| `server.yaml` | Connection settings — host, port, user, **password**, database — one entry per server. Git-ignored. |
| `symbol_map.yaml` | Optional raw-symbol → canonical-symbol overrides layered on the built-in table. Not git-ignored (no secrets); absent by default. |
| `mysql_to_dataframe.ipynb` | MySQL -> SQL query -> `pandas.DataFrame`: picks a server from `server.yaml`, then ad-hoc SQL and the typed client methods. |
| `trading_data/config.py` | `server.yaml` / `symbol_map.yaml` lookup and parsing (`find_config`, `server_names`, `server_settings`, `load_symbol_aliases`). |
| `trading_data/trading_data_client.py` | `TradingDataClient` — owns the engine/pool; `from_yaml`, `query`, `query_chunked`, `execute`, `tables`, `columns`. |
| `trading_data/enums.py` | The one vocabulary both platforms are translated into — request kinds, order kinds, source flags, volume units. Read it before comparing MT4 and MT5 numbers. |
| `trading_data/mt4_data_client.py` | `MT4DataClient` — MT4 database: `get_trade_requests()`, `get_trade_records()`, `symbol_specs()`. |
| `trading_data/mt5_data_client.py` | `MT5DataClient` — MT5 database: `get_trade_requests()`, in the same columns as the MT4 one; `symbol_specs()`. |
| `trading_data/research.py` | Symbol-canonicalisation, account features, recommendations, and the OOS backtest / efficient-frontier scenario engine. |
| `trading_data/risk.py` | Daily P/L drivers (biggest gross/net notional accounts and symbols) and Value-at-Risk (parametric, historical, correlation-adjusted portfolio) built from trade-implied prices. |
| `trading_data/book_assignment.py` | Point-in-time A-book/B-book routing (`assign_books`) and the actual firm P&L, equity curve, and drawdown it would have produced (`firm_daily_pnl`). |
| `symbol_map.example.yaml` | Template for `symbol_map.yaml` — copy it to add a venue-specific symbol alias without a code change. |

## Running

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy server.example.yaml server.yaml     # then fill in host / user / password
jupyter lab
```

Start Jupyter **in this directory** — the notebook imports `trading_data` from
the working directory.

## Web app on a fresh clone

The FastAPI app (`webapp/`) needs three things the repository cannot carry:

1. `server.yaml` — copy `server.example.yaml` and fill in the MySQL settings.
2. The antifraud data files, which are far over GitHub's 100 MB commit limit and
   travel as **release assets** instead (release `data-2026-09-14`):
   `webapp/artifacts/ad/model_frame.parquet` (the account-day corpus the
   antifraud tab loads), `model_features.csv`, `bq_90d_records.parquet` (the
   BigQuery snapshot the daily refresh builds on) and
   `markout_all_servers.parquet` (markout panels). Pull them into place with

   ```
   python tools/github_data_assets.py fetch      # token with Contents: read
   ```

   or unzip `antifraud_data_2026-09-14.zip` over the repository root. The app
   reads them from `webapp/artifacts/ad` (override with `NOTEBOOK_DATA_DIR`).
3. Nothing else: the Vantage copy-trading engine is a separate private
   repository. Without it the `/vantage` page answers "engine not installed",
   the alert and assistant engine metrics report the same, and every other tab
   works.

Then `uvicorn webapp.main:app --port 8000`.

## server.yaml

Everything needed to connect, password included. The file is git-ignored, so
credentials stay local.

```yaml
default: mt4_live01        # used when from_yaml() is called without a name

defaults:                  # merged into every entry; an entry's own key wins
  port: 3306
  user: ecn
  password: ecn

servers:
  mt4_live01:
    host: 127.0.0.1
    database: mt4_live01
  ts_uat:
    host: 10.135.51.17
    user: reader
    password: change-me
    database: tradeserver
```

A single-server file can skip `servers:` and put `host/port/user/password/database`
at the top level. Allowed keys per server: `host`, `port`, `user`, `password`,
`database`, `connect_args`, `echo` — anything else is rejected, so a `passwd:`
typo fails loudly instead of connecting without a password.

Lookup order: an explicit `path=` argument, `$TRADING_DATA_CONFIG`,
`./server.yaml`, `Tools/Analysis/server.yaml`.

```python
from trading_data import MT4DataClient

mt4 = MT4DataClient.from_yaml("mt4_live01")               # by name
mt4 = MT4DataClient.from_yaml()                           # the file's `default:`
mt4 = MT4DataClient.from_yaml("mt4_live01", database="fbr_live01")   # override a key
mt4 = MT4DataClient.from_yaml(path="D:/configs/server.yaml")
```

`TradingDataClient.from_env()` (reading `MYSQL_HOST` / `MYSQL_PORT` / `MYSQL_USER`
/ `MYSQL_PASSWORD` / `MYSQL_DB`) and `from_engine(engine)` remain for cases where
a config file is in the way, e.g. a scheduled job taking credentials from the
environment.

Clear cell outputs before committing a notebook — query results routinely
contain production data.

## Daily research workflow

`mysql_to_dataframe.ipynb` reads every configured MT4/MT5 database, adds explicit
`database`, `platform`, and `account_key` provenance, builds a 60-day account
lookback, and shifts features to the next decision day. A recommendation is
always `review_insufficient_data`, `A_review`, `B_review`, or `review`; it is not
an execution instruction. The current rules are a baseline for comparison, not
a claim of profitability. Use chronological walk-forward tests with future
realized P/L as the label before changing thresholds or introducing a model.

Run the local dashboard with:

```
streamlit run dashboard.py
```

The dashboard is read-only decision support. Keep the MySQL account SELECT-only,
rotate any credential that has been exposed, and do not store notebook outputs
containing production data in source control.

### Dashboard tabs

| Tab | What it answers |
|---|---|
| 📒 Book assignment | The flagship view: which accounts should be A-booked (hedged, no market risk, markup-only economics) vs. B-booked (unhedged, firm profits from the client's own loss) today or any day, the firm's real equity curve and max daily drawdown under that policy, top winning/losing clients by real realised P/L, two clean **A-book**/**B-book CSV exports** ranked by impact (ready for dealing/ops use), and a separate filterable exploration table. See below. |
| 📅 Daily drivers | What actually drove the selected day's result at the *symbol* level: real client P/L, firm proxy P/L, the biggest gross- and net-notional accounts, and canonical-symbol exposure. `Compare to another day` in the sidebar renders the same block for a second, arbitrary historical day. |
| 🔤 Symbol mapping | Every raw symbol traded, its canonical bucket, and whether contract size/mapping came from the server or a fallback — audit this before trusting any other tab's "by symbol" aggregation. |
| 📉 VaR & exposure | Parametric and historical VaR per canonical symbol at 1/7/20-day horizons, plus correlation-adjusted portfolio VaR (undiversified vs. diversified). |
| ⚖️ Efficient frontier | The **real** dollar Pareto frontier — `assign_books` + `firm_daily_pnl` swept across every profit/drawdown preference — the sidebar's sliders move along this exact curve. See below. |
| 🧾 Recommendations | The older, single-day heuristic A/B/review queue — kept as a simple, explainable baseline distinct from the model-driven Book assignment tab. |
| 🕵️ Client intelligence | Behavioral screening, unchanged. |
| 🧪 OOS validation | Classifier-quality diagnostics (precision/recall/base rate of the loss-probability model) — not a dollar estimate; see below. |
| 🏦 Firm risk | Event-flow gross/net notional by day and **canonical** symbol. |

### Book assignment and firm P&L

`trading_data/book_assignment.py` answers the question the rest of the
package builds toward: given only what was knowable at the time, which
accounts should have been A-booked vs. B-booked, and what firm P&L would that
have produced? **B_BOOK is the default** — absent evidence an account is
expected to make money, the firm takes the other side of it, and only hedges
out (A-books) the accounts it has specific reason to expect will win.

**`assign_books` routes via a risk-BUDGET walk, not a probability threshold.**
An earlier version derived a single classifier-probability cutoff
(`loss_threshold`) directly from the profit_weight/drawdown_weight sliders and
moved it as the dial moved — which meant *which accounts counted as good*
changed every time the dial moved. Confirmed on real 90-day production data:
raising profit_weight pushed accounts the model still expected to *win* out of
A_BOOK into the B_BOOK default (a lower `loss_threshold` is a *stricter* bar
to clear for a hedge), which mechanically **reduced** realised firm P&L —
`total_firm_pnl_usd` rose then *fell* across the sweep, the opposite of what
"more profit-seeking" should ever do. The fix replaces the threshold with a
fixed ranking plus a growing budget: every non-override candidate is scored
*once* per decision day by `excess_value_usd - policy.ranking_risk_aversion *
sigma_usd**2` (a Markowitz-style risk-adjusted dollar value — `excess_value_usd`
is `-model_expected_client_profit - markup_usd`, the dollar case for
B-booking over hedging; `sigma_usd` is `expanding_account_pnl_volatility`'s
notional-scaled EWMA P&L-rate estimate). This ranking never depends on the
dial. `risk_appetite_from_weights(profit_weight, drawdown_weight)` (just
`alpha = p/(p+d)`) controls how far down that *fixed* ranking the day's
B-book extends, admitting accounts while the cumulative `sigma_usd**2`
consumed stays within `alpha` times that day's total eligible risk. Because
the ranking never changes and the budget only grows with alpha, the admitted
set is **nested** — it only ever grows as the dial moves toward
"more profit-seeking" — which makes both expected firm P&L and the risk
budget consumed provably monotonically non-decreasing (a real proof: adding
an account to a strictly-positive-`excess_value_usd`-only set can only add a
positive amount to the P&L sum and a non-negative amount to the variance sum;
see `assign_books`'s docstring for the full argument).

**The scope of that guarantee is stated honestly, not oversold.** It covers
*expected* firm P&L (against the model's own `model_expected_client_profit`)
and the *ex-ante*, undiversified risk budget the walk enforces. It does
**not** cover: (a) the *realised*, path-dependent max drawdown
`firm_daily_pnl`/`real_efficient_frontier` plot from one particular
historical equity curve — that is a minimum over cumulative sums with a
different slope per day, which is concave (not monotonic) in any per-account
dial, for *any* routing rule, a fact confirmed independently for three
candidate designs (mean-variance, mean-CVaR, and Kelly position-sizing) during
this redesign; (b) protection against a mis-calibrated or regime-lagging
model — an expanding-window regression that hasn't caught up to a recent
shift can still make realised P&L worse as more of its (wrong) high-scoring
accounts get admitted, which is a fundamental limit of routing by an
estimate, not a defect specific to this formula. Both are disclosed in the
dashboard captions next to the numbers they affect, rather than left
implicit.

Two point-in-time (no-lookahead) ingredients feed the ranking, computed the
same walk-forward way `research.fit_oos_predictions` already did —
`research.fit_oos_expected_value`'s single-day-ahead *dollar* forecast (a
Ridge regression on `supervised_dataset`'s continuous `target_profit`, not
just a loss-probability classifier — replacing direction-only with
magnitude), and `expanding_client_profile`'s *persistent* trader-type signal,
accumulated day-by-day so a decision for day D never sees day D+1 or later. A
demonstrated persistent edge or arbitrage pattern always wins (A-book: hedge a
proven-profitable or latency-exploiting client, never take the other side of
them, whatever the risk budget says) — this override is unchanged from the
original design. `firm_daily_pnl` turns the routing decision into actual daily
P&L — A-book from markup economics only (no client-P&L exposure, since it's
hedged), B-book from the negative of the client's *real* realised P&L (not a
placeholder proxy) — and an equity curve, max/current drawdown, win-day rate,
and a Sharpe-like ratio.

The original classifier path (`research.fit_oos_predictions`/`predict_live`,
`research.coupled_scenario`, `research.classifier_performance`,
`research.evaluate_scenario`/`efficient_frontier`) is kept exactly as it was —
a separate calibration diagnostic (the OOS validation tab), not part of real
routing any more. `research.risk_appetite_from_weights` is the new, much
simpler function real routing reads instead of `coupled_scenario`.

**Every account with any activity in the window gets a book on every active
day, including its very first one — the invariant "A-book accounts + B-book
accounts == total active accounts" holds by construction.** This wasn't
always true: `firm_daily_pnl`'s as-of join (below) requires a *prior* routing
decision, and an account's first-ever activity day in the window has none —
an earlier version of this code silently dropped that row (`dropna`) instead
of giving it a decision, which meant every account's first day (and any
account whose *entire* window activity was a single day) was invisible to the
Book Assignment tab, its CSVs, and the firm P&L, despite being fully counted
in "All known accounts." It's now defaulted to B_BOOK
(`reason_codes == "NO_PRIOR_DECISION_DEFAULT_BBOOK"`) instead of dropped.

`research.predict_live`/`predict_live_expected_value` fill the other gap this
leaves: `supervised_dataset` correctly drops each account's most recent day
(it has no future outcome to label yet), so `fit_oos_predictions`/
`fit_oos_expected_value` alone have no row — and therefore no routing
decision — for today. Each fits one model on the entire labeled history and
scores exactly each account's own single most-recent dropped row (never a
stale, long-inactive account's), kept structurally separate from any backtest
P&L (its rows carry no label at all). Their data-sufficiency check (enough
distinct labeled days; the classifier twin also needs both outcome classes
present) is necessarily firm-wide, not per-account — but an earlier version
let that firm-wide check decide whether *any* account got a live row at all:
when the check failed, the function returned a completely empty frame,
silently omitting every account needing a live decision, not just the ones
actually short on history. Both now always return one row per account still
needing a decision; only when the model can't be trusted does
`model_probability_loss`/`model_expected_client_profit` get left `NaN`
(and `confidence` 0, for the classifier) instead of the whole account
vanishing — `assign_books` already has a defined fallback for "no
prediction," but only for a row it actually receives.

Two constant-policy baselines make the model's actual value visible instead
of assumed: `firm_daily_pnl_fixed_book(records, "A_BOOK", economics)` and
`...("B_BOOK", ...)` compute what the firm's daily equity would have been if
every account had always been hedged, or always taken the other side of,
with no routing decision at all. The Book Assignment tab plots these
alongside the real ML-routed equity curve — if the model-routed line doesn't
clearly beat both flat-policy baselines, it isn't adding value over the
simplest possible rule.

`firm_daily_pnl` scores a day's real P&L against the **most recent** routing
decision made strictly before it — an as-of match (`pd.merge_asof`,
`direction="backward"`, `allow_exact_matches=False`), not an exact
day-minus-one lookup. This matters in practice, not just in theory: an exact
day-1 join was tried first and immediately collapsed real-data account
coverage from ~1,000/day down to single digits, because almost no real
account trades on two literally consecutive calendar days (weekends, and
plenty of accounts that simply don't trade daily). A routing decision stays
in effect until a newer one supersedes it, exactly like a real policy would
work, while `allow_exact_matches=False` still guarantees the matched decision
predates the activity it's scored against — the same no-lookahead property
`research.evaluate_scenario` gets from scoring `decision_day=D` against
`target_profit` (D+1's outcome), never D's own.

`research.supervised_dataset`'s next-day label has the same real-data lesson
baked in: it is looked up within `max_label_gap_days` (default 7) calendar
days, not by an exact next-day match (which silently dropped the training row
before almost every weekend and starved the walk-forward model of most of
its data) and not by blind row-sequence position (which would mislabel an
account that goes quiet for months with a much-later, unrelated outcome).

The four client-categorisation screens behind this (and behind the
**Client intelligence** tab's whole-history view) are independent, separately
testable functions in `research.py`: `detect_edge_clients`, `detect_toxicity`,
`detect_arbitrage`, `detect_fraud_review` — the last being a compliance/ops
screen only, not a book-routing input.

### Real efficient frontier vs. the classifier-proxy backtest

Two different things now use the phrase "profit/drawdown frontier," and they
answer different questions:

- **Efficient frontier** tab / `book_assignment.real_efficient_frontier` --
  sweeps `assign_books` + `firm_daily_pnl` (the same real, dollar simulation
  as the **Book assignment** tab) across every profit/drawdown preference.
  This is the actual firm-economics frontier the sidebar sliders move along.
  Since `assign_books`'s risk-budget-walk redesign (see "Book assignment and
  firm P&L" above), `total_firm_pnl_usd` is provably monotonically
  non-decreasing across this sweep; `max_daily_drawdown_usd` is not, and the
  tab's caption says so.
- `research.efficient_frontier` / `research.oos_backtest` / `evaluate_scenario`
  -- sweep a much cheaper classifier-only proxy (B-book proxy P/L =
  `-target_profit`; A-book residual = `-target_profit * 0.10`) used for
  calibrating the model/threshold itself, retained for `daily_ab_book_research.ipynb`'s
  existing research workflow. That 10% figure was never meant to model real
  A-book economics (which are markup-only, not a fraction of client P/L) --
  it is a classifier-calibration convenience, and the dashboard no longer
  presents it as if it were a dollar estimate.

The **OOS validation** tab reflects this split: it no longer shows a dollar
equity curve at all, since **Book assignment**'s is the real one. Instead it
shows `research.classifier_performance` -- precision, recall, and the actual
base rate of the loss-probability model at the sidebar's current threshold,
plus how those trade off across the full profit/drawdown sweep. That is a
genuinely different question from "what would the money have been" ("how
good is the model at predicting tomorrow"), and conflating the two under one
misleading dollar figure was the original problem.

### Top client profitability contributors

The **Book assignment** tab also ranks, for the selected day, the biggest
winning and biggest losing clients by *real, realised* P/L side by side — the
account-level complement to the **Daily drivers** tab's by-symbol breakdown.
A big winner is exactly the profile `detect_edge_clients`/`expanding_edge_flag`
is built to catch (route to A-book, hedge it); a big loser is the flow a
B-book policy exists to profit from. Seeing both together, for the same day,
is the fastest way to read "what actually drove today's number."

### Why Streamlit

Streamlit stays the right tool here: this is an internal, read-only,
Python-native decision-support app for a small desk audience, iterating
against a research codebase that already lives in this repo — a bespoke
frontend would be substantial extra engineering (a build pipeline, an API
layer, auth) for a use case Streamlit's script-per-rerun model already
serves well. What changes is craft, not framework: `.streamlit/config.toml`
sets a dark theme by default, every chart uses one validated,
colorblind-safe palette via Altair (its dark-surface steps, to match) rather
than ad hoc defaults, KPIs render as bordered, monospace-numeral card tiles,
and interactive filtering plus CSV export give the ranked tables a proper
analyst workflow.

### Symbol mapping

`canonical_symbol` (in `trading_data/research.py`) merges a broker's raw
symbol names — decorated with suffixes like `.raw`, `.pro`, `m`, `c`, or a
house alias like `GOLD` — into one canonical name, so every "by symbol"
aggregation in the dashboard groups the same instrument together regardless
of which server or account type it traded on. An explicit alias always wins;
otherwise recognised decorator tokens are stripped from the end of the raw
name. A raw symbol the heuristic gets wrong belongs in `symbol_map.yaml`
(copy `symbol_map.example.yaml`), not in a more aggressive stripping rule —
guessing too eagerly risks merging two different instruments. The **Symbol
mapping** tab is the audit view for this: it shows every raw symbol actually
traded, its canonical bucket, and where its contract size came from.

Every aggregation that groups "by symbol" must key on `canonical_symbol`, not
the raw `symbol` column -- `firm_risk_timeseries` (the **Firm risk** tab) was
found grouping by raw `symbol`, so `EURUSD`/`EURUSDe`/`EURUSDmin` appeared as
three separate rows there even though every other symbol-grouped function
already did this correctly. Fixed by grouping on `canonical_symbol` like the
rest of the package.

Two real-data lessons on the stripping heuristic itself, found by inspecting
this venue's actual raw symbol list rather than guessing further: the
length guard on a glued (no-delimiter) suffix strip must be evaluated
per-token, not with one blanket rule -- a single relaxation broad enough to
let `NGmin` merge into `NG` (natural gas) also broke `INTC` (Intel) into
`INT`, since a bare single-letter suffix (`E`/`M`/`C`) is far more ambiguous
on a short root than the distinctive three-letter `MIN` is.
`_GLUED_SUFFIX_MIN_REMAINING` now keeps a stricter floor for the single-letter
tokens and a looser one for `MIN`/`.E`. `CU` (this venue's alternate
commodity code for `COPPER`) and a trailing lowercase `t` on `XAUUSDt` needed
explicit `SYMBOL_ALIASES` entries instead, since no safe general stripping
rule covers either without risking a false merge elsewhere.

Lot-based sizing is deliberately avoided: `attach_symbol_specs` pulls
`contract_size` from each server's own `symbols` table first, and only falls
back to a conventional, asset-class-based contract size (`CONTRACT_SIZE_FALLBACK`
/ `SYMBOL_CONTRACT_SIZE_FALLBACK`) when the server has no spec for a canonical
symbol — flagged via `contract_size_source` so a fallback is never silently
mistaken for a venue-confirmed figure.

### VaR and the profit/drawdown frontier

`trading_data/risk.py` builds a trade-implied daily mark price per canonical
symbol (volume-weighted from trade open/close prices — there is no market
data feed wired in here), estimates EWMA daily volatility with a
published-fallback for thin history, and reports both parametric
(`z · σ · √horizon · exposure`) and historical (empirical quantile of trailing
returns) VaR at 1/7/20-day horizons, gross and net, per symbol —
`symbol_var` — and firm-wide with a correlation adjustment — `portfolio_var`.
Both are monitoring proxies, not a funded position ledger or regulatory
capital figure, until validated against real position and price data.

The sidebar's profit-maximisation and drawdown-minimisation sliders are
linked (`profit_weight + drawdown_weight == 100` always) and feed
`coupled_scenario`, which is scored against the walk-forward OOS predictions
via `evaluate_scenario`. `trading_data.research.fit_oos_predictions` runs the
expensive model refit once per dataset; `evaluate_scenario` re-scores a
scenario against those same predictions with no refit, so a slider move is
cheap. `efficient_frontier` sweeps that same scoring function across every
profit/drawdown split and marks the Pareto-optimal points — the curve the
**Efficient frontier** tab plots, and the curve the sidebar's current
selection is always a single point on.

## MT4/MT5 data notes: fields, quirks, and known limitations

Researched and applied while building the risk system, since a broker-data
analytics product is only as correct as its handling of these platform
quirks. Kept here as the durable reference rather than scattered across
docstrings.

**Commission, swap, and taxes/fee are separate fields `profit` never
includes.** `MT4DataClient.get_trade_records()` now selects `commission`,
`swap`, and `taxes`; `MT5DataClient.get_trade_records()` selects `commission`,
`swap`, and `fee`. On **both** platforms the real database column behind
"swap" is named `storage`, not `swap` — confirmed against this deployment's
actual `traderecord`/`dealrecord` schemas via `client.columns("traderecord")`
after an initial guess at the MT5 column name (`d.swap`) broke every MT5
database outright with `Unknown column 'd.swap'`; `commission`/`taxes`/`fee`
were correct on the first try. Both clients add a computed `net_profit` = the
sum of all of them — the client's true realised economic result — and every
P/L aggregation in `trading_data` reads `net_profit` in preference to raw
`profit` (via `research.realised_trade_pnl`). Using `profit` alone understates or
overstates every realised-P/L figure in the system by whatever was charged in
commission, swap, or tax on that position.

**Money operations (deposits, withdrawals, credit, commission rebates, ...)
are booked as ordinary rows in the same trade/deal tables, and cannot be
filtered out by `state` alone.** MT4 books a manual balance/credit adjustment
with `cmd` = `balance`/`credit`; MT5's `DEAL_TYPE_BALANCE` and neighbouring
action codes cover commissions, dividends, and stop-out compensation too.
Both platforms give these rows whatever `state`/`entry` a one-shot ledger
entry gets — commonly the same value a genuinely closed trade would have, and
for MT5 explicitly "broker-server-setup-dependent" per MetaQuotes' own
documentation. `research.is_realised_trade` (and `realised_trade_pnl` built
on it) therefore gates on **both** `state` and `cmd` — see
`enums.MONEY_OPERATION_KINDS` / `enums.TRADE_KINDS` — never `state` alone.
Every aggregation in `research.py`, `risk.py`, and `book_assignment.py` was
audited against this; a deposit or commission rebate can no longer be counted
as trading P/L (confirmed with a synthetic $50,000 "deposit" injected into
the test data, which the pre-fix code would have added straight into
`realised_profit`).

**A `close_time`/`close_ts` of zero (sometimes displayed as `1970-01-01`,
the Unix epoch) means the order is still open, not that it closed at
midnight on January 1st, 1970.** Both `MT4DataClient` and `MT5DataClient`
already convert this to `NaT` before it reaches a caller — `MT4DataClient`
explicitly (`close_ts.replace(0, pd.NA)`), `MT5DataClient` structurally (the
`CASE` expression only populates `close_time` for an `OUT`/`OUT_BY` deal).
Nothing downstream should ever see a raw zero timestamp; if one appears, that
is itself a bug report.

**Partial closes and `close_by` are structurally different between
platforms.** MT4 is (by convention on most venues) a hedging model: every
trade is its own ticket, and a partial close just shrinks that ticket's
remaining `state=open` row while writing a separate `closed_part` row for the
piece that closed. MT5 is commonly run as a **netting** model instead: only
one position per symbol exists per account, and a partial close is booked as
an opposite-direction deal of the reduced volume that nets against the
existing position — which is exactly why `mt5_data_client.py`'s own docstring
warns that netting volume by side gives a different answer on the two
platforms unless only opening rows are netted. This package's event-level
notional/exposure functions (`rolling_symbol_exposure`, `symbol_var`,
`daily_account_exposure`, ...) sum *signed* per-event notional rather than
maintaining a position ledger, which happens to net out correctly under
either convention for a given day's flow — but a true intraday open-position
snapshot (for margin/floating-P&L purposes) would need a live positions feed,
not this archive.

**`close_by` leaves a remainder position stamped with an old open time.**
MT4's close-by nets two opposing trades against each other and creates a
leftover position carrying one of the original trades' *opening* timestamp,
not the moment of the close-by itself — a known regulatory-reporting pitfall
for T+1 submission regimes, and worth remembering before treating `open_time`
as "when this economic exposure began" for a `close_by` remainder.

**Broker server time is not reliably UTC.** MetaTrader servers run on
whatever timezone the broker configured — GMT, GMT+2/+3 (common), NY+7, or a
fixed offset that may or may not observe DST — and every timestamp this
package reads (`timestamp`, `open_time`, `close_time`, and therefore every
`day`/`decision_day` bucket derived from them) is naive, in that server's own
time. "Daily" P/L, the equity curve, and VaR windows all use `.dt.floor("D")`
on these naive timestamps, so a day boundary here is the **broker's**
midnight, not UTC's or the viewer's. Confirm each configured server's actual
offset (compare its `ping()`/`NOW()` reading against a known UTC clock) before
treating day boundaries as timezone-equivalent across multiple servers with
different broker configurations, and before comparing "daily" figures against
any UTC-anchored external report.

**Account currency is not necessarily USD.** `profit`/`net_profit` are
booked in each account's own deposit currency, and multi-currency account
groups are common. Nothing in this package currently pulls the account's
currency or converts between currencies — every dollar figure implicitly
assumes the configured servers are single-currency (or already
USD-denominated) books. If a deployment mixes account currencies, every P/L,
notional, and VaR figure here needs an FX conversion step (using a rate as of
each trade, not today's rate) before it can be summed across accounts; that
conversion is not yet built.

**MT4's `userinfo` is a per-transaction snapshot log, not a one-row-per-account
registry — do not `COUNT(*)` it and call the result "registered accounts."**
Its primary key is `(ts, sequence)`, the same ARS-archive shape as
`traderecord`; a single login can carry thousands of historical rows (one
live deployment showed 1.6M `userinfo` rows behind just 231 distinct logins).
MT5's `accounts` table is the opposite — `login` is its actual primary key,
one row per account — so this trap is MT4-specific. Consequently, joining
`userinfo` against a date-filtered `traderecord` to resolve account groups
multiplies two large row sets together and can take minutes even for a
ten-day window; `research.account_group_lookup` avoids this by never joining
— it takes the small, already-known list of distinct logins active in a
window and does a single bounded `login IN (...)` point lookup against the
group table instead, chunked at 5,000 logins per query.

**"Distinct accounts that traded" is a definitional choice, not a fixed
number, and the platforms give you no way to avoid picking one.**
`traderecord`/`dealrecord` are event logs — a row is written on every open,
SL/TP modify, and close of an order — so "any row in the window" (which is
what this dashboard counts, by design: see the "All known accounts" caption)
counts an account the moment it has *any* activity, including a position
still open with no realised outcome yet. A system that instead counts only
accounts with a **closed, realised** trade in the same window will show a
much smaller number for the identical period — in one live sample, distinct
accounts with any row outnumbered distinct accounts with a closed trade by
roughly 23x, simply because most open positions in a short window haven't
closed yet. That gap is expected, not a sign that rows are being dropped or
mis-joined; it only becomes a bug if the two systems are silently assumed to
mean the same thing. Separately (and unconditionally worth excluding either
way): `research.non_client_logins` flags logins whose account `group` matches
`enums.NON_CLIENT_GROUP_MARKERS` (currently just `"test"`, kept narrow so a
demo *server*'s legitimate `demo_*` client groups are never caught) —
`load_records` in `dashboard.py` drops these before any account count is
computed. Coverage is best-effort, especially on MT4: a login only resolves
to a group if a `userinfo` snapshot happened to capture it, so most currently
active MT4 logins won't resolve to a group at all and are left in (never
excluded on missing data, only on a positive test-group match).

**Not yet pulled, but worth adding for a more accurate risk picture:** MT5's
`accounts` table (one row per login) would give an authoritative
`balance`/`equity`/`margin`/`leverage` reading rather than one reconstructed
purely from trade deltas — useful as a reconciliation check against this
package's own realised-P/L sums. The equivalent reconciliation on MT4 would
need the *latest* `userinfo` row per login (by `ts`/`sequence`), not a raw
pull of the table, given the snapshot-log shape described above. MT5's
`GwOrder`/`GwVolume` (and MT4's already-pulled but unused
`gw_open_price`/`gw_close_price`) would let an A-book's assumed
markup/LP-cost economics be replaced with the venue's *actual* hedge fill and
cost, instead of the `Economics` bps scenario this package uses today.

## Adding a query

Put it on a client rather than leaving it in a cell:

```python
class MT4DataClient(TradingDataClient):
    def get_something(self, start_time: datetime, symbol: str | None = None) -> pd.DataFrame:
        sql = "SELECT ... WHERE OPEN_TIME >= %(start_time)s"
        params: dict[str, Any] = {"start_time": start_time}
        ...
        return pd.read_sql(sql, self.engine, params=params)
```

SQL is passed to the driver untouched (pandas uses `exec_driver_sql` for plain
strings), so placeholders are pymysql-style `%(name)s` and a literal percent sign
has to be doubled (`LIKE 'EUR%%'`). Never format values into the SQL string.

A new database gets its own `TradingDataClient` subclass here plus an entry in
`server.yaml`. The notebook runs `%autoreload 2`, so edits are picked up without
a kernel restart.
