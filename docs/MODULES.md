# What this system does, module by module

A risk-analytics application for a retail FX/CFD broker. It answers one
commercial question in several forms: **for each client, each trade and each
instrument, is the firm better off taking the other side (B-book) or passing the
flow through (A-book) — and which clients are winning by means the firm should
act on rather than accept?**

Everything else is in service of that: the pipeline that makes the question
answerable, the models that answer it, and the screens that let a desk act on
the answer with the evidence attached.

**Contents** — [Instances](#two-instances-one-codebase) ·
[Engines](#the-anti-fraud-engines) · [Trading tabs](#trading-module-per-client-decisions) ·
[Quant tabs](#quant-module-per-trade-decisions) · [Standalone pages](#standalone-pages) ·
[Data pipeline](#where-the-data-comes-from) · [Module reference](#complete-module-reference) ·
[Access control](#access-control)

---

## Two instances, one codebase

The same code serves both; `AF_BETA=1` selects the cut-down profile.

| | Port | Serves | Writes? |
|---|---|---|---|
| **Development — "the master app"** | 3302 | Everything in this document | Yes — scans, warehouse top-up, copy trader |
| **Beta** | 3310 | Latency Arbitrage, Toxic Flow, Account Detail only | **No** — renders what dev writes |

The beta is deliberately a reader. The tick store is DuckDB and allows a single
writer, so a second scanning instance would invalidate the tape for both. Route
access is default-deny: anything not explicitly allowed redirects to the beta
home, so a component added later cannot appear in the beta by accident.

## Two halves: Trading and Quant

The product splits along the **unit of decision**, and users are granted one or
both (the `views` permission):

- **Trading** — decisions about **clients**. Route this account or not.
- **Quant** — decisions about **trades**. Copy this individual trade or not.

They share the shell, the exposure/regions/cashflow screens and the risk
monitor; they differ in what the model scores. Where a tab appears in both, it
is the same screen fed by a different model — noted per tab below.

---

# The anti-fraud engines

Three engines over a 7-day rolling window, rescanned every 20 minutes
(`auto_rescan_minutes`). Each scan re-measures **every** trade in the window;
results are cached whole, not per order.

### Engine A — Latency Arbitrage
Fast, transient price exploitation: fills instantly profitable against an
independent reference market whose advantage then **decays**. The decay is the
signal — an edge present at 100 ms and gone by 60 s is a speed advantage, not a
view on the market. Produces flagged orders, an economic impact figure, and a
share of the client's winning-trade profit.

- `latency_arb.py` — detection, profiling, decisioning and control
- `latency_spec.py` — spec scoring
- `latency_reference.py` — independent reference feeds (Vantage Raw ECN, IC Markets), cached to parquet per symbol-hour
- `latency_tags.py` — stable automation tags

### Engine B — Toxic Flow
**Persistent** adverse selection rather than transient. Where latency arbitrage
fades, toxic flow keeps building — informed or directional flow. Scores only
trades with tick coverage.

By construction every latency event is also "sharp/fast" toxic flow; a
"persistent" toxic trade is never a latency event. **The two impact figures must
never be added together.**

- `toxic_flow.py` — scan, persistence, audit and automation feed
- `toxic_spec.py` — metrics, score, confidence, profiles, decision state
- `toxic_tags.py` — automation vocabulary

### Engine C — AntiFraud / Behavioural Profiling
Classifies clients into behavioural profiles — persistent edge, toxic flow, high
magnitude, scalper, news/event/vol, bonus arbitrage, swap arbitrage, martingale,
high exposure/recovery — each scored on five axes (score, confidence, stability,
severity, evidence) with a priority. Empty means clean under current rules.

- `antifraud.py` — the spec, implemented; also client screening and action testing
- `af_registry.py` — dynamic category registry: categories can be added, activated and given their own ML model without code changes
- `rule_models.py` — the generic per-category rule engine and its models
- `rule_forecast.py` — early warning: who will *qualify* for each rule in the next five days

---

# Trading module (per-client decisions)

### Summary
One page for management. Four questions: what did the book make, what is the
model worth, where is the risk, what needs attention.

### Overview
The client-level routing model measured against B-booking every account — the
counterfactual that justifies the model existing. Daily P&L, model vs B-book.

### A-Book Manifest
Accounts to hedge on the selected day, **ordered by impact on profitability**.
This is the operational output of the Trading half: the list a dealer acts on.

### Client Intelligence
The behavioural taxonomy across the routable population — edge, toxic,
arbitrage — with the evidence behind each label.

### AntiFraud
Behavioural profiling, screening and **action testing**: backtest a desk
intervention against one client's own trade history before applying it. Hosts
the Latency Arbitrage and Toxic Flow panes.

### Account Detail — the evidence screen
The page a desk opens to justify a decision about one client:

1. **Account P&L and flagged-flow impact** — floating P&L from open positions on
   the trade server, realised P&L from every closed trade on record after
   commission and swap, and each engine's flagged share and money. The server
   *balance* is deliberately unused: order compression books compressed trades
   into the balance while the trades stay in the database, so balance-based
   figures double-count.
2. **Trades** — every trade, latency-flagged in amber and toxic-flagged in
   violet. Clicking an order opens its **markout evidence**: the trade as booked,
   the broker's own tick tape and the independent reference at the fill and at
   seven spec horizons, with a CSV export. Built to be shown to a client.
3. **Model classification** — every profile the client currently qualifies for.
4. **Cumulative realised P&L**, **model score over time**, and a **price path**
   with the account's entries and exits marked — the chart that separates a
   client who is merely *profitable* from one who consistently *buys the low*.

### Exposure
Net notional by symbol, live and historical — what the firm is actually
carrying, in instrument units rather than dollars.

### Regions
Firm P&L and accounts by **client geography**, not by server. Servers and
countries do not align, and routing decisions are commercial decisions about
markets.

### Cash Flow
Deposits, withdrawals and transfers — overall, by region and by account — with
compliance flags. Funding behaviour is itself evidence: a client who withdraws
immediately after a win reads differently from one who compounds.

### Surveillance
Accounts worth investigating, each with the measurement that triggered it. The
patterns an anti-fraud desk needs to see, with evidence attached.

### Book Risk
Net exposure caps — the **drawdown layer**. This addresses risk by hedging the
*book*, not by selecting clients, and is deliberately separate from routing.

### Risk Monitor
Settled risk for a chosen day and the live event stream, **kept apart by
provenance** — settled figures and live figures are never blended. VaR and
concentration.

### Model Performance
Walk-forward diagnostics, model settings and retraining.

### Validation
Reconciliation, AUC and regime stability — evidence that the numbers on the rest
of the site can be believed.

---

# Quant module (per-trade decisions)

### Summary
As Trading: one page for management, same four questions, trade-level model.

### Overview
The trade-level routing model measured against B-booking every trade.

### Trade Signals
Individual trades to copy (A-book) rather than take the other side of, for the
selected day. The operational output of the Quant half.

### Copy Trader
MT5 execution of the Quant book — **expected risk against what the terminal
actually holds**. Reconciliation matters here: the model's intended position and
the broker's actual position drift, and the gap is the thing to watch.

### Exit Policy
Where a copied trade actually closes, and whether that beats mirroring the
client's own exit. Several exit answers measured against each other rather than
one assumed.

### Research
What the trade-level model is actually using and how it behaves by symbol —
feature importance, ablations, markout profiles.

### Strategy Lab
Live walk-forward results as they compute — the instrument for tuning, and for
running experiment variants side by side.

### Exposure · Regions · Cash Flow · Risk Monitor · Model Performance · Validation
The same six screens as Trading, described above, fed by the trade-level model
rather than the client-level one.

---

# Standalone pages

| Page | What it is |
|---|---|
| **Hub** (`/hub`) | Landing page and account search — the way into Account Detail |
| **Replay** (`/replay`) | The biggest or most suspicious recent client wins, as playable cards: watch the trade unfold against the price |
| **Executive** (`/executive`) | Firm-level view for management |
| **TAF** (`/taf`) | Toxic Account Forecasting — which accounts are becoming toxic, before they are |
| **Vantage** (`/vantage`) | Copy-trader control: state, diagnostics, reconciliation, S1 dump, kill switch |
| **Data** (`/data`) | What history is stored, how fresh it is, how to extend it. Manual warehouse refresh with a forced window |
| **Admin** (`/admin`) | Approve users, set permissions, manage credentials and sessions, the IP allowlist, and restart the instance |
| **Settings** (`/settings`) | Model configuration and retraining |

> **Copy trader, operationally:** `vantage.yaml` carries `mode: live` and the
> kill switch is **not persisted** — it returns disengaged on every restart.
> Check `/vantage` after restarting the app.

---

# Where the data comes from

```
MT4/MT5 production MySQL ──> mysql_extract.py     ──> warehouse/*.parquet (monthly)
Kafka (MT4 ms exec times)──> kafka_service.py     ──> live_stream.duckdb
Broker tick tables       ──> tick_bars.py         ──> minute bars
Reference venues         ──> latency_reference.py ──> reference_ticks/*.parquet
BigQuery (IB rebates)    ──> dwh.py
```

**The warehouse is filled by the top-up, not by the scans.** `refresh_recent()`
pulls the last two days from every server every **15 minutes** in the
background; the engines *read* what it has written. A stalled top-up therefore
shows up as engines scanning stale data rather than as an obvious error — check
`artifacts/warehouse_topup.json` for per-server failures.

Writes are atomic (write beside, then rename) and an unreadable partition is
quarantined and rebuilt rather than aborting the refresh.

**The application renders artefacts; it does not generate them on demand.** A
fresh deployment with an empty `webapp/artifacts/` shows empty screens. That is
not a fault — it needs a writer instance with database access, or artefacts
copied from one.

---

# Complete module reference

Every module in `webapp/`, grouped by what it serves.

### Engines and detection
| Module | Purpose |
|---|---|
| `latency_arb.py` | Latency-arbitrage detection, profiling, decisioning and control |
| `latency_spec.py` | Spec scoring for Engine A |
| `latency_reference.py` | Independent reference market feeds |
| `latency_tags.py` | Automation tags for Engine A |
| `toxic_flow.py` | Engine B: scan, persistence, audit, automation feed |
| `toxic_spec.py` | Engine B: metrics, score, confidence, profiles, decision state |
| `toxic_tags.py` | Automation vocabulary for Engine B |
| `antifraud.py` | Behavioural Profiling Engine v2 — the spec, implemented |
| `af_registry.py` | Dynamic anti-fraud category registry |
| `rule_models.py` | Generic per-category rule engine and ML models |
| `rule_forecast.py` | Predicts who qualifies for each rule in the next 5 days |
| `surveillance.py` | Desk-facing detections, with evidence |
| `taf.py` | Toxic Account Forecasting |
| `trade_markout.py` | Per-trade markout evidence for one order, measured as the engine measures it |

### Event impact
| Module | Purpose |
|---|---|
| `event_impact.py` | Who a market event hurt, helped, and what to do about each |
| `event_abuse.py` | Abuse detectors and client classification for that tab |
| `econ_calendar.py` | Dynamic economic-events calendar |

### Risk and exposure
| Module | Purpose |
|---|---|
| `exposure.py` | Net notional exposure by instrument |
| `exposure_days.py` | Active days redefined as **exposure** days |
| `exposure_policy.py` | Hedging the book rather than the client |
| `book_risk.py` | The drawdown layer: exposure caps and what capping is worth |
| `risk_monitor.py` | Historical and live book exposure |
| `validation.py` | Evidence that the rest of the site's numbers can be believed |

### Quant book and copy trading
| Module | Purpose |
|---|---|
| `vantage.py` | Admin-only copy trader: the quant model driving one external MT5 account |
| `copytrader.py` | MT5 copy-trading engine for the Quant book |
| `tape_model.py` | Short-horizon directional signal per instrument |
| `exit_policy.py` | When a copied trade closes — several answers, measured |
| `trade_features.py` | Per-trade features for the trade-level router |
| `path_features.py` | Per-trade MAE/MFE price-path excursions for training |

### Data pipeline
| Module | Purpose |
|---|---|
| `data_store.py` | Two-year warehouse: monthly partitions, overlap-safe incremental refresh, watermarks |
| `mysql_extract.py` | Pull trade history from MySQL — the primary source |
| `trade_feed.py` | Production trade feed from MySQL |
| `backfill_repair.py` | Fill holes in the warehouse, surviving an intermittent VPN |
| `dubai_backfill.py` | Backfill `mt5_dubai_live01` from BigQuery |
| `mt4_kafka_backfill.py` | Backfill MT4 millisecond trade-event times from Kafka's retained log |
| `mt5_pairing.py` | Rebuild MT5 round-trip trades from deal rows |
| `kafka_service.py` | Background Kafka consumer materialising the live stream into DuckDB |
| `tick_bars.py` | Minute bars from the MySQL tick table |
| `dwh.py` | BigQuery data warehouse access |
| `cashflow.py` / `cashflow_store.py` | Cash movements, and their conversion into point-in-time model features |
| `client_profile.py` | Client demographics: country, region, group, acquisition channel |
| `symbol_specs.py` | Contract specifications and USD conversion, per server |
| `ad_refresh.py` | Keeps the account-day corpus at yesterday, always |

### Models and serving
| Module | Purpose |
|---|---|
| `model_service.py` | Model configuration, background training, cached scored artefacts |
| `views.py` | Query helpers turning a cached score artefact into what a screen needs |
| `replay.py` | The biggest / most suspicious recent wins, as playable cards |

### Application, access and assistance
| Module | Purpose |
|---|---|
| `main.py` | The FastAPI application: routes, the beta gate, the IP gate, background loops |
| `auth.py` | Users, sessions, the approval gate, permissions |
| `agent.py` | LLM operations agent across the three data sources |
| `assistant.py` | Plain-language question console answered from the actual data |
| `alert_engine.py` | User-defined alerts: source → field → condition → webhook |
| `glossary.py` | Definitions surfaced as hover tooltips |

---

# Access control

Two independent layers, frequently confused:

1. **Network** — firewall or VPN routing. Decides whether a client can reach the
   port at all. A block here leaves **no trace in the application log**.
2. **IP allowlist** — checked before authentication, managed in the admin
   console (stored in `app.db`, applied within ~2 s without a restart). A
   rejection **always** logs `[ip-allowlist] blocked <ip> -> <path>` and returns
   403. It can only ever deny; it never grants access.

**If a user cannot connect and the log is silent, it is the network. If the log
shows a block, it is the allowlist.** That distinction resolves almost every
"I cannot connect" report.

Accounts are self-registered and inert until an admin approves them. Permissions
are two axes rather than a hierarchy: `role` (admin may approve users and switch
into any user's view) and `views` (Trading, Quant).

---

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for installation, configuration and the
credential files, and **[BETA.md](../BETA.md)** for what the beta profile
exposes and why.
