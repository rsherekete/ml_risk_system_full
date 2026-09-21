# Anti-Fraud Engine — beta instance

The beta is the cut-down build shared with internal teams. It runs the **same
codebase** as the development app, started with `AF_BETA=1`.

| | Development | Beta |
|---|---|---|
| Port | 3302 | 3310 |
| Start | `uvicorn webapp.main:app --host 0.0.0.0 --port 3302` | `.\start_beta.ps1` |
| Sections | all 14 trading tabs, quant, executive, TAF, copy trader | Latency Arbitrage, Toxic Flow, Account Detail |
| Background work | Kafka materialiser, scans, warehouse top-up, copy trader | **none** |
| Tick store | writer | does not open it |

## The three sections

- **Latency Arbitrage** — `/trading/antifraud?af=latency`
- **Toxic Flow** — `/trading/antifraud?af=toxic`
- **Account Detail** — `/trading/account`, including per-trade **replay** on the
  Order tab: the `▶` next to each order plays that trade back over minute bars
  from the broker tick tape, with the entry and exit flags revealing as the
  playhead reaches them.

## Why one codebase and not a fork

A forked copy would run its own scans and open its own handle on
`artifacts/live_stream.duckdb`. DuckDB permits a **single writer**: a second
scanner is exactly what invalidated the tape on 17 Sep 2026, which zeroed tick
coverage and produced a scan with no flagged trades at all. It would also double
the MySQL load and race the full instance on the same artefact files.

So the beta is a **reader**. Its startup hook returns before spawning any
background thread, and `refresh=1` on the latency and toxic endpoints is ignored
there. It renders the artefacts the development instance produces, which means:

> **Beta freshness is development-instance freshness.** If the full app stops
> scanning, the beta shows the last completed scan, not an error.

## How the trimming works

`BETA` in [`webapp/main.py`](webapp/main.py) reads `AF_BETA`. Three things follow:

1. **`_beta_gate` middleware** — default-deny on `BETA_ALLOWED`. Any path not
   explicitly allowed redirects to the beta home (or returns 404 for `/api/`).
   A component added to the full instance later therefore cannot appear in the
   beta by accident; it has to be allowed on purpose.
2. **`_startup`** returns immediately under `BETA`, before any thread starts.
3. **Templates** receive `beta`, `beta_tabs` and `beta_routes`, so `base.html`
   renders the three-item nav and no sidenav, and `antifraud.html` drops its
   section rail and opens on the single requested pane.

## Shared state, and what that means

The beta shares `app.db` (so logins are the same on both) and the detection rule
files. Rule edits and retraining are **enabled** in the beta by choice — which
means a threshold changed there changes what the development engine flags on its
next scan. If that is not wanted, restrict the `/api/antifraud/*` POST routes
under `BETA`.

## Before sharing more widely

The user table still holds only the seeded defaults, `admin/admin` and
`test/test`, and both instances now listen on `0.0.0.0`. Change those passwords
and add per-person accounts first.
