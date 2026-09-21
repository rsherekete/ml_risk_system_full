# Deploying the Anti-Fraud Engine (beta profile)

For the engineer standing this up on a server. It assumes no prior contact with
the codebase.

---

## 1. What "the beta" actually is

There is **no separate beta codebase**. The beta is this same application started
with one environment variable:

```
AF_BETA=1
```

That switches on a default-deny gate (`_beta_gate` in `webapp/main.py`) which
exposes only three sections — Latency Arbitrage, Toxic Flow, and Account Detail —
and redirects everything else. Without the variable you get the full application:
quant, executive, exposure, cash flow, admin.

The gate is an **allowlist of path prefixes** (`BETA_ALLOWED`). Anything added to
the app later cannot appear in the beta by accident; it has to be allowed on
purpose. If you add a screen and it 404s or redirects, that is the gate, and the
list is the place to change it.

## 2. The one architectural constraint that will bite you

The tick/quote store is **DuckDB, and DuckDB allows a single writer process.**

- Exactly **one** process may run scans and the Kafka materialiser.
- Every other instance must be a **reader** of the artefacts that process writes.
- A second scanner does not degrade gracefully — it invalidates the tape for
  everyone. This has happened in production.

Consequences you will hit if you ignore it:

- A second instance cannot open `webapp/artifacts/live_stream.duckdb` at all.
  Importing `webapp.kafka_service` constructs a materialiser at module scope, so
  the *import itself* raises when the file is held. Code that needs it must guard
  the import, not just the call.
- Anything that falls back to "compute live from the tick tape" will fail in the
  reader instance and must cache that failure, or it re-pays the cost on every
  request.

**Do not run more than one writer, and do not point two deployments at one store.**

## 3. Install

Requires Python 3.12+.

```bash
git clone <this repo>
cd notebook-webapp
python -m venv .venv
.venv/bin/pip install -r requirements-webapp.txt
```

Use `requirements-webapp.txt`, **not** `requirements.txt` — the latter belongs to
the notebook/analysis side of this tree and does not install the web app.

Versions are pinned for two real reasons: the data code is written against
pandas 3.x, and duckdb's file format is tied to its minor version, so an older
duckdb cannot open a store written by 1.5.

## 4. Configuration

Copy the examples and fill them in. All real config files are git-ignored and
must never be committed.

| Copy from | To | Holds |
|---|---|---|
| `server.example.yaml` | `server.yaml` | MT4/MT5 MySQL hosts, users, **passwords** |
| `symbol_map.example.yaml` | `symbol_map.yaml` | optional symbol aliases |
| — | `kafka/clusters.yaml` | broker hosts, credentials, schema registry |

Environment variables:

| Variable | Purpose |
|---|---|
| `AF_BETA=1` | beta profile (three sections only) |
| `AF_IP_ALLOWLIST` | comma-separated IPs/CIDRs allowed to reach the app; empty = no restriction |

## 5. Run

```bash
.venv/bin/python -m uvicorn webapp.main:app --host 0.0.0.0 --port 3310
```

Behind a reverse proxy, see §8 first — the IP allowlist interacts with it.

Startup runs a background warm (`_warm_caches`) that loads the score artefacts
and an ~80 MB markout parquet. **The port binds before the warm finishes.** A
bound port is not a ready app: the first Account Detail opened during warm can
take ~100 s, and roughly two minutes after start it settles to a few seconds.
Health-check on a real page, not on the socket.

## 6. Data — read this before promising anyone a working screen

The beta **renders artefacts; it does not generate them.** A fresh deployment
with an empty `webapp/artifacts/` shows empty screens. It is not broken.

Artefacts are produced by the writer instance, which needs:

1. MySQL access to the MT4/MT5 servers (the broker tick tape and trades).
2. Kafka access, for MT4 millisecond execution times. MT4 stamps fills to the
   whole second; the millisecond time comes from `tradeRecord.timeStamp` on the
   trade events. Without it, sub-second markout horizons are not measurable and
   the engine correctly falls back to the 1 s and 5 s horizons.
3. Optionally BigQuery, for IB rebate figures. Absent, they report as
   unavailable rather than as zero.

So decide deliberately how the deployment gets data:

- **Own pipeline** — give it database credentials and let it scan. It becomes a
  writer, so it needs its own tick store and must be the only writer against it.
- **Shipped artefacts** — copy artefacts from an existing writer. Note these
  contain **real client account identifiers, P&L and country data**. Treat any
  transfer as client-data movement, with whatever approval that requires.

## 7. Accounts

The user store is SQLite at `webapp/app.db`, created on first run, seeding one
admin account. **Change that password before the app is reachable by anyone.**

Usernames are compared case-insensitively on both sides; registration lowercases
them. Registration creates a *pending* account that an admin must approve.

## 8. Access control, and a trap

Two independent layers, and they are often confused:

1. **Network** — firewall/security group. This decides whether a client can
   reach the port at all. A block here leaves **no trace in the application log**.
2. **`AF_IP_ALLOWLIST`** — checked before authentication. A rejection here
   **always** logs `[ip-allowlist] blocked <ip> -> <path>` and returns 403.

That distinction resolves almost every "I cannot connect" report: if the log is
silent, it is the network; if it logs a block, it is the allowlist.

Two things to get right:

- **Use CIDRs, not single hosts,** for VPN clients. VPN pools hand out a
  different address per device and per session; `/32` entries look correct and
  admit almost nobody.
- **A reverse proxy breaks the allowlist.** The check reads the peer address, so
  behind a proxy every request appears to come from the proxy. Either keep the
  port in the URL and skip the proxy, or terminate at the proxy and enforce the
  restriction there.

The allowlist is a perimeter, not the access control. Authentication is.

## 9. Operational notes

- **Logs**: uvicorn stdout/stderr. `*.log` and `*.err` are git-ignored; the
  stderr files reach tens of megabytes.
- **Restarts**: the writer instance resumes background work on start. Check what
  it starts before restarting it in market hours.
- **Scans** write artefacts atomically, so a reader never sees a half-written
  scan; a stale artefact is served as stale rather than as current.
- **Never commit**: `server.yaml`, `kafka/*.yaml`, `*.db*`, `*.parquet`,
  `*.duckdb`, `*.csv`. The `.gitignore` covers these, but note it does not
  *untrack* anything already committed.
