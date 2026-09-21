# Event Impact — standalone replication

This package reproduces the **Event Impact** analysis (Anti-fraud tab of the ZFX risk-intelligence
application) directly from the MT4/MT5 MySQL servers, without the application or its local data
store, and was validated against the application on the **NFP release of 4 September 2026 (XAUUSD,
13:29:00–13:31:00 London = 12:29–12:31 UTC)**.

## Contents

| File | What it is |
|---|---|
| `event_impact_standalone.py` | The script: extraction queries, normalisation, analysis logic, Excel workbook, JSON summary. Self-contained. |
| `event_impact_labels.csv` | Behavioural label per account (`account_key,profile`) exported from the application's Anti-fraud classifier. Optional input; context for segmentation only. |
| `event_impact_XAUUSD_2026-09-04.xlsx` | The workbook the script produced for the 4 September event (Summary, All results, one sheet per segment). |
| `summary.json` | The script's totals, segment counts and window for that run. |
| `README.md` | This file. |

Requirements: Python 3.10+, `pymysql`, `pandas`, `numpy`, `scikit-learn`, `xlsxwriter`, `pyyaml`.

## Running it

Create a `servers.yaml` with your own credentials (not distributed):

```yaml
defaults: {port: 3306, user: <user>, password: <password>}
servers:
  mt4_live01: {host: <host>}
  mt4_live02: {host: <host>}
  mt4_live03: {host: <host>}
  mt4_live04: {host: <host>}
  mt5_live01: {host: <host>}
```

Then, for the 4 September event exactly as validated:

```
python event_impact_standalone.py --servers servers.yaml \
  --start "2026-09-04 13:29:00" --end "2026-09-04 13:31:00" --symbols XAUUSD \
  --labels event_impact_labels.csv --as-of "2026-09-13 14:24:21" \
  --out event_impact_XAUUSD_2026-09-04.xlsx --json summary.json
```

* `--start/--end` are Europe/London wall time; the script converts to UTC.
* `--symbols` are canonical names (variants such as XAUUSDe / XAUUSDmin / XAUUSD247 collapse to XAUUSD); empty = all.
* `--loss/--profit` (default 500) are the significant-loss/profit thresholds in dollars.
* `--as-of` pins the "current time" of the cash-movement data (tenure, coverage flag). Omit to use the newest movement on the servers.
* `--cash-since` optionally sets the earliest cash movement included (default: as-of minus 730 days, the application's retention).
* Run time for this event: about 3 minutes (≈2.4M trades read across the five servers, then ≈0.4M cash movements for the 3,055 impacted logins).

## What the script does

1. **Extracts** closed trades from every server for `t0 − 5 days` to `t1 + 5 days` with the application's own queries (MT4 `orders`; MT5 `deals` exit-deal joined to its entry-deal on `position_id`), and cash movements for the impacted logins (MT4 `balance_ops`; MT5 `deals` with `action = 2`).
2. **Normalises** exactly as the application's extractor does: MT4 volume ÷ 100 and MT5 volume ÷ 10,000 to lots; `cmd` 0/1 → buy/sell; UTC timestamps; cent-account money and lots ÷ 100 (logins whose group has `groups.currency = 'CNT'`); accounts keyed `server:login`; deposits/withdrawals classified with internal transfers and credits excluded.
3. **Analyses** with the same logic, in order: impacted clients (`open_time ≤ t1 AND close_time ≥ t0` on the symbols); opened-in / closed-in / held-through actions; window P&L on closes inside the window; the 120-second losing-close stop-out signature; 5-day baseline and 4h/1d/2d/5d activity and volume ratios; funding per horizon, lifetime, tenure and MLTV (net deposits ÷ months); impact class; Isolation-Forest reaction anomaly; high-value (80th percentiles); behaviour-change reasons; the segmentation decision tree; ordering by segment then MLTV; the 800-row cap on listed rows (totals cover everyone).
4. **Writes** the same workbook layout and formats as the application's export.

## Validation against the application (4 September 2026)

The application was run on the same footing (its five MySQL servers; it also holds a sixth server
sourced from BigQuery, which the script cannot reach) with the same cash-movement as-of time and the
same labels. Result:

| Metric | Application | Script |
|---|---|---|
| Impacted clients | 3,055 | 3,055 |
| Window P&L | −$828,438.00 | −$828,438.00 |
| Opens / closes / held-through | 4,915 / 9,437 / 4,599 | 4,915 / 9,437 / 4,599 |
| Actions in window | 16,026 | 16,026 |
| Lots in window | 1,042.46 | 1,042.46 |
| Stopped out / sig. loss / sig. profit | 706 / 79 / 70 | 706 / 79 / 70 |
| Deposited / withdrawn / net, 5 days | $1,544,878.37 / $1,282,614.36 / $262,264.02 | identical |
| Behaviour changed | 2,217 | 2,217 |
| Median MLTV | $132 / month | $132 / month |
| Lifetime net deposits (impacted clients) | $59,739,706 | $59,739,706 |
| Abuse candidates / retain / compensate-review / monitor / minimal | 50 / 303 / 482 / 1,043 / 1,177 | 50 / 303 / 482 / 1,043 / 1,177 |

Every figure is identical (difference zero on every line), with the application's cash-movement
"current to" time pinned to 2026-09-13 14:24:21 in both runs.

Reaching that agreement was itself useful: the first comparison exposed three defects in the
application's local caches, all corrected before the final run above. Its trade warehouse had not
been refreshed through the analysis horizon (645 held-through positions and 51 clients that closed
late were missing); its cash-movement store held cent-account amounts undeflated (×100) for months
before July 2026; and its trade warehouse held cent-account lots and money undeflated (×100) for
rows extracted before ~3 September 2026 — which had inflated the baseline-volume input of the
high-value test for 123 cent accounts. The warehouse and the cash store were rebuilt in full from
MySQL with the current extractor, after which the application matches the script exactly. The
script, pulling live and deflating at source, was correct throughout.

## Known, intended differences from the application

* The application also includes `mt5_dubai_live01`, replicated to BigQuery rather than MySQL; the script covers the five MySQL servers. (That server had no trades in the September window.)
* The script pulls live from the servers; the application reads its local copy. For a closed window the two agree exactly once the copy is current; a position still open at run time is not counted as held through in either.
* Lifetime funding depends on the retention start (`--cash-since`); tenure and MLTV depend on `--as-of`. Pin both to reproduce a given application run.
