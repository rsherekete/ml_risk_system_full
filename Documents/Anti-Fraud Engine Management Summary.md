# Anti-Fraud Engine
## Management Summary

**As of:** 17 September 2026  
**Latest completed scan:** 17 September 2026, 09:52 UTC  
**Purpose:** Detect latency arbitrage and broader toxic flow using broker tick evidence, independent reference prices, trade behaviour, and economic impact.

## 1. Executive Summary

The engine reviews closed trading activity and asks two related but distinct questions:

1. **Latency arbitrage:** Did the client receive a favourable price move within milliseconds or seconds, and did that advantage fade quickly? This identifies fast, transient exploitation of stale or delayed broker prices.
2. **Toxic flow:** Did the market continue moving in the client's favour after entry, at a level and frequency that is unusually adverse for the broker? This captures persistent, informed, event-driven, or otherwise repeated adverse selection.

Latency is treated as a mechanism. Toxic flow is the wider economic effect. A proven latency event is also labelled Sharp/Fast toxic flow, but persistent flow is never labelled latency when the advantage does not fade.

The latest scan processed **1,784,873 trades** with **91.5% markout coverage from broker ticks** and no tape error. It identified **1,021 latency-flagged trades**. The Toxic Flow engine evaluated **1,632,527 tick-covered trades**, identified **131,398 toxic trades**, and scored **7,387 accounts**.

The revised toxic rule is now active: fixed minimums remain through 1 second; at 5 seconds and 60 seconds, a trade must also be in the symbol's top 5% of markouts for the current scan window. This materially reduces ordinary volatility being treated as toxic, especially in gold, while preserving the short-horizon latency signal.

The engine produces investigation and monitoring recommendations. It does **not** automatically conclude fraud or execute a client action without the configured governance layer and human review.

## 2. How the Engine Works

### 2.1 Common markout engine

Both detection engines consume one shared markout pass. For every eligible closed trade, the engine aligns the fill to the broker's own tick tape and measures the direction-adjusted price move at:

- 100 ms
- 200 ms
- 300 ms
- 500 ms
- 1 second
- 5 seconds
- 60 seconds

A positive markout means the market continued in the client's favour after the fill. That is adverse selection for the broker.

The engine also records:

- Fill timestamp precision and millisecond availability
- Broker quote age at entry
- Entry spread relative to the symbol's normal spread
- Hold duration and realised P&L
- Markout in basis points and estimated USD impact
- Independent reference-market markouts
- Symbol, direction, event timing, clustering, and cross-account repetition

Kafka-published quotes are excluded from markout calculations because their publish timestamp is delayed relative to the broker execution clock. The markout source is the MySQL broker tick tape materialised into the shared quote store.

### 2.2 Latency arbitrage detection

A trade is a latency candidate when it is:

- Fast and profitable
- Favourable to the client at an early checkpoint
- Followed by a sufficiently large decay by 60 seconds
- Within the configured holding-time limit

The engine distinguishes this from directional or informed flow. A favourable move that remains or grows at 60 seconds is classified as directional, not latency arbitrage.

The account-level latency assessment also considers:

- Early markout magnitude: 20%
- Early hit rate: 15%
- Stale quote or price-age evidence: 20%
- Decay consistency: 15%
- Event or repeated activity: 10%
- Profit concentration: 10%
- Independent reference confirmation: 10%

Scores are calibrated against the population where appropriate, while fixed floors prevent noise from receiving full marks. Score and confidence are reported separately.

### 2.3 Toxic Flow detection

A trade is materially adverse when it clears the applicable minimum at one or more markout horizons and has valid tick coverage. Missing markouts are not treated as zero and cannot create a toxic signal.

The toxic engine records the curve signature:

- **Sharp/Fast:** strong early adverse move that fades, including latency events
- **Persistent:** adverse move remains or grows to the late horizon
- **Other adverse:** adverse at one or more checkpoints without either main signature

The account-level Toxicity Score uses these components:

- Seven-horizon adverse markout: 25%
- Toxic trade rate: 15%
- Economic impact in USD: 15%
- Repetition and persistence: 15%
- Profit concentration: 10%
- Independent reference corroboration: 10%
- LP or venue execution evidence: 10%

LP feedback is not currently available. That component is removed and the remaining weights are renormalised; the result reports this limitation explicitly.

## 3. Revised 5-Second and 60-Second Rule

The previous fixed minimums were 2.5 bps at 5 seconds and 3.0 bps at 60 seconds. Those floors remain in place.

The revised rule adds a symbol-relative requirement:

> At 5 seconds and 60 seconds, the markout must be at or above the 95th percentile for that canonical symbol in the current scan window, subject to the fixed minimum floor.

This prevents a normal move in a volatile instrument from being treated as toxic merely because it crossed a small absolute threshold. For example, the latest XAUUSD 60-second threshold is approximately **7.3 bps**, rather than the old fixed 3.0 bps floor.

The rule is measured within each symbol, rather than comparing gold, currencies, indices, and crypto on one common scale. Latency detection through 1 second is unchanged.

## 4. Latest Results

### 4.1 Latency scan

- Scan completed: **09:52 UTC, 17 September 2026**
- Trades scanned: **1,784,873**
- Broker tick coverage: **91.5%**
- Tape errors: **None**
- Latency-flagged trades: **1,021**
- Latency clients currently listed: **15**

The latest available trade timestamp represented in the latency result is approximately **06:29 UTC**. The scan is therefore current to the latest trade data loaded into the warehouse at scan time, subject to source-feed and ingestion lag.

### 4.2 Toxic Flow scan

- Scan completed: **09:51 UTC, 17 September 2026**
- Tick-covered trades evaluated: **1,632,527**
- Toxic trades: **131,398**
- Toxic trade rate: **8.05%**
- Accounts scored: **7,387**
- Accounts listed in the management table: **500** due to display limits
- 95th-percentile rule: **Active at 5 seconds and 60 seconds**

Current trade signatures:

- Sharp/Fast: **9,417**
- Persistent: **81,853**
- Other adverse: **40,128**
- Latency events included in Toxic Flow: **1,021**

Current curve profiles across scored accounts:

- None: **7,181**
- Mixed: **174**
- Persistent: **23**
- Sharp/Fast: **6**
- Event: **3**

Current risk states include:

- Passive monitoring: **6,944 accounts**
- Monitor evidence: **423 accounts**
- Manual investigation: **10 accounts**
- High-priority review: **5 accounts**
- Enhanced monitoring: **5 accounts**

These are screening outputs, not final fraud determinations.

## 5. What the Results Mean

The revised rule has changed the meaning of a persistent toxic flag. It now represents an unusually large move for that symbol, not simply a move that exceeds a low absolute threshold.

Expected benefits:

- Fewer false positives caused by normal gold and index volatility
- A substantially lower toxic trade rate than the previous fixed-floor approach
- Better separation between ordinary flow and accounts that stand out relative to their market
- Preservation of genuine latency evidence at the short horizons
- More defensible explanations to clients and management because every flag includes its checkpoints and coverage status

A relative rule cannot detect a situation where the whole market moves adversely together. The fixed minimum remains as a protection against low-level noise, and longer historical storage is recommended for a stable 30-day benchmark.

## 6. Evidence and Governance Controls

The engine includes the following controls:

- No toxic classification without tick-covered markout evidence
- No latency classification from missing or fabricated markouts
- Independent reference feeds are used where available
- Reference absence is reported and does not count as positive corroboration
- Replicated patterns across accounts can be capped as a possible market-data or execution issue
- Confidence is separate from risk score and is reduced for small samples or weak corroboration
- LP/venue evidence is reported as unavailable rather than implied
- Actions operate in shadow mode unless explicitly changed through governance
- Suggested states require human review for adverse decisions
- Completed scan artifacts are written atomically, including tape errors, so stale results are not silently presented as current

## 7. Management Interpretation

The engine should be used as a prioritisation and evidence system:

- **Latency alerts** identify fast, transient price exploitation that may warrant execution and feed review.
- **Persistent toxic alerts** identify accounts whose post-fill advantage is unusually strong and repeated for the traded symbol.
- **Mixed or event profiles** require context, because news and broad market moves can affect many accounts simultaneously.
- **Low-confidence results** should remain in monitoring until more evidence is collected.
- **Tick coverage matters:** uncovered trades are excluded from markout-based conclusions, not treated as benign or toxic.

Recommended management workflow:

1. Review the highest-score accounts and their per-order evidence.
2. Confirm that broker and reference ticks cover the relevant fills.
3. Check whether the pattern is replicated across unrelated accounts.
4. Separate broker execution or feed issues from client behaviour.
5. Apply controls only after evidence review and an approved decision.

## 8. Limitations and Next Steps

Current limitations:

- Tick coverage is high but not complete: latest coverage is 91.5%.
- The relative benchmark currently uses the current scan window rather than a 30-day historical distribution.
- LP or venue feedback is not connected.
- The current management table displays the top 500 Toxic Flow accounts, while the full scored population remains in the feature artifact.
- Latest-trade freshness depends on warehouse and tick-feed ingestion timing.

Recommended next steps:

- Store symbol-level markout samples for at least 30 days and use a rolling percentile benchmark.
- Add LP or venue execution feedback to restore the final 10% evidence component.
- Monitor coverage by server, symbol, and horizon, not only the aggregate percentage.
- Add management export of the full scored population and an audit trail of analyst decisions.
- Backtest the revised rule against confirmed historical cases and known benign volatile periods.

## 9. Conclusion

The engine now combines timestamp-accurate broker markouts, independent market corroboration, behavioural patterns, and economic impact. The latest rule change makes the long-horizon Toxic Flow signal more selective and more comparable across symbols, while preserving the short-horizon latency detector. Its outputs are suitable for investigation prioritisation and execution-risk management, with the stated data and governance limitations kept visible in the results.
