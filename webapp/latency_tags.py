"""Automation tags for the Latency Arbitrage engine.

Stable, machine-readable codes the Anti-Fraud automation layer maps to
execution treatments (wider spread, speed bump, forced slippage, ... A-Book).
Codes never change meaning; new ones are only ever added. Every client row in
a scan carries `tags`, every change of an account's tag set is written to the
audit log, and /api/antifraud/latency/tags serves the catalog plus the
current tag state for every scanned account and feed condition.

Categories, in the order an automation rule should read them:
  VERDICT   the engine's recommendation (exactly one per client)
  BAND      the spec s5.3 score band (exactly one)
  CAP       why a verdict was held at monitor -- a CAP tag means "do NOT
            apply adverse client treatment automatically" (spec s5.5.4)
  PROFILE   observable behaviour (spec s5.4 L-profiles), zero or more
  CURVE     the account's markout-curve shape (exactly one)
  EVIDENCE  strength / basis of the evidence, zero or more
  DATA      data-quality caveats, zero or more
  REVIEW    operator review state (exactly one)
  FEED      conditions of the broker's price feed, not of a client
"""
from __future__ import annotations

TAG_CATALOG: dict[str, dict] = {
    # ---- VERDICT
    "LA_VERDICT_ESCALATE": {"category": "VERDICT", "meaning": "Critical band, every gate passed: strongest latency-arbitrage evidence"},
    "LA_VERDICT_RESTRICT": {"category": "VERDICT", "meaning": "High band, every gate passed"},
    "LA_VERDICT_MONITOR": {"category": "VERDICT", "meaning": "Emerging or moderate band, or a higher band capped by a failed gate"},
    "LA_VERDICT_CLEAR": {"category": "VERDICT", "meaning": "Normal band: no actionable latency risk"},
    # ---- BAND
    "LA_BAND_CRITICAL": {"category": "BAND", "meaning": "Latency Risk Score 85-100"},
    "LA_BAND_HIGH": {"category": "BAND", "meaning": "Latency Risk Score 70-84"},
    "LA_BAND_MODERATE": {"category": "BAND", "meaning": "Latency Risk Score 50-69"},
    "LA_BAND_EMERGING": {"category": "BAND", "meaning": "Latency Risk Score 25-49"},
    "LA_BAND_NORMAL": {"category": "BAND", "meaning": "Latency Risk Score 0-24"},
    # ---- CAP (automation must not treat the client adversely on these alone)
    "LA_CAP_LOW_CONFIDENCE": {"category": "CAP", "meaning": "Confidence below the configured minimum"},
    "LA_CAP_REPLICATED": {"category": "CAP", "meaning": "Pattern replicated across unrelated accounts at the same price moment: investigate the feed first (s5.5.2)"},
    "LA_CAP_MARKET_DATA_CONDITION": {"category": "CAP", "meaning": "Most latency events sit inside a broker feed condition flagged as architecture risk (s5.5.4)"},
    "LA_CAP_EARLY_HIT_RATE": {"category": "CAP", "meaning": "Early hit rate below the configured minimum"},
    "LA_CAP_EVENT_SUCCESS": {"category": "CAP", "meaning": "Event success rate below the configured minimum"},
    "LA_CAP_PROFIT_CONCENTRATION": {"category": "CAP", "meaning": "Profit from latency events below the configured minimum"},
    "LA_CAP_REALIZED_PNL": {"category": "CAP", "meaning": "Account realised P&L below the configured minimum"},
    # ---- PROFILE (spec s5.4)
    "LA_L1_STALE_PRICE_CAPTURE": {"category": "PROFILE", "meaning": "Repeated fills while the broker price sat behind the independent reference"},
    "LA_L2_FAST_POSITIVE_MARKOUT": {"category": "PROFILE", "meaning": "Consistently favourable markout within the early window (100 ms - 1 s)"},
    "LA_L3_DECAYING_ADVANTAGE": {"category": "PROFILE", "meaning": "Early advantage fades by 60 s (latency signature)"},
    "LA_L4_REFERENCE_LEAD_LAG": {"category": "PROFILE", "meaning": "Entries after the reference moved but before the broker quote did"},
    "LA_L5_SHORT_HOLD_EXTRACTION": {"category": "PROFILE", "meaning": "Very short holds with repeated quick profits"},
    "LA_L7_CLUSTERED_EVENTS": {"category": "PROFILE", "meaning": "Latency events arrive in bursts"},
    "LA_L8_SYMBOL_CONCENTRATED": {"category": "PROFILE", "meaning": "Latency events concentrated in one symbol"},
    "LA_L9_CROSS_ACCOUNT_REPLICATION": {"category": "PROFILE", "meaning": "Several accounts profit on the same price at the same moment"},
    "LA_L10_PROFIT_CONCENTRATED": {"category": "PROFILE", "meaning": "Most of the account's profit comes from latency events"},
    # ---- CURVE
    "LA_CURVE_FAST_TRANSIENT": {"category": "CURVE", "meaning": "Account markout peaks early and fades (latency-shaped)"},
    "LA_CURVE_PERSISTENT": {"category": "CURVE", "meaning": "Account markout holds or grows (directional / informed flow, not latency)"},
    "LA_CURVE_NEUTRAL": {"category": "CURVE", "meaning": "No consistent advantage across horizons"},
    "LA_CURVE_NO_TICKS": {"category": "CURVE", "meaning": "No tick coverage to draw a curve"},
    # ---- EVIDENCE
    "LA_EVIDENCE_REFERENCE_CONFIRMED": {"category": "EVIDENCE", "meaning": "At least half of the latency events are confirmed as latency by the independent reference"},
    "LA_EVIDENCE_STALE_QUOTES": {"category": "EVIDENCE", "meaning": "Fills on broker quotes older than the 200 ms throttle + tolerance, well above normal flow"},
    "LA_EVIDENCE_QUOTE_AGE_ONLY": {"category": "EVIDENCE", "meaning": "L1 judged on broker quote age alone (no reference tick covered the account)"},
    "LA_EVIDENCE_DIRECTIONAL_FLOW": {"category": "EVIDENCE", "meaning": "Repeated fast profitable trades whose advantage persisted (toxic/informed flow rather than latency)"},
    # ---- DATA
    "LA_DATA_SECOND_PRECISION": {"category": "DATA", "meaning": "Most fills stamped only to the second: sub-second horizons and quote age not assessable"},
    "LA_DATA_LOW_TICK_COVERAGE": {"category": "DATA", "meaning": "Under 80% of the account's trades have tick coverage"},
    "LA_DATA_NO_REFERENCE": {"category": "DATA", "meaning": "No reference ticks covered the account's latency events"},
    # ---- REVIEW
    "LA_REVIEW_PENDING": {"category": "REVIEW", "meaning": "No operator decision yet"},
    "LA_REVIEW_CONFIRMED": {"category": "REVIEW", "meaning": "Operator confirmed latency arbitrage (with comment)"},
    "LA_REVIEW_DISMISSED": {"category": "REVIEW", "meaning": "Operator dismissed as a false positive (with comment)"},
    # ---- FEED (conditions of the broker's feed, attached to feed x symbol)
    "LA_FEED_ARCHITECTURE_RISK": {"category": "FEED", "meaning": "A server feed and symbol where several clients repeatedly profit from stale broker prices: fix the feed before client action"},
    "LA_FEED_STALE_TOB": {"category": "FEED", "meaning": "A server feed and symbol with fills behind the reference price (below architecture-risk level)"},
    "LA_FEED_REPLICATED_MOMENT": {"category": "FEED", "meaning": "A price moment where several accounts profited on the same symbol and side within 2 s"},
}

_PROFILE_CODES = {"L1": "LA_L1_STALE_PRICE_CAPTURE", "L2": "LA_L2_FAST_POSITIVE_MARKOUT",
                  "L3": "LA_L3_DECAYING_ADVANTAGE", "L4": "LA_L4_REFERENCE_LEAD_LAG",
                  "L5": "LA_L5_SHORT_HOLD_EXTRACTION", "L7": "LA_L7_CLUSTERED_EVENTS",
                  "L8": "LA_L8_SYMBOL_CONCENTRATED", "L9": "LA_L9_CROSS_ACCOUNT_REPLICATION",
                  "L10": "LA_L10_PROFIT_CONCENTRATED"}
_GATE_CODES = {"confidence": "LA_CAP_LOW_CONFIDENCE", "not_replicated": "LA_CAP_REPLICATED",
               "market_data_condition": "LA_CAP_MARKET_DATA_CONDITION",
               "early_hit_rate": "LA_CAP_EARLY_HIT_RATE", "event_success_rate": "LA_CAP_EVENT_SUCCESS",
               "profit_concentration": "LA_CAP_PROFIT_CONCENTRATION", "realized_pnl": "LA_CAP_REALIZED_PNL"}


def account_tags(row: dict, review: str | None = None, min_events: int = 5) -> list[str]:
    """Tags for one account row of the scan (the fields latency_arb emits)."""
    tags = [f"LA_VERDICT_{str(row.get('verdict') or 'clear').upper()}",
            f"LA_BAND_{str(row.get('band') or 'normal').upper()}"]
    tags += [_GATE_CODES[g] for g in str(row.get("gates_failed") or "").split(",") if g in _GATE_CODES]
    tags += [_PROFILE_CODES[p] for p in str(row.get("profiles") or "").split(",") if p in _PROFILE_CODES]
    curve = str(row.get("curve_profile") or "no_ticks").upper()
    tags.append(f"LA_CURVE_{curve}" if f"LA_CURVE_{curve}" in TAG_CATALOG else "LA_CURVE_NO_TICKS")
    if (row.get("flag_refcov") or 0) > 0 and (row.get("reference_confirmed_share") or 0) >= 0.5:
        tags.append("LA_EVIDENCE_REFERENCE_CONFIRMED")
    if (row.get("c_price_age") or 0) >= 0.5:
        tags.append("LA_EVIDENCE_STALE_QUOTES")
    if "L1" in str(row.get("profiles") or "").split(",") and row.get("l1_basis") == "quote_age_only":
        tags.append("LA_EVIDENCE_QUOTE_AGE_ONLY")
    if (row.get("directional_flagged") or 0) >= min_events:
        tags.append("LA_EVIDENCE_DIRECTIONAL_FLOW")
    if row.get("ms_share") is not None and row.get("ms_share") < 0.5:
        tags.append("LA_DATA_SECOND_PRECISION")
    if row.get("tick_coverage") is not None and row.get("tick_coverage") < 0.8:
        tags.append("LA_DATA_LOW_TICK_COVERAGE")
    if (row.get("latency_events") or 0) > 0 and not (row.get("flag_refcov") or 0):
        tags.append("LA_DATA_NO_REFERENCE")
    tags.append({"confirm": "LA_REVIEW_CONFIRMED", "dismiss": "LA_REVIEW_DISMISSED"}.get(review or "", "LA_REVIEW_PENDING"))
    unknown = [t for t in tags if t not in TAG_CATALOG]
    assert not unknown, f"tags missing from the catalog: {unknown}"
    return tags


def feed_tags(root_cause: list[dict], top_replicated: list[dict]) -> list[dict]:
    """Feed-level tag records (not attached to any client)."""
    out = []
    for r in root_cause or []:
        if (r.get("stale_tob_events") or 0) <= 0:
            continue
        code = "LA_FEED_ARCHITECTURE_RISK" if r.get("architecture_risk") is True else "LA_FEED_STALE_TOB"
        out.append({"tag": code, "feed": r.get("feed"), "symbol": r.get("canonical"),
                    "stale_tob_events": r.get("stale_tob_events"), "clients": r.get("clients"),
                    "success_rate": r.get("stale_tob_success_rate"), "impact_usd": r.get("stale_tob_impact_usd")})
    for m in top_replicated or []:
        symbol, side, moment = (str(m.get("condition")).split("|") + ["", "", ""])[:3]
        out.append({"tag": "LA_FEED_REPLICATED_MOMENT", "symbol": symbol, "side": "buy" if side == "1" else "sell",
                    "moment_utc": moment, "accounts": m.get("accounts"), "events": m.get("events"), "pnl": m.get("pnl")})
    return out


def catalog() -> list[dict]:
    order = ["VERDICT", "BAND", "CAP", "PROFILE", "CURVE", "EVIDENCE", "DATA", "REVIEW", "FEED"]
    return sorted(({"tag": k, **v} for k, v in TAG_CATALOG.items()),
                  key=lambda t: (order.index(t["category"]), t["tag"]))
