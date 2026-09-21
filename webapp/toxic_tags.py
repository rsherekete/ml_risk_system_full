"""Stable automation vocabulary for P1 Engine B (Toxic Flow).

Same contract as webapp/latency_tags.py: a tag's MEANING NEVER CHANGES, so the
Anti-Fraud UI can wire a code straight to an execution treatment. The prefix is
`TF_`; Engine A's `LA_` codes are untouched, and an account can legitimately
carry both -- s1 of the specification: "Latency Arbitrage and Toxic Flow
overlap, but latency is a mechanism while toxicity is the broader economic
effect."

Read a tag set in this order:

  STATE     exactly one -- the s10 Suggested Decision Matrix outcome.
  TIER      exactly one -- the s9 risk tier.
  CAP       zero or more -- WHY the state was held back. A CAP tag means DO NOT
            apply adverse client treatment automatically.
  ACTION    one or two -- the recommended treatment rung. s13: "Automated
            actions are governed separately from classification", and s15.3
            "Fix the broker-created opportunity first". An ACTION tag is a
            recommendation to the decision engine, never a routing order.
  PROFILE   zero or more -- the s5.3.1 observable behaviours (T1-T12).
  CURVE     exactly one -- the s5.3 toxic flow profile.
  SIGNATURE exactly one -- the s9 composite markout signature.
  EVIDENCE  zero or more -- what the case rests on.
  DATA      zero or more -- why the evidence is weaker than it looks.
  REVIEW    exactly one -- where the human sits.
"""
from __future__ import annotations

from webapp.toxic_spec import CURVE_PROFILES, PROFILE_LABELS

TAG_CATALOG: dict[str, dict] = {
    # ---- STATE (s10 Suggested Decision Matrix)
    "TF_STATE_PASSIVE_MONITORING": {"category": "STATE", "meaning": "Score below 50: passive monitoring only"},
    "TF_STATE_MONITOR_EVIDENCE": {"category": "STATE", "meaning": "Score 50-69 with confidence under 70%: monitor and collect evidence"},
    "TF_STATE_ENHANCED_MONITORING": {"category": "STATE", "meaning": "Score 50-69 with confidence 70%+: enhanced monitoring / Risk review"},
    "TF_STATE_MANUAL_INVESTIGATION": {"category": "STATE", "meaning": "Score 70-84 with confidence under 70%: manual investigation"},
    "TF_STATE_HIGH_PRIORITY_REVIEW": {"category": "STATE", "meaning": "Score 70-84 with confidence 70%+: high-priority Risk review"},
    "TF_STATE_URGENT_EVIDENCE_REVIEW": {"category": "STATE", "meaning": "Score 85+ with confidence under 80%: urgent evidence review, no automatic adverse conclusion"},
    "TF_STATE_CRITICAL_REVIEW": {"category": "STATE", "meaning": "Score 85+ with confidence 80%+: critical Risk review, eligible for approved controls subject to governance"},
    # ---- TIER (s9 Composite P1 Client Profile)
    "TF_TIER_CRITICAL": {"category": "TIER", "meaning": "Toxicity score 85-100"},
    "TF_TIER_HIGH": {"category": "TIER", "meaning": "Toxicity score 70-84"},
    "TF_TIER_REVIEW": {"category": "TIER", "meaning": "Toxicity score 50-69"},
    "TF_TIER_MONITOR": {"category": "TIER", "meaning": "Toxicity score 25-49"},
    "TF_TIER_NORMAL": {"category": "TIER", "meaning": "Toxicity score 0-24"},
    # ---- CAP: the state was held back
    "TF_CAP_SMALL_SAMPLE": {"category": "CAP", "meaning": "Fewer toxic trades than the s5.2 minimum sample"},
    "TF_CAP_LOW_CONFIDENCE": {"category": "CAP", "meaning": "Confidence below the desk minimum -- do not act automatically"},
    "TF_CAP_MARKET_DATA_CONDITION": {"category": "CAP", "meaning": "Same opportunity available to multiple clients: systemic market-data issue first (s15)"},
    "TF_CAP_LATENCY_DRIVEN": {"category": "CAP", "meaning": "Most of this flow is Engine A's latency case -- treat there, not twice"},
    "TF_CAP_CONFLICTING_CURVE": {"category": "CAP", "meaning": "Markout curve conflicts across horizons: classify mixed (s7)"},
    "TF_CAP_TOTAL_PNL": {"category": "CAP", "meaning": "Account realised P&L below the configured minimum (s5.1)"},
    # ---- ACTION: the recommended rung, for the decision engine
    "TF_ACTION_NONE": {"category": "ACTION", "meaning": "No treatment; flow stays as it is"},
    "TF_ACTION_WATCHLIST": {"category": "ACTION", "meaning": "Observe only; no execution change"},
    "TF_ACTION_WIDEN_SPREAD": {"category": "ACTION", "meaning": "Widen spread on the concentrated instrument"},
    "TF_ACTION_PARTIAL_HEDGE": {"category": "ACTION", "meaning": "Externalise a share of the flow / hedge the edge"},
    "TF_ACTION_ABOOK_RECOMMENDED": {"category": "ACTION", "meaning": "Recommend full A-book routing, subject to governance sign-off"},
    "TF_ACTION_REVIEW_REQUIRED": {"category": "ACTION", "meaning": "Human review required before any treatment is applied"},
    # ---- PROFILE (s5.3.1 T1-T12)
    "TF_T1_SHARP_FAST": {"category": "PROFILE", "meaning": PROFILE_LABELS["T1"]},
    "TF_T2_PERSISTENT_INFORMED": {"category": "PROFILE", "meaning": PROFILE_LABELS["T2"]},
    "TF_T3_HIGH_TOXIC_RATE": {"category": "PROFILE", "meaning": PROFILE_LABELS["T3"]},
    "TF_T4_HIGH_ECONOMIC_TOXICITY": {"category": "PROFILE", "meaning": PROFILE_LABELS["T4"]},
    "TF_T5_PROFIT_CONCENTRATED": {"category": "PROFILE", "meaning": PROFILE_LABELS["T5"]},
    "TF_T6_EVENT_TOXICITY": {"category": "PROFILE", "meaning": PROFILE_LABELS["T6"]},
    "TF_T7_SYMBOL_SPECIFIC": {"category": "PROFILE", "meaning": PROFILE_LABELS["T7"]},
    "TF_T8_EXECUTION_CONDITION": {"category": "PROFILE", "meaning": PROFILE_LABELS["T8"]},
    "TF_T9_DIRECTIONAL": {"category": "PROFILE", "meaning": PROFILE_LABELS["T9"]},
    "TF_T10_REPEATED_CLUSTERS": {"category": "PROFILE", "meaning": PROFILE_LABELS["T10"]},
    "TF_T11_CROSS_ACCOUNT": {"category": "PROFILE", "meaning": PROFILE_LABELS["T11"]},
    "TF_T12_LP_CONFIRMED": {"category": "PROFILE", "meaning": PROFILE_LABELS["T12"]},
    # ---- CURVE (s5.3 Toxic Flow Profiles)
    "TF_CURVE_SHARP_FAST": {"category": "CURVE", "meaning": CURVE_PROFILES["sharp_fast"]},
    "TF_CURVE_PERSISTENT": {"category": "CURVE", "meaning": CURVE_PROFILES["persistent"]},
    "TF_CURVE_EVENT": {"category": "CURVE", "meaning": CURVE_PROFILES["event"]},
    "TF_CURVE_SYMBOL_SPECIFIC": {"category": "CURVE", "meaning": CURVE_PROFILES["symbol_specific"]},
    "TF_CURVE_EXECUTION_CONDITION": {"category": "CURVE", "meaning": CURVE_PROFILES["execution_condition"]},
    "TF_CURVE_MIXED": {"category": "CURVE", "meaning": CURVE_PROFILES["mixed"]},
    "TF_CURVE_NONE": {"category": "CURVE", "meaning": CURVE_PROFILES["none"]},
    # ---- SIGNATURE (s9 Markout Signature)
    "TF_SIGNATURE_HEALTHY": {"category": "SIGNATURE", "meaning": "Markout around the book's own baseline"},
    "TF_SIGNATURE_FAST": {"category": "SIGNATURE", "meaning": "Early advantage that decays (latency-shaped)"},
    "TF_SIGNATURE_INFORMED": {"category": "SIGNATURE", "meaning": "Advantage persists to 60 s (informed flow)"},
    "TF_SIGNATURE_MIXED": {"category": "SIGNATURE", "meaning": "Several signatures at once -- composite investigation"},
    "TF_SIGNATURE_ADVERSE": {"category": "SIGNATURE", "meaning": "Flow is worse than the book's baseline: benign for the broker"},
    # ---- EVIDENCE
    "TF_EVIDENCE_REFERENCE_CONFIRMED": {"category": "EVIDENCE", "meaning": "The independent reference market confirms the adverse move"},
    "TF_EVIDENCE_MATERIAL_USD": {"category": "EVIDENCE", "meaning": "Adverse selection is material in money terms"},
    "TF_EVIDENCE_PROFIT_CONCENTRATED": {"category": "EVIDENCE", "meaning": "A material share of client profit comes from the toxic trades"},
    "TF_EVIDENCE_REPEATED_CLUSTERS": {"category": "EVIDENCE", "meaning": "Toxic trades arrive in repeated bursts, not at random"},
    "TF_EVIDENCE_BROAD_BOOK": {"category": "EVIDENCE", "meaning": "Toxicity spread across several instruments"},
    "TF_EVIDENCE_LATENCY_OVERLAP": {"category": "EVIDENCE", "meaning": "Engine A also flags part of this flow"},
    # ---- DATA
    "TF_DATA_NO_REFERENCE": {"category": "DATA", "meaning": "No independent reference ticks: corroboration scored zero"},
    "TF_DATA_NO_LP_FEEDBACK": {"category": "DATA", "meaning": "No LP/venue feedback: the s5.4 LP component (10%) is removed from the model and T12 cannot fire"},
    "TF_DATA_SHORT_HISTORY": {"category": "DATA", "meaning": "Few trading days in the window: repeatability is provisional"},
    # ---- REVIEW
    "TF_REVIEW_PENDING": {"category": "REVIEW", "meaning": "No analyst decision recorded yet"},
    "TF_REVIEW_CONFIRMED": {"category": "REVIEW", "meaning": "Analyst confirmed the toxic classification"},
    "TF_REVIEW_DISMISSED": {"category": "REVIEW", "meaning": "Analyst dismissed it; suppress treatment"},
}

_PROFILE_CODES = {
    "T1": "TF_T1_SHARP_FAST", "T2": "TF_T2_PERSISTENT_INFORMED",
    "T3": "TF_T3_HIGH_TOXIC_RATE", "T4": "TF_T4_HIGH_ECONOMIC_TOXICITY",
    "T5": "TF_T5_PROFIT_CONCENTRATED", "T6": "TF_T6_EVENT_TOXICITY",
    "T7": "TF_T7_SYMBOL_SPECIFIC", "T8": "TF_T8_EXECUTION_CONDITION",
    "T9": "TF_T9_DIRECTIONAL", "T10": "TF_T10_REPEATED_CLUSTERS",
    "T11": "TF_T11_CROSS_ACCOUNT", "T12": "TF_T12_LP_CONFIRMED",
}
_GATE_CODES = {
    "sample": "TF_CAP_SMALL_SAMPLE", "confidence": "TF_CAP_LOW_CONFIDENCE",
    "market_data_condition": "TF_CAP_MARKET_DATA_CONDITION",
    "latency_driven": "TF_CAP_LATENCY_DRIVEN",
    "conflicting_curve": "TF_CAP_CONFLICTING_CURVE",
    "total_pnl": "TF_CAP_TOTAL_PNL",
}
#: s10 state -> the treatment rung the decision engine may consider. Every rung
#: above watchlist also carries TF_ACTION_REVIEW_REQUIRED: s10 allows approved
#: controls only for critical review, and only "subject to governance".
_ACTION_CODES = {
    "passive_monitoring": "TF_ACTION_NONE",
    "monitor_evidence": "TF_ACTION_WATCHLIST",
    "enhanced_monitoring": "TF_ACTION_WATCHLIST",
    "manual_investigation": "TF_ACTION_REVIEW_REQUIRED",
    "high_priority_review": "TF_ACTION_PARTIAL_HEDGE",
    "urgent_evidence_review": "TF_ACTION_REVIEW_REQUIRED",
    "critical_review": "TF_ACTION_ABOOK_RECOMMENDED",
}


def account_tags(row: dict, review: str | None = None) -> list[str]:
    """The full tag set for one scored account. Reads the scan row only."""
    def num(key, default=0.0):
        value = row.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    tags: list[str] = []
    state = str(row.get("state") or "passive_monitoring")
    tags.append(f"TF_STATE_{state.upper()}")
    tags.append(f"TF_TIER_{str(row.get('risk_tier') or 'normal').upper()}")

    gates = [g for g in str(row.get("gates_failed") or "").split(",") if g]
    for gate in gates:
        code = _GATE_CODES.get(gate)
        if code and code not in tags:
            tags.append(code)

    action = _ACTION_CODES.get(state, "TF_ACTION_NONE")
    # A cap stops the ladder: nothing above a watchlist without a human.
    if gates and action not in ("TF_ACTION_NONE", "TF_ACTION_WATCHLIST"):
        action = "TF_ACTION_WATCHLIST"
    # Toxicity concentrated in one instrument is a pricing change, not a
    # book-wide one (s5.3 Symbol-Specific).
    elif action == "TF_ACTION_PARTIAL_HEDGE" and num("top_symbol_share") >= 0.9:
        action = "TF_ACTION_WIDEN_SPREAD"
    tags.append(action)
    if action in ("TF_ACTION_WIDEN_SPREAD", "TF_ACTION_PARTIAL_HEDGE",
                  "TF_ACTION_ABOOK_RECOMMENDED"):
        tags.append("TF_ACTION_REVIEW_REQUIRED")

    for code in str(row.get("profiles") or "").split(","):
        tag = _PROFILE_CODES.get(code.strip())
        if tag and tag not in tags:
            tags.append(tag)

    curve = str(row.get("curve_profile") or "none").upper()
    tags.append(f"TF_CURVE_{curve}")
    tags.append(f"TF_SIGNATURE_{str(row.get('markout_signature') or 'healthy').upper()}")

    if num("ref_agree_share") >= 0.5 and num("ref_checked") >= 10:
        tags.append("TF_EVIDENCE_REFERENCE_CONFIRMED")
    if "TF_T4_HIGH_ECONOMIC_TOXICITY" in tags:
        tags.append("TF_EVIDENCE_MATERIAL_USD")
    if "TF_T5_PROFIT_CONCENTRATED" in tags:
        tags.append("TF_EVIDENCE_PROFIT_CONCENTRATED")
    if "TF_T10_REPEATED_CLUSTERS" in tags:
        tags.append("TF_EVIDENCE_REPEATED_CLUSTERS")
    if num("n_symbols") >= 3 and num("top_symbol_share", 1.0) < 0.6:
        tags.append("TF_EVIDENCE_BROAD_BOOK")
    if num("latency_events") > 0:
        tags.append("TF_EVIDENCE_LATENCY_OVERLAP")

    if not num("ref_checked"):
        tags.append("TF_DATA_NO_REFERENCE")
    tags.append("TF_DATA_NO_LP_FEEDBACK")
    if num("active_days") < 3:
        tags.append("TF_DATA_SHORT_HISTORY")

    tags.append({"confirm": "TF_REVIEW_CONFIRMED",
                 "dismiss": "TF_REVIEW_DISMISSED"}.get(review or "", "TF_REVIEW_PENDING"))

    unknown = [t for t in tags if t not in TAG_CATALOG]
    assert not unknown, f"tags missing from the catalog: {unknown}"
    return tags


def catalog() -> list[dict]:
    order = ["STATE", "TIER", "CAP", "ACTION", "PROFILE", "CURVE",
             "SIGNATURE", "EVIDENCE", "DATA", "REVIEW"]
    return sorted(({"tag": k, **v} for k, v in TAG_CATALOG.items()),
                  key=lambda t: (order.index(t["category"]), t["tag"]))
