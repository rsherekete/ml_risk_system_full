"""P1 Engine B -- Toxic Flow: metrics, score, confidence, profiles, decision state.

Implements section 5 of "P1_Risk_Detection_Formal_Specification.docx" (P1 Risk
Detection Engine -- Formal Functional & Scoring Specification), together with
the shared sections that govern every P1 engine:

  s2   seven-horizon markout framework (100/200/300/500 ms, 1 s, 5 s, 60 s)
  s2.2 universal curve profiles; "Persistent / Directional Advantage ...
       potential informed, directional or toxic flow"
  s3   common markout metrics (median, hit rate, USD, peak, decay, persistence
       ratio, curve slope, area under curve, consistency, population benchmark)
  s5.1 basic sharp-deal parameters
  s5.2 recommended toxicity parameters
  s5.3 toxic flow curve profiles; s5.3.1 observable behaviours T1-T12
  s5.4 the toxicity score weights -- BINDING, reproduced verbatim below
  s7   common confidence framework (evidence-count bands + adjustments)
  s9   composite client profile (markout signature, risk tier)
  s10  suggested decision matrix (score x confidence -> suggested state)
  s15  mandatory market-data aggregation root-cause control

DESIGN NOTES THAT THE SPEC FIXES, NOT US
  * "The Toxic Flow engine should profile the FORM of adverse flow, not merely
    count profitable or 'sharp' trades. A client can be profitable without
    being toxic." (s5.3.1) -- so client P&L is one 10% component, never a gate.
  * "Risk Score and Confidence are separate." (s13) -- two numbers, always.
  * "Small samples cannot automatically generate high-confidence
    classifications." (s13) -- s7's event bands cap confidence directly.
  * "Toxic Flow quantifies economic impact in USD." (s13) -- adverse markout is
    converted to money per trade, not left in basis points.
  * "Automated actions are governed separately from classification." (s13) and
    s15.3 "Fix the broker-created opportunity first; then measure and manage the
    client behaviour." -- the engine emits a SUGGESTED STATE from s10, never a
    routing instruction.

SIGN CONVENTION: markouts are direction-adjusted and CLIENT-POSITIVE. A
positive markout means the market kept moving the client's way after the fill,
i.e. the flow was adverse to the broker. That is the toxic direction, and it is
what "adverse markout" means throughout this module.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from webapp.latency_spec import _usd_per_price_lot

#: s5.4 Toxicity Score -- the specification's weight table, verbatim.
SPEC_WEIGHTS = {
    "seven_horizon_markout": 0.25,   # Seven-horizon adverse markout
    "toxic_trade_rate": 0.15,        # Toxic trade rate
    "economic_impact": 0.15,         # Economic impact / USD markout
    "repetition_persistence": 0.15,  # Repetition / persistence
    "profit_concentration": 0.10,    # Profit concentration
    "reference_corroboration": 0.10,  # Reference-market corroboration
    "lp_execution_evidence": 0.10,   # LP / execution evidence
}

#: s5.3 Toxic Flow Profiles -- the six curve signatures.
CURVE_PROFILES = {
    "sharp_fast": "Strong 100-500 ms adverse markout, decays later",
    "persistent": "Adverse markout remains or grows to 60 s",
    "event": "Toxicity concentrated around news/volatility",
    "symbol_specific": "Toxicity concentrated in selected instruments",
    "execution_condition": "Concentrated around execution conditions",
    "mixed": "Multiple signatures -- composite investigation",
    "none": "No toxic signature",
}

#: s5.3.1 Observable Toxic Flow Behaviour Profiles.
PROFILE_LABELS = {
    "T1": "Sharp / Fast Toxic Flow -- very short-lived trades, favourable early markout",
    "T2": "Persistent Toxic / Informed Flow -- advantage remains or grows through 5-60 s",
    "T3": "High Toxic Trade Rate -- a high proportion of trades are adverse",
    "T4": "High Economic Toxicity -- material adverse markout in USD",
    "T5": "Profit-Concentrated Toxic Flow -- client profit comes from the adverse trades",
    "T6": "Event Toxicity -- adverse flow clusters around news or volatility",
    "T7": "Symbol-Specific Toxicity -- concentrated in selected instruments",
    "T8": "Execution-Condition Toxicity -- better during specific spread/feed conditions",
    "T9": "Directional Toxicity -- one direction generates stronger adverse markout",
    "T10": "Repeated Toxic Clusters -- bursts rather than random arrivals",
    "T11": "Cross-Account Toxicity -- similar adverse flow across synchronised accounts",
    "T12": "LP / Venue-Confirmed Toxicity -- independently identified as adverse",
}
#: s5.3.1 T12 and the s5.4 "LP / execution evidence" component both need
#: liquidity-provider feedback. s11 Data Requirements lists LP feedback as
#: "Highly useful" for Toxic Flow, not Required -- and we do not receive it, so
#: the component scores zero for every account and T12 can never fire. The
#: scan summary reports this so the missing 10 points are never mistaken for a
#: clean result.
NOT_MEASURABLE = {
    "T12": ("LP / venue feedback is not received from any liquidity provider, so "
            "independent execution corroboration cannot be evaluated and T12 "
            "never fires. Its s5.4 component (10%) is currently REMOVED from the "
            "model and its weight redistributed across the other six, so the "
            "score keeps its full 0-100 range; restore it by setting "
            "renormalise_unavailable to false once an LP feed exists."),
}

TOXIC_DEFAULT_RULES = {
    # ---- s5.1 Basic Sharp Deal Parameters
    #: Count of Sharp Deals: minimum profitable transactions in last 24 hours.
    "sharp_deal_count": 5,
    #: Check Profit/Spread Ratio + the ratio itself: profit relative to the full
    #: opening-position cost (spread, and commission where booked).
    "check_profit_spread_ratio": True,
    "profit_spread_ratio": 2.0,
    #: Profit: minimum order profit USD for a sharp deal.
    "sharp_min_profit_usd": 1.0,
    #: Order Life Time: maximum position lifetime for a sharp deal, seconds.
    "sharp_order_life_time_s": 300,
    #: Check Total PnL / Minimum TOTAL PnL, USD.
    "check_total_pnl": False,
    "min_total_pnl_usd": 0.0,

    # ---- s5.2 Recommended Toxicity Parameters
    #: A trade is MATERIALLY ADVERSE when its markout clears the minimum at the
    #: horizon. MO100 is analytical only per s5.2 / s13 -- it counts as markout
    #: evidence but never as quote-age evidence, which this engine does not use.
    "min_mo100_bps": 1.0,
    "min_mo500_bps": 1.5,
    "min_mo1s_bps": 2.0,
    "min_mo5s_bps": 2.5,
    "min_mo60s_bps": 3.0,
    #: Relative book filter for the longer horizons. The fixed minimum remains
    #: a floor; the percentile is measured on this scan week's tick-covered
    #: trades for each canonical symbol.
    "relative_markout_percentile": 95.0,
    "relative_markout_horizons_seconds": [5.0, 60.0],
    #: Minimum Toxic Trades: minimum sample before an account can be alerted.
    "min_toxic_trades": 10,
    #: Toxic Trade Rate %: share of trades materially adverse (full marks).
    "toxic_trade_rate_pct": 25.0,
    #: Average / Total Adverse Markout USD: broker economic impact.
    "avg_adverse_markout_usd": 25.0,
    "total_adverse_markout_usd": 5000.0,
    #: Maximum Holding Time: OPTIONAL short-duration filter (0 = disabled).
    "max_holding_time_s": 0,
    #: Minimum Toxic PnL Concentration %.
    "min_toxic_pnl_concentration_pct": 40.0,
    #: Reference Market Check / LP-Venue Feedback Check.
    "reference_market_check": True,
    "lp_venue_feedback_check": False,

    # ---- scoring and governance
    "toxic_weights": dict(SPEC_WEIGHTS),
    #: s5.4 gives seven weights. Where a component has no data source at all --
    #: today that is LP / execution evidence, since no liquidity provider sends
    #: us feedback -- leaving its weight in place would cap every account below
    #: 100 and quietly compress the bands. DESK DECISION (16 Sep 2026): drop
    #: such a component and renormalise the rest, so the score uses its full
    #: range. Both the spec weights and the applied weights are reported on
    #: every scan. Set false to score strictly to the s5.4 table instead, which
    #: is what to do once an LP feed exists.
    "renormalise_unavailable": True,
    "toxic_min_confidence": 40.0,
    #: s15 root-cause control: a pattern replicated across unrelated accounts is
    #: a market-data condition first. Above this share the state is capped.
    "max_replicated_share": 0.5,
    #: Engine A owns the fast-decaying curve; above this share of an account's
    #: trades being latency events, Engine B defers instead of treating twice.
    "max_latency_share": 0.5,
    #: A trade's advantage has FADED when its 60 s markout is at most
    #: (100 - this)% of its early peak. toxic_flow.build overrides it with
    #: Engine A's `min_decay_pct`, so both engines share one fade definition.
    "min_decay_pct": 50.0,
    #: T7 / T8 / T9 / T10 profile parameters.
    #: Smallest early peak (bps) a persistence RATIO may be computed from.
    #: Below it the denominator is noise and the quotient is meaningless.
    "persistence_min_base_bps": 0.5,
    "symbol_concentration": 0.8,
    "event_share": 0.25,
    "direction_skew": 0.7,
    "cluster_window_s": 300,
    "cluster_min_trades": 3,
    "cluster_share": 0.3,
    "execution_condition_ratio": 1.5,
    #: Population calibration, as Engine A: full marks at the percentile-99
    #: account of the book, bounded by the fixed standard below.
    "toxic_calibration": {"mode": "population", "percentile": 99, "min_trades": 50},
    "toxic_scale": {"auc_excess_bps": 2.0, "toxic_rate_excess": 0.25,
                    "adverse_usd": 5000.0, "confidence_full_events": 50},
    "toxic_floor": {"auc_excess_bps": 0.5, "toxic_rate_excess": 0.05,
                    "adverse_usd": 500.0},
}

#: s10 Suggested Decision Matrix, verbatim. (score_low, score_high) x confidence
#: -> (state key, human label).
DECISION_MATRIX = [
    (0, 50, None, "passive_monitoring", "Passive monitoring"),
    (50, 70, 70, "monitor_evidence", "Monitor / collect evidence"),
    (50, 70, None, "enhanced_monitoring", "Enhanced monitoring / Risk review"),
    (70, 85, 70, "manual_investigation", "Manual investigation"),
    (70, 85, None, "high_priority_review", "High-priority Risk review"),
    (85, 101, 80, "urgent_evidence_review", "Urgent evidence review; no automatic adverse conclusion"),
    (85, 101, None, "critical_review", "Critical Risk review; eligible for approved controls subject to governance"),
]
#: s9 Risk Tier / s8 Severity vocabulary.
RISK_TIERS = [(85, "critical"), (70, "high"), (50, "review"), (25, "monitor"), (0, "normal")]
#: s9 Markout Signature.
SIGNATURES = ("healthy", "fast", "informed", "mixed", "adverse")


def toxic_rules(rules: dict) -> dict:
    """Effective Engine B rules: defaults, with nested dicts merged key by key
    so a partial save never silently drops a threshold."""
    out = dict(TOXIC_DEFAULT_RULES)
    for key, value in (rules or {}).items():
        if key in TOXIC_DEFAULT_RULES and isinstance(TOXIC_DEFAULT_RULES[key], dict) \
                and isinstance(value, dict):
            merged = dict(TOXIC_DEFAULT_RULES[key])
            merged.update(value)
            out[key] = merged
        elif key in TOXIC_DEFAULT_RULES:
            out[key] = value
    return out


def decision_state(score: float, confidence: float) -> tuple[str, str]:
    """s10 Suggested Decision Matrix: the state for a (score, confidence) pair."""
    for low, high, conf_below, key, label in DECISION_MATRIX:
        if low <= score < high and (conf_below is None or confidence < conf_below):
            return key, label
    return "passive_monitoring", "Passive monitoring"


def risk_tier(score: float) -> str:
    for low, label in RISK_TIERS:
        if score >= low:
            return label
    return "normal"


def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _mo_minimums(r: dict, horizons: list, hlabel) -> dict:
    """s5.2: the per-horizon 'materially adverse' minimum, in bps. Horizons the
    spec does not name (200 ms, 300 ms) interpolate from their neighbours so
    every one of the seven can qualify a trade."""
    named = {"100ms": float(r["min_mo100_bps"]), "500ms": float(r["min_mo500_bps"]),
             "1s": float(r["min_mo1s_bps"]), "5s": float(r["min_mo5s_bps"]),
             "60s": float(r["min_mo60s_bps"])}
    known_x = [math.log10(h) for h in horizons if hlabel(h) in named]
    known_y = [named[hlabel(h)] for h in horizons if hlabel(h) in named]
    out = {}
    for h in horizons:
        label = hlabel(h)
        out[label] = named[label] if label in named else float(
            np.interp(math.log10(h), known_x, known_y))
    return out


def account_metrics(trades: pd.DataFrame, rules: dict, horizons: list, early: list,
                    fallback: list, late: float, mcol, hlabel,
                    replicated: dict | None = None,
                    event_share: pd.Series | None = None) -> tuple[pd.DataFrame, dict]:
    """s3 metrics + s5.2 toxicity metrics + s5.3/5.3.1 profiles + s5.4 score +
    s7 confidence + s10 state, per account.

    `trades` is the shared Markout Engine's per-trade frame (s12: "The Markout
    Engine should be implemented once and consumed by all three P1 engines"),
    i.e. the latency scan's frame. It must carry the per-horizon markouts
    (mcol(h)), `account_key`, `canonical`, `direction`, `open_time`,
    `open_price`, `close_price`, `volume_lots`, `net_profit`, `hold_seconds`,
    `flagged` (Engine A's latency events), and where available `spread_ratio`,
    `entry_spread_rel`, `quote_age_ms` and `ref_exec_<label>_bps`.

    Every tick-covered account is scored, including those Engine A cleared:
    the persistent curve is precisely what Engine A discards.
    """
    r = toxic_rules(rules)
    late_label = hlabel(late)
    labels = [hlabel(h) for h in horizons]
    summary: dict = {"not_measurable_profiles": NOT_MEASURABLE,
                     "weights": dict(r["toxic_weights"]),
                     "persistence_horizon_s": float(late)}

    t = trades.loc[trades[mcol(late)].notna()].copy()
    # s5.2 Maximum Holding Time -- an OPTIONAL short-duration filter.
    max_hold = float(r.get("max_holding_time_s") or 0)
    if max_hold > 0:
        t = t.loc[t["hold_seconds"] <= max_hold]
    if not len(t):
        return pd.DataFrame(), summary

    mo_min = _mo_minimums(r, horizons, hlabel)
    relative_min = {}
    percentile = float(r.get("relative_markout_percentile", 0) or 0)
    relative_horizons = {float(h) for h in (r.get("relative_markout_horizons_seconds") or [])}
    if 0 < percentile < 100 and relative_horizons:
        for h in horizons:
            label = hlabel(h)
            if float(h) not in relative_horizons:
                continue
            by_symbol = t.groupby("canonical", observed=True)[mcol(h)].quantile(percentile / 100.0)
            relative_min[label] = by_symbol.clip(lower=mo_min[label])
            mo_min[label] = float(max(mo_min[label], by_symbol.median())) if len(by_symbol) else mo_min[label]
    summary["markout_minimums_bps"] = {k: round(v, 3) for k, v in mo_min.items()}
    summary["relative_markout"] = {
        "percentile": percentile,
        "horizons": sorted(relative_min),
        "scope": "canonical symbol, tick-covered trades in the current scan window",
        "thresholds_bps": {
            label: {str(symbol): round(float(value), 3) for symbol, value in series.items()}
            for label, series in relative_min.items()},
    }
    t["_day"] = pd.to_datetime(t["open_time"]).dt.normalize()
    lots = pd.to_numeric(t.get("volume_lots"), errors="coerce").fillna(0.0)
    t["_lots"] = lots

    # ---- s5.2: a trade is MATERIALLY ADVERSE when it clears the minimum at any
    # of the seven horizons. Which horizons it clears is kept, because that is
    # what separates the s5.3 signatures from each other.
    hit = {}
    for h in horizons:
        label = hlabel(h)
        threshold = (t["canonical"].map(relative_min[label]).fillna(mo_min[label])
                     if label in relative_min else mo_min[label])
        hit[label] = (t[mcol(h)] >= threshold).fillna(False)
        t[f"_hit_{label}"] = hit[label]
    t["_toxic"] = pd.concat(list(hit.values()), axis=1).any(axis=1)
    early_labels = [hlabel(h) for h in horizons if h <= 0.5]
    late_labels = [hlabel(h) for h in horizons if h >= 1.0]
    t["_toxic_early"] = t[[f"_hit_{c}" for c in early_labels]].any(axis=1)
    t["_toxic_late"] = t[[f"_hit_{c}" for c in late_labels]].any(axis=1)

    # ---- s5.3 per-trade CURVE SIGNATURE (desk decision, 16 Sep 2026).
    # s1: "latency is a mechanism while toxicity is the broader economic
    # effect", so the two engines overlap in ONE direction only:
    #   sharp_fast  an Engine A latency event, or a 100-500 ms toxic hit whose
    #               advantage FADED by 60 s. Every latency event is here.
    #   persistent  60 s markout clears its minimum and did NOT fade (T2);
    #               never a latency event, since a latency event has faded.
    #   other       toxic at some horizon, neither of the above (e.g. adverse
    #               at 1 s / 5 s only and reversed by 60 s).
    # "Faded" is Engine A's test on Engine A's early peak, so the two engines
    # cannot disagree about the same curve.
    keep = 1.0 - float(r.get("min_decay_pct", 50.0)) / 100.0
    late_mo = t[mcol(late)]
    if "early_peak_bps" in t.columns:
        peak = pd.to_numeric(t["early_peak_bps"], errors="coerce")
    else:
        peak = t[[mcol(h) for h in horizons if h <= 1.0]].max(axis=1)
    faded = ((late_mo <= keep * peak) & (peak > 0)).fillna(False)
    latency_event = (t["flagged"].fillna(False).astype(bool) if "flagged" in t.columns
                     else pd.Series(False, index=t.index))
    t["_toxic"] = t["_toxic"] | latency_event
    t["_sharp_fast"] = latency_event | (t["_toxic_early"] & faded)
    t["_persistent"] = (late_mo >= mo_min[late_label]).fillna(False) & ~faded & ~latency_event
    t["_sig"] = np.select([t["_sharp_fast"], t["_persistent"]], ["sharp_fast", "persistent"], "other")
    tox_sig = t.loc[t["_toxic"], "_sig"]
    summary["trade_signatures"] = {
        "sharp_fast": int((tox_sig == "sharp_fast").sum()),
        "persistent": int((tox_sig == "persistent").sum()),
        "other": int((tox_sig == "other").sum()),
        "latency_events": int(latency_event.sum()),
        # Latency events counted as toxic only because Engine A flagged them
        # (Engine A's early thresholds sit slightly below the s5.2 minimums).
        "latency_events_below_toxic_minimums": int(
            (latency_event & ~t[[f"_hit_{c}" for c in labels]].any(axis=1)).sum())}
    # The materially adverse orders themselves, so the account page can label
    # them. Private key: toxic_flow.build pops it before the summary is saved.
    if "order" in t.columns:
        summary["_toxic_orders"] = t.loc[t["_toxic"], ["account_key", "order", "open_time",
                                                       "_toxic_early", "_toxic_late", "_sig"]
                                         + [f"_hit_{hlabel(h)}" for h in horizons]]

    # ---- s3 Markout USD: the economic impact of the flow, per s13 "Toxic Flow
    # quantifies economic impact in USD".
    # A symbol whose trades never moved far enough to imply a contract size
    # gets the book's median instead of NaN: silently scoring the 15% economic
    # component as zero would read as "harmless", which is the wrong failure.
    usd_lot = _usd_per_price_lot(t)
    per_lot = t["canonical"].map(usd_lot)
    if usd_lot.notna().any():
        # A symbol with no qualifying move of its own takes the book's median
        # rather than NaN: silently scoring the 15% economic component as zero
        # would read as "harmless", which is the wrong way to fail.
        per_lot = per_lot.fillna(float(usd_lot.median()))
    usd_available = bool(np.isfinite(per_lot).any())
    summary["economic_impact"] = {
        "available": usd_available,
        "symbols_with_basis": int(usd_lot.notna().sum()),
        "trades_with_own_basis": int(t["canonical"].isin(usd_lot.index).sum()),
        "trades_total": int(len(t)),
        "note": ("" if usd_available else
                 "No symbol had a price move large enough to imply its contract "
                 "size, so adverse selection cannot be expressed in USD and the "
                 "s5.4 economic component (15%) is unavailable.")}
    t["_mo_usd"] = (t[mcol(late)] / 1e4 * t["open_price"] * per_lot * t["_lots"])
    t["_adverse_usd"] = t["_mo_usd"].where(t["_toxic"], 0.0).clip(lower=0)
    t["_gross_profit"] = t["net_profit"].clip(lower=0)
    t["_gross_loss"] = (-t["net_profit"]).clip(lower=0)
    t["_toxic_profit"] = t["_gross_profit"].where(t["_toxic"], 0.0)

    # ---- s5.1 Basic Sharp Deal Parameters: profitable, short-lived, and (when
    # enabled) paying more than a configured multiple of the round-trip cost.
    sharp = (t["net_profit"] >= float(r["sharp_min_profit_usd"])) \
        & (t["hold_seconds"] <= float(r["sharp_order_life_time_s"]))
    if r.get("check_profit_spread_ratio") and "entry_spread_rel" in t.columns:
        spread_cost = (pd.to_numeric(t["entry_spread_rel"], errors="coerce")
                       * t["open_price"] * per_lot * t["_lots"])
        t["_profit_spread_ratio"] = (t["net_profit"]
                                     / spread_cost.where(spread_cost > 0)).clip(-50, 50)
        sharp &= t["_profit_spread_ratio"] >= float(r["profit_spread_ratio"])
    else:
        t["_profit_spread_ratio"] = np.nan
    t["_sharp"] = sharp.fillna(False)

    # ---- POPULATION BENCHMARK (s3 "Population Benchmark Difference"): every
    # fill starts about half a spread behind, so an account is measured against
    # what the book does, not against zero.
    bench_curve = {label: float(t[mcol(h)].mean()) for h, label in zip(horizons, labels)}
    bench_rate = float(t["_toxic"].mean())
    x = np.log10(np.asarray(horizons, dtype=float))
    bench_y = np.nan_to_num(np.array([bench_curve[c] for c in labels], dtype=float))
    bench_auc = float((((bench_y[1:] + bench_y[:-1]) / 2) * np.diff(x)).sum() / (x[-1] - x[0]))
    summary["benchmark"] = {"curve_bps": {k: round(v, 4) for k, v in bench_curve.items()},
                            "auc_bps": round(bench_auc, 4),
                            "toxic_trade_rate": round(bench_rate, 4),
                            "trades": int(len(t))}

    g = t.groupby("account_key", observed=True)
    out = g.agg(
        trades=("_toxic", "size"),
        toxic_trades=("_toxic", "sum"),
        toxic_early=("_toxic_early", "sum"),
        toxic_late=("_toxic_late", "sum"),
        sharp_deals=("_sharp", "sum"),
        adverse_usd=("_adverse_usd", "sum"),
        markout_usd=("_mo_usd", "sum"),
        realized_pnl=("net_profit", "sum"),
        gross_profit=("_gross_profit", "sum"),
        gross_loss=("_gross_loss", "sum"),
        toxic_profit=("_toxic_profit", "sum"),
        latency_events=("flagged", "sum"),
        med_hold_s=("hold_seconds", "median"),
        lots_total=("_lots", "sum"),
        active_days=("_day", "nunique"),
        first_day=("_day", "min"),
        last_day=("_day", "max"),
        profit_spread_ratio=("_profit_spread_ratio", "median"),
    )
    out["toxic_trade_rate"] = out["toxic_trades"] / out["trades"].clip(lower=1)
    out["toxic_rate_excess"] = out["toxic_trade_rate"] - bench_rate
    out["avg_adverse_usd"] = (out["adverse_usd"]
                              / out["toxic_trades"].clip(lower=1)).where(out["toxic_trades"] > 0, 0.0)
    out["profit_factor"] = (out["gross_profit"]
                            / out["gross_loss"].where(out["gross_loss"] > 0)).clip(0, 50)
    #: s5.2 Minimum Toxic PnL Concentration %.
    out["toxic_pnl_concentration"] = (out["toxic_profit"]
                                      / out["gross_profit"].where(out["gross_profit"] > 0)).fillna(0).clip(0, 1)
    out["latency_share"] = (out["latency_events"] / out["trades"].clip(lower=1)).fillna(0)
    #: s5.1 Count of Sharp Deals is a 24-hour count: keep the busiest day.
    out["sharp_deals_24h"] = t.loc[t["_sharp"]].groupby(
        ["account_key", "_day"], observed=True).size().groupby(level=0).max() \
        .reindex(out.index).fillna(0)

    # ---- s3 COMMON MARKOUT METRICS, over the full seven-horizon curve
    curve = pd.DataFrame({label: g[mcol(h)].mean() for h, label in zip(horizons, labels)})
    for h, label in zip(horizons, labels):
        out[f"mo_{label}_bps"] = curve[label]
        out[f"med_{label}_bps"] = g[mcol(h)].median()
        # Vectorised hit rate: a per-group Python lambda over seven horizons
        # costs minutes at book scale. Masking to NaN keeps the semantics --
        # uncovered trades are skipped, an all-uncovered account stays NaN.
        out[f"hit_{label}"] = (t[mcol(h)] > 0).where(t[mcol(h)].notna()) \
            .groupby(t["account_key"], observed=True).mean()
    out["peak_markout_bps"] = curve[labels].max(axis=1)
    early_cols = [hlabel(h) for h in (list(early) + list(fallback)) if hlabel(h) in curve.columns]
    early_peak = curve[early_cols].max(axis=1) if early_cols else curve[[labels[0]]].max(axis=1)
    out["early_peak_bps"] = early_peak
    # PERSISTENCE RATIO (s3: "60 s markout relative to early/peak markout").
    # The ratio is only meaningful when there was an early advantage to persist
    # FROM. On a curve whose early peak sits at noise level the quotient blows
    # up -- 2.8 bps over an 0.09 bps peak reads as 32x and pins the clip -- so
    # the ratio is left undefined below the base floor and the account is
    # instead described by where its edge appeared. `late_emerging` marks the
    # accounts whose advantage only shows up at the long horizons.
    base_floor = float(r["persistence_min_base_bps"])
    usable_base = early_peak >= base_floor
    out["persistence_ratio"] = (curve[late_label]
                                / early_peak.where(usable_base)).clip(-10, 10)
    out["late_emerging"] = (~usable_base) & (curve[late_label] >= mo_min[late_label])
    out["decay_rate_pct"] = ((1.0 - out["persistence_ratio"]) * 100).clip(-500, 100)
    ys = curve[labels].to_numpy(dtype=float)
    xc = x - x.mean()
    with np.errstate(invalid="ignore"):
        out["curve_slope_bps_per_decade"] = (
            np.nansum((ys - np.nanmean(ys, axis=1, keepdims=True)) * xc, axis=1)
            / (xc ** 2).sum())
        yz = np.nan_to_num(ys)
        out["auc_bps"] = (((yz[:, 1:] + yz[:, :-1]) / 2) * np.diff(x)).sum(axis=1) / (x[-1] - x[0])
    out["auc_excess_bps"] = out["auc_bps"] - bench_auc
    #: Markout Consistency (s3): stability across days and across symbols.
    day_mean = t.groupby(["account_key", "_day"], observed=True)[mcol(late)].mean()
    out["consistency_days"] = day_mean.gt(bench_curve[late_label]).groupby(level=0).mean() \
        .reindex(out.index)
    sym = t.groupby(["account_key", "canonical"], observed=True)[mcol(late)].agg(["mean", "size"])
    sym = sym[sym["size"] >= 3]
    out["consistency_symbols"] = sym["mean"].gt(bench_curve[late_label]).groupby(level=0).mean() \
        .reindex(out.index)
    out["markout_consistency"] = out[["consistency_days", "consistency_symbols"]].mean(axis=1)
    out["n_symbols"] = g["canonical"].nunique()

    # ---- CONCENTRATION AND CONDITION FEATURES (T6-T10)
    tox = t.loc[t["_toxic"]]
    tox_g = tox.groupby("account_key", observed=True)
    sym_n = tox.groupby(["account_key", "canonical"], observed=True).size()
    out["top_symbol_share"] = (sym_n.groupby(level=0).max()
                               / out["toxic_trades"].clip(lower=1)).reindex(out.index)
    out["top_symbol"] = sym_n.groupby(level=0).idxmax().map(
        lambda k: str(k[1]) if isinstance(k, tuple) else "").reindex(out.index).fillna("")
    #: T9 Directional Toxicity: is one side of the book doing the damage?
    buy_tox = tox.loc[tox["direction"] > 0].groupby("account_key", observed=True).size()
    out["buy_toxic_share"] = (buy_tox / out["toxic_trades"].clip(lower=1)) \
        .reindex(out.index).fillna(0.0)
    out["direction_skew"] = (out["buy_toxic_share"] - 0.5).abs() * 2
    #: T10 Repeated Toxic Clusters: toxic trades arriving in bursts. A trade is
    #: clustered when another toxic trade from the same account lands within the
    #: cluster window.
    window = float(r["cluster_window_s"])
    times = tox.sort_values(["account_key", "open_time"])
    gap = times.groupby("account_key", observed=True)["open_time"].diff().dt.total_seconds()
    near = (gap <= window)
    near = near | near.groupby(times["account_key"], observed=True).shift(-1).fillna(False)
    out["clustered_share"] = near.groupby(times["account_key"], observed=True).mean() \
        .reindex(out.index).fillna(0.0)
    #: T8 Execution-Condition Toxicity: does the account do materially better
    #: when the spread is unusually wide or the quote unusually old?
    if "spread_ratio" in t.columns:
        wide = pd.to_numeric(t["spread_ratio"], errors="coerce") >= float(r["execution_condition_ratio"])
        t["_wide"] = wide.fillna(False)
        cond = t.groupby(["account_key", "_wide"], observed=True)[mcol(late)].mean().unstack()
        out["condition_lift_bps"] = (cond.get(True) - cond.get(False)).reindex(out.index) \
            if (True in cond.columns and False in cond.columns) else np.nan
        out["wide_spread_share"] = t.groupby("account_key", observed=True)["_wide"].mean() \
            .reindex(out.index).fillna(0.0)
    else:
        out["condition_lift_bps"] = np.nan
        out["wide_spread_share"] = 0.0
    out["event_share"] = (event_share.reindex(out.index).fillna(0.0)
                          if event_share is not None else 0.0)
    out["replicated_share"] = pd.Series(replicated or {}, dtype=float) \
        .reindex(out.index).fillna(0.0)

    # ---- s5.2 Reference Market Check: does the independent feed agree that the
    # market kept moving the client's way? Measured on the toxic trades only.
    ref_col = f"ref_exec_{late_label}_bps"
    has_ref = bool(r.get("reference_market_check")) and ref_col in t.columns \
        and t[ref_col].notna().any()
    if has_ref:
        ref = tox.loc[tox[ref_col].notna()].copy()
        ref["_agree"] = ref[ref_col] >= mo_min[late_label]
        rg = ref.groupby("account_key", observed=True)
        out["ref_checked"] = rg.size().reindex(out.index).fillna(0)
        out["ref_agree_share"] = rg["_agree"].mean().reindex(out.index)
        out["ref_mo_bps"] = rg[ref_col].mean().reindex(out.index)
        summary["reference_feed"] = "Vantage Raw ECN (independent)"
    else:
        out["ref_checked"] = 0.0
        out["ref_agree_share"] = np.nan
        out["ref_mo_bps"] = np.nan
        summary["reference_feed"] = ("reference market check disabled" if not r.get("reference_market_check")
                                     else "no reference ticks for this window")
    #: s5.2 LP/Venue Feedback Check -- no LP feed is received (see NOT_MEASURABLE).
    lp_available = bool(r.get("lp_venue_feedback_check"))
    out["lp_flagged_share"] = np.nan
    summary["lp_feedback"] = ("enabled but no LP feed is connected" if lp_available
                              else "not available -- s5.4 LP/execution component (10%) scores zero")

    # ---- s5.4 SCORE COMPONENTS (0..1), population-calibrated
    scale_fixed = dict(r["toxic_scale"])
    floor = r["toxic_floor"] or {}
    cal = r["toxic_calibration"] or {}
    raw = pd.DataFrame({"auc_excess_bps": out["auc_excess_bps"],
                        "toxic_rate_excess": out["toxic_rate_excess"],
                        "adverse_usd": out["adverse_usd"]}, index=out.index)
    used_scale: dict = {"mode": cal.get("mode", "fixed")}
    if cal.get("mode") == "population":
        pctl = float(cal.get("percentile", 99)) / 100
        active = out["trades"] >= int(cal.get("min_trades", 50))
        for key in ("auc_excess_bps", "toxic_rate_excess", "adverse_usd"):
            series = raw.loc[active, key].dropna()
            target = float(series.quantile(pctl)) if len(series) else np.nan
            if np.isfinite(target):
                scale_fixed[key] = float(min(max(target, float(floor.get(key, 0))),
                                             float(scale_fixed[key])))
        used_scale.update(percentile=cal.get("percentile", 99),
                          population_accounts=int(active.sum()))
    used_scale["full_marks_at"] = {k: round(float(scale_fixed[k]), 4)
                                   for k in ("auc_excess_bps", "toxic_rate_excess", "adverse_usd")}
    summary["score_scale"] = used_scale

    comp = pd.DataFrame(index=out.index)
    #: 25% Seven-horizon adverse markout -- the whole curve against the book's,
    #: as area under the curve, which is s3's "Aggregate advantage across horizons".
    comp["seven_horizon_markout"] = (raw["auc_excess_bps"] / scale_fixed["auc_excess_bps"]).clip(0, 1)
    #: 15% Toxic trade rate, in excess of the population rate.
    comp["toxic_trade_rate"] = (raw["toxic_rate_excess"] / scale_fixed["toxic_rate_excess"]).clip(0, 1)
    #: 15% Economic impact: total USD, with the s5.2 average-impact minimum as a
    #: second gate so many tiny adverse fills cannot score like real damage.
    comp["economic_impact"] = ((raw["adverse_usd"] / scale_fixed["adverse_usd"]).clip(0, 1)
                               * (out["avg_adverse_usd"] / float(r["avg_adverse_markout_usd"])).clip(0, 1))
    #: 15% Repetition / persistence: does the curve hold (persistence ratio) AND
    #: does the behaviour repeat (consistent days and symbols)?
    # Persistence: how much of the early edge survives to 60 s. Where there was
    # no early edge to survive, the account is measured on the late level it
    # reached instead, so a late-emerging advantage still counts as persistent
    # without borrowing a meaningless ratio.
    persist = out["persistence_ratio"].clip(0, 1).where(curve[late_label] > 0, 0.0)
    late_level = (curve[late_label] / mo_min[late_label]).clip(0, 1)
    persist = persist.fillna(late_level.where(out["late_emerging"], 0.0)).fillna(0)
    repeat = (out["markout_consistency"].fillna(0)
              * (out["active_days"] / 3.0).clip(upper=1)).clip(0, 1)
    comp["repetition_persistence"] = (0.5 * persist + 0.5 * repeat).clip(0, 1)
    #: 10% Profit concentration against the s5.2 minimum.
    comp["profit_concentration"] = (out["toxic_pnl_concentration"]
                                    / (float(r["min_toxic_pnl_concentration_pct"]) / 100)).clip(0, 1)
    #: 10% Reference-market corroboration, scaled by how much was checkable --
    #: one confirmed trade out of one is not repeated evidence.
    comp["reference_corroboration"] = (out["ref_agree_share"].fillna(0)
                                       * (out["ref_checked"] / 10.0).clip(upper=1)).fillna(0)
    #: 10% LP / execution evidence -- unavailable (NOT_MEASURABLE).
    comp["lp_execution_evidence"] = out["lp_flagged_share"].fillna(0.0)

    weights = dict(r["toxic_weights"])
    unavailable = ["lp_execution_evidence"]
    if not has_ref:
        unavailable.append("reference_corroboration")
    if not usd_available:
        unavailable.append("economic_impact")
    summary["unavailable_components"] = {k: round(float(weights.get(k, 0)) * 100, 1)
                                         for k in unavailable}
    summary["renormalised"] = bool(r.get("renormalise_unavailable"))
    if r.get("renormalise_unavailable") and unavailable:
        keep = {k: w for k, w in weights.items() if k not in unavailable}
        total = sum(keep.values()) or 1.0
        weights = {k: w / total for k, w in keep.items()}
    summary["weights_applied"] = {k: round(float(v), 4) for k, v in weights.items()}
    summary["weights_spec"] = {k: round(float(v), 4) for k, v in SPEC_WEIGHTS.items()}
    out["toxic_score"] = (100 * sum(comp[k].fillna(0) * float(w)
                                    for k, w in weights.items())).clip(0, 100).round(1)
    for key in SPEC_WEIGHTS:
        out[f"c_{key}"] = comp[key].round(4)

    # ---- s7 COMMON CONFIDENCE FRAMEWORK. Evidence-count bands set the base;
    # the listed factors increase or reduce it. s13: "Small samples cannot
    # automatically generate high-confidence classifications."
    events = out["toxic_trades"].astype(float)
    base = pd.Series(np.select(
        [events > 50, events >= 20, events >= 10, events >= 5],
        [0.90, 0.75, 0.55, 0.35], default=0.15), index=out.index)
    adj = pd.Series(0.0, index=out.index)
    adj += np.where(out["active_days"] >= 3, 0.05, 0.0)            # multiple days
    adj += np.where(out["active_days"] >= 10, 0.05, 0.0)           # multiple weeks
    adj += np.where((out["n_symbols"] >= 3)
                    & (out["consistency_symbols"].fillna(0) >= 0.6), 0.05, 0.0)
    adj += np.where(out["ref_agree_share"].fillna(0) >= 0.5, 0.10, 0.0)  # external confirmation
    adj += np.where(out["lp_flagged_share"].fillna(0) > 0, 0.05, 0.0)    # LP corroboration
    adj -= np.where((out["toxic_trades"] <= 1) | (out["n_symbols"] <= 1), 0.10, 0.0)
    #: Poor timestamp / reference data.
    adj -= np.where(out["ref_checked"] <= 0, 0.05, 0.0)
    #: Conflicting markout curve: adverse early, benign late or the reverse.
    conflicting = (out["toxic_early"] > 0) & (out["toxic_late"] > 0) \
        & (out["persistence_ratio"].fillna(0) < 0.2)
    adj -= np.where(conflicting, 0.10, 0.0)
    out["toxic_confidence"] = (100 * (base + adj).clip(0.0, 1.0)).round(1)
    out["conf_base"] = (100 * base).round(1)
    out["conflicting_curve"] = conflicting

    # ---- s5.3 CURVE SIGNATURE and s9 MARKOUT SIGNATURE
    # s5.3 describes CURVE SHAPES, so the classification reads the account's own
    # mean curve against the s5.2 minimums -- not how many individual trades
    # happened to clear a threshold on noise. An account whose average curve
    # sits at zero is not "sharp/fast" however many of its trades wobble past
    # the 100 ms minimum.
    sig = pd.DataFrame(False, index=out.index, columns=list(CURVE_PROFILES))
    toxic_enough = out["toxic_trades"] >= int(r["min_toxic_trades"])
    curve_early_ok = pd.concat([curve[c] >= mo_min[c] for c in early_labels],
                               axis=1).any(axis=1)
    curve_late_ok = curve[late_label] >= mo_min[late_label]
    any_signal = curve_early_ok | curve_late_ok
    sig["sharp_fast"] = curve_early_ok & (out["persistence_ratio"].fillna(0) < 0.5) \
        & ~curve_late_ok
    sig["persistent"] = curve_late_ok & ((out["persistence_ratio"] >= 0.5)
                                         | out["late_emerging"]).fillna(False)
    sig["event"] = any_signal & (out["event_share"] >= float(r["event_share"]))
    sig["symbol_specific"] = any_signal & (out["top_symbol_share"].fillna(0) >= float(r["symbol_concentration"])) \
        & (out["n_symbols"] > 1)
    sig["execution_condition"] = any_signal & (out["condition_lift_bps"].fillna(0) > 0) \
        & (out["wide_spread_share"].fillna(0) > 0.1)
    n_sig = sig[["sharp_fast", "persistent", "event", "symbol_specific",
                 "execution_condition"]].sum(axis=1)
    profile = pd.Series("none", index=out.index)
    for name in ("execution_condition", "symbol_specific", "event", "sharp_fast", "persistent"):
        profile[sig[name]] = name
    profile[n_sig >= 2] = "mixed"
    out["curve_profile"] = profile
    # s9 Markout Signature. It must never contradict the score: an account
    # whose curve sits above the book's baseline and is still positive at 60 s
    # is informed flow, even when it falls short of the s5.3 thresholds that
    # would give it a named curve profile. "Healthy" is reserved for flow that
    # is genuinely unremarkable, and "adverse" for flow that does WORSE than
    # the book -- which is benign for the broker, not a risk.
    # "Above the baseline" must mean MATERIALLY above it. Half the book sits
    # above its own mean by construction, so a bare `> 0` test labels 2,000
    # unremarkable accounts as fast or informed and leaves "healthy" empty.
    # The margin is a quarter of the full-marks scale, which is calibrated to
    # the book, so it tracks the population rather than a fixed guess.
    margin = max(0.25 * float(scale_fixed["auc_excess_bps"]), 0.1)
    summary["signature_margin_bps"] = round(margin, 4)
    above = out["auc_excess_bps"] >= margin
    signature = pd.Series("healthy", index=out.index)
    signature[out["auc_excess_bps"] <= -margin] = "adverse"
    signature[above & (curve[late_label] > 0)] = "informed"
    signature[above & (curve[late_label] <= 0)] = "fast"
    signature[sig["sharp_fast"]] = "fast"
    signature[sig["persistent"]] = "informed"
    signature[n_sig >= 2] = "mixed"
    out["markout_signature"] = signature

    # ---- s5.3.1 OBSERVABLE BEHAVIOUR PROFILES T1-T12
    prof = pd.DataFrame(False, index=out.index, columns=list(PROFILE_LABELS))
    prof["T1"] = (out["sharp_deals_24h"] >= int(r["sharp_deal_count"])) & sig["sharp_fast"]
    prof["T2"] = sig["persistent"] & (out["curve_slope_bps_per_decade"] > -0.1)
    prof["T3"] = toxic_enough & (out["toxic_trade_rate"] >= float(r["toxic_trade_rate_pct"]) / 100)
    prof["T4"] = (out["adverse_usd"] >= float(r["total_adverse_markout_usd"])) \
        | (out["avg_adverse_usd"] >= float(r["avg_adverse_markout_usd"]))
    prof["T5"] = toxic_enough & (out["toxic_pnl_concentration"]
                                 >= float(r["min_toxic_pnl_concentration_pct"]) / 100) \
        & (out["realized_pnl"] > 0)
    prof["T6"] = sig["event"]
    prof["T7"] = sig["symbol_specific"]
    prof["T8"] = sig["execution_condition"]
    prof["T9"] = toxic_enough & (out["direction_skew"] >= float(r["direction_skew"]))
    prof["T10"] = toxic_enough & (out["clustered_share"] >= float(r["cluster_share"]))
    prof["T11"] = out["replicated_share"] >= float(r["max_replicated_share"])
    prof["T12"] = False  # NOT_MEASURABLE: no LP/venue feedback
    out["profiles"] = prof.apply(lambda row: ",".join(row.index[row.to_numpy()]), axis=1)
    out["n_profiles"] = prof.sum(axis=1)

    # ---- GATES. s13: "Automated actions are governed separately from
    # classification" -- a gate never changes the score, only the state.
    gates = pd.DataFrame(index=out.index)
    gates["sample"] = out["toxic_trades"] < int(r["min_toxic_trades"])
    gates["confidence"] = out["toxic_confidence"] < float(r["toxic_min_confidence"])
    #: s15 Mandatory Market-Data Aggregation Root-Cause Control: "Same pricing
    #: opportunity is available to multiple clients -> treat as evidence of a
    #: systemic market-data/execution issue rather than solely a client
    #: behaviour issue."
    gates["market_data_condition"] = out["replicated_share"] > float(r["max_replicated_share"])
    #: s1: "latency is a mechanism while toxicity is the broader economic
    #: effect" -- where Engine A already owns the flow, defer to it.
    gates["latency_driven"] = out["latency_share"] > float(r["max_latency_share"])
    #: s7: a conflicting curve reduces confidence and classifies as mixed.
    gates["conflicting_curve"] = conflicting
    if r.get("check_total_pnl"):
        gates["total_pnl"] = out["realized_pnl"] < float(r["min_total_pnl_usd"])
    out["gates_failed"] = gates.apply(lambda row: ",".join(row.index[row.to_numpy()]), axis=1)

    # ---- s10 SUGGESTED DECISION MATRIX (score x confidence), then the s9 tier.
    states = [decision_state(float(s), float(c))
              for s, c in zip(out["toxic_score"], out["toxic_confidence"])]
    out["state"] = [s[0] for s in states]
    out["state_label"] = [s[1] for s in states]
    #: A failed gate can only lower the state, never raise it (s13 governance).
    capped = gates.any(axis=1) & ~out["state"].isin(["passive_monitoring", "monitor_evidence"])
    out.loc[capped, "state"] = "monitor_evidence"
    out.loc[capped, "state_label"] = "Monitor / collect evidence (held back by a gate)"
    out["capped"] = capped
    out["risk_tier"] = out["toxic_score"].map(risk_tier)
    #: s8 Severity: what the flow costs, not how sure we are of it.
    sev_scale = float(max(out["adverse_usd"].quantile(0.99), 1000.0))
    out["severity"] = (100 * (0.5 * (out["adverse_usd"] / sev_scale).clip(0, 1)
                              + 0.5 * comp["seven_horizon_markout"])).clip(0, 100).round(1)
    out["stability"] = (100 * out["markout_consistency"].fillna(0)).round(1)

    summary["gates"] = {k: int(v.sum()) for k, v in gates.items()}
    summary["accounts_scored"] = int(len(out))
    summary["curve_profiles"] = {k: int(v) for k, v in out["curve_profile"].value_counts().items()}
    summary["states"] = {k: int(v) for k, v in out["state"].value_counts().items()}
    return out, summary
