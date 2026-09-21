"""Spec scoring for the Latency Arbitrage engine.

Implements, over the per-trade markouts `latency_arb` produces, the parts of
"Latency Arbitrage Detection Engine -- Functional & Scoring Specification"
that turn evidence into a client assessment:

- s3   Common markout metrics per account (mean/median/hit rate per horizon,
       markout USD, peak, decay, persistence ratio, curve slope, area under
       curve, consistency across days and symbols, population benchmark).
- s5.1 Excluded account groups / symbols, account realised-PnL condition.
- s5.2 Minimum early hit rate, event success rate, profit concentration and
       confidence as ALERT GATES (a failed gate caps the verdict at monitor).
- s5.3 Latency Risk Score: the seven weighted components, the five bands, and
       confidence reported SEPARATELY from the score.
- s5.4 The observable behaviour profiles: L1 (reference dislocation; broker
       quote age only where no reference tick covers the account), L2, L3,
       L4 (reference lead/lag), L5, L7, L8 (symbols; feed IDs are not recorded),
       L9, L10. L6 needs constituent-feed update data and is reported as not
       measurable rather than silently scored as zero evidence.

Reference-market corroboration (10% of the score, 15% of confidence) comes
from the independent reference feeds -- Vantage Raw ECN and IC Markets Raw,
combined in latency_reference. Where no
reference tick covers an account's latency events, the component is 0 --
an absent reference must not inflate the score.
"""
from __future__ import annotations

import fnmatch

import numpy as np
import pandas as pd

#: s5.3 component weights (sum 1.0).
SPEC_WEIGHTS = {
    "early_markout": 0.20,
    "early_hit_rate": 0.15,
    "price_age": 0.20,
    "decay_consistency": 0.15,
    "event_count": 0.10,
    "profit_concentration": 0.10,
    "reference": 0.10,
}

#: s5.3 bands, lower bounds.
SPEC_BANDS = [(85, "critical"), (70, "high"), (50, "moderate"), (25, "emerging"), (0, "normal")]

#: Band -> the action layer's verdict (escalate / restrict / monitor / clear).
BAND_VERDICT = {"critical": "escalate", "high": "restrict", "moderate": "monitor",
                "emerging": "monitor", "normal": "clear"}

PROFILE_LABELS = {
    "L1": "Stale-price capture (broker price behind the reference at the fill)",
    "L2": "Fast positive markout",
    "L3": "Decaying latency advantage",
    "L4": "Reference-feed lead/lag exploitation",
    "L5": "Short-hold fast extraction",
    "L7": "Event / clustered latency activity",
    "L8": "Symbol-specific latency (symbols only, no feed IDs)",
    "L9": "Cross-account replication",
    "L10": "Profit-concentrated latency flow",
}
NOT_MEASURABLE = {
    "L6": "Timing after feed delays/update gaps: needs constituent-feed update data",
}

SPEC_DEFAULT_RULES = {
    #: s5.1 exclusions: fnmatch patterns, case-insensitive. Symbols match the
    #: raw OR canonical name; groups match the platform account group.
    "excluded_symbols": [],
    "excluded_account_groups": [],
    #: s5.1 account realised-PnL condition over the scan window.
    "check_total_realized_pnl": False,
    "min_total_realized_pnl_usd": 0.0,
    #: s5.2 alert gates. 0 disables a gate; calibrate from the scan's
    #: `spec_distributions` before tightening.
    "min_early_hit_rate_pct": 0.0,
    "min_event_success_rate_pct": 0.0,
    "min_profit_concentration_pct": 0.0,
    "min_confidence": 40.0,
    #: s5.2 maximum broker quote age that still counts as a stale quote (a
    #: longer gap is a data hole, not venue staleness).
    "max_price_age_ms": 60_000,
    "score_weights": dict(SPEC_WEIGHTS),
    #: POPULATION CALIBRATION (15 Sep 2026, user-directed): each magnitude
    #: component reaches full marks at the `percentile` account of this book,
    #: bounded below by `score_floor` (never award full marks for noise) and
    #: above by the fixed `score_scale` (never demand more than the standard).
    #: mode "fixed" uses `score_scale` alone. The shares that are already 0-1
    #: evidence (decay consistency, reference confirmation) are not rescaled.
    "score_calibration": {"mode": "population", "percentile": 99, "min_trades": 20},
    "score_floor": {
        "early_markout_bps": 0.25, "early_hit_rate_excess": 0.10, "price_age_share": 0.02,
        "event_count": 5, "profit_concentration": 0.05,
    },
    #: Where each component reaches full marks (fixed standard / upper bound).
    "score_scale": {
        "early_markout_bps": None,        # None = mean early flag threshold
        "early_hit_rate_excess": 0.40,    # hit rate above the population's
        "price_age_share": 0.10,          # share of ms fills with stale quotes
        "event_count": 20,                # latency events
        "profit_concentration": 0.50,     # latency-event PnL / gross profit
        "confidence_min_trades": 50,
    },
    "profile_params": {
        "cluster_window_s": 300, "cluster_min_events": 3,
        "replication_bucket_s": 2, "replication_min_accounts": 3,
        "symbol_concentration": 0.80,
    },
}


def spec_rules(rules: dict) -> dict:
    """Rules with every spec key present (nested dicts merged over defaults)."""
    out = dict(rules)
    for key, default in SPEC_DEFAULT_RULES.items():
        if isinstance(default, dict):
            out[key] = {**default, **(rules.get(key) or {})}
        elif key not in out:
            out[key] = default
    return out


def band_of(score: float) -> str:
    for lower, name in SPEC_BANDS:
        if score >= lower:
            return name
    return "normal"


def _matches(values: pd.Series, patterns: list) -> pd.Series:
    pats = [str(p).upper() for p in patterns or [] if str(p).strip()]
    if not pats:
        return pd.Series(False, index=values.index)
    upper = values.astype(str).str.upper()
    uniq = {v: any(fnmatch.fnmatchcase(v, p) for p in pats) for v in upper.unique()}
    return upper.map(uniq).astype(bool)


def apply_exclusions(trades: pd.DataFrame, rules: dict, groups: pd.Series | None = None
                     ) -> tuple[pd.DataFrame, dict]:
    """s5.1 Excluded Symbol Groups / Excluded Account Groups.
    `groups`: account_key -> platform group (only needed when group patterns exist)."""
    rules = spec_rules(rules)
    drop = _matches(trades["symbol"], rules["excluded_symbols"])
    if "canonical" in trades.columns:
        drop |= _matches(trades["canonical"], rules["excluded_symbols"])
    n_sym = int(drop.sum())
    n_grp = 0
    if rules["excluded_account_groups"] and groups is not None and len(groups):
        acct_group = trades["account_key"].map(groups).fillna("")
        g = _matches(acct_group, rules["excluded_account_groups"]) & ~drop
        n_grp = int(g.sum())
        drop |= g
    return trades.loc[~drop].copy(), {"excluded_symbol_trades": n_sym,
                                      "excluded_group_trades": n_grp}


def _usd_per_price_lot(trades: pd.DataFrame) -> pd.Series:
    """USD value of a 1.0 price move on 1 lot, per canonical symbol, inferred
    from the trades' own realised P&L (so contract size and FX are implied).
    Uses trades whose price moved > 5 bps; the median resists outliers."""
    move = trades["direction"] * (trades["close_price"] - trades["open_price"])
    lots = pd.to_numeric(trades.get("volume_lots"), errors="coerce")
    ok = ((move.abs() / trades["open_price"]) > 5e-4) & (lots > 0) & trades["net_profit"].notna()
    ratio = (trades.loc[ok, "net_profit"] / (move[ok] * lots[ok])).abs()
    return ratio.groupby(trades.loc[ok, "canonical"]).median()


def account_metrics(trades: pd.DataFrame, rules: dict, horizons: list, early: list,
                    fallback: list, late: float, mcol, hlabel, hkey,
                    arch_share: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """s3 metrics + s5.3 components + confidence + s5.4 profiles, per account.

    `trades` must already carry per-horizon markouts (mcol(h)), `ms_fill`,
    `quote_age_ms`, `flagged`, `fast`, `net_profit`, `hold_seconds`,
    `account_key`, `canonical`, `direction`, `open_time`, `open_price`,
    `close_price`, `volume_lots`; optionally the reference columns
    `ref_covered`, `ref_confirm`, `ref_dislocation`, `ref_leadlag`,
    `ref_exec_<h>_bps`, `ref_mid_<h>_bps` (latency_reference). `arch_share`:
    account -> share of its latency events inside a Market-Data Architecture
    Risk condition (s5.5). Returns (per-account frame, scan summary)."""
    rules = spec_rules(rules)
    scale = rules["score_scale"]
    pp = rules["profile_params"]
    thr = rules["flag_markout_bps"]
    t = trades.loc[trades[mcol(late)].notna()].copy()
    for col in ("ref_covered", "ref_confirm", "ref_validated", "ref_dislocation", "ref_leadlag"):
        t[col] = t[col].fillna(False).astype(bool) if col in t.columns else False
    has_ref = bool(t["ref_covered"].any())
    summary = {"not_measurable_profiles": NOT_MEASURABLE,
               "reference_feed": ("independent reference feeds (see reference.feeds)" if has_ref
                                  else "no reference ticks for this window (0 of 10 score points available)")}
    if not len(t):
        return pd.DataFrame(), summary

    precise = t["ms_fill"].fillna(False).astype(bool)
    sub = t[[mcol(h) for h in early]].mean(axis=1)
    sec = t[[mcol(h) for h in fallback]].mean(axis=1)
    t["early_mo_bps"] = sub.where(precise, sec)
    # Early hit on the window this fill's timestamp precision supports.
    hit_sub = pd.concat([t[mcol(h)] >= float(thr.get(hkey(h), 999)) for h in early], axis=1).any(axis=1)
    hit_sec = pd.concat([t[mcol(h)] >= float(thr.get(hkey(h), 999)) for h in fallback], axis=1).any(axis=1)
    t["early_hit"] = hit_sub.where(precise, hit_sec)
    t["early_pos"] = t["early_mo_bps"] > 0

    # POPULATION BENCHMARK per precision class: every fill starts about half a
    # spread behind, so an account is measured against normal flow, not zero.
    bench_mo = t.groupby(precise)["early_mo_bps"].mean()
    bench_hit = t.groupby(precise)["early_pos"].mean()
    t["excess_bps"] = t["early_mo_bps"] - precise.map(bench_mo)
    t["bench_hit"] = precise.map(bench_hit)
    stale_ms = float(rules.get("quote_throttle_ms", 200)) + float(rules.get("quote_age_tolerance_ms", 100))
    t["stale_quote"] = precise & (t["quote_age_ms"] > stale_ms) & (t["quote_age_ms"] <= float(rules["max_price_age_ms"]))
    pop_stale = float(t.loc[precise, "stale_quote"].mean()) if precise.any() else 0.0
    t["qualifying"] = t["fast"] & (t["net_profit"] >= float(rules["min_trade_profit_usd"]))
    t["_ms"] = precise
    t["_day"] = pd.to_datetime(t["open_time"]).dt.normalize()

    usd_lot = _usd_per_price_lot(t)
    t["early_mo_usd"] = (t["early_mo_bps"] / 1e4 * t["open_price"] * t["canonical"].map(usd_lot)
                         * pd.to_numeric(t.get("volume_lots"), errors="coerce"))
    t["_flag_pnl"] = t["net_profit"].where(t["flagged"], 0.0)
    t["_gross_profit"] = t["net_profit"].clip(lower=0)
    t["_q_success"] = t["qualifying"] & t["early_pos"]
    t["_hit_decayed"] = t["early_hit"] & t["flagged"]
    # Reference evidence (s5.3 corroboration, L1, L4).
    t["_flag_refcov"] = t["flagged"] & t["ref_covered"]
    t["_flag_confirm"] = t["_flag_refcov"] & t["ref_confirm"]
    t["_flag_validated"] = t["_flag_refcov"] & t["ref_validated"]
    t["_q_refcov"] = t["qualifying"] & t["ref_covered"]
    t["_disloc_hit"] = t["_q_refcov"] & t["ref_dislocation"] & t["early_hit"]
    t["_disloc"] = t["_q_refcov"] & t["ref_dislocation"]
    t["_leadlag_hit"] = t["_q_refcov"] & t["ref_leadlag"] & t["early_hit"]
    t["_leadlag"] = t["_q_refcov"] & t["ref_leadlag"]

    g = t.groupby("account_key", observed=True)
    out = g.agg(cov_trades=("early_mo_bps", "size"), ms_share=("_ms", "mean"),
                excess_bps=("excess_bps", "mean"), early_hit_rate=("early_pos", "mean"),
                bench_hit=("bench_hit", "mean"), stale_share=("stale_quote", "mean"),
                early_hits=("early_hit", "sum"), hit_decayed=("_hit_decayed", "sum"),
                latency_events=("flagged", "sum"), qualifying=("qualifying", "sum"),
                q_success=("_q_success", "sum"), flag_pnl=("_flag_pnl", "sum"),
                gross_profit=("_gross_profit", "sum"), realized_pnl=("net_profit", "sum"),
                early_mo_usd=("early_mo_usd", "sum"),
                flag_refcov=("_flag_refcov", "sum"), flag_confirm=("_flag_confirm", "sum"),
                flag_validated=("_flag_validated", "sum"),
                q_refcov=("_q_refcov", "sum"), disloc_hits=("_disloc_hit", "sum"),
                disloc=("_disloc", "sum"), leadlag_hits=("_leadlag_hit", "sum"),
                leadlag=("_leadlag", "sum"))
    out["reference_confirmed_share"] = (out["flag_confirm"] / out["flag_refcov"].clip(lower=1)) \
        .where(out["flag_refcov"] > 0)
    out["reference_validated_share"] = (out["flag_validated"] / out["flag_refcov"].clip(lower=1)) \
        .where(out["flag_refcov"] > 0)
    out["dislocation_share"] = (out["disloc"] / out["q_refcov"].clip(lower=1)).where(out["q_refcov"] > 0)
    out["leadlag_share"] = (out["leadlag"] / out["q_refcov"].clip(lower=1)).where(out["q_refcov"] > 0)
    # s2.1: account mean markout against the reference, executable
    # opposite side and (labelled separately) mid, over reference-covered trades.
    for h in horizons:
        for kind in ("exec", "mid"):
            col = f"ref_{kind}_{hlabel(h)}_bps"
            if col in t.columns:
                out[f"ref_mo_{kind}_{hlabel(h)}_bps"] = t.loc[t["ref_covered"]].groupby(
                    "account_key", observed=True)[col].mean().reindex(out.index)
    out["flag_days"] = t.loc[t["flagged"]].groupby("account_key", observed=True)["_day"].nunique() \
        .reindex(out.index).fillna(0)

    # ---- s3 curve metrics
    labels = [hlabel(h) for h in horizons]
    curve = pd.DataFrame({hlabel(h): g[mcol(h)].mean() for h in horizons})
    for h in horizons:
        out[f"med_{hlabel(h)}_bps"] = g[mcol(h)].median()
    out["peak_markout_bps"] = curve[labels].max(axis=1)
    early_peak = pd.concat([curve[[hlabel(h) for h in early]].max(axis=1),
                            curve[[hlabel(h) for h in fallback]].max(axis=1)], axis=1)
    early_peak = early_peak.iloc[:, 0].where(out["ms_share"] >= 0.5, early_peak.iloc[:, 1])
    out["persistence_ratio"] = (curve[hlabel(late)] / early_peak.where(early_peak > 0)).clip(-10, 10)
    x = np.log10(np.asarray(horizons, dtype=float))
    ys = curve[labels].to_numpy(dtype=float)
    xc = x - x.mean()
    with np.errstate(invalid="ignore"):
        out["curve_slope_bps_per_decade"] = (np.nansum((ys - np.nanmean(ys, axis=1, keepdims=True)) * xc, axis=1)
                                             / (xc ** 2).sum())
        # Area under the curve on a log-time axis, as an average bps level.
        yz = np.nan_to_num(ys)
        out["auc_bps"] = (((yz[:, 1:] + yz[:, :-1]) / 2) * np.diff(x)).sum(axis=1) / (x[-1] - x[0])
    day_pos = t.groupby(["account_key", "_day"], observed=True)["excess_bps"].mean().gt(0)
    out["consistency_days"] = day_pos.groupby(level=0).mean().reindex(out.index)
    sym = t.groupby(["account_key", "canonical"], observed=True)["excess_bps"].agg(["mean", "size"])
    sym = sym[sym["size"] >= 3]
    out["consistency_symbols"] = sym["mean"].gt(0).groupby(level=0).mean().reindex(out.index)
    out["markout_consistency"] = out[["consistency_days", "consistency_symbols"]].mean(axis=1)
    out["bench_diff_bps"] = out["excess_bps"]
    out["event_success_rate"] = (out["q_success"] / out["qualifying"].clip(lower=1)).where(out["qualifying"] > 0)
    out["profit_concentration"] = (out["flag_pnl"].clip(lower=0) / out["gross_profit"].where(out["gross_profit"] > 0)).fillna(0).clip(0, 1)

    # ---- s5.3 components (0..1)
    ms_thr = np.mean([float(thr.get(hkey(h), 1.0)) for h in early])
    sec_thr = np.mean([float(thr.get(hkey(h), 1.5)) for h in fallback])
    mag_scale = scale.get("early_markout_bps") or None
    fixed_mag = (pd.Series(float(mag_scale), index=out.index) if mag_scale
                 else (ms_thr * out["ms_share"] + sec_thr * (1 - out["ms_share"])))
    min_ev = max(int(rules.get("min_flagged_trades", 5)), 1)
    raw = pd.DataFrame({
        "early_markout_bps": out["excess_bps"],
        "early_hit_rate_excess": out["early_hit_rate"] - out["bench_hit"],
        "price_age_share": (out["stale_share"] - pop_stale).where(out["ms_share"] > 0),
        "event_count": out["latency_events"].astype(float),
        "profit_concentration": out["profit_concentration"]}, index=out.index)
    cal = rules.get("score_calibration") or {}
    floor = rules.get("score_floor") or {}
    fixed = {"early_hit_rate_excess": float(scale["early_hit_rate_excess"]),
             "price_age_share": float(scale["price_age_share"]),
             "event_count": float(scale["event_count"]),
             "profit_concentration": float(scale["profit_concentration"])}
    used_scale: dict = {"mode": cal.get("mode", "fixed")}
    if cal.get("mode") == "population":
        pctl = float(cal.get("percentile", 99)) / 100
        active = out["cov_trades"] >= int(cal.get("min_trades", 20))
        with_events = out["latency_events"] >= 1
        pop = {"early_markout_bps": raw.loc[active, "early_markout_bps"],
               "early_hit_rate_excess": raw.loc[active, "early_hit_rate_excess"],
               "price_age_share": raw.loc[active, "price_age_share"],
               "event_count": raw.loc[with_events, "event_count"],
               "profit_concentration": raw.loc[with_events, "profit_concentration"]}
        target = {k: (float(v.dropna().quantile(pctl)) if v.notna().any() else np.nan) for k, v in pop.items()}
        for k in fixed:
            t_k = target[k] if np.isfinite(target[k]) else fixed[k]
            fixed[k] = float(min(max(t_k, float(floor.get(k, 0))), fixed[k]))
        t_mag = target["early_markout_bps"] if np.isfinite(target["early_markout_bps"]) else np.inf
        fixed_mag = np.minimum(fixed_mag, max(t_mag, float(floor.get("early_markout_bps", 0.25))))
        used_scale.update(percentile=cal.get("percentile", 99),
                          population_accounts=int(active.sum()), accounts_with_events=int(with_events.sum()),
                          raw_percentile={k: (round(v, 4) if np.isfinite(v) else None) for k, v in target.items()})
    used_scale["full_marks_at"] = {**{k: round(v, 4) for k, v in fixed.items()},
                                   "early_markout_bps": round(float(np.nanmedian(np.asarray(fixed_mag, dtype=float))), 4)}
    comp = pd.DataFrame(index=out.index)
    comp["early_markout"] = (raw["early_markout_bps"] / fixed_mag).clip(0, 1)
    comp["early_hit_rate"] = (raw["early_hit_rate_excess"] / fixed["early_hit_rate_excess"]).clip(0, 1)
    # Price age is only eligible on millisecond fills (s2.0.1).
    comp["price_age"] = (raw["price_age_share"] / fixed["price_age_share"]).clip(0, 1).fillna(0.0)
    comp["decay_consistency"] = ((out["hit_decayed"] / out["early_hits"].clip(lower=1))
                                 * (out["early_hits"] / min_ev).clip(upper=1)).fillna(0)
    comp["event_count"] = (raw["event_count"] / fixed["event_count"]).clip(0, 1)
    comp["profit_concentration"] = (raw["profit_concentration"] / fixed["profit_concentration"]).clip(0, 1)
    summary["score_scale"] = used_scale
    # Reference corroboration: the share of latency events the independent
    # feed confirms, scaled by how many were checkable (one confirmed event
    # out of one is not repeated evidence).
    comp["reference"] = (out["reference_confirmed_share"].fillna(0)
                         * (out["flag_refcov"] / min_ev).clip(upper=1)).clip(0, 1)
    weights = {**SPEC_WEIGHTS, **(rules.get("score_weights") or {})}
    out["spec_score"] = (100 * sum(comp[k].fillna(0) * float(w) for k, w in weights.items())).clip(0, 100).round(1)
    for k in comp.columns:
        out[f"c_{k}"] = comp[k].round(3)
    out["band"] = out["spec_score"].map(band_of)

    # ---- confidence (separate from the score): sample size, data quality,
    # persistence, corroboration.
    sample = (out["cov_trades"] / float(scale["confidence_min_trades"])).clip(0, 1)
    base_rate = max(float(t["flagged"].mean()), 1e-4)
    mu = out["cov_trades"] * base_rate
    sd = np.sqrt(out["cov_trades"] * base_rate * (1 - base_rate)).clip(lower=1e-6)
    z = ((out["latency_events"] - mu) / sd).to_numpy(dtype=float)
    from math import erf, sqrt
    signif = pd.Series([0.5 * (1 + erf(v / sqrt(2))) for v in z], index=out.index)
    cov_all = trades.groupby("account_key", observed=True)[mcol(late)].apply(lambda s: s.notna().mean())
    quality = cov_all.reindex(out.index).fillna(0) * (0.5 + 0.5 * out["ms_share"])
    persistence = (out["flag_days"] / 3.0).clip(0, 1)
    # Corroboration of the MEASUREMENT: the independent market shows the same
    # favourable move (the latency-specific confirmation is in the score).
    corroboration = out["reference_validated_share"].fillna(0)
    out["conf_sample"] = (sample * signif).round(3)
    out["conf_quality"] = quality.round(3)
    out["conf_persistence"] = persistence.round(3)
    out["conf_corroboration"] = corroboration.round(3)
    out["spec_confidence"] = (100 * (0.35 * out["conf_sample"] + 0.25 * out["conf_quality"]
                                     + 0.25 * out["conf_persistence"] + 0.15 * corroboration)).round(1)

    # ---- s5.4 profiles
    flagged = t.loc[t["flagged"]].sort_values("open_time")
    prof = pd.DataFrame(False, index=out.index, columns=list(PROFILE_LABELS))
    enough = out["latency_events"] >= min_ev
    # L1: repeated fills while the broker price sat behind the reference (with
    # a fast positive markout). Accounts no reference tick covers fall back
    # to post-throttle broker quote age alone (recorded in l1_basis).
    ref_l1 = (out["disloc_hits"] >= min_ev) & (out["dislocation_share"].fillna(0) >= 0.25)
    age_l1 = enough & (comp["price_age"] >= 0.5) & (out["excess_bps"] > 0)
    prof["L1"] = ref_l1.where(out["q_refcov"] > 0, age_l1)
    out["l1_basis"] = np.where(out["q_refcov"] > 0, "reference", "quote_age_only")
    prof["L2"] = (comp["early_markout"] >= 0.5) & (comp["early_hit_rate"] >= 0.5)
    prof["L3"] = enough & (comp["decay_consistency"] >= 0.6)
    # L4: entries after the reference had already moved while the broker quote
    # had not (lead window = reference_price_delay_ms), repeatedly.
    prof["L4"] = (out["leadlag_hits"] >= min_ev) & (out["leadlag_share"].fillna(0) >= 0.25)
    if len(flagged):
        fh = flagged.groupby("account_key", observed=True)["hold_seconds"].median()
        prof["L5"] = enough & (fh.reindex(out.index) <= float(rules["max_hold_seconds"]) / 2) \
            & (out["profit_concentration"] >= 0.3)
        # L7: share of an account's latency events inside bursts.
        win = pd.Timedelta(seconds=float(pp["cluster_window_s"]))
        ot = pd.to_datetime(flagged["open_time"])
        gap = ot.groupby(flagged["account_key"]).diff() > win
        burst_id = gap.fillna(True).astype(int).groupby(flagged["account_key"]).cumsum()
        bsize = flagged.groupby([flagged["account_key"], burst_id])["flagged"].transform("size")
        in_burst = (bsize >= int(pp["cluster_min_events"])).groupby(flagged["account_key"]).mean()
        out["clustered_share"] = in_burst.reindex(out.index).fillna(0)
        prof["L7"] = enough & (out["clustered_share"] >= 0.5)
        # L8: symbol concentration of latency events.
        top_sym = flagged.groupby("account_key", observed=True)["canonical"].agg(
            lambda s: s.value_counts(normalize=True).iloc[0])
        out["top_symbol_share"] = top_sym.reindex(out.index).fillna(0)
        prof["L8"] = enough & (out["top_symbol_share"] >= float(pp["symbol_concentration"]))
        # L9: the same symbol + direction + time bucket flagged for several
        # distinct accounts -- shared strategy or a systemic pricing condition.
        bucket = ot.dt.floor(f"{int(pp['replication_bucket_s'])}s")
        key = flagged["canonical"].astype(str) + "|" + flagged["direction"].astype(str) + "|" + bucket.astype(str)
        n_acc = flagged.groupby(key)["account_key"].transform("nunique")
        replicated = n_acc >= int(pp["replication_min_accounts"])
        out["replicated_share"] = replicated.groupby(flagged["account_key"]).mean().reindex(out.index).fillna(0)
        prof["L9"] = enough & (out["replicated_share"] >= 0.5)
        conds = flagged.loc[replicated].assign(_k=key[replicated]).groupby("_k").agg(
            accounts=("account_key", "nunique"), events=("flagged", "size"), pnl=("net_profit", "sum"))
        summary["replicated_conditions"] = int(len(conds))
        summary["top_replicated"] = [
            {"condition": k, "accounts": int(r.accounts), "events": int(r.events), "pnl": round(float(r.pnl), 2)}
            for k, r in conds.sort_values("accounts", ascending=False).head(10).iterrows()]
    else:
        out["clustered_share"] = 0.0
        out["top_symbol_share"] = 0.0
        out["replicated_share"] = 0.0
    prof["L10"] = enough & (out["profit_concentration"] >= 0.5)
    out["profiles"] = [",".join(c for c in prof.columns if row[c]) for _, row in prof.iterrows()]

    # ---- s5.1 / s5.2 alert gates: failing any caps the verdict at monitor.
    gates = pd.DataFrame(index=out.index)
    if rules.get("check_total_realized_pnl"):
        gates["realized_pnl"] = out["realized_pnl"] >= float(rules["min_total_realized_pnl_usd"])
    gates["early_hit_rate"] = 100 * out["early_hit_rate"] >= float(rules["min_early_hit_rate_pct"])
    gates["event_success_rate"] = 100 * out["event_success_rate"].fillna(0) >= float(rules["min_event_success_rate_pct"])
    gates["profit_concentration"] = 100 * out["profit_concentration"] >= float(rules["min_profit_concentration_pct"])
    gates["confidence"] = out["spec_confidence"] >= float(rules["min_confidence"])
    # s5.5.2 / s5.5.4: a pattern replicated across unrelated accounts points to
    # a pricing condition first -- no adverse client action until isolated.
    gates["not_replicated"] = ~prof["L9"]
    # s5.5.4: most of the account's latency events sit inside a feed/symbol
    # condition classified as Market-Data Architecture Risk -- the system,
    # not (only) the client, created the opportunity.
    out["architecture_share"] = pd.Series(arch_share or {}, dtype=float).reindex(out.index).fillna(0.0)
    gates["market_data_condition"] = out["architecture_share"] < 0.5
    out["gates_failed"] = [",".join(c for c in gates.columns if not row[c]) for _, row in gates.iterrows()]

    verdict = out["band"].map(BAND_VERDICT)
    capped = (out["gates_failed"] != "") & verdict.isin(["escalate", "restrict"])
    out["spec_verdict"] = verdict.where(~capped, "monitor")

    summary["population_benchmark"] = {
        "early_markout_bps": {("ms" if k else "sec"): round(float(v), 3) for k, v in bench_mo.items()},
        "early_hit_rate": {("ms" if k else "sec"): round(float(v), 3) for k, v in bench_hit.items()},
        "stale_quote_share_ms": round(pop_stale, 4)}
    summary["usd_per_price_lot_symbols"] = int(len(usd_lot))
    summary["weights"] = weights
    ev = out[out["latency_events"] >= min_ev]
    if len(ev):
        q = lambda s: {p: round(float(s.quantile(p / 100)), 3) for p in (25, 50, 75, 90)}
        summary["spec_distributions"] = {
            "accounts": int(len(ev)),
            "early_hit_rate_pct": q(100 * ev["early_hit_rate"]),
            "event_success_rate_pct": q(100 * ev["event_success_rate"].fillna(0)),
            "profit_concentration_pct": q(100 * ev["profit_concentration"]),
            "confidence": q(ev["spec_confidence"]), "score": q(ev["spec_score"]),
            "reference_confirmed_pct": q(100 * ev["reference_confirmed_share"].fillna(0)),
            "dislocation_share_pct": q(100 * ev["dislocation_share"].fillna(0)),
            "leadlag_share_pct": q(100 * ev["leadlag_share"].fillna(0))}
    return out, summary
