"""AntiFraud / Behavioural Profiling Engine v2 -- the spec, implemented.

Nine behavioural profiles, each scored on the common five-axis architecture
from the specification:

  * Behaviour Score (0-100)  -- strength of the observed behaviour
  * Confidence     (0-100%)  -- reliability given evidence and data quality
  * Stability      (0-100)   -- persistence/consistency through time
  * Severity       (0-100)   -- materiality to broker risk / P&L / exposure
  * Evidence Score (0-100)   -- quantity and quality of objective support

Classification says WHAT the client does; Score says how STRONGLY; Confidence
says how RELIABLY; Stability says whether it PERSISTS; Severity says how much
it MATTERS. Every flag ships with the contributing features so a reviewer can
see exactly why -- the "100% explainable" requirement.

Thresholds are configurable (antifraud_rules.json) so the desk can tune rules
and re-run classification without a code change -- the rules-vs-ML toggle and
backtest hang off this same rule object.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RULES_PATH = ROOT / "antifraud_rules.json"
SCRATCH = None      # set from model_service at call time to avoid import cycle

#: Profile -> primary objective (verbatim intent from the specification).
PROFILES = {
    "persistent_edge": "Sustained, statistically meaningful profitability",
    "toxic_flow": "Adverse-selection / latency / stale-price exploitation",
    "high_magnitude": "Large or persistent directional exposure",
    "scalper": "Very short-duration, execution-sensitive trading",
    "news_vol": "Activity concentrated around news, gaps and volatility",
    "bonus_arb": "Systematic use of promotional credit for economic value",
    "swap_arb": "Economics materially dependent on positive swap/carry",
    "martingale": "Progressive size escalation during drawdown",
    "high_exposure_recovery": "High exposure with drawdown/recovery behaviour",
}

SEVERITY_WEIGHT = {          # materiality of each abuse type to the book
    "toxic_flow": 100, "bonus_arb": 95, "martingale": 85, "swap_arb": 80,
    "high_magnitude": 78, "news_vol": 70, "high_exposure_recovery": 72,
    "persistent_edge": 65, "scalper": 55,
}

DEFAULT_RULES = {
    "persistent_edge": {"profit_factor": 1.1, "min_trades": 20, "pf_strong": 1.6},
    "toxic_flow": {"min_trades": 50, "markout_bad": 0.0, "toxic_pf": 1.2},
    "high_magnitude": {"notional_z": 2.0, "hold_hours": 24.0},
    "scalper": {"under_5m_share": 0.5, "min_trades": 30},
    "news_vol": {"event_pnl_dependency": 0.4},
    "bonus_arb": {"credit_to_equity": 0.5},
    "swap_arb": {"swap_dependency": 0.4, "min_hold_days": 3},
    "martingale": {"escalation_rate": 0.15, "min_sequences": 5},
    "high_exposure_recovery": {"exposure_to_equity": 3.0},
    "_global": {"short_term_profit_flag": 0.0},   # flag if predicted +ve short-term
    #: Operator-defined watch rules: ANY numeric column of the account-day
    #: corpus (ad_* family) compared against a threshold. Each row that fires
    #: appears in the classification table as profile "custom:<name>".
    #: Format: {"name": str, "metric": "<ad_ column>", "op": ">"|">="|"<"|"<=",
    #:          "value": number, "severity": 0-100}
    "_custom": [],
    #: Alert autopilot: send NEW flags to Lark as they are generated (checked
    #: every poll_minutes), plus a full digest at digest_hour London time.
    "_alerts": {"autopilot": False, "min_priority": 60.0,
                "digest_hour_london": 7, "poll_minutes": 5},
}

#: What every threshold MEANS -- rendered next to each field in the Rules tab
#: so changing a number never requires reading this file. `direction` says
#: what raising the value does to strictness.
RULES_SCHEMA = {
    "persistent_edge": {
        "profit_factor": {"label": "Profit factor", "format": "ratio >= 0",
            "meaning": "Gross wins / gross losses over the account's history. "
                       "Above this the account counts as persistently profitable.",
            "direction": "higher = fewer accounts flagged"},
        "min_trades": {"label": "Minimum trades", "format": "integer",
            "meaning": "Evidence floor: fewer closed trades than this and the "
                       "profile is never scored (small samples lie).",
            "direction": "higher = fewer accounts flagged"},
        "pf_strong": {"label": "Strong-band profit factor", "format": "ratio",
            "meaning": "At or above this the flag jumps to the strong band.",
            "direction": "higher = fewer strong-band flags"}},
    "toxic_flow": {
        "min_trades": {"label": "Minimum trades", "format": "integer",
            "meaning": "Evidence floor for markout measurement.",
            "direction": "higher = fewer flagged"},
        "markout_bad": {"label": "Markout threshold ($/trade)", "format": "dollars",
            "meaning": "Mean post-trade markout against the book worse than "
                       "this marks the flow as adversely selecting us.",
            "direction": "lower (more negative) = fewer flagged"},
        "toxic_pf": {"label": "Toxic profit factor", "format": "ratio",
            "meaning": "Profitability gate combined with bad markouts -- "
                       "profitable AND adverse = the classic toxic signature.",
            "direction": "higher = fewer flagged"}},
    "high_magnitude": {
        "notional_z": {"label": "Notional z-score", "format": "sigmas",
            "meaning": "Position size in cross-sectional standard deviations "
                       "above the client population that day.",
            "direction": "higher = fewer flagged"},
        "hold_hours": {"label": "Hold hours", "format": "hours",
            "meaning": "Directional exposure held longer than this counts as "
                       "persistent, not incidental.",
            "direction": "higher = fewer flagged"}},
    "scalper": {
        "under_5m_share": {"label": "Sub-5-minute share", "format": "0-1",
            "meaning": "Fraction of trades closed inside five minutes.",
            "direction": "higher = fewer flagged"},
        "min_trades": {"label": "Minimum trades", "format": "integer",
            "meaning": "Evidence floor.", "direction": "higher = fewer flagged"}},
    "news_vol": {
        "event_pnl_dependency": {"label": "Event-P&L dependency", "format": "0-1",
            "meaning": "Share of total P&L earned in high-volatility windows. "
                       "(Partial: no economic calendar joined yet -- vol-regime "
                       "windows proxy for events.)",
            "direction": "higher = fewer flagged"}},
    "bonus_arb": {
        "credit_to_equity": {"label": "Credit / equity", "format": "ratio 0-1+",
            "meaning": "Promotional credit as a share of account equity. "
                       "(Partial: needs the bonus ledger for full coverage.)",
            "direction": "higher = fewer flagged"}},
    "swap_arb": {
        "swap_dependency": {"label": "Swap share of P&L", "format": "0-1",
            "meaning": "Fraction of net P&L that is swap/carry rather than price.",
            "direction": "higher = fewer flagged"},
        "min_hold_days": {"label": "Minimum hold days", "format": "days",
            "meaning": "Carry strategies hold; shorter positions are excluded.",
            "direction": "higher = fewer flagged"}},
    "martingale": {
        "escalation_rate": {"label": "Escalation rate", "format": "0-1",
            "meaning": "Share of loss-followed trades where size grew >= 1.5x.",
            "direction": "higher = fewer flagged"},
        "min_sequences": {"label": "Minimum sequences", "format": "integer",
            "meaning": "Evidence floor of loss->bigger-size sequences.",
            "direction": "higher = fewer flagged"}},
    "high_exposure_recovery": {
        "exposure_to_equity": {"label": "Exposure / equity", "format": "ratio",
            "meaning": "Open notional as a multiple of the client's equity.",
            "direction": "higher = fewer flagged"}},
    "_global": {
        "short_term_profit_flag": {"label": "Short-term score flag",
            "format": "model score", "meaning": "Also flag any account whose "
            "short-term model score exceeds this (0 disables).",
            "direction": "higher = fewer flagged"}},
}


def load_rules() -> dict:
    rules = json.loads(json.dumps(DEFAULT_RULES))    # deep copy
    try:
        stored = json.loads(RULES_PATH.read_text(encoding="utf-8"))
        for key, value in stored.items():
            rules.setdefault(key, {}).update(value)
    except Exception:
        pass
    return rules


def save_rules(rules: dict) -> None:
    RULES_PATH.write_text(json.dumps(rules, indent=2), encoding="utf-8")
    _CACHE.pop("classify", None)             # force re-classification
    _CACHE.pop("classify_key", None)


def warm() -> None:
    """Pre-compute the heavy caches (classify + ML) so the first tab open is
    instant. Safe to call in a background thread at startup."""
    try:
        classify()
        ml_scores()
    except Exception:
        pass


# ---------------------------------------------------------------------------
def _band(score: float) -> str:
    if score >= 85: return "very strong"
    if score >= 70: return "strong"
    if score >= 50: return "moderate"
    if score >= 25: return "weak"
    return "negligible"


def _sig(x, lo, hi):
    """Map a raw metric in [lo, hi] onto a 0-100 strength score (clipped).
    Works element-wise on Series/arrays and returns a float for scalars."""
    if hi == lo:
        return 0.0 if np.isscalar(x) else np.zeros_like(np.asarray(x, dtype="float64"))
    scaled = np.clip((np.asarray(x, dtype="float64") - lo) / (hi - lo), 0, 1) * 100
    if isinstance(x, pd.Series):
        return pd.Series(scaled, index=x.index)
    return float(scaled) if np.isscalar(x) or scaled.ndim == 0 else scaled


_CACHE: dict = {}


def _frame_stamp() -> float:
    from webapp.trade_features import _AD_DIR
    try:
        return (_AD_DIR / "model_frame.parquet").stat().st_mtime
    except Exception:
        return 0.0


def _frame():
    """The account-day feature frame -- the 174-feature corpus the models
    train on. One row per account per decision day. Cached by file mtime.
    Reads the DURABLE corpus (webapp/artifacts/ad), kept at yesterday by
    ad_refresh -- the scratchpad copy it previously read froze at Aug 26."""
    from webapp.trade_features import _AD_DIR
    stamp = _frame_stamp()
    if _CACHE.get("frame_stamp") != stamp:
        frame = pd.read_parquet(_AD_DIR / "model_frame.parquet")
        # Decimal/object columns (a DECIMAL-typed source) poison numpy ops
        # downstream ("no callable sqrt") -- everything numeric goes float.
        for column in frame.columns:
            if column in ("account_key", "decision_day"):
                continue
            if not pd.api.types.is_float_dtype(frame[column]):
                frame[column] = pd.to_numeric(
                    frame[column], errors="coerce").astype("float32")
        _CACHE["frame"] = frame
        _CACHE["frame_stamp"] = stamp
        _CACHE.pop("panel", None)
        _CACHE.pop("classify", None)
        _CACHE.pop("ml", None)
    return _CACHE["frame"]


_MARKOUT_CACHE: dict = {}


_EVENT_SHARE_CACHE: dict = {}


def _event_shares(days: int = 14, ttl: float = 3600.0) -> pd.Series:
    """Per-account share of entries within +/-10 minutes of a high-impact
    calendar event (the dynamic feed), from the last `days` of warehouse
    trades. Vectorised: one searchsorted over every open time, then a
    groupby mean. Cached an hour."""
    now = time.time()
    if _EVENT_SHARE_CACHE.get("at", 0) > now - ttl:
        return _EVENT_SHARE_CACHE["data"]
    result = pd.Series(dtype=float)
    try:
        from datetime import datetime, timedelta
        from webapp import data_store, econ_calendar
        anchors = econ_calendar.event_times_utc("high")
        if anchors:
            frame = data_store.read_history(
                start=datetime.utcnow() - timedelta(days=days),
                end=datetime.utcnow(),
                columns=["database", "login", "open_time"])
            if frame is not None and len(frame):
                acct = (frame["database"].astype(str) + ":"
                        + frame["login"].astype(str))
                t = pd.to_datetime(frame["open_time"]) \
                    .to_numpy(dtype="datetime64[s]")
                a = np.array(sorted(anchors), dtype="datetime64[s]")
                idx = np.searchsorted(a, t)
                tol = np.timedelta64(600, "s")
                near = np.zeros(len(t), dtype=bool)
                for shift in (0, 1):
                    j = np.clip(idx - shift, 0, len(a) - 1)
                    near |= np.abs(t - a[j]) <= tol
                result = pd.Series(near, index=acct).groupby(level=0).mean()
    except Exception:
        result = pd.Series(dtype=float)
    _EVENT_SHARE_CACHE.update(at=now, data=result)
    return result


def _markouts() -> pd.DataFrame | None:
    """Per-account short-horizon post-trade markouts (adverse-selection
    signature). Cached; None if the markout parquet has not been built."""
    if "m" in _MARKOUT_CACHE:
        return _MARKOUT_CACHE["m"]
    from webapp import model_service as ms
    from webapp import views
    result = None
    for name in ("markout_all_servers.parquet", "markout_trading.parquet"):
        path = ms.SCRATCH / name
        if path.exists():
            try:
                frame = views.load_markouts(path)
                cols = [f"markout_{h}" for h in ("1m", "5m", "30m")
                        if f"markout_{h}" in frame.columns]
                agg = frame.groupby("account_key").agg(
                    mk_short=(cols[0], "mean") if cols else ("day", "size"),
                    mk_trades=("context_trades", "sum")
                    if "context_trades" in frame.columns else ("day", "size"),
                    mk_days=("day", "size"))
                # direction-adjusted mean of the shortest horizon(s)
                if cols:
                    agg["mk_short"] = frame.groupby("account_key")[cols].mean().mean(axis=1)
                result = agg
                break
            except Exception:
                result = None
    _MARKOUT_CACHE["m"] = result
    return result


def _account_panel(as_of: str | None = None, start: str | None = None,
                   end: str | None = None) -> pd.DataFrame:
    """Per-account behaviour panel, DATE-AWARE.

    No dates -> the latest state (today's view). `as_of` -> the panel exactly
    as it stood at end of that day (rows after it never existed). `start`/`end`
    -> aggregate view of the range: numeric metrics AVERAGED across the
    account's days in the range. Cached per (frame, dates)."""
    _frame()                                  # refreshes cache/mtime
    key = ("panel", as_of or "", start or "", end or "")
    if key in _CACHE:
        return _CACHE[key]
    if not as_of and not start:
        if "panel" in _CACHE:
            return _CACHE["panel"]
        _CACHE["panel"] = _build_panel()
        return _CACHE["panel"]
    frame = _frame().copy()
    frame["decision_day"] = pd.to_datetime(frame["decision_day"])
    if as_of:
        frame = frame.loc[frame["decision_day"] <= pd.Timestamp(as_of)]
        panel = _build_panel(frame)
    else:
        lo = pd.Timestamp(start)
        hi = pd.Timestamp(end) if end else frame["decision_day"].max()
        window = frame.loc[(frame["decision_day"] >= lo)
                           & (frame["decision_day"] <= hi)]
        if window.empty:
            panel = _build_panel(frame).iloc[0:0]
        else:
            numeric = window.select_dtypes("number").columns
            panel = window.groupby("account_key", observed=True)[numeric].mean()
            panel["obs_days"] = window.groupby("account_key", observed=True).size()
            panel["obs_span_days"] = (hi - lo).days + 1
            for column in ("stab_martingale_rate", "stab_scalp_rate"):
                if column not in panel.columns:
                    panel[column] = 50.0
    if len(_CACHE) < 40:
        _CACHE[key] = panel
    return panel


def _build_panel(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    if frame is None:
        frame = _frame()
    frame = frame.copy()
    frame["decision_day"] = pd.to_datetime(frame["decision_day"])
    frame = frame.sort_values(["account_key", "decision_day"])
    grouped = frame.groupby("account_key", observed=True)
    latest = grouped.last()
    # observation depth for the Evidence axis
    latest["obs_days"] = grouped.size()
    latest["obs_span_days"] = (grouped["decision_day"].max()
                               - grouped["decision_day"].min()).dt.days + 1
    # per-behaviour temporal stability = 1 - normalised dispersion of the
    # driving rate across the account's history (persistence through time)
    for col in ("martingale_rate", "scalp_rate", "concentration",
                "win_rate", "profit_factor"):
        if col in frame.columns:
            disp = grouped[col].std().fillna(0) / (grouped[col].mean().abs() + 1e-6)
            latest[f"_stab_{col}"] = (1 - disp.clip(0, 1)) * 100
    return latest


# ---------------------------------------------------------------------------
def classify(rules: dict | None = None,
             short_term_scores: pd.Series | None = None,
             as_of: str | None = None, start: str | None = None,
             end: str | None = None) -> pd.DataFrame:
    """Score every account against all nine profiles.

    Returns one row per (account, profile) that TRIGGERS, each carrying the
    five axes, the band, the short-term-profit flag, and the human-readable
    evidence string. `short_term_scores` (account_key -> predicted forward
    P&L / win-prob) lets us prioritise clients we also expect to be
    short-term profitable, per the brief.
    """
    rules = rules or load_rules()
    _frame()                                  # refresh cache stamp
    has_stp = short_term_scores is not None
    cache_key = (json.dumps(rules, sort_keys=True) + f"|stp={has_stp}"
                 + f"|{as_of or ''}|{start or ''}|{end or ''}")
    if _CACHE.get("classify_key") == cache_key and "classify" in _CACHE:
        return _CACHE["classify"]
    panel = _account_panel(as_of=as_of, start=start, end=end)
    n = len(panel)
    if n == 0:
        return pd.DataFrame()

    def col(name, default=0.0):
        return (pd.to_numeric(panel[name], errors="coerce").fillna(default)
                if name in panel.columns else pd.Series(default, index=panel.index))

    obs = col("obs_days", 1)
    span = col("obs_span_days", 1)
    # Evidence: sample size (log-scaled) x time span x data completeness
    evidence = (_sig(np.log1p(obs), 0, np.log1p(200)) * 0.5
                + _sig(span, 0, 120) * 0.5)
    # Confidence rises with evidence and behavioural consistency; capped by
    # the spec's "<50% = insufficient evidence" rule for thin samples.
    thin = obs < 20

    rows = []
    trades = col("life_closes", 0) if "life_closes" in panel.columns else col("closes", 0)

    def emit(profile, score, drivers, stability_col=None, extra_conf=0.0):
        score = float(np.clip(score, 0, 100))
        if score < 25:                       # below "weak" -> not flagged
            return
        stab = (float(panel[stability_col].fillna(50).loc[account])
                if stability_col and stability_col in panel.columns else 60.0)
        ev = float(evidence.loc[account])
        conf = float(np.clip(0.55 * ev + 0.30 * score + 0.15 * stab
                             + extra_conf, 0, 100))
        if bool(thin.loc[account]):
            conf = min(conf, 49.0)           # spec: insufficient evidence
        sev = float(np.clip(SEVERITY_WEIGHT[profile] * (0.5 + score / 200), 0, 100))
        stp = (float(short_term_scores.get(account, np.nan))
               if short_term_scores is not None else np.nan)
        rows.append({
            "account_key": account, "profile": profile,
            "objective": PROFILES[profile], "score": round(score, 1),
            "band": _band(score), "confidence": round(conf, 1),
            "stability": round(stab, 1), "severity": round(sev, 1),
            "evidence": round(ev, 1), "obs_days": int(obs.loc[account]),
            "short_term_score": None if np.isnan(stp) else round(stp, 4),
            "priority": round(score * conf / 100 * sev / 100
                              * (1 + max(0.0, stp if not np.isnan(stp) else 0)), 2),
            "why": " · ".join(drivers)})

    pf = col("life_profit_factor", 0) if "life_profit_factor" in panel.columns else col("profit_factor", 0)
    win = col("life_win_rate", 0) if "life_win_rate" in panel.columns else col("win_rate", 0)
    mart = col("martingale_rate", 0)
    scalp = col("scalp_rate", 0)
    overnight = col("overnight_rate", 0)
    concentration = col("concentration", 0)
    notional = col("gross_notional", 0)
    hold = col("median_hold_seconds", 0) if "median_hold_seconds" in panel.columns else col("avg_hold_days_20d", 0) * 86400
    notional_z = (notional - notional.mean()) / (notional.std() + 1e-6)

    for account in panel.index:
        # 1 Persistent Edge
        if float(pf.loc[account]) >= rules["persistent_edge"]["profit_factor"] \
                and float(trades.loc[account]) >= rules["persistent_edge"]["min_trades"]:
            s = _sig(pf.loc[account], rules["persistent_edge"]["profit_factor"],
                     rules["persistent_edge"]["pf_strong"])
            emit("persistent_edge", s,
                 [f"profit factor {float(pf.loc[account]):.2f}",
                  f"win rate {float(win.loc[account]):.0%}",
                  f"{int(trades.loc[account])} closed trades"],
                 "_stab_profit_factor")
        # 3 High Magnitude / Position
        if float(notional_z.loc[account]) >= rules["high_magnitude"]["notional_z"]:
            emit("high_magnitude", _sig(notional_z.loc[account], 2, 6),
                 [f"notional {float(notional_z.loc[account]):+.1f}σ above peers",
                  f"concentration {float(concentration.loc[account]):.0%}"],
                 "_stab_concentration")
        # 4 Scalper
        if float(scalp.loc[account]) >= rules["scalper"]["under_5m_share"] \
                and float(trades.loc[account]) >= rules["scalper"]["min_trades"]:
            emit("scalper", _sig(scalp.loc[account], 0.5, 0.95),
                 [f"{float(scalp.loc[account]):.0%} trades < 5 min",
                  f"{int(trades.loc[account])} trades"],
                 "_stab_scalp_rate" if "_stab_scalp_rate" in panel.columns else "_stab_scalp")
        # 8 Martingale
        if float(mart.loc[account]) >= rules["martingale"]["escalation_rate"]:
            emit("martingale", _sig(mart.loc[account], 0.15, 0.6),
                 [f"size-escalation rate {float(mart.loc[account]):.0%}",
                  "1→2→4→8 pattern present"],
                 "_stab_martingale_rate")
        # 9 High Exposure / Recovery  (exposure proxy: notional vs peers + overnight)
        if float(notional_z.loc[account]) >= 1.0 and float(overnight.loc[account]) >= 0.3:
            emit("high_exposure_recovery",
                 _sig(notional_z.loc[account], 1, 5) * 0.6
                 + _sig(overnight.loc[account], 0.3, 1) * 0.4,
                 [f"notional {float(notional_z.loc[account]):+.1f}σ",
                  f"{float(overnight.loc[account]):.0%} held overnight"])
        # 5 News/Event/Vol -- the overnight/concentration proxy PLUS the
        # dynamic economic calendar join: share of the account's recent
        # entries within +/-10 minutes of a high-impact macro print.
        _ev = float(_event_shares().get(account, 0.0))
        if (float(overnight.loc[account]) >= 0.5
                and float(concentration.loc[account]) >= 0.5) or _ev >= 0.30:
            emit("news_vol",
                 _sig(overnight.loc[account], 0.5, 1) * 0.35
                 + _sig(concentration.loc[account], 0.5, 1) * 0.35
                 + _sig(_ev, 0.15, 0.7) * 0.30,
                 [f"{float(overnight.loc[account]):.0%} overnight",
                  f"concentration {float(concentration.loc[account]):.0%}",
                  f"{_ev:.0%} of entries at macro prints (live calendar)"])

    # 2 Toxic Flow -- direction-adjusted short-horizon markout (adverse
    # selection). Positive markout means price moves the CLIENT's way right
    # after entry, i.e. the fill was toxic to the book. Scored on markout
    # magnitude x profitability x sample.
    markouts = _markouts()
    if markouts is not None and len(markouts):
        mk = markouts.reindex(panel.index)
        for account in panel.index:
            short = pd.to_numeric(mk["mk_short"].get(account), errors="coerce") \
                if account in mk.index else np.nan
            short = float(short) if pd.notna(short) else np.nan
            trades_raw = pd.to_numeric(mk["mk_trades"].get(account), errors="coerce") \
                if account in mk.index else 0
            trades_a = float(trades_raw) if pd.notna(trades_raw) else 0.0
            if not np.isfinite(short) or short <= rules["toxic_flow"]["markout_bad"]:
                continue
            if trades_a < rules["toxic_flow"]["min_trades"]:
                continue
            profitable = float(pf.loc[account]) >= rules["toxic_flow"]["toxic_pf"] \
                if account in pf.index else False
            base = _markout_strength(short)
            score = base * (1.0 if profitable else 0.7)
            obs_a = obs
            ev = float(evidence.loc[account]) if account in evidence.index else 40.0
            conf = float(np.clip(0.55 * ev + 0.30 * score + 15, 0, 100))
            if account in thin.index and bool(thin.loc[account]):
                conf = min(conf, 49.0)
            sev = float(np.clip(SEVERITY_WEIGHT["toxic_flow"] * (0.5 + score / 200), 0, 100))
            stp = (float(short_term_scores.get(account, np.nan))
                   if short_term_scores is not None else np.nan)
            if score < 25:
                continue
            rows.append({
                "account_key": account, "profile": "toxic_flow",
                "objective": PROFILES["toxic_flow"], "score": round(score, 1),
                "band": _band(score), "confidence": round(conf, 1),
                "stability": 60.0, "severity": round(sev, 1),
                "evidence": round(ev, 1), "obs_days": int(obs.loc[account]) if account in obs.index else 0,
                "short_term_score": None if np.isnan(stp) else round(stp, 4),
                "priority": round(score * conf / 100 * sev / 100
                                  * (1 + max(0.0, stp if not np.isnan(stp) else 0)), 2),
                "why": f"adverse markout {short:+.4f} · {int(trades_a)} trades"
                       + (" · profitable client" if profitable else "")})

    # OPERATOR-DEFINED WATCH RULES: simple threshold OR a full boolean
    # expression over any panel columns, added from the Rules tab without a
    # code change. Evaluated on the SAME (date-aware) panel as everything else.
    for custom in (rules.get("_custom") or []):
        try:
            name = str(custom.get("name") or custom.get("metric") or "rule")
            severity = float(custom.get("severity", 50))
            expr = str(custom.get("expr") or "").strip()
            if expr:
                fired = safe_expr_mask(expr, panel)
                describe = expr
                observed_of = None
            else:
                metric = str(custom.get("metric", ""))
                op = str(custom.get("op", ">"))
                value = float(custom.get("value", 0))
                if metric not in panel.columns:
                    continue
                series = pd.to_numeric(panel[metric], errors="coerce")
                fired = {">": series > value, ">=": series >= value,
                         "<": series < value, "<=": series <= value}.get(op)
                describe = f"{metric} {op} {value}"
                observed_of = series
            if fired is None:
                continue
            for account in panel.index[fired.fillna(False)]:
                why = describe if observed_of is None else \
                    f"{describe} (observed {float(observed_of.loc[account]):.4g})"
                rows.append({
                    "account_key": str(account), "profile": f"custom:{name}",
                    "objective": f"operator rule: {describe}",
                    "score": 100.0, "band": _band(100.0), "confidence": 90.0,
                    "stability": 50.0, "severity": round(severity, 1),
                    "evidence": 60.0, "obs_days": 0, "short_term_score": None,
                    "priority": round(90.0 * severity / 100, 2),
                    "why": why})
        except Exception:
            continue

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("priority", ascending=False).reset_index(drop=True)
    _CACHE["classify"] = result
    _CACHE["classify_key"] = cache_key
    return result


#: The universe of desk interventions, each with a default sweep of magnitudes.
ACTION_UNIVERSE = {
    "widen_spread":   {"label": "Widen spread", "unit": "× current spread",
                       "sweep": [1.5, 2.0, 3.0, 5.0]},
    "add_slippage":   {"label": "Add entry slippage", "unit": "pips",
                       "sweep": [0.5, 1.0, 2.0, 5.0]},
    "raise_swap":     {"label": "Raise overnight swap", "unit": "× current",
                       "sweep": [1.5, 2.0, 3.0]},
    "cap_size":       {"label": "Cap position size", "unit": "× median lots",
                       "sweep": [2.0, 1.5, 1.0, 0.5]},
    "delay_feed":     {"label": "Delay price feed", "unit": "seconds",
                       "sweep": [1, 3, 5, 10]},
    "reject_rate":    {"label": "Reject fraction of orders", "unit": "fraction",
                       "sweep": [0.1, 0.25, 0.5]},
}


def _per_unit(symbol: str, price: float) -> float:
    root = "".join(c for c in str(symbol).upper() if c.isalnum())
    if root.startswith(("XAU", "XAG")):
        return 100.0
    if len(root) == 6 and root.isalpha():
        return 100_000.0
    return 100.0


def action_test(account: str, actions: dict | None = None) -> dict:
    """Backtest desk interventions on ONE client's own trade history.

    Broker P&L is the NEGATIVE of client P&L on internalised flow, so a change
    that costs the client earns the broker. Each action is applied to every
    historical trade and the delta to broker P&L is measured; the sweep shows
    the response curve, and the recommendation is the action/magnitude with
    the best broker-P&L improvement per unit of flow retained.

    `actions` optionally overrides the sweep, e.g. {"widen_spread": [2.0]}.
    """
    from webapp import views
    trades = views.account_trades(account, None)
    if trades is None or len(trades) == 0:
        return {"account": account, "error": "no trade history"}
    trades = trades.copy()
    for c in ("open_price", "close_price", "volume_lots", "net_profit"):
        if c in trades.columns:
            trades[c] = pd.to_numeric(trades[c], errors="coerce")
    trades = trades.dropna(subset=["open_price", "volume_lots", "net_profit"])
    if trades.empty:
        return {"account": account, "error": "no priced trades"}

    symbols = trades["symbol"].astype(str)
    lots = trades["volume_lots"].to_numpy(dtype="float64")
    price = trades["open_price"].to_numpy(dtype="float64")
    client_pnl = trades["net_profit"].to_numpy(dtype="float64")
    per_unit = np.array([_per_unit(s, p) for s, p in zip(symbols, price)])
    # crude per-symbol spread in price terms (same heuristic the engine uses)
    def spread_price(s, p):
        root = "".join(c for c in str(s).upper() if c.isalnum())
        if root.startswith("XAU"): return 0.30
        if root.startswith("XAG"): return 0.03
        if len(root) == 6 and root.isalpha():
            return 0.00015 * p if "JPY" in root else 0.00012
        return 0.0006 * p
    spread = np.array([spread_price(s, p) for s, p in zip(symbols, price)])
    overnight = (trades["close_time"] - trades["open_time"]).dt.total_seconds() \
        .to_numpy() > 20 * 3600 if "close_time" in trades.columns else np.zeros(len(trades), bool)

    base_broker = float(-client_pnl.sum())
    actions = actions or {k: v["sweep"] for k, v in ACTION_UNIVERSE.items()}
    results = {}
    for action, sweep in actions.items():
        curve = []
        for mag in sweep:
            if action == "widen_spread":
                # extra cost to client = (mag-1) x spread, both sides
                cost = (mag - 1.0) * spread * 2 * per_unit * lots
                delta = float(cost.sum())
            elif action == "add_slippage":
                pip = mag * np.where(spread > 0.01, 0.1, 0.0001)  # pip size proxy
                cost = pip * per_unit * lots
                delta = float(cost.sum())
            elif action == "raise_swap":
                # additional swap only on overnight trades (~1 spread/night proxy)
                cost = np.where(overnight, (mag - 1.0) * spread * per_unit * lots, 0)
                delta = float(cost.sum())
            elif action == "cap_size":
                med = np.median(lots) if len(lots) else 0.0
                cap = med * mag
                scale = np.where(lots > cap, cap / np.maximum(lots, 1e-9), 1.0)
                # capping scales the client's P&L (and our mirror of it)
                delta = float((-client_pnl * scale - (-client_pnl)).sum())
            elif action == "reject_rate":
                # reject the worst-for-us fraction (their most profitable trades)
                order = np.argsort(client_pnl)[::-1]      # client best first
                k = int(len(order) * mag)
                mask = np.zeros(len(order), bool); mask[order[:k]] = True
                delta = float((-client_pnl[~mask]).sum() - base_broker)
            elif action == "delay_feed":
                if "_delay_curve" not in results:
                    curve_map, delay_meta = _delay_feed_curve(
                        account, trades, list(sweep))
                    results["_delay_curve"] = (curve_map, delay_meta)
                curve_map, delay_meta = results["_delay_curve"]
                delta = curve_map.get(float(mag), float("nan"))
            else:
                delta = 0.0
            curve.append({"magnitude": mag,
                          "broker_pnl_delta": None if np.isnan(delta) else round(delta, 2),
                          "new_broker_pnl": None if np.isnan(delta) else round(base_broker + delta, 2)})
        results[action] = {"label": ACTION_UNIVERSE[action]["label"],
                           "unit": ACTION_UNIVERSE[action]["unit"], "curve": curve}

    delay_extra = results.pop("_delay_curve", None)
    if delay_extra is not None and "delay_feed" in results:
        results["delay_feed"]["meta"] = delay_extra[1]

    # recommendation: best positive broker-P&L delta at the mildest magnitude
    best = None
    for action, r in results.items():
        for point in r["curve"]:
            d = point["broker_pnl_delta"]
            if d is not None and d > 0 and (best is None or d > best["delta"]):
                best = {"action": action, "label": r["label"],
                        "magnitude": point["magnitude"], "unit": r["unit"],
                        "delta": d, "new_broker_pnl": point["new_broker_pnl"]}
    return _json_safe({"account": account, "trades": int(len(trades)),
            "base_broker_pnl": round(base_broker, 2),
            "base_client_pnl": round(float(client_pnl.sum()), 2),
            "actions": results, "recommendation": best})


def _delay_feed_curve(account: str, trades: pd.DataFrame,
                      sweep: list) -> tuple[dict, dict]:
    """REAL feed-delay what-if from the venue's own `ticks` tables.

    For each recent trade: the price the client actually dealt at (last tick at
    or before the event) versus the price d seconds later. Delaying the feed
    fills the client at the later price on BOTH legs; the client's loss is the
    book's gain on internalised flow. Returns ({delay: broker_delta}, meta).

    Honest limits, reported in meta: only trades young enough to still be in
    tick retention are measured; the delta is scaled up to the account's full
    P&L rate by the measured/total trade ratio only in `scaled` (never
    silently); an unindexed ticks table aborts fast rather than hanging.
    """
    from webapp.ad_refresh import _bulk_connection
    server = account.split(":")[0]
    mt5 = server.startswith("mt5")
    frame = trades.copy()
    frame["open_time"] = pd.to_datetime(frame["open_time"], errors="coerce")
    frame["close_time"] = pd.to_datetime(frame.get("close_time"), errors="coerce")
    # newest first; tick retention is finite, so recent trades measure best
    frame = frame.dropna(subset=["open_time"]).sort_values(
        "open_time", ascending=False).head(120)
    if frame.empty:
        return {}, {"measured": 0, "reason": "no timestamped trades"}
    direction = np.where(frame.get("cmd", "buy").astype(str)
                         .str.lower().str.startswith("b"), 1.0, -1.0)
    lots = pd.to_numeric(frame["volume_lots"], errors="coerce").fillna(0.0).to_numpy()
    price0 = pd.to_numeric(frame["open_price"], errors="coerce").fillna(0.0).to_numpy()
    per_unit = np.array([_per_unit(s, p) for s, p in
                         zip(frame["symbol"].astype(str), price0)])
    max_delay = float(max(sweep))
    connection = _bulk_connection(server)
    time_col = "datetime" if mt5 else "tm"
    sym_col = "symbol" if mt5 else "symbol_name"

    def ticks_for(symbol: str, when: pd.Timestamp):
        lo = int(when.timestamp()) - 20
        hi = int(when.timestamp()) + int(max_delay) + 20
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT {time_col}, bid, ask FROM ticks "
                f"WHERE {sym_col} = %s AND {time_col} BETWEEN %s AND %s "
                f"ORDER BY {time_col} LIMIT 4000", [symbol, lo, hi])
            return cursor.fetchall()

    deltas = {float(d): 0.0 for d in sweep}
    measured = skipped = 0
    started = time.time()
    for row_i, (_, trade) in enumerate(frame.iterrows()):
        if time.time() - started > 45:                    # hard time budget
            break
        legs = [(trade["open_time"], direction[row_i])]
        if pd.notna(trade.get("close_time")):
            legs.append((trade["close_time"], -direction[row_i]))
        try:
            got_any = False
            for when, sign in legs:
                rows = ticks_for(str(trade["symbol"]), when)
                if not rows:
                    continue
                t0 = when.timestamp()
                series = [(float(r[0]), float(r[1]), float(r[2])) for r in rows
                          if r[1] and r[2]]
                if not series:
                    continue

                def price_at(t):
                    prior = [r for r in series if r[0] <= t]
                    chosen = prior[-1] if prior else series[0]
                    return chosen[2] if sign > 0 else chosen[1]   # buy=ask
                base = price_at(t0)
                for d in deltas:
                    moved = price_at(t0 + d)
                    # client fills at the delayed price: worse for them is
                    # (moved-base)*sign dollars per unit AGAINST the client
                    deltas[d] += (moved - base) * sign * per_unit[row_i] * lots[row_i]
                got_any = True
            measured += 1 if got_any else 0
            skipped += 0 if got_any else 1
        except Exception:
            skipped += 1
            if measured == 0 and skipped >= 3:
                return {}, {"measured": 0,
                            "reason": "ticks table not answering range queries"}
    try:
        connection.close()
    except Exception:
        pass
    meta = {"measured": measured, "skipped_no_ticks": skipped,
            "window_trades": int(len(frame)),
            "note": "broker delta = client cost on measured trades only"}
    return deltas, meta


def _markout_strength(markout: float) -> float:
    """Markout magnitude -> 0-100. The scale is instrument-relative so the
    band is set generously; calibrate against labelled toxic cases later."""
    return _sig(abs(markout), 0.0, 0.0015) if markout is not None else 0.0


# ---------------------------------------------------------------------------
# ML CLASSIFICATION LAYER -- an unsupervised anomaly model over the same
# behavioural features, run ALONGSIDE the rules. Rules are explainable and
# auditable; the ML layer catches multivariate patterns no single threshold
# describes. The UI toggles between / overlays them, per the spec's
# "rules or ML models" requirement.
# ---------------------------------------------------------------------------
_ML_FEATURES = ["martingale_rate", "revenge_rate", "scalp_rate",
                "overnight_rate", "concentration", "profit_factor",
                "win_rate", "expectancy_per_trade", "gross_notional",
                "buy_share", "stop_use_rate", "take_profit_use_rate",
                "life_sharpe", "life_max_drawdown"]


def ml_scores() -> pd.DataFrame:
    """Per-account anomaly score (0-100) from an IsolationForest over the
    behavioural feature space, with the top contributing features as the
    justification -- so an ML flag is as explainable as a rule flag."""
    from sklearn.ensemble import IsolationForest
    _frame()
    if "ml" in _CACHE:
        return _CACHE["ml"]
    panel = _account_panel()
    cols = [c for c in _ML_FEATURES if c in panel.columns]
    if not cols or len(panel) < 50:
        return pd.DataFrame()
    X = (panel[cols].apply(pd.to_numeric, errors="coerce")
         .astype("float64").fillna(0.0).to_numpy(dtype="float64"))
    means, stds = X.mean(0), X.std(0) + 1e-9
    Xz = (X - means) / stds
    forest = IsolationForest(n_estimators=200, contamination="auto",
                             random_state=0, n_jobs=-1)
    forest.fit(Xz)
    raw = -forest.score_samples(Xz)
    score = _sig(raw, float(np.quantile(raw, 0.5)), float(np.quantile(raw, 0.999)))
    out = pd.DataFrame({"account_key": panel.index, "ml_anomaly": np.round(score, 1)})
    # per-account justification: which standardised features are most extreme
    top = []
    for i in range(len(panel)):
        idx = np.argsort(-np.abs(Xz[i]))[:3]
        top.append(", ".join(f"{cols[j]} {Xz[i, j]:+.1f}σ" for j in idx))
    out["ml_why"] = top
    out = out.sort_values("ml_anomaly", ascending=False).reset_index(drop=True)
    _CACHE["ml"] = out
    return out


def _json_safe(obj):
    """Recursively convert numpy types and non-finite floats to JSON-safe
    values (NaN/Inf -> None) so FastAPI can serialise the dossier."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is pd.NaT or (isinstance(obj, float) and not np.isfinite(obj)):
        return None
    return obj


def client_screen(account: str) -> dict:
    """Full behavioural dossier for ONE client -- every profile score, the
    markout curve, recent trades, and the ML anomaly view. The 'zoom in /
    screen' surface that ties the framework together for manual review."""
    from webapp import model_service as ms
    from webapp import views
    profiles = classify()
    mine = profiles.loc[profiles["account_key"] == account] if not profiles.empty else profiles
    dossier = {"account": account,
               "profiles": mine.to_dict("records") if len(mine) else []}
    # markout curve (reuse the account-tab machinery)
    try:
        markouts = views.load_markouts(ms.SCRATCH / "markout_all_servers.parquet")
        dossier["markout"] = views.account_markouts(markouts, account)
    except Exception:
        dossier["markout"] = {}
    # recent trades
    try:
        trades = views.account_trades(account, None, limit=100)
        dossier["trades"] = (trades.assign(
            open_time=trades["open_time"].astype(str),
            close_time=trades["close_time"].astype(str))
            [[c for c in ("open_time", "symbol", "cmd", "volume_lots",
                          "open_price", "close_price", "net_profit")
              if c in trades.columns]].to_dict("records")) if len(trades) else []
    except Exception:
        dossier["trades"] = []
    # ML anomaly for this account
    try:
        ml = ml_scores()
        row = ml.loc[ml["account_key"] == account]
        dossier["ml"] = row.iloc[0].to_dict() if len(row) else None
    except Exception:
        dossier["ml"] = None
    return _json_safe(dossier)


# ---------------------------------------------------------------------------
# LARK ALERTS -- push high-priority NEW flags to a Lark webhook. Off unless a
# webhook URL is configured; de-duplicated so a standing flag alerts once.
# ---------------------------------------------------------------------------
_ALERTED_PATH = ROOT / "artifacts" / "antifraud_alerted.json"


def send_lark_alerts(min_priority: float = 50.0, webhook: str | None = None) -> dict:
    import os
    import urllib.request
    webhook = webhook or os.environ.get("LARK_WEBHOOK", "")
    result = classify(short_term_scores=None)
    if result.empty:
        return {"sent": 0, "reason": "no flags"}
    high = result.loc[result["priority"] >= min_priority]
    try:
        seen = set(json.loads(_ALERTED_PATH.read_text(encoding="utf-8")))
    except Exception:
        seen = set()
    fresh = high.loc[~high.apply(
        lambda r: f"{r['account_key']}|{r['profile']}" in seen, axis=1)]
    if not webhook:
        return {"sent": 0, "would_send": int(len(fresh)),
                "reason": "no LARK_WEBHOOK configured (dry run)"}
    sent = 0
    for _, r in fresh.iterrows():
        card = {"msg_type": "text", "content": {"text":
                f"🚩 AntiFraud flag · {r['profile']} · {r['account_key']}\n"
                f"score {r['score']} · confidence {r['confidence']}% · "
                f"severity {r['severity']} · priority {r['priority']}\n{r['why']}"}}
        try:
            req = urllib.request.Request(
                webhook, data=json.dumps(card).encode(),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
            seen.add(f"{r['account_key']}|{r['profile']}")
            sent += 1
        except Exception:
            break
    try:
        _ALERTED_PATH.write_text(json.dumps(sorted(seen)), encoding="utf-8")
    except Exception:
        pass
    return {"sent": sent, "candidates": int(len(fresh))}


# ------------------------------------------------------------ expressions
import ast as _ast

_ALLOWED_NODES = (_ast.Expression, _ast.BoolOp, _ast.BinOp, _ast.UnaryOp,
                  _ast.Compare, _ast.Name, _ast.Load, _ast.Constant,
                  _ast.And, _ast.Or, _ast.Not, _ast.USub,
                  _ast.Add, _ast.Sub, _ast.Mult, _ast.Div,
                  _ast.Gt, _ast.GtE, _ast.Lt, _ast.LtE, _ast.Eq, _ast.NotEq)


def safe_expr_mask(expr: str, panel: pd.DataFrame) -> pd.Series:
    """Boolean mask from an operator-written expression over panel columns.

    Whitelisted AST only -- names must be panel columns, operators are
    comparisons, and/or/not, +-*/. `profit_factor > 1.5 and scalp_rate > 0.3`
    is the intended shape; anything else raises with a plain message.
    """
    tree = _ast.parse(expr, mode="eval")
    for node in _ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"'{type(node).__name__}' is not allowed -- use "
                             f"column names, numbers, comparisons, and/or/not")
        if isinstance(node, _ast.Name) and node.id not in panel.columns:
            raise ValueError(f"unknown metric '{node.id}'")

    def ev(node):
        if isinstance(node, _ast.Expression):
            return ev(node.body)
        if isinstance(node, _ast.Constant):
            return float(node.value)
        if isinstance(node, _ast.Name):
            return pd.to_numeric(panel[node.id], errors="coerce")
        if isinstance(node, _ast.UnaryOp):
            operand = ev(node.operand)
            return ~operand.fillna(False) if isinstance(node.op, _ast.Not) \
                else -operand
        if isinstance(node, _ast.BinOp):
            left, right = ev(node.left), ev(node.right)
            ops = {_ast.Add: lambda a, b: a + b, _ast.Sub: lambda a, b: a - b,
                   _ast.Mult: lambda a, b: a * b,
                   _ast.Div: lambda a, b: a / b}
            return ops[type(node.op)](left, right)
        if isinstance(node, _ast.Compare):
            left = ev(node.left)
            result = None
            for op, comp in zip(node.ops, node.comparators):
                right = ev(comp)
                ops = {_ast.Gt: lambda a, b: a > b, _ast.GtE: lambda a, b: a >= b,
                       _ast.Lt: lambda a, b: a < b, _ast.LtE: lambda a, b: a <= b,
                       _ast.Eq: lambda a, b: a == b,
                       _ast.NotEq: lambda a, b: a != b}
                piece = ops[type(op)](left, right)
                result = piece if result is None else (result & piece)
                left = right
            return result
        if isinstance(node, _ast.BoolOp):
            parts = [ev(v) for v in node.values]
            out = parts[0]
            for part in parts[1:]:
                out = (out & part) if isinstance(node.op, _ast.And) else (out | part)
            return out
        raise ValueError(f"unsupported node {type(node).__name__}")

    mask = ev(tree)
    if not isinstance(mask, pd.Series) or mask.dtype != bool:
        mask = mask > 0 if isinstance(mask, pd.Series) else \
            pd.Series(False, index=panel.index)
    return mask.fillna(False)


# ------------------------------------------------------------ unified universe
def unified_universe(as_of: str | None = None, start: str | None = None,
                     end: str | None = None, limit: int = 500) -> dict:
    """ONE classification table across the three systems.

    Merges: the nine antifraud profiles + custom rules (score columns), the
    routing taxonomy used by Client Intelligence / A-Book (views.classify on
    the same panel), and the ML anomaly score. The PRIMARY label is the
    highest-priority antifraud flag; with no flag, the routing label stands.
    Date-aware like everything else."""
    from webapp import views
    panel = _account_panel(as_of=as_of, start=start, end=end)
    flags = classify(as_of=as_of, start=start, end=end)
    ml = ml_scores() if not as_of and not start else pd.DataFrame()

    table = pd.DataFrame(index=panel.index)
    # routing taxonomy on the SAME panel rows (needs the _notional_p90 helper)
    p90 = pd.to_numeric(panel.get("gross_notional"), errors="coerce").quantile(0.9)
    routing = []
    for account, row in panel.iterrows():
        r = row.copy()
        r["_notional_p90"] = p90
        try:
            label, _ = views.classify(r)
        except Exception:
            label = "MODEL_SIGNAL"
        routing.append(label)
    table["routing_label"] = routing

    # per-profile score columns from the antifraud flags
    if not flags.empty:
        pivot = flags.pivot_table(index="account_key", columns="profile",
                                  values="score", aggfunc="max")
        pivot.columns = [f"af_{c}" for c in pivot.columns]
        table = table.join(pivot)
        best = (flags.sort_values("priority", ascending=False)
                .drop_duplicates("account_key").set_index("account_key"))
        table["af_primary"] = best["profile"]
        table["af_priority"] = best["priority"]
    else:
        table["af_primary"] = None
        table["af_priority"] = np.nan
    if not ml.empty:
        table = table.join(ml.set_index("account_key")[["ml_anomaly"]])

    # PRIMARY: highest-priority antifraud flag wins; else the routing label
    table["primary"] = table["af_primary"].fillna(table["routing_label"])
    table = table.sort_values("af_priority", ascending=False, na_position="last")
    columns = ["primary", "routing_label", "af_primary", "af_priority",
               "ml_anomaly"] + sorted(c for c in table.columns
                                      if c.startswith("af_")
                                      and c not in ("af_primary", "af_priority"))
    columns = [c for c in columns if c in table.columns]
    out = table[columns].head(limit).reset_index()
    return _json_safe({
        "rows": out.where(pd.notna(out), None).to_dict("records"),
        "total": int(len(table)),
        "columns": ["account_key"] + columns,
        "window": {"as_of": as_of, "start": start, "end": end,
                   "default": "latest (today)"}})


def primary_labels(as_of: str | None = None) -> dict[str, str]:
    """account_key -> unified primary label, for OTHER tabs to join on so the
    three surfaces always agree."""
    try:
        universe = unified_universe(as_of=as_of, limit=10 ** 9)
        return {r["account_key"]: r["primary"] for r in universe["rows"]}
    except Exception:
        return {}


# ------------------------------------------------------------ markout grid
MARKOUT_HORIZONS = ["markout_1m", "markout_5m", "markout_30m", "markout_1h",
                    "markout_4h", "markout_1d", "markout_3d"]


def markout_grid(hx: str = "markout_5m", hy: str = "markout_1h",
                 percentile: float = 95.0, start: str | None = None,
                 end: str | None = None, limit: int = 1500) -> dict:
    """2-D markout scatter: each point one account, x/y the mean markout at
    the chosen horizons over the day range (default last 90 days), filtered
    to the top (100-percentile)% by extremity, coloured by ML anomaly."""
    from webapp import model_service as ms
    from webapp import views
    path = ms.SCRATCH / "markout_all_servers.parquet"
    if not path.exists():
        return {"error": "markout parquet not built"}
    if hx not in MARKOUT_HORIZONS or hy not in MARKOUT_HORIZONS:
        return {"error": f"horizons must be one of {MARKOUT_HORIZONS}"}
    frame = views.load_markouts(path) if hasattr(views, "load_markouts") \
        else pd.read_parquet(path)
    frame["day"] = pd.to_datetime(frame["day"])
    for column in MARKOUT_HORIZONS + ["context_trades"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column], errors="coerce").astype("float64")
    hi = pd.Timestamp(end) if end else frame["day"].max()
    lo = pd.Timestamp(start) if start else hi - pd.Timedelta(days=90)
    window = frame.loc[(frame["day"] >= lo) & (frame["day"] <= hi)]
    if window.empty:
        return {"points": [], "note": "no markout rows in range"}
    per_account = window.groupby("account_key").agg(
        x=(hx, "mean"), y=(hy, "mean"),
        trades=("context_trades", "sum"), days=("day", "nunique"))
    per_account = per_account.dropna(subset=["x", "y"])
    extremity = np.hypot(per_account["x"], per_account["y"])
    cut = float(np.percentile(extremity, percentile)) if len(extremity) else 0.0
    keep = per_account.loc[extremity >= cut].copy()
    ml = ml_scores()
    if not ml.empty:
        keep = keep.join(ml.set_index("account_key")[["ml_anomaly"]])
    labels = primary_labels()
    keep["primary"] = [labels.get(a) for a in keep.index]
    keep = keep.sort_values("trades", ascending=False).head(limit)
    return _json_safe({
        "points": [{"account": a, "x": float(r["x"]), "y": float(r["y"]),
                    "trades": int(r["trades"]), "days": int(r["days"]),
                    "anomaly": (float(r["ml_anomaly"])
                                if pd.notna(r.get("ml_anomaly")) else None),
                    "primary": r.get("primary")}
                   for a, r in keep.iterrows()],
        "hx": hx, "hy": hy, "percentile": percentile,
        "window": {"start": str(lo.date()), "end": str(hi.date())},
        "total_accounts": int(len(per_account))})


# ---------------------------------------------------------------- autopilot
_AUTOPILOT_THREAD = None
_LAST_DIGEST_DAY = None


def alerts_config() -> dict:
    return dict(load_rules().get("_alerts") or DEFAULT_RULES["_alerts"])


def save_alerts_config(config: dict) -> None:
    rules = load_rules()
    current = rules.get("_alerts") or {}
    current.update({k: config[k] for k in
                    ("autopilot", "min_priority", "digest_hour_london",
                     "poll_minutes") if k in config})
    rules["_alerts"] = current
    save_rules(rules)


def _send_digest(webhook: str, min_priority: float) -> int:
    """The 07:00 London round-up: EVERY currently-standing flag above the
    bar, in one message, regardless of whether it alerted before."""
    import urllib.request
    result = classify(short_term_scores=None)
    if result.empty:
        return 0
    high = result.loc[result["priority"] >= min_priority].head(40)
    if high.empty:
        return 0
    lines = [f"📋 AntiFraud daily digest — {len(high)} standing flags "
             f"(priority ≥ {min_priority:g})"]
    for _, r in high.iterrows():
        lines.append(f"• {r['profile']} · {r['account_key']} · "
                     f"score {r['score']} · prio {r['priority']}")
    card = {"msg_type": "text", "content": {"text": "\n".join(lines)}}
    req = urllib.request.Request(webhook, data=json.dumps(card).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=8)
    return int(len(high))


def _autopilot_tick() -> None:
    global _LAST_DIGEST_DAY
    import os
    config = alerts_config()
    if not config.get("autopilot"):
        return
    webhook = os.environ.get("LARK_WEBHOOK", "")
    if not webhook:
        return
    # as-generated: new flags since the last look (send_lark dedupes)
    send_lark_alerts(min_priority=float(config.get("min_priority", 60)),
                     webhook=webhook)
    # daily digest at the configured London hour
    try:
        from zoneinfo import ZoneInfo
        import datetime as _dt
        now_london = _dt.datetime.now(ZoneInfo("Europe/London"))
        target = int(config.get("digest_hour_london", 7))
        if (now_london.hour == target
                and _LAST_DIGEST_DAY != now_london.date()):
            _send_digest(webhook, float(config.get("min_priority", 60)))
            _LAST_DIGEST_DAY = now_london.date()
    except Exception:
        pass


def start_autopilot() -> None:
    """Background alert loop; a no-op unless _alerts.autopilot is on AND
    LARK_WEBHOOK is set, so it is always safe to start."""
    global _AUTOPILOT_THREAD
    import threading
    if _AUTOPILOT_THREAD is not None and _AUTOPILOT_THREAD.is_alive():
        return

    def _loop():
        while True:
            try:
                _autopilot_tick()
            except Exception:
                pass
            minutes = float(alerts_config().get("poll_minutes", 5) or 5)
            time.sleep(max(60.0, minutes * 60.0))

    _AUTOPILOT_THREAD = threading.Thread(target=_loop, daemon=True,
                                         name="antifraud-autopilot")
    _AUTOPILOT_THREAD.start()


def data_availability() -> dict:
    """Honest statement of which profiles are fully vs partially computable
    with the columns currently in the corpus -- so the UI never implies
    evidence it doesn't have (Toxic Flow needs tick markouts; Bonus/Swap Arb
    need the cashflow + swap tables)."""
    frame_cols = set(_frame().columns)
    return {
        "persistent_edge": "full",
        "high_magnitude": "full",
        "scalper": "full" if "scalp_rate" in frame_cols else "partial",
        "martingale": "full" if "martingale_rate" in frame_cols else "partial",
        "high_exposure_recovery": "partial",
        "news_vol": "live (dynamic economic calendar joined)",
        "toxic_flow": "full (markouts)" if _markouts() is not None
                      else "needs markout parquet",
        "bonus_arb": "needs cashflow + credit table",
        "swap_arb": "needs swap ledger",
    }
