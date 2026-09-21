"""Dynamic Anti-Fraud category registry.

Every category the system detects is a REGISTRY ENTRY: a key, a label, a
complex-rule definition (a safe expression over the account panel's fields),
an `active` flag (inactive categories disappear from every tab), a `use_ml`
flag (the category gets its own auto-generated tab and a dedicated model
trained on its rule as the target), and an ordering.

Persistence: code defaults -> admin defaults (saved by an admin) -> per-user
overrides (each user's own definitions, applied whenever they are logged in).
"Restore defaults" deletes the user's overrides, falling back to the admin's.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ADMIN_PATH = ROOT / "af_registry_admin.json"
USER_DIR = ROOT / "artifacts" / "af_registry_users"

DEFAULT_CATEGORIES = [
    {"key": "latency_arbitrage", "label": "Latency Arbitrage", "order": 1,
     "active": True, "use_ml": True,
     "definition": "med_hold_s <= 120 and fast_win_rate >= 0.55 and profit_per_min > 0",
     "description": "Dealing on stale quotes: tick-tape-proved entry markouts, "
                    "fast holds, extraction. ML target = tape-proved labels."},
    {"key": "news_vol", "label": "News/Event/Vol", "order": 2,
     "active": True, "use_ml": True,
     "definition": "event_trades >= 5 and event_pnl > 0",
     "description": "Profitable activity on the fixed event calendar (macro "
                    "prints + market-derived synchronized-volatility minutes). "
                    "ML target = the rule holding over the next 5 days with "
                    "client equity rising."},
    {"key": "persistent_edge", "label": "Persistent Edge", "order": 3,
     "active": True, "use_ml": False,
     "definition": "profit_factor >= 1.4 and win_rate >= 0.55 and n_trades >= 50",
     "description": "Sustained statistically meaningful profitability."},
    {"key": "toxic_flow", "label": "Toxic Flow", "order": 4,
     "active": True, "use_ml": False,
     "definition": "mk_short > 0 and mk_trades >= 30",
     "description": "Positive short-horizon markout: fills that are adverse "
                    "selection against the book."},
    {"key": "high_magnitude", "label": "High Magnitude", "order": 5,
     "active": True, "use_ml": False,
     "definition": "notional_z >= 2",
     "description": "Position sizes far above peer scale."},
    {"key": "scalper", "label": "Scalper", "order": 6,
     "active": True, "use_ml": False,
     "definition": "scalp_rate >= 0.6",
     "description": "Dominantly sub-2-minute round trips."},
    {"key": "bonus_arb", "label": "Bonus Arbitrage", "order": 7,
     "active": True, "use_ml": False,
     "definition": "bonus_extraction >= 0.9 and life_pnl > 0 "
                   "and life_days_active <= 20 and life_profit_factor >= 1.5",
     "description": "Deposit-bonus extraction patterns."},
    {"key": "swap_arb", "label": "Swap Arbitrage", "order": 8,
     "active": True, "use_ml": False,
     "definition": "swap_capture_rate >= 0.5",
     "description": "Positioning purely for swap capture."},
    {"key": "martingale", "label": "Martingale", "order": 9,
     "active": True, "use_ml": False,
     "definition": "martingale_rate >= 0.4",
     "description": "Loss-doubling size escalation."},
    {"key": "high_exposure_recovery", "label": "High Exposure / Recovery",
     "order": 10, "active": True, "use_ml": False,
     "definition": "notional_z >= 1 and overnight_share >= 0.3",
     "description": "Oversized positions held to recover drawdowns."},
]

#: Field documentation for the definitions editor -- what each identifier in
#: a rule expression means. Extended automatically with panel columns.
FIELD_DOCS = {
    "med_hold_s": "Median position hold time in seconds.",
    "fast_win_rate": "Win rate on trades held <= 120s.",
    "profit_per_min": "Net profit per minute of market exposure ($).",
    "event_share": "Share of entries within ±10 min of a high-impact "
                   "calendar event (fixed historical calendar).",
    "overnight_share": "Share of positions held overnight.",
    "concentration": "Symbol/time concentration of activity (0-1).",
    "profit_factor": "Gross wins / gross losses.",
    "win_rate": "Share of profitable closed trades.",
    "trades": "Closed trades in the observation window.",
    "mk_short": "Mean tape markout (bps) at the short horizon over the "
                "account's tick-covered trades (latency scan); positive = "
                "price moves the client's way (toxic to the book).",
    "mk_trades": "Tick-covered trades in the latency scan window.",
    "notional_z": "Peer z-score of position scale (log max daily notional "
                  "vs all accounts).",
    "scalp_rate": "Share of sub-2-minute round trips.",
    "bonus_extraction": "Withdrawn / deposited money over the window (0-1); "
                        "1 = every deposited dollar pulled back out — the "
                        "deposit-bonus extraction shape.",
    "swap_capture_rate": "Collected positive swap / (positive swap + gross "
                         "trading wins): share of the win column that is "
                         "swap capture.",
    "martingale_rate": "Rate of 1-2-4-8 size-doubling sequences.",
    "stale_share": "Share of fast profitable entries on quotes >= 2s old.",
    "illiquid_share": "Share of flagged entries at >= 1.5x median spread.",
    "consistency": "Flagged trades / total trades.",
    "persistence": "Active flagged days / calendar span.",
    "confidence": "Statistical confidence the pattern is not luck (0-1).",
    "econ_usd": "Dollars extracted by flagged trades.",
    # trade-shape window features (computed over the 90-day observation window)
    "n_trades": "Closed trades in the 90-day observation window.",
    "fast60_share": "Share of trades held <= 60 seconds.",
    "fast120_share": "Share of trades held <= 120 seconds.",
    "p25_hold_s": "25th-percentile hold time in seconds.",
    "fast_profit_share": "Share of gross profit earned on <= 120s trades.",
    "top_hour_share": "Share of entries in the account's busiest hour.",
    "night_share": "Share of entries between 21:00 and 01:59 server time.",
    "fx_share": "Share of trades on FX pairs.",
    "metal_share": "Share of trades on XAU/XAG.",
    "crypto_share": "Share of trades on crypto symbols.",
    "mean_lots": "Average position size in lots.",
    "profit_std_ratio": "Mean trade P&L / its standard deviation "
                        "(consistency of results).",
    "event_trades": "Trades entered within ±10 min of a calendar event "
                    "(macro prints + synchronized-volatility minutes).",
    "event_pnl": "Net P&L ($) of trades entered within ±10 min of a "
                 "calendar event.",
    "obs_active_days": "Days with at least one trade in the 90-day window.",
    "avg_daily_pnl": "Realized client P&L per ACTIVE day over the 90-day "
                     "window ($). The client's win is the book's cost, so "
                     "this is the empirical expected $ cost per day while "
                     "the account is hitting a rule.",
}


def _read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load(user_id: str | None = None) -> dict:
    """Effective registry for this user: user overrides > admin > code."""
    base = {"categories": [dict(c) for c in DEFAULT_CATEGORIES],
            "source": "defaults"}
    admin = _read(ADMIN_PATH)
    if admin and admin.get("categories"):
        base = {"categories": admin["categories"], "source": "admin"}
    if user_id:
        user = _read(USER_DIR / f"{user_id}.json")
        if user and user.get("categories"):
            base = {"categories": user["categories"], "source": "user"}
    base["categories"] = sorted(base["categories"],
                                key=lambda c: c.get("order", 99))
    return base


def save(data: dict, user_id: str | None = None,
         as_admin: bool = False) -> dict:
    cats = data.get("categories") or []
    clean = []
    for c in cats:
        if not c.get("key") or not str(c.get("label", "")).strip():
            continue
        clean.append({
            "key": str(c["key"])[:40],
            "label": str(c["label"])[:60],
            "order": int(c.get("order", 99)),
            "active": bool(c.get("active", True)),
            "use_ml": bool(c.get("use_ml", False)),
            "definition": str(c.get("definition", ""))[:2000],
            "description": str(c.get("description", ""))[:500],
        })
    payload = {"categories": clean}
    if as_admin:
        ADMIN_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        USER_DIR.mkdir(parents=True, exist_ok=True)
        (USER_DIR / f"{user_id}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
    return load(user_id)


def restore(user_id: str) -> dict:
    try:
        (USER_DIR / f"{user_id}.json").unlink(missing_ok=True)
    except Exception:
        pass
    return load(user_id)


def field_universe() -> list[dict]:
    """Every identifier usable in a rule definition = every column of the
    per-client FEATURE TABLE the rules are evaluated over: the 174-column
    account-day corpus + trade-shape + event features. Definitions come from
    the curated docs, the glossary, or a humanised derivation of the name."""
    try:
        from webapp import glossary
        gloss = dict(getattr(glossary, "GLOSSARY", {}))
    except Exception:
        gloss = {}

    def _doc(name: str, source: str) -> str:
        if name in FIELD_DOCS:
            return FIELD_DOCS[name]
        for key in (name, name.replace("_", " ")):
            if key in gloss:
                return str(gloss[key])
        base, deriv = name, ""
        for tag, means in (("_lag1", "; value 1 day earlier"),
                           ("_lag2", "; value 2 days earlier"),
                           ("_lag3", "; value 3 days earlier"),
                           ("_lag4", "; value 4 days earlier"),
                           ("_lag5", "; value 5 days earlier"),
                           ("_trend", "; day-over-day slope"),
                           ("_roll7", "; 7-day rolling mean"),
                           ("_roll30", "; 30-day rolling mean"),
                           ("_delta1", "; 1-day change"),
                           ("_delta5", "; 5-day change"),
                           ("_z", "; z-score vs the peer population"),
                           ("_pct", "; percentile rank vs peers")):
            if name.endswith(tag):
                base, deriv = name[: -len(tag)], means
                break
        for pre, means in (("roll5_", "; 5-day rolling mean"),
                           ("roll20_", "; 20-day rolling mean"),
                           ("roll30_", "; 30-day rolling mean"),
                           ("expanding_", "; expanding (all-history) stat"),
                           ("lag_", "; prior-day value")):
            if base.startswith(pre) and not deriv:
                base, deriv = base[len(pre):], means
                break
        words = (base.replace("hist_", "historical ")
                 .replace("acct_", "account ").replace("ctx_", "context ")
                 .replace("_", " "))
        unit = ""
        for tag, means in (("_rate", " (rate 0-1)"), ("_share", " (share 0-1)"),
                           ("_usd", " ($)"), ("_bps", " (basis points)"),
                           ("_s", " (seconds)"), ("_pd", " (per day)")):
            if base.endswith(tag):
                unit = means
        return words.capitalize() + unit + deriv + f" — {source}."

    # AUTHORITATIVE composition -- exactly what the rule engine evaluates
    # over: trade-shape features + event features + every column of the
    # 174-feature account-day corpus (collision names carry _corpus).
    trade_shape = [
        "n_trades", "fast60_share", "fast120_share", "med_hold_s",
        "p25_hold_s", "fast_win_rate", "fast_profit_share", "profit_per_min",
        "win_rate", "profit_factor", "top_hour_share", "night_share",
        "fx_share", "metal_share", "crypto_share", "mean_lots",
        "profit_std_ratio", "swap_capture_rate", "overnight_share",
        "obs_active_days", "avg_daily_pnl",
        # derived registry fields computed onto the table
        "notional_z", "mk_short", "mk_trades", "bonus_extraction"]
    event = ["event_share", "event_trades", "event_pnl"]
    corpus: list[str] = []
    try:
        from webapp import rule_models
        base = list(rule_models._ad_latest().columns)
        overlap = set(trade_shape) | set(event)
        corpus = [(c + "_corpus") if c in overlap else c for c in base]
    except Exception:
        try:
            from webapp import antifraud
            frame = antifraud._frame()
            corpus = [c for c in frame.columns
                      if c not in ("account_key", "decision_day")]
        except Exception:
            corpus = []
    seen, out = set(), []
    sources = ([(c, "90-day trade-shape feature") for c in trade_shape]
               + [(c, "event-calendar feature") for c in event]
               + [(c, "account-day corpus feature")
                  for c in sorted(corpus)])
    for c, src in sources:
        if c not in seen:
            seen.add(c)
            doc = _doc(c[:-7] if c.endswith("_corpus") else c, src)
            if c.endswith("_corpus"):
                doc += " [account-day corpus copy of a colliding name]"
            out.append({"field": c, "doc": doc})
    return out
