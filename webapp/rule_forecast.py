"""Early warning: predict who QUALIFIES for each rule in the next 5 days.

Hindsight flags tell you who already became a problem; the desk asked for
foresight. For each profile we train a separate model:

  label(account, day D) = does NOT qualify at D, but qualifies at some day in
                          (D, D+5]  ->  1, else 0
  features             = the account-day corpus row at D (the same behaviour
                          features everything else uses)

Qualification at a day is computed ROW-WISE from the live rules (the same
thresholds the Rules tab edits -- change a rule and the forecast retrains
against the new definition), on the corpus's own daily/expanding columns.
Profiles whose defining columns the corpus lacks are skipped and say so.
Custom expression rules are forecastable the same way.

Output: for every account NOT currently qualifying, P(qualifies within 5d),
per profile -- the watchlist before the fact. Cached per (rules, corpus).
"""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import pandas as pd

_CACHE: dict = {}
_LOCK = threading.Lock()
_TRAINING: dict = {"running": False, "started": 0.0}


def _disk_path():
    from webapp.trade_features import _AD_DIR
    return _AD_DIR / "forecast_cache.json"


def _current_key(rules: dict, horizon_days: int) -> str:
    from webapp.trade_features import _AD_DIR
    stamp = (_AD_DIR / "model_frame.parquet").stat().st_mtime
    return json.dumps({k: v for k, v in rules.items() if k != "_alerts"},
                      sort_keys=True) + f"|{stamp}|{horizon_days}"


def _load_disk() -> dict | None:
    try:
        return json.loads(_disk_path().read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_disk(key: str, result: dict) -> None:
    try:
        _disk_path().write_text(json.dumps({"key": key, "result": result}),
                                encoding="utf-8")
    except Exception:
        pass

#: Row-wise qualification masks per profile: (needed columns, mask builder).
#: These mirror the panel rules on the DAILY grain -- one row = one account-day.
def _masks(rules: dict) -> dict:
    def get(profile, field, default):
        return float((rules.get(profile) or {}).get(field, default))

    return {
        "persistent_edge": (
            ["expanding_profit_factor", "life_closes"],
            lambda f: (pd.to_numeric(f["expanding_profit_factor"], errors="coerce")
                       > get("persistent_edge", "profit_factor", 1.1))
                      & (pd.to_numeric(f["life_closes"], errors="coerce")
                         >= get("persistent_edge", "min_trades", 20))),
        "scalper": (
            ["scalp_rate", "life_closes"],
            lambda f: (pd.to_numeric(f["scalp_rate"], errors="coerce")
                       >= get("scalper", "under_5m_share", 0.5))
                      & (pd.to_numeric(f["life_closes"], errors="coerce")
                         >= get("scalper", "min_trades", 30))),
        "martingale": (
            ["martingale_rate"],
            lambda f: pd.to_numeric(f["martingale_rate"], errors="coerce")
                      >= get("martingale", "escalation_rate", 0.15)),
        "high_magnitude": (
            ["gross_notional"],
            lambda f: pd.to_numeric(f["gross_notional"], errors="coerce")
                      >= pd.to_numeric(f["gross_notional"], errors="coerce")
                      .groupby(f["decision_day"]).transform(
                          lambda s: s.quantile(0.9))),
        # Same proxies the classify() step uses, on the daily grain, so these
        # profiles get an early-warning model too (they auto-skip if the base
        # rate is degenerate on the corpus).
        "news_vol": (
            ["overnight_rate", "concentration"],
            lambda f: (pd.to_numeric(f["overnight_rate"], errors="coerce") >= 0.5)
                      & (pd.to_numeric(f["concentration"], errors="coerce") >= 0.5)),
        "high_exposure_recovery": (
            ["rank_gross_notional", "overnight_rate"],
            lambda f: (pd.to_numeric(f["rank_gross_notional"], errors="coerce")
                       >= get("high_exposure_recovery", "exposure_rank", 0.9))
                      & (pd.to_numeric(f["overnight_rate"], errors="coerce") >= 0.3)),
    }


def _custom_masks(rules: dict) -> dict:
    from webapp import antifraud
    out = {}
    for custom in (rules.get("_custom") or []):
        name = str(custom.get("name") or "rule")
        expr = str(custom.get("expr") or "").strip()
        metric = str(custom.get("metric") or "")
        op = str(custom.get("op") or ">")
        value = float(custom.get("value") or 0)

        def make(expr=expr, metric=metric, op=op, value=value):
            def mask(frame):
                if expr:
                    return antifraud.safe_expr_mask(expr, frame)
                series = pd.to_numeric(frame[metric], errors="coerce")
                return {">": series > value, ">=": series >= value,
                        "<": series < value, "<=": series <= value}[op]
            return mask
        needed = [metric] if metric and not expr else []
        out[f"custom:{name}"] = (needed, make())
    return out


def forecast(horizon_days: int = 5, max_rows: int = 400000) -> dict:
    """Non-blocking entry: return the pre-trained models' watchlists instantly
    (memory -> disk), and if the corpus/rules changed, retrain in the BACKGROUND
    while still serving the last result. Never blocks the page on a 7-minute
    train -- that was the 'not working' symptom after every restart."""
    from webapp import antifraud
    rules = antifraud.load_rules()
    try:
        key = _current_key(rules, horizon_days)
    except Exception:
        key = None
    if key and _CACHE.get("key") == key:
        return _CACHE["result"]
    disk = _load_disk()
    if key and disk and disk.get("key") == key:
        _CACHE.update(key=key, result=disk["result"])
        return disk["result"]
    # Not fresh: kick a background retrain and serve the last good result (or an
    # empty-but-training placeholder) so the tab shows something immediately.
    _ensure_training(horizon_days, max_rows)
    base = dict((disk or {}).get("result") or
                {"horizon_days": horizon_days, "profiles": {}, "skipped": {},
                 "trained_minutes": 0})
    base["training"] = True
    base["note"] = ("early-warning models are training in the background "
                    "(~5-7 min the first time); showing the last results -- "
                    "refresh shortly for the update")
    return base


def _ensure_training(horizon_days: int, max_rows: int) -> None:
    with _LOCK:
        if _TRAINING["running"]:
            return
        _TRAINING["running"] = True
        _TRAINING["started"] = time.time()

    def run():
        try:
            from webapp import antifraud
            result = _compute(horizon_days, max_rows)
            key = _current_key(antifraud.load_rules(), horizon_days)
            _CACHE.update(key=key, result=result)
            _save_disk(key, result)
        except Exception:
            pass
        finally:
            _TRAINING["running"] = False

    threading.Thread(target=run, daemon=True).start()


def ensure(horizon_days: int = 5) -> None:
    """Startup pre-train: adopt the disk cache if it matches the current corpus/
    rules, else train in the background so the tab is ready before it's opened."""
    from webapp import antifraud
    try:
        key = _current_key(antifraud.load_rules(), horizon_days)
    except Exception:
        key = None
    disk = _load_disk()
    if key and disk and disk.get("key") == key:
        _CACHE.update(key=key, result=disk["result"])
        return
    _ensure_training(horizon_days, 400000)


def _compute(horizon_days: int = 5, max_rows: int = 400000) -> dict:
    """Train per-profile become-a-member models and score today's non-members."""
    from webapp import antifraud
    import lightgbm as lgb

    rules = antifraud.load_rules()
    frame = antifraud._frame().copy()
    frame["decision_day"] = pd.to_datetime(frame["decision_day"])
    frame = frame.sort_values(["account_key", "decision_day"])
    numeric = [c for c in frame.columns
               if c not in ("account_key", "decision_day")
               and pd.api.types.is_numeric_dtype(frame[c])]

    all_masks = {**_masks(rules), **_custom_masks(rules)}
    results, skipped = {}, {}
    started = time.time()
    for profile, (needed, build) in all_masks.items():
        missing = [c for c in needed if c not in frame.columns]
        if missing:
            skipped[profile] = f"corpus lacks {missing}"
            continue
        try:
            qualifies = build(frame).fillna(False).to_numpy()
        except Exception as error:
            skipped[profile] = f"mask failed: {error}"
            continue
        member_rate = float(qualifies.mean())
        if member_rate < 0.0005 or member_rate > 0.6:
            skipped[profile] = f"degenerate base rate {member_rate:.2%}"
            continue
        # will-qualify-within-horizon: rolling forward max per account.
        # Reverse each account's series and take a trailing max window.
        flag = pd.Series(qualifies, index=frame.index)
        grouped = flag.groupby(frame["account_key"].to_numpy())
        future = (grouped.transform(
            lambda s: s[::-1].rolling(horizon_days, min_periods=1)
                             .max()[::-1].shift(-1)).fillna(0).astype(bool))
        label = (~pd.Series(qualifies)) & future            # not now, soon yes
        usable = ~pd.Series(qualifies)                       # train on non-members
        X_all = frame[numeric].to_numpy("float32")
        idx = np.flatnonzero(usable.to_numpy())
        if len(idx) > max_rows:
            idx = np.random.default_rng(0).choice(idx, max_rows, replace=False)
        y = label.to_numpy()[idx]
        if y.sum() < 200:
            skipped[profile] = f"only {int(y.sum())} conversion events"
            continue
        X = X_all[idx]
        np.putmask(X, ~np.isfinite(X), np.nan)
        model = lgb.LGBMClassifier(
            n_estimators=150, num_leaves=31, learning_rate=0.08,
            min_child_samples=50, n_jobs=-1, verbosity=-1)
        # last 20% of rows (time-ordered within sample) as a sanity holdout
        order = np.argsort(frame["decision_day"].to_numpy()[idx])
        split = int(len(order) * 0.8)
        model.fit(X[order[:split]], y[order[:split]])
        from sklearn.metrics import roc_auc_score
        try:
            auc = float(roc_auc_score(y[order[split:]],
                                      model.predict_proba(X[order[split:]])[:, 1]))
        except Exception:
            auc = None
        # score TODAY's non-members on their latest row
        latest_idx = (frame.groupby("account_key", observed=True)
                      .tail(1).index.to_numpy())
        latest_mask = ~pd.Series(qualifies)[latest_idx].to_numpy()
        candidates = latest_idx[latest_mask]
        Xc = X_all[candidates]
        np.putmask(Xc, ~np.isfinite(Xc), np.nan)
        probability = model.predict_proba(Xc)[:, 1]
        accounts = frame["account_key"].to_numpy()[candidates]
        top = np.argsort(-probability)[:100]
        importance = sorted(zip(numeric, model.feature_importances_),
                            key=lambda kv: -kv[1])[:5]
        results[profile] = {
            "holdout_auc": round(auc, 3) if auc is not None else None,
            "base_rate": round(float(y.mean()), 5),
            "conversions_trained_on": int(y.sum()),
            "drivers": [name for name, _ in importance],
            "watchlist": [{"account_key": str(accounts[i]),
                           "probability": round(float(probability[i]), 4)}
                          for i in top if probability[i] >= 0.05]}
    result = {"horizon_days": horizon_days,
              "profiles": results, "skipped": skipped,
              "trained_minutes": round((time.time() - started) / 60, 1),
              "note": "probability an account NOT currently qualifying joins "
                      "the rule within the horizon; retrains when rules or "
                      "corpus change"}
    return result
