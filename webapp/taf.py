"""TAF -- Toxic Account Forecasting.

The framework, in code:

  U = B u C                       every account-day is B-book or Toxic/Arbitrage
  C = union of measurable classes (the rule_forecast masks -- configurable)
  A = C n {account net-profitable}  -- the ABUSE: toxic behaviour that costs us
  cost(A) = net realised USD extracted by A accounts (cent/contract normalised;
            client gain = B-book loss)

This module computes the walk-forward cost accounting (by class + universe),
joins it to the per-class forecast models' accuracy (rule_forecast), and finds
candidate NEW abuse classes (profitable + predictable + unclassified). Results
are cached to disk keyed on the corpus + rules so the tab loads instantly.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

TARGET_ACC = 0.75          # SMART target: >=75% OOS detection
HORIZON = 5                # n days (configurable)
_CACHE: dict = {}
_COMPUTING: dict = {"running": False}


def _disk():
    from webapp.trade_features import _AD_DIR
    return _AD_DIR / "taf_overview.json"


def _key(rules: dict) -> str:
    from webapp.trade_features import _AD_DIR
    stamp = (_AD_DIR / "model_frame.parquet").stat().st_mtime
    return json.dumps({k: v for k, v in rules.items() if k != "_alerts"},
                      sort_keys=True) + f"|{stamp}|{HORIZON}"


def _pnl_col(frame) -> str:
    return next(c for c in ("realised_pnl", "day_pnl", "realized_pnl", "pnl")
               if c in frame.columns)


def _walkforward(frame) -> dict:
    """Honest out-of-sample walk-forward. At the END of each week W-1 we predict,
    from prior data only, who will be an ABUSER in week W (in C AND profitable
    that week). We then align to what ACTUALLY happened and score precision,
    recall and USD captured vs the perfect oracle -- the numbers the desk would
    have seen a week *ahead* of the money moving.

    Protocol (no test leakage):
      * split weeks into train / validation / test (time-ordered);
      * fit ONE model on the train weeks;
      * pick the operating threshold on the VALIDATION week to hit the SMART
        USD-capture target (never on test);
      * walk each TEST week forward at that fixed threshold, scoring OOS.

    `frame` must already carry decision_day, week, pnl, inC."""
    try:
        import lightgbm as lgb
    except Exception:
        return {"available": False, "note": "lightgbm unavailable"}

    num = [c for c in frame.columns
           if c not in ("account_key", "decision_day", "week", "pnl", "inC",
                        "is_abuse_day")
           and pd.api.types.is_numeric_dtype(frame[c])]
    if len(num) < 5:
        return {"available": False, "note": "too few features"}

    f = frame.sort_values(["account_key", "decision_day"])
    agg = {"week_pnl": ("pnl", "sum"), "inC": ("inC", "max")}
    agg.update({c: (c, "last") for c in num})
    wk = f.groupby(["account_key", "week"], observed=True).agg(**agg).reset_index()
    wk["abuse"] = ((wk["inC"] > 0) & (wk["week_pnl"] > 0)).astype(int)
    wk = wk.sort_values(["account_key", "week"])
    # features predicting week W = the account's state at the END of W-1
    wk[num] = wk.groupby("account_key", observed=True)[num].shift(1)
    wk = wk.dropna(subset=num, how="all")
    weeks = sorted(wk["week"].unique())
    if len(weeks) < 4:
        return {"available": False, "note": "too few weeks for walk-forward"}

    n_test = max(1, int(round(len(weeks) * 0.42)))
    n_val = 2 if len(weeks) - n_test >= 4 else 1    # pool 2 weeks to de-noise
    n_test = min(n_test, len(weeks) - n_val - 1)    # leave >=1 train week
    test_weeks = weeks[-n_test:]
    val_weeks = weeks[-n_test - n_val:-n_test]
    train_weeks = weeks[:-n_test - n_val]

    def _mat(sub):
        X = sub[num].to_numpy("float32", copy=True)
        np.putmask(X, ~np.isfinite(X), np.nan)
        return X

    tr = wk[wk["week"].isin(train_weeks)]
    # weight rows by the USD at stake: the framework optimises USD captured, not
    # head-count, so the model must care most about the accounts carrying the money
    sw = 1.0 + np.log1p(np.clip(tr["week_pnl"].to_numpy(), 0, None))
    model = lgb.LGBMClassifier(n_estimators=250, num_leaves=48, learning_rate=0.06,
                               min_child_samples=60, n_jobs=-1, verbosity=-1)
    model.fit(_mat(tr), tr["abuse"].to_numpy(), sample_weight=sw)

    def _score(sub, thr):
        p = model.predict_proba(_mat(sub))[:, 1]
        flag = p >= thr
        actual = sub["abuse"].to_numpy() == 1
        tp = flag & actual
        wp = sub["week_pnl"].to_numpy()
        cap = float(wp[tp].sum())
        up = float(wp[actual].sum())
        nflag, nact, ntp = int(flag.sum()), int(actual.sum()), int(tp.sum())
        return {"flagged": nflag, "actual": nact, "tp": ntp,
                "precision": round(ntp / max(nflag, 1), 3),
                "recall": round(ntp / max(nact, 1), 3),
                "captured_usd": round(cap, 0), "upper_usd": round(up, 0),
                "capture_pct": round(cap / max(up, 1), 3)}

    # ---- frontier + threshold calibration on the (pooled) validation weeks.
    # Operating point = the MOST PRECISE threshold whose validation USD-capture
    # still meets the SMART target -- honest (never sees test), and precision is
    # maximised subject to hitting the >=75% capture goal.
    val = wk[wk["week"].isin(val_weeks)]
    frontier = []
    thr_op = 0.30
    for thr in [round(0.15 + 0.05 * i, 2) for i in range(14)]:     # 0.15..0.80
        s = _score(val, thr)
        frontier.append({"threshold": thr, "flagged": s["flagged"],
                         "usd_capture_pct": s["capture_pct"],
                         "precision": s["precision"], "recall": s["recall"]})
    # validation capture is optimistic vs unseen weeks, so calibrate to a margin
    # above the goal -- the live operating point then robustly clears 75% OOS.
    cal_target = min(0.92, TARGET_ACC + 0.15)
    meeting = [r for r in frontier if r["usd_capture_pct"] >= cal_target]
    if meeting:
        thr_op = max(r["threshold"] for r in meeting)   # most precise that still clears
    else:                                               # target unreachable: max capture
        thr_op = min(r["threshold"] for r in frontier)

    weekly = []
    cum_cap = cum_up = tp_a = fl_a = ac_a = 0
    for w in test_weeks:
        s = _score(wk[wk["week"] == w], thr_op)
        s["week"] = w
        weekly.append(s)
        cum_cap += s["captured_usd"]; cum_up += s["upper_usd"]
        tp_a += s["tp"]; fl_a += s["flagged"]; ac_a += s["actual"]

    aggregate = {
        "usd_capture_pct": round(cum_cap / max(cum_up, 1), 3),
        "captured_usd": round(cum_cap, 0), "upper_usd": round(cum_up, 0),
        "precision": round(tp_a / max(fl_a, 1), 3),
        "recall": round(tp_a / max(ac_a, 1), 3),
        "flagged": int(fl_a), "actual": int(ac_a),
        "avg_lead_days": HORIZON}
    return {"available": True, "operating_threshold": thr_op,
            "target_acc": TARGET_ACC,
            "train_weeks": list(train_weeks), "val_weeks": list(val_weeks),
            "test_weeks": list(test_weeks), "n_features": len(num),
            "weekly": weekly, "aggregate": aggregate, "frontier": frontier}


def _compute() -> dict:
    """WINDOWED (per-week) accounting. Abuse is a STATE, evaluated per calendar
    week: an account is Abuse in a week if it is in C during the week AND makes
    money that week. The perfect-oracle UPPER BOUND is the sum of every
    profitable toxic week -- the money a flawless 'A-book them for that week'
    policy would capture. Our system recovers the fraction it flags in advance.
    Also emits the Monitor's weekly + daily series in the same pass."""
    from webapp import antifraud, rule_forecast
    rules = antifraud.load_rules()
    frame = antifraud._frame().copy()
    frame["decision_day"] = pd.to_datetime(frame["decision_day"])
    frame["pnl"] = pd.to_numeric(frame[_pnl_col(frame)], errors="coerce").fillna(0.0)
    n_days = int(frame["decision_day"].dt.normalize().nunique())
    iso = frame["decision_day"].dt.isocalendar()
    frame["week"] = (iso["year"].astype(str) + "-W"
                     + iso["week"].astype(str).str.zfill(2))
    n_weeks = int(frame["week"].nunique())

    masks = rule_forecast._masks(rules)
    inC = pd.Series(False, index=frame.index)
    class_day = {}
    for cls, (needed, build) in masks.items():
        if any(c not in frame.columns for c in needed):
            continue
        try:
            m = build(frame).fillna(False)
        except Exception:
            continue
        if m.mean() < 0.0005 or m.mean() > 0.6:
            continue
        class_day[cls] = m
        inC = inC | m
    frame["inC"] = inC.to_numpy()

    # per (account, week): total P&L and whether it was in C that week
    wk = (frame.groupby(["account_key", "week"])
          .agg(pnl=("pnl", "sum"), inC=("inC", "max")).reset_index())
    wk["abuse_week"] = wk["inC"] & (wk["pnl"] > 0)
    ab = wk[wk["abuse_week"]]
    upper = float(ab["pnl"].sum())                         # perfect-oracle cost
    A_accounts = int(ab["account_key"].nunique())          # ever an abuser
    C_accounts = int(wk.loc[wk["inC"], "account_key"].nunique())
    uni = int(frame["account_key"].nunique())
    total_win = float(wk.loc[wk["pnl"] > 0, "pnl"].sum())
    cost_day = upper / n_days

    # Two distinct, clearly-separated cost bases per class:
    #   WINDOWED (framework)  -- profitable-toxic-week P&L / days (the perfect
    #                            oracle basis; ties to the $/day headline).
    #   LIVE (day-level)      -- realised positive day-P&L of accounts matching
    #                            the class, for today / avg-day / this-week.
    # NOTE: classes OVERLAP (an account can match several), so per-class figures
    # sum to MORE than the union total on BOTH bases. The union totals below
    # count each account once.
    latest_day = frame["decision_day"].max()
    latest_week = frame.loc[frame["decision_day"] == latest_day, "week"].iloc[0]
    wk_key = wk.set_index(["account_key", "week"])["pnl"]
    pos = frame["pnl"] > 0
    today_sel = pos & (frame["decision_day"] == latest_day)
    week_sel = pos & (frame["week"] == latest_week)
    fc = rule_forecast.forecast(HORIZON)
    accs = {p: (d or {}).get("holdout_auc")
            for p, d in (fc.get("profiles") or {}).items()}
    by_class = []
    for cls, mser in class_day.items():
        marr = mser.to_numpy()
        pairs = frame.loc[marr, ["account_key", "week"]].drop_duplicates()
        wpnl = wk_key.reindex(list(map(tuple, pairs.to_numpy()))).fillna(0.0)
        ab_w = wpnl[wpnl > 0]                               # windowed abuse weeks
        td = frame.loc[today_sel & mser]                   # live: today
        by_class.append({
            "class": cls, "abuse_weeks": int(len(ab_w)),
            "A_accounts": int(pairs.loc[wpnl.to_numpy() > 0, "account_key"].nunique())
            if len(ab_w) else 0,
            "cost_total": round(float(ab_w.sum()), 0),
            "cost_per_day": round(float(ab_w.sum()) / n_days, 0),   # windowed
            "live_today": round(float(td["pnl"].sum()), 0),         # day-level
            "today_accounts": int(td["account_key"].nunique()),
            "live_avg_daily": round(float(frame.loc[pos & mser, "pnl"].sum()) / n_days, 0),
            "live_week": round(float(frame.loc[week_sel & mser, "pnl"].sum()), 0),
            "auc": accs.get(cls)})
    by_class.sort(key=lambda r: -r["cost_per_day"])

    # UNION totals (each account counted once) -- the honest denominators
    live_today_total = float(frame.loc[today_sel & frame["inC"], "pnl"].sum())
    live_today_accounts = int(frame.loc[today_sel & frame["inC"], "account_key"].nunique())
    live_avg_daily_total = float(frame.loc[pos & frame["inC"], "pnl"].sum()) / n_days
    live_week_total = float(frame.loc[week_sel & frame["inC"], "pnl"].sum())
    aucs = [a for a in accs.values() if a]
    mean_auc = float(np.mean(aucs)) if aucs else None

    # ---- Monitor series: weekly upper bound + at-target capture, daily breakdown
    weekly = (ab.groupby("week")
              .agg(abuse_usd=("pnl", "sum"), abuse_accounts=("account_key", "nunique"))
              .reset_index().sort_values("week"))
    weekly_rows = [{"week": r.week, "abuse_usd": round(float(r.abuse_usd), 0),
                    "abuse_accounts": int(r.abuse_accounts),
                    "capture_at_target": round(float(r.abuse_usd) * TARGET_ACC, 0)}
                   for r in weekly.itertuples()]
    # daily abuse USD = P&L on days that fall in an account's abuse-week
    abuse_pairs = set(map(tuple, ab[["account_key", "week"]].to_numpy()))
    frame["is_abuse_day"] = [(a, w) in abuse_pairs
                             for a, w in zip(frame["account_key"], frame["week"])]
    daily = (frame[frame["is_abuse_day"] & (frame["pnl"] > 0)]
             .groupby(frame["decision_day"].dt.strftime("%Y-%m-%d"))["pnl"]
             .sum().reset_index())
    daily.columns = ["day", "abuse_usd"]
    daily_rows = [{"day": r.day, "abuse_usd": round(float(r.abuse_usd), 0)}
                  for r in daily.itertuples()]

    # ---- genuine out-of-sample walk-forward (predicted vs actual, ahead of time)
    try:
        wf = _walkforward(frame)
    except Exception as exc:                       # never break the overview
        wf = {"available": False, "note": f"walk-forward error: {exc}"}

    return {
        "generated": time.time(), "n_days": n_days, "n_weeks": n_weeks,
        "horizon": HORIZON,
        "date_min": str(frame["decision_day"].min().date()),
        "date_max": str(frame["decision_day"].max().date()),
        "universe_accounts": uni, "C_accounts": C_accounts,
        "B_accounts": uni - C_accounts, "A_accounts": A_accounts,
        "C_share": round(C_accounts / uni, 3),
        "A_share_of_universe": round(A_accounts / uni, 3),
        "abuse_weeks": int(len(ab)),
        "total_cost": round(upper, 0), "cost_per_day": round(cost_day, 0),
        "cost_per_week": round(upper / n_weeks, 0),
        "cost_per_year": round(cost_day * 252, 0),
        "abuse_share_of_winnings": round(upper / max(total_win, 1), 3),
        "by_class": by_class,
        "mean_auc": None if mean_auc is None else round(mean_auc, 3),
        "target_acc": TARGET_ACC,
        "recoverable_per_day": round(cost_day * TARGET_ACC, 0),
        "recoverable_per_year": round(cost_day * TARGET_ACC * 252, 0),
        # LIVE day-level union totals (each account once) -- what the class
        # columns roll up to after removing cross-class overlap
        "cost_today": round(live_today_total, 0),
        "today_accounts": live_today_accounts,
        "live_avg_daily_total": round(live_avg_daily_total, 0),
        "live_week_total": round(live_week_total, 0),
        "latest_day": str(latest_day.date()), "latest_week": latest_week,
        "monitor": {"weekly": weekly_rows, "daily": daily_rows,
                    "walkforward": wf},
    }


def _bg_compute() -> None:
    import threading
    if _COMPUTING["running"]:
        return
    _COMPUTING["running"] = True

    def run():
        try:
            overview(force=True)
        finally:
            _COMPUTING["running"] = False
    threading.Thread(target=run, daemon=True).start()


def overview(force: bool = False) -> dict:
    """The framework's headline accounting, cached to disk. NON-BLOCKING: the
    endpoint path never runs the ~1-2 min scan itself -- it serves the cache or a
    'computing' placeholder while the background thread finishes."""
    from webapp import antifraud
    try:
        key = _key(antifraud.load_rules())
    except Exception:
        key = None
    if key and _CACHE.get("key") == key:
        return _CACHE["data"]
    try:
        disk = json.loads(_disk().read_text(encoding="utf-8"))
        if key and disk.get("key") == key:
            _CACHE.update(key=key, data=disk["data"])
            return disk["data"]
    except Exception:
        disk = None
    if not force:
        _bg_compute()
        base = dict((disk or {}).get("data") or {"by_class": []})
        base["computing"] = True
        base["note"] = "cost accounting is computing (~1-2 min); refresh shortly"
        return base
    data = _compute()
    _CACHE.update(key=key, data=data)
    try:
        _disk().write_text(json.dumps({"key": key, "data": data}), encoding="utf-8")
    except Exception:
        pass
    return data


def ensure() -> None:
    """Startup warm: adopt the disk cache or compute in the background."""
    from webapp import antifraud
    try:
        key = _key(antifraud.load_rules())
        disk = json.loads(_disk().read_text(encoding="utf-8"))
        if disk.get("key") == key:
            _CACHE.update(key=key, data=disk["data"])
            return
    except Exception:
        pass
    _bg_compute()


def monitor() -> dict:
    """Predicted-vs-actual tracking. Two layers:
      (1) BASELINE walk-forward from history -- weekly perfect-oracle abuse USD
          and the at-target capture, with a daily breakdown (populated now).
      (2) LIVE prediction log -- each day's watchlist is snapshotted (below) and,
          once the week elapses, scored against realised ground truth. This layer
          fills in as weeks pass; today it shows whatever has accrued."""
    ov = overview()
    mon = dict(ov.get("monitor") or {"weekly": [], "daily": []})
    mon["computing"] = ov.get("computing", False)
    mon["target_acc"] = ov.get("target_acc", TARGET_ACC)
    mon["cost_per_week"] = ov.get("cost_per_week")
    mon["cost_today"] = ov.get("cost_today")
    mon["today_accounts"] = ov.get("today_accounts")
    mon["live_avg_daily_total"] = ov.get("live_avg_daily_total")
    mon["live_week_total"] = ov.get("live_week_total")
    mon["latest_day"] = ov.get("latest_day")
    mon["mean_auc"] = ov.get("mean_auc")
    mon["by_class"] = ov.get("by_class", [])       # for live cost-by-class panel
    mon["live"] = _score_prediction_log()          # accrues as real weeks elapse
    return mon


def _log_path():
    from webapp.trade_features import _AD_DIR
    return _AD_DIR / "taf_predictions.json"


def log_predictions() -> dict:
    """Snapshot today's per-class watchlist so it can be scored later against
    ground truth -- the spine of the live monitor. Idempotent per day."""
    import datetime as _dt
    day = _dt.date.today().isoformat()
    try:
        store = json.loads(_log_path().read_text(encoding="utf-8"))
    except Exception:
        store = {}
    if day in store:
        return {"logged": day, "already": True}
    wl = watchlist()
    store[day] = {cls: [w["account_key"] for w in (d.get("watchlist") or [])
                        if (w.get("probability") or 0) >= 0.5]
                  for cls, d in (wl.get("classes") or {}).items()}
    # keep ~120 days
    for k in sorted(store)[:-120]:
        store.pop(k, None)
    try:
        _log_path().write_text(json.dumps(store), encoding="utf-8")
    except Exception:
        pass
    return {"logged": day, "classes": len(store[day])}


def _score_prediction_log() -> list:
    """Score logged predictions whose horizon has elapsed against the corpus's
    realised abuse. Returns per-snapshot precision/recall/USD captured."""
    try:
        store = json.loads(_log_path().read_text(encoding="utf-8"))
    except Exception:
        return []
    if not store:
        return []
    from webapp import antifraud, rule_forecast
    frame = antifraud._frame().copy()
    frame["decision_day"] = pd.to_datetime(frame["decision_day"])
    frame["pnl"] = pd.to_numeric(frame[_pnl_col(frame)], errors="coerce").fillna(0.0)
    masks = rule_forecast._masks(antifraud.load_rules())
    inC = pd.Series(False, index=frame.index)
    for cls, (needed, build) in masks.items():
        if any(c not in frame.columns for c in needed):
            continue
        try:
            inC = inC | build(frame).fillna(False)
        except Exception:
            continue
    frame["inC"] = inC.to_numpy()
    out = []
    import datetime as _dt
    for day, preds in sorted(store.items()):
        d0 = pd.Timestamp(day)
        end = d0 + pd.Timedelta(days=HORIZON)
        if end > frame["decision_day"].max():
            continue                              # horizon not elapsed yet
        win = frame[(frame["decision_day"] > d0) & (frame["decision_day"] <= end)]
        gp = win.groupby("account_key").agg(pnl=("pnl", "sum"), inC=("inC", "max"))
        actual = set(gp[(gp["inC"] > 0) & (gp["pnl"] > 0)].index)
        flagged = set().union(*[set(v) for v in preds.values()]) if preds else set()
        tp = flagged & actual
        captured = float(gp.loc[list(tp), "pnl"].sum()) if tp else 0.0
        upper = float(gp.loc[list(actual), "pnl"].sum()) if actual else 0.0
        out.append({"snapshot": day,
                    "flagged": len(flagged), "actual": len(actual),
                    "precision": round(len(tp) / max(len(flagged), 1), 3),
                    "recall": round(len(tp) / max(len(actual), 1), 3),
                    "captured_usd": round(captured, 0), "upper_usd": round(upper, 0),
                    "capture_pct": round(captured / max(upper, 1), 3)})
    return out[-14:]


def watchlist(horizon: int = HORIZON) -> dict:
    """Forward-looking: who becomes Abuse within n days, per class, + action."""
    from webapp import rule_forecast
    fc = rule_forecast.forecast(horizon)
    action = {                      # optimal response per predicted class
        "persistent_edge": "Route to A-book (hedge the edge)",
        "high_magnitude": "Route to A-book + cap size",
        "scalper": "Widen spread / add execution latency guard",
        "news_vol": "Restrict around events / widen spread",
        "martingale": "Margin + exposure limits",
        "high_exposure_recovery": "Reduce leverage / exposure cap",
        "toxic_flow": "Route to A-book (adverse selection)",
        "swap_arb": "Adjust swap / restrict carry",
        "bonus_arb": "Withhold credit / review",
    }
    profiles = fc.get("profiles") or {}
    out = {}
    for cls, d in profiles.items():
        out[cls] = {"auc": d.get("holdout_auc"),
                    "action": action.get(cls, "Review"),
                    "watchlist": (d.get("watchlist") or [])[:50],
                    "drivers": d.get("drivers", [])}
    return {"horizon": horizon, "training": fc.get("training", False),
            "classes": out, "note": fc.get("note", "")}


def discovery(horizon: int = HORIZON, top: int = 40) -> dict:
    """The frontier: accounts that are PROFITABLE over the window and
    PREDICTABLE, yet fit NO existing class -- candidate new abuse classes.

    Predictability proxy: the account is a consistent net-winner across its
    recent days (low-variance positive P&L), which is exactly the signature a
    class model would learn. Surfaced for a human to name, or to act on early."""
    from webapp import antifraud, rule_forecast
    rules = antifraud.load_rules()
    frame = antifraud._frame().copy()
    frame["decision_day"] = pd.to_datetime(frame["decision_day"])
    frame["pnl"] = pd.to_numeric(frame[_pnl_col(frame)], errors="coerce").fillna(0.0)

    masks = rule_forecast._masks(rules)
    classified = pd.Series(False, index=frame.index)
    for cls, (needed, build) in masks.items():
        if any(c not in frame.columns for c in needed):
            continue
        try:
            classified = classified | build(frame).fillna(False)
        except Exception:
            continue
    # account-level: net winner, and never classified
    ever_classified = set(frame.loc[classified.to_numpy(), "account_key"].unique())
    g = frame.groupby("account_key")["pnl"]
    net = g.sum()
    days = g.size()
    mean = g.mean()
    std = g.std().fillna(0)
    # consistency = mean/std (Sharpe-like); high => predictable winner
    consistency = (mean / std.replace(0, np.nan)).fillna(0)
    cand = pd.DataFrame({"net": net, "days": days, "consistency": consistency})
    cand = cand[(cand["net"] > 0) & (cand["days"] >= 8)
                & (~cand.index.isin(ever_classified))]
    cand["score"] = cand["consistency"] * np.log1p(cand["net"].clip(lower=0))
    cand = cand.sort_values("score", ascending=False).head(top)
    rows = [{"account_key": str(k), "net_usd": round(float(r.net), 0),
             "active_days": int(r.days), "consistency": round(float(r.consistency), 2)}
            for k, r in cand.iterrows()]
    return {"candidates": rows,
            "note": "profitable + consistent (predictable) + fits no current "
                    "class -> candidate new abuse class for naming or early action"}
