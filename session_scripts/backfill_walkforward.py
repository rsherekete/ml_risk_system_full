"""BACKFILL the realised walk-forward, honestly:
  - dedicated models per rule trained with cutoff Aug 25 (they never see
    any backfill day);
  - for each trading day D in Aug 26 .. latest: features as of D-1 (with a
    DAY-ACCURATE corpus snapshot -- no future rows), predictions for every
    account, verified against the accounts ACTUALLY ACTIVE on D and their
    true rule state at D;
  - writes rule_verification.json (latest day -> the tabs' green cards)
    with the full per-day history attached.
"""
import sys, time, json
from datetime import datetime, timedelta
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import numpy as np
import pandas as pd

from webapp import rule_models as rm
from webapp import af_registry, antifraud

t0 = time.time()
CUTOFF = datetime(2026, 8, 25)
rules = [c for c in af_registry.load().get("categories", [])
         if c.get("active") and c.get("use_ml")]
print(f"rules: {[c['key'] for c in rules]}", flush=True)

# ---- train backfill models at the early cutoff ----
models = {}
for c in rules:
    key = c["key"]
    data = rm._panel_dataset(key, cutoff=CUTOFF)
    if data is None:
        print(f"{key}: no panel"); continue
    X = data["X"]
    keep = (X["n_trades"] >= 10) if "n_trades" in X else pd.Series(True, index=X.index)
    X = X.loc[keep].astype(float)
    y_now = data["y_now"].loc[keep]
    groups = data["groups"].loc[keep]
    mdl, met = rm._fit_scored(rm._now_features(X, data["definition"]),
                              y_now, groups)
    if mdl is None:
        print(f"{key}: {met}", flush=True); continue
    models[key] = {"model": mdl, "thr": float(met["threshold"]),
                   "features": list(rm._now_features(X, data["definition"]).columns),
                   "definition": data["definition"], "cv": met}
    print(f"[{time.time()-t0:.0f}s] {key}: trained@{CUTOFF:%Y-%m-%d} "
          f"AUC {met['auc']} thr {met['threshold']}", flush=True)

# ---- day-by-day tables ----
end = datetime.utcnow()
trades = rm._window_trades(CUTOFF - timedelta(days=rm.OBS_DAYS), end)
trades["day"] = trades["open_time"].dt.strftime("%Y-%m-%d")
anchors = rm.combined_anchors()
frame = antifraud._frame().sort_values("decision_day")
frame["decision_day"] = pd.to_datetime(frame["decision_day"])
days = sorted(d for d in trades["day"].unique()
              if d >= CUTOFF.strftime("%Y-%m-%d"))
print(f"[{time.time()-t0:.0f}s] trades {len(trades):,} | days {len(days)}",
      flush=True)

def corpus_asof(day: str) -> pd.DataFrame:
    sub = frame[frame["decision_day"] <= pd.Timestamp(day)]
    latest = sub.groupby("account_key", observed=True).last()
    cols = [c for c in latest.columns if c != "decision_day"
            and pd.api.types.is_numeric_dtype(latest[c])]
    return latest[cols]

def table_asof(day: str) -> pd.DataFrame:
    d_end = pd.Timestamp(day) + pd.Timedelta(days=1)
    win = trades[(trades["open_time"] < d_end)
                 & (trades["open_time"] >= d_end - pd.Timedelta(days=rm.OBS_DAYS))]
    return rm.feature_table(win, anchors, corpus=corpus_asof(day))

history = {c["key"]: [] for c in rules}
prev_day = None
prev_table = None
for day in days:
    table = table_asof(day)
    if prev_table is not None:
        active = set(trades.loc[trades["day"] == day, "account_key"].astype(str))
        cohort = [a for a in table.index if str(a) in active]
        for c in rules:
            key = c["key"]
            m = models.get(key)
            if m is None or not cohort:
                continue
            X = prev_table.reindex(columns=m["features"]).fillna(0).astype(float)
            p = pd.Series(m["model"].predict_proba(X.to_numpy())[:, 1],
                          index=prev_table.index)
            pred = (p >= m["thr"])
            truth = rm._rule_mask(m["definition"], table)
            tp = fp = fn = tn = 0
            for a in cohort:
                pr = bool(pred.get(a, False))
                tr = bool(truth.get(a, False))
                tp += pr and tr; fp += pr and not tr
                fn += (not pr) and tr; tn += (not pr) and not tr
            n = max(tp + fp + fn + tn, 1)
            history[key].append({
                "day": day, "snapshot_day": prev_day,
                "active_accounts": n,
                "predicted_flag_active": tp + fp,
                "actually_flagged": tp + fn,
                "precision": round(tp / max(tp + fp, 1), 3),
                "recall": round(tp / max(tp + fn, 1), 3),
                "accuracy": round((tp + tn) / n, 3),
                "base_rate": round((tp + fn) / n, 4)})
    prev_day, prev_table = day, table
    print(f"[{time.time()-t0:.0f}s] {day} done", flush=True)

out = {"verified_against_day": days[-1] if days else None,
       "backfilled": True, "model_cutoff": CUTOFF.strftime("%Y-%m-%d"),
       "rules": {}, "history": history}
for key, hist in history.items():
    if not hist:
        continue
    last = hist[-1]
    mean_p = float(np.mean([h["precision"] for h in hist]))
    mean_r = float(np.mean([h["recall"] for h in hist]))
    out["rules"][key] = dict(
        last, status="verified",
        precision=last["precision"], recall=last["recall"],
        mean_precision=round(mean_p, 3), mean_recall=round(mean_r, 3),
        days_verified=len(hist),
        explain=(f"BACKFILLED walk-forward (model cutoff "
                 f"{CUTOFF:%b %d}, never saw these days): over "
                 f"{len(hist)} trading days, average precision "
                 f"{mean_p:.0%} / recall {mean_r:.0%} on each day's "
                 f"active accounts; latest day {last['day']}: predicted "
                 f"{last['predicted_flag_active']} of "
                 f"{last['active_accounts']} active would flag, "
                 f"{last['actually_flagged']} truly did."))
(rm.ART / "rule_verification.json").write_text(json.dumps(out),
                                               encoding="utf-8")
print(f"[{time.time()-t0:.0f}s] WROTE rule_verification.json", flush=True)
for key, hist in history.items():
    if hist:
        print(f"{key}: days {len(hist)} | "
              f"P mean {np.mean([h['precision'] for h in hist]):.3f} | "
              f"R mean {np.mean([h['recall'] for h in hist]):.3f} | "
              f"base mean {np.mean([h['base_rate'] for h in hist]):.4f}",
              flush=True)
