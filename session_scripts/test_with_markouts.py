"""Two-stage model WITH tick-derived market context, in both stages.

Baseline to beat (behavioural features only, leak-free):
  stage 1  P(win)  ROC AUC ~0.762

Markout features are joined per account-day and fed to BOTH the classifier
(does market reaction predict who wins?) and the magnitude model (does it
predict how big the win is?). Reported per stage so it is clear which one --
if either -- actually benefits.
"""
import sys, gc, time, warnings
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from trading_data.behaviour_features import build_active_day_frame, feature_columns
from trading_data.bigquery_data_client import compact_memory
from trading_data.research import _rank_discrimination

RECORDS = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\bq_90d_records.parquet"
MARKOUTS = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\markout_all_servers.parquet"
MIN_TRAIN, CADENCE = 20, 5

parts = []
for database in sorted(pd.read_parquet(RECORDS, columns=["database"])["database"].unique()):
    part = compact_memory(pd.read_parquet(RECORDS, filters=[("database", "==", database)]))
    parts.append(build_active_day_frame(part, max_gap_days=None))
    del part; gc.collect()
frame = pd.concat(parts, ignore_index=True)
del parts; gc.collect()
base_columns = feature_columns(frame)

markouts = pd.read_parquet(MARKOUTS)
markout_columns = [c for c in markouts.columns
                   if c not in {"database", "login", "day", "account_key"}
                   and pd.api.types.is_numeric_dtype(markouts[c])]
frame["day"] = pd.to_datetime(frame["day"])
frame = frame.merge(markouts[["account_key", "day", *markout_columns]], on=["account_key", "day"], how="left")
coverage = frame["context_trades"].notna().mean() if "context_trades" in frame else 0.0
print(f"{len(frame):,} rows | behavioural {len(base_columns)} + markout {len(markout_columns)} features "
      f"| markout coverage {coverage:.1%}\n", flush=True)

# Label columns are prefixed so they can never collide with a feature name --
# `wins` is already a real feature (winning trades closed today), and
# overwriting it with the label produced a fake ROC AUC of 1.0 once already.
frame["label_wins"] = frame["target_client_wins"].astype(bool)
frame["label_firm_cost"] = frame["target_profit"].clip(lower=0)
frame["label_log_win_size"] = np.log1p(frame["label_firm_cost"])
frame["decision_day"] = pd.to_datetime(frame["decision_day"]).dt.normalize()
days = sorted(frame["decision_day"].unique())


def run(columns, label):
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    p_win = pd.Series(np.nan, index=frame.index, dtype="float64")
    size = pd.Series(np.nan, index=frame.index, dtype="float64")
    clf = lgb.LGBMClassifier(n_estimators=200, verbose=-1, random_state=0)
    reg = lgb.LGBMRegressor(n_estimators=200, verbose=-1, random_state=0)
    clf_ok = reg_ok = False
    t0 = time.time()
    for offset in range(MIN_TRAIN, len(days)):
        test_mask = frame["decision_day"] == days[offset]
        if not test_mask.any():
            continue
        if (offset - MIN_TRAIN) % CADENCE == 0 or not (clf_ok and reg_ok):
            train_mask = frame["decision_day"].isin(days[:offset])
            y = frame.loc[train_mask, "label_wins"]
            if y.notna().sum() > 100 and y.nunique() >= 2:
                clf.fit(frame.loc[train_mask, columns], y); clf_ok = True
            winners = train_mask & frame["label_wins"]
            if winners.sum() > 100:
                reg.fit(frame.loc[winners, columns], frame.loc[winners, "label_log_win_size"]); reg_ok = True
        if clf_ok:
            p_win.loc[test_mask] = clf.predict_proba(frame.loc[test_mask, columns])[:, 1]
        if reg_ok:
            size.loc[test_mask] = reg.predict(frame.loc[test_mask, columns])

    valid = p_win.notna() & size.notna()
    scored = frame.loc[valid].copy()
    scored["p_win"] = p_win[valid]
    scored["expected_size"] = np.expm1(size[valid]).clip(lower=0)
    scored["priority"] = scored["p_win"] * scored["expected_size"]

    disc = _rank_discrimination(scored["p_win"].to_numpy(), scored["label_wins"].to_numpy())
    winners = scored.loc[scored["label_wins"]]
    spearman = float(pd.Series(winners["expected_size"].to_numpy()).rank().corr(winners["label_firm_cost"].rank()))
    total = scored["label_firm_cost"].sum()
    capture = {}
    for pct in (0.05, 0.10, 0.20):
        cutoff = scored["priority"].quantile(1 - pct)
        capture[pct] = scored.loc[scored["priority"] >= cutoff, "label_firm_cost"].sum() / total
    print(f"{label:<28} stage1 AUC {disc['roc_auc']:.4f} | stage2 Spearman {spearman:+.4f} | "
          f"capture 5% {capture[0.05]:.1%}  10% {capture[0.10]:.1%}  20% {capture[0.20]:.1%}  [{time.time()-t0:.0f}s]", flush=True)
    return disc["roc_auc"], capture[0.10]


auc_base, cap_base = run(base_columns, "behavioural only")
auc_full, cap_full = run(base_columns + markout_columns, "behavioural + markouts")
print(f"\nDELTA: AUC {auc_full - auc_base:+.4f}   top-10% capture {cap_full - cap_base:+.1%}")

rng = np.random.default_rng(0)
random_capture = []
for _ in range(20):
    shuffled = pd.Series(rng.random(len(frame)), index=frame.index)
    cutoff = shuffled.quantile(0.9)
    random_capture.append(frame.loc[shuffled >= cutoff, "label_firm_cost"].sum() / frame["label_firm_cost"].sum())
print(f"random top-10% capture baseline: {np.mean(random_capture):.1%}")
