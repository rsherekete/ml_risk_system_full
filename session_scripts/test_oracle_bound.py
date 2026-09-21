"""Why does 0.76 AUC on 'client wins tomorrow' not beat flat B-book?

The claim to test: direction and magnitude are close to independent here, so
knowing WHO wins says little about WHO COSTS MONEY. Hedging pays off on the
tail of win SIZES, and that tail is what nine attempts have failed to predict.

Three oracles bound the problem and separate "our model is weak" from "this
information is insufficient":

  1. DIRECTION ORACLE  -- perfect foresight on win/loss, nothing about size.
     If this barely beats flat B-book, then no direction model, however good,
     can route profitably and the ceiling is the information, not the fit.
  2. MAGNITUDE ORACLE  -- perfect foresight on |P&L|, nothing about direction.
  3. FULL ORACLE       -- both. The absolute ceiling.

Then the decomposition that explains the gap: among account-days our real model
hedges, how many dollars come from correctly-hedged winners versus wrongly
forfeited losers -- and critically, whether the winners we CATCH are smaller
than the winners we MISS.
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

BASE = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad"
MIN_TRAIN, CADENCE = 20, 5

parts = []
for database in sorted(pd.read_parquet(f"{BASE}\\bq_90d_records.parquet", columns=["database"])["database"].unique()):
    part = compact_memory(pd.read_parquet(f"{BASE}\\bq_90d_records.parquet", filters=[("database", "==", database)]))
    parts.append(build_active_day_frame(part, max_gap_days=None))
    del part; gc.collect()
frame = pd.concat(parts, ignore_index=True)
del parts; gc.collect()

columns = feature_columns(frame)
for column in columns:
    frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
frame["decision_day"] = pd.to_datetime(frame["decision_day"]).dt.normalize()
frame["label_wins"] = frame["target_client_wins"].astype(bool)
frame["pnl"] = pd.to_numeric(frame["target_profit"], errors="coerce")
frame = frame.loc[frame["pnl"].notna()].reset_index(drop=True)
days = sorted(frame["decision_day"].unique())
print(f"{len(frame):,} account-days, {frame['account_key'].nunique():,} accounts, {len(days)} days\n", flush=True)

# --- the real model, walk-forward -------------------------------------------
probability = pd.Series(np.nan, index=frame.index, dtype="float64")
model = lgb.LGBMClassifier(n_estimators=200, verbose=-1, random_state=0)
fitted = False
t0 = time.time()
for offset in range(MIN_TRAIN, len(days)):
    test_mask = frame["decision_day"] == days[offset]
    if not test_mask.any():
        continue
    if (offset - MIN_TRAIN) % CADENCE == 0 or not fitted:
        train = frame["decision_day"].isin(days[:offset])
        y = frame.loc[train, "label_wins"]
        if y.notna().sum() > 100 and y.nunique() >= 2:
            model.fit(frame.loc[train, columns], y); fitted = True
    if fitted:
        probability.loc[test_mask] = model.predict_proba(frame.loc[test_mask, columns])[:, 1]

scored = frame.loc[probability.notna()].copy()
scored["p_win"] = probability[probability.notna()]
auc = _rank_discrimination(scored["p_win"].to_numpy(), scored["label_wins"].to_numpy())["roc_auc"]
print(f"walk-forward model: {len(scored):,} scored rows, ROC AUC {auc:.4f} [{time.time()-t0:.0f}s]\n", flush=True)

# --- is direction independent of magnitude? ---------------------------------
scored["abs_pnl"] = scored["pnl"].abs()
print("=== THE CRUX: does predicted win probability say anything about SIZE? ===")
print(f"  corr(p_win, |pnl|)          {scored['p_win'].corr(scored['abs_pnl']):+.4f}")
print(f"  Spearman(p_win, |pnl|)      {scored['p_win'].rank().corr(scored['abs_pnl'].rank()):+.4f}")
winners = scored.loc[scored["label_wins"]]
print(f"  among WINNERS only, Spearman(p_win, win size) "
      f"{winners['p_win'].rank().corr(winners['pnl'].rank()):+.4f}")
print("  (near zero means the model cannot tell a $50 winner from a $50,000 one)\n")

print("win rate vs profitability by predicted decile:")
scored["decile"] = pd.qcut(scored["p_win"], 10, labels=False, duplicates="drop")
table = scored.groupby("decile").agg(
    rows=("pnl", "size"), win_rate=("label_wins", "mean"),
    client_pnl=("pnl", "sum"), mean_win=("pnl", lambda s: s[s > 0].mean()),
    mean_loss=("pnl", lambda s: s[s < 0].mean()))
for decile, row in table.iterrows():
    print(f"  d{int(decile)}  win {row['win_rate']:>5.1%}  client P&L ${row['client_pnl']:>14,.0f}  "
          f"avg win ${row['mean_win']:>9,.0f}  avg loss ${row['mean_loss']:>10,.0f}")
print("  (firm earns the NEGATIVE of client P&L -- every decile profitable to B-book)\n")

# --- oracle bounds ----------------------------------------------------------
baseline = -scored["pnl"].sum()
print(f"=== ORACLE BOUNDS (firm P&L; flat B-book = ${baseline:,.0f}) ===")


def firm_pnl(hedged: pd.Series) -> float:
    """Firm keeps the negative of client P&L on B-booked rows, nothing on hedged."""
    return float(-scored.loc[~hedged, "pnl"].sum())


rng = np.random.default_rng(0)
for pct in (0.05, 0.10, 0.20, 0.50):
    n = int(len(scored) * pct)
    # 1. Direction oracle: hedge the account-days that WILL win, chosen at
    #    random among them -- perfect direction, zero size information.
    winner_index = scored.index[scored["label_wins"]]
    pick = rng.choice(winner_index, size=min(n, len(winner_index)), replace=False)
    direction_oracle = pd.Series(False, index=scored.index); direction_oracle[pick] = True
    # 2. Magnitude oracle: hedge the largest |P&L| regardless of sign.
    magnitude_oracle = scored["abs_pnl"] >= scored["abs_pnl"].quantile(1 - pct)
    # 3. Full oracle: hedge the biggest WINNERS -- the only rows that actually
    #    cost the firm money.
    win_size = scored["pnl"].where(scored["pnl"] > 0, -np.inf)
    full_oracle = win_size >= win_size.quantile(1 - pct)
    # 4. Our model.
    model_hedge = scored["p_win"] >= scored["p_win"].quantile(1 - pct)

    print(f"\n  hedge {int(pct*100):>2}% of account-days:")
    for label, mask in (("direction oracle (perfect win/loss)", direction_oracle),
                        ("magnitude oracle (perfect |size|)", magnitude_oracle),
                        ("FULL oracle (biggest winners)", full_oracle),
                        ("our model (AUC %.2f)" % auc, model_hedge)):
        value = firm_pnl(mask)
        print(f"    {label:<38} ${value:>14,.0f}  vs flat {value - baseline:>+14,.0f}")

# --- where the model's dollars actually go ----------------------------------
print("\n=== WHY THE MODEL LOSES: dollar decomposition at 10% hedged ===")
hedged = scored["p_win"] >= scored["p_win"].quantile(0.9)
caught = scored.loc[hedged & scored["label_wins"], "pnl"]
missed = scored.loc[~hedged & scored["label_wins"], "pnl"]
forfeited = scored.loc[hedged & ~scored["label_wins"], "pnl"]
print(f"  winners CAUGHT   {len(caught):>7,} rows, ${caught.sum():>14,.0f} saved   "
      f"(mean ${caught.mean():>9,.0f})")
print(f"  winners MISSED   {len(missed):>7,} rows, ${missed.sum():>14,.0f} still paid "
      f"(mean ${missed.mean():>9,.0f})")
print(f"  losers FORFEITED {len(forfeited):>7,} rows, ${-forfeited.sum():>14,.0f} given up "
      f"(mean ${-forfeited.mean():>9,.0f})")
print(f"\n  net effect of hedging: ${caught.sum() + forfeited.sum():>+14,.0f}")
if len(caught) and len(missed):
    print(f"\n  mean size of winners we CATCH  ${caught.mean():,.0f}")
    print(f"  mean size of winners we MISS   ${missed.mean():,.0f}")
    print("  -> if MISSED >> CAUGHT, the model finds frequent small winners and "
          "walks past the rare huge ones,\n     which is exactly the failure mode that "
          "makes a strong direction model worthless for routing.")
