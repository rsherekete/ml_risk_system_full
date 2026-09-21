import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import numpy as np, pandas as pd
from webapp import model_service as ms
from webapp import trade_features as tf

RAW = tf.SCORING_RAW_COLUMNS
df = pd.read_parquet(ms.SCRATCH / "quant_feature_cache.parquet")

CTX = {"ctx_return_1h","ctx_return_4h","ctx_return_24h","ctx_vol_24h","ctx_flow_prev",
       "zscore_24h","range_pos_24h","vol_regime","mom_1h_vol","mom_4h_vol",
       "mom_24h_vol","with_momentum","momentum_align","sl_dist_vol","tp_dist_vol",
       "hist_wr_trend","hist_wr_chop","hist_mp_trend","hist_mp_chop"}

counts = df["account_key"].value_counts()
picks = counts[(counts >= 40) & (counts <= 300)].head(30).index.tolist()
sub = df[df["account_key"].isin(picks)].copy().sort_values("open_time")

targets, tape_rows = [], []
for acct, g in sub.groupby("account_key"):
    g = g.sort_values("open_time")
    targets.append(g.iloc[-1]); tape_rows.append(g.iloc[:-1])
target_df = pd.DataFrame(targets)
tape = pd.concat(tape_rows, ignore_index=True)[RAW].copy()

new = target_df[RAW].copy().reset_index(drop=True)
new["close_time"] = pd.NaT; new["net_profit"] = np.nan
new["close_price"] = np.nan; new["state"] = "open"

built = tf.build_features_for_scoring(new, tape)
print(f"scored {len(built)} open trades against tape of {len(tape)}")

feats = list(tf.TRADE_FEATURES)
tgt = target_df.reset_index(drop=True)
non_ctx_ok = ctx_diff = 0; mismatches = []
for f in feats:
    a = pd.to_numeric(tgt[f], errors="coerce").to_numpy(dtype="float64")
    b = pd.to_numeric(built[f], errors="coerce").to_numpy(dtype="float64")
    both_nan = np.isnan(a) & np.isnan(b)
    d = np.abs(a - b); d[both_nan] = 0.0
    match = np.mean((d < 1e-5) | both_nan)
    if f in CTX: ctx_diff += 1; continue
    if match > 0.999: non_ctx_ok += 1
    else: mismatches.append((f, float(np.nanmax(d)), float(match)))

n_noctx = len([f for f in feats if f not in CTX])
print(f"\nNON-ctx features matching training exactly: {non_ctx_ok}/{n_noctx}")
print(f"ctx/bar features skipped (live overlays real bars): {ctx_diff}")
if mismatches:
    print("\nUNEXPECTED mismatches (should be empty):")
    for f, mx, m in mismatches[:30]:
        print(f"  {f:26s} maxdiff={mx:.4g}  match={m:.1%}")
else:
    print("\nPARITY CONFIRMED: every non-bar feature reproduced training exactly.")
