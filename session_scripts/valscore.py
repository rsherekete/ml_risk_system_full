import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import numpy as np, pandas as pd
from pathlib import Path
from webapp import model_service as ms
from webapp import trade_features as tf
import lightgbm as lgb

ART = Path(r"c:\Users\RoyVivasi\Documents\notebook\webapp\artifacts")
feats = (ART / "quant_model_features.txt").read_text(encoding="utf-8").splitlines()
model = lgb.Booster(model_file=str(ART / "quant_model.txt"))

# the history family the live path had been getting wrong (constant per account)
HIST = ["hist_win_rate","hist_mean_pnl","hist_pnl_std","hist_mean_notional",
        "hist_recent_pnl5","trade_index","notional_vs_usual",
        "hist_wr_20p","hist_wr_10p","hist_wr_5p","hist_wr_2p",
        "hist_mp_20p","hist_mp_10p","hist_mp_5p","hist_mp_2p",
        "hist_wr_trend","hist_wr_chop","hist_mp_trend","hist_mp_chop",
        "acct_pnl_5d","acct_pnl_20d","acct_pnl_60d","acct_vol_20d","acct_winrate_20d",
        "acct_trades_5d","acct_trades_20d","acct_dd_20d","acct_best_20d",
        "acct_worst_20d","acct_tenure_days","equity_proxy","notional_to_equity"]
HIST = [h for h in HIST if h in feats]

df = pd.read_parquet(ms.SCRATCH / "quant_feature_cache.parquet",
                     columns=list(set(feats + HIST + ["account_key","net_profit"])))
# take the last ~600k rows (most recent) for a fast, representative slice
df = df.sort_values("account_key").tail(800000).reset_index(drop=True)
y = (pd.to_numeric(df["net_profit"], errors="coerce") > 0).to_numpy()
print(f"rows: {len(df):,}  base client-win {y.mean():.1%}")

def gated(p, tag):
    for thr in (0.85, 0.90):
        m = p >= thr
        wr = y[m].mean() if m.sum() else float("nan")
        print(f"  [{tag}] score>={thr}: n={m.sum():,}  client-win={wr:.1%}")

# 1) CORRECT (per-trade parity history) -- what the fix produces
Xc = df[feats].to_numpy("float32"); np.putmask(Xc, ~np.isfinite(Xc), np.nan)
pc = model.predict(Xc)
gated(pc, "PARITY per-trade history (the fix)")

# 2) BROKEN: replace history family with a CONSTANT per account (the 45-day-mean
#    style snapshot the live path used) -- same value for all of an account's trades
broken = df.copy()
means = broken.groupby("account_key")[HIST].transform("mean")
for h in HIST:
    broken[h] = means[h]
Xb = broken[feats].to_numpy("float32"); np.putmask(Xb, ~np.isfinite(Xb), np.nan)
pb = model.predict(Xb)
gated(pb, "BROKEN constant-per-account history (old live path)")
