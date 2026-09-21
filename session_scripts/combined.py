import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import numpy as np, pandas as pd
from pathlib import Path
from webapp import model_service as ms
import lightgbm as lgb

ART = Path(r"c:\Users\RoyVivasi\Documents\notebook\webapp\artifacts")
feats = (ART / "quant_model_features.txt").read_text(encoding="utf-8").splitlines()
model = lgb.Booster(model_file=str(ART / "quant_model.txt"))
idx = {f: i for i, f in enumerate(feats)}

CTX = [f for f in feats if f.startswith(("ctx_","mom_")) or f in ("with_momentum",
       "momentum_align","vol_regime","zscore_24h","range_pos_24h","sl_dist_vol","tp_dist_vol")]
AD = [f for f in feats if f.startswith("ad_")]

df = pd.read_parquet(ms.SCRATCH / "quant_feature_cache.parquet",
                     columns=list(set(feats + ["net_profit","symbol"])))
gold = df[df["symbol"].astype(str).str.upper().str.startswith("XAU")].tail(400000).reset_index(drop=True)
y = (pd.to_numeric(gold["net_profit"], errors="coerce") > 0).to_numpy()
base = gold[feats].to_numpy("float32"); np.putmask(base, ~np.isfinite(base), np.nan)
print(f"GOLD trades: {len(gold):,}  base client-win {y.mean():.1%}")

def gate(p, tag):
    for thr in (0.85, 0.90):
        m = p >= thr
        wr = y[m].mean() if m.sum() else float("nan")
        print(f"  [{tag}] >= {thr}: n={m.sum():>7,}  client-win={wr:.1%}")

gate(model.predict(base), "FULL (correct)")

# what the '+' book actually got: symbol_code NaN + ALL ctx NaN together
X = base.copy()
X[:, idx["symbol_code"]] = np.nan
for f in CTX: X[:, idx[f]] = np.nan
gate(model.predict(X), "symbol_code + ctx NaN (the '+' book bug)")

# add AD stale-ish: not NaN but shuffle within to simulate wrong values
X2 = X.copy()
for f in AD:
    col = X2[:, idx[f]]
    X2[:, idx[f]] = np.random.permutation(col)   # decorrelate AD (stale/wrong proxy)
gate(model.predict(X2), "+ AD decorrelated (stale proxy)")
