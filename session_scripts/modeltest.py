import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import numpy as np, pandas as pd
from pathlib import Path
from webapp import model_service as ms
from webapp import trade_features as tfmod
import lightgbm as lgb

ART = Path(r"c:\Users\RoyVivasi\Documents\notebook\webapp\artifacts")
SCRATCH = ms.SCRATCH
print("SCRATCH:", SCRATCH)
print("AD_DIR :", tfmod._AD_DIR)

feats = (ART / "quant_model_features.txt").read_text(encoding="utf-8").splitlines()
print(f"\nmodel features: {len(feats)}  | TRADE_FEATURES: {len(tfmod.TRADE_FEATURES)}")
print("order identical to TRADE_FEATURES?", feats == list(tfmod.TRADE_FEATURES))
# where first differs
if feats != list(tfmod.TRADE_FEATURES):
    for i,(a,b) in enumerate(zip(feats, tfmod.TRADE_FEATURES)):
        if a != b:
            print(f"  first diff at {i}: model={a!r} vs TRADE_FEATURES={b!r}"); break

model = lgb.Booster(model_file=str(ART / "quant_model.txt"))
print("model n_features:", model.num_feature())

cache = SCRATCH / "quant_feature_cache.parquet"
if not cache.exists():
    print("\nNO training feature cache at", cache); sys.exit()
tf = pd.read_parquet(cache)
print(f"\ntraining feature frame: {len(tf):,} rows, {tf.shape[1]} cols")
print("open_time range:", pd.to_datetime(tf['open_time']).min(), "->", pd.to_datetime(tf['open_time']).max())

# does the model hit high gated win-rate on TRAINING features?
X = tf[feats].to_numpy('float32')
np.putmask(X, ~np.isfinite(X), np.nan)
p = model.predict(X)
y = (pd.to_numeric(tf['net_profit'], errors='coerce') > 0).to_numpy()
print(f"\nbase client win rate: {y.mean():.1%}")
for thr in (0.85, 0.90):
    m = p >= thr
    print(f"  score>={thr}: n={m.sum():,}  client-win={y[m].mean():.1%}  (COPY band -> want HIGH)")
for thr in (0.25, 0.10):
    m = p <= thr
    print(f"  score<={thr}: n={m.sum():,}  client-win={y[m].mean():.1%}  (INVERT band -> want LOW)")

# staleness of AD corpus
adf = pd.read_parquet(SCRATCH / "model_frame.parquet", columns=['decision_day'])
dd = pd.to_datetime(adf['decision_day'])
print(f"\nAD corpus decision_day: {dd.min()} -> {dd.max()}  (today ~ {pd.Timestamp.now().date()})")

# per-feature NaN rate in TRAINING (baseline to compare live against)
nan_rate = tf[feats].isna().mean().sort_values(ascending=False)
print("\ntop-12 TRAINING NaN-rate features (live should match these):")
for name, r in nan_rate.head(12).items():
    print(f"  {name:28s} {r:.1%}")
print(f"features with 0% NaN in training: {(nan_rate==0).sum()} / {len(feats)}")
