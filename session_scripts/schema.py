import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import pandas as pd
from webapp import model_service as ms
from webapp import trade_features as tf
cache = ms.SCRATCH / "quant_feature_cache.parquet"
df = pd.read_parquet(cache, columns=None)
cols = list(df.columns)
feats = set(tf.TRADE_FEATURES)
raw = [c for c in cols if c not in feats]
print("TOTAL cols:", len(cols))
print("\nRAW (non-feature) columns present:", len(raw))
for c in raw:
    print("  ", c, "|", str(df[c].dtype))
print("\nsample row raw values:")
r = df.iloc[len(df)//2]
for c in raw[:25]:
    print(f"  {c:16s} = {r[c]!r}")
