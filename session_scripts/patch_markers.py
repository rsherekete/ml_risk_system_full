"""Record the REQUESTED symbol set (not just the symbols that had bars) in
every day marker of the incremental bar cache, so symbols with no ticks on the
feed can never make a cached day look missing again."""
import sys, json
import pandas as pd
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import model_service as ms, path_features as pf

symbols = set(pd.read_parquet(ms.SCRATCH / "quant_feature_cache.parquet",
                              columns=["symbol"])["symbol"].astype(str).unique())
root = pf.bars_root(ms.SCRATCH)
markers = sorted((root / "_days").glob("*.json"))
grown = 0
for mk in markers:
    have = pf._marker_symbols(mk)
    merged = have | symbols
    if merged != have:
        mk.write_text(json.dumps({"symbols": sorted(merged)}), encoding="utf-8")
        grown += 1
print(f"frame symbols {len(symbols)} | markers {len(markers)} | updated {grown}")
