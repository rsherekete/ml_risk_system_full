"""Seed the REAL-FLOW feature cache (identical logic and signature to the
trainer's cache block, so the next training run hits it too)."""
import hashlib
import json
import sys
import time

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import model_service as ms  # noqa: E402

import pandas as pd  # noqa: E402

from webapp.trade_features import TRADE_FEATURES, build_trade_features  # noqa: E402

config = ms.load_config(ms.VIEW_QUANT)
config.max_trade_rows = 11_000_000
signature = hashlib.md5(
    ("|".join(TRADE_FEATURES)
     + f"|{config.history_days}|{config.max_trade_rows}").encode()).hexdigest()

cache_path = ms.SCRATCH / "quant_feature_cache.parquet"
meta_path = ms.SCRATCH / "quant_feature_cache.json"
if cache_path.exists() and meta_path.exists():
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("signature") == signature:
        print(f"cache already valid: {meta['rows']:,} rows"); sys.exit(0)

t0 = time.time()
trades = ms._load_trade_history(config, ms.VIEW_QUANT)
window_start = trades["close_time"].min().floor("D")
trades = trades.loc[trades["open_time"] >= window_start]
sample_fraction = float(trades.attrs.get("sample_fraction", 1.0) or 1.0)
if len(trades) > config.max_trade_rows:
    fraction = config.max_trade_rows / len(trades)
    trades = (trades.assign(_d=pd.to_datetime(trades["open_time"]).dt.normalize())
              .groupby("_d", group_keys=False, observed=True)
              .apply(lambda g: g.sample(frac=fraction, random_state=0))
              .drop(columns="_d").reset_index(drop=True))
    sample_fraction *= fraction
print(f"loaded+sampled {len(trades):,} rows in {time.time()-t0:.0f}s; building features...")
trades = build_trade_features(trades)
trades.to_parquet(cache_path)
meta_path.write_text(json.dumps(
    {"signature": signature, "built_at": time.time(), "rows": len(trades),
     "sample_fraction": sample_fraction}), encoding="utf-8")
print(f"CACHE SEEDED: {len(trades):,} rows, {time.time()-t0:.0f}s total")
