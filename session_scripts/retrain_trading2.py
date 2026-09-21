"""Offline retrain of the trading (account-day) model on the deflated corpus,
mirroring model_service._run exactly, with visible output."""
import sys, json, time
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from dataclasses import asdict
from webapp import model_service as ms

config = ms.load_config(ms.VIEW_TRADING)
print("config:", asdict(config))
started = time.time()
try:
    frame, metrics = ms._train_client_model(config)
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)
scores_path, meta_path = ms.artifact_paths(ms.VIEW_TRADING)
frame.to_parquet(scores_path, index=False)
meta_path.write_text(json.dumps({
    "fingerprint": config.fingerprint(),
    "config": asdict(config),
    "metrics": metrics,
    "rows": int(len(frame)),
    "trained_at": time.time(),
}, indent=2))
print(f"\nDONE in {(time.time()-started)/60:.1f} min | rows {len(frame):,}")
print("flat_bbook:", metrics.get("flat_bbook"))
print("roc_auc:", metrics.get("roc_auc"))
