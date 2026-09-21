import sys, tempfile, time, json, shutil
from pathlib import Path
import numpy as np
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import model_service as ms

tmp = Path(tempfile.mkdtemp(prefix="path_"))
(tmp / "artifacts").mkdir(); (tmp / "scratch").mkdir()
ms.ARTIFACTS = tmp / "artifacts"
ms.SCRATCH = tmp / "scratch"

cfg = ms.TrainingConfig()
cfg.history_days = 12          # short window -> small tick pull (~12d x symbols)
cfg.max_trade_rows = 120000
cfg.max_fit_rows = 120000
cfg.min_train_days = 3
cfg.refit_cadence_days = 3
cfg.n_estimators = 60

t0 = time.time()
output, metrics = ms._train_trade_model(cfg)
print(f"\nrun finished in {time.time()-t0:.0f}s | rows {len(output):,}")

print("\n--- path coverage & excursion sanity ---")
print("path_coverage:", metrics.get("path_coverage"))
# the trainer keeps the columns on the output? (they are on `trades`, not output)
# so re-derive from the saved artifacts instead: check the path models exist
print("\n--- holdout correlations (pooled + per class) ---")
for name in ("quant_mae_q50", "quant_mae_q80", "quant_exit_fe", "quant_perlot"):
    print(f"  {name:14}:", metrics.get(name))

print("\n--- artifacts written ---")
for f in sorted(ms.ARTIFACTS.glob("*")):
    print(f"  {f.name:28} {f.stat().st_size/1024:8.1f} KB")

must = ["quant_mae_q50.txt", "quant_mae_q80.txt", "quant_exit_fe.txt",
        "entry_model.txt", "quant_perlot.txt"]
missing = [m for m in must if not (ms.ARTIFACTS / m).exists()]
print("\nrequired pooled path artifacts missing:", missing or "none")

# bars cache landed?
bars = list(ms.SCRATCH.glob("path_bars_*.parquet"))
print("bars cache files:", [b.name for b in bars])
if bars:
    import pandas as pd
    b = pd.read_parquet(bars[0])
    print(f"  bars: {len(b):,} rows | symbols {b['symbol'].nunique()} | "
          f"{b['minute'].min()} -> {b['minute'].max()}")

shutil.rmtree(tmp, ignore_errors=True)
print("\nPATH TEST DONE (temp cleaned, production untouched)")
