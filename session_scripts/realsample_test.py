import sys, tempfile, time
from pathlib import Path
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import model_service as ms

# Redirect artifact + scratch dirs to temp so the REAL production models are
# never overwritten by this small-sample run.
tmp = Path(tempfile.mkdtemp(prefix="perclass_"))
(tmp / "artifacts").mkdir(); (tmp / "scratch").mkdir()
ms.ARTIFACTS = tmp / "artifacts"
ms.SCRATCH = tmp / "scratch"
print("temp artifacts ->", ms.ARTIFACTS)

cfg = ms.TrainingConfig()
cfg.history_days = 30          # short window = fast real load + feature build
cfg.max_trade_rows = 150000    # cap rows so it runs in a couple of minutes
cfg.max_fit_rows = 150000
cfg.min_train_days = 5
cfg.refit_cadence_days = 3
cfg.n_estimators = 80

t0 = time.time()
output, metrics = ms._train_trade_model(cfg)
print(f"\n_train_trade_model finished in {time.time()-t0:.0f}s | output rows {len(output):,}")

print("\noutput has score_cal:", "score_cal" in output.columns,
      "| model_class:", "model_class" in output.columns)
print("model_class distribution:")
print(output["model_class"].value_counts().to_string())

print("\nper-class metrics (AUC unchanged by calibration; Brier should drop):")
for c, r in (metrics.get("per_class") or {}).items():
    print(f"  {c:7} rows={r['rows']:>7} base={r.get('base_rate')} "
          f"auc={r.get('auc')} brier_raw={r.get('brier_raw')} "
          f"brier_cal={r.get('brier_cal')}")

print("\ncalibrated classes:", metrics.get("calibrated_classes"))
print("walk_curve points:", len(metrics.get("walk_curve") or []))
if metrics.get("walk_curve"):
    print("  first:", metrics["walk_curve"][0])
    print("  last :", metrics["walk_curve"][-1])

print("\nartifacts written:")
for f in sorted(ms.ARTIFACTS.glob("*")):
    print(f"  {f.name}  ({f.stat().st_size/1024:.1f} KB)")

# sanity: per-class model files + calibrators exist for classes that had data
import json
calib = json.loads((ms.ARTIFACTS / "quant_calibrators.json").read_text())
print("\ncalibrators json classes:", sorted(calib.keys()))
for c in ("metals", "fx", "index", "crypto"):
    mp = ms.ARTIFACTS / f"quant_model_{c}.txt"
    print(f"  quant_model_{c}.txt:", "EXISTS" if mp.exists() else "(fell back to pooled)")

# cleanup
import shutil
shutil.rmtree(tmp, ignore_errors=True)
print("\nREAL-SAMPLE TEST DONE (temp cleaned, production artifacts untouched)")
