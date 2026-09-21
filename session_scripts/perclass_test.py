import sys
import numpy as np
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import lightgbm as lgb
from webapp import model_service as ms

rng = np.random.default_rng(0)
N = 45000
# 4 classes with different base rates and different signal directions, so a
# pooled model is genuinely worse-calibrated per class than per-class models.
classes = np.array(["metals", "fx", "index", "crypto"])
mclass = rng.choice(classes, size=N, p=[0.55, 0.20, 0.15, 0.10])   # gold-heavy
X = rng.normal(size=(N, 5)).astype("float32")
# class-specific linear signal -> profit sign
coef = {"metals": 1.5, "fx": -1.2, "index": 0.8, "crypto": 2.0}
bias = {"metals": 0.2, "fx": -0.5, "index": 0.0, "crypto": 0.4}
logit = np.array([coef[c] for c in mclass]) * X[:, 0] + np.array([bias[c] for c in mclass])
p = 1 / (1 + np.exp(-logit))
win = rng.random(N) < p
profit = np.where(win, rng.uniform(1, 50, N), -rng.uniform(1, 50, N))
symbol_weight = np.ones(N)
usable = np.ones(N, dtype=bool)

cfg = ms.TrainingConfig()
cfg.n_estimators = 60          # small = fast for the smoke test
cfg.max_fit_rows = 0

# 1) per-class fit
models = ms._fit_class_models(X, profit, mclass, symbol_weight, usable, cfg, lgb)
print("1) fitted classes:", sorted(models.keys()))
assert ms._POOL_KEY in models and "metals" in models and "fx" in models

# 2) routing: score each row with its class model
score = np.full(N, np.nan)
for cls in np.unique(mclass):
    rows = np.flatnonzero(mclass == cls)
    mdl = models.get(cls) or models.get(ms._POOL_KEY)
    score[rows] = mdl.predict_proba(X[rows])[:, 1]
print("2) routed scores finite:", np.isfinite(score).all(),
      "| range [%.3f, %.3f]" % (score.min(), score.max()))
assert np.isfinite(score).all()

# 3) calibration
cal = ms._fit_calibrators(score, profit, mclass)
print("3) calibrators for:", sorted(cal.keys()))
score_cal = ms._apply_calibrators(score, mclass, cal)
assert score_cal.min() >= 0 and score_cal.max() <= 1

# 4) per-class metrics: calibration should not hurt AUC and should lower Brier
import pandas as pd
out = pd.DataFrame({"model_class": mclass, "pnl": profit,
                    "score": score, "score_cal": score_cal})
pc = ms._per_class_metrics(out)
print("4) per-class metrics:")
for c, r in pc.items():
    print(f"   {c}: rows={r['rows']} base={r.get('base_rate')} auc={r.get('auc')} "
          f"brier_raw={r.get('brier_raw')} brier_cal={r.get('brier_cal')}")
    if "brier_cal" in r:
        assert r["brier_cal"] <= r["brier_raw"] + 1e-6, "calibration should not worsen Brier"

# 5) empty-calibrator fallback returns raw untouched
raw_back = ms._apply_calibrators(score, mclass, {})
assert np.allclose(raw_back, score)
print("5) empty-calibrator fallback returns raw: OK")

# 6) vantage fallback path: no per-class artifacts yet -> uses pooled.predict raw
from webapp import vantage
class FakePooled:
    def predict(self, v): return np.array([0.73])
# force-clear caches so it reads current (absent) artifacts
vantage._CLASS_BOOSTERS = None; vantage._CALIBRATORS = None
val = vantage._predict_calibrated("EURUSD", np.zeros((1, 5)), FakePooled())
print("6) vantage fallback (no artifacts) ->", val, "(expect 0.73)")
assert abs(val - 0.73) < 1e-9
print("\nALL PER-CLASS TESTS PASSED")
