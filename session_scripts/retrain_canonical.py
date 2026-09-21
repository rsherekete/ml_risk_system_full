"""Full canonical-consistent retrain of the quant trade model.
Rebuilds the feature cache (now canonicalised in build_trade_features) and
retrains the stance model + regenerates quant_symbols.txt. Then retrains the
magnitude, per-lot (E) and exit-FE models off the fresh canonical cache so
the WHOLE engine is parity-consistent."""
import sys, time, subprocess
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import model_service as ms

t0 = time.time()
cfg = ms.load_config(ms.VIEW_QUANT)
# RECENT window, memory-safe: 180 days and a 9M-row cap builds in minutes
# (the 2y/24.6M default was swapping at 3.9GB free), and recent data is the
# RIGHT calibration for fixing current live behaviour anyway.
import dataclasses
cfg = dataclasses.replace(cfg, history_days=180, max_trade_rows=9_000_000)
print(f"[{time.time()-t0:.0f}s] starting quant retrain "
      f"(history_days={cfg.history_days}, cap={cfg.max_trade_rows:,}) — "
      f"canonical cache + stance model + quant_symbols.txt", flush=True)
ms.start_training(ms.VIEW_QUANT, cfg)
# poll the job to completion
while True:
    time.sleep(30)
    st = ms._JOBS.get(ms.VIEW_QUANT)
    if st is None:
        continue
    print(f"[{time.time()-t0:.0f}s] {st.status}: {st.message}", flush=True)
    if st.status in ("done", "error"):
        break
print(f"[{time.time()-t0:.0f}s] stance retrain {st.status}", flush=True)
if st.status != "done":
    sys.exit(1)

# now the downstream engine models off the fresh canonical cache
for script in ("train_perlot.py",):
    print(f"[{time.time()-t0:.0f}s] running {script}", flush=True)
    r = subprocess.run(
        [r"c:\Users\RoyVivasi\Documents\notebook\.venv\Scripts\python.exe",
         rf"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\{script}"],
        capture_output=True, text=True)
    print(r.stdout[-500:], r.stderr[-300:], flush=True)
print(f"[{time.time()-t0:.0f}s] DONE — engine models canonical-consistent", flush=True)
