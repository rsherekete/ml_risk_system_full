import sys, time
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import model_service as ms
c = ms.load_config(ms.VIEW_QUANT)
# Half the rows pay for the FULL 174-column account-day corpus: day-
# stratified sampling keeps the whole window represented, and the artifact
# metrics are scale-corrected by the recorded sample fraction.
c.max_trade_rows = 11_000_000
ms.start_training(ms.VIEW_QUANT, c)
last = ""
while True:
    s = ms.job_state(ms.VIEW_QUANT)
    if s.message != last: print(f"[{s.progress:5.1%}] {s.message}", flush=True); last = s.message
    if s.status in ("done", "error"): break
    time.sleep(10)
print("STATUS:", s.status)
from pathlib import Path
print("booster saved:", (ms.ARTIFACTS / "quant_model.txt").exists())
