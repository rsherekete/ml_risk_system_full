"""Where do the ~190 seconds on /quant/overview actually go? Measure each step."""
import sys
import time

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import pandas as pd

from webapp import model_service as ms
from webapp import views


def step(label, fn):
    t0 = time.time()
    result = fn()
    print(f"  {label:<44} {time.time() - t0:>8.1f}s", flush=True)
    return result


print("cold path for /quant/overview:")
frame = step("read parquet (load_scores)", lambda: ms.load_scores("quant"))
print(f"    -> {len(frame):,} rows, {frame.memory_usage(deep=True).sum()/1e9:.2f} GB")

step("available_days", lambda: views.available_days(frame))
step("coverage_note", lambda: views.coverage_note(frame))
config = ms.load_config("quant")
step("trade_equity_curves", lambda: ms.trade_equity_curves(
    frame, config.hedge_fraction, config.probability_threshold))
step("summary_stats", lambda: views.summary_stats(frame))

print("\nsecond call (should be cached):")
step("load_scores again", lambda: ms.load_scores("quant"))
step("frame_facts again", lambda: ms.frame_facts("quant", frame))
step("cached_equity_curves again", lambda: ms.cached_equity_curves(
    "quant", frame, config.hedge_fraction, config.probability_threshold))
