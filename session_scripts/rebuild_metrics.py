"""Recompute stored metrics with the point-in-time routable-population simulation.

The artefact's scores are unchanged -- only how the routing policy is EVALUATED.
The previous table used one global quantile as the cutoff, which set today's
threshold from scores that did not exist yet, and ranked only accounts that
turned out to be active. Both flattered the model.
"""
import json
import sys
import time

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import model_service as ms

for view in ("trading",):
    frame = ms.load_scores(view)
    if frame is None:
        print(f"{view}: no artefact")
        continue
    meta = ms.artifact_meta(view) or {}
    old = meta.get("metrics", {})

    t0 = time.time()
    config = ms.TrainingConfig(**{k: v for k, v in meta.get("config", {}).items()
                                  if k in {f.name for f in __import__("dataclasses").fields(ms.TrainingConfig)}})
    metrics = ms._routing_metrics(frame, config)
    meta["metrics"] = metrics
    _, meta_path = ms.artifact_paths(view)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"{view}: recomputed in {time.time()-t0:.0f}s\n")

    flat = metrics["flat_bbook"]
    print(f"{'policy':<14}{'profit':>16}{'vs flat':>15}{'maxDD':>14}{'vs flat':>13}"
          f"{'Sharpe':>8}{'routable':>10}{'hedged':>9}")
    print(f"{'FLAT B-BOOK':<14}{flat['total_pnl_usd']:>16,.0f}{'':>15}"
          f"{flat['max_drawdown_usd']:>14,.0f}{'':>13}{flat['sharpe']:>8.2f}")
    for key, row in metrics["by_fraction"].items():
        dp = row["total_pnl_usd"] - flat["total_pnl_usd"]
        dd = row["max_drawdown_usd"] - flat["max_drawdown_usd"]
        flag = "  <-- BEATS BOTH" if (dp > 0 and dd > 0) else ""
        print(f"{'hedge ' + key:<14}{row['total_pnl_usd']:>16,.0f}{dp:>+15,.0f}"
              f"{row['max_drawdown_usd']:>14,.0f}{dd:>+13,.0f}{row['sharpe']:>8.2f}"
              f"{row['mean_routable']:>10,.0f}{row['mean_hedged']:>9,.0f}{flag}")

    if old.get("by_fraction"):
        print("\nprevious (lookahead cutoff, active-only population):")
        for key, row in old["by_fraction"].items():
            print(f"  hedge {key}: {row['total_pnl_usd']:>16,.0f}  DD {row['max_drawdown_usd']:>13,.0f}")
