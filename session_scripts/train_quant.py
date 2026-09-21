"""Build the Quant (trade-level) artefact so the web app has real data."""
import sys
import time

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import model_service as ms

config = ms.load_config(ms.VIEW_QUANT)
print(f"fingerprint {config.fingerprint()} | cadence {config.refit_cadence_days}d", flush=True)
ms.start_training(ms.VIEW_QUANT, config)

last = ""
while True:
    state = ms.job_state(ms.VIEW_QUANT)
    if state.message != last:
        print(f"  [{state.progress:5.1%}] {state.message}", flush=True)
        last = state.message
    if state.status in ("done", "error"):
        break
    time.sleep(5)

print("STATUS:", state.status, "|", state.message)
if state.status == "error":
    print("\n".join(state.log[-4:]))
else:
    meta = ms.artifact_meta(ms.VIEW_QUANT)
    metrics = meta["metrics"]
    flat = metrics["flat_bbook"]
    print(f"\nrows {meta['rows']:,} | ROC AUC {metrics['roc_auc']:.4f}")
    print(f"{'policy':<14}{'profit':>16}{'vs flat':>15}{'maxDD':>14}{'Sharpe':>8}")
    print(f"{'FLAT':<14}{flat['total_pnl_usd']:>16,.0f}{'':>15}{flat['max_drawdown_usd']:>14,.0f}"
          f"{flat['sharpe']:>8.2f}")
    for key, row in metrics["by_fraction"].items():
        dp = row["total_pnl_usd"] - flat["total_pnl_usd"]
        dd = row["max_drawdown_usd"] - flat["max_drawdown_usd"]
        flag = "  <-- BEATS BOTH" if (dp > 0 and dd > 0) else ""
        print(f"{'hedge ' + key:<14}{row['total_pnl_usd']:>16,.0f}{dp:>+15,.0f}"
              f"{row['max_drawdown_usd']:>14,.0f}{row['sharpe']:>8.2f}{flag}")
