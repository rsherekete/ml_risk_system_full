"""Retrain both models on the full two-year warehouse.

Also answers the warm-start question by MEASURING it rather than assuming:
continuing from the previous booster cost 0.16 AUC on a daily cadence earlier in
this project. A weekly cadence over two years is a different regime, so both are
run and compared.
"""
import sys
import time

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import model_service as ms


def run(view, config, label):
    print(f"\n=== {view}: {label} ===", flush=True)
    ms.start_training(view, config)
    last, t0 = "", time.time()
    while True:
        state = ms.job_state(view)
        if state.message != last:
            print(f"  [{state.progress:5.1%}] {state.message}", flush=True)
            last = state.message
        if state.status in ("done", "error"):
            break
        time.sleep(5)
    if state.status == "error":
        print("  FAILED:", state.message)
        print("\n".join(state.log[-4:]))
        return None

    meta = ms.artifact_meta(view)
    metrics = meta["metrics"]
    flat = metrics["flat_bbook"]
    print(f"  {meta['rows']:,} rows | AUC {metrics['roc_auc']:.4f} "
          f"| {time.time()-t0:.0f}s")
    print(f"  {'policy':<14}{'profit':>16}{'vs flat':>15}{'maxDD':>14}{'Sharpe':>8}")
    print(f"  {'FLAT':<14}{flat['total_pnl_usd']:>16,.0f}{'':>15}"
          f"{flat['max_drawdown_usd']:>14,.0f}{flat['sharpe']:>8.2f}")
    for key, row in metrics["by_fraction"].items():
        dp = row["total_pnl_usd"] - flat["total_pnl_usd"]
        dd = row["max_drawdown_usd"] - flat["max_drawdown_usd"]
        flag = "  <-- BEATS BOTH" if (dp > 0 and dd > 0) else ""
        print(f"  {'hedge ' + key:<14}{row['total_pnl_usd']:>16,.0f}{dp:>+15,.0f}"
              f"{row['max_drawdown_usd']:>14,.0f}{row['sharpe']:>8.2f}{flag}")
    return metrics["roc_auc"]


config = ms.load_config(ms.VIEW_TRADING)
config.history_days = 730
config.use_exposure_days = True
config.refit_cadence_days = 7
config.warm_start = False
ms.save_config(ms.VIEW_TRADING, config)
cold = run(ms.VIEW_TRADING, config, "2 years, weekly refit, COLD start")

import dataclasses
warm_config = dataclasses.replace(config, warm_start=True)
warm = run(ms.VIEW_TRADING, warm_config, "2 years, weekly refit, WARM start")

if cold and warm:
    print(f"\nwarm start vs cold: {warm - cold:+.4f} AUC "
          f"({'better' if warm > cold else 'WORSE -- keeping cold'})")
    # Keep whichever actually won.
    ms.save_config(ms.VIEW_TRADING, warm_config if warm > cold else config)
    if cold >= warm:
        run(ms.VIEW_TRADING, config, "restoring the cold-start artefact")

quant = ms.load_config(ms.VIEW_QUANT)
quant.history_days = 730
quant.refit_cadence_days = 7
quant.warm_start = False
ms.save_config(ms.VIEW_QUANT, quant)
run(ms.VIEW_QUANT, quant, "2 years, weekly refit")
