"""Retrain the trade-level (Quant) model on two years.

The Quant artefact covers 71 days while Trading covers 670, so every Quant
screen has been quoting a two-month sample against the two-year sample the rest
of the site uses. The two views are not comparable until this matches.
"""
import sys
import time

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import model_service as ms

config = ms.load_config(ms.VIEW_QUANT)
config.history_days = 730
config.refit_cadence_days = 7
config.warm_start = False
ms.save_config(ms.VIEW_QUANT, config)

print(f"config: history={config.history_days}d refit={config.refit_cadence_days}d "
      f"max_train={getattr(config, 'max_train_days', None)}d "
      f"max_rows={getattr(config, 'max_trade_rows', None):,}", flush=True)

ms.start_training(ms.VIEW_QUANT, config)
last, started = "", time.time()
while True:
    state = ms.job_state(ms.VIEW_QUANT)
    if state.message != last:
        print(f"  [{state.progress:5.1%}] {state.message}", flush=True)
        last = state.message
    if state.status in ("done", "error"):
        break
    time.sleep(5)

print(f"\nSTATUS: {state.status} | {state.message} | {(time.time() - started) / 60:.1f} min")
if state.status == "error":
    print("\n".join(state.log[-6:]))
    raise SystemExit(1)

meta = ms.artifact_meta(ms.VIEW_QUANT)
metrics = meta["metrics"]
print(f"\nrows {meta['rows']:,} | walk-forward ROC AUC {metrics.get('roc_auc', float('nan')):.4f}")
flat = metrics.get("flat_bbook")
if flat:
    print(f"{'policy':<14}{'profit':>18}{'vs flat':>16}{'maxDD':>16}{'vs flat':>15}{'Sharpe':>8}")
    print(f"{'FLAT B-BOOK':<14}{flat['total_pnl_usd']:>18,.0f}{'':>16}"
          f"{flat['max_drawdown_usd']:>16,.0f}{'':>15}{flat['sharpe']:>8.2f}")
    for key, row in metrics.get("by_fraction", {}).items():
        dp = row["total_pnl_usd"] - flat["total_pnl_usd"]
        dd = row["max_drawdown_usd"] - flat["max_drawdown_usd"]
        flag = "  <-- BEATS BOTH" if (dp > 0 and dd > 0) else ""
        print(f"{'hedge ' + key:<14}{row['total_pnl_usd']:>18,.0f}{dp:>+16,.0f}"
              f"{row['max_drawdown_usd']:>16,.0f}{dd:>+15,.0f}{row['sharpe']:>8.2f}{flag}")
