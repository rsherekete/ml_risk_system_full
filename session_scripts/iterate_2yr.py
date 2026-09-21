"""Fast 90-day iteration of BOTH models with tonight's changes, sequentially.

What is new in this cycle and needs validating before committing 9+ hours:

  TRADING: economic sample weights (dollar-weighted loss) + 9 cashflow
  features + six servers + repaired MT5 P&L.
  QUANT: 9 cashflow features + sample-fraction recording. At 90 days the
  history fits UNDER the 25M row cap, so this runs at FULL density -- both a
  faster iteration and a direct check on what sampling costs the 2-year run.

Sequential because two of these do not fit in RAM together; that lesson has
been paid for twice.
"""
import sys
import time

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import model_service as ms


def run(view: str, setup) -> dict | None:
    config = ms.load_config(view)
    setup(config)
    ms.save_config(view, config)
    print(f"\n=== {view}: 2-year run (fingerprint {config.fingerprint()}) ===", flush=True)
    ms.start_training(view, config)
    last, started = "", time.time()
    while True:
        state = ms.job_state(view)
        if state.message != last:
            print(f"  [{state.progress:5.1%}] {state.message}", flush=True)
            last = state.message
        if state.status in ("done", "error"):
            break
        time.sleep(10)
    print(f"STATUS: {state.status} | {(time.time() - started) / 60:.1f} min", flush=True)
    if state.status == "error":
        print("\n".join(state.log[-6:]))
        return None
    return ms.artifact_meta(view)


def show(meta: dict, label: str):
    if not meta:
        return
    metrics = meta["metrics"]
    flat = metrics["flat_bbook"]
    fraction = metrics.get("sample_fraction")
    print(f"\n--- {label}: rows {meta['rows']:,} | AUC {metrics.get('roc_auc') or float('nan'):.4f}"
          + (f" | sample fraction {fraction:.3f}" if fraction else ""))
    print(f"{'policy':<14}{'profit':>16}{'vs flat':>15}{'maxDD':>14}{'vs flat':>13}{'Sharpe':>8}")
    print(f"{'FLAT':<14}{flat['total_pnl_usd']:>16,.0f}{'':>15}"
          f"{flat['max_drawdown_usd']:>14,.0f}{'':>13}{flat['sharpe']:>8.2f}")
    for key, row in metrics["by_fraction"].items():
        dp = row["total_pnl_usd"] - flat["total_pnl_usd"]
        dd = row["max_drawdown_usd"] - flat["max_drawdown_usd"]
        flag = "  <== BEATS BOTH" if (dp > 0 and dd > 0) else ""
        print(f"{'hedge ' + key:<14}{row['total_pnl_usd']:>16,.0f}{dp:>+15,.0f}"
              f"{row['max_drawdown_usd']:>14,.0f}{dd:>+13,.0f}{row['sharpe']:>8.2f}{flag}")


def fast(config):
    """The efficiency settings, applied identically to both models.

    150 trees at lr 0.1 is the standard equivalent-budget swap for 300 at
    0.05; max_bin=63 and the 3M-row per-fit cap live in the training code.
    Together they turn a 7-hour walk-forward into tens of minutes, and the
    90-day AUC printed by this run is the check that accuracy survived.
    """
    config.n_estimators = 150
    config.learning_rate = 0.1
    config.max_fit_rows = 3_000_000
    config.warm_start = False
    config.refit_cadence_days = 7


def trading_setup(config):
    fast(config)
    config.history_days = 730
    config.min_train_days = 60
    config.max_train_days = 365
    config.use_exposure_days = True
    config.economic_weights = True


def quant_setup(config):
    fast(config)
    config.history_days = 730
    config.min_train_days = 60


show(run(ms.VIEW_TRADING, trading_setup), "TRADING 730d (weighted + cashflow)")
show(run(ms.VIEW_QUANT, quant_setup), "QUANT 730d (sampled + cashflow)")

# Feature importances: did the cashflow features actually earn their place?
meta = ms.artifact_meta(ms.VIEW_QUANT) or {}
importance = (meta.get("metrics") or {}).get("feature_importance") or {}
if importance:
    print("\n--- quant feature importance (cashflow features marked) ---")
    cash = {"deposits_to_date", "withdrawals_to_date", "net_funding_to_date",
            "deposit_count_to_date", "withdrawal_count_to_date",
            "days_since_deposit", "days_since_withdrawal",
            "withdrawal_ratio", "funding_churn"}
    for name, value in list(importance.items())[:25]:
        mark = "  <== CASHFLOW" if name in cash else ""
        print(f"  {name:<26}{value:>8,}{mark}")
print("\n2-YEAR ITERATION COMPLETE", flush=True)

