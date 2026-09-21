import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import model_service as ms

meta = ms.artifact_meta("trading")
metrics = meta["metrics"]
flat = metrics["flat_bbook"]
print(f"rows {meta['rows']:,} | walk-forward ROC AUC {metrics['roc_auc']:.4f}\n")
print(f"{'policy':<16}{'profit':>16}{'vs flat':>15}{'maxDD':>14}{'vs flat':>13}{'Sharpe':>8}{'Calmar':>8}")
print(f"{'FLAT B-BOOK':<16}{flat['total_pnl_usd']:>16,.0f}{'':>15}"
      f"{flat['max_drawdown_usd']:>14,.0f}{'':>13}{flat['sharpe']:>8.2f}{flat['calmar']:>8.1f}")
for key, value in metrics["by_fraction"].items():
    dp = value["total_pnl_usd"] - flat["total_pnl_usd"]
    dd = value["max_drawdown_usd"] - flat["max_drawdown_usd"]
    flag = "   <-- BEATS FLAT ON BOTH" if (dp > 0 and dd > 0) else ""
    print(f"{'hedge ' + key:<16}{value['total_pnl_usd']:>16,.0f}{dp:>+15,.0f}"
          f"{value['max_drawdown_usd']:>14,.0f}{dd:>+13,.0f}"
          f"{value['sharpe']:>8.2f}{value['calmar']:>8.1f}{flag}")
