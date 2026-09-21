"""Does an independent exit beat mirroring the client's close?

Run on the trades the Quant model actually selects, not on all flow -- the
policy only ever applies to copied trades.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import exit_policy, model_service, views

frame = model_service.load_scores("quant")
config = model_service.load_config("quant")
print(f"artefact: {len(frame):,} trades")

# The copied set: the top slice by model score, which is what the book trades.
cutoff = frame.groupby("day")["score"].transform(
    lambda s: s.quantile(1 - config.hedge_fraction))
copied = frame.loc[frame["score"] >= cutoff].copy()
print(f"copied at {config.hedge_fraction:.0%}: {len(copied):,} trades")

# Expected profit per trade: confidence above the base rate scaled by the
# account's typical outcome size. This is what the stop is measured against.
copied["expected_profit"] = ((copied["score"] - 0.5).clip(lower=0) * 2
                             * copied["pnl"].abs().median())
if "open_time" in copied.columns:
    copied["hold_hours"] = 1.0

print(f"\nP&L concentration on the copied set:")
losses = copied.loc[copied["pnl"] < 0, "pnl"]
print(f"  worst single trade      ${copied['pnl'].min():,.0f}")
print(f"  total losses            ${losses.sum():,.0f}")
print(f"  worst 1% of losses      ${losses.nsmallest(max(1, len(losses)//100)).sum():,.0f}"
      f"  ({losses.nsmallest(max(1, len(losses)//100)).sum() / losses.sum():.0%} of all loss)")

table = exit_policy.compare(copied)
print(f"\n{'policy':<10}{'stop':>6}{'target':>8}{'total P&L':>15}{'vs mirror':>14}"
      f"{'maxDD':>14}{'vs mirror':>13}{'Sharpe':>8}{'Calmar':>8}{'tail%':>8}")
for row in table.itertuples():
    stop = f"{row.stop_multiple:.1f}" if hasattr(row, "stop_multiple") and pd.notna(getattr(row, "stop_multiple", np.nan)) else "--"
    target = f"{row.target_multiple:.1f}" if hasattr(row, "target_multiple") and pd.notna(getattr(row, "target_multiple", np.nan)) else "--"
    flag = "  <-- DOMINATES" if row.dominates else ""
    print(f"{row.policy:<10}{stop:>6}{target:>8}{row.total_pnl:>15,.0f}"
          f"{row.pnl_vs_mirror:>+14,.0f}{row.max_drawdown:>14,.0f}"
          f"{row.dd_vs_mirror:>+13,.0f}{row.sharpe:>8.2f}{row.calmar:>8.1f}"
          f"{row.tail_loss_share:>8.0%}{flag}")

dominant = table.loc[table["dominates"]]
print(f"\n{len(dominant)} of {len(table)} policies beat mirroring on BOTH profit and drawdown.")
