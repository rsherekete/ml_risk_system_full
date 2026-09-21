"""Does an ACCOUNT WATCHLIST beat pure per-trade selection? Measured, not argued.

Two policies on identical walk-forward scores, same trades, same window:

  A  PER-TRADE (what runs today): each day, act on the top decile of that
     day's scores (copy) and the bottom decile (invert). Every trade judged
     on its own merit, no account gate.

  B  WATCHLIST + PER-TRADE (the proposal): maintain a copy list and an invert
     list of ACCOUNTS -- top/bottom 5% by mean score, ranked by win rate then
     trade count -- and only act on trades FROM those accounts that also clear
     the per-trade bar.

The watchlist is rebuilt WEEKLY from data strictly before that week. Ranking
accounts on the whole sample would leak the answer: an account's win rate over
the period trivially predicts its trades within the period, and B would win by
construction. That mistake produced an AUC of 1.0000 earlier in this project.

P&L convention: copying earns the client's P&L; inverting earns its negative.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import model_service as ms

frame = ms.load_scores(ms.VIEW_QUANT)
frame["day"] = pd.to_datetime(frame["day"])
frame = frame.loc[np.isfinite(frame["score"].to_numpy(dtype="float64"))]
frame["week"] = frame["day"].dt.to_period("W")
weeks = sorted(frame["week"].unique())
print(f"{len(frame):,} scored trades | {frame['day'].min().date()} -> "
      f"{frame['day'].max().date()} | {len(weeks)} weeks", flush=True)

TOP = 0.10          # per-trade decile, matching the live policy
ACCOUNT_SHARE = 0.05
MIN_TRADES = 20


def evaluate(daily: pd.Series, label: str) -> dict:
    curve = daily.cumsum()
    drawdown = float((curve - curve.cummax()).min())
    deviation = float(daily.std())
    return {
        "policy": label,
        "pnl": float(daily.sum()),
        "drawdown": drawdown,
        "sharpe": float(daily.mean() / deviation * np.sqrt(252)) if deviation else 0.0,
        "calmar": float(daily.sum() / abs(drawdown)) if drawdown else float("inf"),
        "days": int(len(daily)),
    }


rows_a, rows_b = [], []
watchlist_sizes = []

for index, week in enumerate(weeks):
    current = frame.loc[frame["week"] == week]
    if current.empty:
        continue

    # ---- A: per-trade deciles, computed per DAY (point-in-time by day) ----
    for day, group in current.groupby("day", observed=True):
        scores = group["score"].to_numpy(dtype="float64")
        if len(group) < 20:
            continue
        high = np.quantile(scores, 1 - TOP)
        low = np.quantile(scores, TOP)
        copied = group.loc[scores >= high, "pnl"].sum()
        inverted = -group.loc[scores <= low, "pnl"].sum()
        rows_a.append({"day": day, "pnl": float(copied + inverted)})

    # ---- B: watchlist from STRICTLY EARLIER weeks, then the same per-trade bar
    history = frame.loc[frame["week"] < week]
    if len(history) < 10_000:
        continue
    stats = history.groupby("account_key", observed=True).agg(
        trades=("score", "size"),
        mean_score=("score", "mean"),
        win_rate=("pnl", lambda s: float((s > 0).mean())),
    ).reset_index()
    stats = stats.loc[stats["trades"] >= MIN_TRADES]
    if stats.empty:
        continue

    # Top/bottom by mean score, then ranked as proposed: win rate, then volume.
    count = max(1, int(len(stats) * ACCOUNT_SHARE))
    copy_list = (stats.nlargest(count, "mean_score")
                 .sort_values(["win_rate", "trades"], ascending=False)["account_key"])
    invert_list = (stats.nsmallest(count, "mean_score")
                   .sort_values(["win_rate", "trades"], ascending=[True, False])["account_key"])
    copy_set, invert_set = set(copy_list), set(invert_list)
    watchlist_sizes.append(len(copy_set) + len(invert_set))

    for day, group in current.groupby("day", observed=True):
        scores = group["score"].to_numpy(dtype="float64")
        if len(group) < 20:
            continue
        high = np.quantile(scores, 1 - TOP)
        low = np.quantile(scores, TOP)
        on_copy = group["account_key"].isin(copy_set).to_numpy()
        on_invert = group["account_key"].isin(invert_set).to_numpy()
        copied = group.loc[(scores >= high) & on_copy, "pnl"].sum()
        inverted = -group.loc[(scores <= low) & on_invert, "pnl"].sum()
        rows_b.append({"day": day, "pnl": float(copied + inverted)})

daily_a = pd.DataFrame(rows_a).groupby("day")["pnl"].sum().sort_index()
daily_b = pd.DataFrame(rows_b).groupby("day")["pnl"].sum().sort_index()
# Compare over the SAME days -- B skips the first weeks while its watchlist
# has no history, and crediting A for days B never traded is not a comparison.
common = daily_a.index.intersection(daily_b.index)
daily_a, daily_b = daily_a.loc[common], daily_b.loc[common]

print(f"\ncomparable window: {len(common)} days "
      f"({common.min().date()} -> {common.max().date()})")
print(f"mean watchlist size: {np.mean(watchlist_sizes):,.0f} accounts\n")

results = [evaluate(daily_a, "A  per-trade only (live policy)"),
           evaluate(daily_b, "B  watchlist + per-trade")]
print(f"{'policy':<34}{'P&L':>16}{'maxDD':>14}{'Sharpe':>9}{'Calmar':>9}")
for row in results:
    print(f"{row['policy']:<34}{row['pnl']:>16,.0f}{row['drawdown']:>14,.0f}"
          f"{row['sharpe']:>9.2f}{row['calmar']:>9.1f}")

delta = results[1]["pnl"] - results[0]["pnl"]
dd_delta = results[1]["drawdown"] - results[0]["drawdown"]
print(f"\nB minus A: {delta:+,.0f} P&L | {dd_delta:+,.0f} drawdown")
if delta > 0 and dd_delta > 0:
    print("  -> the watchlist DOMINATES: more profit AND less drawdown")
elif delta > 0:
    print("  -> more profit, but deeper drawdown")
elif dd_delta > 0:
    print("  -> less drawdown, but less profit")
else:
    print("  -> the watchlist is worse on both axes")
