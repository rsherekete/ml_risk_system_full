"""Does a per-client markout-derived hold beat pure mirror exits? Path-true test.

THE PROPOSAL (the user's): each client's markout profile says at which horizon
their trades' post-entry move peaks. Close OUR copy at that horizon rather than
waiting for their exit -- they demonstrably hold past their own edge. For an
INVERTED trade the sign flips: the client's most ADVERSE horizon (their mean
markout minimum) is our most favourable, so that is our deadline.

Policy tested: exit at min(client's exit, open + hold(account, stance)) --
mirror remains the fallback; the deadline can only fire when the client holds
LONGER than their own historical edge.

HONESTY RULES
  * Profiles come from markout days STRICTLY BEFORE the study window, so an
    account's profile cannot contain the trades being evaluated.
  * Exit prices at the deadline come from the cached minute bars (the actual
    path), never from interpolation or the realised outcome.
  * Dollar conversion per trade reuses the trade's own pnl/price-move ratio --
    exact for that instrument as that venue traded it.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import model_service as ms

HORIZON_SECONDS = {"1m": 60, "5m": 300, "30m": 1800, "1h": 3600,
                   "4h": 14400, "1d": 86400, "3d": 259200}
MARKOUT_COLUMNS = [f"markout_{h}" for h in HORIZON_SECONDS]

study = pd.read_parquet(ms.SCRATCH / "quant_mae_mfe.parquet")
study["open_time"] = pd.to_datetime(study["open_time"])
study["close_time"] = pd.to_datetime(study["close_time"])
window_start = study["open_time"].min()
print(f"copy-book trades: {len(study):,} | window from {window_start.date()}")

bars = pd.read_parquet(ms.SCRATCH / "exit_study_bars.parquet")
bars["minute"] = pd.to_datetime(bars["minute"])
print(f"bars: {len(bars):,} across {bars['symbol'].nunique()} symbols")

markouts = pd.read_parquet(ms.SCRATCH / "markout_all_servers.parquet")
markouts["day"] = pd.to_datetime(markouts["day"])
# STRICTLY BEFORE the study window -- the profile may not see the answer.
history = markouts.loc[markouts["day"] < window_start]
print(f"markout history rows before window: {len(history):,} "
      f"({history['account_key'].nunique():,} accounts)")

profile = history.groupby("account_key")[MARKOUT_COLUMNS].mean()
# Copy deadline: horizon of the client's PEAK mean markout, held only when
# that peak is positive (a client with no positive edge anywhere gets no cap).
best_col = profile[MARKOUT_COLUMNS].idxmax(axis=1)
best_val = profile[MARKOUT_COLUMNS].max(axis=1)
copy_hold = best_col.map(lambda c: HORIZON_SECONDS[c.replace("markout_", "")])
copy_hold = copy_hold.where(best_val > 0)
# Invert deadline: horizon of the client's WORST mean markout (our best).
worst_col = profile[MARKOUT_COLUMNS].idxmin(axis=1)
worst_val = profile[MARKOUT_COLUMNS].min(axis=1)
invert_hold = worst_col.map(lambda c: HORIZON_SECONDS[c.replace("markout_", "")])
invert_hold = invert_hold.where(worst_val < 0)
print(f"profiles: {copy_hold.notna().sum():,} copy holds, "
      f"{invert_hold.notna().sum():,} invert holds")
print("copy hold distribution:", copy_hold.dropna().map(
    {v: k for k, v in HORIZON_SECONDS.items()}).value_counts().to_dict())

# --- price at deadline, from the actual path -------------------------------
bar_index = {}
for symbol, group in bars.groupby("symbol", observed=True):
    group = group.sort_values("minute")
    bar_index[symbol] = (group["minute"].to_numpy(), group["close"].to_numpy())


def price_at(symbol: str, when: pd.Timestamp) -> float | None:
    entry = bar_index.get(symbol)
    if entry is None:
        return None
    minutes, closes = entry
    position = minutes.searchsorted(np.datetime64(when), side="right") - 1
    if position < 0:
        return None
    # A bar more than 5 minutes older than the deadline is not "the price
    # then" -- gap periods return None and the trade falls back to mirror.
    if (when - pd.Timestamp(minutes[position])).total_seconds() > 300:
        return None
    return float(closes[position])


def evaluate(trades: pd.DataFrame, hold: pd.Series, invert: bool) -> dict:
    """Total/DD for mirror vs markout-capped exits on the same trades."""
    direction = trades["direction"].to_numpy() * (-1 if invert else 1)
    pnl_mirror = trades["pnl"].to_numpy() * (-1 if invert else 1)
    move = ((trades["close_price"] - trades["open_price"]).to_numpy()
            * trades["direction"].to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        per_unit = np.where(np.abs(move) > 1e-12,
                            np.abs(trades["pnl"].to_numpy() / move), np.nan)
    hold_seconds = trades["account_key"].map(hold).to_numpy()

    pnl_policy = pnl_mirror.copy()
    fired = 0
    for i in range(len(trades)):
        h = hold_seconds[i]
        if not np.isfinite(h) or not np.isfinite(per_unit[i]):
            continue
        deadline = trades["open_time"].iat[i] + pd.Timedelta(seconds=float(h))
        if deadline >= trades["close_time"].iat[i]:
            continue                    # client left first -- mirror as usual
        exit_price = price_at(str(trades["symbol"].iat[i]), deadline)
        if exit_price is None:
            continue
        our_move = (exit_price - trades["open_price"].iat[i]) * direction[i]
        pnl_policy[i] = our_move * per_unit[i]
        fired += 1

    day = trades["open_time"].dt.normalize()
    def shape(values):
        daily = pd.Series(values, index=day).groupby(level=0).sum().sort_index()
        curve = daily.cumsum()
        return float(values.sum()), float((curve - curve.cummax()).min())
    mirror_pnl, mirror_dd = shape(pnl_mirror)
    policy_pnl, policy_dd = shape(pnl_policy)
    return {"trades": len(trades), "deadline_fired": fired,
            "mirror_pnl": mirror_pnl, "mirror_dd": mirror_dd,
            "policy_pnl": policy_pnl, "policy_dd": policy_dd}


# --- copy book -------------------------------------------------------------
result = evaluate(study, copy_hold, invert=False)
print(f"\nCOPY BOOK ({result['trades']:,} trades, deadline fired on "
      f"{result['deadline_fired']:,}):")
print(f"  mirror        : {result['mirror_pnl']:>14,.0f}  maxDD {result['mirror_dd']:>12,.0f}")
print(f"  markout hold  : {result['policy_pnl']:>14,.0f}  maxDD {result['policy_dd']:>12,.0f}")
delta = result["policy_pnl"] - result["mirror_pnl"]
dd_delta = result["policy_dd"] - result["mirror_dd"]
print(f"  delta         : {delta:>+14,.0f}        DD {dd_delta:>+12,.0f}"
      + ("   <== DOMINATES" if delta > 0 and dd_delta > 0 else ""))

# --- invert book: bottom decile of the same artifact, same symbols ---------
frame = ms.load_scores(ms.VIEW_QUANT)
frame["open_time"] = pd.to_datetime(frame["open_time"])
frame["close_time"] = pd.to_datetime(frame["close_time"])
score = pd.to_numeric(frame["score"], errors="coerce")
low_cut = float(score.quantile(0.10))
inverts = frame.loc[(score <= low_cut)
                    & frame["symbol"].isin(bar_index.keys())
                    & frame["close_time"].notna()].copy()
print(f"\nINVERT BOOK: {len(inverts):,} bottom-decile trades on covered symbols")
result_inv = evaluate(inverts, invert_hold, invert=True)
print(f"  mirror-invert : {result_inv['mirror_pnl']:>14,.0f}  maxDD {result_inv['mirror_dd']:>12,.0f}")
print(f"  markout hold  : {result_inv['policy_pnl']:>14,.0f}  maxDD {result_inv['policy_dd']:>12,.0f} "
      f"(fired {result_inv['deadline_fired']:,})")
delta_i = result_inv["policy_pnl"] - result_inv["mirror_pnl"]
dd_i = result_inv["policy_dd"] - result_inv["mirror_dd"]
print(f"  delta         : {delta_i:>+14,.0f}        DD {dd_i:>+12,.0f}"
      + ("   <== DOMINATES" if delta_i > 0 and dd_i > 0 else ""))
print("\nSTUDY COMPLETE")
