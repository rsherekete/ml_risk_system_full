"""When does a copied trade close? Several answers, measured against each other.

A GAP THAT WAS NOT PREVIOUSLY STATED

Every backtest so far assumed the copy book mirrors the client's entire round
trip -- in at their entry, out at their exit. That is a real policy and it is
what the reported figures measure, but it is only one option, and it inherits
the client's worst habit: holding losers. The single largest loss in the signal
list comes from exactly that.

An independent exit changes the distribution rather than the direction: it
cannot make a bad signal good, but it can stop one trade destroying a month.

THE POLICIES

* **mirror**    -- close when the client closes. The incumbent, and the baseline
                   every other policy must beat.
* **stop**      -- cap the loss at a multiple of the trade's expected profit.
                   The proposal being tested: if a trade is worth $100 in
                   expectation, refuse to lose more than $100 on it.
* **target**    -- take profit at a multiple of expectation, let losers run to
                   the client's exit.
* **bracket**   -- both a stop and a target.
* **time**      -- close after a fixed holding period regardless.

WHAT THIS CAN AND CANNOT MEASURE

The warehouse stores entry and exit, not the path between them, so an intrabar
high/low is unavailable. A stop is therefore evaluated against the trade's
REALISED outcome: it is treated as hit when the final loss exceeds the stop.
That is a CONSERVATIVE approximation of a target (a trade that touched the
target and came back is not counted as a win) and an OPTIMISTIC one for a stop
(a trade that dipped through the stop and recovered is not counted as stopped).
Both directions are stated because the second one flatters the result, and a
tick-level replay would be needed to settle it properly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

POLICIES = ("mirror", "stop", "target", "bracket", "time")


def apply_policy(trades: pd.DataFrame, policy: str = "mirror",
                 stop_multiple: float = 1.0, target_multiple: float = 2.0,
                 max_hold_hours: float = 24.0,
                 expected_column: str = "expected_profit") -> pd.Series:
    """Realised P&L per trade under an exit policy.

    `expected_profit` is the model's own estimate for that trade and is what the
    stop is scaled from -- a large expected winner is allowed a proportionally
    larger loss, which is what makes this different from a flat dollar stop.
    """
    pnl = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
    if policy == "mirror":
        return pnl

    # `DataFrame.get` on a missing column returns None, and `pd.to_numeric(None)`
    # yields a scalar NaN rather than a Series -- which then fails on `.isna()`.
    # Check for the column before converting instead of after.
    if expected_column in trades.columns:
        expected = pd.to_numeric(trades[expected_column], errors="coerce")
    else:
        expected = None
    if expected is None or expected.isna().all():
        # No per-trade expectation available: fall back to the account's typical
        # trade size so the policy still has a scale to work from.
        expected = pnl.abs().rolling(50, min_periods=1).median()
    expected = expected.abs().replace(0, np.nan)

    stop_level = -(expected * stop_multiple)
    target_level = expected * target_multiple

    # PATH-AWARE where the excursions are known. `mae` (worst adverse move) and
    # `mfe` (best favourable move) come from replaying minute bars over the
    # trade's life, so a stop can be tested against what the price ACTUALLY did
    # rather than only where it finished.
    #
    # This distinction is the whole validity of the exercise. Applied to the
    # realised outcome alone, a stop truncates losses and touches nothing else,
    # so it raises total P&L BY CONSTRUCTION -- it cannot lose, and a table of
    # such results measures the arithmetic rather than the policy. With
    # excursions, a trade that dipped through the stop and recovered is correctly
    # counted as stopped, which is where the cost of stops actually shows up.
    has_path = "mae" in trades.columns and "mfe" in trades.columns
    if has_path:
        mae = pd.to_numeric(trades["mae"], errors="coerce")
        mfe = pd.to_numeric(trades["mfe"], errors="coerce")
        stop_hit = mae <= stop_level
        target_hit = mfe >= target_level

    if policy == "stop":
        if has_path:
            return pnl.where(~stop_hit, stop_level).fillna(pnl)
        # A loss worse than the stop is truncated at the stop; winners unchanged.
        return pnl.where(pnl >= stop_level, stop_level).fillna(pnl)
    if policy == "target":
        if has_path:
            return pnl.where(~target_hit, target_level).fillna(pnl)
        return pnl.where(pnl <= target_level, target_level).fillna(pnl)
    if policy == "bracket":
        if has_path:
            # Whichever level the path reached first would have closed the trade.
            # Without intrabar ordering the stop is assumed to bind, which is the
            # conservative reading rather than the flattering one.
            out = pnl.where(~target_hit, target_level)
            return out.where(~stop_hit, stop_level).fillna(pnl)
        capped = pnl.where(pnl >= stop_level, stop_level)
        return capped.where(capped <= target_level, target_level).fillna(pnl)
    if policy == "time":
        # Trades held beyond the limit are closed at the limit. Without a price
        # path the outcome is unknown, so it is scaled by the fraction of the
        # holding period that elapsed -- crude, and flagged as such.
        if "hold_hours" not in trades.columns:
            return pnl
        hold = pd.to_numeric(trades["hold_hours"], errors="coerce").fillna(0.0)
        fraction = (max_hold_hours / hold.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)
        return pnl * fraction
    return pnl


def evaluate(trades: pd.DataFrame, policy: str, **params) -> dict:
    """Profit, drawdown and tail behaviour for one policy."""
    realised = apply_policy(trades, policy, **params)
    days = pd.to_datetime(trades["day"]).dt.normalize()
    # The copy book earns what the client earns on mirrored trades.
    daily = realised.groupby(days).sum().sort_index()
    curve = daily.cumsum()
    drawdown = float((curve - curve.cummax()).min())
    deviation = float(daily.std())
    losses = realised[realised < 0]

    return {
        "policy": policy,
        "total_pnl": float(realised.sum()),
        "max_drawdown": drawdown,
        "sharpe": float(daily.mean() / deviation * np.sqrt(252)) if deviation else 0.0,
        "calmar": float(realised.sum() / abs(drawdown)) if drawdown else float("inf"),
        "win_rate": float((realised > 0).mean()),
        "worst_trade": float(realised.min()),
        # The number this exists to move: how much of total loss comes from the
        # worst 1% of trades. A stop should cut it sharply.
        "tail_loss_share": float(losses.nsmallest(max(1, len(losses) // 100)).sum()
                                 / losses.sum()) if len(losses) else 0.0,
        "trades": int(len(realised)),
        # Whether this row means anything. Without excursions, a stop or target
        # applied to the realised outcome can only move P&L in its own favour,
        # so the number is structurally biased and must be labelled as such
        # wherever it is displayed. The TIME policy never uses the path at all
        # -- it scales realised P&L linearly by elapsed holding time, which is
        # an approximation whatever else the frame carries -- so it is never
        # marked path-aware and always flagged biased.
        "path_aware": bool(policy != "time"
                           and "mae" in trades.columns and "mfe" in trades.columns),
        "biased": bool(policy == "time"
                       or (policy in ("stop", "target", "bracket")
                           and not ("mae" in trades.columns and "mfe" in trades.columns))),
        **params,
    }


def compare(trades: pd.DataFrame,
            stop_multiples: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0),
            target_multiples: tuple[float, ...] = (2.0, 3.0)) -> pd.DataFrame:
    """Every policy and parameter, ranked so the trade-offs are visible.

    Reported together rather than reduced to a winner: the best policy by total
    profit and the best by drawdown are usually different, and which one a desk
    wants depends on its constraint, not on this function's opinion.
    """
    rows = [evaluate(trades, "mirror")]
    for multiple in stop_multiples:
        rows.append(evaluate(trades, "stop", stop_multiple=multiple))
    for stop in stop_multiples:
        for target in target_multiples:
            rows.append(evaluate(trades, "bracket",
                                 stop_multiple=stop, target_multiple=target))
    for target in target_multiples:
        rows.append(evaluate(trades, "target", target_multiple=target))
    if "hold_hours" in trades.columns:
        for hours in (1.0, 4.0, 24.0):
            rows.append(evaluate(trades, "time", max_hold_hours=hours))

    table = pd.DataFrame(rows)
    baseline = table.loc[table["policy"] == "mirror"].iloc[0]
    table["pnl_vs_mirror"] = table["total_pnl"] - baseline["total_pnl"]
    table["dd_vs_mirror"] = table["max_drawdown"] - baseline["max_drawdown"]
    # Dominant means better on BOTH axes -- the only unambiguous improvement.
    table["dominates"] = (table["pnl_vs_mirror"] > 0) & (table["dd_vs_mirror"] > 0)
    return table.sort_values("calmar", ascending=False).reset_index(drop=True)
