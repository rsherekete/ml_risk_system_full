"""Hedging the BOOK, not the client. The layer that addresses the drawdown.

WHAT THE EVIDENCE SAID

The firm's worst drawdown -- $31.3M over sixteen days in January 2026 -- was not
a whale and not bad luck. It was one position:

  * 84.8% of it was XAUUSD, across 12,879 separate accounts;
  * the per-account model was blind to it, scoring the top-1% winners at 0.5164
    against 0.4513 for everyone else, and flagging 0.5% of their rows;
  * and the net exposure built up monotonically for two weeks BEFORE the losses
    landed: -556 lots on 2 January, 36,496 by the 8th, 93,759 by the 19th.

A per-account model cannot see this by construction. Each of those 12,879
clients looked individually unremarkable; what was dangerous was that they
agreed. The risk was never in any client -- it was in the book, and the book was
observable the whole time.

THE REFRAME

The firm is not trying to pick winning clients. It is running a book. Its P&L is
a short position in whatever its clients are collectively long. Two separable
things follow:

  * THE EDGE is that clients lose on average -- about $250 per account-day,
    worth $1.096bn over two years. Hedging destroys edge, so hedge narrowly.
  * THE RISK is that clients sometimes agree with each other. That is a
    portfolio question with a portfolio answer: cap net exposure per instrument.

This module is the second layer. It leaves account selection alone.

HOW THE SIMULATION WORKS, AND WHAT IT ASSUMES

Net exposure is measured in USD at the START of each day, from positions already
open -- never from the day's outcome, which would be lookahead. Where net
exposure exceeds a limit, the excess is treated as passed to market: the firm
neither gains nor loses on that share of the day's P&L.

Two assumptions are stated rather than buried. Hedging is modelled as
frictionless, so real spread and slippage would reduce the benefit; and the
hedge is applied to the whole instrument pro rata rather than to specific
positions, which is how a desk would actually flatten a net book.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Instruments are the same risk regardless of the suffix a server gives them:
#: XAUUSD, XAUUSDe, XAUUSDmin and XAUUSD247 all move with gold. Treating them
#: separately is what let a single crowded position look like four small ones.
_SUFFIX = r"(MIN|MICRO|247|PRO|ECN|RAW|[EXZCM])+$"


def canonical_symbol(symbol: pd.Series) -> pd.Series:
    raw = symbol.astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    return raw.str.replace(_SUFFIX, "", regex=True)


def daily_book(trades: pd.DataFrame, notional_column: str = "notional_usd") -> pd.DataFrame:
    """Net and gross USD exposure per (day, instrument), plus realised P&L.

    A position contributes exposure on every day it is open -- opened, carried
    and closed alike -- because the firm carries that risk on each of them.
    Counting only closing days is what previously made a two-week build-up
    invisible until it resolved.
    """
    frame = trades.copy()
    frame["open_time"] = pd.to_datetime(frame["open_time"], errors="coerce")
    frame["close_time"] = pd.to_datetime(frame["close_time"], errors="coerce")
    frame = frame.loc[frame["open_time"].notna()]
    if frame.empty:
        return pd.DataFrame(columns=["day", "canonical", "net_usd", "gross_usd", "client_pnl"])

    frame["canonical"] = canonical_symbol(frame["symbol"])
    side = frame["cmd"].astype(str).str.lower()
    # Signed by direction: a client long is firm short, and the sign is the
    # entire point -- summing unsigned notional makes a balanced book look
    # identical to a one-way one.
    sign = np.where(side.str.startswith("b"), 1.0, -1.0)
    notional = pd.to_numeric(frame[notional_column], errors="coerce").fillna(0.0).abs()
    frame["signed_usd"] = sign * notional.to_numpy()
    frame["abs_usd"] = notional.to_numpy()

    as_of = frame["close_time"].max()
    start = frame["open_time"].dt.normalize()
    end = frame["close_time"].fillna(as_of).dt.normalize()
    span = ((end - start).dt.days + 1).clip(lower=1, upper=400).to_numpy(dtype="int64")

    index = np.repeat(np.arange(len(frame)), span)
    offsets = np.concatenate([np.arange(n) for n in span]) if len(span) else np.empty(0, "int64")
    expanded = pd.DataFrame({
        "day": start.to_numpy()[index] + offsets.astype("timedelta64[D]"),
        "canonical": frame["canonical"].to_numpy()[index],
        "signed_usd": frame["signed_usd"].to_numpy()[index],
        "abs_usd": frame["abs_usd"].to_numpy()[index],
    })
    exposure = expanded.groupby(["day", "canonical"], observed=True).agg(
        net_usd=("signed_usd", "sum"),
        gross_usd=("abs_usd", "sum"),
    ).reset_index()

    # P&L is realised on the CLOSING day only; spreading it would invent marks.
    realised = frame.loc[frame["close_time"].notna()].groupby(
        [frame["close_time"].dt.normalize().rename("day"), "canonical"],
        observed=True)["net_profit"].sum().reset_index(name="client_pnl")

    book = exposure.merge(realised, on=["day", "canonical"], how="left")
    book["client_pnl"] = book["client_pnl"].fillna(0.0)
    return book.sort_values(["day", "canonical"], kind="mergesort").reset_index(drop=True)


def simulate(book: pd.DataFrame, limit_usd: float,
             participation: float = 1.0) -> pd.DataFrame:
    """Firm P&L per day if net exposure above `limit_usd` were passed to market.

    `participation` scales how much of the excess is actually hedged, since a
    desk rarely flattens to the limit exactly.

    The decision uses YESTERDAY's closing exposure. Using today's would let the
    policy act on information it could not have had until the day was over,
    which is the same lookahead the rest of this project is careful to avoid.
    """
    frame = book.copy()
    frame["net_usd"] = pd.to_numeric(frame["net_usd"], errors="coerce").fillna(0.0)

    # Exposure as known at the START of the day.
    prior = frame.sort_values("day", kind="mergesort").groupby(
        "canonical", observed=True)["net_usd"].shift(1).fillna(0.0)
    frame["exposure_known"] = prior

    excess = (frame["exposure_known"].abs() - limit_usd).clip(lower=0.0)
    share = (excess / frame["exposure_known"].abs().replace(0, np.nan)).fillna(0.0)
    frame["hedged_share"] = (share * participation).clip(0.0, 1.0)

    # The firm earns the negative of client P&L on the share it keeps; the
    # hedged share is passed through and earns nothing either way.
    frame["firm_flat"] = -frame["client_pnl"]
    frame["firm_hedged"] = -frame["client_pnl"] * (1.0 - frame["hedged_share"])

    daily = frame.groupby("day", observed=True).agg(
        flat=("firm_flat", "sum"),
        model=("firm_hedged", "sum"),
        hedged_usd=("exposure_known", lambda s: float(np.abs(s).sum())),
    ).reset_index()
    for column in ("flat", "model"):
        daily[f"{column}_cum"] = daily[column].cumsum()
        daily[f"{column}_dd"] = daily[f"{column}_cum"] - daily[f"{column}_cum"].cummax()
    return daily


def simulate_anomaly(book: pd.DataFrame, percentile: float = 0.95,
                     window_days: int = 120, min_history: int = 40,
                     participation: float = 1.0) -> pd.DataFrame:
    """Hedge exposure only when it is ABNORMAL for that instrument.

    The static cap fails economically, and the backtest says so without
    ambiguity: every fixed limit from $5M to $500M destroys between $621M and
    $1.1bn of profit, because it binds on 374-706 days out of 730. The book is
    STRUCTURALLY directional -- clients sit permanently net long gold, and the
    firm's core edge is precisely being short that crowd. A cap that fires
    every day passes the edge to market along with the risk.

    What distinguished January 2026 was not that net exposure was large but
    that it was ABNORMAL: 93,759 lots against a typical book. So the trigger
    here is each instrument's own trailing distribution -- hedge only the
    excess above its `percentile` over the past `window_days`, computed from
    days strictly before the decision day. On a normal day nothing is hedged
    and the edge is untouched; on a crowding day the abnormal share passes to
    market.

    Same honesty rules as `simulate`: prior-day exposure only, frictionless
    hedge, pro rata across the instrument.
    """
    frame = book.copy()
    frame["net_usd"] = pd.to_numeric(frame["net_usd"], errors="coerce").fillna(0.0)
    frame = frame.sort_values(["canonical", "day"], kind="mergesort")

    grouped = frame.groupby("canonical", observed=True)["net_usd"]
    # The trigger level: this instrument's own |net| percentile over the past
    # window, shifted one day so today's exposure never sets today's threshold.
    abs_net = frame["net_usd"].abs()
    threshold = (abs_net.groupby(frame["canonical"], observed=True)
                 .transform(lambda s: s.rolling(window_days, min_periods=min_history)
                            .quantile(percentile).shift(1)))
    prior = grouped.shift(1).fillna(0.0)
    frame["exposure_known"] = prior

    # Hedge only the share ABOVE the trailing threshold. No threshold yet
    # (young instrument) means no hedge -- act only on evidence.
    excess = (prior.abs() - threshold).clip(lower=0.0)
    share = (excess / prior.abs().replace(0, np.nan)).fillna(0.0)
    frame["hedged_share"] = (share * participation).clip(0.0, 1.0).fillna(0.0)

    frame["firm_flat"] = -frame["client_pnl"]
    frame["firm_hedged"] = -frame["client_pnl"] * (1.0 - frame["hedged_share"])

    daily = frame.groupby("day", observed=True).agg(
        flat=("firm_flat", "sum"),
        model=("firm_hedged", "sum"),
        instruments_hedged=("hedged_share", lambda s: int((s > 0).sum())),
    ).reset_index().sort_values("day")
    for column in ("flat", "model"):
        daily[f"{column}_cum"] = daily[column].cumsum()
        daily[f"{column}_dd"] = daily[f"{column}_cum"] - daily[f"{column}_cum"].cummax()
    return daily


def sweep_anomaly(book: pd.DataFrame,
                  percentiles: tuple[float, ...] = (0.90, 0.95, 0.98, 0.995),
                  window_days: int = 120) -> pd.DataFrame:
    """The anomaly policy across trigger percentiles."""
    rows = []
    baseline = simulate(book, limit_usd=float("inf"))
    flat_profit = float(baseline["flat"].sum())
    flat_dd = float(baseline["flat_dd"].min())
    for percentile in percentiles:
        daily = simulate_anomaly(book, percentile=percentile, window_days=window_days)
        profit = float(daily["model"].sum())
        drawdown = float(daily["model_dd"].min())
        deviation = float(daily["model"].std())
        rows.append({
            "percentile": percentile,
            "profit": profit,
            "profit_delta": profit - flat_profit,
            "drawdown": drawdown,
            "drawdown_delta": drawdown - flat_dd,
            "sharpe": float(daily["model"].mean() / deviation * np.sqrt(252)) if deviation else 0.0,
            "calmar": float(profit / abs(drawdown)) if drawdown else float("inf"),
            "days_hedged": int((daily["model"] != daily["flat"]).sum()),
            "dominates": bool(profit > flat_profit and drawdown > flat_dd),
        })
    table = pd.DataFrame(rows)
    table.attrs["flat_profit"] = flat_profit
    table.attrs["flat_drawdown"] = flat_dd
    return table


def sweep(book: pd.DataFrame, limits: tuple[float, ...],
          participation: float = 1.0) -> pd.DataFrame:
    """Every limit, so the profit/drawdown trade-off is visible rather than argued."""
    rows = []
    baseline = simulate(book, limit_usd=float("inf"))
    flat_profit = float(baseline["flat"].sum())
    flat_dd = float(baseline["flat_dd"].min())
    for limit in limits:
        daily = simulate(book, limit_usd=limit, participation=participation)
        profit = float(daily["model"].sum())
        drawdown = float(daily["model_dd"].min())
        deviation = float(daily["model"].std())
        rows.append({
            "limit_usd": limit,
            "profit": profit,
            "profit_delta": profit - flat_profit,
            "drawdown": drawdown,
            "drawdown_delta": drawdown - flat_dd,
            "sharpe": float(daily["model"].mean() / deviation * np.sqrt(252)) if deviation else 0.0,
            "calmar": float(profit / abs(drawdown)) if drawdown else float("inf"),
            "days_hedged": int((daily["model"] != daily["flat"]).sum()),
            "dominates": bool(profit > flat_profit and drawdown > flat_dd),
        })
    table = pd.DataFrame(rows)
    table.attrs["flat_profit"] = flat_profit
    table.attrs["flat_drawdown"] = flat_dd
    return table
