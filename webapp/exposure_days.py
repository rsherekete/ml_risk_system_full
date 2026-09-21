"""Active days redefined as EXPOSURE days.

THE CHANGE, AND WHY IT IS MORE CORRECT

Until now an "active day" meant a day on which an account closed a trade. That
definition has a hole a risk desk would immediately object to: a client who
opened a large position on Monday and closes it on Friday is carrying risk all
week, yet appeared only on Friday. Every intervening day the firm was exposed to
that position and the model was blind to it.

An account is now active on any day it was EXPOSED:

* it opened a position that day, or
* it closed a position that day, or
* it held a position open across that day, or
* it holds a position that is still open now.

This matters differently in the three places it is used:

* **Features** -- carrying days now exist, so holding behaviour (duration,
  overnight risk, weekend gaps) becomes visible instead of being collapsed into
  the closing day.
* **Targets** -- "the next active day" now means the next day the account is
  exposed, which is the day a routing decision actually applies to.
* **Risk** -- exposure on a given day now includes positions opened earlier and
  not yet closed, which is what the desk is actually carrying. The previous
  definition systematically understated open interest.

COST

Expanding every trade to the days it spans is an interval explosion, so it is
done by integer day arithmetic on numpy arrays and capped: a position open for
years contributes its whole span, but `MAX_SPAN_DAYS` guards against a corrupt
timestamp generating millions of rows from a single trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: A single position may not contribute more than this many exposure days.
#: Real positions do run for months; a value beyond this is a data error, and
#: without the cap one bad timestamp would explode the frame.
MAX_SPAN_DAYS = 400


def exposure_days(trades: pd.DataFrame, as_of: pd.Timestamp | None = None,
                  open_positions: pd.DataFrame | None = None,
                  window_start: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per (account, day) the account was exposed.

    `trades` needs ``account_key``, ``open_time``, ``close_time`` and
    ``net_profit``; `open_positions` (optional) contributes still-running
    positions, which by definition have no close time and would otherwise be
    invisible.

    Returns the exposure calendar with, per day: whether the account opened,
    closed, or merely carried, how many positions were live, and the P&L
    realised that day.
    """
    if trades.empty:
        return pd.DataFrame(columns=["account_key", "day", "opened", "closed",
                                     "carried", "live_positions", "realised_pnl"])

    frame = trades.copy()
    frame["open_time"] = pd.to_datetime(frame["open_time"], errors="coerce")
    frame["close_time"] = pd.to_datetime(frame["close_time"], errors="coerce")
    as_of = pd.Timestamp(as_of) if as_of is not None else frame["close_time"].max()

    # Still-open trades run to the as-of date rather than being dropped.
    close = frame["close_time"].fillna(as_of)
    frame = frame.loc[frame["open_time"].notna()]
    if frame.empty:
        return pd.DataFrame(columns=["account_key", "day", "opened", "closed",
                                     "carried", "live_positions", "realised_pnl"])

    start_day = frame["open_time"].dt.normalize()
    end_day = close.loc[frame.index].dt.normalize()

    # Clip the expansion to the requested window. The warehouse selects on
    # CLOSE time, so a position opened years earlier but closed inside the
    # window arrives with it -- mt4_live01's history spans 2,658 days against a
    # 730-day request. Expanding those over their full life manufactures
    # exposure days BEFORE the window starts, which is both wrong and the reason
    # that server showed 88 exposure days per account against mt5_live01's 16.
    if window_start is not None:
        floor = pd.Timestamp(window_start).normalize()
        start_day = start_day.clip(lower=floor)
        # A position that also closed before the window contributes nothing.
        keep = end_day >= floor
        frame, start_day, end_day = frame.loc[keep], start_day.loc[keep], end_day.loc[keep]
        close = close.loc[frame.index]
        if frame.empty:
            return pd.DataFrame(columns=["account_key", "day", "opened", "closed",
                                         "carried", "live_positions", "realised_pnl"])

    span = ((end_day - start_day).dt.days + 1).clip(lower=1, upper=MAX_SPAN_DAYS)

    # Repeat each trade once per day it spanned, then offset by position within
    # the run. Integer arithmetic on numpy arrays -- a per-trade date_range would
    # be orders of magnitude slower at this scale.
    counts = span.to_numpy(dtype="int64")
    repeat_index = np.repeat(np.arange(len(frame)), counts)
    offsets = np.concatenate([np.arange(n) for n in counts]) if len(counts) else np.empty(0, dtype="int64")

    expanded = pd.DataFrame({
        "account_key": frame["account_key"].to_numpy()[repeat_index],
        "day": start_day.to_numpy()[repeat_index] + offsets.astype("timedelta64[D]"),
    })
    # A trade whose start was clipped to the window floor did NOT open on that
    # day -- it was already running. Marking it as an open would invent activity
    # the account never performed.
    truly_opened = (frame["open_time"].dt.normalize() == start_day).to_numpy()
    expanded["is_open_day"] = (offsets == 0) & np.repeat(truly_opened, counts)
    last_offset = np.repeat(counts - 1, counts)
    expanded["is_close_day"] = (offsets == last_offset) & \
        np.repeat(frame["close_time"].notna().to_numpy(), counts)
    # P&L is realised on the CLOSING day only -- spreading it across the holding
    # period would invent daily marks the warehouse never recorded.
    realised = np.where(expanded["is_close_day"],
                        np.repeat(frame["net_profit"].fillna(0).to_numpy(), counts), 0.0)
    expanded["realised_pnl"] = realised

    calendar = expanded.groupby(["account_key", "day"], observed=True).agg(
        opened=("is_open_day", "sum"),
        closed=("is_close_day", "sum"),
        live_positions=("day", "size"),
        realised_pnl=("realised_pnl", "sum"),
    ).reset_index()
    # A carrying day is one with exposure but no transaction -- previously
    # invisible, and the whole point of this redefinition.
    calendar["carried"] = ((calendar["opened"] == 0) & (calendar["closed"] == 0)).astype(int)
    return calendar


def merge_open_positions(calendar: pd.DataFrame, positions: pd.DataFrame,
                         as_of: pd.Timestamp) -> pd.DataFrame:
    """Fold currently-open positions into today's exposure row.

    Open positions come from the broker's own position table, so they include
    everything currently at risk -- including positions opened before any
    history window, which a trade-derived calendar can never see.
    """
    if positions is None or positions.empty:
        return calendar
    today = pd.Timestamp(as_of).normalize()
    live = positions.groupby("account_key", observed=True).agg(
        live_now=("volume_lots", "size"),
        open_lots=("volume_lots", lambda s: float(s.abs().sum())),
    ).reset_index()
    live["day"] = today

    merged = calendar.merge(live, on=["account_key", "day"], how="outer")
    for column, default in (("opened", 0), ("closed", 0), ("live_positions", 0),
                            ("realised_pnl", 0.0), ("carried", 0)):
        merged[column] = merged[column].fillna(default)
    merged["live_now"] = merged["live_now"].fillna(0)
    merged["open_lots"] = merged["open_lots"].fillna(0.0)
    # An account with a live position but no trade today is exposed and carrying.
    merged.loc[(merged["live_now"] > 0) & (merged["opened"] == 0) & (merged["closed"] == 0),
               "carried"] = 1
    return merged


def summarise(calendar: pd.DataFrame) -> dict:
    """Headline numbers, including how much the redefinition adds."""
    if calendar.empty:
        return {}
    carrying = int(calendar["carried"].sum())
    return {
        "account_days": int(len(calendar)),
        "accounts": int(calendar["account_key"].nunique()),
        "days": int(calendar["day"].nunique()),
        "transacting_days": int((calendar["carried"] == 0).sum()),
        "carrying_days": carrying,
        # The share of exposure that the old close-day definition simply missed.
        "carrying_share": float(carrying / len(calendar)) if len(calendar) else 0.0,
        "mean_live_positions": float(calendar["live_positions"].mean()),
    }
