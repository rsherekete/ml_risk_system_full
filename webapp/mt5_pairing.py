"""Rebuild MT5 round-trip trades from deal rows.

MT5 is deal-based: one position produces an ENTRY row (open_time and open_price
set, no outcome) and an EXIT row (close_time, close_price and net_profit set,
but `open_time` NULL). A trade-level router has to decide at ENTRY, so without
rejoining the halves MT5 cannot be modelled at all -- which is why the Quant
book originally covered only MT4 and reported a $71.1M baseline against the
Trading view's $117.4M.

The two halves join on `order`, and the join is clean: 99.9% of closed rows find
their entry, recovering $48.18M of $48.55M client P&L (99.2%).

TWO TRAPS, both visible in the raw data and both handled here:

1. **The exit row's direction is INVERTED.** A long is opened with `buy` and
   closed with `sell`, so taking `cmd` from the exit reverses every trade. The
   direction is read from the ENTRY row.

2. **Volumes disagree on ~4% of pairs** -- partial closes, where an entry is
   closed in pieces. The EXIT volume is what actually closed and produced the
   P&L, so that is the size used.

`order` is also not perfectly unique (8.34M entry rows against 8.21M distinct
ids), so entries are de-duplicated to their earliest occurrence before joining;
otherwise a repeated id fans out into several spurious trades.
"""

from __future__ import annotations

import pandas as pd

MT5_COLUMNS = ["order", "login", "symbol", "cmd", "volume_lots", "open_time", "close_time",
               "open_price", "close_price", "sl", "tp", "net_profit", "state", "reason",
               "account_key"]

#: Exit states that represent a genuine close. `closed_by` is a position closed
#: against an opposing one and `reversed` is a direction flip; both realise P&L
#: and both belong in the sample.
CLOSING_STATES = ("closed", "closed_by", "reversed")


def pair_mt5_trades(deals: pd.DataFrame) -> pd.DataFrame:
    """Join MT5 entry and exit deals into round-trip trades.

    Returns a frame with the same shape the MT4 path produces -- ``open_time``,
    ``close_time``, ``open_price``, ``close_price``, ``cmd``, ``volume_lots``,
    ``net_profit`` -- so a single feature builder serves both platforms.
    """
    state = deals["state"].astype("string")
    entries = deals.loc[state == "open"].copy()
    exits = deals.loc[state.isin(CLOSING_STATES)].copy()
    if entries.empty or exits.empty:
        return pd.DataFrame(columns=MT5_COLUMNS)

    # De-duplicate entries to the earliest row per order id. Repeated ids would
    # otherwise fan out into several trades that never happened.
    entries = (entries.sort_values("open_time")
                      .drop_duplicates(subset="order", keep="first"))

    entry_side = entries[["order", "open_time", "open_price", "cmd", "sl", "tp", "reason"]].rename(
        columns={"open_price": "entry_price", "cmd": "entry_cmd", "sl": "entry_sl",
                 "tp": "entry_tp", "reason": "entry_reason"})

    merged = exits.merge(entry_side, on="order", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=MT5_COLUMNS)

    # Direction from the ENTRY. The exit row carries the opposite side, so using
    # it would invert every trade in the book.
    merged["cmd"] = merged["entry_cmd"]
    merged["open_time"] = merged["open_time_y"] if "open_time_y" in merged else merged["open_time"]
    # Prefer the entry's own price; the exit row usually carries it too, but the
    # entry is the authoritative record of what was paid.
    merged["open_price"] = merged["entry_price"].where(
        merged["entry_price"].notna() & (merged["entry_price"] > 0), merged["open_price"])
    # Stops belong to the entry decision.
    merged["sl"] = merged["entry_sl"]
    merged["tp"] = merged["entry_tp"]
    merged["reason"] = merged["entry_reason"]

    # Keep only pairs whose exit genuinely follows the entry. A handful invert,
    # which indicates a mis-joined id rather than a real trade.
    valid = merged["close_time"] > merged["open_time"]
    merged = merged.loc[valid]

    merged["state"] = "closed"
    return merged[[c for c in MT5_COLUMNS if c in merged.columns]].reset_index(drop=True)


def load_paired_mt5(parquet_path, database: str) -> pd.DataFrame:
    """Read one MT5 server's deals and return round-trip trades."""
    deals = pd.read_parquet(parquet_path, columns=MT5_COLUMNS,
                            filters=[("database", "==", database)])
    trades = pair_mt5_trades(deals)
    return trades.loc[
        trades["net_profit"].notna()
        & (trades["volume_lots"] > 0)
        & (trades["open_price"] > 0)
        & trades["open_time"].notna()
        & trades["close_time"].notna()
    ].reset_index(drop=True)
