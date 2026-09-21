"""Was the January drawdown a single crowded position? If so, the fix is obvious.

The drawdown was dispersed across 7,570 accounts, which rules out a whale and
points at correlated exposure. This checks the next link in the chain: were
those accounts all in the SAME instrument, on the SAME side?

If yes, the firm's real risk is net book exposure by symbol -- a quantity it
already computes correctly in USD -- and the loss was foreseeable from the
book itself, without predicting any individual client.
"""
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import data_store

PEAK = datetime(2026, 1, 3, tzinfo=timezone.utc)
TROUGH = datetime(2026, 1, 20, tzinfo=timezone.utc)

trades = data_store.read_history(
    start=PEAK, end=TROUGH,
    columns=["database", "login", "symbol", "cmd", "volume_lots",
             "open_time", "close_time", "open_price", "close_price", "net_profit"])
trades["net_profit"] = pd.to_numeric(trades["net_profit"], errors="coerce")
trades = trades.loc[trades["net_profit"].notna()]
print(f"trades closed in the fall: {len(trades):,} "
      f"across {trades['symbol'].nunique():,} symbols")

# Canonical symbol: XAUUSD, XAUUSDe, XAUUSDmin and XAUUSD247 are the same risk.
# Treating them separately would hide exactly the concentration being looked for.
raw = trades["symbol"].astype(str).str.upper()
canon = (raw.str.replace(r"(MIN|MICRO|E|247|\.|_|#|\+)+$", "", regex=True)
            .str.replace(r"[^A-Z0-9]", "", regex=True))
trades["canonical"] = canon

by_symbol = trades.groupby("canonical").agg(
    trades=("net_profit", "size"),
    accounts=("login", "nunique"),
    client_pnl=("net_profit", "sum"),
    lots=("volume_lots", "sum"),
).sort_values("client_pnl", ascending=False)

total_client = trades["net_profit"].sum()
print(f"\ntotal client P&L in the fall: ${total_client:,.0f} "
      f"(firm lost ${-total_client:,.0f})")
print(f"\n{'symbol':<14}{'trades':>10}{'accounts':>10}{'client P&L':>16}{'share':>9}{'lots':>12}")
for symbol, row in by_symbol.head(10).iterrows():
    print(f"{symbol:<14}{int(row['trades']):>10,}{int(row['accounts']):>10,}"
          f"{row['client_pnl']:>16,.0f}{row['client_pnl'] / total_client:>9.1%}"
          f"{row['lots']:>12,.0f}")

top = by_symbol.index[0]
print(f"\n  top symbol {top} is {by_symbol.iloc[0]['client_pnl'] / total_client:.1%} "
      f"of all client winnings in the fall, across "
      f"{int(by_symbol.iloc[0]['accounts']):,} accounts")

# ---- were they all on the same side? --------------------------------------
print(f"\n=== DIRECTIONAL CROWDING IN {top} ===")
focus = trades.loc[trades["canonical"] == top].copy()
side = focus["cmd"].astype(str).str.lower()
for name, group in focus.groupby(side):
    print(f"  {name:<6} {len(group):>8,} trades  {group['volume_lots'].sum():>12,.0f} lots"
          f"  client P&L ${group['net_profit'].sum():>14,.0f}")
buy_lots = focus.loc[side == "buy", "volume_lots"].sum()
sell_lots = focus.loc[side == "sell", "volume_lots"].sum()
net = buy_lots - sell_lots
gross = buy_lots + sell_lots
print(f"\n  net {net:,.0f} lots on a gross of {gross:,.0f} "
      f"({abs(net) / max(gross, 1):.1%} directional)")
print("  A book that is heavily one-way in a single instrument is not diversified;")
print("  it is one position. The firm was short whatever its clients were long.")

# ---- daily build-up: was it visible BEFORE the loss? -----------------------
print("\n=== WAS THE EXPOSURE VISIBLE IN ADVANCE? ===")
focus["open_day"] = pd.to_datetime(focus["open_time"]).dt.normalize()
focus["close_day"] = pd.to_datetime(focus["close_time"]).dt.normalize()
signed = np.where(side == "buy", focus["volume_lots"], -focus["volume_lots"])
opened = pd.Series(signed, index=focus.index).groupby(focus["open_day"]).sum()
realised = focus.groupby("close_day")["net_profit"].sum()
table = pd.DataFrame({"net_lots_opened": opened, "client_pnl_realised": realised}).fillna(0.0)
table["cumulative_net_lots"] = table["net_lots_opened"].cumsum()
print(f"{'day':<12}{'net lots opened':>18}{'cumulative net':>18}{'client P&L':>16}")
for day, row in table.iterrows():
    print(f"{str(day.date()):<12}{row['net_lots_opened']:>18,.0f}"
          f"{row['cumulative_net_lots']:>18,.0f}{row['client_pnl_realised']:>16,.0f}")
print("\n  If cumulative net lots builds up BEFORE the losses land, the risk was")
print("  observable from the book alone -- no client prediction required.")
