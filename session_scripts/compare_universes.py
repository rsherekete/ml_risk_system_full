"""Why do the two flat-B-book baselines differ ($117.4M vs $71.1M)?

They are not the same population, and the difference should be explainable to
the dollar rather than asserted. Three candidate causes:
  1. different servers (Trading covers 8 databases, Quant only 3 MT4 servers)
  2. different unit (account-day next-active-day P&L vs per-trade net profit)
  3. different window (Quant now excludes trades opened before the extract window)
"""
import sys

import pandas as pd

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import model_service as ms

trading = ms.load_scores("trading")
quant = ms.load_scores("quant")

trading["server"] = trading["account_key"].str.split(":").str[0]
quant["server"] = quant["account_key"].str.split(":").str[0]

print(f"TRADING artefact: {len(trading):,} account-days | firm ${-trading['pnl'].sum():,.0f}")
print(trading.groupby("server")["pnl"].agg(rows="size", client_pnl="sum")
      .assign(firm=lambda d: -d.client_pnl).to_string())
print(f"\nQUANT artefact: {len(quant):,} trades | firm ${-quant['pnl'].sum():,.0f}")
print(quant.groupby("server")["pnl"].agg(rows="size", client_pnl="sum")
      .assign(firm=lambda d: -d.client_pnl).to_string())

shared = ("mt4_live01", "mt4_live02", "mt4_live04")
t_shared = trading.loc[trading["server"].isin(shared)]
print(f"\n--- restricted to the three servers BOTH models cover ---")
print(f"  Trading firm P&L : ${-t_shared['pnl'].sum():>14,.0f}  ({len(t_shared):,} account-days)")
print(f"  Quant   firm P&L : ${-quant['pnl'].sum():>14,.0f}  ({len(quant):,} trades)")
print(f"  difference       : ${(-quant['pnl'].sum()) - (-t_shared['pnl'].sum()):>14,.0f}")

print(f"\ndate ranges")
print(f"  Trading: {pd.to_datetime(trading['day']).min().date()} .. "
      f"{pd.to_datetime(trading['day']).max().date()}")
print(f"  Quant  : {pd.to_datetime(quant['day']).min().date()} .. "
      f"{pd.to_datetime(quant['day']).max().date()}")

# The unit difference: the account-day target is the NEXT ACTIVE DAY's P&L, so
# an account's final active day has no successor and is dropped, and days with
# no following activity never appear. Per-trade counts every closed trade.
print(f"\nunit check -- raw closed-trade P&L on the three shared servers, from source:")
total = 0.0
for server in shared:
    part = pd.read_parquet(
        ms.SCRATCH / "bq_90d_records.parquet",
        columns=["state", "net_profit", "volume_lots", "open_time", "close_time"],
        filters=[("database", "==", server)])
    closed = part.loc[(part["state"].astype("string") == "closed")
                      & part["net_profit"].notna() & (part["volume_lots"] > 0)]
    window_start = closed["close_time"].min().floor("D")
    in_window = closed.loc[closed["open_time"] >= window_start]
    total += in_window["net_profit"].sum()
    print(f"  {server}: all closed ${-closed['net_profit'].sum():>14,.0f} | "
          f"opened in window ${-in_window['net_profit'].sum():>14,.0f}")
print(f"  TOTAL opened-in-window firm P&L: ${-total:,.0f}")
