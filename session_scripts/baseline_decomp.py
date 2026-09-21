"""Account for the remaining baseline gap: Trading $117.4M vs Quant $109.5M.

Both now cover the same six servers, so server coverage no longer explains it.
Candidate causes, tested one at a time:
  A. the account-day target is the NEXT ACTIVE DAY, so each account's final
     active day has no successor and is dropped
  B. Quant excludes trades OPENED before the extract window (survivor bias)
  C. Quant counts only CLOSED trades -- positions still open realise nothing
  D. different warm-up rows dropped by the walk-forward
"""
import sys

import pandas as pd

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import model_service as ms
from webapp.mt5_pairing import load_paired_mt5

trading = ms.load_scores("trading")
quant = ms.load_scores("quant")
for frame in (trading, quant):
    frame["server"] = frame["account_key"].str.split(":").str[0]

print(f"TRADING  {len(trading):>12,} account-days  firm ${-trading['pnl'].sum():>14,.0f}")
print(f"QUANT    {len(quant):>12,} trades        firm ${-quant['pnl'].sum():>14,.0f}")
print(f"GAP                                    ${(-trading['pnl'].sum()) - (-quant['pnl'].sum()):>14,.0f}\n")

print("per server (firm P&L):")
t_by = trading.groupby("server")["pnl"].sum().mul(-1)
q_by = quant.groupby("server")["pnl"].sum().mul(-1)
compare = pd.DataFrame({"trading": t_by, "quant": q_by}).fillna(0)
compare["gap"] = compare["trading"] - compare["quant"]
print(compare.round(0).to_string())

print(f"\ndate coverage")
print(f"  trading: {pd.to_datetime(trading['day']).min().date()} .. {pd.to_datetime(trading['day']).max().date()}"
      f"  ({pd.to_datetime(trading['day']).nunique()} days)")
print(f"  quant  : {pd.to_datetime(quant['day']).min().date()} .. {pd.to_datetime(quant['day']).max().date()}"
      f"  ({pd.to_datetime(quant['day']).nunique()} days)")

# --- C: does the OPEN-position exclusion explain it? -----------------------
print("\n--- C. still-open positions (the user's hypothesis) ---")
total_open_float = 0.0
for server in ("mt4_live01", "mt4_live02", "mt4_live03", "mt4_live04"):
    part = pd.read_parquet(ms.SCRATCH / "bq_90d_records.parquet",
                           columns=["state", "profit", "net_profit"],
                           filters=[("database", "==", server)])
    state = part["state"].astype("string")
    still_open = part.loc[state == "open"]
    floating = pd.to_numeric(still_open["profit"], errors="coerce").fillna(0).sum()
    total_open_float += floating
    print(f"  {server}: {len(still_open):>9,} open rows, floating P&L ${floating:>13,.0f}")
print(f"  MT4 total floating: ${total_open_float:,.0f}  (firm side ${-total_open_float:,.0f})")

# --- B: survivor trades excluded from Quant --------------------------------
print("\n--- B. trades opened BEFORE the window (excluded from Quant) ---")
excluded_pnl = 0.0
for server in ("mt4_live01", "mt4_live02", "mt4_live03", "mt4_live04"):
    part = pd.read_parquet(ms.SCRATCH / "bq_90d_records.parquet",
                           columns=["state", "open_time", "close_time", "net_profit", "volume_lots"],
                           filters=[("database", "==", server)])
    closed = part.loc[(part["state"].astype("string") == "closed")
                      & part["net_profit"].notna() & (part["volume_lots"] > 0)
                      & part["open_time"].notna()]
    window_start = closed["close_time"].min().floor("D")
    old = closed.loc[closed["open_time"] < window_start]
    excluded_pnl += old["net_profit"].sum()
    print(f"  {server}: {len(old):>7,} pre-window trades, ${old['net_profit'].sum():>12,.0f}")
print(f"  total excluded client P&L ${excluded_pnl:,.0f}  (firm side ${-excluded_pnl:,.0f})")

# --- A: account-days with no successor -------------------------------------
print("\n--- A. unit difference ---")
print("  Trading target = the NEXT ACTIVE DAY's P&L, so an account's final active")
print("  day has no successor and is dropped; Quant counts every closed trade.")
print(f"  trading rows {len(trading):,} vs distinct accounts {trading['account_key'].nunique():,}")
print(f"  quant  rows {len(quant):,} vs distinct accounts {quant['account_key'].nunique():,}")
