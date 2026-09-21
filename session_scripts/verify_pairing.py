"""Verify the MT5 pairing reconstructs trades correctly before retraining on it."""
import sys

import pandas as pd

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp.model_service import SCRATCH
from webapp.mt5_pairing import load_paired_mt5

for server in ("mt5_live01", "mt5_dubai_live01"):
    trades = load_paired_mt5(SCRATCH / "bq_90d_records.parquet", server)
    if trades.empty:
        print(f"{server}: no trades")
        continue
    raw = pd.read_parquet(SCRATCH / "bq_90d_records.parquet",
                          columns=["state", "net_profit"], filters=[("database", "==", server)])
    closed = raw.loc[raw["state"].astype("string").isin(("closed", "closed_by", "reversed"))]

    print(f"\n{server}")
    print(f"  paired trades   : {len(trades):,}")
    print(f"  client P&L      : ${trades['net_profit'].sum():,.0f}")
    print(f"  raw closed P&L  : ${closed['net_profit'].sum():,.0f}"
          f"  ({trades['net_profit'].sum()/closed['net_profit'].sum():.1%} recovered)")
    hold = (trades["close_time"] - trades["open_time"]).dt.total_seconds() / 3600
    print(f"  holding hours   : median {hold.median():.2f}, p95 {hold.quantile(0.95):.1f}")
    print(f"  all close>open  : {(trades['close_time'] > trades['open_time']).all()}")
    print(f"  direction split : {trades['cmd'].value_counts().to_dict()}")
    print(f"  distinct accounts: {trades['account_key'].nunique():,}")
    print(f"  window          : {trades['open_time'].min().date()} .. {trades['open_time'].max().date()}")

    # A buy that closed above its entry should show a profit more often than not.
    buys = trades.loc[trades["cmd"].astype("string") == "buy"]
    if len(buys):
        moved_up = buys["close_price"] > buys["open_price"] if "close_price" in buys else None
        if moved_up is not None:
            agree = (moved_up == (buys["net_profit"] > 0)).mean()
            print(f"  direction sanity: {agree:.1%} of buys agree "
                  f"(price up <-> profit) -- low would mean an inverted side")
