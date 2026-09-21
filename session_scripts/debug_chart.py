"""Reproduce the account-chart endpoint to get a real traceback."""
import sys
import traceback
from datetime import timedelta

import pandas as pd

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import tick_bars, views

account = "mt4_live02:2395433"
try:
    trades = views.account_trades(account, None, limit=400)
    print(f"trades: {0 if trades is None else len(trades)}")
    trades = trades.copy()
    trades["open_time"] = pd.to_datetime(trades["open_time"])
    trades["close_time"] = pd.to_datetime(trades["close_time"])
    latest = trades["close_time"].max()
    print("latest close:", latest)
    window_end = (latest + timedelta(hours=2)).to_pydatetime()
    window_start = (latest - timedelta(days=3)).to_pydatetime()
    focus = trades.loc[trades["open_time"] >= pd.Timestamp(window_start)]
    print(f"focus rows: {len(focus)}")
    raw_symbol = focus["symbol"].mode().iloc[0] if len(focus) else None
    print("symbol:", raw_symbol, type(raw_symbol))
    database = account.split(":", 1)[0]
    bars = tick_bars.fetch_bars(database, (str(raw_symbol),), window_start, window_end,
                                chunk_hours=12, workers=4)
    print(f"bars: {len(bars)}")
    print(bars.head(3).to_string(index=False))
except Exception:
    traceback.print_exc()
