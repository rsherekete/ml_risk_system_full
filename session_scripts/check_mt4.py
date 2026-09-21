import pandas as pd
BASE = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad"
COLS = ["state", "open_time", "close_time", "net_profit", "volume_lots", "open_price"]
for db in ("mt4_live01", "mt4_live02", "mt4_live04", "mt5_live01"):
    t = pd.read_parquet(f"{BASE}\\bq_90d_records.parquet", columns=COLS, filters=[("database", "==", db)])
    c = t.loc[(t["state"].astype("string") == "closed") & t["net_profit"].notna() & (t["volume_lots"] > 0)]
    if not len(c):
        print(f"{db}: no closed rows")
        continue
    print(f"{db}: closed {len(c):,} | open_time {c['open_time'].notna().mean():.1%} "
          f"| close_time {c['close_time'].notna().mean():.1%} "
          f"| open_price>0 {(c['open_price'] > 0).mean():.1%} "
          f"| client P&L ${c['net_profit'].sum():,.0f}")
