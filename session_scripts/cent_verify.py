import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import pandas as pd
from webapp import trade_feed, mysql_extract as m

db = "mt5_live01"
cents = trade_feed.cent_logins(db)
print(f"{db}: {len(cents)} cent logins")

# raw positions (no deflation) vs deflated
con = trade_feed._connection(db)
raw = pd.read_sql("SELECT login, symbol, volume, profit FROM positions LIMIT 100000", con)
raw["is_cent"] = raw["login"].astype("int64").isin(cents)
raw["lots_raw"] = pd.to_numeric(raw["volume"], errors="coerce") / 10000.0
print("open positions:", len(raw), "| cent positions:", int(raw["is_cent"].sum()),
      "| non-cent:", int((~raw["is_cent"]).sum()))
print("\nRAW lots by tier (before cent fix):")
print("  cent    mean lots_raw:", round(raw.loc[raw["is_cent"], "lots_raw"].mean(), 3),
      "| max:", round(raw.loc[raw["is_cent"], "lots_raw"].max(), 2))
print("  non-cent mean lots_raw:", round(raw.loc[~raw["is_cent"], "lots_raw"].mean(), 3),
      "| max:", round(raw.loc[~raw["is_cent"], "lots_raw"].max(), 2))

# via the fixed open_positions
f = m.open_positions(db)
f["is_cent"] = f["login"].astype("int64").isin(cents)
print("\nDEFLATED via open_positions():")
print("  cent    mean volume_lots:", round(f.loc[f["is_cent"], "volume_lots"].mean(), 4),
      "| max:", round(f.loc[f["is_cent"], "volume_lots"].max(), 3))
print("  non-cent mean volume_lots:", round(f.loc[~f["is_cent"], "volume_lots"].mean(), 4),
      "| max:", round(f.loc[~f["is_cent"], "volume_lots"].max(), 3))
print("\n=> cent lots should be ~1/100 of raw; non-cent unchanged")
