import sys
sys.path.insert(0, r'c:\Users\RoyVivasi\Documents\notebook')

import traceback
import pandas as pd

from trading_data.research import clients_from_yaml, platform_for_database

decision_day = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D")
start = decision_day - pd.Timedelta(days=90)
end = decision_day + pd.Timedelta(days=1)

print(f"window: start={start} end={end}")
print()

clients = clients_from_yaml()

for name, client in clients.items():
    platform = platform_for_database(name)
    print("=" * 80)
    print(f"{name} ({platform})")
    print("=" * 80)
    try:
        if platform == "mt4":
            cols = client.columns("traderecord")
            tm_row = cols.loc[cols["column_name"] == "tm"]
            ts_row = cols.loc[cols["column_name"] == "ts"]
            print("tm column type:", tm_row["column_type"].values if len(tm_row) else "MISSING")
            print("ts column type:", ts_row["column_type"].values if len(ts_row) else "MISSING")
        else:
            cols = client.columns("dealrecord")
            ts_row = cols.loc[cols["column_name"] == "ts"]
            print("ts column type:", ts_row["column_type"].values if len(ts_row) else "MISSING")
    except Exception as e:
        print("column introspection failed:", e)
        traceback.print_exc()

    print()
