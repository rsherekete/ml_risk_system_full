import sys
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from trading_data.research import add_provenance, clients_from_yaml

decision_day = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D")
print("querying for:", decision_day, "to", decision_day + pd.Timedelta(days=1))
print("current UTC now:", pd.Timestamp.now(tz="UTC"))
print()

for name, client in clients_from_yaml().items():
    try:
        r = add_provenance(client.get_trade_records(decision_day.to_pydatetime(), (decision_day + pd.Timedelta(days=1)).to_pydatetime()), name)
        if not r.empty:
            latest = pd.to_datetime(r["timestamp"]).max()
            earliest = pd.to_datetime(r["timestamp"]).min()
            print(f"{name}: {len(r)} rows today, accounts={r['account_key'].nunique()}, earliest={earliest}, latest={latest}")
        else:
            print(f"{name}: 0 rows today")
    except Exception as exc:
        print(f"{name}: FAILED -> {exc}")
