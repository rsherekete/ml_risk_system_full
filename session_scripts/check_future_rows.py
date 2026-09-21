import sys
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from trading_data.research import clients_from_yaml

now = pd.Timestamp.now(tz="UTC").tz_localize(None)
future_end = now + pd.Timedelta(days=2)
print("current UTC now:", pd.Timestamp.now(tz="UTC"))
for name, client in clients_from_yaml().items():
    try:
        r = client.get_trade_records(now.to_pydatetime(), future_end.to_pydatetime())
        if not r.empty:
            latest = pd.to_datetime(r["timestamp"]).max()
            print(f"{name}: {len(r)} rows with a FUTURE timestamp (clock skew?) latest={latest}")
        else:
            print(f"{name}: no future-dated rows (normal)")
    except Exception as exc:
        print(f"{name}: FAILED -> {exc}")
