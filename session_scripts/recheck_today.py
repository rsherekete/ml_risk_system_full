import sys
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from trading_data.research import add_provenance, clients_from_yaml

decision_day = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D")
start = decision_day - pd.Timedelta(days=10)
end = decision_day + pd.Timedelta(days=1)

print("current UTC now:", pd.Timestamp.now(tz="UTC"))
print()

frames = []
for name, client in clients_from_yaml().items():
    try:
        r = add_provenance(client.get_trade_records(start.to_pydatetime(), end.to_pydatetime()), name)
        frames.append(r)
        print(f"{name}: {len(r)} rows in last 10 days")
    except Exception as exc:
        print(f"{name}: FAILED -> {type(exc).__name__}: {exc}")

records = pd.concat(frames, ignore_index=True)
records["day"] = pd.to_datetime(records["timestamp"]).dt.floor("D")
by_day = records.groupby("day")["account_key"].nunique().sort_index()
print()
print("Distinct accounts with any activity, per day (raw ground truth, no analytics applied):")
print(by_day)
print()

# per-database breakdown for the last 2 days specifically
for day_offset in (1, 0):
    d = decision_day - pd.Timedelta(days=day_offset)
    print(f"\n--- {d.date()} breakdown by database ---")
    day_records = records.loc[records["day"] == d]
    if day_records.empty:
        print("  (no rows at all)")
        continue
    for db, grp in day_records.groupby("database"):
        latest = grp["timestamp"].max()
        print(f"  {db}: {len(grp)} rows, {grp['account_key'].nunique()} accounts, latest={latest}")
