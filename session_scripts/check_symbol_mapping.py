import sys
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from trading_data.research import add_provenance, attach_symbol_specs, clients_from_yaml, symbol_mapping_table, canonical_symbol

decision_day = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D")
start = decision_day - pd.Timedelta(days=10)
end = decision_day + pd.Timedelta(days=1)

frames, specs_by_db = [], {}
for name, client in clients_from_yaml().items():
    try:
        frames.append(add_provenance(client.get_trade_records(start.to_pydatetime(), end.to_pydatetime()), name))
        specs_by_db[name] = client.symbol_specs()
    except Exception:
        pass
records = pd.concat(frames, ignore_index=True)

mapping = symbol_mapping_table(records, specs_by_db)
pd.set_option("display.max_rows", 400)
pd.set_option("display.width", 200)

# Group canonical symbols and look for ones that look suspiciously similar
# (same alnum "root" but different canonical bucket) -- evidence of raw
# symbol variants our heuristic failed to merge.
counts = mapping.groupby("canonical_symbol")["symbol"].apply(list)
print(f"Total raw symbols: {len(mapping)}, canonical groups: {mapping['canonical_symbol'].nunique()}")
print()
print("All canonical groups with their raw members:")
for canon, raws in counts.items():
    print(f"  {canon!r}: {raws}")
