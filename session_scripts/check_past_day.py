import sys
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from trading_data.research import (
    Economics, add_provenance, attach_symbol_specs, clients_from_yaml,
    daily_account_features, supervised_dataset, fit_oos_predictions, predict_live,
)
from trading_data.book_assignment import assign_books, daily_intelligence_components, expanding_client_profile, firm_daily_pnl

decision_day = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D")
start = decision_day - pd.Timedelta(days=90)
end = decision_day + pd.Timedelta(days=1)

frames, specs_by_db = [], {}
for name, client in clients_from_yaml().items():
    try:
        frames.append(add_provenance(client.get_trade_records(start.to_pydatetime(), end.to_pydatetime()), name))
        specs_by_db[name] = client.symbol_specs()
    except Exception:
        pass
records = pd.concat(frames, ignore_index=True)
enriched = [attach_symbol_specs(records.loc[records["database"] == n], s) for n, s in specs_by_db.items() if not records.loc[records["database"] == n].empty]
records = pd.concat(enriched, ignore_index=True)

features = daily_account_features(records)
dataset = supervised_dataset(features)
predictions = fit_oos_predictions(dataset)
live = predict_live(dataset, features)
combined = pd.concat([predictions, live], ignore_index=True) if not live.empty else predictions

components = daily_intelligence_components(records)
profile = expanding_client_profile(components)
assignment = assign_books(combined, profile, 70, 30)
daily_pnl, detail, metrics = firm_daily_pnl(records, assignment, Economics())

# Check the last few NORMAL business days, not just literally "today"
by_day = detail.groupby("decision_day")["account_key"].nunique().sort_index()
print(by_day.tail(10))
print()
print("Raw activity accounts per day (ground truth, no assignment logic at all):")
raw_by_day = records.copy()
raw_by_day["day"] = pd.to_datetime(raw_by_day["timestamp"]).dt.floor("D")
print(raw_by_day.groupby("day")["account_key"].nunique().tail(10))
