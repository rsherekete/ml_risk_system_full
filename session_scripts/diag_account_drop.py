import sys
sys.path.insert(0, r'c:\Users\RoyVivasi\Documents\notebook')

import pandas as pd
import numpy as np

from trading_data.research import (
    clients_from_yaml,
    platform_for_database,
    add_provenance,
    daily_account_features,
    supervised_dataset,
    fit_oos_predictions,
    predict_live,
)
from trading_data.book_assignment import (
    daily_intelligence_components,
    expanding_client_profile,
    assign_books,
)

pd.set_option("display.width", 200)

decision_day = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D")
start = decision_day - pd.Timedelta(days=90)
end = decision_day + pd.Timedelta(days=1)

print(f"window: {start} .. {end}")

clients = clients_from_yaml()
client = clients["mt4_live01"]
database = "mt4_live01"

raw = client.get_trade_records(start.to_pydatetime(), end.to_pydatetime())
print(f"raw rows: {len(raw)}  raw distinct logins: {raw['login'].nunique() if 'login' in raw else 'N/A'}")

records = add_provenance(raw, database)
print(f"records rows: {len(records)}  distinct account_key: {records['account_key'].nunique()}")

features = daily_account_features(records)
print(f"features rows: {len(features)}  distinct account_key: {features['account_key'].nunique()}")

dataset = supervised_dataset(features)
print(f"dataset rows: {len(dataset)}  distinct account_key: {dataset['account_key'].nunique()}")

accounts_in_features = set(features["account_key"].unique())
accounts_in_dataset = set(dataset["account_key"].unique())
dropped_by_supervised = accounts_in_features - accounts_in_dataset
print(f"accounts present in features but ENTIRELY absent from supervised_dataset: {len(dropped_by_supervised)}")

predictions = fit_oos_predictions(dataset)
print(f"predictions rows: {len(predictions)}  distinct account_key: {predictions['account_key'].nunique()}")
print(f"predictions rows with non-null model_probability_loss: {predictions['model_probability_loss'].notna().sum()}")

accounts_in_predictions = set(predictions.loc[predictions['model_probability_loss'].notna(), 'account_key'].unique())
print(f"distinct decision_days in dataset: {pd.to_datetime(dataset['decision_day']).dt.normalize().nunique()}")
print(f"dataset target_loss value counts: {dataset['target_loss'].value_counts(dropna=False).to_dict()}")

live_predictions = predict_live(dataset, features)
print(f"live_predictions rows: {len(live_predictions)}  distinct account_key: {live_predictions['account_key'].nunique() if len(live_predictions) else 0}")

predictions_with_live = pd.concat([predictions, live_predictions], ignore_index=True) if not live_predictions.empty else predictions
print(f"predictions_with_live rows: {len(predictions_with_live)}  distinct account_key: {predictions_with_live['account_key'].nunique()}")

accounts_missing_from_predictions_with_live = accounts_in_features - set(predictions_with_live['account_key'].unique())
print(f"accounts in features but MISSING from predictions_with_live entirely: {len(accounts_missing_from_predictions_with_live)}")
if accounts_missing_from_predictions_with_live:
    sample = list(accounts_missing_from_predictions_with_live)[:10]
    print(f"sample missing accounts: {sample}")
    # Inspect one in detail
    for acct in sample[:3]:
        sub = features.loc[features['account_key'] == acct].sort_values('day')
        print(f"--- {acct} feature rows ---")
        print(sub[['day', 'observations', 'close_count']])
        sub_ds = dataset.loc[dataset['account_key'] == acct]
        print(f"rows in dataset for {acct}: {len(sub_ds)}")

components = daily_intelligence_components(records)
print(f"components rows: {len(components)}  distinct account_key: {components['account_key'].nunique()}")

profile = expanding_client_profile(components)
print(f"profile rows: {len(profile)}  distinct account_key: {profile['account_key'].nunique()}")

assignment = assign_books(predictions_with_live, profile)
print(f"assignment rows: {len(assignment)}  distinct account_key: {assignment['account_key'].nunique()}")

print()
print("=== SUMMARY ===")
print(f"distinct account_key in RAW records:              {records['account_key'].nunique()}")
print(f"distinct account_key in daily_account_features:    {features['account_key'].nunique()}")
print(f"distinct account_key in supervised_dataset:        {dataset['account_key'].nunique()}")
print(f"distinct account_key in fit_oos_predictions:       {predictions['account_key'].nunique()}")
print(f"distinct account_key in predict_live:               {live_predictions['account_key'].nunique() if len(live_predictions) else 0}")
print(f"distinct account_key in predictions_with_live:     {predictions_with_live['account_key'].nunique()}")
print(f"distinct account_key in profile (expanding_client_profile): {profile['account_key'].nunique()}")
print(f"distinct account_key in FINAL assign_books() output: {assignment['account_key'].nunique()}")
print(f"GAP (records vs final assignment): {records['account_key'].nunique() - assignment['account_key'].nunique()}")
