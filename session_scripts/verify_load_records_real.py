import sys
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
import time
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from trading_data.research import (
    account_group_lookup, add_provenance, clients_from_yaml, non_client_logins, platform_for_database,
)

decision_day = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D")
start = decision_day - pd.Timedelta(days=90)
end = decision_day + pd.Timedelta(days=1)

t0 = time.time()
frames, excluded = [], []
for database, client in clients_from_yaml().items():
    db_t0 = time.time()
    frame = add_provenance(client.get_trade_records(start.to_pydatetime(), end.to_pydatetime()), database)
    before = frame["login"].nunique() if "login" in frame else 0
    group_lookup = account_group_lookup(client, platform_for_database(database), frame.get("login", pd.Series(dtype="float64")))
    flagged = non_client_logins(group_lookup)
    if flagged:
        is_flagged = frame["login"].astype("Int64").isin(flagged)
        if is_flagged.any():
            excluded.append(frame.loc[is_flagged, ["database", "login"]].drop_duplicates())
            frame = frame.loc[~is_flagged].copy()
    after = frame["login"].nunique() if "login" in frame else 0
    print(f"{database}: {before} -> {after} distinct accounts ({before - after} excluded), {time.time()-db_t0:.1f}s")
    frames.append(frame)

records = pd.concat(frames, ignore_index=True)
excluded_frame = pd.concat(excluded, ignore_index=True) if excluded else pd.DataFrame(columns=["database", "login"])
print(f"\nTotal distinct accounts (any activity, 90-day window): {records['account_key'].nunique():,}")
print(f"Total test accounts excluded: {len(excluded_frame)}")
print(f"Total wall time: {time.time()-t0:.1f}s")
