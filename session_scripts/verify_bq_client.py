import sys
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
import shutil
import time
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from trading_data import BigQueryDataClient, cached_trade_records
from trading_data.research import add_provenance

test_cache_dir = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\bq_cache_test"
shutil.rmtree(test_cache_dir, ignore_errors=True)

decision_day = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D")

for database in ["mt4_live01", "mt5_live01"]:
    print(f"\n{'='*70}\n{database}\n{'='*70}")
    client = BigQueryDataClient(database)

    t0 = time.time()
    ping = client.ping()
    print(f"ping: {ping.iloc[0].to_dict()} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    direct = client.get_trade_records((decision_day - pd.Timedelta(days=5)).to_pydatetime(), (decision_day + pd.Timedelta(days=1)).to_pydatetime())
    print(f"direct get_trade_records (5d): {len(direct):,} rows, {direct['login'].nunique():,} accounts, columns={list(direct.columns)} ({time.time()-t0:.1f}s)")
    print(direct.head(2).to_string())
    frame = add_provenance(direct, database)
    print(f"add_provenance worked: account_key sample = {frame['account_key'].iloc[0] if not frame.empty else 'N/A'}")

    # first cache call -- should hit BQ for the full window
    t0 = time.time()
    first = cached_trade_records(client, (decision_day - pd.Timedelta(days=10)).to_pydatetime(), (decision_day + pd.Timedelta(days=1)).to_pydatetime(), cache_dir=test_cache_dir)
    print(f"\ncached_trade_records first call (10d, cold cache): {len(first):,} rows ({time.time()-t0:.1f}s)")

    # second cache call, same window -- should only re-query the overlap tail
    t0 = time.time()
    second = cached_trade_records(client, (decision_day - pd.Timedelta(days=10)).to_pydatetime(), (decision_day + pd.Timedelta(days=1)).to_pydatetime(), cache_dir=test_cache_dir)
    print(f"cached_trade_records second call (warm cache): {len(second):,} rows ({time.time()-t0:.1f}s, should be much faster)")
    assert len(first) == len(second) or abs(len(first) - len(second)) < 100, "warm-cache result should closely match cold-cache result for the same window"

print("\nALL BQ CLIENT CHECKS PASSED")
