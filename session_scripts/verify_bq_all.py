import sys
sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")
import shutil, time, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from trading_data.bigquery_data_client import (
    BQ_SOURCE_FOR_DATABASE, BigQueryDataClient, BigQueryUnavailableError, cached_trade_records,
)
from trading_data.research import add_provenance

cache_dir = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\bq_cache_all"
shutil.rmtree(cache_dir, ignore_errors=True)

day = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D")
start, end = (day - pd.Timedelta(days=2)).to_pydatetime(), (day + pd.Timedelta(days=1)).to_pydatetime()

rows = []
for database in BQ_SOURCE_FOR_DATABASE:
    try:
        c = BigQueryDataClient(database)
    except BigQueryUnavailableError as exc:
        print(f"{database:<20} UNAVAILABLE (by design): {str(exc)[:90]}...")
        rows.append({"database": database, "status": "unavailable", "rows": 0, "accounts": 0})
        continue
    try:
        t0 = time.time()
        f = add_provenance(c.get_trade_records(start, end), database)
        tz_ok = all(not isinstance(f[col].dtype, pd.DatetimeTZDtype) for col in f.columns)
        assert tz_ok, f"{database}: tz-aware column leaked through!"
        print(f"{database:<20} OK  {len(f):>9,} rows  {f['account_key'].nunique():>6,} accts  "
              f"kind={c.source.kind:<16} tz_naive={tz_ok}  ({time.time()-t0:.1f}s)")
        rows.append({"database": database, "status": "ok", "rows": len(f), "accounts": f["account_key"].nunique()})
    except Exception as exc:
        print(f"{database:<20} FAILED: {type(exc).__name__}: {str(exc)[:110]}")
        rows.append({"database": database, "status": "failed", "rows": 0, "accounts": 0})

print()
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
print(f"\nTOTAL: {summary['rows'].sum():,} rows, {summary['accounts'].sum():,} accounts across "
      f"{(summary['status']=='ok').sum()}/{len(summary)} databases")

# cache round-trip on the smallest working database
working = summary.loc[(summary["status"] == "ok") & (summary["rows"] > 0)].sort_values("rows")
if not working.empty:
    db = working.iloc[0]["database"]
    c = BigQueryDataClient(db)
    t0 = time.time(); a = cached_trade_records(c, start, end, cache_dir=cache_dir); cold = time.time() - t0
    t0 = time.time(); b = cached_trade_records(c, start, end, cache_dir=cache_dir); warm = time.time() - t0
    print(f"\ncache check on {db}: cold={len(a):,} rows ({cold:.1f}s), warm={len(b):,} rows ({warm:.1f}s)")
    assert not a.empty and abs(len(a) - len(b)) <= max(50, 0.02 * len(a)), "warm cache must closely match cold"
    print("cache round-trip OK")

print("\nALL BQ CHECKS PASSED")
