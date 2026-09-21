"""Two data-correctness questions before building anything on top.

1. Why do the day selectors stop short of today?
2. Is EURUSDe a separate instrument or a broker suffix on EURUSD?
"""
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import model_service as ms

print(f"today (UTC): {datetime.now(timezone.utc).date()}\n")

for view in ("trading", "quant"):
    frame = ms.load_scores(view)
    if frame is None:
        continue
    days = pd.to_datetime(frame["day"])
    print(f"{view}: {days.min().date()} .. {days.max().date()}  ({days.nunique()} days)")

# The artefacts derive from the cached 90-day extract; its own bound is the
# real ceiling, so check the source rather than blaming the model.
raw = pd.read_parquet(ms.SCRATCH / "bq_90d_records.parquet",
                      columns=["close_time", "open_time"],
                      filters=[("database", "==", "mt5_live01")])
print(f"\nsource extract close_time: {raw['close_time'].min()} .. {raw['close_time'].max()}")
print(f"source extract open_time : {raw['open_time'].min()} .. {raw['open_time'].max()}")

# --- symbol shape ---------------------------------------------------------
print("\n--- symbol naming across servers ---")
for database in ("mt4_live01", "mt4_live02", "mt4_live03", "mt4_live04",
                 "mt5_live01", "mt5_dubai_live01"):
    symbols = pd.read_parquet(ms.SCRATCH / "bq_90d_records.parquet", columns=["symbol"],
                              filters=[("database", "==", database)])["symbol"]
    unique = symbols.dropna().astype(str).unique()
    eur = sorted(s for s in unique if s.upper().startswith("EURUSD"))
    gold = sorted(s for s in unique if s.upper().startswith("XAUUSD"))
    print(f"  {database:<20} {len(unique):>4} symbols | EURUSD*: {eur[:6]} | XAUUSD*: {gold[:6]}")

quant = ms.load_scores("quant")
if quant is not None and "symbol" in quant:
    counts = quant["symbol"].value_counts()
    suffixed = [s for s in counts.index if len(str(s)) > 6 and str(s)[:6].isalpha()]
    print(f"\nsuffixed-looking symbols in the Quant artefact ({len(suffixed)}):")
    for name in suffixed[:20]:
        print(f"  {name:<14} {counts[name]:>9,} trades")
