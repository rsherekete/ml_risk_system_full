"""Reproduce the exposure endpoint end-to-end so the 500 has a stack trace."""
import sys
import traceback

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import pandas as pd

from webapp import exposure as exposure_module
from webapp import mysql_extract, views

frames, errors = [], []
for database in mysql_extract.MYSQL_DATABASES:
    try:
        frame = mysql_extract.open_positions(database)
        print(f"{database}: {len(frame):,} positions")
        if not frame.empty:
            frames.append(views.add_canonical_symbol(frame))
    except Exception as error:
        errors.append(f"{database}: {type(error).__name__}: {error}")
        print(f"{database}: FAILED {type(error).__name__}: {str(error)[:120]}")

if not frames:
    print("no frames"); raise SystemExit

positions = pd.concat(frames, ignore_index=True)
print(f"\ncombined {len(positions):,} rows, columns: {list(positions.columns)}")
print(positions.dtypes.to_string())

try:
    by_symbol = exposure_module.exposure_by_symbol(positions)
    print(f"\nexposure rows: {len(by_symbol)}")
    print(by_symbol.head(6).to_string())
    print(f"\nalerts: {len(exposure_module.concentration_alerts(by_symbol))}")
    print(f"stress: {len(exposure_module.stress_test(by_symbol))}")
    # The endpoint serialises to JSON; numpy/bool types are the usual culprit.
    import json
    json.dumps(by_symbol.to_dict("records"))
    print("JSON serialisation OK")
except Exception:
    traceback.print_exc()
