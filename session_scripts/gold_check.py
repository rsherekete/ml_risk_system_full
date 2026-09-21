"""Trace gold from raw positions through to USD notional."""
import sys

import pandas as pd

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import exposure as exposure_module
from webapp import mysql_extract, symbol_specs, views

specs, rates = symbol_specs.load_all_specs(tuple(mysql_extract.MYSQL_DATABASES))
frames = []
for database in mysql_extract.MYSQL_DATABASES:
    try:
        frame = mysql_extract.open_positions(database)
        if not frame.empty:
            frames.append(views.add_canonical_symbol(frame))
    except Exception:
        pass
positions = pd.concat(frames, ignore_index=True)

gold = positions.loc[positions["canonical_symbol"] == "XAUUSD"]
print(f"gold positions: {len(gold):,}")
print("\nby raw ticker and server:")
print(gold.groupby(["database", "symbol"], observed=True).agg(
    positions=("volume_lots", "size"),
    lots=("volume_lots", lambda s: float(s.abs().sum()))).to_string())

priced = symbol_specs.usd_notional(positions, specs, rates)
gold_priced = priced.loc[priced["canonical_symbol"] == "XAUUSD"]
print("\nafter pricing, by raw ticker:")
print(gold_priced.groupby(["database", "symbol"], observed=True).agg(
    lots=("volume_lots", lambda s: float(s.abs().sum())),
    contract=("contract_size", "first"),
    gross_usd=("gross_notional_usd", "sum"),
    status=("notional_status", "first")).to_string())

result = exposure_module.exposure_by_symbol(positions, specs, rates)
row = result.loc[result["canonical_symbol"] == "XAUUSD"]
print("\naggregated XAUUSD row:")
print(row.to_string(index=False))

print("\ntop 6 by GROSS USD:")
top = result.copy()
top["g"] = pd.to_numeric(top["gross_notional"], errors="coerce")
print(top.nlargest(6, "g")[["canonical_symbol", "gross_lots", "gross_notional",
                            "net_notional", "notional_status"]].to_string(index=False))
