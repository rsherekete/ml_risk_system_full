"""Verify USD notional is computed correctly across the three conversion routes."""
import sys

import pandas as pd

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import exposure as exposure_module
from webapp import mysql_extract, symbol_specs, views

specs, rates = symbol_specs.load_all_specs(tuple(mysql_extract.MYSQL_DATABASES))
print(f"specs: {len(specs):,} rows across {specs['database'].nunique()} servers")
print(f"FX rates derived: {len(rates)} currencies")
for currency in ("USD", "EUR", "GBP", "JPY", "AUD", "CHF", "CAD", "NZD"):
    print(f"  {currency}: {rates.get(currency, 'MISSING')}")

sample = specs.loc[specs["canonical_symbol"].isin(
    ["EURUSD", "USDJPY", "GBPJPY", "XAUUSD", "USTEC", "BTCUSD"])]
print("\nspecs for reference instruments:")
print(sample[["database", "canonical_symbol", "contract_size",
              "currency_base", "currency_profit"]].drop_duplicates(
    "canonical_symbol").to_string(index=False))

frames = []
for database in mysql_extract.MYSQL_DATABASES:
    try:
        frame = mysql_extract.open_positions(database)
        if not frame.empty:
            frames.append(views.add_canonical_symbol(frame))
    except Exception as error:
        print(f"{database}: {type(error).__name__}")
positions = pd.concat(frames, ignore_index=True)
print(f"\n{len(positions):,} open positions")

result = exposure_module.exposure_by_symbol(positions, specs, rates)
print(f"\n{len(result)} instruments\n")
print(f"{'symbol':<12}{'net lots':>11}{'gross lots':>12}{'net USD':>18}"
      f"{'gross USD':>18}  status")
for row in result.head(14).itertuples():
    net = row.net_notional if row.net_notional is not None else float('nan')
    gross = row.gross_notional if row.gross_notional is not None else float('nan')
    print(f"{row.canonical_symbol:<12}{row.net_lots:>11,.2f}{row.gross_lots:>12,.2f}"
          f"{net:>18,.0f}{gross:>18,.0f}  {row.notional_status}")

usable = result.loc[result["net_notional"].notna()]
print(f"\ntotal net  USD: {pd.to_numeric(usable['net_notional']).sum():>20,.0f}")
print(f"total gross USD: {pd.to_numeric(usable['gross_notional']).sum():>20,.0f}")
print(f"\nby conversion route:")
print(result["notional_status"].value_counts().to_string())
