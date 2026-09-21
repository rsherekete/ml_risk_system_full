"""What account-financial data exists? Equity, margin, leverage, stop levels.

Every model so far has been behavioural. None of them know how big a client's
position is RELATIVE TO THEIR EQUITY -- which is the arithmetic driver of how
much they can lose, and plausibly why magnitude has been unpredictable.
"""
from google.cloud import bigquery
client = bigquery.Client(project="zfx-dwh-prod")

RISK_TERMS = ("balance", "equity", "margin", "credit", "leverage", "free", "level", "stop", "sl", "tp", "profit")

for ref in ("mt5_live01.accounts", "mt5_live01.userrecord", "mt5_live01.dailyrecord",
            "mt4_live01.accounts", "mt4_live01.userrecord", "mt4_live01.orders",
            "mt5_live01.deals", "mt5_live01.positions"):
    try:
        table = client.get_table(f"zfx-dwh-prod.{ref}")
        matched = [f.name for f in table.schema if any(t in f.name.lower() for t in RISK_TERMS)]
        part = table.time_partitioning.field if table.time_partitioning else None
        print(f"{ref:<28} {table.num_rows:>14,} rows  part={str(part):<20}")
        print(f"    risk-relevant columns: {matched}\n")
    except Exception as exc:
        print(f"{ref:<28} FAILED {type(exc).__name__}: {str(exc)[:70]}\n")

# Does the MT4 userrecord CDC log carry per-moment balance/margin?
try:
    t = client.get_table("zfx-dwh-prod.mt4_live01.userrecord")
    print("mt4_live01.userrecord full column list:")
    print([f.name for f in t.schema])
except Exception as exc:
    print(f"userrecord detail failed: {exc}")
