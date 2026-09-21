from google.cloud import bigquery

client = bigquery.Client(project='zfx-dwh-prod')

targets = [
    ("topbook", "topbook_2024"),
    ("topbook", "topbook_20250618"),
    ("topbook", "topbook_20250715"),
    ("topbook", "topbook_20250716"),
    ("topbook", "topbook_dq_bid_ask_2024"),
    ("topbook", "topbook_dq_bid_ask_test_20250916"),
    ("topbook", "symbols_test"),
    ("topbook", "price_delay_hourly_all_hours"),
    ("markouts_v2", "markouts"),
    ("ecn_reports", "FilteredQuotes"),
    ("ecn_reports", "FilteredBookQuotes"),
    ("ecn_ts", "OrdersJournal"),
    ("ecn_ts", "DoneTrades"),
]

for ds, tbl in targets:
    try:
        table = client.get_table(f"zfx-dwh-prod.{ds}.{tbl}")
    except Exception as e:
        print(f"{ds}.{tbl}  ERROR: {e}")
        continue
    print(f"{ds}.{tbl}: rows={table.num_rows}  bytes={table.num_bytes}  created={table.created}  modified={table.modified}")
