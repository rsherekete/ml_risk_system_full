from google.cloud import bigquery

client = bigquery.Client(project='zfx-dwh-prod')

targets = [
    ("ecn_ts", "DoneTrades"),
    ("ecn_ts", "DoneTrades_temp"),
    ("ecn_ts", "OrdersJournal"),
    ("ecn_ts", "TradeRequests"),
    ("ecn_ts", "NetPositions"),
    ("ecn_ts", "NetPositionSnapshots"),
    ("ecn_ts", "QuoteCache"),
    ("ecn_ts", "LiquiditySettingsData"),
    ("ecn_ts", "DoneTradeRouteData"),
    ("ecn_ts", "TradeMessageSequences"),
    ("ecn_ts_asia", "LiquiditySettingsData"),
    ("ecn_ts_asia", "Symbols"),
    ("ecn_configuration", "LiquidityPools"),
    ("ecn_configuration", "Banks"),
    ("ecn_configuration", "FixConnectors"),
    ("ecn_configuration", "ClientTradePermissions"),
    ("ecn_configuration", "Bank2Connectors"),
    ("ecn_reports", "BankExecutionReports"),
    ("ecn_reports", "BankOrders"),
    ("ecn_reports", "BankQuotesOld"),
    ("ecn_reports", "ExecutionReports"),
    ("ecn_reports", "FilteredQuotes"),
    ("ecn_reports", "FilteredBookQuotes"),
    ("bank_quotes", "bank_quotes"),
    ("bank_quotes", "bank_quotes_gold"),
    ("bank_quotes", "bank_quotes_dbs04p"),
    ("bank_quotes", "bank_quotes_6"),
    ("bankquotes_sandbox", "banquotes_meta_last"),
    ("bankquotes_sandbox", "closed_by_server_jul2026"),
    ("bankquotes_sandbox", "closed_vs_sales_jul2026_summary"),
    ("exness_ticks_from_site", "raw_exness_ticks_external"),
    ("topbook", "topbook"),
    ("topbook", "price_delay"),
    ("topbook", "price_delay_hourly"),
    ("topbook", "senders"),
    ("topbook", "symbols"),
    ("topbook", "symbol_family"),
    ("markouts_v2", "markouts"),
    ("markouts_v2", "markouts_order_close"),
    ("markouts_v2", "markouts_order_open"),
]

for ds, tbl in targets:
    print(f"\n{'='*70}\n{ds}.{tbl}\n{'='*70}")
    try:
        table = client.get_table(f"zfx-dwh-prod.{ds}.{tbl}")
    except Exception as e:
        print(f"  ERROR: {e}")
        continue
    print(f"  table_type: {table.table_type}")
    print(f"  description: {table.description!r}")
    print(f"  num_rows: {table.num_rows}")
    print(f"  num_bytes: {table.num_bytes}")
    print(f"  created: {table.created}")
    print(f"  modified: {table.modified}")
    if table.time_partitioning:
        tp = table.time_partitioning
        print(f"  time_partitioning: type={tp.type_} field={tp.field}")
    if table.range_partitioning:
        print(f"  range_partitioning: {table.range_partitioning}")
    if table.clustering_fields:
        print(f"  clustering_fields: {table.clustering_fields}")
    if table.external_data_configuration:
        edc = table.external_data_configuration
        print(f"  EXTERNAL source_format: {edc.source_format}")
        print(f"  EXTERNAL source_uris: {edc.source_uris}")
    print(f"  schema ({len(table.schema)} fields):")
    for f in table.schema:
        print(f"    - {f.name}: {f.field_type} ({f.mode}) desc={f.description!r}")
