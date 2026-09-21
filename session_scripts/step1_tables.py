import sys; sys.path.insert(0, r'c:\Users\RoyVivasi\Documents\notebook')

from trading_data.research import clients_from_yaml, platform_for_database

clients = clients_from_yaml()

for db, client in clients.items():
    platform = platform_for_database(db)
    print(f"\n===== {db} ({platform}) =====")
    try:
        t = client.tables()
    except Exception as e:
        print(f"  ERROR listing tables: {e}")
        continue
    # print all tables sorted by name for readability, with rows/mb
    t2 = t.sort_values("table_name")
    for _, row in t2.iterrows():
        print(f"  {row['table_name']:<30} rows~={row['table_rows']:<12} data_mb={row['data_mb']}")
