"""Where do `rebate_payout` (per trade) and `primary_trading_account_number`
live? Search every schema visible on each MySQL host, then try BigQuery."""
import sys, re
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import event_abuse

PAT = "rebate|payout|primary_trading|trading_account|client_id|crm_id|customer_id|sub_account|subaccount|parent_account|master_account"
seen_hosts = set()
cfg, dbs = event_abuse._config()
for server in dbs:
    host = cfg["servers"].get(server, {}).get("host")
    if not host or host in seen_hosts:
        continue
    seen_hosts.add(host)
    try:
        con = event_abuse._connect(server)
        with con.cursor() as cur:
            cur.execute("SHOW DATABASES")
            schemas = [r[0] for r in cur.fetchall()]
            print(f"\n=== {server} @ {host}: schemas {schemas}")
            cur.execute("SELECT table_schema, table_name, column_name, data_type FROM information_schema.columns "
                        "WHERE column_name REGEXP %s AND table_schema NOT IN ('information_schema','mysql','performance_schema','sys') "
                        "ORDER BY 1,2,3", (PAT,))
            rows = cur.fetchall()
            print(f"  columns matching /{PAT}/: {len(rows)}")
            for r in rows[:80]:
                print("   ", r)
            cur.execute("SELECT table_schema, table_name, table_rows FROM information_schema.tables "
                        "WHERE table_name REGEXP 'rebate|payout|commission|partner|ib_|agent|client|crm|link|profile' "
                        "AND table_schema NOT IN ('information_schema','mysql','performance_schema','sys') ORDER BY 1,2")
            print("  tables of interest:")
            for r in cur.fetchall()[:60]:
                print("   ", r)
        con.close()
    except Exception as e:
        print(f"{server}: {type(e).__name__}: {e}")

print("\n=== BigQuery reachability")
try:
    from google.cloud import bigquery
    client = bigquery.Client()
    print("project:", client.project)
    n = 0
    for ds in client.list_datasets(max_results=50):
        n += 1
        print("  dataset:", ds.dataset_id)
    print("datasets listed:", n)
except Exception as e:
    print("BigQuery:", type(e).__name__, str(e)[:300])
