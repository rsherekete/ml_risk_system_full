"""(1) MySQL: what the rebate tables hold (columns, samples, ranges, how a
row ties to a trade / login); (2) BigQuery zfx-dwh-prod: any column called
rebate_payout / primary_trading_account_number, and which tables."""
import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_colwidth", 40)
from webapp import event_abuse

def q(cur, sql, n=6, label=""):
    try:
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        print(f"\n--- {label or sql[:90]}")
        print(pd.DataFrame(rows[:n], columns=cols).to_string() if rows else "(no rows)")
        return rows
    except Exception as e:
        print(f"\n--- {label or sql[:90]} FAILED: {type(e).__name__}: {e}")
        return []

con = event_abuse._connect("mt4_live01")
cur = con.cursor()
q(cur, "SHOW COLUMNS FROM mt4_live01.ib_rebates_summary", 40, "ib_rebates_summary columns")
q(cur, "SELECT * FROM mt4_live01.ib_rebates_summary ORDER BY 1 DESC LIMIT 5", 5, "ib_rebates_summary newest")
q(cur, "SHOW COLUMNS FROM mt4_live01.debug_timer_ib_rebates", 40, "debug_timer_ib_rebates columns")
q(cur, "SELECT * FROM mt4_live01.debug_timer_ib_rebates ORDER BY 1 DESC LIMIT 3", 3, "debug_timer_ib_rebates newest")
q(cur, "SHOW COLUMNS FROM mt4_live01.rep_client_volumes", 40, "rep_client_volumes columns")
q(cur, "SELECT * FROM mt4_live01.rep_client_volumes ORDER BY 1 DESC LIMIT 3", 3, "rep_client_volumes newest")
q(cur, "SHOW COLUMNS FROM rep_general.profit_summary", 60, "rep_general.profit_summary columns")
q(cur, "SELECT * FROM rep_general.profit_summary ORDER BY 1 DESC LIMIT 3", 3, "profit_summary newest")
q(cur, "SHOW COLUMNS FROM reporting.tbl_zfx_mt4_eod", 60, "reporting.tbl_zfx_mt4_eod columns")
q(cur, "SELECT * FROM reporting.tbl_zfx_mt4_eod ORDER BY 1 DESC LIMIT 3", 3, "tbl_zfx_mt4_eod newest")
q(cur, "SHOW COLUMNS FROM reporting.tbl_ib_toxic", 30, "reporting.tbl_ib_toxic columns")
q(cur, "SELECT * FROM reporting.tbl_ib_toxic LIMIT 5", 5, "tbl_ib_toxic sample")
q(cur, "SHOW COLUMNS FROM mt5_live01.concommission", 40, "mt5_live01.concommission columns")
q(cur, "SELECT * FROM mt5_live01.concommission ORDER BY 1 DESC LIMIT 3", 3, "concommission newest")
q(cur, "SHOW COLUMNS FROM mt5_live01.commissions", 40, "mt5_live01.commissions columns")
q(cur, "SELECT * FROM mt5_live01.commissions LIMIT 3", 3, "commissions sample")
q(cur, "SHOW COLUMNS FROM mt5_real01.mt5_clients", 40, "mt5_real01.mt5_clients columns")
q(cur, "SELECT * FROM mt5_real01.mt5_clients LIMIT 3", 3, "mt5_clients sample")
q(cur, "SHOW TABLES FROM mt5_real01", 80, "mt5_real01 tables")
q(cur, "SHOW TABLES FROM rep_general", 80, "rep_general tables")
q(cur, "SHOW TABLES FROM reporting", 120, "reporting tables")
q(cur, "SHOW TABLES FROM mt4_svc", 80, "mt4_svc tables")
con.close()

print("\n=== BigQuery zfx-dwh-prod")
try:
    from google.cloud import bigquery
    client = bigquery.Client(project="zfx-dwh-prod")
    dss = [d.dataset_id for d in client.list_datasets()]
    print("datasets:", dss)
    for ds in dss:
        try:
            sql = (f"SELECT table_name, column_name, data_type FROM `zfx-dwh-prod.{ds}.INFORMATION_SCHEMA.COLUMNS` "
                   f"WHERE REGEXP_CONTAINS(LOWER(column_name), r'rebate|payout|primary_trading|trading_account|sub_account|subaccount') LIMIT 200")
            rows = list(client.query(sql).result(timeout=120))
            if rows:
                print(f"  [{ds}]")
                for r in rows:
                    print("    ", r.table_name, r.column_name, r.data_type)
        except Exception as e:
            print(f"  [{ds}] {type(e).__name__}: {str(e)[:160]}")
except Exception as e:
    print("BigQuery:", type(e).__name__, str(e)[:400])
