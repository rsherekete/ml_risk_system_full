"""Schemas, sizes, partitioning and recent coverage of the DWH tables that
carry rebate_payout / primary_trading_account_number (source-bound transport)."""
import sys, time, re
WIFI_IP = sys.argv[1] if len(sys.argv) > 1 else "10.240.17.21"
import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
import google.auth
from google.auth.transport.requests import AuthorizedSession
from google.cloud import bigquery

class BoundAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False, **kw):
        kw["source_address"] = (WIFI_IP, 0)
        self.poolmanager = PoolManager(num_pools=connections, maxsize=maxsize, block=block, **kw)

creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
session = AuthorizedSession(creds); session.mount("https://", BoundAdapter())
client = bigquery.Client(project="zfx-dwh-prod", _http=session)

try:
    import psutil
    print("psutil interfaces:", {k: [a.address for a in v if a.family.name == 'AF_INET'] for k, v in psutil.net_if_addrs().items()})
except Exception as e:
    print("psutil:", type(e).__name__, e)

TABLES = ["data_marts.trading_accounts", "data_marts.accounts", "data_marts.closed_trades", "data_marts.closed_trades_hist",
          "data_marts.mt5_live01_closed_trades", "data_marts_multi_region.closed_trades", "data_marts_multi_region.accounts",
          "data_marts.transactions", "data_marts_multi_region.mt5_dubai_live01_closed_trades", "data_marts_multi_region.cid_login_mapping"]
for t in TABLES:
    try:
        tb = client.get_table(f"zfx-dwh-prod.{t}")
        cols = [f"{f.name}:{f.field_type}" for f in tb.schema]
        part = tb.time_partitioning.field if tb.time_partitioning else None
        print(f"\n=== {t}: rows {tb.num_rows:,} | {tb.num_bytes/1e9:.2f} GB | type {tb.table_type} | partition {part} | cluster {tb.clustering_fields}")
        print("   ", ", ".join(cols)[:1600])
        tcol = part or next((f.name for f in tb.schema if re.search(r"close_time|closed_at|close_ts|close_date", f.name, re.I)), None)
        keycols = [f.name for f in tb.schema if f.name in ("login", "trading_account_number", "primary_trading_account_number", "server", "server_name", "platform", "rebate_payout", "trade_id", "order_id", "ticket")]
        if tcol and "rebate_payout" in [f.name for f in tb.schema]:
            sql = (f"SELECT COUNT(*) n, MIN({tcol}) t0, MAX({tcol}) t1, SUM(rebate_payout <> 0) reb_rows, ROUND(SUM(rebate_payout),2) reb_sum "
                   f"FROM `zfx-dwh-prod.{t}` WHERE {tcol} >= TIMESTAMP('2026-08-01')")
            r = list(client.query(sql).result(timeout=120))[0]
            print(f"    since Aug: rows {r.n:,} | {r.t0} -> {r.t1} | rebate rows {r.reb_rows:,} | rebate sum {r.reb_sum}")
            sql = f"SELECT {', '.join(keycols)} FROM `zfx-dwh-prod.{t}` WHERE {tcol} >= TIMESTAMP('2026-09-04') AND rebate_payout <> 0 LIMIT 3"
            for r in client.query(sql).result(timeout=120):
                print("    sample:", dict(r))
        elif "primary_trading_account_number" in [f.name for f in tb.schema] and tb.num_rows and tb.num_rows < 50_000_000 and not tcol:
            cand = [c for c in keycols if c != "rebate_payout"]
            sql = f"SELECT {', '.join(cand)} FROM `zfx-dwh-prod.{t}` WHERE primary_trading_account_number <> trading_account_number LIMIT 3" \
                if "trading_account_number" in cand else f"SELECT {', '.join(cand)} FROM `zfx-dwh-prod.{t}` LIMIT 3"
            try:
                for r in client.query(sql).result(timeout=120):
                    print("    sample:", dict(r))
                if "trading_account_number" in cand:
                    r = list(client.query(f"SELECT COUNT(*) n, COUNT(DISTINCT primary_trading_account_number) prim, SUM(primary_trading_account_number <> trading_account_number) subs FROM `zfx-dwh-prod.{t}`").result(timeout=120))[0]
                    print(f"    accounts {r.n:,} | primaries {r.prim:,} | rows that are sub-accounts {r.subs:,}")
            except Exception as e:
                print("    sample failed:", str(e)[:160])
    except Exception as e:
        print(f"\n=== {t}: {type(e).__name__}: {str(e)[:200]}")
