"""Server naming and coverage of data_marts.closed_trades / trading_accounts,
and the rebate_payout figures around the 4 Sep window."""
import sys
WIFI_IP = sys.argv[1] if len(sys.argv) > 1 else "10.240.17.21"
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
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

def q(sql, label):
    try:
        df = client.query(sql).to_dataframe()
        print(f"\n--- {label}\n{df.to_string()}")
        return df
    except Exception as e:
        print(f"\n--- {label} FAILED: {type(e).__name__}: {str(e)[:220]}")

q("""SELECT server_name, COUNT(*) n, COUNT(DISTINCT login) logins, MIN(login) lo, MAX(login) hi,
            COUNTIF(rebate_payout IS NOT NULL AND rebate_payout <> 0) reb_rows, ROUND(SUM(rebate_payout), 0) reb_sum,
            ROUND(SUM(qty_usd), 0) qty_usd, COUNTIF(primary_trading_account_number IS NULL) prim_null
     FROM `zfx-dwh-prod.data_marts.closed_trades`
     WHERE close_time_london_trading_date BETWEEN '2026-09-01' AND '2026-09-13'
     GROUP BY 1 ORDER BY 2 DESC""", "data_marts.closed_trades by server_name, Sept 1-13")

q("""SELECT server_name, COUNT(*) n, COUNT(DISTINCT login) logins, MIN(login) lo, MAX(login) hi,
            COUNTIF(rebate_payout <> 0) reb_rows, ROUND(SUM(rebate_payout), 0) reb_sum
     FROM `zfx-dwh-prod.data_marts_multi_region.closed_trades`
     WHERE close_time_london_trading_date BETWEEN '2026-09-01' AND '2026-09-13'
     GROUP BY 1 ORDER BY 2 DESC""", "data_marts_multi_region.closed_trades by server_name, Sept 1-13")

q("""SELECT server_name, login, trade_id, symbol_name, volume_lots, qty_usd, trade_profit_usd, commission_usd,
            rebate_payout_markup, rebate_payout_commission, rebate_payout_subsidy, rebate_payout, primary_trading_account_number, close_time
     FROM `zfx-dwh-prod.data_marts.closed_trades`
     WHERE close_time_london_trading_date = '2026-09-04' AND rebate_payout <> 0
     ORDER BY rebate_payout DESC LIMIT 8""", "biggest rebate_payout trades on 4 Sep")

q("""SELECT COUNT(*) trades, COUNT(DISTINCT login) logins, ROUND(SUM(rebate_payout),0) reb_sum, COUNTIF(rebate_payout <> 0) reb_rows
     FROM `zfx-dwh-prod.data_marts.closed_trades`
     WHERE close_time_london_trading_date BETWEEN '2024-09-01' AND '2026-09-13' AND login IN (105034975, 2891280, 2969082, 1062886, 1044610)""",
  "known logins over 2 years: rebate totals")

q("""SELECT login, server_name, COUNT(*) trades, ROUND(SUM(rebate_payout),2) reb, ROUND(SUM(qty_usd),0) qty_usd, MIN(close_time) t0, MAX(close_time) t1
     FROM `zfx-dwh-prod.data_marts.closed_trades`
     WHERE close_time_london_trading_date BETWEEN '2024-09-01' AND '2026-09-13' AND login IN (105034975, 2891280, 2969082, 1062886, 1044610)
     GROUP BY 1,2""", "known logins per login")

q("""SELECT trading_server, COUNT(*) n, COUNTIF(primary_trading_account_number IS NULL) prim_null,
            COUNTIF(primary_trading_account_number <> account_number) subs, COUNT(DISTINCT primary_trading_account_number) primaries,
            MIN(account_number) lo, MAX(account_number) hi
     FROM `zfx-dwh-prod.data_marts.trading_accounts` GROUP BY 1 ORDER BY 2 DESC""", "trading_accounts by trading_server")

q("""SELECT trading_server, account_number, primary_trading_account_number, parent_trading_account_number, type, currency,
            total_deposit_usd, net_deposit_usd, total_withdrawal_usd, total_volume_usd, total_rebates_generated_usd, equity_usd, last_trade_datetime
     FROM `zfx-dwh-prod.data_marts.trading_accounts`
     WHERE account_number IN (105034975, 2891280, 2969082, 1062886, 1044610, 6825721, 6834256)""", "trading_accounts rows for known logins")

q("""SELECT primary_trading_account_number, COUNT(*) n, ARRAY_TO_STRING(ARRAY_AGG(CONCAT(trading_server, ':', CAST(account_number AS STRING)) ORDER BY account_number LIMIT 12), ', ') members
     FROM `zfx-dwh-prod.data_marts.trading_accounts`
     WHERE primary_trading_account_number IN (SELECT primary_trading_account_number FROM `zfx-dwh-prod.data_marts.trading_accounts` WHERE account_number IN (1062886, 6825721, 6834256))
     GROUP BY 1""", "members of the primaries behind three 'linked' logins")

q("""SELECT n, COUNT(*) primaries FROM (SELECT primary_trading_account_number, COUNT(*) n FROM `zfx-dwh-prod.data_marts.trading_accounts`
     WHERE primary_trading_account_number IS NOT NULL GROUP BY 1) GROUP BY 1 ORDER BY 1 LIMIT 12""", "accounts per primary distribution")
