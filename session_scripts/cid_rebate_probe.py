"""The reporting schema looks like the real source: tbl_cid_login (client id
-> logins = primary/sub-accounts?), tbl_abuse_db (the desk register in MySQL),
tbl_zfx_mt4_eod.rebate (daily rebate per login, current to yesterday). Pin
down columns, coverage and semantics."""
import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_colwidth", 60)
from webapp import event_abuse

def q(cur, sql, n=8, label=""):
    try:
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        print(f"\n--- {label or sql[:100]}")
        print(pd.DataFrame(rows[:n], columns=cols).to_string() if rows else "(no rows)")
        return rows
    except Exception as e:
        print(f"\n--- {label or sql[:100]} FAILED: {type(e).__name__}: {e}")
        return []

con = event_abuse._connect("mt4_live01")
cur = con.cursor()
for t in ("tbl_cid_login", "tbl_cid_toxic", "tbl_cid_equitytoxic", "tbl_last_cid_ts", "tbl_debug_cid", "tbl_abuse_db", "tbl_eod", "tbl_mt4_balances", "tbl_login_history", "tbl_zfx_directors"):
    q(cur, f"SHOW COLUMNS FROM reporting.{t}", 40, f"reporting.{t} columns")
    q(cur, f"SELECT * FROM reporting.{t} ORDER BY 1 DESC LIMIT 4", 4, f"reporting.{t} newest")
    q(cur, f"SELECT COUNT(*) FROM reporting.{t}", 1, f"reporting.{t} count")

q(cur, "SELECT COUNT(*), COUNT(DISTINCT cid), COUNT(DISTINCT login), MIN(login), MAX(login) FROM reporting.tbl_cid_login", 1, "cid_login: rows, cids, logins, login range")
q(cur, "SELECT n, COUNT(*) FROM (SELECT cid, COUNT(*) n FROM reporting.tbl_cid_login GROUP BY cid) t GROUP BY n ORDER BY n LIMIT 15", 15, "logins per cid distribution")
q(cur, "SELECT cid, COUNT(*) n, GROUP_CONCAT(login ORDER BY login) FROM reporting.tbl_cid_login GROUP BY cid HAVING n BETWEEN 2 AND 6 ORDER BY RAND() LIMIT 5", 5, "sample multi-login cids")
q(cur, "SELECT server, COUNT(*) FROM reporting.tbl_cid_login GROUP BY server", 20, "cid_login by server (if column exists)")

q(cur, "SELECT MIN(d), MAX(d), COUNT(*), COUNT(DISTINCT login), MIN(login), MAX(login) FROM reporting.tbl_zfx_mt4_eod", 1, "eod coverage")
q(cur, "SELECT DATE_FORMAT(d,'%Y-%m') m, COUNT(*) rows_, SUM(rebate<>0) rebate_rows, ROUND(SUM(rebate),2) rebate_sum, ROUND(SUM(net_deposit),0) netdep, ROUND(SUM(deposit),0) dep, ROUND(SUM(withdrawal),0) wd, ROUND(SUM(commission),0) comm FROM reporting.tbl_zfx_mt4_eod WHERE d >= '2026-01-01' GROUP BY 1 ORDER BY 1", 12, "eod monthly rebate/deposit sums 2026")
q(cur, "SELECT login, d, std_lots, rebate, trade_profit, commission, deposit, withdrawal, net_deposit FROM reporting.tbl_zfx_mt4_eod WHERE rebate <> 0 AND d >= '2026-09-01' ORDER BY ABS(rebate) DESC LIMIT 8", 8, "biggest September rebates")
q(cur, "SELECT login, COUNT(*) days, ROUND(SUM(rebate),2) reb, ROUND(SUM(std_lots),2) lots FROM reporting.tbl_zfx_mt4_eod WHERE d >= '2026-06-01' AND rebate <> 0 GROUP BY login ORDER BY reb DESC LIMIT 8", 8, "top rebate logins since June")
q(cur, "SELECT COUNT(DISTINCT login) FROM reporting.tbl_zfx_mt4_eod WHERE login >= 105000000", 1, "eod: MT5 logins present?")
q(cur, "SELECT COUNT(DISTINCT login) FROM reporting.tbl_zfx_mt4_eod WHERE d >= '2026-09-01' AND login BETWEEN 2000000 AND 2999999", 1, "eod: mt4_live02-range logins in Sept")
q(cur, "SELECT COUNT(DISTINCT login) FROM reporting.tbl_zfx_mt4_eod WHERE d >= '2026-09-01' AND login BETWEEN 6000000 AND 6999999", 1, "eod: mt4_live04-range logins in Sept")
q(cur, "SELECT COUNT(DISTINCT login) FROM reporting.tbl_zfx_mt4_eod WHERE d >= '2026-09-01' AND login BETWEEN 4000000 AND 4999999", 1, "eod: mt4_live03-range logins in Sept")

# Does a rebate in the EOD table show up as a balance operation on the login?
rows = q(cur, "SELECT login, d, rebate FROM reporting.tbl_zfx_mt4_eod WHERE rebate <> 0 AND d BETWEEN '2026-09-01' AND '2026-09-10' ORDER BY ABS(rebate) DESC LIMIT 3", 3, "rebate rows to cross-check")
for login, d, reb in rows:
    q(cur, f"SELECT tm, profit, type, comment FROM mt4_live01.balance_ops WHERE login={int(login)} AND tm BETWEEN '{d}' - INTERVAL 1 DAY AND '{d}' + INTERVAL 2 DAY ORDER BY tm", 12, f"balance_ops around {d} for {login} (rebate {reb})")
    q(cur, f"SELECT `order`, symbol_name, cmd, volume/100 lots, FROM_UNIXTIME(close_ts) closed, profit, commission, comment FROM mt4_live01.orders WHERE login={int(login)} AND close_ts BETWEEN UNIX_TIMESTAMP('{d} 00:00:00') AND UNIX_TIMESTAMP('{d} 23:59:59') AND cmd IN (0,1) ORDER BY close_ts LIMIT 6", 6, f"orders closed on {d} for {login}")

q(cur, "SHOW COLUMNS FROM mt5_live01.deals", 60, "mt5 deals columns")
q(cur, "SELECT COUNT(*), SUM(fee<>0), ROUND(SUM(fee),2), SUM(commission<>0), ROUND(SUM(commission),2) FROM mt5_live01.deals WHERE `time` >= '2026-09-01' AND action IN (0,1)", 1, "mt5 deals: fee / commission usage in Sept")
con.close()
