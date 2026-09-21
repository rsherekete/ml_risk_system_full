"""Continue after the dropped connection: cid semantics, EOD rebate coverage
and semantics, MT5 deal columns. Fresh connection per block."""
import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_colwidth", 60)
from webapp import event_abuse

def run(block, label):
    con = event_abuse._connect("mt4_live01")
    cur = con.cursor()
    for sql, n, lab in block:
        try:
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
            print(f"\n--- {lab}")
            print(pd.DataFrame(rows[:n], columns=cols).to_string() if rows else "(no rows)")
        except Exception as e:
            print(f"\n--- {lab} FAILED: {type(e).__name__}: {e}")
            break
    con.close()

run([
    ("SELECT COUNT(DISTINCT cid) cids, COUNT(DISTINCT login) logins FROM reporting.tbl_cid_login WHERE login BETWEEN 1044600 AND 1044620", 1, "cid_login: sample login range -> cids/logins"),
    ("SELECT login, COUNT(*) cids FROM reporting.tbl_cid_login WHERE login IN (1044610, 969683, 2403247, 6869677, 105034975) GROUP BY login", 10, "cids per known login"),
    ("SELECT cid, COUNT(*) n FROM reporting.tbl_cid_login WHERE login IN (1044610, 969683) GROUP BY cid ORDER BY n DESC LIMIT 5", 5, "cids of two logins"),
    ("SELECT l2.login FROM reporting.tbl_cid_login l1 JOIN reporting.tbl_cid_login l2 ON l1.cid = l2.cid WHERE l1.login = 1044610 AND l2.login <> 1044610 GROUP BY l2.login LIMIT 20", 20, "other logins sharing a cid with 1044610"),
], "cid")

run([
    ("SELECT MIN(d), MAX(d), COUNT(*), COUNT(DISTINCT login), MIN(login), MAX(login) FROM reporting.tbl_zfx_mt4_eod WHERE d >= '2026-08-01'", 1, "eod coverage since Aug"),
    ("SELECT DATE_FORMAT(d,'%Y-%m') m, COUNT(*) rows_, SUM(rebate<>0) rebate_rows, ROUND(SUM(rebate),2) rebate_sum, ROUND(SUM(net_deposit),0) netdep, ROUND(SUM(deposit),0) dep, ROUND(SUM(withdrawal),0) wd, ROUND(SUM(commission),0) comm, ROUND(SUM(std_lots),0) lots FROM reporting.tbl_zfx_mt4_eod WHERE d >= '2026-06-01' GROUP BY 1 ORDER BY 1", 12, "eod monthly sums since June"),
    ("SELECT login, d, std_lots, rebate, trade_profit, commission, deposit, withdrawal, net_deposit, `group` FROM reporting.tbl_zfx_mt4_eod WHERE rebate <> 0 AND d >= '2026-09-01' ORDER BY ABS(rebate) DESC LIMIT 8", 8, "biggest September rebates"),
    ("SELECT login, COUNT(*) days, ROUND(SUM(rebate),2) reb, ROUND(SUM(std_lots),2) lots, ROUND(SUM(rebate)/NULLIF(SUM(std_lots),0),2) per_lot FROM reporting.tbl_zfx_mt4_eod WHERE d >= '2026-06-01' AND rebate <> 0 GROUP BY login ORDER BY reb DESC LIMIT 8", 8, "top rebate logins since June"),
    ("SELECT SUM(login >= 105000000) mt5_rows, SUM(login BETWEEN 2000000 AND 2999999) l02, SUM(login BETWEEN 4000000 AND 4999999) l03, SUM(login BETWEEN 6000000 AND 6999999) l04, SUM(login < 2000000) l01 FROM reporting.tbl_zfx_mt4_eod WHERE d >= '2026-09-01'", 1, "eod: login ranges present in Sept (server coverage)"),
], "eod")

run([
    ("SELECT login, d, rebate FROM reporting.tbl_zfx_mt4_eod WHERE rebate <> 0 AND d BETWEEN '2026-09-01' AND '2026-09-10' AND login < 2000000 ORDER BY ABS(rebate) DESC LIMIT 2", 2, "rebate rows to cross-check (mt4_live01 range)"),
], "pick")

con = event_abuse._connect("mt4_live01"); cur = con.cursor()
cur.execute("SELECT login, d, rebate FROM reporting.tbl_zfx_mt4_eod WHERE rebate <> 0 AND d BETWEEN '2026-09-01' AND '2026-09-10' AND login < 2000000 ORDER BY ABS(rebate) DESC LIMIT 2")
picks = cur.fetchall(); con.close()
for login, d, reb in picks:
    run([
        (f"SELECT tm, profit, type, comment FROM mt4_live01.balance_ops WHERE login={int(login)} AND tm BETWEEN '{d}' - INTERVAL 1 DAY AND '{d}' + INTERVAL 3 DAY ORDER BY tm", 12, f"balance_ops around {d} for {login} (eod rebate {reb})"),
        (f"SELECT `order`, symbol_name, cmd, volume/100 lots, FROM_UNIXTIME(close_ts) closed, profit, commission, swaps, comment FROM mt4_live01.orders WHERE login={int(login)} AND close_ts BETWEEN UNIX_TIMESTAMP('{d} 00:00:00') AND UNIX_TIMESTAMP('{d} 23:59:59') AND cmd IN (0,1) ORDER BY close_ts LIMIT 6", 6, f"orders closed on {d} for {login}"),
        (f"SELECT `group`, agent_account, comment FROM mt4_live01.accounts WHERE login={int(login)}", 1, f"account {login}"),
    ], "xcheck")

run([
    ("SHOW COLUMNS FROM mt5_live01.deals", 60, "mt5 deals columns"),
    ("SELECT COUNT(*), SUM(fee<>0), ROUND(SUM(fee),2), SUM(commission<>0), ROUND(SUM(commission),2) FROM mt5_live01.deals WHERE `time` >= '2026-09-01' AND action IN (0,1)", 1, "mt5 deals: fee / commission usage in Sept"),
    ("SELECT AbuseType, COUNT(*), ROUND(SUM(AmountOfSaved)), ROUND(SUM(AmountOfLoss)), MIN(DateOfAbuse), MAX(DateOfAbuse) FROM reporting.tbl_abuse_db GROUP BY 1 ORDER BY 2 DESC", 20, "tbl_abuse_db by type"),
], "mt5+abuse")
