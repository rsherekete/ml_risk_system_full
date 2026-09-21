"""Sanity-check the lifetime figures of the biggest clients straight from
MySQL: deal span, lots, notional, and the cash store's first/last movement."""
import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import pandas as pd
from webapp import event_abuse, cashflow_store

checks = [("mt5_live01", 105034975), ("mt4_live02", 2969082), ("mt4_live02", 2891280), ("mt4_live01", 1062886)]
flows = cashflow_store.read_cashflows()
flows["account_key"] = flows["account_key"].astype(str)
for server, login in checks:
    key = f"{server}:{login}"
    con = event_abuse._connect(server)
    with con.cursor() as cur:
        if server.startswith("mt5"):
            cur.execute("SELECT COUNT(*), MIN(`time`), MAX(`time`), SUM(volume)/10000, SUM(volume*price)/10000 "
                        "FROM deals WHERE login=%s AND action IN (0,1) AND entry IN (0,2)", (login,))
            n, t0, t1, lots, lxp = cur.fetchone()
            cur.execute("SELECT symbol, COUNT(*), SUM(volume)/10000 FROM deals WHERE login=%s AND action IN (0,1) AND entry IN (0,2) GROUP BY symbol ORDER BY 3 DESC LIMIT 5", (login,))
        else:
            cur.execute("SELECT COUNT(*), FROM_UNIXTIME(MIN(open_ts)), FROM_UNIXTIME(MAX(GREATEST(open_ts, close_ts))), SUM(volume)/100, SUM(volume*open_price)/100 "
                        "FROM orders WHERE login=%s AND cmd IN (0,1)", (login,))
            n, t0, t1, lots, lxp = cur.fetchone()
            cur.execute("SELECT symbol_name, COUNT(*), SUM(volume)/100 FROM orders WHERE login=%s AND cmd IN (0,1) GROUP BY symbol_name ORDER BY 3 DESC LIMIT 5", (login,))
        top = cur.fetchall()
        cur.execute("SELECT name, country, `group`, balance, credit FROM accounts WHERE login=%s", (login,))
        acct = cur.fetchone()
    con.close()
    f = flows[flows["account_key"] == key]
    print(f"\n{key}: {acct}")
    print(f"  deals/orders {n} | span {t0} -> {t1} | lots {float(lots or 0):,.2f} | sum(lots*price) {float(lxp or 0):,.0f}")
    print(f"  top symbols: {top}")
    print(f"  cashflows {len(f)} | first {f['when'].min()} | last {f['when'].max()} | dep {f.loc[f['kind']=='deposit','amount'].sum():,.0f} | wd {f.loc[f['kind']=='withdrawal','amount'].sum():,.0f}")
