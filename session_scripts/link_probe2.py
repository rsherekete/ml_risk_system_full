"""Deeper: (1) do MBO / DT / WT transfer comments carry the counterpart login
(a hard link between accounts of one client)? (2) what does MT4 balance_ops
type='Other' contain (rebates?) and do MT5 balance deals ever say rebate?
(3) name+country collisions as a linkage key."""
import sys, re
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import pandas as pd
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 60)
from webapp import cashflow_store, event_abuse

flows = cashflow_store.read_cashflows(start=pd.Timestamp("2026-01-01"))
c = flows["comment"].astype(str)
for label, rx in (("MBO", r"^MBO"), ("DT-", r"^DT-"), ("WT-", r"^WT-"), ("REFUND", r"REFUND"), ("IB-", r"^IB-"), ("AGENT", r"^AGENT"), ("PPF", r"^PPF"), ("FEE", r"^FEE")):
    m = c.str.contains(rx, regex=True)
    print(f"\n--- {label}: {int(m.sum())} rows; samples:")
    print(flows.loc[m, ["database", "account_key", "when", "amount", "kind", "comment"]].sample(min(8, int(m.sum())), random_state=1).to_string())

for server in ("mt4_live01", "mt5_live01"):
    try:
        con = event_abuse._connect(server)
        with con.cursor() as cur:
            if server.startswith("mt4"):
                cur.execute("SELECT LEFT(comment, 24), COUNT(*), ROUND(SUM(profit)) FROM balance_ops WHERE type='Other' AND tm >= '2026-01-01' GROUP BY 1 ORDER BY 2 DESC LIMIT 30")
                print(f"\n=== {server} balance_ops type=Other comment prefixes")
                for r in cur.fetchall(): print("   ", r)
                cur.execute("SELECT comment, COUNT(*), ROUND(SUM(profit)) FROM balance_ops WHERE tm >= '2025-09-01' AND UPPER(comment) REGEXP 'REBATE|CASHBACK|CASH BACK|COMMISSION|COMM REB|KICKBACK' GROUP BY 1 ORDER BY 2 DESC LIMIT 20")
                print(f"=== {server} rebate-word comments (12m)")
                for r in cur.fetchall(): print("   ", r)
            else:
                cur.execute("SELECT LEFT(comment, 24), COUNT(*), ROUND(SUM(profit)) FROM deals WHERE action=2 AND time >= '2026-01-01' GROUP BY 1 ORDER BY 2 DESC LIMIT 40")
                print(f"\n=== {server} deals action=2 comment prefixes")
                for r in cur.fetchall(): print("   ", r)
                cur.execute("SELECT comment, COUNT(*), ROUND(SUM(profit)) FROM deals WHERE action=2 AND time >= '2025-09-01' AND UPPER(comment) REGEXP 'REBATE|CASHBACK|CASH BACK|COMMISSION|KICKBACK' GROUP BY 1 ORDER BY 2 DESC LIMIT 20")
                print(f"=== {server} rebate-word comments (12m)")
                for r in cur.fetchall(): print("   ", r)
                cur.execute("SELECT COUNT(*), SUM(commission_agent_monthly<>0), SUM(agent<>0) FROM accounts")
                print("    accounts: n, agent_comm_monthly<>0, agent<>0 =", cur.fetchone())
            # name + country collisions
            cur.execute("SELECT COUNT(*) FROM (SELECT UPPER(TRIM(name)) n, country, COUNT(*) k FROM accounts WHERE balance<>0 GROUP BY 1,2 HAVING k>1) t")
            print("    name+country groups with >1 funded login:", cur.fetchone())
            cur.execute("SELECT UPPER(TRIM(name)), country, COUNT(*) k, GROUP_CONCAT(login) FROM accounts WHERE balance<>0 GROUP BY 1,2 HAVING k>1 ORDER BY k DESC LIMIT 5")
            for r in cur.fetchall(): print("    ", r)
        con.close()
    except Exception as e:
        print(f"{server}: {type(e).__name__}: {e}")
