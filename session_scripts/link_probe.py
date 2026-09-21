"""What links accounts of one client (MT4 + MT5 `accounts` columns), what a
rebate looks like in the cashflow store, and what the warehouse rows carry."""
import sys, re, glob
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import pandas as pd
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)

from webapp import event_abuse
for server in ("mt4_live01", "mt5_live01"):
    try:
        con = event_abuse._connect(server)
        with con.cursor() as cur:
            cur.execute("SHOW COLUMNS FROM accounts")
            cols = [r[0] for r in cur.fetchall()]
            print(f"\n=== {server} accounts columns ({len(cols)}):", cols)
            cand = [c for c in cols if re.search(r"client|email|mail|phone|id$|^id|name|agent|lead|zip|address|country|regist|regdate|comment", c, re.I)]
            cur.execute(f"SELECT {', '.join('`'+c+'`' for c in cand)} FROM accounts WHERE balance <> 0 LIMIT 3")
            for r in cur.fetchall():
                print("   ", dict(zip(cand, r)))
            # how many logins share an identity key?
            for key in ("email", "client_id", "id", "phone", "name"):
                if key in cols:
                    cur.execute(f"SELECT COUNT(*), COUNT(DISTINCT `{key}`), SUM(`{key}` IS NULL OR `{key}`='') FROM accounts")
                    print(f"    {key}: rows/distinct/empty =", cur.fetchone())
            if server.startswith("mt4"):
                cur.execute("SHOW COLUMNS FROM balance_ops")
                print("    balance_ops columns:", [r[0] for r in cur.fetchall()])
                cur.execute("SELECT type, COUNT(*), ROUND(SUM(profit)) FROM balance_ops WHERE tm >= '2026-06-01' GROUP BY type ORDER BY 2 DESC LIMIT 25")
                for r in cur.fetchall():
                    print("    type:", r)
        con.close()
    except Exception as e:
        print(f"{server}: {type(e).__name__}: {e}")

print("\n=== cashflow store comment patterns (2026-06 .. now)")
from webapp import cashflow_store
flows = cashflow_store.read_cashflows(start=pd.Timestamp("2026-06-01"))
print("rows", len(flows), "cols", list(flows.columns))
print(flows["kind"].value_counts().to_string())
if "comment" in flows.columns:
    c = flows["comment"].astype(str).str.upper()
    pat = c.str.replace(r"\d+", "#", regex=True).str.slice(0, 28)
    top = flows.assign(pat=pat).groupby("pat")["amount"].agg(["size", "sum"]).sort_values("size", ascending=False)
    print(top.head(45).to_string())
    m = c.str.contains("REBATE|CASHBACK|CASH BACK|COMM|IB |IB-|REFUND|PROMO|BONUS", regex=True)
    print("\nrebate-like comments:", int(m.sum()), "sum", round(float(flows.loc[m, "amount"].sum())))
    print(flows.loc[m].assign(pat=pat[m]).groupby(["database", "pat"])["amount"].agg(["size", "sum"]).sort_values("size", ascending=False).head(30).to_string())

print("\n=== warehouse columns")
from webapp import data_store
p = sorted(glob.glob(str(data_store.WAREHOUSE / "mt5_live01" / "2026-09.parquet")))
if p:
    f = pd.read_parquet(p[0])
    print(list(f.columns)); print(f.head(2).T.to_string())
