import sys, datetime as dt
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import mysql_extract
for server in ("mt4_live01", "mt4_live04", "mt4_live02"):
    try:
        conn = mysql_extract._connection(server, timeout=40)
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(tm), MIN(tm) FROM ticks")
            mx, mn = cur.fetchone()
        smx = dt.datetime.fromtimestamp(mx) if mx else None
        smn = dt.datetime.fromtimestamp(mn) if mn else None
        print("%s ticks: tm %s -> %s" % (server, smn, smx))
        if mx:
            with conn.cursor() as cur:
                cur.execute("SELECT symbol_name, COUNT(*) FROM ticks WHERE tm BETWEEN %s AND %s "
                            "GROUP BY symbol_name HAVING symbol_name LIKE %s LIMIT 10",
                            (mx - 3600, mx, "%XAU%"))
                print("   gold syms in last hr:", cur.fetchall())
    except Exception as e:
        print("%s ERR: %s" % (server, type(e).__name__))
