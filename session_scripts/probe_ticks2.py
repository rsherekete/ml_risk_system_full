import sys, datetime as dt
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import mysql_extract
for server in ("mt4_live04", "mt4_live01"):
    conn = mysql_extract._connection(server, timeout=40)
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT symbol_name FROM ticks WHERE symbol_name LIKE %s LIMIT 20", ("%XAU%",))
        syms = [r[0] for r in cur.fetchall()]
    print("%s gold tick symbols: %s" % (server, syms))
    for s in syms[:4]:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*), MIN(tm), MAX(tm) FROM ticks WHERE symbol_name=%s", (s,))
            r = cur.fetchone()
        mn = dt.datetime.fromtimestamp(r[1]) if r[1] else None
        mx = dt.datetime.fromtimestamp(r[2]) if r[2] else None
        print("   %-14s n=%s  tm range %s -> %s" % (s, r[0], mn, mx))
