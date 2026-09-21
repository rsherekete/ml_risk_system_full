import sys, json, datetime as dt
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import mysql_extract
server = "mt4_live04"; login = 6857706
conn = mysql_extract._connection(server, timeout=30)

def q(sql, args=()):
    with conn.cursor() as cur:
        cur.execute(sql, args); return cur.fetchall()

print("== orders columns ==")
for r in q("SHOW COLUMNS FROM orders"):
    print("  ", r[0], r[1])
print("\n== latest 5 orders for login %s ==" % login)
for r in q("SELECT `order`, symbol_name, cmd, volume, open_price, open_ts, close_price, close_ts, profit "
           "FROM orders WHERE login=%s ORDER BY open_ts DESC LIMIT 5", (login,)):
    print("  ", r)
print("\n== any XAU orders for login (latest 3) ==")
for r in q("SELECT `order`, symbol_name, open_ts, close_ts FROM orders "
           "WHERE login=%s AND symbol_name LIKE %s ORDER BY open_ts DESC LIMIT 3", (login, "%XAU%")):
    print("  ", r, "open=", dt.datetime.fromtimestamp(r[2]) if isinstance(r[2], int) else r[2])
print("\n== ticks table columns ==")
for r in q("SHOW COLUMNS FROM ticks"):
    print("  ", r[0], r[1])
print("\n== distinct gold-ish tick symbols ==")
for r in q("SELECT DISTINCT symbol_name FROM ticks WHERE symbol_name LIKE %s LIMIT 10", ("%XAU%",)):
    print("  ", r[0])
print("\n== latest tick tm for XAUUSD ==")
for r in q("SELECT symbol_name, MAX(tm), MIN(tm) FROM ticks WHERE symbol_name LIKE %s", ("%XAU%",)):
    print("  ", r[0], "max=", r[1], "min=", r[2])
