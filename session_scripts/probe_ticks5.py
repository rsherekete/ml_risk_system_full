import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import mysql_extract
conn = mysql_extract._connection("mt4_live01", timeout=40)
with conn.cursor() as cur:
    cur.execute("SHOW COLUMNS FROM ticks")
    print("ticks columns:")
    for r in cur.fetchall():
        print("  ", r[0], r[1])
with conn.cursor() as cur:
    cur.execute("SELECT MAX(tm) FROM ticks")
    mx = cur.fetchone()[0]
    print("\nMAX(tm) =", repr(mx), type(mx).__name__)
# sample a recent row near max
with conn.cursor() as cur:
    cur.execute("SELECT symbol_name, tm, bid, ask FROM ticks WHERE tm BETWEEN %s AND %s "
                "AND symbol_name LIKE %s LIMIT 5",
                (str(mx)[:19].replace(str(mx)[11:13], str(int(str(mx)[11:13])-1).zfill(2), 1), str(mx), "%XAU%"))
    for r in cur.fetchall():
        print("  sample:", r)
