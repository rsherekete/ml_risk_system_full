import sys, json, datetime as dt
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import mysql_extract
M = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\matches.json"
mm = [t for t in json.load(open(M)) if t["c"]["close_ts"]]
c = mm[0]["c"]
server, cts = c["server"], c["close_ts"]
print("probe server=%s  close_ts=%d (%s)  client symbol=%s"
      % (server, cts, dt.datetime.fromtimestamp(cts), c["symbol"]))
conn = mysql_extract._connection(server, timeout=60)
# tm-indexed small window, NO symbol filter -> what symbols/tm exist here
with conn.cursor() as cur:
    cur.execute("SELECT symbol_name, tm, bid, ask FROM ticks "
                "WHERE tm BETWEEN %s AND %s ORDER BY tm LIMIT 40", (cts - 5, cts + 30))
    rows = cur.fetchall()
print("rows in [cts-5, cts+30]: %d" % len(rows))
syms = {}
for r in rows:
    syms.setdefault(r[0], 0); syms[r[0]] += 1
print("symbols present:", syms)
for r in rows[:6]:
    print("  ", r[0], dt.datetime.fromtimestamp(int(r[1])), "bid", r[2], "ask", r[3])
# also try a wider window to be safe about tz
if not rows:
    for off_h in (-3, 2, 3):
        cur = conn.cursor()
        cur.execute("SELECT symbol_name, tm FROM ticks WHERE tm BETWEEN %s AND %s "
                    "AND symbol_name LIKE %s LIMIT 5",
                    (cts + off_h*3600 - 30, cts + off_h*3600 + 30, "%XAU%"))
        rr = cur.fetchall()
        print("  tz probe %+dh: %d rows %s" % (off_h, len(rr), rr[:2]))
