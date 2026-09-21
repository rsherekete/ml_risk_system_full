import sys, json, datetime as dt
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import mysql_extract
from webapp.trade_feed import _canonical
IN = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\s1_dump.json"
tr = json.load(open(IN))["trades"]
print("MYSQL_DATABASES:", dict(mysql_extract.MYSQL_DATABASES))

# pick a matchable gold trade with an mt4 source
g = [t for t in tr if t["symbol"] == "XAUUSD+" and (t.get("source_from_comment") or "").startswith("mt4")]
print("gold w/ mt4 source:", len(g))
sample = g[0]
src = sample["source_from_comment"]; server, login = src.split(":")
print("sample:", src, "in_time", sample["in_time"],
      dt.datetime.fromtimestamp(sample["in_time"]))

conn = mysql_extract._connection(server, timeout=30)
# client MT4 orders for this login around the window (open_ts is epoch secs?)
lo = sample["in_time"] - 3600; hi = sample["in_time"] + 3600
with conn.cursor() as cur:
    cur.execute(
        "SELECT `order`, login, symbol_name, cmd, volume, open_price, open_ts, "
        "close_price, profit, close_ts FROM orders "
        "WHERE login=%s AND open_ts BETWEEN %s AND %s ORDER BY open_ts LIMIT 20",
        (login, lo, hi))
    rows = cur.fetchall()
print("\nclient orders near window (%d):" % len(rows))
for r in rows[:8]:
    print("  order=%s sym=%s cmd=%s vol=%s open=%s@%s close=%s@%s prof=%s" % (
        r[0], r[2], r[3], r[4], r[5], r[6], r[7], r[9], r[8]))

# ticks for the symbol on this server
canon = _canonical(sample["symbol"])
for symtry in (sample["symbol"], "XAUUSD", canon):
    with conn.cursor() as cur:
        try:
            cur.execute("SELECT tm, bid, ask FROM ticks WHERE symbol_name=%s "
                        "AND tm BETWEEN %s AND %s ORDER BY tm LIMIT 5",
                        (symtry, sample["in_time"], sample["in_time"] + 60))
            tk = cur.fetchall()
            print("\nticks for symbol '%s': %d rows" % (symtry, len(tk)))
            for r in tk[:3]:
                print("   tm=%s bid=%s ask=%s" % (r[0], r[1], r[2]))
            if tk:
                break
        except Exception as e:
            print("  ticks query err for '%s': %s" % (symtry, e))
