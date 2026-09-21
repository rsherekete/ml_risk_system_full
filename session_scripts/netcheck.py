import sys, sqlite3, time
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
cx = sqlite3.connect(r"c:\Users\RoyVivasi\Documents\notebook\webapp\app.db")
cx.row_factory = sqlite3.Row
cur = cx.cursor()
cur.execute("SELECT status, COUNT(*) n FROM vantage_orders GROUP BY status ORDER BY n DESC")
print("order status counts:")
for r in cur.fetchall():
    print("  %-12s %d" % (r["status"], r["n"]))
cur.execute("SELECT stance, COUNT(*) n FROM vantage_orders WHERE status='netted' GROUP BY stance")
print("netted by stance:", [(r["stance"], r["n"]) for r in cur.fetchall()])
cur.execute("SELECT COUNT(*) n, MIN(created) mn, MAX(created) mx FROM vantage_orders WHERE status='netted'")
r = cur.fetchone()
print("netted total:", r["n"], "| span:",
      time.ctime(r["mn"]) if r["mn"] else "-", "->", time.ctime(r["mx"]) if r["mx"] else "-")
# how many filled vs netted in the last 48h?
since = time.time() - 48 * 3600
cur.execute("SELECT status, COUNT(*) n FROM vantage_orders WHERE created > ? GROUP BY status ORDER BY n DESC", (since,))
print("last 48h status:", [(r["status"], r["n"]) for r in cur.fetchall()])
