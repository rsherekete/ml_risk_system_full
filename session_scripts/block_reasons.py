import sqlite3
cx = sqlite3.connect(r"c:\Users\RoyVivasi\Documents\notebook\webapp\app.db")
cx.row_factory = sqlite3.Row
RESET = 1789073589
print("=== decisions SINCE reset ===")
rows = cx.execute("SELECT created, symbol, stance, status, our_lots, "
                  "expected_usd, detail FROM vantage_orders "
                  "WHERE created > ? ORDER BY id DESC LIMIT 20",
                  (RESET,)).fetchall()
print("count since reset:", len(rows))
for r in rows:
    print(f"  {r['symbol']:8} {r['stance']:6} {r['status']:9} "
          f"lots={r['our_lots']} exp=${r['expected_usd']} | {str(r['detail'])[:110]}")
print("\n=== recent REJECTED (broker) any time ===")
for r in cx.execute("SELECT symbol, status, our_lots, detail FROM "
                    "vantage_orders WHERE status='rejected' "
                    "ORDER BY id DESC LIMIT 6").fetchall():
    print(f"  {r['symbol']:8} lots={r['our_lots']} | {str(r['detail'])[:130]}")
print("\n=== block reason breakdown since reset ===")
from collections import Counter
c = Counter()
for r in cx.execute("SELECT status, detail FROM vantage_orders WHERE "
                    "created > ?", (RESET,)).fetchall():
    d = str(r["detail"] or "")[:40]
    c[(r["status"], d)] += 1
for k, v in c.most_common(12):
    print(f"  {v:4}  {k[0]:9} {k[1]}")
