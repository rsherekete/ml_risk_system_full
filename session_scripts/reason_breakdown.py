import sqlite3, re
from collections import Counter
cx = sqlite3.connect(r"c:\Users\RoyVivasi\Documents\notebook\webapp\app.db")
cx.row_factory = sqlite3.Row
rows = cx.execute("SELECT created, status, our_lots, detail FROM "
                  "vantage_orders ORDER BY id DESC LIMIT 400").fetchall()
print("last 400 decisions status:", Counter(r["status"] for r in rows))
# normalize detail into a reason bucket
def bucket(d):
    d = (d or "").lower()
    for key in ("no quote", "edge", "below hurdle", "max open", "leverage wall",
                "symbol", "share", "limit rejected", "retcode", "averaging",
                "side-score", "stopped out", "margin", "paused", "held",
                "no headroom", "market closed"):
        if key in d:
            return key
    return d[:45]
print("\nblock/reject reason breakdown (last 400):")
for k, v in Counter(bucket(r["detail"]) for r in rows
                    if r["status"] in ("blocked", "rejected")).most_common(15):
    print(f"  {v:4}  {k}")
print("\nany FILLED in last 400:", sum(1 for r in rows if r["status"] == "filled"))
# time span of these 400
if rows:
    span = (rows[0]["created"] - rows[-1]["created"]) / 60
    print(f"span of last 400 decisions: {span:.0f} min "
          f"({400/max(span,1):.1f} decisions/min)")
