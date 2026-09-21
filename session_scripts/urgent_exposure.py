import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import requests
from webapp import auth
with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])
cook = {"zfx_session": tok}
d = requests.get("http://127.0.0.1:8000/api/vantage/status", cookies=cook, timeout=40).json()
rep = d.get("report", {}); snap = rep.get("snapshot", {})
print("ACCOUNT: balance=%s equity=%s free_margin=%s margin=%s currency=%s"
      % (snap.get("balance"), snap.get("equity"), snap.get("free_margin"),
         snap.get("margin"), snap.get("currency")))
lvl = (snap.get("equity")/snap.get("margin")*100) if snap.get("margin") else None
print("margin level: %s%%" % (round(lvl) if lvl else "n/a (no open margin)"))
pos = rep.get("positions", [])
print("\nOPEN POSITIONS: %d" % len(pos))
tot = 0.0
for p in pos:
    print("  %s dir=%s lots=%s open=%s profit=%s comment=%s"
          % (p.get("symbol"), p.get("direction"), p.get("lots"), p.get("open_price"),
             p.get("profit"), p.get("comment")))
    tot += float(p.get("profit") or 0)
print("total floating P&L: %.2f" % tot)
