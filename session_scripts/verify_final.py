import sys, urllib.request, json, time
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import auth
with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])
def get(p):
    req = urllib.request.Request("http://127.0.0.1:8000" + p)
    req.add_header("Cookie", "zfx_session=" + tok)
    return json.loads(urllib.request.urlopen(req, timeout=60).read())

# 1) replay account_events for a known active account, recent 2h window
import datetime as dt, urllib.parse
end = dt.datetime.now(dt.timezone.utc)
start = end - dt.timedelta(hours=3)
fmt = "%Y-%m-%d %H:%M"
qs = urllib.parse.urlencode({"account": "mt5_live01:105036806",
                             "start": start.strftime(fmt), "end": end.strftime(fmt)})
ev = get("/api/replay/account_events?" + qs)
print("account_events: events=%d balance=%s note=%s" % (
    len(ev.get("events", [])), ev.get("balance"), ev.get("note", "")))
for e in (ev.get("events") or [])[:4]:
    print("  %s %-8s dir=%s lots=%.3f %s @ %s profit=%.2f" % (
        e["time"], e["symbol"], e["direction"], e["lots"], e["kind"], e["price"], e["profit"]))

# 2) engine: is it inverting now?
import sqlite3
cx = sqlite3.connect(r"c:\Users\RoyVivasi\Documents\notebook\webapp\app.db")
since = time.time() - 5*60
rows = cx.execute("SELECT stance, status, COUNT(*) n FROM vantage_orders WHERE created>? GROUP BY stance,status ORDER BY n DESC", (since,)).fetchall()
print("\nlast 5min orders by stance/status:", rows)

# 3) engine + kafka status
v = get("/api/vantage/status")["engine"]
print("ENGINE running=%s scored=%s acted=%s filled=%s" % (v["running"], v["scored"], v["acted"], v["filled"]))
k = get("/api/live/status")
print("KAFKA streaming=%d/%d" % (sum(1 for t in k["topics"] if t["status"]=="streaming"), len(k["topics"])))
