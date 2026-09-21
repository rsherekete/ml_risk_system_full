import sys, urllib.request, json, time, datetime as dt
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import auth
with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])
def get(p):
    req = urllib.request.Request("http://127.0.0.1:8000" + p)
    req.add_header("Cookie", "zfx_session=" + tok)
    return json.loads(urllib.request.urlopen(req, timeout=60).read())

# Kafka streaming?
k = get("/api/live/status")
streaming = sum(1 for t in k["topics"] if t["status"] == "streaming")
print("KAFKA env=%s streaming=%d/%d" % (k.get("environment"), streaming, len(k["topics"])))
for t in k["topics"]:
    if "deals.live01" in t["topic"] or "trades.live01" in t["topic"]:
        print("  %-40s %s consumed=%s age=%ss" % (t["topic"][-40:], t["status"], t["consumed"],
              round(t["seconds_since_record"],1) if t["seconds_since_record"] else None))

# Engine stream freshness: now - newest scored trade
for _ in range(3):
    v = get("/api/vantage/status")
    e = v["engine"]
    sigs = v.get("signals") or []
    if sigs and sigs[0].get("traded"):
        traded = dt.datetime.fromisoformat(sigs[0]["traded"].replace("Z", "+00:00"))
        lag = (dt.datetime.now(dt.timezone.utc) - traded).total_seconds()
        print("ENGINE running=%s scored=%s acted=%s filled=%s | freshest scored trade lag=%.1fs (%s)" % (
            e["running"], e["scored"], e["acted"], e["filled"], lag, sigs[0].get("symbol")))
    else:
        print("ENGINE running=%s scored=%s (no signals yet)" % (e["running"], e["scored"]))
    time.sleep(3)

# Strategy metrics realtime
s1 = (v["report"]["strategies"]["s1"] or {})
s2 = (v["report"]["strategies"]["s2"] or {})
print("S1 floating=$%s open=%s closed=%s realized=$%s" % (
    round(s1.get("floating",0),2), len(s1.get("positions") or []),
    s1.get("perf",{}).get("closed"), s1.get("perf",{}).get("realized_usd")))
print("S2 floating=$%s open=%s closed=%s realized=$%s" % (
    round(s2.get("floating",0),2), len(s2.get("positions") or []),
    s2.get("perf",{}).get("closed"), s2.get("perf",{}).get("realized_usd")))
