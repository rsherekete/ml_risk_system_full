import sys, json, time
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import requests
from webapp import auth
with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])
cook = {"zfx_session": tok}
base = "http://127.0.0.1:8000"

for attempt in range(20):
    ov = requests.get(base + "/api/taf/overview", cookies=cook, timeout=30).json()
    if not ov.get("computing"):
        break
    print("overview still computing... (%d)" % attempt); time.sleep(15)

print("\n== OVERVIEW ==")
print("cost_today:", ov.get("cost_today"), "latest_day:", ov.get("latest_day"))
bc = ov.get("by_class", [])
print("by_class keys:", list(bc[0].keys()) if bc else "none")
for c in bc[:3]:
    print("  %-22s today=%s day=%s wk=%s auc=%s" % (
        c["class"], c.get("cost_today"), c.get("cost_per_day"),
        c.get("cost_latest_week"), c.get("auc")))

mon = requests.get(base + "/api/taf/monitor", cookies=cook, timeout=30).json()
print("\n== MONITOR ==")
print("keys:", sorted(mon.keys()))
wf = mon.get("walkforward", {})
print("walkforward available:", wf.get("available"), "op_thr:", wf.get("operating_threshold"))
ag = wf.get("aggregate", {})
print("AGG capture%%: %s  precision: %s  recall: %s  captured: %s" % (
    ag.get("usd_capture_pct"), ag.get("precision"), ag.get("recall"), ag.get("captured_usd")))
print("test_weeks:", wf.get("test_weeks"))
print("frontier rows:", len(wf.get("frontier", [])))
print("mon.by_class rows:", len(mon.get("by_class", [])), "| cost_today:", mon.get("cost_today"))
print("weekly(oracle) rows:", len(mon.get("weekly", [])), "| daily rows:", len(mon.get("daily", [])))
