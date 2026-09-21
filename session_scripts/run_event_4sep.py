"""Run the 4 Sep NFP analysis through the live server (fills its cache and
event_impact_last.json), then download the workbook to docs/."""
import sys, sqlite3, json, urllib.request, urllib.parse, time, shutil
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import auth
uid = sqlite3.connect(r"c:\Users\RoyVivasi\Documents\notebook\webapp\app.db").execute(
    "SELECT id FROM users WHERE role='admin' ORDER BY id LIMIT 1").fetchone()[0]
tok = auth.start_session(uid)

def get(path, timeout=900):
    req = urllib.request.Request("http://127.0.0.1:8000" + path); req.add_header("Cookie", f"zfx_session={tok}")
    return urllib.request.urlopen(req, timeout=timeout).read()

preset = json.loads(get("/api/antifraud/event_impact/preset?name=nfp_2026_09_04"))
print("preset:", preset)
q = urllib.parse.urlencode({"start": preset["start"], "end": preset["end"], "symbols": preset["symbols"],
                            "loss_limit": 500, "profit_limit": 500})
t0 = time.time()
data = json.loads(get("/api/antifraud/event_impact?" + q + "&refresh=1"))
print(f"[{time.time()-t0:.0f}s] impacted {data.get('n_impacted')} | abuse {data.get('abuse_counts')} | classes {data.get('classes')} | segments {data.get('segments')}")
print("abuse_notes:", data.get("abuse_notes"))
blob = get("/api/antifraud/event_impact.xlsx?" + q)
out = r"c:\Users\RoyVivasi\Documents\notebook\docs\event_impact_XAUUSD_2026-09-04_v2.xlsx"
open(out, "wb").write(blob)
print(f"excel {len(blob):,} bytes -> {out}")
last = json.loads(get("/api/antifraud/event_impact/preset?name=last_nfp")); print("last_nfp preset:", last)
