import sys, time, collections
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import requests
from webapp import auth, vantage as v
with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])
cook = {"zfx_session": tok}; base = "http://127.0.0.1:8000"
cfg = v.load_config()
print("anchors: copy>=%.2f invert<=%.2f" % (cfg.copy_anchor_override, cfg.invert_anchor_override))
d = requests.get(base + "/api/vantage/status", cookies=cook, timeout=40).json()
print("\nSTREAM (score / decision) -- correct keys:")
for s in d.get("signals", []):
    print("  %-11s side=%-4s score=%s decision=%-7s s2=%s seen=%s traded=%s"
          % (s.get("symbol"), s.get("side"), s.get("score"), s.get("decision"),
             s.get("s2_decision"), (s.get("seen") or "")[-9:], (s.get("traded") or "")[-9:]))
dec = collections.Counter(s.get("decision") for s in d.get("signals", []))
print("decision counts:", dict(dec))
scores = [s.get("score") for s in d.get("signals", []) if s.get("score") is not None]
print("scores present: %d ; range %s..%s" % (len(scores), min(scores) if scores else None, max(scores) if scores else None))
# watch acted advance over 30s
e0 = d["engine"]; print("\n[t0] scored=%s acted=%s filled=%s" % (e0.get("scored"), e0.get("acted"), e0.get("filled")))
time.sleep(30)
e1 = requests.get(base + "/api/vantage/status", cookies=cook, timeout=40).json()["engine"]
print("[t+30] scored=%s acted=%s filled=%s" % (e1.get("scored"), e1.get("acted"), e1.get("filled")))
