import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import requests
from webapp import auth, vantage as v
with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])
cook = {"zfx_session": tok}
d = requests.get("http://127.0.0.1:8000/api/vantage/status", cookies=cook, timeout=40).json()
cfg = v.load_config()
print("anchors: copy>=%.2f invert<=%.2f" % (cfg.copy_anchor_override, cfg.invert_anchor_override))
print("\nSTREAM SIGNALS (what the panel shows):")
for s in d.get("signals", []):
    val = s.get("value")
    print("  %-11s dir=%s value=%s decision=%s s2=%s traded=%s"
          % (s.get("symbol"), s.get("direction"),
             round(val, 3) if val is not None else None,
             s.get("decision"), s.get("s2_decision"), (s.get("traded") or "")[-9:]))
# count decisions
import collections
dec = collections.Counter(s.get("decision") for s in d.get("signals", []))
print("\ndecision counts in stream:", dict(dec))
