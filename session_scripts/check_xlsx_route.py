"""The exact request the tab's Excel button makes (London wall-time params),
against the freshly restarted server whose memory cache is empty: it must
come back in seconds from the disk copy, never recompute."""
import sys, sqlite3, json, urllib.request, urllib.parse, urllib.error, time
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import auth
uid = sqlite3.connect(r"c:\Users\RoyVivasi\Documents\notebook\webapp\app.db").execute(
    "SELECT id FROM users WHERE role='admin' ORDER BY id LIMIT 1").fetchone()[0]
tok = auth.start_session(uid)

def get(path, timeout=120):
    req = urllib.request.Request("http://127.0.0.1:8000" + path); req.add_header("Cookie", f"zfx_session={tok}")
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()

q = urllib.parse.urlencode({"start": "2026-09-04 13:29:00", "end": "2026-09-04 13:31:00", "symbols": "XAUUSD",
                            "loss_limit": 500, "profit_limit": 500})
t0 = time.time(); st, hd, body = get("/api/antifraud/event_impact.xlsx?" + q)
print(f"[{time.time()-t0:.1f}s] xlsx with tab params -> {st} {len(body):,} bytes | {hd.get('Content-Disposition')}")
t0 = time.time(); st, hd, body = get("/api/antifraud/event_impact?" + q)
d = json.loads(body); print(f"[{time.time()-t0:.1f}s] json with tab params -> {st} clients={d.get('n_impacted')} from_disk={d.get('from_disk')} cache_key={d.get('cache_key')}")
q2 = urllib.parse.urlencode({"start": "2026-09-05 13:29:00", "end": "2026-09-05 13:31:00", "symbols": "XAUUSD", "loss_limit": 500, "profit_limit": 500})
t0 = time.time(); st, hd, body = get("/api/antifraud/event_impact.xlsx?" + q2)
print(f"[{time.time()-t0:.1f}s] xlsx for an UNCACHED window -> {st}: {body[:120]!r}")
