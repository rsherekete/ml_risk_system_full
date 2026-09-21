import sys, urllib.request
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import auth
with auth.connect() as c:
    row = c.execute("SELECT id, username FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])
    print("admin user:", row["username"])
def get(path):
    req = urllib.request.Request("http://127.0.0.1:8000" + path)
    req.add_header("Cookie", "zfx_session=" + tok)
    r = urllib.request.urlopen(req, timeout=20)
    return r.status, r.read().decode("utf-8", "replace")

st, html = get("/hub")
print("/hub status:", st)
print('nav has href="/taf":', 'href="/taf"' in html)
print('nav has >TAF<:', '>TAF<' in html)
st2, _ = get("/taf")
print("/taf status:", st2)
# show the view-switch nav block
import re
m = re.search(r'<nav class="view-switch".*?</nav>', html, re.S)
print("\n--- masthead nav ---")
print(re.sub(r'\s+', ' ', m.group(0))[:600] if m else "nav not found")
