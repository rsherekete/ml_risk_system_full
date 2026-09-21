import sys, json
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
import requests
from webapp import auth
OUT = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\s1_dump.json"
with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])
d = requests.get("http://127.0.0.1:8000/api/vantage/s1_dump?window_days=45",
                 cookies={"zfx_session": tok}, timeout=120).json()
print("available:", d.get("available"), "reason:", d.get("reason"), "count:", d.get("count"))
tr = d.get("trades", [])
if tr:
    json.dump(d, open(OUT, "w"))
    print("saved ->", OUT)
    # quick profile
    import collections
    syms = collections.Counter(t["symbol"] for t in tr)
    srcs = collections.Counter(t.get("source_account") for t in tr)
    matched = sum(1 for t in tr if t.get("matched"))
    times = [t["out_time"] for t in tr if t["out_time"]]
    import datetime as dt
    print("matched to order store: %d / %d" % (matched, len(tr)))
    print("out_time range: %s -> %s" % (dt.datetime.fromtimestamp(min(times)),
                                        dt.datetime.fromtimestamp(max(times))))
    print("top symbols:", syms.most_common(8))
    print("distinct source logins:", len(srcs), "top:", srcs.most_common(6))
    print("\nsample trade:", json.dumps(tr[0], indent=1))
