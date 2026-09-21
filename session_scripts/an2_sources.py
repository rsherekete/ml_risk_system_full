import json, collections
import numpy as np
IN = r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193\scratchpad\s1_dump.json"
tr = json.load(open(IN))["trades"]
gold = [t for t in tr if t["symbol"] == "XAUUSD+"]

def real(src): return bool(src) and ":" in str(src) and str(src).split(":")[0].startswith(("mt4", "mt5"))

by_src = collections.defaultdict(lambda: {"n": 0, "net": 0.0, "wins": 0})
for t in gold:
    s = t.get("source_account") or "NONE"
    b = by_src[s]; b["n"] += 1; b["net"] += t["net"]; b["wins"] += 1 if t["net"] > 0 else 0
print("GOLD trades by source (%d total):" % len(gold))
for s, b in sorted(by_src.items(), key=lambda kv: kv[1]["net"]):
    print("  %-24s n=%3d net=$%9.2f win=%3.0f%%  real=%s" % (
        s, b["n"], b["net"], 100*b["wins"]/b["n"], real(s)))

realg = [t for t in gold if real(t.get("source_account"))]
print("\nGOLD matchable to a real client login: %d / %d  (net $%.0f)" % (
    len(realg), len(gold), sum(t["net"] for t in realg)))
# direction internal-consistency check on matchable gold
def route_dir(t):
    v = t.get("live_score"); cd = t.get("client_direction")
    if v is None or cd is None: return None, None
    if v >= 0.85: return "copy", cd
    if v <= 0.25: return "invert", -cd
    return "ignore", None
ok = bad = amb = 0
for t in realg:
    st, exp = route_dir(t)
    if exp is None: amb += 1; continue
    if exp == t.get("our_direction"): ok += 1
    else: bad += 1
print("direction consistency (route(score) vs our_direction): ok=%d bad=%d ambiguous=%d" % (ok, bad, amb))
print("\nsample matchable gold trade:", json.dumps(realg[0] if realg else {}, indent=1))
