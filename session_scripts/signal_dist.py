import sys, collections
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import vantage
reset = vantage.stats_reset_at() or 0
rows = vantage.recent_orders(20000)
win = [r for r in rows if (r.get("created") or 0) >= reset]
print("vantage_orders since reset: %d (of %d total)" % (len(win), len(rows)))

def stance_of(r):
    v = r.get("live_score")
    st = r.get("stance") or ""
    if st.startswith("fade"):
        return "s2_fade"
    if v is None:
        return st or "?"
    if v >= 0.85:
        return "copy"
    if v <= 0.25:
        return "invert"
    return "mid(%.2f)" % v

print("\nby STANCE x STATUS:")
tab = collections.defaultdict(int)
for r in win:
    tab[(stance_of(r), r.get("status"))] += 1
for k in sorted(tab):
    print("  %-12s %-10s %d" % (k[0], k[1], tab[k]))

print("\nby recorded STANCE field:")
c = collections.Counter((r.get("stance"), r.get("status")) for r in win)
for k, n in c.most_common(20):
    print("  stance=%-8s status=%-10s %d" % (k[0], k[1], n))

# gold specifically
gold = [r for r in win if "XAU" in (r.get("symbol") or "")]
print("\nGOLD signals since reset: %d" % len(gold))
cg = collections.Counter(stance_of(r) for r in gold)
print("  by stance:", dict(cg))
cs = collections.Counter(r.get("status") for r in gold)
print("  by status:", dict(cs))
