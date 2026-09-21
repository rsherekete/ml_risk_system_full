import sys, urllib.request, json, os
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import auth
with auth.connect() as c:
    row = c.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
    tok = auth.start_session(row["id"])
RF = r"c:\Users\RoyVivasi\Documents\notebook\webapp\artifacts\stats_reset.txt"
BAK = RF + ".bak"
moved = os.path.exists(RF)
if moved: os.rename(RF, BAK)

def get(path):
    req = urllib.request.Request("http://127.0.0.1:8000" + path)
    req.add_header("Cookie", "zfx_session=" + tok)
    return json.loads(urllib.request.urlopen(req, timeout=90).read())

try:
    st = get("/api/vantage/status")
    strat = st.get("report", {}).get("strategies", {})
    for k, name in (("s1", "MIRROR"), ("s2", "FADE")):
        s = (strat.get(k) or {}).get("perf", {})
        print(f"\n===== STRATEGY {k.upper()} ({name}) =====")
        print("  closed=%s  realized=$%s  EXPECTED=$%s  (accuracy gap $%.2f)" % (
            s.get("closed"), s.get("realized_usd"), s.get("expected_usd"),
            (s.get("realized_usd") or 0) - (s.get("expected_usd") or 0)))
        print("  hit_rate=%s  win_rate=%s  profit_factor=%s  max_dd=$%s" % (
            s.get("hit_rate"), s.get("win_rate"), s.get("profit_factor"), s.get("max_dd_usd")))
        print("  client@our_size=$%s  our_edge_vs_client=$%s  cost_paid=$%s" % (
            s.get("client_equiv_usd"), s.get("edge_vs_client_usd"), s.get("cost_usd")))
        bysym = s.get("by_symbol") or []
        print("  by symbol (worst first):")
        for b in sorted(bysym, key=lambda x: x.get("net", 0))[:8]:
            print("    %-8s net=$%8.2f  trades=%s  wins=%s" % (
                b.get("symbol"), b.get("net"), b.get("count"), b.get("wins")))
        rec = s.get("recent") or []
        print("  recent closed (expected vs actual):")
        for r in rec[:12]:
            print("    %s %-8s %-6s net=%7.2f exp=%7.2f client_eq=%7.2f cost=%6.2f score=%s" % (
                (r.get("close_time") or "")[-8:], r.get("symbol"), r.get("stance"),
                r.get("net"), r.get("expected"), r.get("client_equiv"), r.get("cost"), r.get("score")))
finally:
    if moved: os.rename(BAK, RF)
    print("\n(reset restored)")
