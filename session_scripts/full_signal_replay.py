import sys, bisect, calendar, collections
import datetime as dt
import numpy as np
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import vantage, mysql_extract
XAU, LEV, OFF = 100.0, 500.0, 10800

reset = vantage.stats_reset_at() or 0
rows = vantage.recent_orders(20000)
# every COPY/INVERT signal the engine routed since reset, on gold, with a source
def stance_of(r):
    v = r.get("live_score")
    if v is None or (r.get("stance") or "").startswith("fade"):
        return None
    return "copy" if v >= 0.85 else ("invert" if v <= 0.25 else None)
sigs = []
for r in rows:
    if (r.get("created") or 0) < reset:
        continue
    if "XAU" not in (r.get("symbol") or ""):
        continue
    st = stance_of(r)
    src = r.get("source_account") or ""
    if st is None or ":" not in src or not src.split(":")[0].startswith("mt4"):
        continue
    sigs.append({"src": src, "server": src.split(":")[0], "login": int(src.split(":")[1]),
                 "cdir": int(r.get("client_direction") or 0), "stance": st,
                 "created": float(r.get("created") or 0), "score": r.get("live_score")})
print("gold copy/invert signals w/ mt4 source since reset: %d (copy %d, invert %d)"
      % (len(sigs), sum(s["stance"] == "copy" for s in sigs), sum(s["stance"] == "invert" for s in sigs)))

# pull client gold orders per server, match each signal to a UNIQUE client order
byserver = collections.defaultdict(list)
for s in sigs:
    byserver[s["server"]].append(s)
lo = min(s["created"] for s in sigs) - OFF - 1800
hi = max(s["created"] for s in sigs) - OFF + 1800
matched = []
for server, ss in byserver.items():
    logins = sorted({s["login"] for s in ss})
    conn = mysql_extract._connection(server, timeout=90)
    ph = ",".join(["%s"]*len(logins))
    with conn.cursor() as cur:
        cur.execute(f"SELECT login, cmd, open_price, open_ts, close_price, close_ts "
                    f"FROM orders WHERE login IN ({ph}) AND symbol_name LIKE %s "
                    f"AND open_ts BETWEEN %s AND %s AND close_ts>0", (*logins, "%XAU%", lo, hi))
        co = cur.fetchall()
    idx = collections.defaultdict(list)
    for r in co:
        idx[int(r[0])].append({"cmd": int(r[1]), "open": float(r[2]), "open_ts": int(r[3]),
                               "close": float(r[4]), "close_ts": int(r[5]), "used": False})
    for s in sorted(ss, key=lambda x: x["created"]):
        cd = 1 if s["cdir"] > 0 else -1
        target = s["created"] - OFF
        best = None; bestd = 1e18
        for r in idx.get(s["login"], []):
            if r["used"] or (1 if r["cmd"] == 0 else -1) != cd:
                continue
            d = abs(r["open_ts"] - target)
            if d < bestd:
                bestd, best = d, r
        if best and bestd <= 300:
            best["used"] = True
            matched.append({"stance": s["stance"], "dir": (cd if s["stance"] == "copy" else -cd),
                            "open_ts": best["open_ts"], "close_ts": best["close_ts"],
                            "open": best["open"], "close": best["close"]})
print("unique client gold trades matched to signals: %d (copy %d, invert %d)"
      % (len(matched), sum(m["stance"] == "copy" for m in matched),
         sum(m["stance"] == "invert" for m in matched)))

# tape
conn = mysql_extract._connection("mt4_live01", timeout=120)
t0 = min(m["open_ts"] for m in matched) - 120
t1 = max(m["close_ts"] for m in matched) + 300
with conn.cursor() as cur:
    cur.execute("SELECT tm, bid, ask FROM ticks WHERE symbol_name=%s AND tm BETWEEN %s AND %s ORDER BY tm",
                ("XAUUSD", dt.datetime.utcfromtimestamp(t0), dt.datetime.utcfromtimestamp(t1)))
    tk = cur.fetchall()
T = np.array([calendar.timegm(r[0].timetuple()) + r[0].microsecond/1e6 for r in tk])
P = np.array([(float(r[1]) + float(r[2]))/2.0 for r in tk])
def price_at(ep):
    i = bisect.bisect_right(T, ep) - 1
    return float(P[i]) if i >= 0 else float(P[0])

LOT = 0.01
for leg in ("copy", "invert", "ALL"):
    ms = [m for m in matched if leg == "ALL" or m["stance"] == leg]
    pnl = sum((price_at(m["close_ts"]) - price_at(m["open_ts"])) * m["dir"] * LOT * XAU for m in ms)
    print("  %-6s n=%4d  tape-mirror P&L @0.01lot = $%+.0f  ($%+.2f/trade)"
          % (leg, len(ms), pnl, pnl/max(len(ms), 1)))

# --- MIN ACCOUNT: deepest equity drawdown of the full signal book, held to mirror
for m in matched:
    m["entry"] = price_at(m["open_ts"])
opens = sorted(matched, key=lambda m: m["open_ts"]); oi = 0
book = []; realized = 0.0; min_eq = 0.0; max_margin = 0.0
grid = list(range(0, len(T), 5))
for k in grid:
    now, px = T[k], P[k]
    while oi < len(opens) and opens[oi]["open_ts"] <= now:
        book.append(opens[oi]); oi += 1
    still = []
    for p in book:
        if p["close_ts"] <= now:
            realized += (px - p["entry"]) * p["dir"] * LOT * XAU
        else:
            still.append(p)
    book = still
    floating = sum((px - p["entry"]) * p["dir"] * LOT * XAU for p in book)
    margin = sum(LOT * px * XAU for p in book) / LEV
    max_margin = max(max_margin, margin)
    min_eq = min(min_eq, realized + floating)
need = -min_eq + 0.5 * max_margin
total = sum((price_at(m["close_ts"]) - price_at(m["open_ts"])) * m["dir"] * LOT * XAU for m in matched)
print("\nMIN ACCOUNT to capture the FULL gold signal edge (0.01 lot, 500x):")
print("  deepest equity drawdown: $%.0f | peak margin: $%.0f | peak concurrent: %d"
      % (min_eq, max_margin, max(len(book) for book in [book])))
print("  => MIN ACCOUNT ~ $%.0f   (session edge $%.0f => %.0f%% return)"
      % (need, total, 100*total/max(need, 1)))
print("  scale linearly: at 0.05 lot min acct ~ $%.0f (5x)" % (need*5))
