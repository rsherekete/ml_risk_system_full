import sys, bisect, calendar, collections
import datetime as dt
import numpy as np
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import vantage, mysql_extract
XAU, OFF, DD_FRAC, DD_PEAK, LOT = 100.0, 10800, 0.35, 3.0, 0.01
reset = vantage.stats_reset_at() or 0
rows = vantage.recent_orders(20000)
def stance_of(r):
    v = r.get("live_score")
    if v is None or (r.get("stance") or "").startswith("fade"):
        return None
    return "copy" if v >= 0.85 else ("invert" if v <= 0.25 else None)
sigs = []
for r in rows:
    if (r.get("created") or 0) < reset or "XAU" not in (r.get("symbol") or ""):
        continue
    st = stance_of(r); src = r.get("source_account") or ""
    if st is None or ":" not in src or not src.split(":")[0].startswith("mt4"):
        continue
    sigs.append({"server": src.split(":")[0], "login": int(src.split(":")[1]),
                 "cdir": int(r.get("client_direction") or 0), "stance": st, "created": float(r.get("created") or 0)})
byserver = collections.defaultdict(list)
for s in sigs:
    byserver[s["server"]].append(s)
lo = min(s["created"] for s in sigs) - OFF - 1800; hi = max(s["created"] for s in sigs) - OFF + 1800
matched = []
for server, ss in byserver.items():
    logins = sorted({s["login"] for s in ss}); conn = mysql_extract._connection(server, timeout=90)
    ph = ",".join(["%s"]*len(logins))
    with conn.cursor() as cur:
        cur.execute(f"SELECT login,cmd,open_price,open_ts,close_price,close_ts FROM orders "
                    f"WHERE login IN ({ph}) AND symbol_name LIKE %s AND open_ts BETWEEN %s AND %s AND close_ts>0",
                    (*logins, "%XAU%", lo, hi))
        co = cur.fetchall()
    idx = collections.defaultdict(list)
    for r in co:
        idx[int(r[0])].append({"cmd": int(r[1]), "open_ts": int(r[3]), "close_ts": int(r[5]), "used": False})
    for s in sorted(ss, key=lambda x: x["created"]):
        cd = 1 if s["cdir"] > 0 else -1; best = None; bestd = 1e18
        for r in idx.get(s["login"], []):
            if r["used"] or (1 if r["cmd"] == 0 else -1) != cd:
                continue
            dd = abs(r["open_ts"] - (s["created"] - OFF))
            if dd < bestd:
                bestd, best = dd, r
        if best and bestd <= 300:
            best["used"] = True
            matched.append({"dir": cd if s["stance"] == "copy" else -cd, "open_ts": best["open_ts"], "close_ts": best["close_ts"]})
conn = mysql_extract._connection("mt4_live01", timeout=120)
t0 = min(m["open_ts"] for m in matched) - 120; t1 = max(m["close_ts"] for m in matched) + 300
with conn.cursor() as cur:
    cur.execute("SELECT tm,bid,ask FROM ticks WHERE symbol_name=%s AND tm BETWEEN %s AND %s ORDER BY tm",
                ("XAUUSD", dt.datetime.utcfromtimestamp(t0), dt.datetime.utcfromtimestamp(t1)))
    tk = cur.fetchall()
T = np.array([calendar.timegm(r[0].timetuple()) + r[0].microsecond/1e6 for r in tk])
P = np.array([(float(r[1]) + float(r[2]))/2.0 for r in tk])
def price_at(ep):
    i = bisect.bisect_right(T, ep) - 1
    return float(P[i]) if i >= 0 else float(P[0])
for m in matched:
    m["entry"] = price_at(m["open_ts"])
# gold daily vol fraction from the tape
logret = np.diff(np.log(P))
dt_med = np.median(np.diff(T)); steps_day = 86400.0 / max(dt_med, 1e-9)
gold_vol = float(np.std(logret) * np.sqrt(steps_day))
print("gold daily vol (tape): %.2f%%   signals: %d" % (gold_vol*100, len(matched)))

grid = list(range(0, len(T), 5)); opens = sorted(matched, key=lambda m: m["open_ts"]); oi = 0
book = []; realized = 0.0; curve = []
peak_net_notional = 0.0; peak_gross_notional = 0.0; peak_concurrent = 0
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
    net = abs(sum(p["dir"] for p in book)) * LOT * px * XAU
    gross = len(book) * LOT * px * XAU
    peak_net_notional = max(peak_net_notional, net)
    peak_gross_notional = max(peak_gross_notional, gross)
    peak_concurrent = max(peak_concurrent, len(book))
    floating = sum((px - p["entry"]) * p["dir"] * LOT * XAU for p in book)
    curve.append(realized + floating)
curve = np.array(curve); realized_dd = float((np.maximum.accumulate(curve) - curve).max())
peak_net_1s = peak_net_notional * gold_vol
peak_gross_1s = peak_gross_notional * gold_vol
print("peak concurrent positions: %d  | peak NET notional $%.0f | peak GROSS notional $%.0f"
      % (peak_concurrent, peak_net_notional, peak_gross_notional))
print("realized peak drawdown @0.01 lot: $%.0f" % realized_dd)
print("\n=== MIN DEPOSIT (gold signal book, 0.01 lot floor) ===")
print("  A) realized DD basis:            $%.0f  (= peak DD $%.0f / 0.35)" % (realized_dd/DD_FRAC, realized_dd))
print("  B) sizer model NET (dd_peak=3):  $%.0f  (= 3 x net-1sigma $%.0f / 0.35)" % (DD_PEAK*peak_net_1s/DD_FRAC, peak_net_1s))
print("  C) sizer model GROSS (worst,no hedge): $%.0f  (= 3 x gross-1sigma $%.0f / 0.35)" % (DD_PEAK*peak_gross_1s/DD_FRAC, peak_gross_1s))
