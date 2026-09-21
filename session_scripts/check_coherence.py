import warnings, sys, json
warnings.filterwarnings("ignore")
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import taf
ov = taf._compute()
json.dumps(ov)  # serialization guard
print("latest_day:", ov["latest_day"])
bc = ov["by_class"]
s_today = sum(c["live_today"] for c in bc)
s_week = sum(c["live_week"] for c in bc)
s_avg = sum(c["live_avg_daily"] for c in bc)
mx = max(c["live_today"] for c in bc)
print("\n--- TODAY (live, day-level) ---")
for c in bc:
    print("  %-22s today=$%-10s avg/day=$%-10s week=$%-11s" % (
        c["class"], f"{c['live_today']:,.0f}", f"{c['live_avg_daily']:,.0f}",
        f"{c['live_week']:,.0f}"))
print("\n  SUM of class today   = $%s" % f"{s_today:,.0f}")
print("  MAX single class     = $%s" % f"{mx:,.0f}")
print("  UNION total (today)  = $%s  (accts %s)" % (
    f"{ov['cost_today']:,.0f}", f"{ov['today_accounts']:,}"))
print("  coherent? union in [max, sum]:",
      mx <= ov["cost_today"] <= s_today + 1)
print("\n  UNION week   = $%s   (sum classes $%s)" % (
    f"{ov['live_week_total']:,.0f}", f"{s_week:,.0f}"))
print("  UNION avg/day= $%s   (sum classes $%s)" % (
    f"{ov['live_avg_daily_total']:,.0f}", f"{s_avg:,.0f}"))
print("\n  windowed cost/day headline (perfect oracle) = $%s" % f"{ov['cost_per_day']:,.0f}")
