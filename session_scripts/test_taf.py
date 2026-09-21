import warnings, time, sys
warnings.filterwarnings("ignore")
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import taf
t0 = time.time()
ov = taf._compute()
print("computed in %.0fs" % (time.time() - t0))
print("cost/day upper:", f"{ov['cost_per_day']:,.0f}", "| cost_today:",
      f"{ov['cost_today']:,.0f}", "| latest_day:", ov["latest_day"])
print("\nBY CLASS (cost/day | today | latest_week | A_accts | auc):")
for c in ov["by_class"]:
    print("  %-24s day=$%-11s today=$%-10s wk=$%-11s A=%-6d auc=%s" % (
        c["class"], f"{c['cost_per_day']:,.0f}", f"{c['cost_today']:,.0f}",
        f"{c['cost_latest_week']:,.0f}", c["A_accounts"], c["auc"]))
wf = ov["monitor"]["walkforward"]
print("\nWALK-FORWARD available:", wf.get("available"), "| op_thr:",
      wf.get("operating_threshold"))
ag = wf.get("aggregate", {})
print("AGG: capture%%=%.1f  precision=%.2f  recall=%.2f  captured=$%s upper=$%s" % (
    100 * ag.get("usd_capture_pct", 0), ag.get("precision", 0), ag.get("recall", 0),
    f"{ag.get('captured_usd', 0):,.0f}", f"{ag.get('upper_usd', 0):,.0f}"))
print("test weeks:", wf.get("test_weeks"))
for w in wf.get("weekly", []):
    print("  %s P=%.2f R=%.2f capt=$%-11s upper=$%-11s cap%%=%.0f%%" % (
        w["week"], w["precision"], w["recall"], f"{w['captured_usd']:,.0f}",
        f"{w['upper_usd']:,.0f}", 100 * w["capture_pct"]))
print("\nfrontier:")
for r in wf.get("frontier", []):
    print("  thr=%.2f cap=%.0f%% P=%.2f R=%.2f" % (
        r["threshold"], 100 * r["usd_capture_pct"], r["precision"], r["recall"]))
import json
json.dumps(ov)   # will raise if any np type leaks through
print("\nJSON-serializable: OK")
