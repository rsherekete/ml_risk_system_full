import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import rule_models

c = rule_models.registry_counts()
print("active_day:", c.get("active_day"), "| n:", c.get("n_accounts"),
      "| n_active:", c.get("n_active_today"),
      "| join_err:", c.get("join_error"))
for k, v in c.get("categories", {}).items():
    print(f"  {k:24s} live={v['hits']:<7} ({v['pct']}%)  "
          f"active={v['hits_active']:<5} ({v['pct_active']}%)")
