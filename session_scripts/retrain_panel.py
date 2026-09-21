"""One-off manual retrain of every use_ml rule under the NEW sequential
active-day panel framework (weekly boundaries, account-grouped CV)."""
import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import rule_models, af_registry

for c in af_registry.load().get("categories", []):
    if not (c.get("use_ml") and c.get("active")):
        continue
    key = c["key"]
    print(f"=== {key} ===", flush=True)
    m = rule_models.train_rule(key)
    print(f"  framework: {m.get('framework')} | pairs {m.get('panel_pairs')} "
          f"| samples {m.get('samples')}", flush=True)
    for fam in ("now", "ew"):
        d = (m or {}).get(fam) or {}
        if d.get("auc") is not None:
            print(f"  {fam}: AUC {d['auc']} PR-AUC {d.get('pr_auc')} "
                  f"P {d['precision']} R {d['recall']} thr {d['threshold']} "
                  f"({d['positives']}+ of {d['labelable']})", flush=True)
        else:
            print(f"  {fam}: {d.get('error') or m.get('error') or 'n/a'}",
                  flush=True)
print("done", flush=True)
