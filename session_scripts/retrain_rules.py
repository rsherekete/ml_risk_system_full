"""Train the generic NOW+EW model pairs for every use_ml rule so the
Rules & Coverage count chips (hindsight / ML / EW 5d) fully populate.
The latency tape part will report store-busy (server holds the tick
store) -- its existing tape artifact stays; the generic pair still trains."""
import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import rule_models, af_registry

for c in af_registry.load().get("categories", []):
    if not (c.get("use_ml") and c.get("active")):
        continue
    key = c["key"]
    print(f"=== {key} ===", flush=True)
    meta = rule_models.train_rule(key)
    for fam in ("now", "ew"):
        m = (meta or {}).get(fam) or {}
        if m.get("auc") is not None:
            print(f"  {fam}: AUC {m['auc']} PR-AUC {m.get('pr_auc')} "
                  f"P {m['precision']} R {m['recall']} thr {m['threshold']} "
                  f"({m['positives']}+ of {m['labelable']})", flush=True)
        else:
            print(f"  {fam}: {m.get('error') or meta.get('error') or 'n/a'}",
                  flush=True)
print("done", flush=True)
