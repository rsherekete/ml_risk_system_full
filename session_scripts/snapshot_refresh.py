"""Post-retrain refresh: snapshot the NEW models' predictions for every
account, rebuild every rule scan cache from them, refresh counts."""
import sys, json
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import rule_models, af_registry

s = rule_models.snapshot_predictions()
print("SNAPSHOT:", json.dumps(s)[:400], flush=True)
for c in af_registry.load().get("categories", []):
    if c.get("active") and c.get("use_ml") and c["key"] != "latency_arbitrage":
        rule_models.rule_scan(c["key"], refresh=True)
        import time
        for _ in range(60):
            r = rule_models.rule_scan(c["key"])
            if not r.get("building"):
                break
            time.sleep(5)
        print(f"{c['key']}: rows {len(r.get('rows', []))} | "
              f"rule_hits {r.get('rule_hits')}", flush=True)
rule_models._COUNTS_CACHE.clear()
print("counts cache cleared; disk caches rebuilt", flush=True)
