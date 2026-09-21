"""Start the Kafka materialiser and confirm events land in DuckDB."""
import json
import sys
import time

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import kafka_service as ks

print("connectivity:", json.dumps(ks.probe_connectivity(), indent=1)[:500], flush=True)

materialiser = ks.MATERIALISER
materialiser.start(backfill=True)
for step in range(18):
    time.sleep(5)
    coverage = materialiser.coverage()
    print(f"  t+{(step+1)*5:3}s  events={coverage.get('events', 0):,}  "
          f"span={coverage.get('span_days', 0)}d  logins={coverage.get('logins', 0)}", flush=True)
    if coverage.get("events", 0) > 20000:
        break
materialiser.stop()
time.sleep(2)

print("\ncoverage:", json.dumps(materialiser.coverage(), indent=1))
print("\nstatus:", json.dumps(materialiser.status()["topics"], indent=1))
print("\ntop exposure:", json.dumps(materialiser.top_exposure(8), indent=1)[:900])
activity = materialiser.recent_activity(168)
print(f"\nactivity buckets: {len(activity)}")
for row in activity[:4]:
    print("  ", row)
