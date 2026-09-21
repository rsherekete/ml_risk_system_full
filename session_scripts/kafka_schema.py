"""Full decoded shape of a Deal event, so the risk panels can map fields correctly."""
import json
import sys
import time
from dataclasses import replace

KAFKA_DIR = r"c:\Users\RoyVivasi\Documents\notebook\kafka"
sys.path.insert(0, KAFKA_DIR)
from sdk.config import Config
from sdk.kafka_client import KafkaClient

base = Config.load(KAFKA_DIR + r"\config.yaml")
config = replace(base, topic="traze.fsa.uat.oms-mt5.events.deals.live01")

seen = []
with KafkaClient(config, auto_offset_reset="earliest") as client:
    client.consume(lambda r: seen.append(r), count=40, timeout=30.0)

print(f"consumed {len(seen)} records")
if seen:
    print("\nfull payload of the first record:")
    print(json.dumps(seen[0].value, indent=2)[:2600])

    kinds: dict[str, int] = {}
    for record in seen:
        for key in (record.value or {}):
            kinds[key] = kinds.get(key, 0) + 1
    print("\ntop-level keys across the sample:", kinds)

    # Timestamp span tells us how quickly retention is consumed.
    stamps = [r.timestamp for r in seen if r.timestamp]
    if stamps:
        print(f"\nrecord times: {min(stamps)} .. {max(stamps)}")
