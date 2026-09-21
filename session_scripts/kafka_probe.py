"""Which live-flow topics carry data, what shape is it, and how far back?

The decisive questions for the Live Risk screens:
  * do the TRADE topics (not quotes) actually produce?
  * what fields does the decoded payload carry, so risk panels can map them?
  * how far back does retention reach -- can a 7-day window be served?
"""
import sys
import time
from dataclasses import replace

KAFKA_DIR = r"c:\Users\RoyVivasi\Documents\notebook\kafka"
sys.path.insert(0, KAFKA_DIR)
from sdk.config import Config
from sdk.kafka_client import KafkaClient

base = Config.load(KAFKA_DIR + r"\config.yaml")
TOPICS = [
    "traze.fsa.uat.oms-mt4.events.trades.live01",
    "traze.fsa.uat.oms-mt4.events.trades.live02",
    "traze.fsa.uat.oms-mt5.events.deals.live01",
]

for topic in TOPICS:
    config = replace(base, topic=topic)
    print(f"\n=== {topic}")
    try:
        with KafkaClient(config, auto_offset_reset="earliest") as client:
            print("  partitions:", client.partitions(timeout=20.0))
            seen = []
            t0 = time.time()
            got = client.consume(lambda r: seen.append(r), count=3, timeout=20.0)
            print(f"  consumed {got} in {time.time()-t0:.0f}s")
            for record in seen[:2]:
                print(f"    offset {record.offset} @ {record.timestamp} msg={record.message}")
                if record.value:
                    for key, value in list(record.value.items())[:18]:
                        print(f"      {key} = {str(value)[:60]}")
            if seen and seen[0].timestamp:
                age = (time.time() - seen[0].timestamp.timestamp()) / 86400
                print(f"  earliest retained: {seen[0].timestamp}  ({age:.1f} days old)")
    except Exception as error:
        print(f"  FAILED {type(error).__name__}: {str(error)[:200]}")
