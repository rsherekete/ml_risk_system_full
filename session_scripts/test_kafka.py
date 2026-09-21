"""Can we actually reach the Kafka cluster, and what does it hold?

Three questions the UI depends on, answered before anything is wired to it:
  1. do the credentials work and is the host reachable?
  2. which topics may this principal see?
  3. how far back does retention actually go -- i.e. can "last 7 days" be served?
"""
import sys
import time

KAFKA_DIR = r"c:\Users\RoyVivasi\Documents\notebook\kafka"
sys.path.insert(0, KAFKA_DIR)

from sdk.config import Config
from sdk.kafka_client import KafkaClient

config = Config.load(KAFKA_DIR + r"\config.yaml")
print("config:", {k: v for k, v in config.masked().items() if k != "path"})

try:
    with KafkaClient(config, auto_offset_reset="earliest") as client:
        print("\nconnecting…", flush=True)
        topics = client.visible_topics(timeout=20.0)
        print(f"visible topics: {len(topics)}")
        for topic in topics:
            if "events" in topic:
                print("   ", topic)

        print(f"\npartitions for {config.topic}: ", end="", flush=True)
        print(client.partitions(timeout=20.0))

        seen = []
        t0 = time.time()
        got = client.consume(lambda r: seen.append(r), count=5, timeout=25.0)
        print(f"\nconsumed {got} records in {time.time()-t0:.0f}s")
        for record in seen[:3]:
            print(f"  offset {record.offset} @ {record.timestamp} msg={record.message}")
            if record.value:
                print("    fields:", list(record.value)[:14])
        if seen:
            span = (seen[-1].timestamp - seen[0].timestamp) if seen[0].timestamp else None
            print(f"\n  earliest retained record: {seen[0].timestamp}")
            print(f"  -> retention determines how much of a 7-day window can be served")
except Exception as error:
    print(f"\nFAILED: {type(error).__name__}: {error}")
