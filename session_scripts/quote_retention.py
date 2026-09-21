"""How far back do QUOTES actually go on the broker?

The local store holds 2.4 fragmented days, but that may only reflect how long
the consumer has run. The decisive question is what the broker retains: reading
from the earliest offset shows the true horizon.
"""
import sys
import time
from dataclasses import replace

KAFKA_DIR = r"c:\Users\RoyVivasi\Documents\notebook\kafka"
sys.path.insert(0, KAFKA_DIR)
from sdk.config import Config
from sdk.kafka_client import KafkaClient

base = Config.load(KAFKA_DIR + r"\config.yaml")

for topic in ("traze.fsa.uat.oms-mt5.events.quotes.live01",
              "traze.fsa.uat.oms-mt5.events.deals.live01"):
    config = replace(base, topic=topic)
    print(f"\n=== {topic}")
    try:
        with KafkaClient(config, auto_offset_reset="earliest") as client:
            seen = []
            t0 = time.time()
            client.consume(lambda r: seen.append(r), count=3, timeout=25.0)
            if not seen:
                print("  no records in 25s")
                continue
            earliest = min(r.timestamp for r in seen if r.timestamp)
            age_days = (time.time() - earliest.timestamp()) / 86400
            print(f"  earliest retained: {earliest}  ({age_days:.2f} days old)")
            print(f"  -> broker retention for this stream is about {age_days:.1f} days")
    except Exception as error:
        print(f"  FAILED {type(error).__name__}: {str(error)[:160]}")
