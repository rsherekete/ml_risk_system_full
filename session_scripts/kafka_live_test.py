import sys, time
NB = r"c:\Users\RoyVivasi\Documents\notebook"
sys.path.insert(0, NB)
KAFKA = NB + r"\kafka"
sys.path.insert(0, KAFKA)
from pathlib import Path
from sdk.config import Config
from sdk.kafka_client import KafkaClient

cfg = Config.load(Path(KAFKA) / "config.yaml")
print("host:", cfg.host, "| user:", cfg.username, "| registry:", cfg.registry_host)
print("default topic in config:", cfg.topic)

# 1) connectivity + what this principal can actually SEE
try:
    probe = type(cfg)(host=cfg.host, username=cfg.username, password=cfg.password,
                      topic=cfg.topic, registry_host=cfg.registry_host, path=cfg.path)
    with KafkaClient(probe, auto_offset_reset="latest", group_id="zfx-probe") as c:
        vis = c.visible_topics(timeout=20)
        print("\nVISIBLE TOPICS (%d):" % len(vis))
        for t in vis:
            print("   ", t)
except Exception as e:
    print("visible_topics ERROR:", type(e).__name__, e)

# 2) try to actually consume recent DEALS from a couple of candidate topics
CANDIDATES = [
    "traze.fsa.uat.oms-mt5.events.deals.live01",
    "traze.fsa.uat.oms-mt4.events.trades.live01",
    "traze.fsa.uat.oms-mt4.events.trades.demo02",
]
for topic in CANDIDATES:
    tc = type(cfg)(host=cfg.host, username=cfg.username, password=cfg.password,
                   topic=topic, registry_host=cfg.registry_host, path=cfg.path)
    print("\n=== consume", topic, "(earliest, up to 3 msgs / 20s) ===")
    got = []
    try:
        with KafkaClient(tc, auto_offset_reset="earliest", group_id="zfx-probe-earliest") as c:
            try:
                parts = c.partitions(timeout=15)
                print("  partitions:", parts)
            except Exception as pe:
                print("  partitions ERROR:", type(pe).__name__, pe)
            c.consume(lambda r: got.append(r), count=3, timeout=20, poll_timeout=1.0)
        print("  consumed:", len(got))
        for r in got[:3]:
            v = r.value or {}
            keys = list(v.keys())[:8]
            print("   msg ts=%s offset=%s keys=%s" % (r.timestamp, r.offset, keys))
    except Exception as e:
        print("  CONSUME ERROR:", type(e).__name__, e)
