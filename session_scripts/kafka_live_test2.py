import sys, secrets
NB = r"c:\Users\RoyVivasi\Documents\notebook"
sys.path.insert(0, NB); sys.path.insert(0, NB + r"\kafka")
from pathlib import Path
from sdk.config import Config
from sdk.kafka_client import KafkaClient

cfg = Config.load(Path(NB + r"\kafka") / "config.yaml")

# Test each candidate deals/trades topic with a TOPIC-PREFIXED throwaway group
# (matching how the app names its group), reading from EARLIEST to catch history.
CANDIDATES = [
    "traze.fsa.uat.oms-mt5.events.deals.live01",
    "traze.fsa.uat.oms-mt5.events.deals.live-dubai",
    "traze.fsa.uat.oms-mt5.events.deals.live-indonesia",
    "traze.fsa.uat.oms-mt4.events.trades.live01",
    "traze.fsa.uat.oms-mt4.events.trades.demo02",
]
for topic in CANDIDATES:
    tc = type(cfg)(host=cfg.host, username=cfg.username, password=cfg.password,
                   topic=topic, registry_host=cfg.registry_host, path=cfg.path)
    prefix = tc.topic_prefix
    group = f"{prefix}.zfxprobe-{secrets.token_hex(3)}"   # prefixed throwaway
    got = []
    print(f"\n=== {topic}")
    print(f"    group={group}")
    try:
        with KafkaClient(tc, auto_offset_reset="earliest", group_id=group) as c:
            n = c.consume(lambda r: got.append(r), count=3, timeout=18, poll_timeout=1.0)
        print(f"    consumed={len(got)}")
        for r in got[:3]:
            v = r.value or {}
            print(f"      ts={r.timestamp} off={r.offset} keys={list(v.keys())[:10]}")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
