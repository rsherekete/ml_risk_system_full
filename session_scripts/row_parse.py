import sys, secrets
NB = r"c:\Users\RoyVivasi\Documents\notebook"
sys.path.insert(0, NB); sys.path.insert(0, NB + r"\kafka")
from sdk.kafka_client import KafkaClient
from webapp import kafka_service as k

cfg = k._config_for_topic("prod.oms-mt5.events.deals.live01")
group = f"{cfg.topic_prefix}.zfxprobe-{secrets.token_hex(3)}"
recs = []
with KafkaClient(cfg, auto_offset_reset="latest", group_id=group) as c:
    c.consume(lambda r: recs.append(r), count=3, timeout=40, poll_timeout=1.0)
print("got", len(recs), "records\n")

cols = ["topic","server","partition","offset","event_time","ingested_at","key",
        "message","login","symbol","canonical","action","volume","price","profit",
        "payload","entry","event_kind"]
row_fn = k.KafkaMaterialiser._row
for r in recs:
    try:
        row = row_fn(r)
    except TypeError:
        row = row_fn(None, r)  # in case it's an instance method
    d = dict(zip(cols, row))
    print("server=%s login=%s symbol=%s canonical=%s action=%s entry=%s vol=%s price=%s profit=%s kind=%s" % (
        d["server"], d["login"], d["symbol"], d["canonical"], d["action"], d["entry"],
        d["volume"], d["price"], d["profit"], d["event_kind"]))
