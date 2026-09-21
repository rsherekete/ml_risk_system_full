"""Backfill MT4 millisecond trade-event times from Kafka's retained log.

WHY: MT4 `orders` stamp fills to the second. The latency engine gets MT4
millisecond times from the Kafka trade events (`tradeRecord.timeStamp`), but
the app's live store only holds events consumed since its consumer started:
on 15 Sep 2026 just 1,530 of 909,162 MT4 trades in the 7-day scan matched,
leaving half the book without sub-second markouts. Kafka retains ~7-8 days
(about 5.1M MT4 trade events across live01-04), which covers the scan window.

WHAT: a READ-ONLY consumer per MT4 topic with a throwaway group (the app's
stable consumer group and its offsets are never touched) seeks to a start
time, reads to the high-water mark captured at start, and writes only
(server, order, event stamp, kind, mode) to parquet under
artifacts/mt4_event_times/. A manifest records how far each topic was read,
so reruns resume instead of replaying. The latency engine reads these files
alongside the live store (latency_arb._attach_mt4_ms_times).

Run: python -m webapp.mt4_kafka_backfill [--days 8]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "artifacts" / "mt4_event_times"
MANIFEST = OUT_DIR / "manifest.json"
KAFKA_DIR = ROOT.parent / "kafka"
PART_ROWS = 250_000


def _manifest() -> dict:
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        return {}


def last_run_age_hours() -> float:
    m = _manifest()
    stamps = [v.get("finished_at", 0) for v in m.values() if isinstance(v, dict)]
    return (time.time() - max(stamps)) / 3600 if stamps else float("inf")


def backfill(days: float = 8.0, log=print) -> dict:
    sys.path.insert(0, str(KAFKA_DIR))
    import yaml
    from confluent_kafka import TopicPartition
    from sdk.config import Config
    from sdk.kafka_client import KafkaClient

    spec = yaml.safe_load((KAFKA_DIR / "clusters.yaml").read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _manifest()
    results = {}
    for cluster in spec.get("clusters", []):
        for topic in [t for t in cluster.get("topics") or [] if ".oms-mt4.events.trades." in t]:
            server = f"mt4_{topic.split('.')[-1]}"
            since_ms = int((time.time() - days * 86400) * 1000)
            prior = manifest.get(topic, {})
            if prior.get("to_ms"):
                since_ms = max(since_ms, int(prior["to_ms"]) - 5 * 60_000)
            cfg = Config(host=cluster["host"], username=cluster["username"], password=cluster["password"],
                         topic=topic, registry_host=spec["registry"])
            t0 = time.time()
            n_msgs = n_rows = part = 0
            max_ms = prior.get("to_ms") or since_ms
            buffer: list[tuple] = []
            out = OUT_DIR / server
            out.mkdir(exist_ok=True)

            def flush():
                nonlocal buffer, part
                if not buffer:
                    return
                frame = pd.DataFrame(buffer, columns=["server", "ord", "stamp", "kind", "mode"])
                frame["stamp"] = pd.to_datetime(frame["stamp"].str.replace("Z", "", regex=False), errors="coerce")
                frame = frame.dropna(subset=["stamp", "ord"])
                frame["ord"] = frame["ord"].astype("int64")
                frame.to_parquet(out / f"part_{int(t0)}_{part:04d}.parquet", index=False)
                part += 1
                buffer = []

            with KafkaClient(cfg, auto_offset_reset="earliest") as client:
                consumer = client._consumer
                parts = client.partitions()
                starts = consumer.offsets_for_times([TopicPartition(topic, p, since_ms) for p in parts], timeout=30)
                ends, assign = {}, []
                for tp in starts:
                    lo, hi = consumer.get_watermark_offsets(TopicPartition(topic, tp.partition), timeout=30)
                    first = tp.offset if tp.offset >= 0 else hi
                    if first < hi:
                        ends[tp.partition] = hi
                        assign.append(TopicPartition(topic, tp.partition, first))
                todo = sum(ends[tp.partition] - tp.offset for tp in assign)
                log(json.dumps({"topic": topic, "server": server, "messages_to_read": todo}))
                if assign:
                    consumer.assign(assign)
                done = set()
                idle = 0
                while len(done) < len(assign):
                    msg = consumer.poll(1.0)
                    if msg is None:
                        idle += 1
                        if idle > 60:
                            log(json.dumps({"topic": topic, "warning": "no messages for 60 s; stopping"}))
                            break
                        continue
                    idle = 0
                    if msg.error():
                        continue
                    n_msgs += 1
                    if msg.offset() + 1 >= ends.get(msg.partition(), 0):
                        done.add(msg.partition())
                    rec = client._record(msg)
                    value = rec.value or {}
                    kind = next((k for k in ("orderCreated", "orderUpdated", "orderClosedBy") if isinstance(value.get(k), dict)), None)
                    if kind is None:
                        continue
                    body = value[kind]
                    tr = body.get("tradeRecord") or body.get("remain") or {}
                    if tr.get("timeStamp") and tr.get("order") is not None:
                        buffer.append((server, tr["order"], tr["timeStamp"], kind, body.get("mode")))
                        n_rows += 1
                    if rec.timestamp is not None:
                        max_ms = max(max_ms, int(rec.timestamp.timestamp() * 1000))
                    if len(buffer) >= PART_ROWS:
                        flush()
                    if n_msgs % 200_000 == 0:
                        log(json.dumps({"topic": topic, "read": n_msgs, "of": todo,
                                        "rate_per_s": round(n_msgs / max(time.time() - t0, 1e-6))}))
                flush()
            manifest[topic] = {"server": server, "to_ms": max_ms, "finished_at": time.time(),
                               "messages": n_msgs, "rows": n_rows}
            MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
            results[server] = {"messages": n_msgs, "rows": n_rows, "seconds": round(time.time() - t0)}
            log(json.dumps({"done": server, **results[server]}))
    return results


def launch_background() -> bool:
    """Start a backfill run in a detached subprocess (never blocks a scan)."""
    import subprocess
    lock = OUT_DIR / "_running.lock"
    if lock.exists() and time.time() - lock.stat().st_mtime < 3 * 3600:
        return False
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(time.time()), encoding="utf-8")
    log = open(OUT_DIR / "backfill.log", "a", encoding="utf-8")
    subprocess.Popen([sys.executable, "-m", "webapp.mt4_kafka_backfill", "--release-lock"],
                     cwd=str(ROOT.parent), stdout=log, stderr=subprocess.STDOUT)
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=8.0)
    ap.add_argument("--release-lock", action="store_true")
    args = ap.parse_args()
    try:
        backfill(args.days, log=lambda s: print(s, flush=True))
    finally:
        if args.release_lock:
            (OUT_DIR / "_running.lock").unlink(missing_ok=True)
