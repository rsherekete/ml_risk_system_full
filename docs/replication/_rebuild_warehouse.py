"""Full re-backfill of the trade warehouse (5 MySQL servers, 730 days) with the
current extractor, which deflates cent accounts at source. Needed because every
partition extracted before ~3 Sep 2026 still holds cent-account money and lots
x100. Runs detached; progress in _warehouse_rebuild.log."""
import sys, time
from pathlib import Path
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import mysql_extract, model_service as ms

log_path = Path(__file__).with_name("_warehouse_rebuild.log")
log = open(log_path, "a", encoding="utf-8")
def p(m):
    log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}\n"); log.flush()

p("== warehouse rebuild start ==")
for server in mysql_extract.MYSQL_DATABASES:
    p(f"== {server}: full backfill, 730 days ==")
    try:
        r = mysql_extract.backfill(server, days=730, progress=p)
        p(f"{server}: {r['rows']:,} rows written over {len(r['months'])} months; failures: {r['failures']}")
    except Exception as e:
        p(f"{server}: FAILED {type(e).__name__}: {e}")
# the cached feature frame was built from the undeflated warehouse: discard it
for f in (ms.SCRATCH / "quant_feature_cache.parquet", ms.SCRATCH / "quant_feature_cache.json"):
    try:
        f.unlink(); p(f"removed stale {f.name}")
    except FileNotFoundError:
        pass
p("== DONE ==")
