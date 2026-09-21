"""Pull the S3 synthetic-population results out of the session transcript:
tool outputs and assistant text around the S3 runs (population, structures,
transfer), printed as readable excerpts."""
import json, re, sys
PATH = r"C:\Users\RoyVivasi\.claude\projects\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193.jsonl"
KEYS = re.compile(r"SKILL RECOVERY|recovery efficiency|RECOVERY EFFICIENCY|ceiling AUC|S3 TRANSFER|transfer|planted|synthetic", re.I)

def texts(obj):
    """Every string inside a transcript record."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from texts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from texts(v)

wanted = set(range(17300, 17520)) | set(range(46550, 46580))
with open(PATH, encoding="utf-8", errors="replace") as f:
    for n, line in enumerate(f, 1):
        if n not in wanted:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        role = rec.get("type") or rec.get("role") or ""
        for t in texts(rec):
            if len(t) < 40 or not KEYS.search(t):
                continue
            # print only the informative blocks: outputs with numbers, or assistant prose with findings
            if re.search(r"AUC|corr|efficiency|%|\$", t):
                print(f"\n===== line {n} [{role}] =====")
                print(t[:3500])
