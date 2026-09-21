"""Second pass: the structure-study results table (recovery efficiency per
planted mechanism) and the transfer study, from the transcript."""
import json, re
PATH = r"C:\Users\RoyVivasi\.claude\projects\c--Users-RoyVivasi-Documents-notebook\9951e7b4-740a-496a-a92d-689972573193.jsonl"
KEYS = re.compile(r"transfer|virtual|decile|AUC", re.I)

def texts(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from texts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from texts(v)

with open(PATH, encoding="utf-8", errors="replace") as f:
    for n, line in enumerate(f, 1):
        if not (17886 <= n <= 17900):
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        role = rec.get("type") or rec.get("role") or ""
        for t in texts(rec):
            if len(t) < 60 or not KEYS.search(t) or t.lstrip().startswith(('"""', "import", "#")):
                continue
            if re.search(r"\d\.\d{2}|%", t):
                print(f"\n===== line {n} [{role}] =====")
                print(t[:4000])
