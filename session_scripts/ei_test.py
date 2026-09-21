import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import event_impact as ei

nfp = ei.last_nfp()
print("NFP default:", nfp)
d = ei.analyze(nfp["start"], nfp["end"], symbols=nfp["symbols"], refresh=True)
if d.get("error"):
    print("ANALYZE ERROR:", d["error"])
else:
    print("window:", d["window"])
    print("n_impacted:", d["n_impacted"])
    print("totals:", d["totals"])
    print("segments:", d["segments"])
    blob = ei.build_excel(d)
    out = r"c:\Users\RoyVivasi\Documents\notebook\webapp\artifacts\_ei_test.xlsx"
    open(out, "wb").write(blob)
    print(f"excel built: {len(blob):,} bytes -> {out}")
