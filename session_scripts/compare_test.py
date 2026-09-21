import sys, sqlite3, json
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from starlette.testclient import TestClient
from webapp import auth, model_service as ms
from webapp.main import app

# helper: archived history is readable and idempotent
hist = ms.model_history("quant")
print("model_history:", [(h["dir"], round(h["metrics"].get("roc_auc") or 0, 4)) for h in hist])
assert hist and hist[0]["dir"].startswith("quant_20260909")
again = ms.archive_current("quant", note="dup-check")
print("archive_current idempotent (same dir, no new copy):", again.name, "| dirs:", len(ms.model_history("quant")))
assert len(ms.model_history("quant")) == len(hist)

uid = sqlite3.connect(r"c:\Users\RoyVivasi\Documents\notebook\webapp\app.db").execute(
    "SELECT id FROM users WHERE role='admin' ORDER BY id LIMIT 1").fetchone()[0]
client = TestClient(app); client.cookies.set("zfx_session", auth.start_session(uid))

j = client.get("/api/quant/lab").json()
print("\nendpoint keys:", sorted(j.keys()))
print("current: trained_at", j["trained_at"], "| rows", j["rows"], "| flat_bbook keys",
      sorted((j["current_flat_bbook"] or {}).keys()))
print("history entries:", len(j["history"]))
h0 = j["history"][0]
print("  previous:", h0["dir"], "| roc_auc", round(h0["roc_auc"], 4), "| P&L",
      round(h0["flat_bbook"]["total_pnl_usd"]), "| maxDD", round(h0["flat_bbook"]["max_drawdown_usd"]),
      "| walk_curve pts", len(h0["walk_curve"]), "| note:", h0["note"][:40])
same = (h0["trained_at"] == j["trained_at"])
print("  previous == current model right now?", same, "(expected True until the running retrain completes)")

r = client.get("/quant/lab")
print("\npage:", r.status_code, "| bytes", len(r.text))
for m in ("Previous model vs this run", "labCompare", "labPrevNote", "'Previous'", "axisDates"):
    print(f"  contains {m!r}:", m in r.text)
assert all(m in r.text for m in ("labCompare", "axisDates"))
print("\nCOMPARISON FEATURE VERIFIED")
