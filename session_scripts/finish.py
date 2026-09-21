"""The last mile, run after the trading retrain: book risk, server, verification.

Waits for the orchestrator's retrain to finish (process watch, not in-process
state -- the latter is the bug that OOM'd two 12GB jobs into each other), then:

  1. builds the daily book and BOTH exposure sweeps (static cap and the
     anomaly-conditional policy) so the Book Risk tab shows the honest
     comparison and the daily_book.parquet exists for future work;
  2. prints the weighted-vs-uniform trading comparison from the fresh artifact;
  3. restarts the web server and sweeps every route.
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
ROOT = Path(r"c:\Users\RoyVivasi\Documents\notebook")
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def say(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def big_python_running(threshold_mb: int = 4000) -> bool:
    import os

    import psutil
    me = os.getpid()
    for process in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            if process.info["pid"] == me:
                continue
            if "python" not in (process.info["name"] or "").lower():
                continue
            if process.info["memory_info"].rss / 1e6 > threshold_mb:
                return True
        except Exception:
            continue
    return False


say("waiting for the trading retrain to finish (checks every 5 min)")
while big_python_running():
    time.sleep(300)
say("retrain done")

# --- 1. book risk: daily book + both sweeps --------------------------------
say("building daily book + exposure sweeps")
from webapp import book_risk

book_risk._build(730)
report = book_risk.report()
if report.get("available"):
    say(f"book risk ready: {report['days']} days, "
        f"{len(report.get('rows', []))} static limits, "
        f"{len(report.get('anomaly', []))} anomaly triggers")
    for row in report.get("anomaly", []):
        flag = "  <== BEATS BOTH" if row["dominates"] else ""
        say(f"  anomaly p{row['percentile']:.3f}: {row['profit_delta']:+,.0f} profit, "
            f"{row['drawdown_delta']:+,.0f} DD, {row['days_hedged']} days hedged{flag}")
else:
    say(f"book risk build failed: {report.get('error')}")

# --- 2. weighted-vs-uniform trading comparison -----------------------------
say("trading artifact after the weighted retrain:")
import numpy as np
import pandas as pd

from webapp import model_service as ms

meta = ms.artifact_meta(ms.VIEW_TRADING) or {}
say(f"rows {meta.get('rows', 0):,} | AUC {meta.get('metrics', {}).get('roc_auc', float('nan')):.4f}")
frame = ms.load_scores(ms.VIEW_TRADING)
if frame is not None and len(frame):
    score = frame["score"].to_numpy(dtype="float64")
    pnl = frame["pnl"].to_numpy(dtype="float64")
    ok = np.isfinite(score) & np.isfinite(pnl)
    previous = {0.70: -18_184_260, 0.75: 13_781_724, 0.80: 11_170_907,
                0.85: 13_545_534, 0.90: 6_912_246}
    say("threshold ladder (uplift vs flat; uniform-weight run in brackets):")
    for x in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        picked = ok & (score >= x)
        uplift = float(pnl[picked].sum())
        was = previous.get(x)
        note = f"  (was {was:+,.0f})" if was is not None else ""
        say(f"  >= {x:.2f}: {picked.sum():>9,} rows  {uplift:+16,.0f}{note}")

# --- 3. restart the server and sweep ---------------------------------------
say("restarting web server")
subprocess.run(["powershell", "-NoProfile", "-Command",
                "$c=Get-NetTCPConnection -LocalPort 8600 -State Listen -EA SilentlyContinue; "
                "if($c){Stop-Process -Id $c.OwningProcess -Force -EA SilentlyContinue}"],
               capture_output=True)
time.sleep(5)
subprocess.Popen([str(PY), "-m", "uvicorn", "webapp.main:app",
                  "--host", "127.0.0.1", "--port", "8600"],
                 cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(90)

import urllib.request

opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
import urllib.parse
body = urllib.parse.urlencode({"username": "admin", "password": "admin"}).encode()
try:
    opener.open("http://127.0.0.1:8600/login", data=body, timeout=30)
except Exception:
    pass

routes = ["/executive", "/trading/summary", "/trading/overview", "/trading/abook",
          "/trading/clients", "/trading/exposure", "/trading/regions",
          "/trading/cashflow", "/trading/surveillance", "/trading/bookrisk",
          "/trading/risk", "/trading/performance", "/trading/validation",
          "/quant/overview", "/quant/summary", "/quant/exits", "/quant/risk",
          "/quant/validation", "/settings", "/admin", "/data"]
ok_count, bad = 0, []
for route in routes:
    try:
        response = opener.open(f"http://127.0.0.1:8600{route}", timeout=300)
        if response.status == 200:
            ok_count += 1
        else:
            bad.append(f"{route}={response.status}")
    except Exception as error:
        bad.append(f"{route}: {type(error).__name__}")
say(f"route sweep: {ok_count}/{len(routes)} ok" + (f" | FAILED: {bad}" if bad else ""))
say("ALL COMPLETE")
