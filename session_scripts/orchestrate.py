"""Finish the remaining work, in order, without supervision.

SEQUENCING IS THE POINT

The previous attempt chained these by polling `job_state`, which reads state
held IN THE CURRENT PROCESS. The Quant run lives in a different process, so the
call returned "idle" immediately, the Trading run started alongside it, and both
died out of memory -- the allocation that failed was 34 MiB, with 3.4 GB free
against two frames wanting sixteen apiece.

Completion is therefore detected by watching for the other python process to
exit, which is true across processes. Everything here runs strictly one at a
time for the same reason.

THE STEPS

  1. wait for the Quant run already in flight;
  2. cache contract specifications locally, waiting for a VPN window -- exposure
     figures silently became zero without them, and they were re-read from MySQL
     on every call;
  3. finish repairing mt5_live01's missing months;
  4. backtest the book-level exposure cap, which is the layer aimed at the
     drawdown rather than at client selection;
  5. retrain Trading with economic weights, cashflow features and six servers.
"""
import subprocess
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
ROOT = Path(r"c:\Users\RoyVivasi\Documents\notebook")
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SCRATCH = Path(r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude"
               r"\c--Users-RoyVivasi-Documents-notebook"
               r"\9951e7b4-740a-496a-a92d-689972573193\scratchpad")


def say(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def big_python_running(threshold_mb: int = 4000) -> bool:
    """Is another heavy python process alive? Works across processes."""
    try:
        import psutil
    except ImportError:
        return False
    me = __import__("os").getpid()
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


def wait_for_quiet(label: str, timeout_hours: float = 6.0):
    deadline = time.time() + timeout_hours * 3600
    announced = False
    while time.time() < deadline:
        if not big_python_running():
            if announced:
                say(f"{label}: finished")
            return True
        if not announced:
            say(f"{label}: waiting for the running job to finish")
            announced = True
        time.sleep(60)
    say(f"{label}: gave up waiting")
    return False


def vpn_up() -> bool:
    try:
        socket.getaddrinfo("ld4-dbproxy.in.zfx.loc", None)
        return True
    except socket.gaierror:
        return False


def run(script: Path, label: str, timeout_hours: float = 12.0) -> int:
    say(f"START {label}")
    began = time.time()
    result = subprocess.run([str(PY), str(script)], cwd=str(ROOT),
                            capture_output=True, text=True,
                            timeout=timeout_hours * 3600)
    tail = (result.stdout or "").strip().splitlines()[-25:]
    for line in tail:
        print(f"    {line}", flush=True)
    if result.returncode != 0:
        err = (result.stderr or "").strip().splitlines()[-8:]
        for line in err:
            print(f"    ! {line}", flush=True)
    say(f"END {label} rc={result.returncode} in {(time.time()-began)/60:.1f} min")
    return result.returncode


# --- 1. let the Quant run finish -------------------------------------------
wait_for_quiet("quant retrain")

# --- 2. cache contract specs (needs a VPN window) --------------------------
for attempt in range(240):          # up to ~2 hours of 30s polls
    if vpn_up():
        break
    if attempt == 0:
        say("waiting for VPN to cache contract specs")
    time.sleep(30)
if vpn_up():
    run(SCRATCH / "cache_specs.py", "cache contract specs", timeout_hours=1)
else:
    say("VPN never returned -- skipping spec cache; exposure backtest may be thin")

# --- 3. finish the mt5_live01 repair ---------------------------------------
run(SCRATCH / "repair_gaps.py", "repair mt5_live01 gaps", timeout_hours=6)

# --- 4. the book-level exposure backtest -----------------------------------
run(SCRATCH / "book_hedge.py", "book exposure backtest", timeout_hours=3)

# --- 5. retrain Trading -----------------------------------------------------
wait_for_quiet("pre-retrain check")
run(SCRATCH / "retrain_trading_weighted.py", "trading retrain", timeout_hours=14)

say("ALL STEPS COMPLETE")
