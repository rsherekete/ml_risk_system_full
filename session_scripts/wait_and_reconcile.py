"""Wait for the detached warehouse rebuild to finish, then refresh the app's
cached 4 Sep analysis and reconcile it against the standalone's summary."""
import sys, time, subprocess
from pathlib import Path
LOG = Path(r"c:\Users\RoyVivasi\Documents\notebook\docs\replication\_warehouse_rebuild.log")
deadline = time.time() + 110 * 60
while time.time() < deadline:
    text = LOG.read_text(encoding="utf-8") if LOG.exists() else ""
    if "== DONE ==" in text:
        break
    time.sleep(60)
else:
    print("NOT DONE after 55 min; last log lines:")
    print("\n".join(LOG.read_text(encoding="utf-8").splitlines()[-4:]))
    sys.exit(2)
print("rebuild DONE; tail of log:")
print("\n".join(LOG.read_text(encoding="utf-8").splitlines()[-8:]))
py = r"c:\Users\RoyVivasi\Documents\notebook\.venv\Scripts\python.exe"
# refresh the tab's cached analysis on the rebuilt warehouse
code = (
    "import sys,json; sys.path.insert(0, r'c:\\Users\\RoyVivasi\\Documents\\notebook');"
    "from webapp import event_impact as ei;"
    "d=ei.analyze('2026-09-04 13:29:00','2026-09-04 13:31:00','XAUUSD',refresh=True);"
    "print('app refreshed: n',d['n_impacted'],'| segments',d['segments'],'| cash to',d['cashflow_current_to'])"
)
print(subprocess.run([py, "-c", code], capture_output=True, text=True, cwd=r"c:\Users\RoyVivasi\Documents\notebook").stdout)
r = subprocess.run([py, str(Path(__file__).with_name("reconcile_standalone.py"))], capture_output=True, text=True,
                   cwd=r"c:\Users\RoyVivasi\Documents\notebook")
print(r.stdout); print(r.stderr[-800:] if r.returncode else "")
