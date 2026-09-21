"""Render the live account page in-process (authenticated) and inspect the
chart scripts for what breaks in the browser."""
import sys, re, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from fastapi.testclient import TestClient
from webapp.main import app

client = TestClient(app)
# mint a session directly (in-process probe; password unknown/rotated)
from webapp import auth
import sqlite3
cx = sqlite3.connect(auth.DB_PATH if hasattr(auth, "DB_PATH") else
                     r"c:\Users\RoyVivasi\Documents\notebook\webapp\app.db")
row = cx.execute("SELECT id, username FROM users WHERE approved = 1 "
                 "ORDER BY id LIMIT 1").fetchone()
print("user:", row)
token = auth.start_session(row[0])
from webapp.main import SESSION_COOKIE
client.cookies.set(SESSION_COOKIE, token)
acc = "mt4_live01:1062794"
r = client.get(f"/trading/account?account={acc}")
print("page:", r.status_code, "| bytes:", len(r.text))
html = r.text
print("HTTP", r.status_code, "bytes", len(html))
import re as _re
m = _re.search(r"const hDays = (\[[^\]]*\]);", html)
print("hDays array:", m.group(1) if m else "MISSING")
m2 = _re.search(r"const cumPnl = (\[[^\]]*\]);", html)
print("cumPnl array:", m2.group(1) if m2 else "MISSING")
print("draws defensively (IIFE+try):", "function drawEquity()" in html)
print("has account-equity div:", 'id="account-equity"' in html)
print("has account-score div:", 'id="account-score"' in html)
print("equity chart script present:",
      "getElementById('account-equity')" in html)
print("zfxChart lib include:",
      [m for m in re.findall(r'<script[^>]*src="([^"]+)"', html)][:10])
# order: where does the lib script appear vs the equity script?
lib_pos = html.find("zfxChart =")
if lib_pos < 0:
    lib_pos = max(html.find("function zfxChart"), html.find("zfxChart="))
eq_pos = html.find("getElementById('account-equity')")
print(f"inline zfxChart def pos {lib_pos} | equity chart call pos {eq_pos}")
# hunt obvious JS breakers in the inline blocks
for pat in ("NaN,", ": NaN", "None", "Infinity"):
    seg = [m.start() for m in re.finditer(re.escape(pat), html)]
    if seg:
        s = seg[0]
        print(f"suspicious '{pat}' x{len(seg)} first at {s}: "
              f"...{html[max(0, s-80):s+60]!r}...")
        break
# dump the 300 chars before the equity chart call for inspection
if eq_pos > 0:
    print("---- context before equity call ----")
    print(html[eq_pos-400:eq_pos+120])
