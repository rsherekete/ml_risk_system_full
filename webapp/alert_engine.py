"""User-defined alert builder: source -> field -> condition -> webhook.

Each user composes alerts from a FIELD CATALOG that is honest about what each
data source can actually answer right now:

  * mysql   -- live aggregates over the production servers' closed trades,
               computed over the alert's own refresh window (~2.5s-fresh data).
  * bq      -- the behavioural corpus (account-day features): the alert fires
               when ANY account's metric crosses the bar, and names the top
               offenders in the message.
  * kafka   -- the live engine/stream telemetry (feed lag, engine state,
               equity, margin) -- today served from the engine's own snapshot;
               the same fields move to the production Kafka consumers when
               that feed is wired.

Alerts persist per USER in app.db, evaluate on their own frequency in one
background thread, deliver to the alert's webhook (or LARK_WEBHOOK), and every
firing is recorded with a delivery flag read from Lark's response -- the green
tick / red cross the UI shows -- plus a free-text comment per firing.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = ROOT / "app.db"

# ---------------------------------------------------------------- catalog
FIELD_CATALOG = {
    "mysql": {
        "label": "Production MySQL (live, ~2.5s)",
        "fields": {
            "closed_trades": "Closed trades across all servers in the window",
            "closed_net_profit": "Sum of client net P&L closed in the window ($)",
            "max_single_win": "Largest single client win closed in the window ($)",
            "max_single_loss": "Largest single client loss closed in the window ($, negative)",
            "closed_volume_lots": "Total lots closed in the window",
        }},
    "bq": {
        "label": "Behaviour corpus (account-day, daily)",
        "fields": {}},   # filled lazily from the live panel columns
    "kafka": {
        "label": "Engine / stream telemetry (live)",
        "fields": {
            "feed_lag_seconds": "Age of the newest scored client trade (s)",
            "engine_running": "Copy-trader engine running (1) or stopped (0)",
            "equity": "Vantage account equity ($)",
            "floating_pnl": "Vantage floating P&L ($)",
            "open_positions": "Vantage open position count",
            "margin_level_pct": "Vantage margin level (%)",
        }},
    "antifraud": {
        "label": "AntiFraud classifications (rules + custom)",
        "fields": {}},   # filled lazily: per-profile flag counts + expression
}

OPS = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
       "<": lambda a, b: a < b, "<=": lambda a, b: a <= b}


def catalog() -> dict:
    out = json.loads(json.dumps(FIELD_CATALOG))
    try:
        from webapp import antifraud
        import pandas as pd
        panel = antifraud._account_panel()
        out["bq"]["fields"] = {
            c: f"per-account behaviour metric (fires when ANY account crosses)"
            for c in sorted(panel.columns)
            if pd.api.types.is_numeric_dtype(panel[c])}
    except Exception:
        out["bq"]["fields"] = {"profit_factor": "per-account (panel warming)"}
    try:
        from webapp import antifraud
        fields = {f"flags_{p}": f"number of accounts currently flagged {p}"
                  for p in antifraud.PROFILES}
        fields["flags_total"] = "total standing flags across all profiles"
        for custom in (antifraud.load_rules().get("_custom") or []):
            fields[f"flags_custom:{custom.get('name')}"] = \
                f"accounts flagged by custom rule '{custom.get('name')}'"
        fields["expr"] = ("COMPLEX: count of accounts matching a boolean "
                          "expression over behaviour metrics (write it in the "
                          "Expression box; condition compares the COUNT)")
        out["antifraud"]["fields"] = fields
    except Exception:
        pass
    return out


# ---------------------------------------------------------------- storage
def _ensure() -> None:
    with sqlite3.connect(DB) as cx:
        cx.execute("""CREATE TABLE IF NOT EXISTS user_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, name TEXT,
            source TEXT, field TEXT, op TEXT, value REAL,
            freq_minutes REAL, webhook TEXT, enabled INTEGER DEFAULT 1,
            created REAL, last_run REAL DEFAULT 0, last_state INTEGER DEFAULT 0)""")
        cx.execute("""CREATE TABLE IF NOT EXISTS alert_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, alert_id INTEGER,
            fired_at REAL, observed REAL, message TEXT,
            delivered INTEGER, response TEXT, comment TEXT)""")
        try:                          # migration: complex-expression alerts
            cx.execute("ALTER TABLE user_alerts ADD COLUMN expr TEXT DEFAULT ''")
        except Exception:
            pass


def list_alerts(username: str) -> list[dict]:
    _ensure()
    with sqlite3.connect(DB) as cx:
        cx.row_factory = sqlite3.Row
        return [dict(r) for r in cx.execute(
            "SELECT * FROM user_alerts WHERE username = ? ORDER BY id DESC",
            (username,))]


def save_alert(username: str, spec: dict) -> int:
    _ensure()
    with sqlite3.connect(DB) as cx:
        cur = cx.execute(
            "INSERT INTO user_alerts (username, name, source, field, op, value,"
            " freq_minutes, webhook, enabled, created, expr)"
            " VALUES (?,?,?,?,?,?,?,?,1,?,?)",
            (username, str(spec.get("name") or spec.get("field")),
             str(spec.get("source")), str(spec.get("field")),
             str(spec.get("op", ">")), float(spec.get("value", 0)),
             max(1.0, float(spec.get("freq_minutes", 5))),
             str(spec.get("webhook") or ""), time.time(),
             str(spec.get("expr") or "")))
        return int(cur.lastrowid)


def delete_alert(username: str, alert_id: int) -> None:
    _ensure()
    with sqlite3.connect(DB) as cx:
        cx.execute("DELETE FROM user_alerts WHERE id = ? AND username = ?",
                   (alert_id, username))


def toggle_alert(username: str, alert_id: int, enabled: bool) -> None:
    _ensure()
    with sqlite3.connect(DB) as cx:
        cx.execute("UPDATE user_alerts SET enabled = ? WHERE id = ? AND username = ?",
                   (1 if enabled else 0, alert_id, username))


def history(username: str, alert_id: int, limit: int = 50) -> list[dict]:
    _ensure()
    with sqlite3.connect(DB) as cx:
        cx.row_factory = sqlite3.Row
        return [dict(r) for r in cx.execute(
            "SELECT h.* FROM alert_history h JOIN user_alerts a ON a.id = h.alert_id"
            " WHERE h.alert_id = ? AND a.username = ? ORDER BY h.id DESC LIMIT ?",
            (alert_id, username, limit))]


def comment(username: str, history_id: int, text: str) -> None:
    _ensure()
    with sqlite3.connect(DB) as cx:
        cx.execute(
            "UPDATE alert_history SET comment = ? WHERE id = ? AND alert_id IN "
            "(SELECT id FROM user_alerts WHERE username = ?)",
            (text[:2000], history_id, username))


# ---------------------------------------------------------------- evaluate
def _value_mysql(field: str, window_minutes: float) -> tuple[float, str]:
    from webapp import trade_feed as tf
    since = int(time.time() - window_minutes * 60)
    total = trades = biggest_win = 0.0
    biggest_loss = 0.0
    lots = 0.0
    for server in tf._servers():
        try:
            con = tf._connection(server)
            with con.cursor() as cur:
                if server.startswith("mt4"):
                    cur.execute(
                        "SELECT COUNT(*), COALESCE(SUM(profit),0),"
                        " COALESCE(MAX(profit),0), COALESCE(MIN(profit),0),"
                        " COALESCE(SUM(volume),0)/100 FROM orders"
                        " WHERE close_ts > %s AND cmd IN (0,1)", [since])
                else:
                    import datetime as _dt
                    cur.execute(
                        "SELECT COUNT(*), COALESCE(SUM(profit),0),"
                        " COALESCE(MAX(profit),0), COALESCE(MIN(profit),0),"
                        " COALESCE(SUM(volume),0)/10000 FROM deals"
                        " WHERE `time` > %s AND entry IN (1,3) AND action IN (0,1)",
                        [_dt.datetime.fromtimestamp(since)])
                n, s, mx, mn, v = cur.fetchone()
            trades += float(n or 0); total += float(s or 0)
            biggest_win = max(biggest_win, float(mx or 0))
            biggest_loss = min(biggest_loss, float(mn or 0))
            lots += float(v or 0)
        except Exception:
            continue
    values = {"closed_trades": trades, "closed_net_profit": total,
              "max_single_win": biggest_win, "max_single_loss": biggest_loss,
              "closed_volume_lots": lots}
    return values.get(field, float("nan")), f"window {window_minutes:g}m, all servers"


def _value_bq(field: str, op: str, value: float) -> tuple[float, str]:
    import pandas as pd
    from webapp import antifraud
    panel = antifraud._account_panel()
    series = pd.to_numeric(panel[field], errors="coerce").dropna()
    if series.empty:
        return float("nan"), "no data"
    compare = OPS[op]
    hits = series[compare(series, value)]
    extreme = float(series.max() if op in (">", ">=") else series.min())
    detail = (f"{len(hits)} accounts cross; top: "
              + ", ".join(f"{a}={v:.4g}" for a, v in
                          hits.sort_values(ascending=(op in ("<", "<="))).head(3).items())
              if len(hits) else "none cross")
    return extreme, detail


def _value_kafka(field: str) -> tuple[float, str]:
    try:
        from webapp import vantage
    except ImportError:
        return float("nan"), "engine not installed"
    report = vantage.report()
    state = vantage.state()
    snapshot = report.get("snapshot") or {}
    signals = state.signals or []
    lag = float("nan")
    if signals:
        try:
            import pandas as pd
            newest = max(pd.Timestamp(s["traded"]).timestamp()
                         for s in signals if s.get("traded"))
            lag = max(0.0, time.time() - newest)
        except Exception:
            pass
    margin = snapshot.get("margin") or 0.0
    level = (float(snapshot.get("equity") or 0) / margin * 100.0) if margin else 1e9
    values = {"feed_lag_seconds": lag,
              "engine_running": 1.0 if state.running else 0.0,
              "equity": float(snapshot.get("equity") or 0),
              "floating_pnl": float(snapshot.get("profit") or 0),
              "open_positions": float(len(report.get("positions") or [])),
              "margin_level_pct": level}
    return values.get(field, float("nan")), "engine snapshot"


def webhook_problem(url: str) -> str | None:
    """Why a webhook can't work, or None if it looks valid. A Lark BOT webhook
    is https://open.larksuite.com/open-apis/bot/v2/hook/<token> (or the .feishu
    /.larkoffice equivalents). An applink 'add_by_link' URL is a chat-INVITE,
    not a webhook -- posting to it silently does nothing, which is exactly the
    'pasted a link, nothing sent' symptom."""
    url = (url or "").strip()
    if not url:
        return "no webhook set"
    if "add_by_link" in url or "applink." in url:
        return ("that is a chat-INVITE link, not a bot webhook. In Lark: open "
                "the target chat -> Settings -> Bots -> Add Bot -> Custom Bot, "
                "and copy its webhook (open.larksuite.com/open-apis/bot/v2/hook/...)")
    if "/open-apis/bot/v2/hook/" not in url:
        return ("does not look like a Lark bot webhook "
                "(expected .../open-apis/bot/v2/hook/<token>)")
    return None


def _send(webhook: str, text: str) -> tuple[bool, str]:
    """Deliver + CONFIRM: Lark answers {"code":0} on success; anything else is
    a real delivery failure the history shows as the red cross."""
    import urllib.request
    problem = webhook_problem(webhook)
    if problem:
        return False, problem
    card = {"msg_type": "text", "content": {"text": text}}
    try:
        request = urllib.request.Request(
            webhook, data=json.dumps(card).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=8) as response:
            body = response.read().decode("utf-8", "replace")[:500]
        try:
            ok = json.loads(body).get("code", 1) == 0
        except Exception:
            ok = True                                    # 200 with odd body
        return ok, body
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def _value_antifraud(field: str, expr: str) -> tuple[float, str]:
    from webapp import antifraud
    if field == "expr" and expr:
        panel = antifraud._account_panel()
        mask = antifraud.safe_expr_mask(expr, panel)
        hits = panel.index[mask].tolist()
        return float(len(hits)), (f"expr [{expr}]: " +
                                  (", ".join(hits[:5]) if hits else "no accounts"))
    flags = antifraud.classify()
    if flags.empty:
        return 0.0, "no standing flags"
    if field == "flags_total":
        return float(len(flags)), f"{flags['account_key'].nunique()} accounts"
    profile = field.replace("flags_", "", 1)
    mine = flags.loc[flags["profile"] == profile]
    tops = ", ".join(mine.head(3)["account_key"]) if len(mine) else "none"
    return float(len(mine)), f"{profile}: {tops}"


def _evaluate(alert: dict) -> None:
    import math
    field, op = alert["field"], alert["op"]
    if op not in OPS:
        return
    if alert["source"] == "mysql":
        observed, detail = _value_mysql(field, float(alert["freq_minutes"]))
    elif alert["source"] == "bq":
        observed, detail = _value_bq(field, op, float(alert["value"]))
    elif alert["source"] == "antifraud":
        observed, detail = _value_antifraud(field, str(alert.get("expr") or ""))
    else:
        observed, detail = _value_kafka(field)
    if observed is None or (isinstance(observed, float) and math.isnan(observed)):
        return
    fired = OPS[op](observed, float(alert["value"]))
    with sqlite3.connect(DB) as cx:
        cx.execute("UPDATE user_alerts SET last_run = ?, last_state = ? WHERE id = ?",
                   (time.time(), 1 if fired else 0, alert["id"]))
        if not fired or alert["last_state"] == 1:
            return                       # fire on ENTERING the condition only
        webhook = alert["webhook"] or os.environ.get("LARK_WEBHOOK", "")
        text = (f"🔔 {alert['name']} — {alert['source']}.{field} "
                f"{op} {alert['value']:g}\nobserved: {observed:g}\n{detail}")
        delivered, response = (_send(webhook, text) if webhook
                               else (False, "no webhook configured"))
        cx.execute(
            "INSERT INTO alert_history (alert_id, fired_at, observed, message,"
            " delivered, response, comment) VALUES (?,?,?,?,?,?, '')",
            (alert["id"], time.time(), float(observed), text,
             1 if delivered else 0, response[:500]))


def analysis(username: str) -> dict:
    """Per-alert firing analytics: counts, delivery rate, recent observed
    values (chart-ready) -- the Alert Analysis panel's data."""
    _ensure()
    with sqlite3.connect(DB) as cx:
        cx.row_factory = sqlite3.Row
        alerts = [dict(r) for r in cx.execute(
            "SELECT * FROM user_alerts WHERE username = ?", (username,))]
        out = []
        for alert in alerts:
            rows = [dict(r) for r in cx.execute(
                "SELECT fired_at, observed, delivered FROM alert_history"
                " WHERE alert_id = ? ORDER BY id DESC LIMIT 100",
                (alert["id"],))]
            fires = len(rows)
            delivered = sum(1 for r in rows if r["delivered"])
            out.append({
                "id": alert["id"], "name": alert["name"],
                "rule": f"{alert['source']}.{alert['field']} {alert['op']} {alert['value']}"
                        + (f" [{alert['expr']}]" if alert.get("expr") else ""),
                "enabled": bool(alert["enabled"]), "fires": fires,
                "delivered": delivered,
                "delivery_rate": round(delivered / fires, 3) if fires else None,
                "last_fired": rows[0]["fired_at"] if rows else None,
                "series": [{"t": r["fired_at"], "v": r["observed"],
                            "ok": bool(r["delivered"])} for r in reversed(rows)]})
    return {"alerts": out}


_THREAD = None


def start() -> None:
    global _THREAD
    if _THREAD is not None and _THREAD.is_alive():
        return

    def _loop():
        while True:
            try:
                _ensure()
                with sqlite3.connect(DB) as cx:
                    cx.row_factory = sqlite3.Row
                    due = [dict(r) for r in cx.execute(
                        "SELECT * FROM user_alerts WHERE enabled = 1")]
                now = time.time()
                for alert in due:
                    if now - float(alert["last_run"] or 0) >= float(alert["freq_minutes"]) * 60:
                        try:
                            _evaluate(alert)
                        except Exception:
                            pass
            except Exception:
                pass
            time.sleep(30)

    _THREAD = threading.Thread(target=_loop, daemon=True, name="alert-engine")
    _THREAD.start()
