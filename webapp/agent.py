"""LLM operations agent: any question, any task, across the three sources.

The existing /api/ask assistant is deliberately deterministic. This module is
the other half the operators asked for: a tool-using LLM that knows WHAT data
exists (the hardcoded catalog below -- BigQuery snapshot, production MySQL,
Kafka/local caches), runs read-only queries against it, returns tables and CSV
files, and can drive the web app itself ("autopilot") by returning navigation
steps the UI performs visibly.

SAFETY RAILS
- SQL tools are SELECT-only (regex-gated, comments stripped, one statement,
  row-capped). MySQL uses the reporting proxy credentials already in use.
- The agent never writes to any store; CSV exports land in artifacts/exports.
- Admin-only at the route layer: it can read client data, so it inherits the
  same gate as the tabs that show client data.

The Anthropic key comes from the ANTHROPIC_API_KEY environment variable or
artifacts/agent_key.txt (saved via the settings endpoint). With no key the
endpoint degrades to the deterministic assistant so the dock always answers.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPORTS = ROOT / "artifacts" / "exports"
KEY_FILE = ROOT / "artifacts" / "agent_key.txt"
WORKSPACE_FILE = ROOT / "artifacts" / "agent_workspace.txt"

MODEL = "claude-sonnet-4-5"
MAX_TOOL_ROUNDS = 24          # deep multi-step tasks; cancel stops a runaway loop
ROW_CAP = 500

# Cooperative cancellation: the dock POSTs a request id to /api/agent/cancel,
# which lands the id here; run() checks between tool rounds and bails, so a
# cancelled request stops calling the model and the tools instead of running on.
_CANCELLED: set = set()


def request_cancel(request_id: str) -> None:
    if request_id:
        _CANCELLED.add(str(request_id))


def _is_cancelled(request_id) -> bool:
    return bool(request_id) and str(request_id) in _CANCELLED

# ---------------------------------------------------------------- catalog
#: What the agent can actually reach, spelled out so it plans around
#: freshness and cost instead of hallucinating tables.
CATALOG = """
DATA SOURCES (hardcoded truth -- plan queries against THESE):

1. PRODUCTION MYSQL (live, ~2.5s lag; read-only reporting proxy).
   Servers: mt4_live01..04 (tables: orders [one row per trade: login,
   symbol_name, cmd 0=buy 1=sell, volume (lots*100), open_price, open_ts,
   close_price, close_ts, profit, swap, commission, sl, tp], users [login,
   `group`, name, country, ...], groups [`group`, currency -- currency='CNT'
   marks CENT accounts whose money values are x100]), and mt5_live01
   (tables: deals [deal, position_id, login, symbol, action 0=buy 1=sell,
   entry 0=in 1/3=out, volume (lots*10000), price, profit, storage=swap,
   commission, `time`], users, positions). Timestamps are unix epoch on mt4
   (open_ts/close_ts) and DATETIME `time` on mt5.
   Tool: mysql_query(server, sql). Cent accounts: divide money/lots by 100.

2. LOCAL ANALYTICS CACHES (DuckDB/parquet/sqlite -- fast, big, seconds to query).
   Registered in duckdb_query as views:
   - records: 33.5M client trades May 29 -> Aug 27 snapshot from BigQuery
     (database, account_key='server:login', symbol, cmd, volume_lots,
      open_time, close_time, open_price, close_price, net_profit, state...).
   - ad_corpus: per-account-per-day behaviour features (account_key,
     decision_day + 174 ad_* columns), refreshed daily to yesterday.
   - vantage_orders: OUR copy-trader engine's own orders (created epoch,
     source_account, stance copy/invert/fade15, symbol, our_lots,
     expected_usd, live_score, status, ticket, detail).
   - events: the live engine feed store (server, login, symbol, action,
     volume, price, event_time) -- most recent client flow.
   Tool: duckdb_query(sql) -- standard SQL, joins across views allowed.

3. KAFKA (event bus). UAT cluster wired for quotes; PROD clusters (Zeal/AE/ID)
   verified reachable with deals/trades topics for all six servers but not yet
   the decision feed. Status via app_status(); no ad-hoc consume tool.

4. THE WEB APP ITSELF (for autopilot). Sections: /hub, /executive,
   /trading/{summary,overview,abook,clients,antifraud,account,exposure,
   regions,cashflow,surveillance,bookrisk,risk,performance,validation},
   /quant/{summary,overview,signals,copytrader,exposure,regions,cashflow,
   exits,research,risk,performance,validation}, /vantage (live engine),
   /data (warehouse), /admin, /settings.
   To drill an account: /trading/account?account=mt4_live01:12345.
"""

SYSTEM = f"""You are the ZFX Risk Intelligence operations agent, embedded in a
broker risk analytics web app. You answer questions and perform tasks using
ONLY the tools provided. Be precise, quantitative and terse; every number you
state must come from a tool result, never from memory.

{CATALOG}

SCOPE -- STRICT. You exist ONLY to help with THIS web app: its data, trading and
broker risk analytics, client behaviour, the copy-trader engine, anti-fraud
screening, data processing, and driving the app's own pages. You must REFUSE,
briefly and politely, anything outside that: personal advice, general knowledge,
coding help unrelated to this data, current events, chit-chat, creative writing,
or any question a generic chatbot would answer. For an off-topic request, call
final_answer with text like: "That's outside what I do -- I'm the ZFX risk
analytics agent, so I can only help with this app's data and operations." Do NOT
answer the off-topic part even partially. On-topic requests: be maximally useful.

RULES:
- SELECT-only SQL. Always LIMIT (<= {ROW_CAP}). Prefer local caches for
  history/aggregates; MySQL only for live/recent state.
- Cent accounts: groups.currency='CNT' on mt4; deflate money/lots by 100.
- When the user asks to "go to", "show", "open" a page: use the navigate tool.
  You can drive ANY page a user can reach, including drill-downs with query
  params (e.g. /trading/account?account=mt5_live01:105036806, or /replay).
- Every tabular result MUST go in final_answer's `rows`, and set csv=true so the
  operator can download it -- grounded, exportable data is the whole point;
  never dump a big table as prose. State the row count and the source you used.
- If a task is impossible with these tools, say exactly what is missing.
"""

TOOLS = [
    {"name": "mysql_query",
     "description": "Read-only SELECT against one production MySQL server "
                    "(mt4_live01..04, mt5_live01). ~2.5s-fresh live data.",
     "input_schema": {"type": "object", "properties": {
         "server": {"type": "string"},
         "sql": {"type": "string"}}, "required": ["server", "sql"]}},
    {"name": "duckdb_query",
     "description": "SQL over the local analytics views: records (BQ 90d "
                    "trades), ad_corpus, vantage_orders, events.",
     "input_schema": {"type": "object", "properties": {
         "sql": {"type": "string"}}, "required": ["sql"]}},
    {"name": "app_status",
     "description": "Live engine + account + feed snapshot from the running "
                    "copy-trader (balance, equity, positions, lag, strategies).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "navigate",
     "description": "Autopilot: send the user's browser to an app path (e.g. "
                    "/trading/antifraud). The UI shows the step and goes.",
     "input_schema": {"type": "object", "properties": {
         "path": {"type": "string"},
         "reason": {"type": "string"}}, "required": ["path"]}},
    {"name": "final_answer",
     "description": "Finish: the answer text, optional table rows, optional "
                    "csv export of those rows.",
     "input_schema": {"type": "object", "properties": {
         "text": {"type": "string"},
         "rows": {"type": "array", "items": {"type": "object"}},
         "csv": {"type": "boolean"}}, "required": ["text"]}},
]

_SELECT_ONLY = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|grant|truncate|replace|"
    r"attach|copy|export|install|load)\b", re.IGNORECASE)


def _guard_sql(sql: str) -> str:
    cleaned = re.sub(r"--.*?$|/\*.*?\*/", " ", sql, flags=re.S | re.M).strip()
    cleaned = cleaned.rstrip(";").strip()
    if ";" in cleaned:
        raise ValueError("one statement only")
    if not _SELECT_ONLY.match(cleaned) or _FORBIDDEN.search(cleaned):
        raise ValueError("SELECT-only")
    return cleaned


def _rows_out(frame) -> list[dict]:
    import numpy as np
    frame = frame.head(ROW_CAP)
    out = []
    for record in frame.to_dict("records"):
        clean = {}
        for key, value in record.items():
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                clean[key] = None
            elif hasattr(value, "isoformat"):
                clean[key] = value.isoformat(sep=" ")[:19]
            elif isinstance(value, (np.integer,)):
                clean[key] = int(value)
            elif isinstance(value, (np.floating,)):
                clean[key] = round(float(value), 6)
            else:
                clean[key] = value
        out.append(clean)
    return out


# ---------------------------------------------------------------- tools impl
def _tool_mysql(server: str, sql: str) -> dict:
    from webapp import trade_feed
    if server not in trade_feed._servers():
        return {"error": f"unknown server {server}; use one of {trade_feed._servers()}"}
    cleaned = _guard_sql(sql)
    if not re.search(r"\blimit\s+\d+", cleaned, re.IGNORECASE):
        cleaned += f" LIMIT {ROW_CAP}"
    import pandas as pd
    connection = trade_feed._connection(server)
    with connection.cursor() as cursor:
        cursor.execute(cleaned)
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchmany(ROW_CAP)
    return {"columns": columns,
            "rows": _rows_out(pd.DataFrame(rows, columns=columns))}


_DUCK = None


def _duck():
    global _DUCK
    if _DUCK is not None:
        return _DUCK
    import duckdb
    from webapp import trade_features as tf
    con = duckdb.connect()
    con.execute(f"CREATE VIEW records AS SELECT * FROM read_parquet('{(tf._AD_DIR / 'bq_90d_records.parquet').as_posix()}')")
    con.execute(f"CREATE VIEW ad_corpus AS SELECT * FROM read_parquet('{(tf._AD_DIR / 'model_frame.parquet').as_posix()}')")
    con.execute(f"ATTACH '{(ROOT / 'app.db').as_posix()}' AS appdb (TYPE sqlite)")
    con.execute("CREATE VIEW vantage_orders AS SELECT * FROM appdb.vantage_orders")
    store = ROOT / "artifacts" / "live_stream.duckdb"
    if store.exists():
        con.execute(f"ATTACH '{store.as_posix()}' AS live (READ_ONLY)")
        try:
            con.execute("CREATE VIEW events AS SELECT * FROM live.events")
        except Exception:
            pass
    _DUCK = con
    return con


def _tool_duckdb(sql: str) -> dict:
    cleaned = _guard_sql(sql)
    frame = _duck().execute(cleaned).df()
    return {"columns": list(frame.columns), "rows": _rows_out(frame)}


def _tool_status() -> dict:
    try:
        from webapp import vantage
    except ImportError:
        return {"engine_installed": False,
                "note": "the Vantage copy-trading engine is not part of this installation"}
    report = vantage.report()
    engine = vantage.state()
    return {"engine_running": engine.running, "scored": engine.scored,
            "filled": engine.filled,
            "snapshot": report.get("snapshot"),
            "open_positions": len(report.get("positions") or []),
            "strategies": {k: {"enabled": v.get("enabled"),
                               "floating": v.get("floating"),
                               "realized_30d": (v.get("perf") or {}).get("realized_usd")}
                           for k, v in (report.get("strategies") or {}).items()
                           if isinstance(v, dict)},
            "stream": report.get("stream")}


def _save_csv(rows: list[dict]) -> str | None:
    if not rows:
        return None
    import csv
    EXPORTS.mkdir(parents=True, exist_ok=True)
    name = f"agent_{int(time.time())}.csv"
    path = EXPORTS / name
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return f"/agent/export/{name}"


# ---------------------------------------------------------------- the loop
def api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    try:
        key = KEY_FILE.read_text(encoding="utf-8").strip()
        return key or None
    except Exception:
        return None


def workspace_id() -> str | None:
    wid = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
    if wid:
        return wid
    try:
        wid = WORKSPACE_FILE.read_text(encoding="utf-8").strip()
        return wid or None
    except Exception:
        return None


def save_key(key: str) -> None:
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    key = key.strip()
    # A workspace id may be pasted alongside the key as "key|wsid" or on a
    # second line -- newer identity-linked keys need it as a request header.
    wid = ""
    for sep in ("|", "\n", " workspace:"):
        if sep in key:
            key, wid = key.split(sep, 1)[0].strip(), key.split(sep, 1)[1].strip()
            break
    KEY_FILE.write_text(key, encoding="utf-8")
    if wid:
        save_workspace(wid)


def save_workspace(wid: str) -> None:
    WORKSPACE_FILE.parent.mkdir(parents=True, exist_ok=True)
    WORKSPACE_FILE.write_text(wid.strip(), encoding="utf-8")


def run(question: str, history: list[dict] | None = None,
        request_id: str | None = None) -> dict:
    """One agent turn. Returns {text, columns?, table?, csv_url?, navigate?,
    steps: [tool summaries], degraded?}. Cancellable via request_cancel()."""
    key = api_key()
    if not key:
        return _degraded(question)
    try:
        import anthropic
    except ImportError:
        return _degraded(question, note="anthropic package not installed -- "
                                         "pip install anthropic")
    wid = workspace_id()
    headers = {"anthropic-workspace-id": wid} if wid else None
    client = anthropic.Anthropic(api_key=key, default_headers=headers)
    messages = list(history or [])
    messages.append({"role": "user", "content": question})
    steps: list[str] = []
    navigate_to = None
    for _ in range(MAX_TOOL_ROUNDS):
        if _is_cancelled(request_id):
            _CANCELLED.discard(str(request_id))
            return {"text": "Cancelled.", "steps": steps, "navigate": navigate_to,
                    "cancelled": True}
        try:
            response = client.messages.create(
                model=MODEL, max_tokens=2000, system=SYSTEM,
                tools=TOOLS, messages=messages)
        except Exception as error:
            msg = str(error)
            if "anthropic-workspace-id" in msg or "workspace" in msg.lower():
                return {"text": "This API key is identity-linked and needs a "
                        "workspace ID. Paste it in the agent dock's key box as "
                        "`<key>|<workspace-id>`, or set ANTHROPIC_WORKSPACE_ID. "
                        "Find it in the Anthropic Console URL "
                        "(console.anthropic.com/settings/workspaces).",
                        "steps": steps, "degraded": True}
            if "model" in msg.lower() and ("not_found" in msg or "404" in msg):
                return {"text": f"The configured model '{MODEL}' isn't available "
                        "to this key. Ask an admin to enable it or change MODEL.",
                        "steps": steps, "degraded": True}
            return {"text": f"LLM error: {type(error).__name__}: {msg[:300]}",
                    "steps": steps, "degraded": True}
        blocks = response.content
        tool_uses = [b for b in blocks if b.type == "tool_use"]
        text_parts = [b.text for b in blocks if b.type == "text"]
        if not tool_uses:
            return {"text": "\n".join(text_parts).strip() or "(no answer)",
                    "steps": steps, "navigate": navigate_to,
                    "history": _fold(messages, blocks)}
        messages.append({"role": "assistant", "content": blocks})
        results = []
        for use in tool_uses:
            name, args = use.name, use.input or {}
            try:
                if name == "mysql_query":
                    out = _tool_mysql(str(args.get("server", "")), str(args.get("sql", "")))
                    steps.append(f"MySQL {args.get('server')}: {str(args.get('sql'))[:90]}")
                elif name == "duckdb_query":
                    out = _tool_duckdb(str(args.get("sql", "")))
                    steps.append(f"DuckDB: {str(args.get('sql'))[:90]}")
                elif name == "app_status":
                    out = _tool_status()
                    steps.append("engine status")
                elif name == "navigate":
                    navigate_to = str(args.get("path", "/hub"))
                    out = {"ok": True, "navigating": navigate_to}
                    steps.append(f"navigate -> {navigate_to}")
                elif name == "final_answer":
                    rows = args.get("rows") or []
                    csv_url = _save_csv(rows) if args.get("csv") and rows else None
                    return {"text": str(args.get("text", "")),
                            "columns": list(rows[0].keys()) if rows else None,
                            "table": rows or None, "csv_url": csv_url,
                            "navigate": navigate_to, "steps": steps,
                            "history": _fold(messages, blocks)}
                else:
                    out = {"error": f"unknown tool {name}"}
            except Exception as error:
                out = {"error": f"{type(error).__name__}: {error}"}
                steps.append(f"{name} ERROR: {error}")
            payload = json.dumps(out, default=str)
            if len(payload) > 60000:
                payload = payload[:60000] + '... (truncated)"}'
            results.append({"type": "tool_result", "tool_use_id": use.id,
                            "content": payload})
        messages.append({"role": "user", "content": results})
    return {"text": "Stopped after the tool-round limit -- narrow the task.",
            "steps": steps, "navigate": navigate_to}


def _fold(messages, last_blocks) -> list[dict]:
    """Compact serialisable history: user text + final assistant text only
    (tool traffic is too heavy to round-trip through the browser)."""
    folded = []
    for message in messages:
        if isinstance(message.get("content"), str):
            folded.append({"role": message["role"], "content": message["content"]})
    text = " ".join(b.text for b in last_blocks
                    if getattr(b, "type", "") == "text").strip()
    if text:
        folded.append({"role": "assistant", "content": text})
    return folded[-10:]


def _degraded(question: str, note: str = "") -> dict:
    """No key: answer with the deterministic assistant so the dock never dies."""
    prefix = ((note + " -- ") if note else "") + (
        "No LLM key configured (paste one in the agent dock); answered by the "
        "built-in deterministic assistant instead. ")
    try:
        from webapp import assistant, model_service
        frame = model_service.load_scores(model_service.VIEW_TRADING)
        result = assistant.answer(
            question, frame, model_service.artifact_meta(model_service.VIEW_TRADING),
            "trading")
        return {"text": prefix + (result.text or ""), "table": result.table,
                "columns": result.columns, "degraded": True}
    except Exception:
        return {"text": prefix, "degraded": True}
