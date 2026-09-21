"""The data warehouse (BigQuery, project zfx-dwh-prod) from this machine.

Google's front end refuses `bigquery.googleapis.com` from the office egress
address that the Fortinet full tunnel gives every packet (HTML 403 before
authentication; other Google APIs answer normally from the same address).
Bound to another local interface the same request is answered (401 without
a token, data with one). So the client here pins each HTTPS connection to a
source address that works: the operator's override first (`BQ_SOURCE_IP`),
then whichever non-VPN interface Google answers on, found by a 12-second
probe and remembered for the process. Authentication is the gcloud
application-default credential, exactly as `dubai_backfill` uses.

Two facts the analysis needs live only here:

* `rebate_payout` -- the rebate the firm pays per closed trade
  (data_marts.closed_trades, partitioned by close_time_london_trading_date,
  clustered by login), the figure Metabase shows.
* `primary_trading_account_number` -- the client's primary account behind
  every trading account (data_marts.trading_accounts), i.e. the desk's
  primary / sub-account structure.
"""
from __future__ import annotations

import os
import re
import socket
import ssl
import threading
import time
import warnings
from collections import defaultdict

import pandas as pd

# gcloud user credentials without a quota project: the warning is right and
# harmless here (the project is billed), and it fired on every call.
warnings.filterwarnings("ignore", message=".*without a quota project.*")

PROJECT = "zfx-dwh-prod"
CLOSED_TRADES = f"{PROJECT}.data_marts.closed_trades"
TRADING_ACCOUNTS = f"{PROJECT}.data_marts.trading_accounts"
_HOST = "bigquery.googleapis.com"
_LOCK = threading.Lock()
_STATE: dict = {"source_ip": None, "checked": 0.0, "ok": None, "reason": ""}
_TTL = 600.0


# ---------------------------------------------------------------- transport
def _candidates() -> list[str | None]:
    """Source addresses to try: the override, the default route, then every
    routable IPv4 on the box (not loopback, not link-local)."""
    out: list[str | None] = []
    override = os.environ.get("BQ_SOURCE_IP", "").strip()
    if override:
        out.append(override)
    out.append(None)
    try:
        import psutil
        for name, addrs in psutil.net_if_addrs().items():
            for a in addrs:
                if getattr(a.family, "name", "") == "AF_INET" and not a.address.startswith(("127.", "169.254.")):
                    if a.address not in out:
                        out.append(a.address)
    except Exception:
        pass
    return out


def _raw_status(src_ip: str | None, timeout: float = 6.0) -> int:
    """HTTP status of an unauthenticated GET on the BigQuery API from this
    source address: 401 = reachable (a token is all that is missing), 403 =
    refused at the edge, anything else = no answer."""
    ip = socket.gethostbyname(_HOST)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        if src_ip:
            s.bind((src_ip, 0))
        s.connect((ip, 443))
        with ssl.create_default_context().wrap_socket(s, server_hostname=_HOST) as ss:
            ss.sendall(b"GET /bigquery/v2/projects/" + PROJECT.encode() + b"/datasets HTTP/1.1\r\nHost: "
                       + _HOST.encode() + b"\r\nConnection: close\r\n\r\n")
            head = ss.recv(64).split(b"\r\n")[0]
        m = re.search(rb"HTTP/1\.\d (\d{3})", head)
        return int(m.group(1)) if m else 0
    except Exception:
        return 0
    finally:
        try:
            s.close()
        except Exception:
            pass


def source_ip() -> tuple[str | None, bool, str]:
    """(source address to bind or None, reachable, reason). Cached 10 min."""
    with _LOCK:
        if time.time() - _STATE["checked"] < _TTL and _STATE["ok"] is not None:
            return _STATE["source_ip"], _STATE["ok"], _STATE["reason"]
    tried = []
    for cand in _candidates():
        status = _raw_status(cand)
        tried.append(f"{cand or 'default route'}={status}")
        if status == 401:
            with _LOCK:
                _STATE.update(source_ip=cand, checked=time.time(), ok=True,
                              reason=f"BigQuery answers via {cand or 'the default route'}")
            return cand, True, _STATE["reason"]
    with _LOCK:
        _STATE.update(source_ip=None, checked=time.time(), ok=False,
                      reason="BigQuery unreachable from every local address (" + ", ".join(tried) + ")")
    return None, False, _STATE["reason"]


def available() -> bool:
    return source_ip()[1]


def client():
    """A BigQuery client whose HTTPS connections leave from the working
    source address. Raises RuntimeError when the warehouse is unreachable."""
    src, ok, reason = source_ip()
    if not ok:
        raise RuntimeError(reason)
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    from google.cloud import bigquery
    from requests.adapters import HTTPAdapter
    from urllib3.poolmanager import PoolManager

    class _Bound(HTTPAdapter):
        def init_poolmanager(self, connections, maxsize, block=False, **kw):
            if src:
                kw["source_address"] = (src, 0)
            self.poolmanager = PoolManager(num_pools=connections, maxsize=maxsize, block=block, **kw)

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    session = AuthorizedSession(creds)
    session.mount("https://", _Bound())
    return bigquery.Client(project=PROJECT, _http=session)


# ------------------------------------------------------------- server names
#: data_marts.trading_accounts labels its servers by brand ('Live', 'Live2',
#: 'Traze-Live03', 'Real', 'ZealCapitalMarketSC-Live' ...); closed_trades
#: uses the app's own keys. Demo servers are not accounts we analyse.
_LABELS = {
    "live": "mt4_live01", "live1": "mt4_live01", "live01": "mt4_live01",
    "live2": "mt4_live02", "live02": "mt4_live02",
    "live3": "mt4_live03", "live03": "mt4_live03", "trazelive03": "mt4_live03",
    "live4": "mt4_live04", "live04": "mt4_live04",
    "real": "mt5_live01", "real01": "mt5_live01", "zealcapitalmarketsclive": "mt5_live01",
}


def server_key(name, login: int | None = None) -> str:
    """The warehouse's server label -> the app's server key (mt4_live01..04,
    mt5_live01, mt5_dubai_live01, mt5_indo_live01); the login's number range
    decides when the label is blank or unknown; '' for demo / unrecognised."""
    s = re.sub(r"[^a-z0-9]", "", str(name or "").lower())
    if s.startswith("demo"):
        return ""
    if s in _LABELS:
        return _LABELS[s]
    if "dubai" in s:
        return "mt5_dubai_live01"
    if "indo" in s:
        return "mt5_indo_live01"
    m = re.search(r"mt(4|5)(?:live)?0?(\d)", s)
    if m:
        return f"mt{m.group(1)}_live0{m.group(2)}"
    if login is not None:
        n = int(login)
        if 105_000_000 <= n < 106_000_000:
            return "mt5_live01"
        if 500_000_000 <= n < 600_000_000:
            return "mt5_dubai_live01"
        if 40_000_000 <= n < 41_000_000:
            return "mt5_indo_live01"
        if 6_000_000 <= n < 7_000_000:
            return "mt4_live04"
        if 4_000_000 <= n < 5_000_000:
            return "mt4_live03"
        if 2_000_000 <= n < 3_000_000:
            return "mt4_live02"
        if 200_000 <= n < 2_000_000:
            return "mt4_live01"
    return ""


def _by_server(accounts) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for a in accounts:
        server, _, login = str(a).rpartition(":")
        if server and login.isdigit():
            out[server].append(int(login))
    return out


def _key_for(server_name, login: int, wanted: dict[int, list[str]]) -> str | None:
    """account_key for a warehouse row: the recognised server label, else
    the one server the requested login lives on."""
    key = server_key(server_name, int(login))
    servers = wanted.get(int(login), [])
    if key and key in servers:
        return f"{key}:{login}"
    if len(servers) == 1:
        return f"{servers[0]}:{login}"
    if key and not servers:
        return f"{key}:{login}"
    return None


# ------------------------------------------------------------------ rebates
def rebate_payouts(accounts, days: int = 730) -> tuple[pd.DataFrame, list[str]]:
    """Per account_key: SUM(rebate_payout) over the closed trades of the last
    `days` days, records, first / last rebated trade, source label."""
    empty = pd.DataFrame(columns=["rebates_usd", "rebate_records", "rebate_first", "rebate_last", "rebate_source"])
    empty.index.name = "account_key"
    by = _by_server(accounts)
    logins = sorted({l for ls in by.values() for l in ls})
    if not logins:
        return empty, []
    wanted: dict[int, list[str]] = defaultdict(list)
    for server, ls in by.items():
        for l in ls:
            wanted[l].append(server)
    try:
        from google.cloud import bigquery
        cx = client()
        sql = f"""
            SELECT server_name, login, SUM(rebate_payout) AS reb, COUNTIF(rebate_payout <> 0) AS n,
                   MIN(close_time) AS t0, MAX(close_time) AS t1
            FROM `{CLOSED_TRADES}`
            WHERE close_time_london_trading_date >= DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY)
              AND login IN UNNEST(@logins) AND rebate_payout IS NOT NULL
            GROUP BY 1, 2"""
        job = cx.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("days", "INT64", int(days)),
            bigquery.ArrayQueryParameter("logins", "INT64", logins)]))
        rows = []
        unmapped = 0
        for r in job.result(timeout=300):
            key = _key_for(r.server_name, int(r.login), wanted)
            if key is None:
                unmapped += 1
                continue
            rows.append({"account_key": key, "rebates_usd": float(r.reb or 0), "rebate_records": int(r.n or 0),
                         "rebate_first": pd.Timestamp(r.t0) if r.t0 else pd.NaT,
                         "rebate_last": pd.Timestamp(r.t1) if r.t1 else pd.NaT,
                         "rebate_source": "DWH rebate_payout (data_marts.closed_trades)"})
        out = pd.DataFrame(rows).groupby("account_key").agg(
            rebates_usd=("rebates_usd", "sum"), rebate_records=("rebate_records", "sum"),
            rebate_first=("rebate_first", "min"), rebate_last=("rebate_last", "max"),
            rebate_source=("rebate_source", "first")) if rows else empty
        note = (f"rebates: DWH rebate_payout per trade, {days}-day window -- {len(out):,} of {len(logins):,} "
                f"affected logins have closed trades" + (f"; {unmapped} rows on an unrecognised server label" if unmapped else ""))
        return out, [note]
    except Exception as error:
        return empty, [f"rebates: DWH unavailable ({type(error).__name__}: {str(error)[:120]})"]


# ---------------------------------------------------------- account links
def account_links(accounts) -> tuple[pd.DataFrame, list[str]]:
    """Per requested account_key: primary_key and every member account_key of
    the same primary (any server), from data_marts.trading_accounts."""
    empty = pd.DataFrame(columns=["primary_key", "members", "link_source"])
    empty.index.name = "account_key"
    by = _by_server(accounts)
    logins = sorted({l for ls in by.values() for l in ls})
    if not logins:
        return empty, []
    wanted: dict[int, list[str]] = defaultdict(list)
    for server, ls in by.items():
        for l in ls:
            wanted[l].append(server)
    try:
        from google.cloud import bigquery
        cx = client()
        sql = f"""
            WITH mine AS (
              SELECT DISTINCT primary_trading_account_number AS prim
              FROM `{TRADING_ACCOUNTS}`
              WHERE account_number IN UNNEST(@logins) AND primary_trading_account_number IS NOT NULL)
            SELECT t.trading_server, t.account_number, t.primary_trading_account_number
            FROM `{TRADING_ACCOUNTS}` t JOIN mine ON t.primary_trading_account_number = mine.prim
            WHERE t.account_number IS NOT NULL"""
        job = cx.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("logins", "INT64", logins)]))
        members: dict[int, set[str]] = defaultdict(set)
        key_of: dict[str, int] = {}
        server_of_login: dict[int, str] = {}
        unmapped = demo = 0
        for r in job.result(timeout=300):
            login = int(r.account_number)
            if str(r.trading_server or "").lower().startswith("demo"):
                demo += 1
                continue                      # demo accounts are not part of a client's live book
            key = server_key(r.trading_server, login)
            if not key:
                servers = wanted.get(login, [])
                if len(servers) == 1:
                    key = servers[0]
                else:
                    unmapped += 1
                    continue
            akey = f"{key}:{login}"
            members[int(r.primary_trading_account_number)].add(akey)
            key_of[akey] = int(r.primary_trading_account_number)
            server_of_login.setdefault(login, key)
        rows = []
        for a in accounts:
            prim = key_of.get(str(a))
            if prim is None:
                continue
            mem = sorted(members[prim] | {str(a)})
            prim_key = f"{server_of_login.get(prim, str(a).rpartition(':')[0])}:{prim}"
            rows.append({"account_key": str(a), "primary_key": prim_key, "members": mem,
                         "link_source": "DWH primary_trading_account_number"})
        out = pd.DataFrame(rows).set_index("account_key") if rows else empty
        linked = int((out["members"].str.len() > 1).sum()) if len(out) else 0
        note = (f"account links: DWH primary_trading_account_number -- {len(out):,} of {len(logins):,} affected "
                f"logins found in trading_accounts, {linked:,} with linked accounts"
                + (f"; {demo} demo accounts of the same clients ignored" if demo else "")
                + (f"; {unmapped} rows on an unrecognised server label skipped" if unmapped else ""))
        return out, [note]
    except Exception as error:
        return empty, [f"account links: DWH unavailable ({type(error).__name__}: {str(error)[:120]})"]


# --------------------------------------------------------------- diagnose
def diagnose() -> str:
    """A readable check: which source address works, the warehouse's server
    labels and rebate coverage, the primary/sub-account structure."""
    lines = []
    src, ok, reason = source_ip()
    lines.append(f"transport: {reason}")
    if not ok:
        return "\n".join(lines)
    cx = client()

    def run(sql, label):
        try:
            # REST only: the Storage Read API (gRPC) does not use the bound
            # session and is refused at the edge like the public endpoint.
            df = cx.query(sql).to_dataframe(create_bqstorage_client=False)
            lines.append(f"\n--- {label}\n{df.to_string()}")
            return df
        except Exception as e:
            lines.append(f"\n--- {label} FAILED: {type(e).__name__}: {str(e)[:200]}")

    run(f"""SELECT server_name, COUNT(*) n, COUNT(DISTINCT login) logins, MIN(login) lo, MAX(login) hi,
                   COUNTIF(rebate_payout <> 0) reb_rows, ROUND(SUM(rebate_payout), 0) reb_sum,
                   COUNTIF(primary_trading_account_number IS NULL) prim_null
            FROM `{CLOSED_TRADES}` WHERE close_time_london_trading_date BETWEEN '2026-09-01' AND '2026-09-13'
            GROUP BY 1 ORDER BY 2 DESC""", "closed_trades by server_name, 1-13 Sep")
    run(f"""SELECT trading_server, COUNT(*) n, COUNTIF(primary_trading_account_number IS NULL) prim_null,
                   COUNTIF(primary_trading_account_number <> account_number) subs,
                   COUNT(DISTINCT primary_trading_account_number) primaries, MIN(account_number) lo, MAX(account_number) hi
            FROM `{TRADING_ACCOUNTS}` GROUP BY 1 ORDER BY 2 DESC""", "trading_accounts by trading_server")
    run(f"""SELECT login, server_name, COUNT(*) trades, ROUND(SUM(rebate_payout), 2) reb, ROUND(SUM(qty_usd), 0) qty_usd,
                   MIN(close_time) t0, MAX(close_time) t1, ANY_VALUE(primary_trading_account_number) prim
            FROM `{CLOSED_TRADES}`
            WHERE close_time_london_trading_date BETWEEN '2024-09-01' AND '2026-09-13'
              AND login IN (105034975, 2891280, 2969082, 1062886, 1044610, 6825721, 6834256)
            GROUP BY 1, 2 ORDER BY 1""", "known logins: trades, rebate_payout, USD qty over 2 years")
    run(f"""SELECT n, COUNT(*) primaries FROM (SELECT primary_trading_account_number, COUNT(*) n FROM `{TRADING_ACCOUNTS}`
            WHERE primary_trading_account_number IS NOT NULL GROUP BY 1) GROUP BY 1 ORDER BY 1 LIMIT 12""",
        "accounts per primary")
    return "\n".join(lines)
