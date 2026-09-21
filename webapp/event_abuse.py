"""Abuse detectors and client classification for the Event Impact tab.

Three abuse tests, each with a written reason so a reviewer can see WHY an
account is flagged:

  0. DESK FLAG -- the account is already marked by the dealing desk. On MT5
     the mark is the account comment ("Toxic", "Toxic 2/3/4", "Rebate
     Abuser"); MT4 accounts carry no such mark in MySQL (their comment field
     holds the group name), so an MT4 account can only be desk-flagged when
     its group or comment carries an abuse token.
  1. LEVERAGE / HEDGE UNWIND -- (a) effective leverage into the event: the
     notional of everything open at t0 over the balance at t0; (b) the hedge
     unwind: long and short open on the same instrument going into the
     window, one side closed inside it, the other kept -- a hedge (little
     margin) turned into a directional bet at the news price.
  2. FAVOURABLE EXECUTION -- fills inside the window BETTER than the market
     at that moment. MT5: the deal's own market_bid / market_ask (stored on
     every deal). MT4: the tick tape (mt4_live01, shared feed) in the second
     around the fill; the fill must beat the BEST price available in that
     second. Stop orders filled beyond their trigger in the client's favour
     (MT5 keeps the order type and trigger) are called out separately.

Classification (the table the business supplied): MLTV = net deposits per
active month, Monthly revenue = (net deposits - equity) per active month;
tiers VIP 40,000 / 32,000, High 10,000 / 6,000, Mid 1,000 / 400, Low 500 /
100. A client takes the higher tier of the two measures; the combined ratio
is revenue / MLTV.
"""
from __future__ import annotations

import datetime as dt
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

TIERS = [("VIP", 40_000.0, 32_000.0), ("High", 10_000.0, 6_000.0),
         ("Mid", 1_000.0, 400.0), ("Low", 500.0, 100.0)]
TIER_ORDER = {"VIP": 4, "High": 3, "Mid": 2, "Low": 1, "Micro": 0}
DESK_TOKENS = re.compile(r"toxic|abus|arbitr|scalp|fraud|blacklist|restrict|suspic|latency|desk", re.I)
FAVOURABLE_BPS = 5.0            # a fill must beat the market by this much to count (gold: ~$2.2 at 4,400)
MIN_ADVANTAGE_USD = 100.0       # ... and the account's total advantage must be worth at least this to be flagged
HIGH_LEVERAGE = 100.0           # effective leverage into the event that is flagged ...
MIN_NOTIONAL = 10_000.0         # ... on at least this much open notional (micro accounts run 300x on $200)
HEDGE_LEVERAGE = 50.0           # the leftover side of an unwound hedge must carry at least this leverage
MT5_LOTS = 10_000.0
MT4_LOTS = 100.0


# ------------------------------------------------------------------ MySQL
def _config():
    import yaml
    from webapp.mysql_extract import MYSQL_DATABASES, SERVER_CONFIG
    cfg = yaml.safe_load(open(SERVER_CONFIG))
    return cfg, MYSQL_DATABASES


def _connect(server: str):
    import pymysql
    from webapp.mysql_extract import connect_settings
    cfg, dbs = _config()
    host = cfg["servers"].get(server, {}).get("host")
    if not host or server not in dbs:
        return None
    return pymysql.connect(**connect_settings(server), connect_timeout=8, read_timeout=240,
                           charset="utf8mb4")


def _chunks(seq, n=900):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _by_server(accounts) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for a in accounts:
        server, _, login = str(a).rpartition(":")
        if server and login.isdigit():
            out[server].append(int(login))
    return out


_CONTRACT: dict[tuple, float] = {}


def contract_size(server: str, symbol: str, cur=None) -> float:
    """Contract size from the server's symbol table (cached), with a sane
    fallback (gold 100, silver 5000, 6-letter FX 100,000, else 1)."""
    key = (server, symbol)
    if key in _CONTRACT:
        return _CONTRACT[key]
    value = None
    try:
        if cur is not None:
            cur.execute("SELECT contract_size FROM symbols WHERE symbol = %s", (symbol,))
            row = cur.fetchone()
            if row and row[0]:
                value = float(row[0])
    except Exception:
        value = None
    if value is None:
        s = symbol.upper()
        crypto = s[:3] in ("BTC", "ETH", "LTC", "XRP", "BCH", "ADA", "DOT", "SOL", "DOG", "AVA", "LNK", "BNB", "TRX", "XLM")
        fx = (len(s) >= 6 and s[:6].isalpha() and s[:3] in ("EUR", "GBP", "USD", "AUD", "NZD", "CAD", "CHF", "JPY")
              and s[3:6] in ("USD", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD", "EUR", "SGD", "HKD", "MXN", "ZAR", "TRY", "CNH"))
        value = (100.0 if s.startswith("XAU") else 5000.0 if s.startswith("XAG") else 1.0 if crypto
                 else 100_000.0 if fx else 1.0)
    _CONTRACT[key] = value
    return value


def load_contract_sizes(server: str, symbols) -> None:
    """Fill the contract-size cache for these symbols from the server's own
    symbol table -- the authority; the heuristic above is only a fallback."""
    missing = [str(s) for s in set(symbols) if (server, str(s)) not in _CONTRACT]
    if not missing:
        return
    cx = None
    try:
        cx = _connect(server)
        if cx is None:
            return
        cur = cx.cursor()
        for chunk in _chunks(missing, 500):
            ph = ",".join(["%s"] * len(chunk))
            cur.execute(f"SELECT symbol, contract_size FROM symbols WHERE symbol IN ({ph})", chunk)
            for sym, cs in cur.fetchall():
                if cs is not None and float(cs) > 0:
                    _CONTRACT[(server, str(sym))] = float(cs)
    except Exception:
        pass
    finally:
        try:
            cx and cx.close()
        except Exception:
            pass


# ------------------------------------------------------------ 0. accounts
def account_state(accounts, since: dt.datetime | None = None) -> tuple[pd.DataFrame, list[str]]:
    """group, account leverage, balance, credit, floating, equity, status,
    comment and the DESK FLAG per account_key -- plus, when `since` is given,
    the P&L the platform realised after that moment (every close, not just
    the analysis frame), so the balance at `since` can be reconstructed as
    balance now - realised since - net funding since. Servers without MySQL
    (Dubai) come back empty and are named in the notes."""
    frames, notes = [], []
    e_since = int(since.replace(tzinfo=dt.timezone.utc).timestamp()) if since else None
    for server, logins in _by_server(accounts).items():
        cx = None
        try:
            cx = _connect(server)
            if cx is None:
                notes.append(f"{server}: no MySQL source -- equity, leverage and desk flag unavailable")
                continue
            cur = cx.cursor()
            mt5 = server.startswith("mt5")
            # CENT accounts keep balance / credit / floating in cents on the
            # platform; the warehouse trades are already in dollars, so the
            # account money is deflated the same way (groups.currency = CNT).
            try:
                from webapp.trade_feed import cent_logins
                cents = set(int(x) for x in cent_logins(server))
            except Exception:
                cents = set()
            rows: dict[int, dict] = {}
            for chunk in _chunks(logins):
                ph = ",".join(["%s"] * len(chunk))
                cur.execute(f"SELECT login, `group`, status, comment, leverage, balance, credit FROM accounts "
                            f"WHERE login IN ({ph})", chunk)
                for login, group, status, comment, lev, bal, credit in cur.fetchall():
                    scale = 100.0 if int(login) in cents else 1.0
                    rows[int(login)] = {"group": str(group or ""), "status": str(status or ""),
                                        "comment": str(comment or ""), "account_leverage": float(lev or 0),
                                        "balance": float(bal or 0) / scale, "credit": float(credit or 0) / scale,
                                        "floating": 0.0, "cent_account": scale > 1,
                                        "realised_after_t0": 0.0 if e_since else float("nan")}
                if mt5:
                    cur.execute(f"SELECT login, SUM(profit + storage) FROM positions WHERE login IN ({ph}) GROUP BY login", chunk)
                else:
                    cur.execute(f"SELECT login, SUM(profit + storage + commission) FROM orders "
                                f"WHERE close_ts = 0 AND cmd IN (0,1) AND login IN ({ph}) GROUP BY login", chunk)
                for login, fl in cur.fetchall():
                    if int(login) in rows:
                        rows[int(login)]["floating"] = float(fl or 0) / (100.0 if int(login) in cents else 1.0)
                if e_since:
                    if mt5:
                        cur.execute(f"SELECT login, SUM(profit + storage + commission) FROM deals "
                                    f"WHERE `time` > %s AND entry IN (1,3) AND login IN ({ph}) GROUP BY login",
                                    [since] + list(chunk))
                    else:
                        cur.execute(f"SELECT login, SUM(profit + storage + commission) FROM orders "
                                    f"WHERE close_ts > %s AND cmd IN (0,1) AND login IN ({ph}) GROUP BY login",
                                    [e_since] + list(chunk))
                    for login, pl in cur.fetchall():
                        if int(login) in rows:
                            rows[int(login)]["realised_after_t0"] = float(pl or 0) / (100.0 if int(login) in cents else 1.0)
            for login, r in rows.items():
                text = f"{r['comment']} {r['group']}"
                desk = ""
                if DESK_TOKENS.search(r["comment"] or ""):
                    desk = r["comment"].strip()
                elif DESK_TOKENS.search(r["group"] or ""):
                    desk = f"group {r['group']}"
                r["desk_flag"] = desk
                r["equity"] = r["balance"] + r["credit"] + r["floating"]
                r["account_key"] = f"{server}:{login}"
                frames.append(r)
        except Exception as error:
            notes.append(f"{server}: account state failed ({type(error).__name__}: {str(error)[:80]})")
        finally:
            try:
                cx and cx.close()
            except Exception:
                pass
    if not frames:
        return pd.DataFrame(columns=["account_key"]).set_index("account_key"), notes
    return pd.DataFrame(frames).set_index("account_key"), notes


# ------------------------------------------------------- 2. execution
# ------------------------------------------------ linked accounts (one client)
#: A client's identity across logins. The authoritative source is the desk's
#: primary / sub-account report (dropped into docs/ as account_links*.csv or
#: .xlsx, or any 'primary ... sub-account' export in Downloads); without it
#: the platform's own identity fields link accounts: the same normalised NAME
#: + COUNTRY on a server (MT4 and MT5 both), and the same e-mail on MT5 (MT4
#: e-mails are placeholders). A name shared by more than LINK_MAX_GROUP funded
#: logins is a collision (common names, test accounts), not a client.
LINK_MAX_GROUP = 60
LINK_REPORT_GLOBS = (
    str(Path(__file__).resolve().parent.parent / "docs" / "account_links*.csv"),
    str(Path(__file__).resolve().parent.parent / "docs" / "account_links*.xlsx"),
    str(Path.home() / "Downloads" / "*[Pp]rimary*[Ss]ub*"),
    str(Path.home() / "Downloads" / "*[Ss]ub*[Aa]ccount*"),
    str(Path.home() / "Downloads" / "*[Ll]inked*[Aa]ccount*"),
)
_NAME_BLOCK = re.compile(r"^(test|demo|copy|trade|ib|admin|n/?a|none|null|-+)?$|test_|_test|^\W*$", re.I)
_EMAIL_RX = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.I)


def _norm_name(value) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().upper()
    return re.sub(r"[.,;:'\"`´]+", "", text)


_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
_HIGH = re.compile(r"[\x80-\xff]")


def _name_ok(name: str) -> bool:
    """A name specific enough to be an identity key. Two-character Chinese
    names (thousands of people each) are refused: 3+ CJK characters, or 6+
    bytes of a mojibake'd (GBK-in-Latin-1, as MT4 stores it) name, or a
    Latin name of two or more words and 6+ characters."""
    if not name or _NAME_BLOCK.search(name):
        return False
    if _CJK.search(name):
        return len(_CJK.findall(name)) >= 3
    if len(_HIGH.findall(name)) >= len(name) / 2:
        return len(name) >= 6
    return len(name) >= 6 and len(name.split()) >= 2


def _load_link_report() -> tuple[dict[int, int], str]:
    """login -> primary login from the desk's report, if one is on disk.
    Accepts any sheet with a 'primary' column and a 'sub'/'login' column."""
    import glob
    paths: list[str] = []
    for pattern in LINK_REPORT_GLOBS:
        paths += glob.glob(pattern)
    for path in sorted(paths, key=lambda p: Path(p).stat().st_mtime, reverse=True):
        try:
            frame = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
        except Exception:
            continue
        cols = {c: str(c).strip().lower() for c in frame.columns}
        primary = next((c for c, n in cols.items() if "primary" in n or n in ("parent", "master", "main")), None)
        sub = next((c for c, n in cols.items() if c != primary and ("sub" in n or "login" in n or "account" in n)), None)
        if primary is None or sub is None:
            continue
        out: dict[int, int] = {}
        for p, s in zip(pd.to_numeric(frame[primary], errors="coerce"), pd.to_numeric(frame[sub], errors="coerce")):
            if pd.notna(p) and pd.notna(s):
                out[int(s)] = int(p); out.setdefault(int(p), int(p))
        if out:
            return out, Path(path).name
    return {}, ""


def linked_accounts(accounts) -> tuple[pd.DataFrame, list[str]]:
    """One row per input account_key: client_key (the group id), link_source,
    members (every account_key of that client, on every server, impacted or
    not). Unlinked accounts are their own client."""
    notes: list[str] = []
    accounts = [str(a) for a in accounts]
    parent: dict[str, str] = {}
    nodes: set[str] = set(accounts)          # every key ever linked (roots included)

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x]); x = parent[x]
        return x

    def union(a, b):
        nodes.add(a); nodes.add(b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    source = pd.Series("", index=pd.Index(accounts, name="account_key"), dtype="object")
    report_primary: dict[str, str] = {}
    by_srv = _by_server(accounts)
    # ---- 0. the warehouse: primary_trading_account_number per trading account
    #      (data_marts.trading_accounts) -- the client register itself.
    dwh_linked = 0
    try:
        from webapp import dwh
        links_dwh, n_ = dwh.account_links(accounts)
        notes += n_
        for a, r in links_dwh.iterrows():
            mem = list(r["members"]) if isinstance(r["members"], (list, tuple, set)) else [a]
            for k in mem:
                union(str(a), str(k))
            source[a] = "DWH primary_trading_account_number"
            if r.get("primary_key"):
                report_primary[str(a)] = str(r["primary_key"])
                for k in mem:
                    report_primary.setdefault(str(k), str(r["primary_key"]))
            dwh_linked += 1
    except Exception as error:
        notes.append(f"account links: DWH unavailable ({type(error).__name__}: {str(error)[:100]})")
    covered = {a for a in accounts if source[a]}
    # ---- 1. the desk's report (for accounts the warehouse did not cover)
    report, report_name = _load_link_report()
    login_server: dict[int, str] = {}
    if report:
        wanted = {l for s, ls in by_srv.items() for l in ls}
        groups: dict[int, set[int]] = defaultdict(set)
        for login, prim in report.items():
            groups[prim].add(login)
        touched = {prim for prim, members in groups.items() if members & wanted or prim in wanted}
        # a bare login in the report has no server: resolve it on every server
        need = {l for prim in touched for l in groups[prim] | {prim}}
        for server in (_config()[1] or {}):
            cx = None
            try:
                cx = _connect(server)
                if cx is None:
                    continue
                cur = cx.cursor()
                for chunk in _chunks(sorted(need)):
                    ph = ",".join(["%s"] * len(chunk))
                    cur.execute(f"SELECT login FROM accounts WHERE login IN ({ph})", chunk)
                    for (login,) in cur.fetchall():
                        login_server.setdefault(int(login), server)
            except Exception:
                pass
            finally:
                try:
                    cx and cx.close()
                except Exception:
                    pass
        n_links = 0
        for prim in touched:
            keys = [f"{login_server[l]}:{l}" for l in groups[prim] | {prim} if l in login_server]
            prim_key = f"{login_server[prim]}:{prim}" if prim in login_server else ""
            for k in keys[1:]:
                union(keys[0], k); n_links += 1
            for k in keys:
                if k in source.index:
                    source[k] = f"report ({report_name})"
                if prim_key:
                    report_primary[k] = prim_key
        notes.append(f"account links: desk report '{report_name}' ({len(report):,} logins) linked {n_links} accounts")
    # ---- 2. platform identity: name + country per server, e-mail on MT5
    ident: dict[str, dict] = {}
    # only the accounts the warehouse / report did not place: a name key must
    # never merge a registered client with a namesake
    uncovered = _by_server([a for a in accounts if not source[a]])
    for server, logins in uncovered.items():
        cx = None
        try:
            cx = _connect(server)
            if cx is None:
                notes.append(f"{server}: no MySQL source -- accounts cannot be linked")
                continue
            cur = cx.cursor()
            for chunk in _chunks(logins):
                ph = ",".join(["%s"] * len(chunk))
                cur.execute(f"SELECT login, name, country, email FROM accounts WHERE login IN ({ph})", chunk)
                for login, name, country, email in cur.fetchall():
                    ident[f"{server}:{login}"] = {"name": _norm_name(name), "country": _norm_name(country),
                                                  "email": str(email or "").strip().lower()}
            names = sorted({v["name"] for v in ident.values() if _name_ok(v["name"])})
            emails = sorted({v["email"] for v in ident.values() if _EMAIL_RX.match(v["email"]) and v["email"] not in ("email",)}) \
                if server.startswith("mt5") else []
            # every login on THIS server sharing one of those keys (funded or traded at some point)
            name_groups: dict[tuple, list[str]] = defaultdict(list)
            for chunk in _chunks(names, 400):
                ph = ",".join(["%s"] * len(chunk))
                cur.execute(f"SELECT login, name, country FROM accounts WHERE UPPER(TRIM(name)) IN ({ph})", chunk)
                for login, name, country in cur.fetchall():
                    name_groups[(_norm_name(name), _norm_name(country))].append(f"{server}:{login}")
            email_groups: dict[str, list[str]] = defaultdict(list)
            for chunk in _chunks(emails, 400):
                ph = ",".join(["%s"] * len(chunk))
                cur.execute(f"SELECT login, email FROM accounts WHERE LOWER(TRIM(email)) IN ({ph})", chunk)
                for login, email in cur.fetchall():
                    email_groups[str(email or "").strip().lower()].append(f"{server}:{login}")
            collisions = 0
            for key, members in name_groups.items():
                members = sorted(set(members) - covered)
                if len(members) < 2 or not _name_ok(key[0]):
                    continue
                if len(members) > LINK_MAX_GROUP:
                    collisions += 1
                    continue
                for k in members[1:]:
                    union(members[0], k)
                for k in members:
                    if k in source.index and not source[k]:
                        source[k] = "name + country"
            for key, members in email_groups.items():
                members = sorted(set(members) - covered)
                if len(members) < 2 or len(members) > LINK_MAX_GROUP:
                    continue
                for k in members[1:]:
                    union(members[0], k)
                for k in members:
                    if k in source.index and not source[k]:
                        source[k] = "e-mail"
            if collisions:
                notes.append(f"{server}: {collisions} name+country keys shared by more than {LINK_MAX_GROUP} logins "
                             f"were treated as collisions, not links")
        except Exception as error:
            notes.append(f"{server}: account linking failed ({type(error).__name__}: {str(error)[:80]})")
        finally:
            try:
                cx and cx.close()
            except Exception:
                pass
    # ---- 3. resolve
    members: dict[str, set[str]] = defaultdict(set)
    for k in nodes | set(parent.keys()):
        members[find(k)].add(k)
    rows = []
    for a in accounts:
        root = find(a)
        mem = sorted(members[root] | {a})
        rows.append({"account_key": a, "client_key": root, "members": mem,
                     "link_source": source[a] if len(mem) > 1 else "",
                     "report_primary": report_primary.get(a, "")})
    out = pd.DataFrame(rows).set_index("account_key")
    linked = int((out["members"].str.len() > 1).sum())
    n_clients = out["client_key"].nunique()
    if not report:
        notes.append("account links: no primary/sub-account report found (docs/account_links*.csv|xlsx) -- "
                     "accounts linked by the platform's identity fields (same name + country per server, same "
                     "e-mail on MT5)")
    notes.append(f"account links: {len(accounts):,} impacted accounts -> {n_clients:,} clients; "
                 f"{linked:,} accounts have at least one linked account")
    return out, notes


# ------------------------------------------ lifetime trading (USD notional)
#: Approximate USD value of one unit of each currency, for the notional of FX
#: crosses and non-USD-quoted CFDs (a lifetime figure; intraday rates are not
#: what the number is for).
USD_PER = {"USD": 1.0, "EUR": 1.17, "GBP": 1.35, "AUD": 0.66, "NZD": 0.60, "CAD": 0.73, "CHF": 1.25,
           "JPY": 0.0068, "SGD": 0.78, "HKD": 0.128, "CNH": 0.14, "CNY": 0.14, "MXN": 0.054, "ZAR": 0.057,
           "TRY": 0.024, "NOK": 0.10, "SEK": 0.105, "DKK": 0.157, "PLN": 0.275, "HUF": 0.0029, "CZK": 0.048}
_CCY = set(USD_PER)
_METALS = ("XAU", "XAG", "XPT", "XPD")


def usd_notional(symbol: str, lots: float, lots_x_price: float, csize: float) -> float:
    """USD notional of `lots` traded on `symbol` given sum(lots x price) and
    the contract size. FX: the contract is in the BASE currency -> lots x
    contract x USD per base unit. Everything else is contract units x price
    in the quote currency -> converted when that is not USD."""
    s = re.sub(r"[^A-Z]", "", str(symbol).upper())
    base, quote = s[:3], s[3:6]
    if len(s) >= 6 and base in _CCY and quote in _CCY:
        return float(lots) * csize * USD_PER[base]
    factor = 1.0
    if len(s) >= 6 and quote in _CCY and quote != "USD" and (base in _METALS or base in _CCY):
        factor = USD_PER[quote]
    return float(lots_x_price) * csize * factor


LIFETIME_CACHE = Path(__file__).resolve().parent / "artifacts" / "lifetime_trading.parquet"
LIFETIME_TTL_HOURS = 24.0


def lifetime_trading(accounts) -> tuple[pd.DataFrame, list[str]]:
    """Per account_key, over the platform's whole history: USD notional
    traded (entries only -- one side of each round trip), lots, trades,
    first and last trade. Cent accounts deflated. A whole-history GROUP BY
    per login is the slow part of the event analysis, so answers are kept
    on disk for LIFETIME_TTL_HOURS and only missing / stale accounts are
    queried."""
    notes: list[str] = []
    accounts = [str(a) for a in accounts]
    cached = pd.DataFrame()
    try:
        if LIFETIME_CACHE.exists():
            cached = pd.read_parquet(LIFETIME_CACHE)
            fresh = pd.to_datetime(cached["computed_at"]) >= pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=LIFETIME_TTL_HOURS)
            cached = cached[fresh].set_index("account_key")
    except Exception:
        cached = pd.DataFrame()
    hit = cached.reindex([a for a in accounts if a in cached.index]) if len(cached) else pd.DataFrame()
    todo = [a for a in accounts if a not in getattr(hit, "index", [])]
    fresh_rows, n_ = _lifetime_trading_query(todo) if todo else (pd.DataFrame(), [])
    notes += n_
    if len(fresh_rows):
        fresh_rows = fresh_rows.copy()
        fresh_rows["computed_at"] = pd.Timestamp.utcnow().tz_localize(None)
        try:
            keep = cached.drop(index=[a for a in fresh_rows.index if a in cached.index], errors="ignore") if len(cached) else pd.DataFrame()
            store = pd.concat([keep, fresh_rows]) if len(keep) else fresh_rows
            LIFETIME_CACHE.parent.mkdir(parents=True, exist_ok=True)
            store.reset_index().to_parquet(LIFETIME_CACHE, index=False)
        except Exception as error:
            notes.append(f"lifetime cache not written: {type(error).__name__}: {str(error)[:60]}")
    if len(hit):
        notes.append(f"lifetime trading: {len(hit):,} accounts from the {LIFETIME_TTL_HOURS:.0f}h cache, {len(todo):,} queried")
    parts = [p.drop(columns=["computed_at"], errors="ignore") for p in (hit, fresh_rows) if len(p)]
    if not parts:
        return pd.DataFrame(columns=["account_key"]).set_index("account_key"), notes
    out = pd.concat(parts)
    out = out[~out.index.duplicated(keep="last")]
    out.index.name = "account_key"
    return out, notes


def _lifetime_trading_query(accounts) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    rows: dict[str, dict] = {}
    for server, logins in _by_server(accounts).items():
        cx = None
        try:
            cx = _connect(server)
            if cx is None:
                notes.append(f"{server}: no MySQL source -- lifetime notional unavailable")
                continue
            cur = cx.cursor()
            try:
                from webapp.trade_feed import cent_logins
                cents = set(int(x) for x in cent_logins(server))
            except Exception:
                cents = set()
            mt5 = server.startswith("mt5")
            symbols_seen: set[str] = set()
            raw: list[tuple] = []
            for chunk in _chunks(logins):
                ph = ",".join(["%s"] * len(chunk))
                if mt5:
                    cur.execute(f"SELECT login, symbol, SUM(volume) / {MT5_LOTS}, SUM(volume * price) / {MT5_LOTS}, COUNT(*), "
                                f"MIN(`time`), MAX(`time`) FROM deals WHERE action IN (0,1) AND entry IN (0,2) "
                                f"AND login IN ({ph}) GROUP BY login, symbol", chunk)
                else:
                    cur.execute(f"SELECT login, symbol_name, SUM(volume) / {MT4_LOTS}, SUM(volume * open_price) / {MT4_LOTS}, COUNT(*), "
                                f"MIN(open_ts), MAX(GREATEST(open_ts, close_ts)) FROM orders WHERE cmd IN (0,1) "
                                f"AND login IN ({ph}) GROUP BY login, symbol_name", chunk)
                for r in cur.fetchall():
                    raw.append(r); symbols_seen.add(str(r[1]))
            load_contract_sizes(server, symbols_seen)
            for login, symbol, lots, lxp, n, first, last in raw:
                scale = 100.0 if int(login) in cents else 1.0
                lots = float(lots or 0) / scale; lxp = float(lxp or 0) / scale
                key = f"{server}:{login}"
                rec = rows.setdefault(key, {"notional_usd": 0.0, "lots_lifetime": 0.0, "trades_lifetime": 0,
                                            "first_trade": None, "last_trade": None})
                rec["notional_usd"] += usd_notional(str(symbol), lots, lxp, contract_size(server, str(symbol), cur))
                rec["lots_lifetime"] += lots; rec["trades_lifetime"] += int(n or 0)
                f = (pd.Timestamp(int(first), unit="s") if (not mt5 and first) else (pd.Timestamp(first) if (mt5 and first) else None))
                l = (pd.Timestamp(int(last), unit="s") if (not mt5 and last) else (pd.Timestamp(last) if (mt5 and last) else None))
                if f is not None and (rec["first_trade"] is None or f < rec["first_trade"]):
                    rec["first_trade"] = f
                if l is not None and (rec["last_trade"] is None or l > rec["last_trade"]):
                    rec["last_trade"] = l
        except Exception as error:
            notes.append(f"{server}: lifetime trading failed ({type(error).__name__}: {str(error)[:80]})")
        finally:
            try:
                cx and cx.close()
            except Exception:
                pass
    if not rows:
        return pd.DataFrame(columns=["account_key"]).set_index("account_key"), notes
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "account_key"
    out["first_trade"] = pd.to_datetime(out["first_trade"]); out["last_trade"] = pd.to_datetime(out["last_trade"])
    return out, notes


# ------------------------------------------------------------ rebates
#: In-account rebate credits: the partner-programme payouts ('PPF-<ib>-<login>'
#: on both platforms), IB payments and anything the back office labels a
#: rebate / cashback. Positive amounts only (the payout side).
REBATE_RX = re.compile(r"^PPF-|^IB-|^AGENT\b|REBATE|CASH ?BACK|KICKBACK|COMM(?:ISSION)? ?REB", re.I)


def rebate_mask(comment: pd.Series, amount: pd.Series) -> pd.Series:
    text = comment.astype(str).fillna("")
    return text.str.contains(REBATE_RX, regex=True) & (pd.to_numeric(amount, errors="coerce").fillna(0.0) > 0)


#: The rebate the firm actually pays per trade (`rebate_payout` in Metabase,
#: computed in the data warehouse from the IB agreements) is not stored on
#: the MT4/MT5 servers: the platforms' EOD table carries an empty rebate
#: column since June 2026, ib_rebates_summary stopped in October 2020 and MT5
#: deals have no rebate field. Until BigQuery is reachable from this network
#: the source is a Metabase export dropped on disk: any CSV / XLSX under
#: docs/ or Downloads whose header carries `rebate_payout` plus a login /
#: trading-account column (a close-time column is used when present).
REBATE_EXPORT_GLOBS = (
    str(Path(__file__).resolve().parent.parent / "docs" / "*.csv"),
    str(Path(__file__).resolve().parent.parent / "docs" / "*.xlsx"),
    str(Path.home() / "Downloads" / "*rebate*.csv"),
    str(Path.home() / "Downloads" / "*rebate*.xlsx"),
    str(Path.home() / "Downloads" / "query_result*.csv"),
)
_REBATE_EXPORT: dict = {"path": None, "mtime": None, "frame": None}


def _load_rebate_export() -> tuple[pd.DataFrame | None, str]:
    import glob, os
    paths: list[str] = []
    for pattern in REBATE_EXPORT_GLOBS:
        paths += glob.glob(pattern)
    for path in sorted(set(paths), key=os.path.getmtime, reverse=True):
        try:
            head = (pd.read_excel(path, nrows=0) if path.lower().endswith((".xlsx", ".xls"))
                    else pd.read_csv(path, nrows=0, encoding="utf-8-sig"))
        except Exception:
            continue
        cols = {str(c).strip().lower(): c for c in head.columns}
        amount = next((c for n, c in cols.items() if "rebate_payout" in n or n in ("rebate", "rebate_usd", "rebates")), None)
        login = next((c for n, c in cols.items() if n in ("login", "trading_account_number", "account", "account_number", "mt_login")
                      or ("trading_account" in n and "primary" not in n)), None)
        if amount is None or login is None:
            continue
        mtime = os.path.getmtime(path)
        if _REBATE_EXPORT["frame"] is not None and _REBATE_EXPORT["path"] == path and _REBATE_EXPORT["mtime"] == mtime:
            return _REBATE_EXPORT["frame"], os.path.basename(path)
        frame = (pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls"))
                 else pd.read_csv(path, encoding="utf-8-sig", low_memory=False))
        when = next((c for c in frame.columns if re.search(r"close|date|time|day", str(c), re.I)), None)
        out = pd.DataFrame({
            "login": pd.to_numeric(frame[login].astype(str).str.replace(",", ""), errors="coerce"),
            "rebate": pd.to_numeric(frame[amount].astype(str).str.replace(",", ""), errors="coerce").fillna(0.0),
            "when": pd.to_datetime(frame[when], errors="coerce") if when is not None else pd.NaT,
        }).dropna(subset=["login"])
        out["login"] = out["login"].astype(int)
        _REBATE_EXPORT.update(path=path, mtime=mtime, frame=out)
        return out, os.path.basename(path)
    return None, ""


def metabase_rebates(accounts) -> tuple[pd.DataFrame, list[str]]:
    """Per account_key: rebates paid (sum of rebate_payout), records, first
    and last rebate date, and the source file -- from the Metabase / DWH
    export on disk. Empty (with a note) when no export is present."""
    empty = pd.DataFrame(columns=["rebates_usd", "rebate_records", "rebate_first", "rebate_last", "rebate_source"])
    empty.index.name = "account_key"
    frame, name = _load_rebate_export()
    if frame is None or frame.empty:
        return empty, ["rebates: no Metabase export (rebate_payout per trade) found under docs/ or Downloads -- "
                       "rebate figures fall back to in-account credits (PPF/IB payouts), which understate the "
                       "rebates the DWH computes"]
    by_login = _by_server(accounts)
    wanted = {login for logins in by_login.values() for login in logins}
    hit = frame[frame["login"].isin(wanted)]
    if hit.empty:
        return empty, [f"rebates: export '{name}' has no rows for the affected logins"]
    g = hit.groupby("login")
    summary = pd.DataFrame({"rebates_usd": g["rebate"].sum(), "rebate_records": g.size(),
                            "rebate_first": g["when"].min(), "rebate_last": g["when"].max()})
    servers_per_login: dict[int, list[str]] = defaultdict(list)
    for server, logins in by_login.items():
        for login in logins:
            servers_per_login[login].append(server)
    rows = []
    for login, r in summary.iterrows():
        for server in servers_per_login.get(int(login), []):
            rows.append({"account_key": f"{server}:{login}", "rebates_usd": float(r["rebates_usd"]),
                         "rebate_records": int(r["rebate_records"]), "rebate_first": r["rebate_first"],
                         "rebate_last": r["rebate_last"], "rebate_source": f"Metabase rebate_payout ({name})"})
    out = pd.DataFrame(rows).set_index("account_key") if rows else empty
    return out, [f"rebates: Metabase export '{name}' -- {len(frame):,} trade rows, {len(out):,} affected accounts matched"]


def rebate_payouts_any(accounts, days: int = 730) -> tuple[pd.DataFrame, list[str]]:
    """The rebate the firm paid per account, from the best source available:
    the warehouse's per-trade rebate_payout (data_marts.closed_trades) first,
    a Metabase export on disk for whatever it did not cover, else nothing
    (the caller falls back to in-account credits)."""
    notes: list[str] = []
    parts = []
    covered: set[str] = set()
    try:
        from webapp import dwh
        got, n_ = dwh.rebate_payouts(accounts, days=days)
        notes += n_
        if len(got):
            parts.append(got); covered = set(got.index)
    except Exception as error:
        notes.append(f"rebates: DWH unavailable ({type(error).__name__}: {str(error)[:100]})")
    rest = [a for a in accounts if a not in covered]
    if rest:
        got, n_ = metabase_rebates(rest)
        if len(got):
            parts.append(got); notes += n_
        elif not parts:
            notes += n_
    if not parts:
        empty = pd.DataFrame(columns=["rebates_usd", "rebate_records", "rebate_first", "rebate_last", "rebate_source"])
        empty.index.name = "account_key"
        return empty, notes
    out = pd.concat(parts)
    return out[~out.index.duplicated(keep="first")], notes


def _mt4_ticks(symbol: str, t0: dt.datetime, t1: dt.datetime) -> pd.DataFrame:
    cx = _connect("mt4_live01")
    if cx is None:
        return pd.DataFrame(columns=["tm", "bid", "ask"])
    try:
        frame = pd.read_sql("SELECT tm, bid, ask FROM ticks WHERE symbol_name = %s AND tm >= %s AND tm < %s ORDER BY tm",
                            cx, params=(symbol, t0 - dt.timedelta(seconds=3), t1 + dt.timedelta(seconds=3)))
    finally:
        cx.close()
    frame["tm"] = pd.to_datetime(frame["tm"])
    frame["bid"] = pd.to_numeric(frame["bid"], errors="coerce"); frame["ask"] = pd.to_numeric(frame["ask"], errors="coerce")
    return frame.dropna()


def _best_in_second(ticks: pd.DataFrame, when: pd.Timestamp) -> tuple[float, float] | None:
    """(best bid, best ask) available within one second either side of `when`."""
    lo, hi = when - pd.Timedelta(seconds=1), when + pd.Timedelta(seconds=1)
    i0 = ticks["tm"].searchsorted(lo, side="left"); i1 = ticks["tm"].searchsorted(hi, side="right")
    if i1 <= i0:
        return None
    seg = ticks.iloc[i0:i1]
    return float(seg["bid"].max()), float(seg["ask"].min())


def execution_check(t0: dt.datetime, t1: dt.datetime, canon_of, want: set, accounts) -> tuple[pd.DataFrame, list[str]]:
    """Per account: fills inside the window, how many beat the market, by how
    much, the money that was worth, and the examples that explain the flag."""
    per: dict[str, dict] = defaultdict(lambda: {"fills_in_window": 0, "favourable_fills": 0, "max_favourable_bps": 0.0,
                                                "advantage_usd": 0.0, "stop_gap_fills": 0, "examples": []})
    notes: list[str] = []
    e0 = int(t0.replace(tzinfo=dt.timezone.utc).timestamp()); e1 = int(t1.replace(tzinfo=dt.timezone.utc).timestamp())
    servers = _by_server(accounts)
    wanted_logins = {s: set(l) for s, l in servers.items()}
    tick_cache: dict[str, pd.DataFrame] = {}

    def note_fill(key, side_paid, price, best_bid, best_ask, lots, csize, label):
        """side_paid: +1 the client BOUGHT at price (paid), -1 the client SOLD (received)."""
        rec = per[key]; rec["fills_in_window"] += 1
        if side_paid > 0 and best_ask > 0 and price < best_ask:
            bps = (best_ask - price) / best_ask * 1e4
        elif side_paid < 0 and best_bid > 0 and price > best_bid:
            bps = (price - best_bid) / best_bid * 1e4
        else:
            return
        if bps < FAVOURABLE_BPS:
            return
        usd = bps / 1e4 * price * lots * csize
        rec["favourable_fills"] += 1; rec["advantage_usd"] += usd
        rec["max_favourable_bps"] = max(rec["max_favourable_bps"], bps)
        if len(rec["examples"]) < 3:
            rec["examples"].append(f"{label} at {price:g} vs market {'ask ' + format(best_ask, 'g') if side_paid > 0 else 'bid ' + format(best_bid, 'g')} ({bps:.1f} bps, ${usd:,.0f})")

    for server, logins in servers.items():
        cx = None
        try:
            cx = _connect(server)
            if cx is None:
                notes.append(f"{server}: no MySQL source -- execution check skipped")
                continue
            cur = cx.cursor()
            if server.startswith("mt5"):
                cur.execute("SELECT deal, login, symbol, action, entry, reason, price, market_bid, market_ask, volume, contract_size, `order`, `time` "
                            "FROM deals WHERE `time` >= %s AND `time` < %s AND action IN (0,1)", (t0, t1))
                deals = cur.fetchall()
                stops: dict[int, tuple] = {}
                cur.execute("SELECT `order`, type, price_order FROM orders WHERE type IN (4,5) AND state = 4 AND time_done >= %s AND time_done < %s", (t0, t1))
                for order, otype, trigger in cur.fetchall():
                    stops[int(order)] = (int(otype), float(trigger or 0))
                for deal, login, symbol, action, entry, reason, price, mbid, mask, vol, csize, order, when in deals:
                    if int(login) not in logins or (want and canon_of(symbol).upper() not in want):
                        continue
                    if int(entry) == 3:
                        continue        # OUT_BY: two positions closed against each other at the counterpart's price, not a market fill
                    key = f"{server}:{login}"; lots = float(vol or 0) / MT5_LOTS
                    side_paid = 1 if int(action) == 0 else -1
                    label = f"{'buy' if side_paid > 0 else 'sell'} {'open' if int(entry) == 0 else 'close'} {lots:g} lots {symbol} {str(when)[11:19]}"
                    # the deal's own snapshot AND the shared tick tape in the
                    # second around the fill: a fill only counts as better than
                    # the market when it beats the best of both.
                    best_bid, best_ask = float(mbid or 0), float(mask or 0)
                    canon = canon_of(symbol).upper()
                    ticks = tick_cache.get(canon)
                    if ticks is None:
                        ticks = tick_cache[canon] = _mt4_ticks(canon, t0, t1)
                    if not ticks.empty:
                        tape = _best_in_second(ticks, pd.Timestamp(when))
                        if tape:
                            best_bid = max(best_bid, tape[0]) if best_bid > 0 else tape[0]
                            best_ask = min(best_ask, tape[1]) if best_ask > 0 else tape[1]
                    note_fill(key, side_paid, float(price), best_bid, best_ask, lots, float(csize or 1), label)
                    st = stops.get(int(order or 0))
                    if st and st[1] > 0:
                        otype, trigger = st
                        gap = (trigger - float(price)) if otype == 4 else (float(price) - trigger)   # favourable = beyond the trigger
                        if gap > 0 and gap / trigger * 1e4 >= FAVOURABLE_BPS:
                            rec = per[key]; rec["stop_gap_fills"] += 1
                            if len(rec["examples"]) < 3:
                                rec["examples"].append(f"{'buy' if otype == 4 else 'sell'} stop {trigger:g} filled at {price:g} "
                                                       f"({gap / trigger * 1e4:.1f} bps beyond the trigger in the client's favour)")
            else:
                cur.execute("SELECT `order`, login, symbol_name, cmd, volume, open_price, close_price, open_ts, close_ts, comment "
                            "FROM orders WHERE cmd IN (0,1) AND ((open_ts BETWEEN %s AND %s) OR (close_ts BETWEEN %s AND %s))",
                            (e0, e1, e0, e1))
                mt4_rows = cur.fetchall()
                # CLOSE-BY pairs: "close hedge by #N" closes N and itself at N's
                # counterpart price -- not market fills. Skip both legs.
                close_by = set()
                for order, login, symbol, cmd, vol, op, cp, ots, cts, comment in mt4_rows:
                    text = str(comment or "")
                    m = re.search(r"close hedge by #(\d+)", text)
                    if m:
                        close_by.add(int(order)); close_by.add(int(m.group(1)))
                for order, login, symbol, cmd, vol, op, cp, ots, cts, comment in mt4_rows:
                    if int(login) not in logins or int(order) in close_by:
                        continue
                    canon = canon_of(symbol).upper()
                    if want and canon not in want:
                        continue
                    key = f"{server}:{login}"; lots = float(vol or 0) / MT4_LOTS
                    ticks = tick_cache.get(canon)
                    if ticks is None:
                        ticks = tick_cache[canon] = _mt4_ticks(canon, t0, t1)
                    if ticks.empty:
                        continue
                    csize = contract_size(server, str(symbol), cur)
                    buy = int(cmd) == 0
                    if e0 <= int(ots or 0) <= e1:
                        best = _best_in_second(ticks, pd.Timestamp(int(ots), unit="s"))
                        if best:
                            note_fill(key, 1 if buy else -1, float(op), best[0], best[1], lots, csize,
                                      f"{'buy' if buy else 'sell'} open {lots:g} lots {symbol} {dt.datetime.utcfromtimestamp(int(ots)):%H:%M:%S}")
                    if e0 <= int(cts or 0) <= e1 and float(cp or 0) > 0:
                        best = _best_in_second(ticks, pd.Timestamp(int(cts), unit="s"))
                        if best:
                            tag = str(comment or "").strip()
                            kind = "stop-loss close" if tag.startswith("[sl]") else "take-profit close" if tag.startswith("[tp]") else "stop-out close" if tag.startswith("so") else "close"
                            note_fill(key, -1 if buy else 1, float(cp), best[0], best[1], lots, csize,
                                      f"{'buy' if buy else 'sell'} {kind} {lots:g} lots {symbol} {dt.datetime.utcfromtimestamp(int(cts)):%H:%M:%S}")
        except Exception as error:
            notes.append(f"{server}: execution check failed ({type(error).__name__}: {str(error)[:80]})")
        finally:
            try:
                cx and cx.close()
            except Exception:
                pass
    if not per:
        return pd.DataFrame(columns=["account_key"]).set_index("account_key"), notes
    out = pd.DataFrame.from_dict(per, orient="index")
    out.index.name = "account_key"
    out["execution_detail"] = out["examples"].apply(lambda xs: "; ".join(xs))
    return out.drop(columns=["examples"]), notes


# ------------------------------------------------- close-by legs
def close_by_orders(t0: dt.datetime, t1: dt.datetime, accounts) -> tuple[set, list[str]]:
    """(server, order) of every CLOSE-BY leg closed inside the window. A
    close-by nets a long against a short at the counterpart's price: the
    P&L is real (the locked spread) but neither leg is a market fill and the
    net exposure does not change. MT4: 'close hedge by #N' on one leg, N is
    the other; MT5: the OUT_BY deal (entry 3) on each position."""
    out: set = set(); notes: list[str] = []
    e0 = int(t0.replace(tzinfo=dt.timezone.utc).timestamp()); e1 = int(t1.replace(tzinfo=dt.timezone.utc).timestamp())
    for server, logins in _by_server(accounts).items():
        cx = None
        try:
            cx = _connect(server)
            if cx is None:
                continue
            cur = cx.cursor()
            if server.startswith("mt5"):
                cur.execute("SELECT DISTINCT position_id FROM deals WHERE entry = 3 AND `time` >= %s AND `time` < %s", (t0, t1))
                for (pid,) in cur.fetchall():
                    out.add((server, int(pid or 0)))
            else:
                cur.execute("SELECT `order`, comment FROM orders WHERE close_ts BETWEEN %s AND %s "
                            "AND comment LIKE 'close hedge by #%%'", (e0, e1))
                for order, comment in cur.fetchall():
                    out.add((server, int(order)))
                    m = re.search(r"close hedge by #(\d+)", str(comment or ""))
                    if m:
                        out.add((server, int(m.group(1))))
        except Exception as error:
            notes.append(f"{server}: close-by lookup failed ({type(error).__name__}: {str(error)[:80]})")
        finally:
            try:
                cx and cx.close()
            except Exception:
                pass
    return out, notes


# ------------------------------------------------- 1. leverage & hedges
def hedge_leverage_check(frame: pd.DataFrame, t0, t1, sym_mask: pd.Series, accounts,
                         balance_now: pd.Series, net_flow_after: pd.Series,
                         realised_after: pd.Series | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Effective leverage into the event and the hedge-unwind pattern, from
    the warehouse trades of the impacted accounts. The unwind is measured on
    NET exposure of the positions carried into the window: a close-by nets
    both legs and cannot register as one."""
    notes: list[str] = []
    acc = set(str(a) for a in accounts)
    f = frame[frame["account_key"].isin(acc)].copy()
    f["dir"] = np.where(f["cmd"].astype(str).str.lower().eq("buy"), 1, -1)
    f["lots"] = pd.to_numeric(f["volume_lots"], errors="coerce").fillna(0.0)
    f["op"] = pd.to_numeric(f["open_price"], errors="coerce").fillna(0.0)
    # contract sizes from each server's symbol table (BTC is 1, oil is not FX)
    for server in f["database"].astype(str).unique():
        load_contract_sizes(server, f.loc[f["database"].astype(str) == server, "symbol"].astype(str).unique())
    csize = np.array([contract_size(str(s), str(y)) for s, y in zip(f["database"], f["symbol"])])
    f["notional"] = f["lots"] * csize * f["op"]
    open_t0 = (f["open_time"] < t0) & (f["close_time"] >= t0)
    open_t1 = (f["open_time"] < t1) & (f["close_time"] > t1)
    closed_in = (f["close_time"] >= t0) & (f["close_time"] <= t1)
    idx = pd.Index(sorted(acc), name="account_key")
    out = pd.DataFrame(index=idx)
    g0 = f[open_t0].groupby("account_key")
    # GROSS notional (both legs of a hedge) for the record; the leverage that
    # is flagged uses NET exposure per instrument (|long - short|) -- a hedged
    # grid carries little directional risk until it is unwound, and the
    # unwind is its own test below.
    out["gross_notional_at_t0"] = g0["notional"].sum().reindex(idx).fillna(0.0)
    out["positions_at_t0"] = g0.size().reindex(idx).fillna(0).astype(int)
    from webapp.trade_feed import _canonical
    o0 = f[open_t0].copy()
    o0["canon"] = [(_canonical(str(s)) or str(s)).upper() for s in o0["symbol"]]
    o0["signed"] = o0["dir"] * o0["notional"]
    out["notional_at_t0"] = (o0.groupby(["account_key", "canon"])["signed"].sum().abs()
                             .groupby("account_key").sum().reindex(idx).fillna(0.0))
    # balance at t0 ~ balance now - P&L realised after t0 - net funding after
    # t0. The platform's own realised figure (every close since t0) is used
    # where available; the analysis frame (5 days) is the fallback.
    frame_after = f[f["close_time"] > t0].groupby("account_key")["net_profit"].sum().reindex(idx).fillna(0.0)
    if realised_after is not None:
        platform_after = pd.to_numeric(realised_after.reindex(idx), errors="coerce")
        after = platform_after.where(platform_after.notna(), frame_after)
    else:
        after = frame_after
    bal_now = pd.to_numeric(balance_now.reindex(idx), errors="coerce")
    flow = pd.to_numeric(net_flow_after.reindex(idx), errors="coerce").fillna(0.0)
    out["balance_at_t0_est"] = (bal_now - after - flow).round(2)
    denom = out["balance_at_t0_est"].where(out["balance_at_t0_est"] > 1.0)
    out["effective_leverage_t0"] = (out["notional_at_t0"] / denom).round(1)
    out["gross_leverage_t0"] = (out["gross_notional_at_t0"] / denom).round(1)
    # hedge unwind on the EVENT symbols only
    e = f[sym_mask.reindex(f.index).fillna(False).astype(bool)] if sym_mask is not None else f
    e_t0 = e[open_t0.reindex(e.index).fillna(False).astype(bool)]
    long0 = e_t0[e_t0["dir"] > 0].groupby("account_key")["lots"].sum().reindex(idx).fillna(0.0)
    short0 = e_t0[e_t0["dir"] < 0].groupby("account_key")["lots"].sum().reindex(idx).fillna(0.0)
    e_in = e[closed_in.reindex(e.index).fillna(False).astype(bool)]
    long_closed = e_in[e_in["dir"] > 0].groupby("account_key")["lots"].sum().reindex(idx).fillna(0.0)
    short_closed = e_in[e_in["dir"] < 0].groupby("account_key")["lots"].sum().reindex(idx).fillna(0.0)
    # what the window LEFT of the positions carried into it (not new opens)
    carried = e[(open_t0 & open_t1).reindex(e.index).fillna(False).astype(bool)]
    long1 = carried[carried["dir"] > 0].groupby("account_key")["lots"].sum().reindex(idx).fillna(0.0)
    short1 = carried[carried["dir"] < 0].groupby("account_key")["lots"].sum().reindex(idx).fillna(0.0)
    notional1 = carried.groupby("account_key")["notional"].sum().reindex(idx).fillna(0.0)
    # close-by legs closed in the window: netted, not traded (marked upstream)
    cb = e_in[e_in["close_by"]] if "close_by" in e_in.columns else e_in.iloc[0:0]
    cb_lots = cb.groupby("account_key")["lots"].sum().reindex(idx).fillna(0.0)
    out["long_lots_t0"] = long0.round(2); out["short_lots_t0"] = short0.round(2)
    out["long_lots_after"] = long1.round(2); out["short_lots_after"] = short1.round(2)
    net0 = long0 - short0; net1 = long1 - short1
    out["net_lots_t0"] = net0.round(2); out["net_lots_after"] = net1.round(2)
    out["close_by_lots_in_window"] = cb_lots.round(2)
    out["leverage_after"] = (notional1 / out["balance_at_t0_est"].where(out["balance_at_t0_est"] > 1.0)).round(1)
    hedged = (long0 > 0) & (short0 > 0)
    smaller = np.minimum(long0, short0)
    # UNWIND = the hedge's directional exposure GREW inside the window by at
    # least half of the hedged amount, and what is left is a leveraged bet.
    # A close-by nets both legs (net unchanged) so it cannot register here;
    # closing one leg at market does.
    grew = (net1.abs() - net0.abs())
    out["hedged_at_t0"] = hedged
    out["hedge_unwind"] = (hedged & (smaller > 0) & (grew >= 0.5 * smaller)
                           & (out["leverage_after"].fillna(0.0) >= HEDGE_LEVERAGE))

    def detail(a):
        if not out.at[a, "hedge_unwind"]:
            return ""
        return (f"hedged {long0.get(a, 0):g}L/{short0.get(a, 0):g}S (net {net0.get(a, 0):+g}) into the window; of those, "
                f"{long1.get(a, 0):g}L/{short1.get(a, 0):g}S remain after it (net {net1.get(a, 0):+g}) -> directional exposure "
                f"grew by {grew.get(a, 0):g} lots through market closes"
                + (f" ({cb_lots.get(a, 0):g} lots of close-by netting excluded)" if cb_lots.get(a, 0) > 0 else "")
                + f", leverage after {out.at[a, 'leverage_after']}x")
    out["hedge_detail"] = [detail(a) for a in idx]
    return out, notes


# --------------------------------------------------------- classification
#: The third measure -- USD notional traded per active month -- has no fixed
#: dollar table, so its tiers are PERCENTILES fitted to the population the
#: analysis returns (the affected accounts): VIP = top 2.5%, High = top 10%,
#: Mid = top 40%, Low = top 70%, Micro = the rest (accounts that traded
#: nothing sit in Micro whatever the percentiles say).
NOTIONAL_PCTS = (("VIP", 0.975), ("High", 0.90), ("Mid", 0.60), ("Low", 0.30))


def notional_tiers(notional_monthly: pd.Series) -> dict[str, float]:
    """Dollar cut-offs per tier, fitted to this population (accounts with
    any notional). Empty when fewer than 20 accounts carry a figure."""
    v = pd.to_numeric(notional_monthly, errors="coerce")
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < 20:
        return {}
    return {name: float(v.quantile(p)) for name, p in NOTIONAL_PCTS}


def classify(mltv: pd.Series, monthly_revenue: pd.Series,
             notional_monthly: pd.Series | None = None,
             notional_cuts: dict[str, float] | None = None) -> pd.DataFrame:
    """Tier by MLTV, tier by monthly revenue, tier by monthly USD notional
    (percentile cut-offs fitted to the population), the highest of the three,
    and the combined ratio (revenue / MLTV)."""
    def tier(value, col):
        if value is None or not np.isfinite(value):
            return "Micro"
        for name, m, r in TIERS:
            if value >= (m if col == 0 else r):
                return name
        return "Micro"

    def tier_notional(value):
        if not notional_cuts or value is None or not np.isfinite(value) or value <= 0:
            return "Micro"
        for name, _ in NOTIONAL_PCTS:
            if value >= notional_cuts[name]:
                return name
        return "Micro"
    out = pd.DataFrame(index=mltv.index)
    out["class_mltv"] = [tier(v, 0) for v in mltv.to_numpy(float)]
    out["class_revenue"] = [tier(v, 1) for v in monthly_revenue.reindex(mltv.index).to_numpy(float)]
    if notional_monthly is not None:
        out["class_notional"] = [tier_notional(v) for v in notional_monthly.reindex(mltv.index).to_numpy(float)]
    else:
        out["class_notional"] = "Micro"
    out["client_class"] = [max((a, b, c), key=lambda t: TIER_ORDER[t])
                           for a, b, c in zip(out["class_mltv"], out["class_revenue"], out["class_notional"])]
    out["combined_ratio"] = (monthly_revenue.reindex(mltv.index) / mltv.where(mltv.abs() > 0)).round(2)
    return out


# ------------------------------------------------ 0b. the desk's register
ABUSEDB_GLOBS = (r"C:\Users\RoyVivasi\Documents\notebook\docs\AbuseDb*.csv",
                 r"C:\Users\RoyVivasi\Documents\AbuseDb*.csv",
                 r"C:\Users\RoyVivasi\Downloads\AbuseDb*.csv")
_ABUSEDB: dict = {"path": None, "mtime": None, "frame": None}


#: The desk's register lives in MySQL on the reporting proxy
#: (`reporting.tbl_abuse_db`: ID, Login, OrderId, AbuseType, AmountOfLoss,
#: DateOfAbuse, AmountOfSaved, Comment -- 94k rows, types Stop Order /
#: zero_slippage / Rebate / Mirror / Market Manipulation / Credit / Arbitrage /
#: Rollover / Swap / Warning_Swap / Leverage Abuse). The CSV export the desk
#: hands out is the same table and stays as the fallback.
ABUSEDB_SQL = ("SELECT Login, OrderId, AbuseType, AmountOfLoss, DateOfAbuse, AmountOfSaved, Comment "
               "FROM reporting.tbl_abuse_db")
ABUSEDB_TTL = 3600.0


def abusedb_load() -> pd.DataFrame | None:
    """The dealing desk's own abuse register: MySQL first (reporting.tbl_abuse_db,
    cached an hour), else the newest CSV export on disk. Login only -- it
    carries no server, so a match is by login."""
    import glob, os, time
    now = time.time()
    if (_ABUSEDB["frame"] is not None and _ABUSEDB["path"] == "mysql"
            and now - float(_ABUSEDB["mtime"] or 0) < ABUSEDB_TTL):
        return _ABUSEDB["frame"]
    cx = None
    try:
        cx = _connect("mt4_live01")
        if cx is not None:
            frame = pd.read_sql(ABUSEDB_SQL, cx)
            if len(frame):
                frame["Login"] = pd.to_numeric(frame["Login"], errors="coerce")
                frame = frame.dropna(subset=["Login"])
                frame["Login"] = frame["Login"].astype(int)
                frame["DateOfAbuse"] = pd.to_datetime(frame["DateOfAbuse"], errors="coerce", utc=True)
                frame["source"] = "reporting.tbl_abuse_db"
                _ABUSEDB.update(path="mysql", mtime=now, frame=frame)
                return frame
    except Exception:
        pass
    finally:
        try:
            cx and cx.close()
        except Exception:
            pass
    paths = [p for g in ABUSEDB_GLOBS for p in glob.glob(g)]
    if not paths:
        return _ABUSEDB["frame"]            # a stale MySQL copy beats nothing
    path = max(paths, key=os.path.getmtime)
    mtime = os.path.getmtime(path)
    if _ABUSEDB["frame"] is not None and _ABUSEDB["path"] == path and _ABUSEDB["mtime"] == mtime:
        return _ABUSEDB["frame"]
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame.columns = [str(c).strip() for c in frame.columns]
    frame["Login"] = pd.to_numeric(frame["Login"], errors="coerce")
    frame = frame.dropna(subset=["Login"])
    frame["Login"] = frame["Login"].astype(int)
    frame["DateOfAbuse"] = pd.to_datetime(frame.get("DateOfAbuse"), errors="coerce", utc=True)
    frame["source"] = f"CSV {os.path.basename(path)}"
    _ABUSEDB.update(path=path, mtime=mtime, frame=frame)
    return frame


def abusedb_flags(accounts) -> pd.DataFrame:
    """Per account_key: the register's verdict -- abuse types, the number of
    records, the last date, the money the desk recorded as saved."""
    frame = abusedb_load()
    empty = pd.DataFrame(columns=["abusedb_flag", "abusedb_records", "abusedb_saved_usd"])
    empty.index.name = "account_key"
    if frame is None or frame.empty:
        return empty
    by_login = _by_server(accounts)
    wanted = {login for logins in by_login.values() for login in logins}
    hit = frame[frame["Login"].isin(wanted)]
    if hit.empty:
        return empty
    g = hit.groupby("Login")
    summary = pd.DataFrame({
        "types": g["AbuseType"].agg(lambda s: ", ".join(sorted(set(str(x) for x in s.dropna())))),
        "n": g.size(),
        "last": g["DateOfAbuse"].max(),
        "saved": pd.to_numeric(hit["AmountOfSaved"], errors="coerce").fillna(0.0).groupby(hit["Login"]).sum(),
    })
    servers_per_login: dict[int, list[str]] = defaultdict(list)
    for server, logins in by_login.items():
        for login in logins:
            servers_per_login[login].append(server)
    rows = []
    for login, r in summary.iterrows():
        servers = servers_per_login.get(int(login), [])
        ambiguous = len(servers) > 1
        for server in servers:
            when = r["last"].strftime("%Y-%m-%d") if pd.notna(r["last"]) else "n/a"
            rows.append({"account_key": f"{server}:{login}",
                         "abusedb_flag": f"{r['types']} ({int(r['n'])} record{'s' if r['n'] != 1 else ''}, last {when}"
                                         + ("; login matched on more than one server" if ambiguous else "") + ")",
                         "abusedb_records": int(r["n"]), "abusedb_saved_usd": float(r["saved"])})
    return pd.DataFrame(rows).set_index("account_key") if rows else empty


def reasons(row: pd.Series) -> list[str]:
    """The written case for an abuse flag, one clause per test."""
    out = []
    if row.get("desk_flag"):
        out.append(f"desk flag: {row['desk_flag']}")
    if row.get("abusedb_flag"):
        out.append(f"desk abuse register: {row['abusedb_flag']}")
    lev = row.get("effective_leverage_t0"); notional = row.get("notional_at_t0") or 0.0
    if lev is not None and np.isfinite(lev) and lev >= HIGH_LEVERAGE and notional >= MIN_NOTIONAL:
        out.append(f"high effective leverage into the event: {lev:.0f}x net exposure (net notional ${notional:,.0f}, "
                   f"gross ${row.get('gross_notional_at_t0', 0) or 0:,.0f}, on an estimated balance of "
                   f"${row.get('balance_at_t0_est', 0):,.0f}, {int(row.get('positions_at_t0') or 0)} positions)")
    if row.get("hedge_unwind"):
        out.append(f"hedge unwind at the event: {row.get('hedge_detail', '')}")
    if row.get("favourable_fills", 0) > 0 and (row.get("advantage_usd") or 0.0) >= MIN_ADVANTAGE_USD:
        out.append(f"favourable execution: {int(row['favourable_fills'])} of {int(row.get('fills_in_window', 0))} window fills beat the market "
                   f"by up to {row.get('max_favourable_bps', 0):.1f} bps (${row.get('advantage_usd', 0):,.0f} advantage): {row.get('execution_detail', '')}")
    elif row.get("stop_gap_fills", 0) > 0:
        out.append(f"stop order filled beyond its trigger in the client's favour: {row.get('execution_detail', '')}")
    return out
