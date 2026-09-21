"""ZFX Risk Intelligence -- FastAPI application.

Two halves of one product, gated by permission:

* **Trading** -- the client-level (account-day) routing model: who to A-book
  tomorrow, why, and what it is worth against B-booking everything.
* **Quant**   -- the trade-level model: per-trade copy/hedge decisions, the
  comparative equity curve, and the live copy-trading book.

FastAPI rather than Streamlit, deliberately. This needs session authentication
with an admin approval gate, a Kafka consumer that outlives a page render, and
WebSocket pushes for live risk -- none of which fit Streamlit's rerun model.
"""

from __future__ import annotations

import ipaddress
import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import (
    HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from webapp import (  # noqa: E402
    auth, book_risk, copytrader, glossary, model_service, risk_monitor, validation, views,
)

SESSION_COOKIE = "zfx_session"

#: BETA INSTANCE. Set AF_BETA=1 to run this same codebase as the cut-down
#: instance shared with internal teams: Latency Arbitrage, Toxic Flow and
#: Account Detail only. Everything else -- quant, executive, TAF, copy trader,
#: exposure, cash flow, admin -- stays behind the full instance, which is where
#: development continues.
#:
#: This is ONE codebase deliberately. A forked copy would have to run its own
#: scans and open its own handle on the tick store, and DuckDB allows a single
#: writer: a second scanner is exactly what invalidated the tape on 17 Sep 2026.
#: The beta therefore READS the artefacts the full instance writes.
BETA = os.environ.get("AF_BETA") == "1"

#: The three areas the beta exposes, in nav order.
BETA_TABS = [
    ("latency", "Latency Arbitrage", "Fast, transient price exploitation"),
    ("toxic", "Toxic Flow", "Persistent adverse selection"),
    ("account", "Account Detail", "Drill-down, trades, markouts, replay"),
]

#: Beta nav key -> (path, antifraud pane). Latency and Toxic are two panes of
#: the one antifraud page, promoted to top-level items so the beta reads as
#: three plain sections rather than a rail inside a tab.
BETA_ROUTES = {
    "latency": ("/trading/antifraud?af=latency", "latency"),
    "toxic": ("/trading/antifraud?af=toxic", "toxic"),
    "account": ("/trading/account", None),
}

#: Landing page for the beta.
BETA_HOME = BETA_ROUTES["latency"][0]

app = FastAPI(title="ZFX Anti-Fraud (Beta)" if BETA else "ZFX Risk Intelligence",
              docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))


def _format_timestamp(value) -> str:
    """Epoch seconds -> a compact local stamp for the order log."""
    if not value:
        return "--"
    return datetime.fromtimestamp(value).strftime("%m-%d %H:%M:%S")


templates.env.filters["timestamp"] = _format_timestamp

# Glossary available to every template, so an ambiguous label can carry its
# definition on hover instead of being expanded into a sentence.
templates.env.globals["glossary"] = glossary.GLOSSARY
templates.env.globals["define"] = glossary.term


def _warm_caches() -> None:
    """Load and pre-compute both artefacts before anyone asks for them.

    Cold, the first request had to read the parquet (19.5M rows for Quant),
    derive the day list and coverage, and run the routing simulation -- ~7s for
    Trading and considerably more for Quant. That cost landed on whoever signed
    in first, which is precisely how "the sign-in button does nothing" gets
    reported. Warm is ~35ms.

    Runs in a daemon thread so the server starts accepting connections
    immediately; a request arriving mid-warm simply pays the old cost once.
    """
    for view in (auth.VIEW_TRADING, auth.VIEW_QUANT):
        try:
            frame = model_service.load_scores(view)
            if frame is None or frame.empty:
                continue
            model_service.frame_facts(view, frame)
            # The two aggregates every screen derives. Computing them here means
            # no user ever pays for a groupby over 19.5M rows.
            model_service.daily_series(view, frame)
            model_service.account_totals(view, frame)
            config = model_service.load_config(view)
            model_service.cached_equity_curves(
                view, frame, config.hedge_fraction, config.probability_threshold)
            if view == auth.VIEW_TRADING:
                # Surveillance scans the markout file and iterates per account;
                # warming it here keeps that 17s off a user's first page load.
                views.surveillance_findings(frame)
                # The account-day markout corpus: ~80 MB on OneDrive, ~37 s to
                # read, and Account Detail needs it on every open. Left lazy it
                # landed on whoever opened the first account after a restart --
                # and the beta gets restarted often, so that was most opens.
                try:
                    views.load_markouts(
                        model_service.SCRATCH / "markout_all_servers.parquet")
                except Exception:
                    pass
        except Exception:
            # Warming is an optimisation; a failure here must never stop the
            # app from serving, it just means the first request is slow.
            continue
    # PARALLEL warms -- the slow latency tick-scan must never block the
    # feature table every other tab depends on (both are skipped instantly
    # when a fresh disk copy already serves them).
    def _warm_latency():
        try:
            # Must run IN THIS PROCESS (single-process DuckDB tick store);
            # rebuilds the tab cache + markout side-table with real coverage.
            from webapp import latency_arb
            latency_arb.scan(refresh=not latency_arb.SCAN_CACHE.exists())
        except Exception:
            pass

    def _warm_obs():
        try:
            # The shared 90-day feature table (counts, rule scans,
            # profiling). Loads from its disk copy in seconds when fresh.
            from webapp import rule_models
            rule_models._obs_table_cached(ttl=3600.0)
        except Exception:
            pass
    threading.Thread(target=_warm_latency, daemon=True).start()
    threading.Thread(target=_warm_obs, daemon=True).start()
    threading.Thread(target=_midnight_refresh_loop, daemon=True).start()


def _midnight_refresh_loop() -> None:
    """At the stroke of midnight LONDON time, every night: rebuild every
    tab's cache from live data (latency scan, the shared feature table,
    each active rule's scan, the registry counts) so every day starts with
    fresh precomputed results and the tabs stay instant."""
    import time as _time
    from datetime import datetime as _dt, timedelta as _td
    from zoneinfo import ZoneInfo
    ldn = ZoneInfo("Europe/London")
    while True:
        now = _dt.now(ldn)
        nxt = (now + _td(days=1)).replace(hour=0, minute=0, second=30,
                                          microsecond=0)
        _time.sleep(max(60.0, (nxt - now).total_seconds()))
        try:
            from webapp import latency_arb, rule_models, af_registry
            # 1) shared feature table: drop disk+memory, rebuild, persist.
            try:
                rule_models.OBS_META.unlink(missing_ok=True)
                rule_models._TABLE_CACHE.clear()
                rule_models._obs_table_cached()
            except Exception:
                pass
            # 2) latency scan (background build in-process).
            latency_arb.scan(refresh=True)
            # 3) every active use_ml rule's scan off the fresh table.
            for c in af_registry.load().get("categories", []):
                if c.get("active") and c.get("use_ml") \
                        and c["key"] != "latency_arbitrage":
                    try:
                        rule_models.rule_scan(c["key"], refresh=True)
                    except Exception:
                        pass
            rule_models._COUNTS_CACHE.clear()
            # WALK-FORWARD LOOP, fully scheduled: (1) verify yesterday's
            # snapshot against today's actual active universe; (2) RETRAIN
            # every use_ml rule with the through-yesterday cutoff (training
            # never sees the current day); (3) snapshot the new models'
            # predictions for EVERY account for tomorrow's verification.
            try:
                rule_models.verify_predictions()
            except Exception:
                pass
            try:
                # latency's bespoke tape detector (needs the in-process
                # tick store), then ALL generic pairs off ONE shared panel
                # build -- 3x cheaper than per-rule training.
                from webapp import latency_arb
                latency_arb.train_model()
            except Exception:
                pass
            try:
                rule_models.train_all_rules()
            except Exception:
                pass
            try:
                rule_models.snapshot_predictions()
            except Exception:
                pass
        except Exception:
            pass


def _start_stream() -> None:
    """Begin consuming the event stream at boot.

    Previously the consumer sat 'stopped' until an admin found the button,
    which meant the live risk screens were empty by default -- and an empty
    risk screen reads as "no risk" rather than "not connected". It starts
    automatically and simply reports an error if the cluster is unreachable.
    """
    # Start UNCONDITIONALLY in a background thread. The per-topic consumer
    # threads already report their own connection errors (status "error"),
    # so gating on a synchronous multi-broker probe only risked a slow or
    # transiently-failing probe leaving the whole stream stopped at boot --
    # which is exactly what happened once the feed moved to three prod
    # clusters. Backgrounding it also keeps a slow first connect off the
    # startup path.
    def _boot():
        try:
            from webapp import kafka_service
            # deals only -- materialising high-volume QUOTES into the single-
            # writer duckdb locked out the engine's reads ('store busy') and
            # stalled trading. The engine gets quotes from MT5 directly.
            # DEALS ONLY: materialising high-volume quotes into the
            # single-writer duckdb held the file lock near-constantly and
            # starved the trading engine's opening-poll reads ('store busy'),
            # stalling trading. Quotes for the engine come from MT5 directly;
            # latency markouts use the independent MySQL ticks feed.
            kafka_service.MATERIALISER.start(backfill=True, with_quotes=False)
        except Exception:
            pass
    import threading
    threading.Thread(target=_boot, daemon=True).start()


def _auto_start_vantage() -> None:
    try:
        from webapp import vantage
        if getattr(vantage.load_config(), "auto_start", False):
            vantage.start()
    except Exception:
        pass


def _warehouse_topup() -> None:
    """Keep the local warehouse current: the last 2 days of closed trades from
    every server, every 15 minutes. Without this the scans (latency, rules,
    profiling) re-read a warehouse that stops at the last manual backfill.
    """
    import time as _time
    from webapp import mysql_extract
    _time.sleep(60)
    while True:
        try:
            result = mysql_extract.refresh_recent(days=2)
            Path(ROOT / "artifacts" / "warehouse_topup.json").write_text(
                json.dumps({"at": str(datetime.utcnow())[:19], **result}, default=str), encoding="utf-8")
        except Exception:
            pass
        _time.sleep(15 * 60)


def _latency_autoscan() -> None:
    """Keep the latency scan fresh (latency_arb rules: auto_rescan_minutes).

    Checked every 30 s; the scan itself runs in its own background thread and
    is never started while one is running, so a slow scan simply delays the
    next one instead of stacking up in the trading process.
    """
    import time as _time
    from webapp import latency_arb
    _time.sleep(90)          # let the boot scan and the copier settle first
    while True:
        try:
            latency_arb.autoscan_tick()
        except Exception:
            pass
        _time.sleep(30)


@app.on_event("startup")
def _startup() -> None:
    # This is THE startup hook. It was previously undecorated, so none of the
    # boot work below ran -- most visibly the Kafka materialiser, which is why
    # the live stream sat empty until an admin hit the button. The engine only
    # came up because _auto_start_vantage carried its own decorator; that is now
    # spawned from here instead, so there is one registered entrypoint.
    auth.initialise()

    # THE BETA INSTANCE IS A READER. It starts no background work at all: no
    # Kafka materialiser, no scans, no warehouse top-up, no copy trader.
    #
    # This is not tidiness, it is correctness. The tick store is DuckDB, which
    # permits ONE writer; a second process materialising ticks is precisely what
    # locked the .wal and invalidated the tape on 17 Sep 2026. The scans and the
    # top-up would likewise double the MySQL load and race the full instance on
    # the same artefact files. The beta renders what the full instance produced.
    if BETA:
        return

    threading.Thread(target=_warm_caches, daemon=True).start()
    threading.Thread(target=_start_stream, daemon=True).start()
    threading.Thread(target=_auto_start_vantage, daemon=True).start()
    threading.Thread(target=_latency_autoscan, daemon=True).start()
    threading.Thread(target=_warehouse_topup, daemon=True).start()

    def _warm_antifraud():
        try:
            from webapp import antifraud
            antifraud.warm()
            antifraud.start_autopilot()   # no-op unless enabled + webhook set
        except Exception:
            pass
        try:
            from webapp import alert_engine
            alert_engine.start()          # user-defined alerts resume
        except Exception:
            pass
        try:
            # Pre-train the Early-Warning models so the tab shows results
            # instantly instead of blocking a page load on a ~7-minute train.
            # ensure() adopts the persisted disk cache if it still matches the
            # corpus/rules, otherwise trains once in the background.
            from webapp import rule_forecast
            rule_forecast.ensure()
        except Exception:
            pass
        try:
            from webapp import taf         # TAF overview cache (cost accounting)
            taf.ensure()
            taf.log_predictions()           # snapshot today's watchlist (live monitor)
        except Exception:
            pass
    threading.Thread(target=_warm_antifraud, daemon=True).start()

    def _ad_ticker():
        try:
            from webapp import ad_refresh
            # checks hourly; rebuilds the account-day corpus whenever its
            # newest decision_day falls behind YESTERDAY, so live always joins
            # the previous trading day the way the walk-forward did.
            ad_refresh.start_daily_ticker()
        except Exception:
            pass
    threading.Thread(target=_ad_ticker, daemon=True).start()


def current_user(request: Request) -> auth.User | None:
    return auth.user_for_session(request.cookies.get(SESSION_COOKIE))


def render(request: Request, template: str, **context) -> HTMLResponse:
    user = context.pop("user", None) or current_user(request)
    return templates.TemplateResponse(
        request, template,
        {"user": user, "views": auth.ALL_VIEWS,
         # Every template can ask `beta` whether it is being served by the
         # cut-down instance, so one template serves both.
         "beta": BETA, "beta_tabs": BETA_TABS, "beta_routes": BETA_ROUTES,
         **context}
    )


# --------------------------------------------------------------------------
# beta instance gate
# --------------------------------------------------------------------------
#: Path prefixes the beta serves. DEFAULT-DENY: anything not matched here is
#: redirected to the beta home, so a component added to the full instance later
#: cannot appear in the beta by accident -- it has to be allowed on purpose.
BETA_ALLOWED = (
    "/login", "/logout", "/register", "/static/", "/favicon",
    "/trading/antifraud", "/trading/account",
    "/trading/latency_arb.csv", "/trading/toxic_flow.csv", "/trading/antifraud.csv",
    "/api/antifraud/",          # both engines, their rules, audit and training
    "/api/replay/",             # the replay player on the Order tab
    "/api/account/chart",       # the price chart ON the allowed Account Detail
)

@app.middleware("http")
async def _beta_gate(request: Request, call_next):
    if not BETA:
        return await call_next(request)
    path = request.url.path
    if path.startswith(BETA_ALLOWED):
        return await call_next(request)
    # Anything else -- quant, executive, TAF, copy trader, exposure, cash flow,
    # admin -- belongs to the development instance only.
    if path.startswith("/api/"):
        return JSONResponse({"error": "not available in the beta instance"},
                            status_code=404)
    return RedirectResponse(BETA_HOME, status_code=303)


#: IP ALLOWLIST. `AF_IP_ALLOWLIST` holds a comma-separated list of addresses or
#: CIDR blocks -- the VPN office egress addresses -- that may reach the app at
#: all; every other source gets 403 before authentication is even considered.
#:
#: FAIL-OPEN BY DESIGN. Unset or empty means no IP restriction, which is the
#: behaviour this app has always had. A typo in the variable therefore degrades
#: to "as before" rather than locking every office out of a running instance,
#: and loopback is always allowed so the host itself keeps working regardless.
#:
#: This is a SECOND lock, not the first one: it can only ever deny. What lets a
#: remote office reach the port at all is the Windows Firewall rule, which is a
#: separate, elevated change (tools/firewall_allow_offices.ps1).
def _parse_allowlist(raw: str) -> tuple:
    networks = []
    for entry in (raw or "").replace(";", ",").split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            networks.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            print(f"[ip-allowlist] ignoring malformed entry {entry!r}", flush=True)
    return tuple(networks)


#: Setting key backing the admin console's allowlist editor.
IP_ALLOWLIST_KEY = "ip_allowlist"

#: Cached parse of the EFFECTIVE list. The admin console can change it at
#: runtime, so this cannot be resolved once at import; equally it must not cost
#: a database read per request. Two seconds is short enough that an admin sees
#: their edit take effect immediately and long enough to be free under load.
_ALLOWLIST_CACHE: dict = {"raw": None, "nets": (), "at": 0.0}


def allowlist_raw() -> str:
    """The configured list: whatever the admin saved, else the launch environment.

    Presence of the stored row decides, NOT whether it is blank. An admin who
    clears the box is deciding "no restriction", and that has to beat the
    environment variable -- otherwise the console reports the allowlist cleared
    while the env var silently reimposes itself, which is precisely what it did
    on 2026-09-18. The env var remains the bootstrap for a deployment that has
    never been configured, so a fresh install can still come up restricted.
    """
    meta = auth.setting_meta(IP_ALLOWLIST_KEY)
    if meta:
        return meta.get("value", "") or ""
    return os.environ.get("AF_IP_ALLOWLIST", "")


def current_allowlist() -> tuple:
    import time as _time
    now = _time.monotonic()
    if now - _ALLOWLIST_CACHE["at"] > 2.0:
        raw = allowlist_raw()
        if raw != _ALLOWLIST_CACHE["raw"]:
            _ALLOWLIST_CACHE["nets"] = _parse_allowlist(raw)
            _ALLOWLIST_CACHE["raw"] = raw
            print("[ip-allowlist] " + ("active: "
                  + ", ".join(str(n) for n in _ALLOWLIST_CACHE["nets"])
                  + " (+ loopback)" if _ALLOWLIST_CACHE["nets"]
                  else "no restriction"), flush=True)
        _ALLOWLIST_CACHE["at"] = now
    return _ALLOWLIST_CACHE["nets"]


def ip_allowed(host: str, nets=None) -> bool:
    """Whether `host` passes. Exposed so the admin console can refuse to save a
    list that would lock out the admin making the change."""
    nets = current_allowlist() if nets is None else nets
    if not nets:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or any(address in net for net in nets)


@app.middleware("http")
async def _ip_gate(request: Request, call_next):
    # Registered last, so it runs FIRST -- ahead of the beta gate and auth.
    if not current_allowlist():
        return await call_next(request)
    host = request.client.host if request.client else ""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        print(f"[ip-allowlist] blocked unparseable client {host!r}", flush=True)
        return PlainTextResponse("Forbidden", status_code=403)
    if address.is_loopback or any(address in net for net in current_allowlist()):
        return await call_next(request)
    # Logged so "a teammate cannot connect" is answerable from the log rather
    # than guessed at -- the blocked source address is the whole diagnosis.
    print(f"[ip-allowlist] blocked {host} -> {request.url.path}", flush=True)
    return PlainTextResponse("Forbidden", status_code=403)


# --------------------------------------------------------------------------
# authentication
# --------------------------------------------------------------------------
@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request, registered: str = "", error: str = ""):
    if current_user(request):
        return RedirectResponse("/", status_code=303)
    return render(request, "login.html", notice=registered, error=error)


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    user = auth.authenticate(username, password)
    if user is None:
        return render(request, "login.html", error="Incorrect username or password.")
    if not user.approved:
        # Deliberately explicit: a pending user should know their account exists
        # and is waiting, not be left guessing at a wrong password.
        return render(request, "login.html",
                      error="This account is awaiting administrator approval.")
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(SESSION_COOKIE, auth.start_session(user.id),
                        httponly=True, samesite="lax", max_age=auth.SESSION_TTL_SECONDS)
    return response


@app.get("/register", response_class=HTMLResponse)
def register_form(request: Request):
    return render(request, "register.html")


@app.post("/register")
def register(request: Request, username: str = Form(...), email: str = Form(""),
             password: str = Form(...), views: list[str] = Form(default=[])):
    ok, message = auth.register(username, email, password, tuple(views))
    if not ok:
        return render(request, "register.html", error=message)
    return render(request, "login.html", notice=message)


@app.get("/logout")
def logout(request: Request):
    auth.end_session(request.cookies.get(SESSION_COOKIE))
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE)
    return response


# --------------------------------------------------------------------------
# main views
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    # Everyone lands on the command hub -- the flywheel of sections.
    return RedirectResponse("/hub", status_code=303)


@app.get("/hub", response_class=HTMLResponse)
def hub(request: Request):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    return render(request, "hub.html", user=user, tabs=None, view="hub")


@app.get("/replay", response_class=HTMLResponse)
def replay_page(request: Request):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    if not user.may_see(auth.VIEW_TRADING):
        return RedirectResponse("/hub", status_code=303)
    return render(request, "replay.html", user=user, tabs=None, view="replay")


@app.get("/taf", response_class=HTMLResponse)
def taf_page(request: Request):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    if not user.may_see(auth.VIEW_TRADING):
        return RedirectResponse("/hub", status_code=303)
    return render(request, "taf.html", user=user, tabs=None, view="taf")


@app.get("/api/taf/overview")
def api_taf_overview(request: Request):
    if current_user(request) is None:
        return {"error": "unauthenticated"}
    from webapp import taf
    try:
        return taf.overview()
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


@app.get("/api/taf/watchlist")
def api_taf_watchlist(request: Request, horizon: int = 5):
    if current_user(request) is None:
        return {"error": "unauthenticated"}
    from webapp import taf
    try:
        return taf.watchlist(max(1, min(int(horizon), 30)))
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


@app.get("/api/taf/discovery")
def api_taf_discovery(request: Request):
    if current_user(request) is None:
        return {"error": "unauthenticated"}
    from webapp import taf
    try:
        return taf.discovery()
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


@app.get("/api/taf/monitor")
def api_taf_monitor(request: Request):
    if current_user(request) is None:
        return {"error": "unauthenticated"}
    from webapp import taf
    try:
        return taf.monitor()
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


TRADING_TABS = [
    ("summary", "Summary", "One page for management"),
    ("overview", "Overview", "Model vs B-book, daily P&L"),
    ("abook", "A-Book Manifest", "Today's routing decisions"),
    ("clients", "Client Intelligence", "Edge / toxic / arbitrage taxonomy"),
    ("antifraud", "AntiFraud", "Behavioural profiling, screening, action testing"),
    ("account", "Account Detail", "Drill-down, trades, markouts"),
    ("exposure", "Exposure", "Net notional by symbol, live and historical"),
    ("regions", "Regions", "P&L and accounts by client geography"),
    ("cashflow", "Cash Flow", "Deposits, withdrawals, compliance flags"),
    ("surveillance", "Surveillance", "Anti-fraud detections with evidence"),
    ("bookrisk", "Book Risk", "Net exposure caps -- the drawdown layer"),
    ("risk", "Risk Monitor", "VaR, concentration, live stream"),
    ("performance", "Model Performance", "Walk-forward diagnostics"),
    ("validation", "Validation", "Reconciliation, AUC, regime stability"),
]

QUANT_TABS = [
    ("summary", "Summary", "One page for management"),
    ("overview", "Overview", "Trade-level model vs B-book"),
    ("signals", "Trade Signals", "Per-trade routing decisions"),
    ("copytrader", "Copy Trader", "MT5 execution + live vs expected"),
    ("exposure", "Exposure", "Net notional by symbol, live and historical"),
    ("regions", "Regions", "P&L and accounts by client geography"),
    ("cashflow", "Cash Flow", "Deposits, withdrawals, compliance flags"),
    ("exits", "Exit Policy", "Where copied trades actually close"),
    ("research", "Research", "Feature importance, ablations"),
    ("risk", "Risk Monitor", "VaR, concentration, live stream"),
    ("performance", "Model Performance", "Walk-forward diagnostics"),
    ("lab", "Strategy Lab", "Live walk-forward + experiment variants"),
    ("validation", "Validation", "Reconciliation, AUC, regime stability"),
]


def _guard(request: Request, view: str):
    """Returns (user, redirect). A redirect means the caller must return it."""
    user = current_user(request)
    if user is None:
        return None, RedirectResponse("/login", status_code=303)
    if not user.may_see(view):
        return user, render(request, "no_access.html", user=user, denied_view=view)
    return user, None


# NOTE: declared BEFORE `/trading/{tab}`. FastAPI matches routes in definition
# order, so the catch-all would otherwise swallow "abook.csv" as an unknown tab
# and redirect instead of serving the file.
# ---- AntiFraud / Behavioural Profiling -------------------------------------
# Declared before the `/trading/{tab}` catch-all so the JSON/CSV routes are not
# swallowed by it.
def _short_term_scores():
    """Predicted short-term profitability per account (the quant model's live
    win-probability, averaged) -- used to PRIORITISE flags on clients we also
    expect to be short-term profitable, per the brief. Empty if unavailable."""
    try:
        frame = model_service.load_scores(auth.VIEW_QUANT)
        if frame is None or frame.empty:
            return None
        recent = frame.copy()
        recent["day"] = pd.to_datetime(recent["day"])
        recent = recent.loc[recent["day"] >= recent["day"].max() - pd.Timedelta(days=14)]
        return recent.groupby("account_key")["score"].mean()
    except Exception:
        return None


@app.get("/api/antifraud/classify")
def antifraud_classify(request: Request, profile: str = "", q: str = "",
                       min_confidence: float = 0.0, limit: int = 500,
                       as_of: str = "", start: str = "", end: str = ""):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import antifraud
    try:
        result = antifraud.classify(
            short_term_scores=_short_term_scores() if not (as_of or start) else None,
            as_of=as_of or None, start=start or None, end=end or None)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}", "rows": []}
    if result.empty:
        return {"rows": [], "availability": antifraud.data_availability()}
    if profile:
        result = result.loc[result["profile"] == profile]
    if q:
        result = result.loc[result["account_key"].str.contains(q, case=False)]
    if min_confidence:
        result = result.loc[result["confidence"] >= float(min_confidence)]
    counts = {p: int((result["profile"] == p).sum()) for p in antifraud.PROFILES}
    return {"rows": result.head(limit).to_dict("records"),
            "total": int(len(result)), "counts": counts,
            "availability": antifraud.data_availability()}


# ---- Latency Arbitrage module (spec: detection -> decision -> control) -----
@app.get("/api/antifraud/latency")
def antifraud_latency(request: Request, refresh: int = 0,
                      as_of: str = "", start: str = ""):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import latency_arb, rule_models
    # The beta never scans IN PROCESS: a scan opens the DuckDB tick store and a
    # second writer invalidates the tape for both instances. Its Rescan instead
    # queues the request for the full instance, which owns the tape.
    queued = None
    if refresh and BETA:
        queued = latency_arb.request_rescan("beta")
        refresh = 0
    out = latency_arb.scan(refresh=bool(refresh),
                           as_of=as_of or None, start=start or None)
    if queued is not None:
        out = dict(out)
        out["building"] = bool(queued.get("requested")) or out.get("building", False)
        out["note"] = queued.get("note")
    try:
        # both model families' quality metrics for the KPI strip: the tape
        # detector lives in kpis; the generic NOW/EW pair rides along here.
        out["rule_meta"] = rule_models.rule_meta("latency_arbitrage")
        out["verification"] = (rule_models.latest_verification()
                               .get("rules") or {}).get("latency_arbitrage")
        out["rule_stats"] = rule_models.rule_hits_stats("latency_arbitrage")
        out["autoscan"] = latency_arb.autoscan_state()
    except Exception:
        pass
    return out


@app.get("/api/antifraud/latency/rules")
def antifraud_latency_rules(request: Request):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import latency_arb
    return latency_arb.load_rules()


@app.post("/api/antifraud/latency/rules")
async def antifraud_latency_rules_save(request: Request):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import latency_arb
    body = await request.json()
    return latency_arb.save_rules(body or {})


@app.post("/api/antifraud/latency/decide")
async def antifraud_latency_decide(request: Request):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import latency_arb
    body = await request.json()
    return latency_arb.decide(str(body.get("account") or ""),
                              str(body.get("decision") or ""),
                              operator=getattr(user, "email", "") or "risk",
                              note=str(body.get("note") or ""))


@app.post("/api/antifraud/latency/train")
def antifraud_latency_train(request: Request):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    # trains BOTH: the bespoke tape-proved detector and the generic
    # EW-5d/NOW pair that fills the tab's EW column. In the BACKGROUND --
    # training takes minutes and must never hang the tab.
    from webapp import rule_models
    threading.Thread(target=lambda: rule_models.train_rule(
        "latency_arbitrage"), daemon=True).start()
    return {"started": True,
            "note": "training in the background — saved models refresh on "
                    "the next load when it completes"}


@app.get("/api/antifraud/latency/tags")
def antifraud_latency_tags(request: Request, catalog_only: int = 0):
    """Automation feed: the tag catalog, the current tag set of every account
    the latest live scan tagged, and the feed-level tags. Accounts not listed
    are LA_VERDICT_CLEAR; the audit log (/api/antifraud/latency/audit) holds
    every change of an account's tags with a timestamp."""
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import latency_arb, latency_tags
    out = {"catalog": latency_tags.catalog(),
           "unlisted_accounts": "LA_VERDICT_CLEAR"}
    if catalog_only:
        return out
    scan = latency_arb.latest_full_scan()
    if scan.get("building"):
        out["note"] = scan.get("note")
        return out
    out.update(generated_at=scan.get("generated_at"), mode=scan.get("mode"),
               accounts=[{k: r.get(k) for k in ("account", "verdict", "band", "risk", "confidence",
                                                "latency_events", "econ_usd", "tags", "last_ts")}
                         for r in scan.get("rows", [])],
               feed=latency_tags.feed_tags(scan.get("root_cause") or [],
                                           (scan.get("spec") or {}).get("top_replicated") or []))
    return out


@app.get("/api/antifraud/latency/trade_markout")
def antifraud_trade_markout(request: Request, account: str, order: str = "",
                            open_time: str = "", symbol: str = "", cmd: str = "",
                            volume_lots: float = 0.0, close_time: str = "",
                            open_price: float = 0.0, close_price: float = 0.0,
                            net_profit: float = 0.0):
    """Markout evidence for ONE trade (spec horizons, broker tape and the
    independent reference). The trade-row fields are a fallback for trades
    the warehouse has not received yet."""
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import trade_markout
    fallback = {"open_time": open_time, "close_time": close_time or None, "symbol": symbol,
                "cmd": cmd, "volume_lots": volume_lots, "open_price": open_price,
                "close_price": close_price, "net_profit": net_profit} if open_time and open_price else None
    try:
        return trade_markout.trade_markout(account, order or None, open_time or None,
                                           symbol or None, fallback)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


@app.get("/api/antifraud/profiling")
def antifraud_profiling(request: Request, as_of: str = "", start: str = ""):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import rule_models
    return rule_models.profiling_scan(as_of=as_of or None,
                                      start=start or None)


@app.get("/api/antifraud/latency/trend")
def antifraud_latency_trend(request: Request, days: int = 30):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import latency_arb
    return {"rows": latency_arb.history_trend(days)}


@app.get("/api/antifraud/latency/audit")
def antifraud_latency_audit(request: Request, limit: int = 100):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import latency_arb
    return {"rows": latency_arb.audit_tail(limit)}


@app.get("/trading/latency_arb.csv")
def latency_arb_csv(request: Request):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import latency_arb
    import io, csv as _csv
    scan = latency_arb.scan()
    rows = scan.get("rows", [])
    buf = io.StringIO()
    if rows:
        w = _csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return PlainTextResponse(buf.getvalue(), media_type="text/csv",
                             headers={"Content-Disposition":
                                      "attachment; filename=latency_arb.csv"})


# ---- P1 Engine B: Toxic Flow (spec s5) -------------------------------------
# The scan itself is produced inside the latency pass (spec s12: one shared
# Markout Engine), so these routes only read, re-score and record decisions.
@app.get("/api/antifraud/toxic")
def antifraud_toxic(request: Request, refresh: int = 0):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import toxic_flow
    # Same as latency: in the beta a Rescan is a request to the full instance,
    # never a scan here. Engine B rebuilds with Engine A on that one pass.
    if refresh and BETA:
        from webapp import latency_arb
        queued = latency_arb.request_rescan("beta")
        out = dict(toxic_flow.scan(refresh=False))
        out["building"] = bool(queued.get("requested"))
        out["note"] = queued.get("note")
        return out
    return toxic_flow.scan(refresh=bool(refresh))


@app.get("/api/antifraud/toxic/rules")
def antifraud_toxic_rules(request: Request):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import toxic_flow
    return toxic_flow.load_rules()


@app.post("/api/antifraud/toxic/rules")
async def antifraud_toxic_rules_save(request: Request):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import toxic_flow
    body = await request.json()
    return toxic_flow.save_rules(body or {})


@app.post("/api/antifraud/toxic/decide")
async def antifraud_toxic_decide(request: Request):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import toxic_flow
    body = await request.json()
    return toxic_flow.decide(str(body.get("account") or ""),
                             str(body.get("decision") or ""),
                             getattr(user, "email", "") or "risk",
                             str(body.get("note") or ""))


@app.get("/api/antifraud/toxic/tags")
def antifraud_toxic_tags(request: Request, catalog_only: int = 0):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import toxic_flow
    return toxic_flow.tags_feed(catalog_only=bool(catalog_only))


@app.get("/api/antifraud/toxic/trend")
def antifraud_toxic_trend(request: Request, days: int = 30):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import toxic_flow
    return {"rows": toxic_flow.history_trend(days)}


@app.get("/api/antifraud/toxic/audit")
def antifraud_toxic_audit(request: Request, limit: int = 100):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import toxic_flow
    return {"rows": toxic_flow.audit_tail(limit)}


@app.get("/trading/toxic_flow.csv")
def toxic_flow_csv(request: Request):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import toxic_flow
    import io, csv as _csv
    rows = toxic_flow.scan().get("rows", [])
    flat = [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows]
    buf = io.StringIO()
    if flat:
        w = _csv.DictWriter(buf, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
    return PlainTextResponse(buf.getvalue(), media_type="text/csv",
                             headers={"Content-Disposition":
                                      "attachment; filename=toxic_flow.csv"})


# ---- Dynamic category registry ---------------------------------------------
@app.get("/api/antifraud/registry")
def antifraud_registry(request: Request, fields: int = 0):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import af_registry
    out = af_registry.load(str(getattr(user, "id", "") or getattr(user, "email", "")))
    if fields:
        out["fields"] = af_registry.field_universe()
    return out


@app.post("/api/antifraud/registry")
async def antifraud_registry_save(request: Request, as_admin: int = 0):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import af_registry
    body = await request.json()
    uid = str(getattr(user, "id", "") or getattr(user, "email", ""))
    is_admin = getattr(user, "role", "") == "admin"
    out = af_registry.save(body or {}, user_id=uid,
                           as_admin=bool(as_admin) and is_admin)
    try:
        from webapp import rule_models
        rule_models._COUNTS_CACHE.clear()   # definitions changed
    except Exception:
        pass
    return out


@app.post("/api/antifraud/registry/restore")
def antifraud_registry_restore(request: Request):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import af_registry
    return af_registry.restore(
        str(getattr(user, "id", "") or getattr(user, "email", "")))


@app.get("/api/antifraud/registry/counts")
def antifraud_registry_counts(request: Request):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import rule_models
    return rule_models.registry_counts()


@app.get("/api/antifraud/rule/{key}")
def antifraud_rule_scan(request: Request, key: str, refresh: int = 0,
                        as_of: str = "", start: str = ""):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import rule_models
    return rule_models.rule_scan(key, refresh=bool(refresh),
                                 as_of=as_of or None, start=start or None)


@app.post("/api/antifraud/rule/{key}/train")
def antifraud_rule_train(request: Request, key: str):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import rule_models
    threading.Thread(target=lambda: rule_models.train_rule(key),
                     daemon=True).start()
    return {"started": True,
            "note": "training in the background — metrics refresh when done"}


@app.get("/api/antifraud/calendar")
def antifraud_calendar(request: Request, refresh: int = 0):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import econ_calendar
    return {"events": econ_calendar.events(refresh=bool(refresh)),
            "source_link": econ_calendar.INVESTING_URL}


# ---- Event Impact analyzer -------------------------------------------------
@app.get("/api/antifraud/events")
def antifraud_events(request: Request):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import event_impact
    return {"events": event_impact.detect_events()}


@app.get("/api/antifraud/event_impact")
def antifraud_event_impact(request: Request, start: str = "", end: str = "",
                           symbols: str = "", loss_limit: float = 500.0,
                           profit_limit: float = 500.0, refresh: int = 0):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import event_impact
    if not start or not end:
        return event_impact.last()      # instant: the previous analysis
    return event_impact.analyze(start, end, symbols=symbols,
                                loss_limit=loss_limit,
                                profit_limit=profit_limit,
                                refresh=bool(refresh))


@app.get("/api/antifraud/event_impact/nfp")
def antifraud_event_impact_nfp(request: Request):
    """The predefined default window: the last NFP session for XAUUSD."""
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import event_impact
    return event_impact.last_nfp()


@app.get("/api/antifraud/event_impact/preset")
def antifraud_event_impact_preset(request: Request, name: str = "last_nfp"):
    """Named windows for the tab's buttons: `nfp_2026_09_04` (fixed) or
    `last_nfp` (the calendar's most recent first-Friday release)."""
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import event_impact
    return event_impact.preset(name)


@app.get("/api/antifraud/event_impact.xlsx")
def antifraud_event_impact_xlsx(request: Request, start: str = "", end: str = "",
                                symbols: str = "", loss_limit: float = 500.0,
                                profit_limit: float = 500.0):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return PlainTextResponse("auth", status_code=403)
    from webapp import event_impact
    # Cached only: a download must never sit on a multi-minute recompute
    # (the browser gives up on it). Memory, then the disk copy of this
    # window; otherwise tell the user to run Analyze first.
    data = (event_impact.last() if not start or not end
            else event_impact.analyze(start, end, symbols=symbols,
                                      loss_limit=loss_limit,
                                      profit_limit=profit_limit,
                                      cached_only=True))
    if data.get("error") == "not cached":
        return PlainTextResponse("No analysis is cached for this window yet. Press Analyze "
                                 "(or one of the NFP buttons) and download when it finishes.",
                                 status_code=409)
    if data.get("error"):
        return PlainTextResponse(data["error"], status_code=400)
    try:
        blob = event_impact.build_excel(data)
    except Exception as error:
        return PlainTextResponse(f"{type(error).__name__}: {error}",
                                 status_code=500)
    from fastapi.responses import Response
    win = data.get("window", {})
    sy = win.get("symbols"); sy = "_".join(sy) if isinstance(sy, list) else (sy or "all")
    fname = f"event_impact_{sy}_{str(win.get('start',''))[:10]}.xlsx"
    return Response(
        blob, media_type="application/vnd.openxmlformats-officedocument."
                         "spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@app.get("/trading/antifraud.csv")
def antifraud_csv(request: Request, profile: str = ""):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return redirect
    from webapp import antifraud
    result = antifraud.classify(short_term_scores=_short_term_scores())
    if profile and not result.empty:
        result = result.loc[result["profile"] == profile]
    return PlainTextResponse(
        result.to_csv(index=False), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="antifraud_{profile or "all"}.csv"'})


@app.get("/api/antifraud/action_test")
def antifraud_action_test(request: Request, account: str):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import antifraud
    try:
        return antifraud.action_test(account)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


@app.get("/api/antifraud/client")
def antifraud_client(request: Request, account: str):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import antifraud
    try:
        return antifraud.client_screen(account)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


@app.get("/api/antifraud/ml")
def antifraud_ml(request: Request, limit: int = 300):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import antifraud
    try:
        ml = antifraud.ml_scores()
        return {"rows": ml.head(limit).to_dict("records"), "total": int(len(ml))}
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}", "rows": []}


@app.post("/api/antifraud/lark")
async def antifraud_lark(request: Request):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import antifraud
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        return antifraud.send_lark_alerts(
            min_priority=float(body.get("min_priority", 50.0)),
            webhook=body.get("webhook") or None)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


@app.get("/api/antifraud/rules")
def antifraud_get_rules(request: Request):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import antifraud
    # metrics available to custom watch rules: the behaviour panel's numeric
    # columns, so the Add-rule form offers real fields, not guesses
    import pandas as pd
    try:
        panel = antifraud._account_panel()
        metrics = sorted(c for c in panel.columns
                         if pd.api.types.is_numeric_dtype(panel[c]))
    except Exception:
        metrics = []
    return {"rules": antifraud.load_rules(), "schema": antifraud.RULES_SCHEMA,
            "profiles": antifraud.PROFILES,
            "severity": antifraud.SEVERITY_WEIGHT,
            "universe": antifraud.ACTION_UNIVERSE,
            "custom_metrics": metrics,
            "alerts": antifraud.alerts_config()}


@app.get("/api/antifraud/universe")
def antifraud_universe(request: Request, as_of: str = "", start: str = "",
                       end: str = "", limit: int = 500):
    """The unified classification across AntiFraud, routing taxonomy and ML --
    highest-priority flag claims the primary label; columns for every type."""
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import antifraud
    try:
        return antifraud.unified_universe(as_of=as_of or None,
                                          start=start or None,
                                          end=end or None, limit=limit)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


@app.get("/api/antifraud/markout_grid")
def antifraud_markout_grid(request: Request, hx: str = "markout_5m",
                           hy: str = "markout_1h", percentile: float = 95.0,
                           start: str = "", end: str = ""):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import antifraud
    try:
        return antifraud.markout_grid(hx=hx, hy=hy, percentile=percentile,
                                      start=start or None, end=end or None)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


@app.post("/api/antifraud/expr_validate")
async def antifraud_expr_validate(request: Request):
    """Dry-run a rule expression: how many accounts would it flag right now?"""
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import antifraud
    try:
        payload = await request.json()
        panel = antifraud._account_panel()
        mask = antifraud.safe_expr_mask(str(payload.get("expr", "")), panel)
        hits = panel.index[mask].tolist()
        return {"ok": True, "matches": len(hits), "sample": hits[:8]}
    except Exception as error:
        return {"ok": False, "error": str(error)}


@app.get("/api/antifraud/forecast")
def antifraud_forecast(request: Request, horizon: int = 5):
    """Early warning: P(account qualifies for each rule within `horizon` days),
    one model per rule, retrained when rules or the corpus change."""
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import rule_forecast
    try:
        return rule_forecast.forecast(horizon_days=horizon)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


@app.get("/api/alerts/analysis")
def alerts_analysis(request: Request):
    """Per-alert firing/delivery analytics for the analysis panel."""
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import alert_engine
    return alert_engine.analysis(user.username)


@app.get("/api/alerts")
def alerts_list(request: Request):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import alert_engine
    return {"alerts": alert_engine.list_alerts(user.username),
            "catalog": alert_engine.catalog()}


@app.post("/api/alerts")
async def alerts_create(request: Request):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import alert_engine
    try:
        spec = await request.json()
        # Validate the webhook up-front so a chat-invite link (the common
        # mistake) is caught here, not silently swallowed at fire time.
        hook = str(spec.get("webhook") or "").strip()
        warning = alert_engine.webhook_problem(hook) if hook else None
        alert_id = alert_engine.save_alert(user.username, spec)
        alert_engine.start()
        return {"ok": True, "id": alert_id, "webhook_warning": warning}
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


@app.post("/api/alerts/delete")
def alerts_delete(request: Request, id: int = Form(...)):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import alert_engine
    alert_engine.delete_alert(user.username, id)
    return {"ok": True}


@app.post("/api/alerts/toggle")
def alerts_toggle(request: Request, id: int = Form(...), on: str = Form("1")):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import alert_engine
    alert_engine.toggle_alert(user.username, id, on == "1")
    return {"ok": True}


@app.get("/api/alerts/history")
def alerts_history(request: Request, id: int):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import alert_engine
    return {"history": alert_engine.history(user.username, id)}


@app.post("/api/alerts/comment")
def alerts_comment(request: Request, id: int = Form(...), text: str = Form("")):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import alert_engine
    alert_engine.comment(user.username, id, text)
    return {"ok": True}


@app.post("/api/antifraud/alerts")
async def antifraud_set_alerts(request: Request):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import antifraud
    try:
        payload = await request.json()
        antifraud.save_alerts_config(payload)
        antifraud.start_autopilot()
        return {"ok": True, "config": antifraud.alerts_config()}
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


@app.post("/api/antifraud/rules")
async def antifraud_set_rules(request: Request):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return {"error": "auth"}
    from webapp import antifraud
    try:
        payload = await request.json()
        antifraud.save_rules(payload)
        return {"ok": True}
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


@app.get("/trading/abook.csv")
def abook_csv(request: Request, day: str = "", hedge: float = 0.05, threshold: float = 0.0):
    """The manifest as a file the dealing desk (or a bridge job) can consume."""
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return redirect
    frame = model_service.load_scores(auth.VIEW_TRADING)
    if frame is None or frame.empty:
        return PlainTextResponse("no model artefact", status_code=404)
    manifest = views.abook_manifest(frame, day or None, hedge, threshold, limit=100000)
    columns = [c for c in ["account_key", "day", "category", "reason", "score",
                           "gross_notional", "expected_impact", "trades", "life_win_rate",
                           "life_closes", "life_profit_factor"] if c in manifest.columns]
    export = manifest[columns].copy()
    export.insert(0, "priority", range(1, len(export) + 1))
    export.insert(1, "book", "A_BOOK")
    return PlainTextResponse(
        export.to_csv(index=False), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="abook_{day or "latest"}.csv"'})


# Declared before `/quant/{tab}` for the same reason as the A-book export:
# FastAPI matches in definition order and the catch-all would swallow it.
@app.get("/quant/signals.csv")
def signals_csv(request: Request, day: str = "", hedge: float = 0.05,
                threshold: float = 0.0, symbol: str = ""):
    """Trade signals as a file the execution engine or a dealer can consume."""
    user, redirect = _guard(request, auth.VIEW_QUANT)
    if redirect is not None:
        return redirect
    frame = model_service.load_scores(auth.VIEW_QUANT)
    if frame is None or frame.empty:
        return PlainTextResponse("no model artefact", status_code=404)
    if symbol:
        frame = frame.loc[frame["symbol"] == symbol]
    # Bounded on purpose. Exporting every scored trade produced a 25 MB file
    # covering the whole history, which is not a manifest -- it is the artefact.
    # A day's signals is what an execution desk consumes.
    signals = views.trade_signals(frame, day or None, hedge, threshold, limit=20000)
    columns = [c for c in ["open_time", "account_key", "symbol", "direction", "volume_lots",
                           "notional", "score", "expected_impact", "pnl"] if c in signals.columns]
    export = signals[columns].copy()
    export.insert(0, "priority", range(1, len(export) + 1))
    export.insert(1, "action", "COPY")
    # Spell the side out: `direction` as +1/-1 is easy to invert by accident on
    # the way into an execution system, and this file may drive real orders.
    if "direction" in export.columns:
        export["side"] = export["direction"].map(lambda d: "BUY" if d > 0 else "SELL")
    return PlainTextResponse(
        export.to_csv(index=False), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="signals_{day or "latest"}.csv"'})


def _view_context(view: str, day: str | None, hedge: float | None,
                  threshold: float | None) -> dict:
    """Shared state every model-backed screen needs.

    Reads the cached artefact only. `stale` says the live settings no longer
    match what produced these numbers -- the screen keeps showing them (they are
    still correct FOR THE SETTINGS THAT MADE THEM) and offers Retrain, rather
    than blanking or silently refitting behind a filter change.
    """
    config = model_service.load_config(view)
    if hedge is not None:
        config.hedge_fraction = hedge
    if threshold is not None:
        config.probability_threshold = threshold
    frame = model_service.load_scores(view)
    has_data = frame is not None and not frame.empty
    # Day list and coverage are cached per artefact -- both scan the whole frame
    # (19.5M rows for Quant) and only change when the model is retrained.
    facts = model_service.frame_facts(view, frame) if has_data else {"days": [], "coverage": {}}
    return {
        "config": config,
        "job": model_service.job_state(view),
        "stale": model_service.is_stale(view),
        "meta": model_service.artifact_meta(view),
        "has_data": has_data,
        "days": facts["days"],
        "coverage": facts["coverage"],
        "selected_day": day,
        "frame": frame,
        # Days between the newest warehouse day and today. Drives the "extract
        # is N days behind" banner -- historical panels legitimately end before
        # today, and that should be stated rather than left to be noticed.
        "data_age": ((datetime.now(timezone.utc).date()
                      - datetime.fromisoformat(facts["days"][0]).date()).days
                     if facts["days"] else None),
    }


@app.get("/trading/{tab}", response_class=HTMLResponse)
def trading(request: Request, tab: str, day: str = "", hedge: float | None = None,
            threshold: float | None = None, account: str = "", symbol: str = "",
            horizons: str = "", mk_symbol: str = "", af: str = ""):
    user, redirect = _guard(request, auth.VIEW_TRADING)
    if redirect is not None:
        return redirect
    if tab not in {key for key, _, _ in TRADING_TABS}:
        return RedirectResponse(BETA_HOME if BETA else "/trading/overview",
                                status_code=303)

    view = auth.VIEW_TRADING
    context = _view_context(view, day or None, hedge, threshold)
    frame = context.pop("frame")
    config = context["config"]
    # Book risk reads the warehouse, not the model artefact, so it works whether
    # or not anything has been trained -- the net exposure is a fact about the
    # book. Set outside the has_data guard for exactly that reason.
    if tab == "bookrisk":
        context["book"] = book_risk.report()
    # The validation template handles "no trained model" itself, so the report
    # is built with or without an artefact -- as the quant view already does.
    if tab == "validation":
        context["report"] = validation.report(view)

    if context["has_data"]:
        # Default the day selector to the most recent day with activity.
        if not day and context["days"]:
            context["selected_day"] = context["days"][0]
        if tab == "summary":
            context["summary_report"] = views.executive_summary(
                frame, context["meta"], config.hedge_fraction, view)
        elif tab == "surveillance":
            context["findings"] = views.surveillance_findings(frame)
        elif tab == "overview":
            context["curve"] = model_service.cached_equity_curves(
                view, frame, config.hedge_fraction, config.probability_threshold)
            context["summary"] = views.summary_stats(frame)
        elif tab == "abook":
            manifest = views.abook_manifest(
                frame, context["selected_day"], config.hedge_fraction, config.probability_threshold)
            context["manifest"] = manifest
            context["population"] = len(views.eligible_population(frame, context["selected_day"]))
            context["summary"] = views.summary_stats(frame, context["selected_day"])
            # The scoreboard: what following this exact list actually earned.
            context["outcome"] = views.realised_day_outcome(
                frame, context["selected_day"], set(manifest["account_key"]) if len(manifest) else set())
        elif tab == "clients":
            context["manifest"] = views.abook_manifest(
                frame, context["selected_day"], 1.0, 0.0, limit=2000)
        elif tab == "account":
            if account:
                context["account_key"] = account
                hist = views.account_history(frame, account)
                context["history"] = hist
                # JSON-safe chart arrays built HERE, not via .dt.strftime in
                # the template (a non-datetime `day` there blanked the charts
                # silently). Always clean lists the page can plot directly.
                try:
                    import pandas as _pd
                    if hist is not None and len(hist):
                        days = _pd.to_datetime(hist["day"], errors="coerce")
                        context["hist_days"] = [
                            d.strftime("%Y-%m-%d") if _pd.notna(d) else ""
                            for d in days]
                        context["hist_cum_pnl"] = [
                            round(float(v), 2) for v in hist["cum_pnl"]]
                        sc = hist["score"]
                        context["hist_scores"] = [
                            None if _pd.isna(v) else round(float(v), 4)
                            for v in sc]
                        context["hist_has_score"] = bool(sc.notna().any())
                except Exception as error:
                    context["history_chart_error"] = \
                        f"{type(error).__name__}: {error}"
                # NOTE: the account-level Markout profile panel was removed
                # (it averaged across symbols, which made the curve unreadable).
                # Its data load lived here and has gone with it -- load_markouts
                # reads an ~80 MB parquet, ~37 s cold, and nothing on this page
                # consumed it any more. Per-trade markout evidence is unaffected:
                # the order-click overlay fetches /api/antifraud/latency/
                # trade_markout on demand.
                try:
                    from webapp import latency_arb as _la
                    _flags = _la.account_flagged_orders(account)
                except Exception:
                    _flags = {"orders": [], "status": ""}
                # A set: the template tests every trade row against it, and with
                # no cap an account can carry thousands of flagged orders.
                context["flagged_orders"] = set(_flags["orders"])
                context["flagged_status"] = _flags.get("status", "")
                # Engine B's materially adverse orders, labelled beside the
                # latency flags and pinned the same way.
                _toxic, _toxic_status = {}, ""
                try:
                    from webapp import toxic_flow as _tf
                    _toxic = _tf.account_toxic_orders(account)
                    _trow = _tf.account_toxic_current(account)
                    if _trow:
                        _toxic_status = (f"{_trow.get('state_label') or _trow.get('state')}"
                                         f" (toxicity {_trow.get('score')})")
                except Exception:
                    pass
                context["toxic_orders"] = _toxic
                context["toxic_status"] = _toxic_status
                # ACCOUNT P&L + FLAGGED-FLOW IMPACT: live balance / equity /
                # floating from the trade server, realised P&L on record from
                # the warehouse, and each engine's flagged share and money.
                try:
                    from webapp import replay as _rp
                    context["account_live"] = _rp.account_snapshot(account)
                except Exception:
                    context["account_live"] = {}
                try:
                    from webapp import latency_arb as _la_rules
                    _window = float(_la_rules.load_rules().get("window_days", 7))
                    context["pnl_impact"] = views.account_pnl_impact(
                        account, _flags["orders"], _toxic, window_days=_window)
                except Exception:
                    context["pnl_impact"] = {}
                _eq = (context.get("pnl_impact") or {}).get("equity") or {}
                context["eq_days"] = _eq.get("days") or []
                context["eq_cum"] = _eq.get("cum") or []
                try:
                    context["toxic_metrics"] = _tf.account_toxic_metrics(account)
                except Exception:
                    context["toxic_metrics"] = {}
                _pinned = list(dict.fromkeys(list(_flags["orders"]) + list(_toxic)))
                context["trades"] = views.account_trades(account, symbol or None, pinned_orders=_pinned)
                context["symbols"] = views.account_symbols(account)
                context["selected_symbol"] = symbol
                context["fmt_duration"] = views.fmt_duration
                # HOW THE MODELS SEE THIS CLIENT: every qualifying behavioural
                # profile with its five-axis scores, on the account page.
                try:
                    from webapp import antifraud
                    context["classification"] = (
                        antifraud.client_screen(account).get("profiles") or [])
                except Exception:
                    context["classification"] = []
                # Latency-arbitrage view of this client: current five-axis
                # profile + ML score, and the day-by-day score history.
                try:
                    from webapp import latency_arb
                    context["latency_now"] = latency_arb.account_latency_current(account)
                    context["latency_history"] = latency_arb.account_latency_history(account)
                except Exception:
                    context["latency_now"] = None
                    context["latency_history"] = []
                # Registry rules this client currently satisfies, with the
                # most recent trigger stamp.
                try:
                    from webapp import rule_models
                    context["rule_status"] = rule_models.account_rule_status(account)
                except Exception:
                    context["rule_status"] = []

    # In the beta, Latency and Toxic are top-level sections rather than two
    # panes of a rail, so the page is told which single pane to show and the
    # masthead is told which section to mark as current.
    af_pane = af if af in ("latency", "toxic") else ""
    beta_active = ("account" if tab == "account" else (af_pane or "latency")) \
        if BETA else ""

    return render(request, f"trading/{tab}.html", user=user, view="trading",
                  tabs=TRADING_TABS, active=tab,
                  af_only=af_pane if BETA else "", beta_active=beta_active,
                  **context)


@app.get("/quant/{tab}", response_class=HTMLResponse)
def quant(request: Request, tab: str, day: str = "", hedge: float | None = None,
          threshold: float | None = None, symbol: str = ""):
    user, redirect = _guard(request, auth.VIEW_QUANT)
    if redirect is not None:
        return redirect
    if tab not in {key for key, _, _ in QUANT_TABS}:
        return RedirectResponse("/quant/overview", status_code=303)

    view = auth.VIEW_QUANT
    context = _view_context(view, day or None, hedge, threshold)
    frame = context.pop("frame")
    config = context["config"]
    if context["has_data"]:
        if not day and context["days"]:
            context["selected_day"] = context["days"][0]
        if symbol:
            frame = frame.loc[frame["symbol"] == symbol]
        context["symbols"] = sorted(frame["symbol"].unique().tolist()) if "symbol" in frame else []
        context["selected_symbol"] = symbol
        if tab == "overview":
            context["curve"] = model_service.cached_equity_curves(
                view, frame, config.hedge_fraction, config.probability_threshold)
            context["summary"] = views.summary_stats(frame)
        elif tab == "signals":
            context["signals"] = views.trade_signals(
                frame, context["selected_day"], config.hedge_fraction, config.probability_threshold)
            context["summary"] = views.summary_stats(frame, context["selected_day"])

    if tab == "risk" and context.get("has_data"):
        # Historical risk for the chosen day, alongside the live stream. The two
        # are rendered separately and never blended: one is settled fact from
        # the warehouse, the other is only meaningful while the consumer is
        # genuinely connected.
        context["risk"] = risk_monitor.day_risk_report(
            frame, context["selected_day"], config.hedge_fraction, view)
        context["timeline"] = risk_monitor.exposure_timeline(frame, view=view)

    if tab == "summary" and context.get("has_data"):
        context["summary_report"] = views.executive_summary(
            frame, context["meta"], config.hedge_fraction)

    # `days` for the as-at selector already comes from _view_context.

    if tab == "surveillance" and context.get("has_data"):
        context["findings"] = views.surveillance_findings(frame)

    if tab == "validation":
        context["report"] = validation.report(view)

    if tab == "bookrisk":
        context["book"] = book_risk.report()

    if tab == "exits" and context.get("has_data"):
        # Scoped to the trades the model would actually COPY. Measuring exit
        # policies across all client flow answers a question nobody asked: the
        # desk only chooses an exit for trades it has taken on.
        context["exits"] = views.exit_policy_report(
            frame, config.hedge_fraction, config.probability_threshold)

    if tab == "copytrader":
        # The copy-trading screen works with no model artefact -- an operator
        # must be able to inspect limits, connectivity and the kill switch even
        # when nothing has been trained.
        context["copy_config"] = copytrader.load_config()
        context["copy_state"] = copytrader.state()
        context["account"] = copytrader.account_snapshot()
        context["positions"] = copytrader.open_positions()
        context["orders"] = copytrader.order_history(200)
        context["drift"] = copytrader.expected_vs_live(context["copy_config"])

    return render(request, f"quant/{tab}.html", user=user, view="quant",
                  tabs=QUANT_TABS, active=tab, **context)


@app.get("/api/quant/lab")
def api_quant_lab(request: Request):
    """Live walk-forward stream for the Strategy Lab tab: the running job's
    structured per-day records while training, else the last run's persisted
    curve so the page is never blank."""
    user, redirect = _guard(request, auth.VIEW_QUANT)
    if redirect is not None:
        return {"error": "auth"}
    view = auth.VIEW_QUANT
    job = model_service.job_state(view)
    meta = model_service.artifact_meta(view) or {}
    mmetrics = meta.get("metrics") or {}
    return {
        "status": job.status,
        "progress": round(float(job.progress or 0.0), 4),
        "message": job.message,
        "log": job.log[-25:],
        "live": list(job.metrics),
        "persisted": mmetrics.get("walk_curve") or [],
        "trained_at": meta.get("trained_at"),
        "roc_auc": mmetrics.get("roc_auc"),
        "history_days": (meta.get("config") or {}).get("history_days"),
        "per_class": mmetrics.get("per_class") or {},
        "calibrated_classes": mmetrics.get("calibrated_classes") or [],
        "path_coverage": mmetrics.get("path_coverage"),
        "path_models": {k: mmetrics[k] for k in
                        ("quant_mae_q50", "quant_mae_q80", "quant_exit_fe", "quant_perlot")
                        if mmetrics.get(k)},
        "rows": meta.get("rows"),
        # flat_bbook = B-booking everything (NO model: the baseline); by_fraction
        # = the model's routing at each hedge fraction (the model-dependent one).
        "current_flat_bbook": mmetrics.get("flat_bbook"),
        "current_by_fraction": mmetrics.get("by_fraction") or {},
        # archived predecessors (newest first) for the previous-vs-current panel
        "history": [{
            "dir": r["dir"], "trained_at": r["trained_at"], "rows": r["rows"],
            "history_days": r["config"].get("history_days"),
            "roc_auc": r["metrics"].get("roc_auc"),
            "flat_bbook": r["metrics"].get("flat_bbook"),
            "by_fraction": r["metrics"].get("by_fraction") or {},
            "per_class": r["metrics"].get("per_class") or {},
            "walk_curve": r["metrics"].get("walk_curve") or [],
            "note": r["note"],
        } for r in model_service.model_history(auth.VIEW_QUANT)],
    }


@app.post("/api/quant/lab/run")
def api_quant_lab_run(request: Request):
    """Kick a walk-forward retrain so the operator can watch it stream live."""
    user, redirect = _guard(request, auth.VIEW_QUANT)
    if redirect is not None:
        return {"error": "auth"}
    view = auth.VIEW_QUANT
    job = model_service.job_state(view)
    if job.status == "running":
        return {"status": "already running", "progress": job.progress}
    model_service.start_training(view, model_service.load_config(view))
    return {"status": "started"}


# --------------------------------------------------------------------------
# model configuration + explicit retrain
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Vantage copytrader -- admin only. Drives a PERSONAL external MT5 demo
# account, so it sits outside the view-permission system entirely: only an
# administrator ever sees or touches it.
# --------------------------------------------------------------------------
def _admin_only(request: Request):
    user = current_user(request)
    if user is None or not user.is_admin:
        return None, RedirectResponse("/login", status_code=303)
    return user, None


def _engine():
    """The Vantage copy-trading engine module, or None in an installation
    without it -- the engine lives in its own private repository, so a clone
    of the analytics app alone must still serve every other page."""
    try:
        from webapp import vantage
        return vantage
    except ImportError:
        return None


_ENGINE_MISSING = ("<h2>Vantage engine not installed</h2>"
                   "<p>The copy-trading engine is a separate private repository; "
                   "this installation runs the analytics app only.</p>")


@app.get("/vantage", response_class=HTMLResponse)
def vantage_page(request: Request):
    user, redirect = _admin_only(request)
    if redirect is not None:
        return redirect
    vantage = _engine()
    if vantage is None:
        return HTMLResponse(_ENGINE_MISSING, status_code=404)
    vantage.write_example()
    config = vantage.load_config()
    return render(request, "vantage.html", user=user, tabs=None, view=None,
                  config=config, engine=vantage.state(),
                  config_exists=vantage.CONFIG_PATH.exists(),
                  config_path=str(vantage.CONFIG_PATH),
                  has_model=vantage.booster() is not None,
                  report=vantage.report())


@app.post("/vantage/config")
def vantage_config(request: Request, login: str = Form(""), password: str = Form(""),
                   server: str = Form(""), terminal_path: str = Form(""),
                   mode: str = Form("paper"), risk_budget_fraction: float = Form(0.35),
                   copy_quantile: float = Form(0.90), invert_quantile: float = Form(0.10),
                   stop_multiple: float = Form(0.0), target_multiple: float = Form(0.0),
                   max_lots_per_trade: float = Form(1.0), max_open_positions: int = Form(20),
                   leverage: float = Form(30.0),
                   max_signal_age_minutes: float = Form(15.0)):
    from webapp import vantage
    user, redirect = _admin_only(request)
    if redirect is not None:
        return redirect
    config = vantage.load_config()
    # A browser autofilling the site's own username into this field must not
    # crash the save or replace the stored account number.
    if login.strip().isdigit():
        config.login = int(login.strip())
    if password.strip():          # blank means "keep the stored one"
        config.password = password.strip()
    if server.strip():
        config.server = server.strip()
    config.terminal_path = terminal_path.strip()
    config.mode = "live" if mode == "live" else "paper"
    config.risk_budget_fraction = max(0.01, min(0.90, risk_budget_fraction))
    config.copy_quantile = min(0.999, max(0.50, copy_quantile))
    config.invert_quantile = max(0.001, min(0.50, invert_quantile))
    config.stop_multiple = stop_multiple
    config.target_multiple = target_multiple
    config.max_lots_per_trade = max_lots_per_trade
    config.max_open_positions = max_open_positions
    config.leverage = max(1.0, leverage)
    config.max_signal_age_minutes = max(1.0, max_signal_age_minutes)
    vantage.save_config(config)
    return RedirectResponse("/vantage", status_code=303)


@app.post("/vantage/start")
def vantage_start(request: Request):
    from webapp import vantage
    user, redirect = _admin_only(request)
    if redirect is not None:
        return redirect
    vantage.start()
    return RedirectResponse("/vantage", status_code=303)


@app.post("/vantage/stop")
def vantage_stop(request: Request):
    from webapp import vantage
    user, redirect = _admin_only(request)
    if redirect is not None:
        return redirect
    vantage.stop()
    return RedirectResponse("/vantage", status_code=303)


@app.post("/api/vantage/reset_stats")
def vantage_reset_stats(request: Request):
    """Draw a fresh-start line at now: zero the strategy stats, order log and
    live counters (used when the account is re-funded)."""
    from webapp import vantage
    user, redirect = _admin_only(request)
    if redirect is not None:
        return {"error": "admin only"}
    return vantage.reset_stats()


@app.post("/vantage/kill")
def vantage_kill(request: Request, engage: str = Form("1")):
    from webapp import vantage
    user, redirect = _admin_only(request)
    if redirect is not None:
        return redirect
    vantage.set_kill_switch(engage == "1")
    return RedirectResponse("/vantage", status_code=303)


@app.post("/vantage/stream/start")
def vantage_stream_start(request: Request):
    """Restart the Kafka consumer without restarting the app.

    The materialiser starts at boot only when the cluster probe succeeds; a
    server restart during a network wobble leaves it stopped, and the tab then
    shows a stale stream with no way back short of bouncing the whole app.
    """
    user, redirect = _admin_only(request)
    if redirect is not None:
        return redirect
    try:
        from webapp import kafka_service
        # A TRUE restart: start() alone skips topics whose (possibly wedged)
        # threads are still alive, which made this button a silent no-op
        # after every VPN drop.
        kafka_service.MATERIALISER.restart(with_quotes=True)
    except Exception as error:
        from webapp import vantage
        vantage._log(f"stream restart failed: {type(error).__name__}: {error}")
    return RedirectResponse("/vantage", status_code=303)


@app.post("/vantage/strategy/toggle")
def vantage_strategy_toggle(request: Request, which: str = Form(...),
                            on: str = Form("1")):
    """Enable/disable a strategy independently. S1 (mirror) and S2 (fade) run
    off the same engine and account; either can be switched without touching
    the other, so they run alone or concurrently."""
    from webapp import vantage
    user, redirect = _admin_only(request)
    if redirect is not None:
        return redirect
    config = vantage.load_config()
    enabled = on == "1"
    if which == "s1":
        config.strategy1 = enabled
    elif which == "s2":
        config.strategy2 = enabled
    vantage.save_config(config)
    return RedirectResponse("/vantage", status_code=303)


@app.get("/api/vantage/diag")
def vantage_diag(request: Request, hours: float = 4.0):
    from webapp import vantage
    user, redirect = _admin_only(request)
    if redirect is not None:
        return {"error": "admin only"}
    return vantage.window_diag(hours)


@app.get("/api/vantage/reconcile")
def vantage_reconcile(request: Request, days: float = 30.0):
    from webapp import vantage
    user, redirect = _admin_only(request)
    if redirect is not None:
        return {"error": "admin only"}
    return vantage.account_reconcile(days)


@app.get("/api/vantage/s1_dump")
def vantage_s1_dump(request: Request, window_days: int = 45, strat: str = "s1"):
    from webapp import vantage
    user, redirect = _admin_only(request)
    if redirect is not None:
        return {"error": "admin only"}
    return vantage.s1_trades_dump(window_days, strat)


@app.get("/api/vantage/status")
def vantage_status(request: Request):
    from dataclasses import asdict as _asdict

    user, redirect = _admin_only(request)
    if redirect is not None:
        return {"error": "admin only"}
    vantage = _engine()
    if vantage is None:
        return {"error": "engine not installed"}
    engine = vantage.state()
    # The stream panel must show life whether or not the engine runs: running,
    # its own decisions stream through; idle, the newest trades are scored on
    # demand so "quiet" and "broken" never look the same.
    signals = engine.signals
    if not signals:
        try:
            signals = vantage.peek(vantage.load_config(), limit=10)
        except Exception:
            signals = []
    # Chronological, newest TRADED first: the loop processes ingestion batches,
    # whose internal order is not trade order.
    signals = sorted(signals, key=lambda s: s.get("traded") or "", reverse=True)
    config = vantage.load_config()
    return {"engine": _asdict(engine), "signals": signals[:10],
            "feed": {"mysql": bool(config.use_db_feed),
                     "kafka": bool(config.use_kafka_events)},
            "report": vantage.report()}


@app.post("/vantage/feed/toggle")
def vantage_feed_toggle(request: Request, source: str = Form(...),
                        on: str = Form("1")):
    """Switch a decision feed on/off (mysql poll vs kafka events) without
    editing yaml. The engine reads config at start, so this restarts it."""
    from webapp import vantage
    user, redirect = _admin_only(request)
    if redirect is not None:
        return redirect
    config = vantage.load_config()
    enabled = on == "1"
    if source == "mysql":
        config.use_db_feed = enabled
    elif source == "kafka":
        config.use_kafka_events = enabled
    vantage.save_config(config)
    if vantage.state().running:
        vantage.stop()
        vantage.start()
    return RedirectResponse("/vantage", status_code=303)


@app.post("/{view}/bookrisk/build")
def build_book_risk(request: Request, view: str):
    """Recompute the net-exposure sweep in the background.

    It reads the full trade history and prices every position in USD, so it
    cannot run inside a request.
    """
    user, redirect = _guard(request, view)
    if redirect is not None:
        return redirect
    book_risk.start_build()
    return RedirectResponse(f"/{view}/bookrisk", status_code=303)


@app.post("/{view}/validation/reconcile")
def reconcile_validation(request: Request, view: str):
    """Recompute the warehouse reconciliation in the background.

    It scans ~140M trades, so it cannot run inside a request. The page is
    redirected back immediately and picks the result up from disk once the
    thread finishes.
    """
    user, redirect = _guard(request, view)
    if redirect is not None:
        return redirect
    validation.start_reconciliation(view)
    return RedirectResponse(f"/{view}/validation", status_code=303)


@app.post("/{view}/config")
def update_config(request: Request, view: str, retrain: str = Form(""),
                  horizon_active_days: int = Form(5), sigma_threshold: float = Form(0.5),
                  min_train_days: int = Form(20), refit_cadence_days: int = Form(10),
                  n_estimators: int = Form(100), num_leaves: int = Form(31),
                  learning_rate: float = Form(0.1), subsample: float = Form(0.5),
                  colsample_bytree: float = Form(0.5), min_child_samples: int = Form(20),
                  hedge_fraction: float = Form(0.05), probability_threshold: float = Form(0.0)):
    """Save settings, and refit ONLY when Retrain was pressed.

    Saving alone leaves the existing artefact in place and flips the screen to a
    'settings changed' state. That separation is the whole point: nobody should
    trigger an hour of fitting by nudging a slider.
    """
    user, redirect = _guard(request, view)
    if redirect is not None:
        return redirect
    config = model_service.TrainingConfig(
        horizon_active_days=horizon_active_days, sigma_threshold=sigma_threshold,
        min_train_days=min_train_days, refit_cadence_days=refit_cadence_days,
        n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=learning_rate,
        subsample=subsample, colsample_bytree=colsample_bytree,
        min_child_samples=min_child_samples, hedge_fraction=hedge_fraction,
        probability_threshold=probability_threshold)
    model_service.save_config(view, config)
    if retrain:
        model_service.start_training(view, config)
    return RedirectResponse(f"/{view}/performance", status_code=303)


@app.get("/api/{view}/job")
def job_status(view: str):
    """Polled by the training panel so progress is visible without a reload.
    Path moved under /api: the old /{view}/job was SHADOWED by the generic
    /{view}/{tab} route, so the panel always saw the overview page instead of
    job state -- training failures were invisible."""
    state = model_service.job_state(view)
    return {"status": state.status, "message": state.message,
            "progress": state.progress, "log": state.log[-12:],
            "stale": model_service.is_stale(view)}


# --------------------------------------------------------------------------
# admin
# --------------------------------------------------------------------------
@app.get("/admin", response_class=HTMLResponse)
def admin_console(request: Request):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    if not user.is_admin:
        return render(request, "no_access.html", user=user, denied_view="admin")
    people = auth.list_users()
    meta = auth.setting_meta(IP_ALLOWLIST_KEY)
    return render(request, "admin.html", user=user, people=people,
                  pending=[p for p in people if not p.approved],
                  ok=request.query_params.get("ok", ""),
                  err=request.query_params.get("err", ""),
                  allowlist_raw=allowlist_raw(),
                  allowlist_count=len(current_allowlist()),
                  allowlist_source=("admin console" if meta
                                    else "AF_IP_ALLOWLIST environment"),
                  allowlist_by=meta.get("updated_by", ""),
                  client_ip=(request.client.host if request.client else ""),
                  sessions=auth.session_counts())


@app.post("/admin/approve")
def admin_approve(request: Request, user_id: int = Form(...), approved: str = Form("1")):
    user = current_user(request)
    if user is None or not user.is_admin:
        return RedirectResponse("/login", status_code=303)
    auth.set_approval(user_id, approved == "1")
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/restart")
def admin_restart(request: Request):
    """Restart THIS instance, from the browser.

    Python changes need a process restart, and until now that meant an
    Administrator shell and a remembered command line -- the wrong one takes the
    app down rather than restarting it. The work is delegated to the same
    start_*.ps1 the operator would run by hand, so there is exactly one restart
    procedure and the button cannot drift from it.

    The script TASKKILLs this very process, so the child must be DETACHED: a
    normal child dies with its parent, which here would mean the killer dying
    with the killed and the app never coming back. It also sleeps briefly first
    so this response reaches the browser before the socket goes away.
    """
    user, redirect = _admin_only(request)
    if redirect is not None:
        return redirect

    # REFUSE MID-WRITE. The warehouse top-up rewrites whole month partitions
    # every 15 minutes; killing it half-way is precisely how
    # mt4_live02/2026-09.parquet was truncated on 17 Sep. Atomic writes make a
    # torn file survivable now, but losing the fetch is still pointless.
    if _REFRESH_STATE.get("status") == "running":
        return RedirectResponse(
            "/admin?err=A+history+refresh+is+running+--+wait+for+it+to+finish",
            status_code=303)
    inflight = list((ROOT / "warehouse").glob("*/*.tmp"))
    if inflight:
        return RedirectResponse(
            "/admin?err=A+partition+write+is+in+flight+--+try+again+shortly",
            status_code=303)

    script = ROOT.parent / ("start_beta.ps1" if BETA else "start_dev.ps1")
    if not script.exists():
        return RedirectResponse(f"/admin?err=Missing+{script.name}", status_code=303)
    try:
        import subprocess
        flags = getattr(subprocess, "DETACHED_PROCESS", 0x00000008) | \
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-NonInteractive",
             "-ExecutionPolicy", "Bypass", "-Command",
             f"Start-Sleep -Seconds 3; & '{script}'"],
            cwd=str(ROOT.parent), creationflags=flags, close_fds=True,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
    except Exception as error:
        return RedirectResponse(
            f"/admin?err=Could+not+start+the+restart:+{type(error).__name__}",
            status_code=303)
    print(f"[restart] requested by {user.username} via /admin -> {script.name}",
          flush=True)
    return RedirectResponse(
        "/admin?ok=Restarting+now+--+this+page+will+fail+to+load+for+about+a+"
        "minute,+then+come+back.+The+copier+resumes+on+start.",
        status_code=303)


@app.post("/admin/permissions")
def admin_permissions(request: Request, user_id: int = Form(...), role: str = Form("user"),
                      views: list[str] = Form(default=[])):
    user = current_user(request)
    if user is None or not user.is_admin:
        return RedirectResponse("/login", status_code=303)
    auth.set_permissions(user_id, role, tuple(views))
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/delete")
def admin_delete(request: Request, user_id: int = Form(...)):
    user = current_user(request)
    if user is None or not user.is_admin:
        return RedirectResponse("/login", status_code=303)
    if user_id != user.id:      # never let an admin delete themselves out of the console
        auth.delete_user(user_id)
    return RedirectResponse("/admin", status_code=303)


def _admin_back(message: str, ok: bool = True) -> RedirectResponse:
    """Back to the console carrying a result. Every action says what it did --
    "password changed, 14 sessions signed out" is the useful confirmation, not
    a silently reloaded page."""
    from urllib.parse import urlencode
    return RedirectResponse("/admin?" + urlencode({"ok" if ok else "err": message}),
                            status_code=303)


@app.post("/admin/password")
def admin_password(request: Request, user_id: int = Form(...),
                   password: str = Form(...), keep_sessions: str = Form("")):
    """Set another user's password, or rotate your own."""
    user = current_user(request)
    if user is None or not user.is_admin:
        return RedirectResponse("/login", status_code=303)
    target = auth.get_user(user_id)
    if target is None:
        return _admin_back("No such user.", ok=False)
    # Changing your OWN password keeps your session by default, so an admin does
    # not sign themselves out mid-task; changing SOMEONE ELSE'S always drops
    # theirs, because that is normally the point of the change.
    keep = (user_id == user.id) if not keep_sessions else keep_sessions == "1"
    ok, message = auth.set_password(user_id, password, keep_sessions=keep)
    return _admin_back(f"{target.username}: {message}", ok=ok)


@app.post("/admin/rename")
def admin_rename(request: Request, user_id: int = Form(...), username: str = Form(...)):
    user = current_user(request)
    if user is None or not user.is_admin:
        return RedirectResponse("/login", status_code=303)
    ok, message = auth.rename_user(user_id, username)
    return _admin_back(message, ok=ok)


@app.post("/admin/revoke")
def admin_revoke(request: Request, user_id: int = Form(0), scope: str = Form("user")):
    """Sign one user, or everyone, out of every device."""
    user = current_user(request)
    if user is None or not user.is_admin:
        return RedirectResponse("/login", status_code=303)
    if scope == "all":
        n = auth.revoke_sessions(None)
        # The admin doing this is signed out too -- deliberately, because a
        # "sign everyone out" that quietly exempts the person clicking it is not
        # what it says. They land on the login page and sign back in.
        return _admin_back(f"Signed out every session ({n}). Please sign in again.")
    target = auth.get_user(user_id)
    if target is None:
        return _admin_back("No such user.", ok=False)
    n = auth.revoke_sessions(user_id)
    return _admin_back(f"{target.username}: signed out {n} session(s).")


@app.post("/admin/ip_allowlist")
def admin_ip_allowlist(request: Request, allowlist: str = Form(""), force: str = Form("")):
    """Edit the network allowlist from the console.

    REFUSES BY DEFAULT to save a list that would block the admin making the
    change: the whole failure mode of an IP allowlist is locking yourself out of
    the box you administer, and it is trivial to do by pasting the wrong range.
    `force` overrides, for the case where the admin knows they are editing on
    behalf of a network they are not currently on.
    """
    user = current_user(request)
    if user is None or not user.is_admin:
        return RedirectResponse("/login", status_code=303)
    raw = (allowlist or "").strip()
    nets = _parse_allowlist(raw)
    entries = [e for e in raw.replace(";", ",").split(",") if e.strip()]
    if len(nets) != len(entries):
        return _admin_back(
            f"{len(entries) - len(nets)} entry(ies) are not a valid IP or CIDR.", ok=False)
    host = request.client.host if request.client else ""
    if nets and force != "1" and not ip_allowed(host, nets):
        return _admin_back(
            f"Refused: that list would block you ({host}). Add a range covering "
            f"it, or tick Apply anyway.", ok=False)
    auth.set_setting(IP_ALLOWLIST_KEY, raw, by=user.username)
    _ALLOWLIST_CACHE["at"] = 0.0          # take effect on the next request
    if not raw:
        return _admin_back("Allowlist cleared - every source may now reach the app.")
    return _admin_back(f"Allowlist saved: {len(nets)} range(s), plus loopback.")


# --------------------------------------------------------------------------
# per-user data source
# --------------------------------------------------------------------------
@app.get("/settings", response_class=HTMLResponse)
def settings(request: Request, saved: str = ""):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    return render(request, "settings.html", user=user,
                  source=auth.get_data_source(user.id), saved=saved)


@app.post("/settings")
def save_settings(request: Request, kind: str = Form("mysql"), host: str = Form(""),
                  port: str = Form("3306"), username: str = Form(""),
                  password: str = Form(""), bq_project: str = Form("")):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    auth.save_data_source(user.id, kind=kind, host=host, port=port, username=username,
                          password=password, bq_project=bq_project)
    return RedirectResponse("/settings?saved=1", status_code=303)


# --------------------------------------------------------------------------
# live stream (Kafka)
# --------------------------------------------------------------------------
@app.get("/api/live/status")
def live_status(request: Request):
    if current_user(request) is None:
        return {"error": "unauthenticated"}
    from webapp import kafka_service
    status = kafka_service.MATERIALISER.status()
    status["reachable"] = kafka_service.probe_connectivity()
    return status


def _stream_is_stale(max_minutes: float = 15.0) -> bool:
    """Has the Kafka store seen a trade recently? If not, production MySQL
    (four seconds behind the trade servers) is the honest live source."""
    try:
        from webapp import vantage
        health = vantage.stream_health()
        age = health.get("trade_age_seconds")
        return age is None or age > max_minutes * 60
    except Exception:
        return True


@app.get("/api/live/activity")
def live_activity(request: Request, hours: int = 24):
    if current_user(request) is None:
        return {"error": "unauthenticated"}
    # Production MySQL when the stream is stale: this UAT Kafka cluster is
    # mostly idle, and a "live" panel showing days-old test flow while real
    # clients trade thousands of times an hour is worse than no panel.
    if _stream_is_stale():
        from webapp import trade_feed
        return _cached_live(f"activity_mysql_{hours}", lambda: {
            "buckets": trade_feed.recent_activity_mysql(hours),
            "exposure": trade_feed.top_exposure_mysql(),
            "source": "mysql-production (~4s lag)"}, ttl=30.0)
    from webapp import kafka_service
    return {"buckets": kafka_service.MATERIALISER.recent_activity(hours),
            "exposure": kafka_service.MATERIALISER.top_exposure(),
            "source": "kafka-stream"}


@app.get("/api/live/symbol_var")
def live_symbol_var(request: Request):
    """Per-symbol VaR at 1d / 5d / 20d, from the freshest live source."""
    if current_user(request) is None:
        return {"error": "unauthenticated"}
    if _stream_is_stale():
        from webapp import trade_feed
        return _cached_live("symbol_var_mysql", lambda: {
            "rows": trade_feed.symbol_var_mysql(),
            "note": glossary.term("var_multi_day"),
            "source": "mysql-production (~4s lag)"}, ttl=60.0)
    from webapp import kafka_service
    return {"rows": kafka_service.MATERIALISER.symbol_var(),
            "note": glossary.term("var_multi_day"), "source": "kafka-stream"}


@app.get("/api/live/ohlc")
def live_ohlc(request: Request, symbol: str, minutes: int = 5, hours: int = 24):
    """Price bars for one instrument, built from the quote stream."""
    if current_user(request) is None:
        return {"error": "unauthenticated"}
    from webapp import kafka_service
    return {"symbol": symbol, "minutes": minutes,
            "bars": kafka_service.MATERIALISER.ohlc(symbol, minutes, hours),
            "symbols": kafka_service.MATERIALISER.quote_symbols()}


#: Short-lived cache for endpoints that query all five MySQL servers. Several
#: users on one risk screen, each polling, would otherwise multiply into dozens
#: of position queries a minute against production databases. Positions do not
#: change meaningfully inside this window.
_LIVE_CACHE: dict[str, tuple[float, object]] = {}
_LIVE_TTL = 20.0


def _cached_live(key: str, builder, ttl: float = _LIVE_TTL):
    import time as _time

    entry = _LIVE_CACHE.get(key)
    if entry is not None and _time.time() - entry[0] < ttl:
        return entry[1]
    value = builder()
    _LIVE_CACHE[key] = (_time.time(), value)
    return value


@app.get("/api/exposure/live")
def exposure_live(request: Request, as_at: str = ""):
    """Exposure by instrument -- live now, or end-of-day on a chosen past date.

    `as_at` reconstructs the book from the warehouse: positions opened on or
    before that date and not yet closed. Without it, the live broker position
    tables are used.
    """
    if current_user(request) is None:
        return {"error": "unauthenticated"}
    if as_at:
        return _cached_live(f"exposure_asat_{as_at}",
                            lambda: _build_exposure_asat(as_at), ttl=600.0)
    return _cached_live("exposure_live", _build_exposure_live)


def _build_exposure_asat(as_at: str):
    """Historical end-of-day exposure, rebuilt from stored trades."""
    from datetime import timedelta

    import numpy as np
    import pandas as pd

    from webapp import data_store
    from webapp import exposure as exposure_module
    from webapp import mysql_extract, symbol_specs

    target = pd.Timestamp(as_at)
    # Lookback sized to the measured holding-time distribution, not to the worst
    # imaginable case: 97.8% of positions close within 24 hours and only 0.1%
    # survive past 30 days. A 400-day window read the entire store and took 211
    # seconds; 60 days captures essentially every open position at a fraction of
    # the I/O. Only the columns the calculation needs are read.
    trades = data_store.read_history(
        start=(target - timedelta(days=60)).to_pydatetime(),
        end=(target + timedelta(days=2)).to_pydatetime(),
        columns=["database", "login", "symbol", "cmd", "volume_lots",
                 "open_time", "close_time", "open_price"])
    if trades.empty:
        return {"rows": [], "alerts": [], "stress": [], "as_at": as_at,
                "errors": ["warehouse has no data for that window"],
                "total_net": 0.0, "total_gross": 0.0, "basis": "warehouse"}

    specs, rates = symbol_specs.load_all_specs(tuple(mysql_extract.MYSQL_DATABASES))
    by_symbol = exposure_module.exposure_as_at(trades, target, specs, rates)
    if by_symbol.empty:
        return {"rows": [], "alerts": [], "stress": [], "as_at": as_at,
                "errors": [], "total_net": 0.0, "total_gross": 0.0, "basis": "warehouse"}

    alerts = exposure_module.concentration_alerts(by_symbol.fillna(0))
    stress = exposure_module.stress_test(by_symbol.fillna(0))
    net_total = float(pd.to_numeric(by_symbol["net_notional"], errors="coerce").sum())
    gross_total = float(pd.to_numeric(by_symbol["gross_notional"], errors="coerce").sum())

    def clean(records):
        return [{k: (None if isinstance(v, float) and not np.isfinite(v)
                     else bool(v) if isinstance(v, np.bool_)
                     else v.item() if hasattr(v, "item") else v)
                 for k, v in row.items()} for row in records]

    return {
        "rows": clean(by_symbol.to_dict("records")),
        "alerts": clean(alerts),
        "stress": [{k: v for k, v in s.items() if k != "by_symbol"} for s in stress],
        "errors": [], "as_at": as_at, "basis": "warehouse (end of day)",
        "total_net": net_total if np.isfinite(net_total) else 0.0,
        "total_gross": gross_total if np.isfinite(gross_total) else 0.0,
    }


def _build_exposure_live():
    import pandas as pd

    from webapp import exposure as exposure_module
    from webapp import mysql_extract

    frames, errors = [], []
    for database in mysql_extract.MYSQL_DATABASES:
        try:
            frame = mysql_extract.open_positions(database)
            if not frame.empty:
                frames.append(views.add_canonical_symbol(frame))
        except Exception as error:
            errors.append(f"{database}: {type(error).__name__}")
    if not frames:
        return {"rows": [], "alerts": [], "stress": [], "errors": errors}

    import numpy as np

    from webapp import symbol_specs

    positions = pd.concat(frames, ignore_index=True)
    # Contract sizes and currency legs from the venues themselves, plus one FX
    # rate map built from their own live USD pairs. Without these, USD notional
    # is wrong for every USD-base pair and impossible for every cross.
    specs, rates = symbol_specs.load_all_specs(tuple(mysql_extract.MYSQL_DATABASES))
    by_symbol = exposure_module.exposure_by_symbol(positions, specs, rates)
    # Alerts and stress need the numeric frame; the response needs it
    # JSON-safe. Compute first, sanitise second.
    alerts = exposure_module.concentration_alerts(by_symbol.fillna(0))
    stress = exposure_module.stress_test(by_symbol.fillna(0))
    net_total = float(pd.to_numeric(by_symbol["net_notional"], errors="coerce").sum())
    gross_total = float(pd.to_numeric(by_symbol["gross_notional"], errors="coerce").sum())

    def clean(records):
        """Replace non-finite floats with None -- JSON has no NaN or Infinity."""
        return [{k: (None if isinstance(v, float) and not np.isfinite(v)
                     else bool(v) if isinstance(v, (np.bool_,))
                     else v.item() if hasattr(v, "item") else v)
                 for k, v in row.items()} for row in records]

    return {
        "rows": clean(by_symbol.to_dict("records")),
        "alerts": clean(alerts),
        "stress": [{k: v for k, v in s.items() if k != "by_symbol"} for s in stress],
        "errors": errors,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "total_net": net_total if np.isfinite(net_total) else 0.0,
        "total_gross": gross_total if np.isfinite(gross_total) else 0.0,
    }


@app.get("/api/exposure/history")
def exposure_history(request: Request, symbols: str = "", days: int = 90, freq: str = "D"):
    """Net notional over time, per instrument, from the local warehouse."""
    if current_user(request) is None:
        return {"error": "unauthenticated"}
    from datetime import timedelta

    from webapp import data_store
    from webapp import exposure as exposure_module

    end = datetime.now(timezone.utc)

    def build():
        # Column list matters: without one this read pulled EVERY column of
        # ninety days across six servers -- tens of millions of full-width rows
        # inside a request thread, which is precisely how this endpoint spent
        # ninety seconds timing out.
        trades = data_store.read_history(
            start=end - timedelta(days=days), end=end,
            columns=["database", "symbol", "cmd", "volume_lots",
                     "open_time", "close_time", "open_price", "net_profit"])
        if trades.empty:
            return None
        wanted_inner = tuple(s for s in symbols.split(",") if s) or None
        return exposure_module.historical_exposure(trades, wanted_inner, freq=freq)

    history = _cached_live(f"exposure_history_{symbols}_{days}_{freq}", build, ttl=900.0)
    if history is None:
        return {"series": [], "note": "warehouse not yet backfilled"}
    if history.empty:
        return {"series": []}

    series = []
    for symbol, group in history.groupby("symbol", observed=True):
        group = group.sort_values("bucket")
        series.append({
            "symbol": str(symbol),
            "points": [[b.strftime("%Y-%m-%d"), round(float(n), 0)]
                       for b, n in zip(group["bucket"], group["net_notional"])],
        })
    series.sort(key=lambda s: -abs(s["points"][-1][1] if s["points"] else 0))
    return {"series": series[:25]}


@app.get("/api/live/positions")
def live_positions(request: Request):
    """Open positions from the BROKER'S OWN tables, not inferred from the stream.

    A stream-derived view can only see positions opened inside the retained
    window; the server's position table is the whole book, which is what a risk
    desk needs. Falls back to the stream view if MySQL is unreachable, and says
    which source it used.
    """
    if current_user(request) is None:
        return {"error": "unauthenticated"}
    return _cached_live("live_positions", _build_live_positions)


def _build_live_positions():
    import pandas as pd

    from webapp import kafka_service, mysql_extract

    rows, errors, source = [], [], "mysql"
    for database in mysql_extract.MYSQL_DATABASES:
        try:
            frame = mysql_extract.open_positions(database)
            if frame.empty:
                continue
            frame["canonical"] = views.add_canonical_symbol(
                frame.rename(columns={"symbol": "symbol"}))["canonical_symbol"]
            grouped = frame.groupby(["database", "canonical"], observed=True).agg(
                accounts=("account_key", "nunique"),
                positions=("volume_lots", "size"),
                gross_lots=("volume_lots", lambda s: float(s.abs().sum())),
                net_lots=("volume_lots", "sum"),
                floating=("profit", "sum"),
            ).reset_index()
            rows.extend(grouped.to_dict("records"))
        except Exception as error:
            errors.append(f"{database}: {type(error).__name__}")
    if not rows:
        source = "kafka-stream"
        rows = kafka_service.MATERIALISER.open_positions()
    return {"source": source, "rows": rows, "errors": errors,
            "as_of": datetime.now(timezone.utc).isoformat()}


@app.post("/api/live/start")
def live_start(request: Request):
    user = current_user(request)
    if user is None or not user.is_admin:
        return {"error": "admin only"}
    from webapp import kafka_service
    # Quotes too: the OHLC panels need a price stream, and it is only consumed
    # when someone explicitly starts the consumer.
    kafka_service.MATERIALISER.start(backfill=True, with_quotes=True)
    return kafka_service.MATERIALISER.status()


@app.post("/api/live/stop")
def live_stop(request: Request):
    user = current_user(request)
    if user is None or not user.is_admin:
        return {"error": "admin only"}
    from webapp import kafka_service
    kafka_service.MATERIALISER.stop()
    return {"stopped": True}


# --------------------------------------------------------------------------
# copy trader
# --------------------------------------------------------------------------
@app.post("/quant/copytrader/config")
def copytrader_config(request: Request, login: str = Form(""), server: str = Form(""),
                      password: str = Form(""), size_multiplier: float = Form(0.1),
                      min_confidence: float = Form(0.7), max_lots_per_order: float = Form(1.0),
                      max_open_lots_per_symbol: float = Form(10.0),
                      max_total_open_lots: float = Form(50.0),
                      max_orders_per_minute: int = Form(30),
                      allowed_symbols: str = Form(""), paper_mode: str = Form(""),
                      arm: str = Form(""), test_connection: str = Form("")):
    user, redirect = _guard(request, auth.VIEW_QUANT)
    if redirect is not None:
        return redirect
    config = copytrader.load_config()
    if login.strip().isdigit():
        config.login = int(login.strip())
    config.server = server
    if password.strip():          # the page never echoes it; blank keeps the stored one
        config.password = password.strip()
    config.size_multiplier = size_multiplier
    config.min_confidence = min_confidence
    config.max_lots_per_order = max_lots_per_order
    config.max_open_lots_per_symbol = max_open_lots_per_symbol
    config.max_total_open_lots = max_total_open_lots
    config.max_orders_per_minute = max_orders_per_minute
    config.allowed_symbols = allowed_symbols
    # Paper mode is the default and must be switched OFF deliberately; arming is
    # then a second, separate action. Two steps, because one of them sends real
    # orders to a broker.
    config.paper_mode = bool(paper_mode)
    if arm and not config.paper_mode and user.is_admin:
        config.enabled = True
    if config.paper_mode:
        config.enabled = False
    copytrader.save_config(config)
    if test_connection:
        copytrader.connect(config)
    return RedirectResponse("/quant/copytrader", status_code=303)


@app.post("/quant/copytrader/dispatch")
def copytrader_dispatch(request: Request, day: str = Form(""), hedge: float = Form(0.05),
                        threshold: float = Form(0.0), limit: int = Form(50)):
    """Push the selected day's signals through the execution engine.

    Deliberately a manual, bounded action rather than a background loop. The
    model can emit thousands of signals for a single day; an operator pressing
    a button for an explicit batch is far easier to reason about -- and to stop
    -- than a daemon quietly trading. Every order still passes the full limit
    check, and paper mode still applies.
    """
    user, redirect = _guard(request, auth.VIEW_QUANT)
    if redirect is not None:
        return redirect
    frame = model_service.load_scores(auth.VIEW_QUANT)
    if frame is None or frame.empty:
        return RedirectResponse("/quant/copytrader", status_code=303)

    config = copytrader.load_config()
    signals = views.trade_signals(frame, day or None, hedge, threshold, limit=limit)
    for row in signals.itertuples():
        copytrader.submit(config, source_account=row.account_key, symbol=row.symbol,
                          direction=int(row.direction), client_lots=float(row.volume_lots),
                          confidence=float(row.score))
    return RedirectResponse("/quant/copytrader", status_code=303)


@app.post("/quant/copytrader/kill")
def copytrader_kill(request: Request, engage: str = Form("1")):
    user, redirect = _guard(request, auth.VIEW_QUANT)
    if redirect is not None:
        return redirect
    copytrader.set_kill_switch(engage == "1")
    return RedirectResponse("/quant/copytrader", status_code=303)


# --------------------------------------------------------------------------
# warehouse management
# --------------------------------------------------------------------------
_REFRESH_STATE = {"status": "idle", "message": "", "log": []}


@app.get("/api/cashflow")
def api_cashflow(request: Request, days: int = 180, dimension: str = "region"):
    """Deposits, withdrawals and transfers -- overall and by region/country/group.

    Cash movement is the strongest fraud signal in the dataset and is invisible
    to any model reading only trades: money in, a token trade, money out is a
    laundering shape that leaves no trading footprint.
    """
    if current_user(request) is None:
        return {"error": "unauthenticated"}
    return _cached_live(f"cashflow_{days}_{dimension}",
                        lambda: _build_cashflow(days, dimension), ttl=900.0)


def _build_cashflow(days: int, dimension: str):
    import numpy as np
    import pandas as pd

    from webapp import cashflow as cash
    from webapp import client_profile, mysql_extract

    errors = []
    # The LOCAL cashflow store first -- 4.9M movements read in ~1.5 seconds.
    # Pulling 180 days from MySQL live took over two minutes per page load,
    # which is how this endpoint timed out; MySQL remains the fallback for a
    # store that has never been backfilled.
    movements = pd.DataFrame()
    try:
        from webapp import cashflow_store
        movements = cashflow_store.read_cashflows(
            start=datetime.now(timezone.utc).replace(tzinfo=None) - pd.Timedelta(days=days))
    except Exception as error:
        errors.append(f"store: {type(error).__name__}")
    if movements.empty:
        frames = []
        for database in mysql_extract.MYSQL_DATABASES:
            try:
                frames.append(cash.load_cashflows(database, days=days))
            except Exception as error:
                errors.append(f"{database}: {type(error).__name__}")
        if not frames or all(f.empty for f in frames):
            return {"summary": {}, "by_dimension": [], "flags": [], "errors": errors}
        movements = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    profiles = client_profile.load_all_profiles(tuple(mysql_extract.MYSQL_DATABASES))
    summary = cash.summarise(movements)
    breakdown = cash.by_dimension(movements, profiles, dimension)
    features = cash.account_features(movements)
    flags = cash.fraud_flags(features)

    def clean(frame, limit=200):
        if frame is None or frame.empty:
            return []
        return [{k: (None if isinstance(v, float) and not np.isfinite(v)
                     else v.item() if hasattr(v, "item") else v)
                 for k, v in row.items()}
                for row in frame.head(limit).to_dict("records")]

    return {"summary": summary, "by_dimension": clean(breakdown, 60),
            "flags": clean(flags, 100), "errors": errors,
            "accounts_profiled": int(len(profiles)), "dimension": dimension}


@app.get("/api/regions")
def api_regions(request: Request, view: str = "trading"):
    """Firm P&L and accounts by client region.

    Region rather than server: a server says which machine an account lives on,
    which is plumbing. Region is the dimension a business question is actually
    asked in.
    """
    if current_user(request) is None:
        return {"error": "unauthenticated"}
    return _cached_live(f"regions_{view}", lambda: _build_regions(view), ttl=900.0)


def _build_regions(view: str):
    import numpy as np

    from webapp import client_profile, mysql_extract

    frame = model_service.load_scores(view)
    if frame is None or frame.empty:
        return {"rows": [], "note": "no model artefact"}
    profiles = client_profile.load_all_profiles(tuple(mysql_extract.MYSQL_DATABASES))
    if profiles.empty:
        return {"rows": [], "note": "client profiles unavailable (needs the VPN)"}

    enriched = client_profile.attach_region(frame, profiles)
    grouped = enriched.groupby("region", observed=True).agg(
        accounts=("account_key", "nunique"),
        rows=("pnl", "size"),
        client_pnl=("pnl", "sum"),
    ).reset_index()
    grouped["firm_pnl"] = -grouped["client_pnl"]
    grouped = grouped.sort_values("firm_pnl", ascending=False)

    by_country = enriched.groupby("country_code", observed=True).agg(
        accounts=("account_key", "nunique"), client_pnl=("pnl", "sum")).reset_index()
    by_country["firm_pnl"] = -by_country["client_pnl"]
    by_country = by_country.sort_values("firm_pnl", ascending=False).head(25)

    def clean(frame):
        return [{k: (None if isinstance(v, float) and not np.isfinite(v)
                     else v.item() if hasattr(v, "item") else v)
                 for k, v in row.items()} for row in frame.to_dict("records")]

    return {"rows": clean(grouped), "countries": clean(by_country),
            "profiled": int(len(profiles)), "note": ""}


@app.get("/executive", response_class=HTMLResponse)
def executive(request: Request, hedge: float = 0.05):
    """One page across both books, for someone who will not open nine tabs."""
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)

    trading = model_service.load_scores(auth.VIEW_TRADING)
    quant = model_service.load_scores(auth.VIEW_QUANT)
    report = views.ceo_dashboard(
        trading, quant,
        model_service.artifact_meta(auth.VIEW_TRADING),
        model_service.artifact_meta(auth.VIEW_QUANT),
        hedge)
    findings = views.surveillance_findings(trading) if trading is not None else None
    return render(request, "executive.html", user=user, report=report,
                  hedge=hedge, findings=findings,
                  stale_trading=model_service.is_stale(auth.VIEW_TRADING),
                  stale_quant=model_service.is_stale(auth.VIEW_QUANT))


@app.get("/data", response_class=HTMLResponse)
def data_console(request: Request):
    """What history is stored, how fresh it is, and how to extend it."""
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    from webapp import data_store

    return render(request, "data.html", user=user,
                  summary=data_store.store_summary(),
                  refresh=_REFRESH_STATE,
                  history_days=data_store.DEFAULT_HISTORY_DAYS,
                  overlap_days=data_store.OVERLAP_DAYS)


@app.post("/data/refresh")
def data_refresh(request: Request, database: str = Form(""), days: int = Form(0)):
    """Fetch new history incrementally.

    Incremental by default: each server is read from its own watermark minus a
    three-day overlap, which recovers amendments and late-closing positions
    without re-reading two years. `days` forces a longer window when a gap needs
    filling.
    """
    user = current_user(request)
    if user is None or not user.is_admin:
        return RedirectResponse("/login", status_code=303)
    if _REFRESH_STATE["status"] == "running":
        return RedirectResponse("/data", status_code=303)

    from webapp import data_store, dubai_backfill, mysql_extract

    def worker():
        _REFRESH_STATE.update(status="running", message="starting", log=[])
        targets = ([database] if database in mysql_extract.MYSQL_DATABASES
                   else list(mysql_extract.MYSQL_DATABASES))
        # The sixth live server is not in this MySQL instance -- it is
        # replicated to BigQuery instead -- so refreshing "everything" has to
        # reach for it explicitly. Leaving it out is what let a whole live
        # server go missing from every figure on the site without a trace.
        # Now that dubai has its own MariaDB entry it refreshes with the MySQL
        # targets above; BigQuery is only the fallback when it is not listed.
        want_dubai = (database in ("", dubai_backfill.DATABASE)
                      and dubai_backfill.DATABASE not in mysql_extract.MYSQL_DATABASES)
        try:
            for target in targets:
                plan = data_store.plan_refresh(target)
                window = days or max(
                    1, (datetime.now(timezone.utc) - plan.start).days + 1)
                _REFRESH_STATE["message"] = f"{target}: {plan.reason}"
                result = mysql_extract.backfill(
                    target, days=window,
                    progress=lambda m: _REFRESH_STATE["log"].append(m))
                _REFRESH_STATE["log"].append(
                    f"{target}: {result['rows']:,} rows, {len(result['failures'])} failed months")

            if want_dubai:
                # BigQuery bills by bytes scanned, so this one reports its cost.
                _REFRESH_STATE["message"] = f"{dubai_backfill.DATABASE}: BigQuery backfill"
                try:
                    plan = data_store.plan_refresh(dubai_backfill.DATABASE)
                    window = days or max(
                        1, (datetime.now(timezone.utc) - plan.start).days + 1)
                    result = dubai_backfill.backfill(
                        days=window,
                        progress=lambda m: _REFRESH_STATE["log"].append(m))
                    _REFRESH_STATE["log"].append(
                        f"{dubai_backfill.DATABASE}: {result['rows']:,} rows, "
                        f"{len(result['failures'])} failed months, "
                        f"${result['estimated_usd']:.2f} billed")
                except Exception as error:
                    # A BigQuery failure (expired credentials, most likely) must
                    # not discard the five servers that just refreshed fine.
                    _REFRESH_STATE["log"].append(
                        f"{dubai_backfill.DATABASE}: SKIPPED -- "
                        f"{type(error).__name__}: {str(error)[:120]}")

            _REFRESH_STATE.update(status="done", message="refresh complete")
        except Exception as error:
            _REFRESH_STATE.update(status="error",
                                  message=f"{type(error).__name__}: {error}")
        _REFRESH_STATE["log"] = _REFRESH_STATE["log"][-60:]

    threading.Thread(target=worker, daemon=True).start()
    return RedirectResponse("/data", status_code=303)


@app.get("/data/status")
def data_status(request: Request):
    if current_user(request) is None:
        return {"error": "unauthenticated"}
    from webapp import data_store
    return {**_REFRESH_STATE, "store": data_store.store_summary()}


def _quote_store_bars(symbol: str, start, end):
    """Minute OHLC from the live quote store (mid), canonical-matched --
    the price source for mt5 tickers and brand-new accounts."""
    import pandas as pd
    try:
        from webapp.trade_feed import _canonical
        from webapp.kafka_service import shared_cursor as _store
        canon = _canonical(symbol) or symbol
        with _store() as cx:
            frame = cx.execute("""
                WITH q AS (
                    SELECT event_time,
                           COALESCE(mid, (bid + ask) / 2) AS px
                    FROM quotes
                    WHERE (symbol = ? OR canonical = ?)
                      AND event_time BETWEEN ? AND ?)
                SELECT date_trunc('minute', event_time) AS minute,
                       first(px ORDER BY event_time)  AS open,
                       max(px)  AS high, min(px) AS low,
                       last(px ORDER BY event_time)   AS close,
                       count(*)  AS ticks
                FROM q WHERE px > 0
                GROUP BY 1 ORDER BY 1
            """, [symbol, canon, start, end]).df()
        frame["minute"] = pd.to_datetime(frame["minute"])
        return frame
    except Exception:
        return pd.DataFrame()


@app.get("/api/account/chart")
def account_chart(request: Request, account: str, symbol: str = "",
                  days: int = 3, minutes: int = 5):
    """Price path with this account's entries and exits marked.

    Reads minute bars from the tick table and overlays the account's own trades,
    so a reviewer can see WHERE the client bought and sold rather than only what
    it earned. That is the difference between "this account is profitable" and
    "this account consistently buys the low", which is the question surveillance
    actually asks.
    """
    user = current_user(request)
    if user is None:
        return {"error": "unauthenticated"}

    from datetime import timedelta

    import numpy as np
    import pandas as pd

    from webapp import tick_bars

    trades = views.account_trades(account, symbol or None, limit=400)
    if trades is None or trades.empty:
        return {"bars": [], "trades": [], "note": "no trades for this account"}

    trades = trades.copy()
    trades["open_time"] = pd.to_datetime(trades["open_time"])
    trades["close_time"] = pd.to_datetime(trades["close_time"])
    # Window the chart on the account's own most recent activity rather than on
    # today: a dormant account would otherwise render an empty chart.
    latest = trades["close_time"].max()
    if pd.isna(latest):
        latest = trades["open_time"].max()
    window_end = (latest + timedelta(hours=2)).to_pydatetime()
    window_start = (latest - timedelta(days=days)).to_pydatetime()

    focus = trades.loc[trades["open_time"] >= pd.Timestamp(window_start)]
    if focus.empty:
        focus = trades.head(40)
    raw_symbol = (symbol or focus["symbol"].mode().iloc[0]) if len(focus) else symbol
    if not raw_symbol:
        return {"bars": [], "trades": [], "note": "no symbol to chart"}

    database = account.split(":", 1)[0]
    note_suffix = ""
    try:
        bars = tick_bars.fetch_bars(database if database.startswith("mt4") else "mt4_live01",
                                    (str(raw_symbol),), window_start, window_end,
                                    chunk_hours=12, workers=4)
    except Exception:
        bars = pd.DataFrame()
    if bars.empty:
        # FALLBACK: the live quote store. The MySQL tick table only covers the
        # mt4 servers' tickers -- mt5 symbols and brand-new (single-day)
        # accounts land here, where 7-day tick retention covers them exactly.
        bars = _quote_store_bars(str(raw_symbol), window_start, window_end)
        note_suffix = " (live quote store)"
    if bars.empty:
        return {"bars": [], "trades": [],
                "note": f"no price history stored for {raw_symbol} in that "
                        f"window (tick table and quote store both empty)"}

    if minutes > 1:
        # Resample to the requested bar width. Aggregating high/low correctly
        # matters -- taking the last value would hide the excursion the chart
        # exists to show.
        bars = (bars.set_index("minute")
                    .resample(f"{minutes}min")
                    .agg({"open": "first", "high": "max", "low": "min",
                          "close": "last", "ticks": "sum"})
                    .dropna(subset=["close"]).reset_index())

    on_symbol = focus.loc[focus["symbol"].astype(str) == str(raw_symbol)]
    markers = []
    for row in on_symbol.itertuples():
        if pd.notna(row.open_time):
            markers.append({"time": row.open_time.strftime("%Y-%m-%d %H:%M"),
                            "price": float(row.open_price), "kind": "entry",
                            "side": str(row.cmd), "lots": float(row.volume_lots),
                            "pnl": float(row.net_profit) if pd.notna(row.net_profit) else None})
        if pd.notna(row.close_time) and pd.notna(row.close_price):
            markers.append({"time": row.close_time.strftime("%Y-%m-%d %H:%M"),
                            "price": float(row.close_price), "kind": "exit",
                            "side": str(row.cmd), "lots": float(row.volume_lots),
                            "pnl": float(row.net_profit) if pd.notna(row.net_profit) else None})

    return {
        "symbol": str(raw_symbol),
        "bars": [{"time": b.minute.strftime("%Y-%m-%d %H:%M"), "open": float(b.open),
                  "high": float(b.high), "low": float(b.low), "close": float(b.close)}
                 for b in bars.itertuples()],
        "trades": markers,
        # Coerce to str and drop nulls BEFORE sorting: a NaN mixed in with
        # strings makes `sorted` raise on the comparison, which surfaced as a
        # 500 on an endpoint whose actual work had already succeeded.
        "symbols": sorted({str(s) for s in trades["symbol"].dropna().tolist()
                           if str(s) not in ("nan", "<NA>", "None")}),
        "note": "",
    }


@app.get("/api/hub/movers")
def hub_movers(request: Request):
    """Recent biggest client wins — the flywheel's Client Zoom chips."""
    if current_user(request) is None:
        return {"error": "unauthenticated"}
    from webapp import replay
    try:
        board = replay.board("wins", minutes=720, limit=12)
        return {"movers": [{"account_key": c["account_key"], "symbol": c["symbol"],
                            "pnl": c["pnl"]} for c in board["cards"]]}
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}", "movers": []}


@app.get("/api/account/resolve")
def account_resolve(request: Request, q: str = ""):
    """Resolve a bare login (or full server:login) to account_key(s)."""
    if current_user(request) is None:
        return {"error": "unauthenticated"}
    from webapp import replay
    try:
        return replay.resolve_login(q)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}", "candidates": []}


@app.get("/api/replay/board")
def replay_board(request: Request, mode: str = "wins",
                 minutes: int = 1440, limit: int = 30):
    """The replay wall: biggest (mode=wins) or most suspicious client round-trips."""
    if current_user(request) is None:
        return {"error": "unauthenticated"}
    from webapp import replay
    mode = "suspicious" if mode == "suspicious" else "wins"
    minutes = max(30, min(int(minutes), 10080))
    try:
        return replay.board(mode, minutes=minutes, limit=max(1, min(int(limit), 60)))
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}", "cards": []}


@app.get("/api/replay/by_ticket")
def replay_by_ticket(request: Request, ticket: str):
    """Trade Zoom: resolve a trade ticket to its round-trip + tick chart."""
    if current_user(request) is None:
        return {"error": "unauthenticated"}
    from webapp import replay
    try:
        return replay.trade_by_ticket(ticket)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


@app.get("/api/replay/snapshot")
def replay_snapshot(request: Request, account: str):
    """Live account snapshot for an expanded replay card."""
    if current_user(request) is None:
        return {"error": "unauthenticated"}
    from webapp import replay
    try:
        return replay.account_snapshot(account)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


@app.get("/api/replay/account_events")
def replay_account_events(request: Request, account: str, start: str, end: str = ""):
    """The client's deals through the replay window, for the animated Live
    Account panel (balance/floating/exposure evolving with the playhead)."""
    if current_user(request) is None:
        return {"error": "unauthenticated"}
    from webapp import replay
    try:
        return replay.account_events(account, start, end)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}", "events": []}


@app.get("/api/replay/chart")
def replay_chart(request: Request, account: str, symbol: str,
                 open: str = "", close: str = ""):
    """Tick bars windowed tightly around ONE trade — the fast replay chart."""
    if current_user(request) is None:
        return {"error": "unauthenticated"}
    from webapp import replay
    try:
        return replay.trade_chart(account, symbol, open, close)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}", "bars": []}


@app.post("/api/ask")
def api_ask(request: Request, question: str = Form(...), view: str = Form("trading")):
    """Answer a question from the loaded data."""
    user = current_user(request)
    if user is None or not user.may_see(view):
        return {"error": "unauthorised"}
    from webapp import assistant

    frame = model_service.load_scores(view)
    result = assistant.answer(question, frame, model_service.artifact_meta(view), view)
    return {"text": result.text, "table": result.table, "columns": result.columns,
            "source": result.source, "followups": result.followups}


@app.post("/api/agent")
def api_agent(request: Request, question: str = Form(...),
              history: str = Form("[]"), request_id: str = Form("")):
    """The LLM operations agent -- available to any logged-in user. It is
    read-only (SELECT-gated) and scope-restricted to this app's data."""
    user = current_user(request)
    if user is None:
        return {"error": "unauthenticated"}
    from webapp import agent
    try:
        past = json.loads(history)
        if not isinstance(past, list):
            past = []
    except Exception:
        past = []
    try:
        return agent.run(question, past, request_id=request_id or None)
    except Exception as error:
        return {"text": f"agent error: {type(error).__name__}: {error}",
                "error": True}


@app.post("/api/agent/cancel")
def api_agent_cancel(request: Request, request_id: str = Form(...)):
    """Stop an in-flight agent request from eating resources."""
    if current_user(request) is None:
        return {"error": "unauthenticated"}
    from webapp import agent
    agent.request_cancel(request_id)
    return {"ok": True}


@app.post("/api/agent/key")
def api_agent_key(request: Request, key: str = Form(...)):
    user, redirect = _admin_only(request)
    if redirect is not None:
        return {"error": "admin only"}
    from webapp import agent
    agent.save_key(key)
    return {"ok": True, "configured": agent.api_key() is not None,
            "needs_workspace": agent.api_key() is not None and agent.workspace_id() is None}


@app.post("/api/agent/workspace")
def api_agent_workspace(request: Request, workspace: str = Form(...)):
    user, redirect = _admin_only(request)
    if redirect is not None:
        return {"error": "admin only"}
    from webapp import agent
    agent.save_workspace(workspace)
    return {"ok": True, "workspace_set": agent.workspace_id() is not None}


@app.get("/api/agent/health")
def api_agent_health(request: Request):
    user = current_user(request)
    if user is None:
        return {"error": "unauthenticated"}
    from webapp import agent
    has_key = agent.api_key() is not None
    # key management stays admin-only; other users just use the configured key
    return {"configured": has_key, "model": agent.MODEL,
            "can_configure": user.is_admin,
            "needs_workspace": has_key and agent.workspace_id() is None}


@app.get("/agent/export/{name}")
def agent_export(request: Request, name: str):
    from fastapi.responses import FileResponse
    if current_user(request) is None:
        return {"error": "unauthenticated"}
    from webapp import agent
    # basename-only: no traversal out of the exports directory
    safe = Path(name).name
    path = agent.EXPORTS / safe
    if not path.exists() or not safe.endswith(".csv"):
        return {"error": "no such export"}
    return FileResponse(path, media_type="text/csv", filename=safe)


@app.get("/health")
def health():
    return {"status": "ok"}

