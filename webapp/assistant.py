"""Question console: plain-language questions answered from the actual data.

WHAT THIS IS, AND WHAT IT IS NOT

This resolves questions against the warehouse and the live feed and answers with
real figures. It is NOT a language model: there is no API key wired in, and
inventing one would mean either shipping a broken feature or quietly sending
firm P&L to a third party.

That trade is deliberate rather than a limitation. On a risk desk a
deterministic answer that always cites its source beats a fluent one that is
occasionally wrong, and every response here names the rows behind it. If an LLM
key is later configured, `interpret()` is the single seam to extend -- it maps
text to an INTENT, and the handlers below stay exactly as they are.

Intents are matched on explicit patterns. An unrecognised question says so and
lists what it can answer, rather than guessing and returning a confident number
to a question nobody asked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Answer:
    text: str
    table: list[dict] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    source: str = ""
    followups: list[str] = field(default_factory=list)


EXAMPLES = [
    "top 10 accounts by profit",
    "biggest losers this week",
    "what is our exposure to gold",
    "net exposure by symbol",
    "which accounts should we A-book today",
    "worst day in the last 90 days",
    "show account mt4_live02:2395433",
    "how much did XAUUSD make",
    "surveillance findings",
    "what is VaR",
    "how many accounts are active",
    "model performance",
]


def interpret(question: str) -> tuple[str, dict]:
    """Map a question to an intent and its parameters.

    Deliberately explicit patterns rather than fuzzy matching: a wrong intent
    returns a confident answer to a different question, which is worse than
    admitting the question was not understood.
    """
    text = (question or "").strip().lower()
    if not text:
        return "help", {}

    account = re.search(r"(mt[45]_[a-z0-9_]+:\d+)", question, re.IGNORECASE)
    if account:
        return "account", {"account_key": account.group(1)}

    symbol = re.search(r"\b(xauusd|gold|eurusd|gbpusd|usdjpy|btcusd|ethusd|xagusd|"
                       r"silver|ustec|nas100|us30|us500|usoil|oil)\b", text)
    aliases = {"gold": "XAUUSD", "silver": "XAGUSD", "oil": "USOIL", "nas100": "USTEC"}

    if any(k in text for k in ("var", "value at risk", "expected shortfall", "tail")):
        return "var", {}
    if any(k in text for k in ("surveillance", "fraud", "arbitrage", "toxic", "suspicious")):
        return "surveillance", {}
    if any(k in text for k in ("a-book", "abook", "hedge today", "route today", "manifest")):
        return "abook", {}
    if any(k in text for k in ("exposure", "notional", "position", "carrying")):
        return "exposure", {"symbol": aliases.get(symbol.group(1), symbol.group(1).upper())
                            if symbol else None}
    if any(k in text for k in ("model", "auc", "accuracy", "performance", "uplift")):
        return "model", {}
    if any(k in text for k in ("worst day", "best day", "biggest loss", "biggest day")):
        return "extremes", {"best": "best" in text}
    if any(k in text for k in ("active", "how many account", "count of account")):
        return "activity", {}
    if symbol:
        return "symbol", {"symbol": aliases.get(symbol.group(1), symbol.group(1).upper())}
    if any(k in text for k in ("top", "best", "most profitable", "biggest winner")):
        return "top_accounts", {"losers": False, "limit": _limit(text)}
    if any(k in text for k in ("loser", "worst account", "losing")):
        return "top_accounts", {"losers": True, "limit": _limit(text)}
    return "help", {}


def _limit(text: str, default: int = 10) -> int:
    match = re.search(r"\b(\d{1,3})\b", text)
    return min(100, max(1, int(match.group(1)))) if match else default


def answer(question: str, frame: pd.DataFrame | None, meta: dict | None,
           view: str = "trading") -> Answer:
    intent, params = interpret(question)
    if frame is None or frame.empty:
        return Answer("No model artefact is loaded, so I cannot answer from data yet.",
                      source="none")

    handlers = {
        "top_accounts": _top_accounts, "account": _account, "symbol": _symbol,
        "exposure": _exposure, "var": _var, "abook": _abook,
        "surveillance": _surveillance, "model": _model, "extremes": _extremes,
        "activity": _activity,
    }
    handler = handlers.get(intent)
    if handler is None:
        return Answer(
            "I did not recognise that question. I answer from the loaded data "
            "rather than guessing, so here is what I can currently resolve:",
            followups=EXAMPLES, source="none")
    try:
        return handler(frame, meta, params, view)
    except Exception as error:
        return Answer(f"That question failed while querying: {type(error).__name__}: {error}",
                      source="error")


# ---------------------------------------------------------------------------
def _top_accounts(frame, meta, params, view):
    losers = params.get("losers", False)
    limit = params.get("limit", 10)
    from webapp import model_service
    totals = model_service.account_totals(view, frame)
    ranked = totals.tail(limit)[::-1] if losers else totals.head(limit)
    rows = [{"account": str(k), "firm_pnl": round(float(v), 2)} for k, v in ranked.items()]
    label = "cost the firm most" if losers else "earned the firm most"
    return Answer(
        f"The {len(rows)} accounts that {label} over the loaded history "
        f"({len(frame):,} account-rows). Firm P&L is the negative of the client's.",
        table=rows, columns=["account", "firm_pnl"],
        source=f"{view} artefact",
        followups=[f"show account {rows[0]['account']}"] if rows else [])


def _account(frame, meta, params, view):
    key = params["account_key"]
    subset = frame.loc[frame["account_key"] == key]
    if subset.empty:
        return Answer(f"No rows for {key} in the loaded history.", source="none")
    pnl = subset["pnl"].sum()
    rows = [{
        "metric": "rows", "value": f"{len(subset):,}"}, {
        "metric": "client P&L", "value": f"${pnl:,.2f}"}, {
        "metric": "firm P&L", "value": f"${-pnl:,.2f}"}, {
        "metric": "win rate", "value": f"{(subset['pnl'] > 0).mean():.1%}"}, {
        "metric": "latest model score", "value": f"{subset['score'].iloc[-1]:.1%}"}, {
        "metric": "first seen", "value": str(pd.to_datetime(subset['day']).min().date())}, {
        "metric": "last seen", "value": str(pd.to_datetime(subset['day']).max().date())},
    ]
    verdict = ("profitable for the firm to B-book" if pnl < 0
               else "costing the firm money -- a hedging candidate")
    return Answer(f"{key} is {verdict}.", table=rows, columns=["metric", "value"],
                  source=f"{view} artefact",
                  followups=[f"surveillance findings"])


def _symbol(frame, meta, params, view):
    symbol = params["symbol"]
    column = "canonical_symbol" if "canonical_symbol" in frame.columns else "symbol"
    if column not in frame.columns:
        return Answer("This artefact has no symbol column -- it is account-level, "
                      "not trade-level. Ask on the Quant view instead.", source="none")
    subset = frame.loc[frame[column] == symbol]
    if subset.empty:
        return Answer(f"No {symbol} rows in the loaded history.", source="none")
    pnl = subset["pnl"].sum()
    rows = [{"metric": "trades", "value": f"{len(subset):,}"},
            {"metric": "accounts", "value": f"{subset['account_key'].nunique():,}"},
            {"metric": "client P&L", "value": f"${pnl:,.0f}"},
            {"metric": "firm P&L", "value": f"${-pnl:,.0f}"},
            {"metric": "win rate", "value": f"{(subset['pnl'] > 0).mean():.1%}"}]
    return Answer(f"{symbol} over the loaded history.", table=rows,
                  columns=["metric", "value"], source=f"{view} artefact",
                  followups=[f"what is our exposure to {symbol}"])


def _exposure(frame, meta, params, view):
    from webapp import exposure as exposure_module
    from webapp import mysql_extract, symbol_specs, views as views_module

    frames = []
    for database in mysql_extract.MYSQL_DATABASES:
        try:
            positions = mysql_extract.open_positions(database)
            if not positions.empty:
                frames.append(views_module.add_canonical_symbol(positions))
        except Exception:
            continue
    if not frames:
        return Answer("No live positions available -- the servers are unreachable "
                      "(this needs the VPN).", source="mysql")
    specs, rates = symbol_specs.load_all_specs(tuple(mysql_extract.MYSQL_DATABASES))
    table = exposure_module.exposure_by_symbol(pd.concat(frames, ignore_index=True),
                                              specs, rates)
    wanted = params.get("symbol")
    if wanted:
        table = table.loc[table["canonical_symbol"] == wanted]
        if table.empty:
            return Answer(f"No open positions in {wanted}.", source="broker position tables")
    rows = [{"symbol": r.canonical_symbol,
             "net_usd": None if r.net_notional is None else round(float(r.net_notional)),
             "gross_usd": None if r.gross_notional is None else round(float(r.gross_notional)),
             "accounts": int(r.accounts), "basis": r.notional_status}
            for r in table.head(15).itertuples()]
    net = pd.to_numeric(table["net_notional"], errors="coerce").sum()
    return Answer(
        f"Live net exposure {'in ' + wanted if wanted else 'across all instruments'} is "
        f"${net:,.0f} USD. Net is direction-weighted; gross ignores side.",
        table=rows, columns=["symbol", "net_usd", "gross_usd", "accounts", "basis"],
        source="broker position tables (live)",
        followups=["net exposure by symbol", "what is VaR"])


def _var(frame, meta, params, view):
    from webapp import model_service, risk_monitor
    series = model_service.daily_series(view, frame)
    stats = risk_monitor.value_at_risk(series)
    rows = [{"metric": "days observed", "value": f"{stats['days']:,}"},
            {"metric": "VaR 95%", "value": f"${stats['var_95']:,.0f}"},
            {"metric": "expected shortfall 95%", "value": f"${stats['es_95']:,.0f}"},
            {"metric": "VaR 99%", "value": f"${stats['var_99']:,.0f}"},
            {"metric": "expected shortfall 99%", "value": f"${stats['es_99']:,.0f}"},
            {"metric": "worst day", "value": f"${stats['worst_day']:,.0f}"},
            {"metric": "losing days", "value": f"{stats['losing_day_share']:.0%}"}]
    return Answer(
        "Historical VaR, not parametric -- these are empirical quantiles of the "
        "actual daily series. Client P&L is heavy-tailed, so a normal fit would "
        "understate exactly the days that matter.",
        table=rows, columns=["metric", "value"], source=f"{view} artefact",
        followups=["worst day in the last 90 days"])


def _abook(frame, meta, params, view):
    from webapp import model_service, views as views_module
    config = model_service.load_config(view)
    days = views_module.available_days(frame)
    if not days:
        return Answer("No days available in the artefact.", source="none")
    manifest = views_module.abook_manifest(frame, days[0], config.hedge_fraction)
    if manifest.empty:
        return Answer(f"No account met the threshold on {days[0]}.", source=f"{view} artefact")
    rows = [{"account": r.account_key, "category": r.category,
             "confidence": f"{r.score:.0%}",
             "expected_impact": round(float(r.expected_impact))}
            for r in manifest.head(15).itertuples()]
    return Answer(
        f"{len(manifest)} accounts to A-book on {days[0]} at a "
        f"{config.hedge_fraction:.0%} hedge, ranked by expected dollar impact.",
        table=rows, columns=["account", "category", "confidence", "expected_impact"],
        source=f"{view} artefact",
        followups=[f"show account {rows[0]['account']}"] if rows else [])


def _surveillance(frame, meta, params, view):
    from webapp import views as views_module
    findings = views_module.surveillance_findings(frame)
    if findings is None or findings.empty:
        return Answer("No account crossed a surveillance threshold. Thresholds are "
                      "set to be defensible rather than generous.", source=f"{view} artefact")
    rows = [{"account": r.account_key, "severity": r.severity,
             "pattern": r.what, "evidence": r.evidence}
            for r in findings.head(12).itertuples()]
    critical = int((findings["severity"] == "critical").sum())
    return Answer(
        f"{len(findings)} findings across {findings['account_key'].nunique()} accounts, "
        f"{critical} critical. Each carries the measurement that triggered it.",
        table=rows, columns=["account", "severity", "pattern", "evidence"],
        source=f"{view} artefact + markouts")


def _model(frame, meta, params, view):
    metrics = (meta or {}).get("metrics", {})
    if not metrics:
        return Answer("No trained model metadata yet.", source="none")
    flat = metrics.get("flat_bbook", {})
    rows = [{"policy": "flat B-book",
             "profit": round(flat.get("total_pnl_usd", 0)),
             "max_drawdown": round(flat.get("max_drawdown_usd", 0)),
             "sharpe": round(flat.get("sharpe", 0), 2)}]
    for key, value in metrics.get("by_fraction", {}).items():
        rows.append({"policy": f"hedge {float(key):.0%}",
                     "profit": round(value.get("total_pnl_usd", 0)),
                     "max_drawdown": round(value.get("max_drawdown_usd", 0)),
                     "sharpe": round(value.get("sharpe", 0), 2)})
    auc = metrics.get("roc_auc")
    return Answer(
        f"Walk-forward out-of-sample ROC AUC {auc:.4f}. "
        f"Not yet validated on a held-out period, so treat the direction as "
        f"established and the magnitude as provisional." if auc else "Model metrics:",
        table=rows, columns=["policy", "profit", "max_drawdown", "sharpe"],
        source=f"{view} artefact metadata")


def _extremes(frame, meta, params, view):
    from webapp import model_service
    series = model_service.daily_series(view, frame)
    best = params.get("best", False)
    ranked = series.nlargest(10) if best else series.nsmallest(10)
    rows = [{"day": str(pd.Timestamp(k).date()), "firm_pnl": round(float(v))}
            for k, v in ranked.items()]
    return Answer(f"The {'best' if best else 'worst'} ten days for the firm.",
                  table=rows, columns=["day", "firm_pnl"], source=f"{view} artefact")


def _activity(frame, meta, params, view):
    from webapp import views as views_module
    days = views_module.available_days(frame)
    latest = views_module.day_slice(frame, days[0]) if days else frame.iloc[0:0]
    population = len(views_module.eligible_population(frame, days[0])) if days else 0
    rows = [{"metric": "accounts in history", "value": f"{frame['account_key'].nunique():,}"},
            {"metric": f"active on {days[0] if days else '--'}",
             "value": f"{latest['account_key'].nunique():,}"},
            {"metric": "routable (carried forward)", "value": f"{population:,}"},
            {"metric": "days of history", "value": f"{len(days):,}"}]
    return Answer(
        "'Routable' is wider than 'active': a routing decision is made before the "
        "session, so every account with recent data is a candidate whether or not "
        "it ends up trading.",
        table=rows, columns=["metric", "value"], source=f"{view} artefact")
