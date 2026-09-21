"""Surveillance: the patterns an anti-fraud desk needs to see, with evidence.

Separate from the routing taxonomy on purpose. Routing asks "is this account
worth hedging"; surveillance asks "is this account doing something the firm
should investigate". An arbitrageur and a persistently skilled client may both
warrant an A-book, but only one of them warrants a phone call.

EVERY FLAG CARRIES ITS EVIDENCE

A surveillance alert is an accusation. Each detection therefore states the
measurement that triggered it, the threshold it crossed, and how many
observations support it -- so a reviewer can judge the claim rather than trust
it. Thresholds are set to be defensible: an earlier version of the arbitrage
rule fired on any positive reading and tagged 137,000 account-days, which is
noise dressed as a finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Minimum closed trades before any behavioural claim is made about an account.
#: Below this, a "pattern" is a small sample.
MIN_TRADES_FOR_CLAIM = 50

DETECTIONS: dict[str, dict] = {
    "LATENCY_ARBITRAGE": {
        "severity": "critical",
        "what": "Fills consistently precede favourable moves",
        "why": ("Entries land just before the market moves their way, with no "
                "matching prior run-up. That is a speed or feed advantage rather "
                "than a view, and it is not a risk the firm is being paid for."),
        "action": "Review feed latency and consider routing to an LP.",
    },
    "TOXIC_FLOW": {
        "severity": "critical",
        "what": "Systematically adverse after entry",
        "why": ("The market moves against the firm at nearly every horizon after "
                "this account trades -- the firm is on the wrong side by "
                "construction, not by chance."),
        "action": "Hedge and investigate the source of the flow.",
    },
    "MARTINGALE": {
        "severity": "warning",
        "what": "Doubles down into losses",
        "why": ("Position size rises after losing trades. The pattern produces a "
                "long run of small wins and one catastrophic loss, so the account "
                "looks profitable until the day it is not."),
        "action": "Monitor margin headroom; this fails suddenly rather than gradually.",
    },
    "RAPID_FIRE": {
        "severity": "warning",
        "what": "Very high trade frequency at minimal hold",
        "why": ("Hundreds of trades held for seconds. Execution-sensitive and "
                "often latency-driven; also a common shape for abusive "
                "bonus-clearing."),
        "action": "Check execution quality and any bonus eligibility.",
    },
    "SIZE_ANOMALY": {
        "severity": "warning",
        "what": "Sudden step-change in position size",
        "why": ("Typical size jumped by an order of magnitude. Either the account "
                "was funded materially, or it is being used differently -- both "
                "change the risk the firm carries."),
        "action": "Confirm the funding source and re-assess limits.",
    },
    "MIRRORED_ACCOUNTS": {
        "severity": "critical",
        "what": "Trades in lockstep with another account",
        "why": ("Near-identical entries and exits across accounts suggest one "
                "operator behind several logins -- which defeats per-account "
                "limits and can indicate bonus abuse or coordinated arbitrage."),
        "action": "Compare KYC, funding sources and IP where available.",
    },
}


def detect(frame: pd.DataFrame, markouts: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run every detection over an account-level summary.

    `frame` is one row per account with behavioural aggregates; `markouts` adds
    the adverse-selection tests when available. Returns one row per detection
    with the evidence that produced it.
    """
    if frame.empty:
        return pd.DataFrame()

    findings: list[dict] = []

    def _count(value) -> int:
        """Trade counts arrive as NaN for accounts with no lifetime record.
        `int(nan)` raises, so an unknown count is zero -- which also means the
        row is skipped by the minimum-trades guard rather than being flagged on
        no evidence at all."""
        try:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return 0
            return int(value)
        except (TypeError, ValueError):
            return 0

    def add(row, code, evidence, value):
        detail = DETECTIONS[code]
        findings.append({
            "account_key": row.account_key,
            "code": code,
            "severity": detail["severity"],
            "what": detail["what"],
            "why": detail["why"],
            "action": detail["action"],
            "evidence": evidence,
            "value": float(value),
            "trades": _count(getattr(row, "trades", 0)),
        })

    for row in frame.itertuples():
        trades = _count(getattr(row, "trades", 0))
        if trades < MIN_TRADES_FOR_CLAIM:
            continue

        martingale = getattr(row, "martingale_rate", np.nan)
        if pd.notna(martingale) and martingale >= 0.20:
            add(row, "MARTINGALE",
                f"Size increased after a loss on {martingale:.0%} of {trades:,} trades",
                martingale)

        scalp = getattr(row, "scalp_rate", np.nan)
        if pd.notna(scalp) and scalp >= 0.70 and trades >= 200:
            add(row, "RAPID_FIRE",
                f"{scalp:.0%} of {trades:,} trades held under five minutes", scalp)

        jump = getattr(row, "size_jump", np.nan)
        if pd.notna(jump) and jump >= 10:
            add(row, "SIZE_ANOMALY",
                f"Recent median size is {jump:.0f}x the account's historical median", jump)

    if markouts is not None and not markouts.empty:
        merged = markouts.copy()
        for row in merged.itertuples():
            anticipation = getattr(row, "anticipation_5m", np.nan)
            share = getattr(row, "markout_positive_share", np.nan)
            context = getattr(row, "context_trades", 0) or 0
            if (pd.notna(anticipation) and anticipation >= 0.0002
                    and pd.notna(share) and share >= 0.60 and context >= MIN_TRADES_FOR_CLAIM):
                findings.append({
                    "account_key": row.account_key, "code": "LATENCY_ARBITRAGE",
                    "severity": DETECTIONS["LATENCY_ARBITRAGE"]["severity"],
                    "what": DETECTIONS["LATENCY_ARBITRAGE"]["what"],
                    "why": DETECTIONS["LATENCY_ARBITRAGE"]["why"],
                    "action": DETECTIONS["LATENCY_ARBITRAGE"]["action"],
                    "evidence": (f"Anticipation {anticipation:+.5f} with the market "
                                 f"favourable at {share:.0%} of horizons across "
                                 f"{int(context):,} trades"),
                    "value": float(anticipation), "trades": int(context),
                })
            elif pd.notna(share) and share >= 0.80 and context >= MIN_TRADES_FOR_CLAIM:
                findings.append({
                    "account_key": row.account_key, "code": "TOXIC_FLOW",
                    "severity": DETECTIONS["TOXIC_FLOW"]["severity"],
                    "what": DETECTIONS["TOXIC_FLOW"]["what"],
                    "why": DETECTIONS["TOXIC_FLOW"]["why"],
                    "action": DETECTIONS["TOXIC_FLOW"]["action"],
                    "evidence": (f"Market favourable to the client at {share:.0%} of "
                                 f"horizons across {int(context):,} trades"),
                    "value": float(share), "trades": int(context),
                })

    if not findings:
        return pd.DataFrame()
    result = pd.DataFrame(findings)
    order = {"critical": 0, "warning": 1, "info": 2}
    result["_rank"] = result["severity"].map(order)
    return result.sort_values(["_rank", "value"], ascending=[True, False]).drop(columns="_rank")


def find_mirrored_accounts(trades: pd.DataFrame, tolerance_seconds: int = 5,
                           min_matches: int = 30) -> pd.DataFrame:
    """Accounts whose trades coincide too closely to be independent.

    Two accounts entering the same instrument in the same direction within
    seconds, repeatedly, are almost certainly one operator. That matters because
    per-account limits stop working, and it is a common shape for bonus abuse.

    Matched on (symbol, direction, entry second) -- deliberately coarse. A
    tighter join would miss deliberately jittered copying; a looser one would
    flag every account trading a news release.
    """
    if trades.empty or len(trades) > 5_000_000:
        return pd.DataFrame()

    working = trades[["account_key", "canonical_symbol", "cmd", "open_time"]].copy()
    working["bucket"] = (pd.to_datetime(working["open_time"]).astype("int64")
                         // (tolerance_seconds * 1_000_000_000))
    keyed = working.groupby(["canonical_symbol", "cmd", "bucket"], observed=True)["account_key"] \
                   .apply(lambda s: sorted(set(s)))
    pairs: dict[tuple[str, str], int] = {}
    for accounts in keyed:
        if len(accounts) < 2 or len(accounts) > 12:
            continue    # a very wide cluster is a news event, not collusion
        for i, left in enumerate(accounts):
            for right in accounts[i + 1:]:
                pairs[(left, right)] = pairs.get((left, right), 0) + 1

    rows = [{"account_a": a, "account_b": b, "coincidences": n}
            for (a, b), n in pairs.items() if n >= min_matches]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("coincidences", ascending=False).reset_index(drop=True)
