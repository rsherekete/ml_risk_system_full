"""Definitions for terms the screens use, surfaced as hover tooltips.

Several words in this product are ambiguous in ways that change what a number
means. "Events" is a stream count, not a trade count. "Accounts" is sometimes
distinct logins seen and sometimes accounts routable. A dealer reading
"1,204 events" has no way to know whether that is 1,204 trades unless told.

Rather than expand every label into a sentence, terms carry a definition on
hover. Keeping them in one dictionary also means the same word cannot end up
defined two different ways on two screens.
"""

from __future__ import annotations

GLOSSARY: dict[str, str] = {
    # --- stream vocabulary ------------------------------------------------
    "events": (
        "Individual messages consumed from the broker's event stream. One trade "
        "usually produces several events (an entry deal and an exit deal at "
        "minimum), so this is NOT a trade count."
    ),
    "accounts": (
        "Distinct client logins seen in the selected window. An account is "
        "counted once however many times it traded."
    ),
    "routable": (
        "Accounts the desk could route on this day: every account with data on "
        "or before the decision day, carried forward from its most recent "
        "observation, excluding those dormant more than 30 days. A routing "
        "decision is made before the session, when nobody knows who will trade."
    ),
    "active": (
        "An account is active on any day it was EXPOSED -- it opened a "
        "position, closed one, carried one across the day, or holds one now. "
        "This is wider than 'traded today': a position held Monday to Friday "
        "makes the account active all week."
    ),
    "carrying": (
        "A day on which the account held an open position but neither opened "
        "nor closed anything. The firm carried risk; the account transacted "
        "nothing."
    ),

    # --- P&L conventions --------------------------------------------------
    "firm_pnl": (
        "The firm's profit. On B-booked flow it is the negative of the client's "
        "realised P&L -- the firm takes the other side, so a losing client is a "
        "gain. Hedged (A-booked) flow contributes nothing here."
    ),
    "client_pnl": (
        "The client's own realised profit. Negative means clients lost money, "
        "which is a gain for the firm on unhedged flow."
    ),
    "cash_flow": (
        "Deposits, withdrawals, credits and corrections. These carry a 'profit' "
        "field in the raw feed but are NOT trading performance, and are excluded "
        "from every P&L figure here."
    ),

    # --- risk -------------------------------------------------------------
    "var": (
        "Value at Risk: the loss the book exceeds on the worst 5% (or 1%) of "
        "days. Computed as an empirical quantile of the actual daily series, not "
        "a normal approximation -- client P&L is heavy-tailed, and a parametric "
        "fit understates precisely the days that matter."
    ),
    "expected_shortfall": (
        "The AVERAGE loss on days that breach VaR. VaR says how bad a bad day "
        "is; expected shortfall says how bad the bad days are."
    ),
    "herfindahl": (
        "Concentration of profit across accounts, 0 to 1. Near 0 means earnings "
        "are spread across many clients; near 1 means a single client is the "
        "book. Two books with identical P&L carry very different risk."
    ),
    "var_multi_day": (
        "Multi-day VaR is scaled from the daily figure by the square root of "
        "time. That assumes independent daily moves; real moves cluster, so "
        "these are a floor rather than a worst case."
    ),
    "drawdown": (
        "Distance below the running peak of cumulative P&L. The peak-to-trough "
        "loss an operator would actually have lived through."
    ),

    # --- model ------------------------------------------------------------
    "score": (
        "Model probability that this account (or trade) is heading for an "
        "unusually strong run by its own standards. Not a prediction of "
        "profit size."
    ),
    "expected_impact": (
        "Model confidence above the base rate, scaled by the exposure the "
        "account typically carries. Ranks the list by consequence rather than "
        "by confidence -- a certain call on a tiny account is worth nothing."
    ),
    "roc_auc": (
        "Probability the model ranks a randomly chosen winner above a randomly "
        "chosen loser. 0.5 is a coin flip. Measured out-of-sample, walk-forward."
    ),
    "stale": (
        "Settings have changed since these results were produced. The figures "
        "remain correct for the configuration that made them; retrain to apply "
        "the new one."
    ),
    "evidence_age": (
        "Days since this account last traded. The routing call rests on its "
        "most recent behaviour, so older evidence is weaker."
    ),
    "markout": (
        "Average market move after a client's fills. The SHAPE across horizons "
        "is the signal: a move that appears within a minute and decays is a "
        "speed advantage; one that keeps building over a day is information."
    ),
    "drift": (
        "Difference between the exposure the engine intended and what the "
        "terminal actually holds. Caused by rejections, partial fills, slippage "
        "and margin limits."
    ),
}


def term(key: str) -> str:
    return GLOSSARY.get(key, "")
