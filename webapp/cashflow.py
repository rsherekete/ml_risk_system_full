"""Deposits, withdrawals and transfers -- and what they say about an account.

Cash movement is the most under-used signal in this dataset. Trading data says
how a client behaves once funded; cash data says how they behave around the
firm, and several fraud patterns are visible only there:

* **Deposit-and-withdraw with minimal trading** -- money in, a token trade, money
  out. A classic laundering shape, and invisible to any model that only reads
  trades.
* **Internal transfers between accounts** -- moving balance rather than depositing
  defeats per-account limits and is a common bonus-abuse pattern. MT4 labels
  these explicitly (`Internal Transfer`), so they can be separated from genuine
  external funding instead of being counted as deposits.
* **Withdrawal immediately after an unusual win** -- extracting the proceeds of a
  one-off event before the firm can review it.
* **Deposit velocity** -- many small deposits in quick succession is a different
  risk from one large one, even at identical totals.

The signed convention throughout: a POSITIVE amount is money into the client's
account, negative is money out. The firm's cash position is the opposite sign,
which is stated wherever a figure is reported so the direction is never
ambiguous.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

#: MT5 marks non-trade balance operations with this deal action. Verified
#: against the server: 341,964 deals totalling $211.8M.
MT5_BALANCE_ACTION = 2

#: MT4 operation types that are NOT external funding. Counting an internal
#: transfer as a deposit double-counts firm-wide inflow, because the matching
#: withdrawal from the other account is also recorded.
INTERNAL_TYPES = ("Internal Transfer", "Credit", "Bonus", "Correction")


def load_cashflows(database: str, days: int = 730) -> pd.DataFrame:
    """Cash movements for one server, in a common shape.

    MT4 keeps them in a dedicated `balance_ops` table with a human-readable
    `type`; MT5 records them as deals with `action = 2` and encodes the kind in
    the comment. Both become: account_key, when, amount, kind, comment.
    """
    from webapp.mysql_extract import _connection

    since = datetime.now(timezone.utc) - timedelta(days=days)
    if database.startswith("mt5"):
        sql = ("SELECT login, time AS tm, profit, comment, 'deal' AS type "
               "FROM deals WHERE action = %s AND time >= %s")
        params = (MT5_BALANCE_ACTION, since.replace(tzinfo=None))
    else:
        sql = ("SELECT login, tm, profit, comment, type FROM balance_ops "
               "WHERE tm >= %s")
        params = (since.replace(tzinfo=None),)

    connection = _connection(database, timeout=300)
    try:
        frame = pd.read_sql(sql, connection, params=params)
    finally:
        connection.close()
    if frame.empty:
        return pd.DataFrame(columns=["account_key", "when", "amount", "kind",
                                     "comment", "database"])

    frame["database"] = database
    frame["account_key"] = database + ":" + frame["login"].astype("int64").astype(str)
    frame["when"] = pd.to_datetime(frame["tm"], errors="coerce")
    frame["amount"] = pd.to_numeric(frame["profit"], errors="coerce")
    frame["kind"] = _classify(frame)          # sign-based, scale-invariant
    # CENT DEFLATION: a cent account's deposit of "$1000" is 1000 cents = $10.
    # Without this the cash-flow totals are ~100x on cent accounts.
    from webapp import trade_feed
    trade_feed.deflate_cent(frame, database, "login", money_cols=("amount",))
    return frame[["account_key", "database", "when", "amount", "kind", "comment"]].dropna(
        subset=["when", "amount"])


def _classify(frame: pd.DataFrame) -> pd.Series:
    """Deposit, withdrawal, transfer or adjustment.

    Uses the explicit `type` where the platform provides one and falls back to
    the comment prefix, which the back office stamps consistently
    (`D-`/`DT-` deposit, `W-`/`WT-` withdrawal).
    """
    kind = pd.Series("adjustment", index=frame.index, dtype="object")
    amount = pd.to_numeric(frame["profit"], errors="coerce")
    label = frame["type"].astype("string").fillna("")
    comment = frame["comment"].astype("string").fillna("").str.upper()

    is_internal = label.isin(INTERNAL_TYPES) | comment.str.startswith(("DT-", "WT-"))
    kind = kind.mask(amount > 0, "deposit")
    kind = kind.mask(amount < 0, "withdrawal")
    kind = kind.mask(is_internal & (amount > 0), "transfer_in")
    kind = kind.mask(is_internal & (amount < 0), "transfer_out")
    kind = kind.mask(label.str.contains("Credit|Bonus", case=False, na=False), "credit")
    return kind


def summarise(cashflows: pd.DataFrame, profiles: pd.DataFrame | None = None) -> dict:
    """Firm-wide totals, with external funding separated from internal movement."""
    if cashflows.empty:
        return {}
    external = cashflows.loc[cashflows["kind"].isin(("deposit", "withdrawal"))]
    internal = cashflows.loc[cashflows["kind"].isin(("transfer_in", "transfer_out"))]
    deposits = external.loc[external["amount"] > 0, "amount"].sum()
    withdrawals = external.loc[external["amount"] < 0, "amount"].sum()

    return {
        "operations": int(len(cashflows)),
        "accounts": int(cashflows["account_key"].nunique()),
        "deposits": float(deposits),
        "withdrawals": float(withdrawals),
        # Net external funding: what clients actually put in, net of what they
        # took out. Internal transfers net to zero firm-wide by construction and
        # are excluded so they cannot inflate the figure.
        "net_funding": float(deposits + withdrawals),
        "deposit_count": int((external["amount"] > 0).sum()),
        "withdrawal_count": int((external["amount"] < 0).sum()),
        "median_deposit": float(external.loc[external["amount"] > 0, "amount"].median() or 0),
        "median_withdrawal": float(external.loc[external["amount"] < 0, "amount"].median() or 0),
        "internal_volume": float(internal["amount"].abs().sum() / 2),
        "internal_count": int(len(internal)),
        "first": str(cashflows["when"].min()),
        "last": str(cashflows["when"].max()),
    }


def by_dimension(cashflows: pd.DataFrame, profiles: pd.DataFrame,
                 dimension: str = "region") -> pd.DataFrame:
    """Funding broken down by region, country or acquisition group."""
    from webapp.client_profile import attach_region

    if cashflows.empty:
        return pd.DataFrame()
    enriched = attach_region(cashflows, profiles)
    column = {"region": "region", "country": "country_code", "group": "group"}.get(
        dimension, "region")
    if column not in enriched.columns:
        return pd.DataFrame()

    external = enriched.loc[enriched["kind"].isin(("deposit", "withdrawal"))]
    grouped = external.groupby(column, observed=True).apply(
        lambda g: pd.Series({
            "accounts": g["account_key"].nunique(),
            "deposits": g.loc[g["amount"] > 0, "amount"].sum(),
            "withdrawals": g.loc[g["amount"] < 0, "amount"].sum(),
            "net_funding": g["amount"].sum(),
            "operations": len(g),
        }), include_groups=False).reset_index()
    grouped["withdrawal_ratio"] = (grouped["withdrawals"].abs()
                                   / grouped["deposits"].replace(0, np.nan))
    return grouped.sort_values("net_funding", ascending=False).reset_index(drop=True)


def account_features(cashflows: pd.DataFrame, trades: pd.DataFrame | None = None,
                     as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Per-account cash behaviour, usable as model features or fraud flags.

    Every column is a property of the account's funding history, which no
    trade-derived feature can express. The ratios matter more than the levels:
    a client who withdraws 95% of what they deposit behaves differently from one
    who withdraws 20%, at any deposit size.
    """
    if cashflows.empty:
        return pd.DataFrame()
    as_of = pd.Timestamp(as_of) if as_of is not None else cashflows["when"].max()

    external = cashflows.loc[cashflows["kind"].isin(("deposit", "withdrawal"))].copy()
    grouped = external.groupby("account_key", observed=True)
    features = pd.DataFrame({
        "deposits_total": grouped["amount"].apply(lambda s: s[s > 0].sum()),
        "withdrawals_total": grouped["amount"].apply(lambda s: -s[s < 0].sum()),
        "deposit_count": grouped["amount"].apply(lambda s: int((s > 0).sum())),
        "withdrawal_count": grouped["amount"].apply(lambda s: int((s < 0).sum())),
        "first_funded": grouped["when"].min(),
        "last_movement": grouped["when"].max(),
        "largest_deposit": grouped["amount"].max(),
        "largest_withdrawal": grouped["amount"].min(),
    }).reset_index()

    internal = cashflows.loc[cashflows["kind"].isin(("transfer_in", "transfer_out"))]
    if not internal.empty:
        transfers = internal.groupby("account_key", observed=True)["amount"].agg(
            transfer_count="size", transfer_net="sum").reset_index()
        features = features.merge(transfers, on="account_key", how="left")
    for column in ("transfer_count", "transfer_net"):
        if column not in features:
            features[column] = 0
    features[["transfer_count", "transfer_net"]] = \
        features[["transfer_count", "transfer_net"]].fillna(0)

    # The headline ratio: how much of what came in has gone back out.
    features["withdrawal_ratio"] = (features["withdrawals_total"]
                                    / features["deposits_total"].replace(0, np.nan))
    features["net_funding"] = features["deposits_total"] - features["withdrawals_total"]
    features["account_age_days"] = (as_of - features["first_funded"]).dt.days
    features["days_since_movement"] = (as_of - features["last_movement"]).dt.days
    # Many small deposits and one large one are different risks at equal totals.
    features["mean_deposit"] = (features["deposits_total"]
                                / features["deposit_count"].replace(0, np.nan))
    features["deposit_velocity"] = (features["deposit_count"]
                                    / features["account_age_days"].replace(0, np.nan).clip(lower=1))

    if trades is not None and not trades.empty and "account_key" in trades.columns:
        activity = trades.groupby("account_key", observed=True).agg(
            trade_count=("net_profit", "size"),
            trade_pnl=("net_profit", "sum"),
        ).reset_index()
        features = features.merge(activity, on="account_key", how="left")
        features[["trade_count", "trade_pnl"]] = features[["trade_count", "trade_pnl"]].fillna(0)
        # Trades per unit of deposited capital. Near zero with real money moving
        # through is the deposit-and-withdraw shape.
        features["trades_per_1k_deposited"] = (
            features["trade_count"] / (features["deposits_total"] / 1000).replace(0, np.nan))
    return features


def fraud_flags(features: pd.DataFrame, min_deposit: float = 1000.0) -> pd.DataFrame:
    """Cash-behaviour patterns worth a compliance review, with the evidence.

    Thresholds are set so a flag means something: `min_deposit` keeps small
    accounts out entirely, because a $50 account withdrawing 100% is noise, not
    a finding.
    """
    if features.empty:
        return pd.DataFrame()
    material = features.loc[features["deposits_total"] >= min_deposit].copy()
    if material.empty:
        return pd.DataFrame()

    findings = []
    for row in material.itertuples():
        trades = getattr(row, "trade_count", None)
        ratio = row.withdrawal_ratio if pd.notna(row.withdrawal_ratio) else 0.0

        # Money in, money out, almost no trading -- the laundering shape.
        if trades is not None and trades < 10 and ratio >= 0.80:
            findings.append({
                "account_key": row.account_key, "severity": "critical",
                "pattern": "Funded and withdrawn with negligible trading",
                "evidence": (f"${row.deposits_total:,.0f} deposited, "
                             f"{ratio:.0%} withdrawn, only {int(trades)} trades"),
                "value": float(row.deposits_total),
            })
        elif ratio >= 0.95 and row.deposits_total >= 10_000:
            findings.append({
                "account_key": row.account_key, "severity": "warning",
                "pattern": "Withdraws nearly everything deposited",
                "evidence": (f"{ratio:.0%} of ${row.deposits_total:,.0f} withdrawn "
                             f"across {int(row.withdrawal_count)} operations"),
                "value": float(row.deposits_total),
            })

        # Heavy internal transfer use relative to external funding: balance is
        # being moved between accounts rather than deposited.
        if row.transfer_count >= 20 and abs(row.transfer_net) > row.deposits_total * 0.5:
            findings.append({
                "account_key": row.account_key, "severity": "warning",
                "pattern": "Balance moved largely by internal transfer",
                "evidence": (f"{int(row.transfer_count)} transfers netting "
                             f"${row.transfer_net:,.0f} against ${row.deposits_total:,.0f} deposited"),
                "value": abs(float(row.transfer_net)),
            })

        # Rapid repeated funding: many deposits in a short life.
        if row.deposit_velocity > 0.5 and row.deposit_count >= 15:
            findings.append({
                "account_key": row.account_key, "severity": "info",
                "pattern": "High deposit frequency",
                "evidence": (f"{int(row.deposit_count)} deposits in "
                             f"{int(row.account_age_days)} days"),
                "value": float(row.deposit_count),
            })

    if not findings:
        return pd.DataFrame()
    result = pd.DataFrame(findings)
    order = {"critical": 0, "warning": 1, "info": 2}
    result["_rank"] = result["severity"].map(order)
    return result.sort_values(["_rank", "value"], ascending=[True, False]).drop(columns="_rank")
