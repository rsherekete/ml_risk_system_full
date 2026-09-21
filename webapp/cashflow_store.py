"""Cash movements, cached locally and turned into point-in-time model features.

TWO PROBLEMS THIS SOLVES

1. `cashflow.load_cashflows` queries MySQL on every request. The Cash Flow
   screen therefore fails whenever the VPN is down, and no model could ever use
   the data without a live database round trip inside training. Deposits and
   withdrawals are now backfilled to parquet beside the trades, on the same
   monthly-partition pattern, so both work offline.

2. Deposit and withdrawal behaviour is among the strongest signals for
   identifying fraud and abuse -- a client who deposits, wins, and withdraws
   immediately looks nothing like one who funds an account and trades it down --
   yet none of it reached the models.

WHY THE FEATURES ARE BUILT THE WAY THEY ARE

The obvious implementation is fatal. Aggregating a client's deposits and
withdrawals over the whole window and attaching that to every one of their
account-days tells the model on day 1 what the client will do on day 700. It
would score beautifully and be worthless, which is precisely the failure mode
that produced an AUC of 1.0000 earlier in this project.

Every feature here is therefore a CUMULATIVE-TO-DATE quantity, shifted so that a
given day sees only movements that had already settled before it. An account's
first day carries zeros, not its lifetime totals.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import data_store

#: Cash movements live beside the trades, partitioned by month the same way.
STORE = data_store.WAREHOUSE.parent / "cashflow_store"
STORE.mkdir(exist_ok=True)

#: Features handed to the models. Named so a leak is obvious in an importance
#: table: everything is "to date", nothing is a lifetime total.
CASHFLOW_FEATURES = (
    "deposits_to_date",
    "withdrawals_to_date",
    "net_funding_to_date",
    "deposit_count_to_date",
    "withdrawal_count_to_date",
    "days_since_deposit",
    "days_since_withdrawal",
    "withdrawal_ratio",
    "funding_churn",
)


def partition_path(database: str, period: str):
    directory = STORE / database
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{period}.parquet"


def backfill(database: str, days: int = 730, progress=None) -> dict:
    """Pull one server's cash movements into the local store."""
    from webapp import cashflow

    frame = cashflow.load_cashflows(database, days=days)
    if frame.empty:
        if progress:
            progress(f"{database}: no cash movements")
        return {"database": database, "rows": 0}

    frame["when"] = pd.to_datetime(frame["when"], errors="coerce")
    frame = frame.loc[frame["when"].notna()]
    written = 0
    for period, part in frame.groupby(frame["when"].dt.to_period("M").astype(str)):
        part.to_parquet(partition_path(database, period), index=False)
        written += len(part)
    if progress:
        progress(f"{database}: {written:,} cash movements")
    return {"database": database, "rows": written}


def read_cashflows(databases: tuple[str, ...] | None = None,
                   start: datetime | None = None,
                   end: datetime | None = None) -> pd.DataFrame:
    """Cash movements from the local store."""
    servers = databases or tuple(p.name for p in STORE.iterdir() if p.is_dir())
    parts = []
    for server in servers:
        directory = STORE / server
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.parquet")):
            try:
                parts.append(pd.read_parquet(path))
            except Exception:
                continue
    if not parts:
        return pd.DataFrame(columns=["account_key", "when", "amount", "kind"])
    frame = pd.concat(parts, ignore_index=True)
    frame["when"] = pd.to_datetime(frame["when"], errors="coerce")
    if start is not None:
        frame = frame.loc[frame["when"] >= pd.Timestamp(start).tz_localize(None)]
    if end is not None:
        frame = frame.loc[frame["when"] <= pd.Timestamp(end).tz_localize(None)]
    return frame


def daily_features(cashflows: pd.DataFrame) -> pd.DataFrame:
    """Per (account, day) cumulative funding state, known as of that morning.

    Returns one row per account-day on which something moved. Callers merge it
    onto their own calendar with a backward `merge_asof`, which carries the last
    known state forward without inventing days.
    """
    if cashflows.empty:
        return pd.DataFrame(columns=["account_key", "day", *CASHFLOW_FEATURES])

    frame = cashflows.copy()
    frame["day"] = pd.to_datetime(frame["when"], errors="coerce").dt.normalize()
    frame = frame.loc[frame["day"].notna()]
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce").fillna(0.0)

    # Sign convention: deposits positive, withdrawals negative. Some sources
    # record withdrawals as positive rows with a 'kind' instead, so the split is
    # taken from the sign, which both encodings agree on.
    frame["deposit"] = frame["amount"].clip(lower=0)
    frame["withdrawal"] = (-frame["amount"]).clip(lower=0)
    # Counted as columns rather than with a lambda aggregate: a per-group Python
    # callable over 4.9M movements does not finish in reasonable time, while
    # summing a precomputed boolean is a single vectorised pass.
    frame["deposit_n"] = (frame["deposit"] > 0).astype("int32")
    frame["withdrawal_n"] = (frame["withdrawal"] > 0).astype("int32")

    daily = frame.groupby(["account_key", "day"], observed=True).agg(
        deposit=("deposit", "sum"),
        withdrawal=("withdrawal", "sum"),
        deposit_n=("deposit_n", "sum"),
        withdrawal_n=("withdrawal_n", "sum"),
    ).reset_index().sort_values(["account_key", "day"], kind="mergesort")

    grouped = daily.groupby("account_key", sort=False)
    # Cumulative INCLUSIVE of the day the movement happened: a deposit that
    # settled this morning is known to a decision made later the same day. What
    # must never be included is a future movement, and a cumulative sum cannot.
    daily["deposits_to_date"] = grouped["deposit"].cumsum()
    daily["withdrawals_to_date"] = grouped["withdrawal"].cumsum()
    daily["deposit_count_to_date"] = grouped["deposit_n"].cumsum()
    daily["withdrawal_count_to_date"] = grouped["withdrawal_n"].cumsum()
    daily["net_funding_to_date"] = (daily["deposits_to_date"]
                                    - daily["withdrawals_to_date"])

    # The DATE of the last movement of each kind, not a day count. A count
    # computed here would be frozen by the carry-forward merge in `attach`: an
    # account whose last deposit was on day 1 and whose last row is day 10 would
    # still report "9 days since" on day 300. The gap is therefore measured
    # against the decision day, which only `attach` knows.
    for kind in ("deposit", "withdrawal"):
        stamps = daily["day"].where(daily[kind] > 0)
        daily[f"last_{kind}_day"] = stamps.groupby(
            daily["account_key"], sort=False).ffill()

    # The fraud-relevant shapes. A client who has withdrawn most of what they
    # deposited behaves differently from one still funding the account, and
    # churn -- money in and straight back out -- is the classic bonus-abuse and
    # laundering signature.
    #
    # Both are ratios with a denominator that can approach zero, which produces
    # values like a withdrawal ratio of 528,715 or a churn of 1.8e16. Those are
    # arithmetic, not behaviour -- an account that deposited a cent and withdrew
    # a bonus -- and feeding them to a learner lets one meaningless row dominate
    # a split. They are clipped to the range over which they still mean
    # something: "withdrew up to 10x what was deposited" and "money went round
    # up to 100 times", both already far beyond normal.
    deposits = daily["deposits_to_date"].replace(0, np.nan)
    daily["withdrawal_ratio"] = (daily["withdrawals_to_date"] / deposits
                                 ).fillna(0.0).clip(0.0, 10.0)
    daily["funding_churn"] = (
        (daily["deposits_to_date"] + daily["withdrawals_to_date"])
        / daily["net_funding_to_date"].abs().replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 100.0)

    carried = [c for c in CASHFLOW_FEATURES if not c.startswith("days_since_")]
    return daily[["account_key", "day", *carried,
                  "last_deposit_day", "last_withdrawal_day"]]


def attach(frame: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Carry each account's funding state forward onto its trading calendar.

    A backward `merge_asof` per account: each account-day sees the most recent
    funding state at or before it, and nothing after. Accounts with no cash
    movements at all get zeros rather than being dropped -- they are the
    majority, and losing them would silently shrink the training population.
    """
    if frame.empty:
        return frame
    if features.empty:
        for column in CASHFLOW_FEATURES:
            frame[column] = 0.0
        return frame

    # `merge_asof` requires both sides sorted by the `on` key GLOBALLY -- sorting
    # by (account, day) orders days only within each account and it rejects that.
    left = frame.copy()
    left["day"] = pd.to_datetime(left["day"])
    left = left.sort_values("day", kind="mergesort")
    right = features.copy()
    right["day"] = pd.to_datetime(right["day"])
    right = right.sort_values("day", kind="mergesort")

    merged = pd.merge_asof(
        left, right, on="day", by="account_key", direction="backward")

    # Recency measured against the DECISION day, so it keeps counting while an
    # account sits idle instead of freezing at its last movement.
    for kind in ("deposit", "withdrawal"):
        stamp = merged.get(f"last_{kind}_day")
        merged[f"days_since_{kind}"] = (
            (merged["day"] - pd.to_datetime(stamp)).dt.days
            if stamp is not None else np.nan)
    merged = merged.drop(columns=[c for c in ("last_deposit_day", "last_withdrawal_day")
                                  if c in merged.columns])

    for column in CASHFLOW_FEATURES:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
    # "No movement yet" is genuinely zero funding, not missing data. Recency is
    # left as NaN, because "never deposited" is not the same as "deposited today"
    # and filling it with 0 would assert the opposite.
    for column in CASHFLOW_FEATURES:
        if column.startswith("days_since_"):
            continue
        merged[column] = merged[column].fillna(0.0)
    return merged


def backfill_all(days: int = 730, progress=None) -> dict:
    """Every live server's cash movements, MySQL ones only.

    `mt5_dubai_live01` is not in this MySQL instance, so it is skipped here
    rather than failing the run; its cash movements need the same BigQuery
    treatment its trades got.
    """
    from webapp.mysql_extract import MYSQL_DATABASES

    total, failures = 0, []
    for database in MYSQL_DATABASES:
        try:
            total += backfill(database, days=days, progress=progress)["rows"]
        except Exception as error:
            failures.append(f"{database}: {type(error).__name__}: {str(error)[:90]}")
            if progress:
                progress(f"{database}: FAILED {type(error).__name__}")
    return {"rows": total, "failures": failures}
