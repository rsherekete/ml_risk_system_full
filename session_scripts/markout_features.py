"""Per-account-day markout features from `markouts_v2.markouts`.

Markout = how the market moved after a fill, at increasing horizons. It is the
standard microstructure measure of toxic flow: a client whose entries are
consistently followed by favourable moves is either skilled or exploiting
latency, and either way is an account to hedge rather than take the other side
of. The warehouse already computes the price at every second from -60s to +60s
around both open and close, so no tick scan is needed.

LEAKAGE DISCIPLINE -- the reason this file is careful rather than clever:

* `trade_profit_usd`, `commission_usd`, `swap_usd` and anything derived from
  them are NEVER selected. They are the outcome the model is trying to predict;
  joining them in any form would produce a spectacular AUC and a worthless
  model.
* Markouts are aggregated per (account, trading day) and used to predict that
  account's *next active day*. A row therefore only ever sees market context
  from trades that had already closed when the decision was made.
* Post-trade markouts (`open_plus_*`) describe the market's reaction to a fill
  that has already happened -- they are legitimate history at decision time, not
  future information about the day being predicted.
"""

from __future__ import annotations

import pandas as pd

MARKOUT_TABLE = "zfx-dwh-prod.markouts_v2.markouts"

#: Horizons kept from the 240 available per-second columns. Enough to describe
#: the shape of the reaction (immediate, short, medium) without dragging 240
#: highly-collinear columns into the model.
HORIZONS = (1, 5, 10, 30, 60)


def markout_sql(start: str, end: str, servers: tuple[str, ...] | None = None) -> str:
    """Aggregate markouts to one row per (server, account, trading day).

    Markouts are signed by trade direction and normalised by entry price, so
    a positive value always means "the market moved in the client's favour",
    comparably across symbols priced in wildly different units.
    """
    open_terms = ",\n".join(
        f"""    AVG(SAFE_DIVIDE(
      (m.open_plus_{h}_sec - m.open_price) * IF(m.is_buy, 1, -1), NULLIF(m.open_price, 0)
    )) AS markout_open_plus_{h}s,
    AVG(SAFE_DIVIDE(
      (m.open_price - m.open_minus_{h}_sec) * IF(m.is_buy, 1, -1), NULLIF(m.open_price, 0)
    )) AS runup_open_minus_{h}s"""
        for h in HORIZONS
    )
    close_terms = ",\n".join(
        f"""    AVG(SAFE_DIVIDE(
      (m.close_plus_{h}_sec - m.close_price) * IF(m.is_buy, 1, -1), NULLIF(m.close_price, 0)
    )) AS markout_close_plus_{h}s"""
        for h in HORIZONS
    )
    server_filter = ""
    if servers:
        listed = ", ".join(f"'{s}'" for s in servers)
        server_filter = f"AND m.server_name IN ({listed})"

    return f"""
SELECT
    m.server_name,
    m.trading_account_number AS login,
    DATE(m.close_time) AS day,
    COUNT(*) AS markout_trades,
    AVG(SAFE_DIVIDE(m.open_ask - m.open_bid, NULLIF(m.open_price, 0))) AS avg_entry_spread,
    AVG(m.group_spread_diff) AS avg_group_spread_diff,
    AVG(ABS(m.qty_usd)) AS avg_qty_usd,
{open_terms},
{close_terms}
FROM `{MARKOUT_TABLE}` m
WHERE m.close_time_ms >= TIMESTAMP('{start}')
  AND m.close_time_ms <  TIMESTAMP('{end}')
  AND m.order_is_currently_open = FALSE
  {server_filter}
GROUP BY server_name, login, day
"""


def fetch_markout_features(client, start: str, end: str, servers: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Run `markout_sql` and derive the shape features the model actually reads."""
    frame = client.query(markout_sql(start, end, servers)).to_dataframe()
    if frame.empty:
        return frame
    frame["day"] = pd.to_datetime(frame["day"])

    # Slope of the post-entry reaction: a client whose edge grows with time
    # horizon looks different from one who is merely getting a good fill.
    if {"markout_open_plus_1s", "markout_open_plus_60s"} <= set(frame.columns):
        frame["markout_slope_open"] = frame["markout_open_plus_60s"] - frame["markout_open_plus_1s"]
    if {"markout_open_plus_5s", "markout_open_plus_30s"} <= set(frame.columns):
        frame["markout_accel_open"] = frame["markout_open_plus_30s"] - frame["markout_open_plus_5s"]
    # Pre-entry run-up vs post-entry markout: strong run-up with weak markout is
    # momentum chasing; weak run-up with strong markout is anticipation, which
    # is the latency/arbitrage signature.
    if {"runup_open_minus_5s", "markout_open_plus_5s"} <= set(frame.columns):
        frame["anticipation_ratio"] = frame["markout_open_plus_5s"] - frame["runup_open_minus_5s"]
    # Exit quality: did the market keep moving their way after they closed
    # (left money on the table) or reverse (well-timed exit)?
    if {"markout_close_plus_60s"} <= set(frame.columns):
        frame["exit_timing"] = -frame["markout_close_plus_60s"]
    return frame


def attach_markouts(features: pd.DataFrame, markouts: pd.DataFrame, server_to_database: dict[str, str]) -> pd.DataFrame:
    """Left-join markout features onto the per-account-day feature frame.

    Left join, never inner: an account-day with no markout coverage keeps its
    behavioural features and gets NaN market context, which the gradient
    boosters handle natively. An inner join would silently restrict the whole
    study to whatever subset markouts happens to cover.
    """
    if markouts.empty:
        return features
    mapped = markouts.copy()
    mapped["database"] = mapped["server_name"].map(server_to_database)
    mapped = mapped.dropna(subset=["database"])
    mapped["account_key"] = mapped["database"] + ":" + mapped["login"].astype("int64").astype(str)
    columns = [c for c in mapped.columns if c not in {"server_name", "login", "database"}]
    return features.merge(mapped[columns], on=["account_key", "day"], how="left")
