"""Admin-only copy trader: the quant model driving one external MT5 account.

A deliberate rewrite. The first version accumulated a dozen fixes on top of
wrong assumptions and became impossible to reason about; this one keeps only
what the evidence supports and states why each rule exists.

WHAT IT DOES

Every client trade opening on the firm's Kafka stream is scored by the saved
walk-forward booster at the moment it arrives, and judged ON ITS OWN MERIT:

  * top decile of recent scores  -> COPY   (mirror the client's direction)
  * bottom decile                -> INVERT (take the other side)
  * anything between             -> ignore

WHY PER-TRADE, NOT PER-ACCOUNT

Measured, not assumed. A weekly-rebuilt account watchlist (top/bottom 5% by
score, ranked by win rate then volume) was backtested against pure per-trade
selection over the same 63 days: per-trade made $95.2M with a $23.8k drawdown,
the watchlist $2.1M with a $1.44M drawdown. Worse on both axes, because an
account gate discards ~98% of qualifying trades and concentrates risk. The
model's edge is genuinely per-trade -- the same client produces a 0.86 trade
and a 0.31 trade within the hour. Account history survives only as FEATURES
feeding each trade's score.

WHY QUANTILE ROUTING

The validated backtest selects the top 10% of each day's scores, not scores
above a fixed number. A hard 0.80 threshold selected nothing at all on a venue
whose scores sit at 0.50-0.70. The rolling window reproduces the backtest's
quantile behaviour against whatever population is actually flowing.

WHY NO STOPS BY DEFAULT

The path-aware study replayed 948k copied trades against minute bars: every
stop multiple cost $16.8M-$18.3M against simply following the client out, and
every target was worse still. The model's winners routinely draw down through
a stop before recovering, so a stop converts a temporary excursion into a
realised loss. MIRRORING THE CLIENT'S EXIT is the evidence-backed policy.
Brackets remain available as a manual override, off by default.

THE THREE HARD RULES

  1. FRESHNESS -- act only on trades whose own timestamp is recent. Kafka is a
     log; a consumer reading history delivers days-old trades in seconds, and
     acting on them opened positions that closed instantly when the matching
     exit arrived moments later.
  2. ENTRY QUALITY -- never enter worse than the client did. Our buy fills at
     or below their entry, our sell at or above. No quote to verify: no order.
  3. RISK CONTRACT -- lots scale so the strategy's historical drawdown equals
     35% of the live balance, and total notional never exceeds the account's
     leverage.

CREDENTIALS live in `vantage.yaml` at the repository root (login, password,
server). The MT5 terminal must be installed; the API drives it.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT.parent / "vantage.yaml"
EXAMPLE_PATH = ROOT.parent / "vantage.example.yaml"
DB = ROOT / "app.db"
ARTIFACTS = ROOT / "artifacts"

#: MT5 raw volume is ten-thousandths of a lot; some producers send lots
#: directly. Nothing retail trades 500+ lots, so anything larger is raw.
RAW_VOLUME_THRESHOLD = 500.0


# ---------------------------------------------------------------- config
@dataclass
class VantageConfig:
    login: int = 0
    password: str = ""
    server: str = "VantageInternational-Demo"
    terminal_path: str = ""
    mode: str = "paper"                     # paper | live

    #: Fraction of EQUITY the strategy may put at drawdown risk. With
    #: sizing_policy="drawdown", every trade is sized maximally so the book's
    #: projected max drawdown stays within this fraction of live equity.
    risk_budget_fraction: float = 0.35
    #: Worst adverse price move (as a FRACTION of notional) a position is
    #: assumed to suffer over its holding period -- the drawdown driver. This is
    #: a copy strategy with minute-scale holds, so the realised excursion is
    #: small (~0.2% of notional on the tape); the default carries a safety
    #: cushion over that for news spikes and worse sessions.
    dd_stress_move: float = 0.01
    #: Budget is reserved for this many CONCURRENT positions so every triggered
    #: signal can be executed (not just the first) -- each trade takes at most
    #: 1/expected_peak_positions of the 35% drawdown budget. Set to the observed
    #: peak concurrency (tape: ~16 on gold) plus a cushion. Together with the
    #: net-exposure safety cap the book's worst-case drawdown never exceeds the
    #: budget even if concurrency spikes; excess signals are then refused rather
    #: than breaching 35%.
    expected_peak_positions: int = 20
    #: TEMPORARY override: when > 0, EVERY copy/invert signal is taken at this
    #: FIXED lot size, bypassing the drawdown sizer entirely (the leverage wall
    #: and position cap still apply). 0 = off, use the drawdown rule.
    fixed_lot: float = 0.0
    #: Account leverage cap; total open notional may not exceed leverage x equity.
    #: Auto-detected from the broker at runtime (a 500x demo overwrites this).
    leverage: float = 30.0
    #: RISK cap on gross exposure, INDEPENDENT of the broker's max leverage.
    #: The exposure wall is min(leverage, max_gross_leverage) x equity, so a
    #: 500x broker never lets the book exceed this many multiples of equity.
    #: Everything scales with equity, so it adapts to any account size and to
    #: whatever leverage the broker grants.
    max_gross_leverage: float = 30.0
    max_lots_per_trade: float = 1.00
    max_open_positions: int = 20

    #: Routing quantiles over the rolling score window.
    copy_quantile: float = 0.90
    invert_quantile: float = 0.10
    #: Absolute guards so a thin or skewed window cannot route on noise. These
    #: are FALLBACKS -- at engine start they are replaced by the artifact's own
    #: decile cutoffs (~0.85 / ~0.15), anchoring live selection to the tested
    #: standard: overnight the rolling window fills with mediocre gold flow and
    #: its "top decile" fired at 0.62, looser than anything the backtest ever
    #: selected.
    min_copy_score: float = 0.55
    max_invert_score: float = 0.45
    #: Manual anchor overrides (0 = derive from the artifact). When set, these
    #: replace the calibrated deciles outright -- the operator's dial between
    #: selectivity and trade frequency. The rolling window may still tighten
    #: them during hot flow, never loosen.
    copy_anchor_override: float = 0.0
    invert_anchor_override: float = 0.0
    #: Master switch for the INVERT leg. Ground-truth reconcile showed inverts
    #: were 94% of the realized loss (-$2,517 at 56.8% win) because inverting
    #: needs the training tail's ~99.8% client-loss precision, which live scoring
    #: does not yet achieve. Copies are break-even. Off until the live feature
    #: build matches the walk-forward and the invert tail is re-measured live.
    enable_invert: bool = True
    #: NET the book: an opposing signal on a symbol we already hold closes the
    #: oldest opposite position instead of opening a hedge.
    #: ROOT-CAUSE FIX (Sep 2026): OFF on a hedging account. Each of our positions
    #: tracks ONE client trade and must ride to THAT client's exit to capture its
    #: full round-trip (the +$6-9/trade edge the backtest priced, already net of
    #: the double spread). Netting closed positions at an unrelated opposite
    #: signal's arrival instead -- and net_only_profitable closed only the
    #: WINNERS, cutting the very tail that carries a low-hit-rate/high-expectancy
    #: strategy while losers ran. That is a second driver of the live losses, on
    #: top of the exit model cutting inverts short. Saving ~$1 of double spread
    #: is not worth forfeiting a $6-9 mirror capture; a hedging account can hold
    #: both sides, so it does.
    net_by_symbol: bool = False
    #: No entry unless the expected edge clears this multiple of the live
    #: spread cost -- at 0.01 lots the spread rivals the edge, and trading
    #: through it is how small costs eat the P&L. Applied to the SPREAD only;
    #: commission is added at full below.
    min_edge_multiple: float = 1.5
    #: Round-turn commission per lot (open + close). ~$6/lot on gold, FX and
    #: metals here; index/crypto CFDs are spread-only. The full commission
    #: must be cleared by the expected edge -- so the hurdle is
    #: 1.5 x spread + commission, not spread alone.
    commission_per_lot: float = 6.0
    #: Operator floor on OUR position size. The venue minimum is 0.01; setting
    #: this higher forces a deliberate minimum (e.g. 0.05) so positions are
    #: meaningful rather than dust. It overrides the anti-amplification guard
    #: -- the over-sizing is intentional. Watch margin: 0.05-0.1 gold lots are
    #: large for a small account.
    min_lot: float = 0.01
    #: MAE drawdown model in the sizer. mae_sizing: this trade's stress term
    #: becomes its predicted tail pre-profit drawdown (MAE q80) when that
    #: exceeds dd_stress_move (tighten-only). mae_veto: block a trade whose
    #: predicted tail drawdown at MINIMUM lot already exceeds its per-trade
    #: budget share instead of flooring up to min_lot.
    mae_sizing: bool = True
    mae_veto: bool = True
    #: RANGE CAP on the entry limit: rest it no further from the client's
    #: price than the extreme of the preceding N completed minute bars (buy at
    #: max(lowest low, model entry); sell at min(highest high, model entry)).
    #: If the capped wait falls under entry_wait_min_bps the trade goes to
    #: market. OOS (timed fills, 13 Sep 2026): on metals +4% P&L, +10% trades,
    #: same drawdown vs the uncapped model; on index it hurt. 0 = off.
    entry_range_cap_minutes: int = 0
    entry_range_cap_classes: str = "metals"
    #: A position of ours closed BY HAND in profit while its client is still
    #: in gets a replacement limit of the same lots at its original entry
    #: (cancelled when the client exits, expired by entry_limit_ttl_minutes).
    replace_manual_closes: bool = False
    #: STRATEGY 2 = TAPE (replaces Fade-15, whose code stays behind
    #: strategy2_fade). Per symbol with a tape artifact: the engine's own
    #: scored opens/closes over 5-120 min -> P(up over tape_hold_minutes).
    #: Trades the most confident half (the artifact's threshold), one
    #: position per symbol, flipped when the bias flips, closed when the hold
    #: elapses without renewal. Lots = tape_lots_per_5k x equity / 5000.
    strategy2_fade: bool = False
    tape_symbols: str = "*"
    tape_lots_per_5k: float = 1.0
    tape_hold_minutes: int = 15
    tape_min_top50_acc: float = 0.65
    tape_eval_seconds: float = 2.0
    tape_stale_minutes: int = 120
    #: Strategy 1 gate: no new mirror entry AGAINST an active tape bias. The
    #: gate reads its own, shorter horizon (a copy lives ~5 minutes): OOS the
    #: 5-minute gate cut gold drawdown 9% for 2.3% of P&L, the 15-minute one
    #: 3% for 2.8%. 0 = gate on the trading horizon.
    tape_bias_gate: bool = True
    tape_gate_minutes: int = 5
    #: Minutes after an engine start during which no bias is ACTIVE (the
    #: event buffer refills from empty; the 15-minute window is full at 15).
    tape_warm_minutes: int = 15
    #: A bias is only ACTIVE on a LIVE tape: at least this many client opens
    #: in the last 15 minutes, and the newest event no older than
    #: tape_feed_stale_seconds. On 14 Sep 2026 the feed died with the VPN and
    #: the model kept an "UP ACTIVE" long alive for an hour on zero events --
    #: price features alone must never carry a trade.
    tape_min_events_15m: int = 50
    tape_feed_stale_seconds: float = 180.0
    #: MARGIN ENVELOPES, as shares of equity, from the venue's OWN margin per
    #: lot (order_calc_margin). Strategy 1 sizes each position to fit
    #: expected_peak_positions of them inside its envelope; strategy 2 keeps
    #: its own envelope so a full mirror book cannot starve it. IG margins
    #: gold at ~4% (24x) whatever the account's headline leverage says.
    s1_margin_share: float = 0.60
    tape_margin_share: float = 0.25
    #: Which model-classes (metals, fx, index, crypto; comma list, '*' = all)
    #: the ENTRY model (rest a limit for the predicted pullback) and the EXIT
    #: model (favourable-excursion take-profit) apply to. Set from the OOS
    #: entry x exit study per class; a class not listed trades market entry /
    #: mirror exit.
    entry_model_classes: str = "*"
    exit_model_classes: str = "*"
    #: No single instrument may hold more than this share of max positions.
    max_symbol_share: float = 0.4
    #: AVERAGING GUARD: adding to a same-symbol same-direction line whose
    #: floating P&L is negative is allowed only at a BETTER price than the
    #: line's current extreme (buy lower / sell higher) -- never lock in a
    #: worse average while under water.
    avg_entry_guard: bool = False
    #: HARVEST (early profit-taking to admit new signals). The PROFITABLE
    #: backtest has NO harvest -- winners ride to their mirror exits. Live
    #: forensics (Sep 9): harvest scalped 784 winners at ~$4.9 avg while
    #: mirror exits were left realising the adversely-selected remainder at
    #: -$11.1 avg (40% win). Off = capacity overflow sizes down or refuses
    #: instead of amputating winners, matching the backtest's mechanics.
    harvest_enabled: bool = False
    #: SIDE-SCORE GUARD: rolling 2h average signal edge per symbol per side;
    #: stacking a 2nd+ position in a direction requires that side's average
    #: edge to be >= the opposite side's, so the book's NET risk follows the
    #: score differential (Sep-9 forensics: gold buys 0.795 vs sells 0.782
    #: -> both sides stacked, harvest pennies, mirror exits ate the tail).
    side_score_guard: bool = False
    #: RESCUE LIMITS -- for signals whose entry has already run away. Measured
    #: on 786k path-replayed trades: replacing FEASIBLE market entries with
    #: limits loses $3.1-4.1M (the never-dip trades are the best ones), but a
    #: limit on an otherwise-LOST signal is limit-versus-nothing, and the
    #: dipped cohort remains profitable. So: market when the entry qualifies,
    #: resting limit at client entry minus a spread buffer when it does not --
    #: cancelled on the client's exit, expiring after the TTL.
    rescue_limits: bool = True
    limit_buffer_spreads: float = 1.0
    limit_ttl_minutes: float = 10.0
    #: Entry tolerance: a market entry may be up to this many spreads WORSE
    #: than the client's, not only at-or-better. Strict better-only admitted a
    #: buy only after price dipped -- exactly the "dipped cohort" the limit
    #: study measured ~25% worse than the full set. The backtest entered at the
    #: client's price on everything; a small symmetric band recovers that
    #: population. Beyond the band, the rescue limit takes over.
    entry_tolerance_spreads: float = 1.0
    #: ENTRY-IMPROVEMENT MODEL: regressor on the stance model's own features
    #: predicting the bps of better entry available before the client's exit.
    #: When the prediction clears the threshold, the entry rests as a limit
    #: that many bps beyond the client's price instead of going to market
    #: (cancelled on the client's exit; expiring after its own TTL). OOS
    #: Aug 10-31: selective waiting (>=12bps) beat client-price entries with
    #: a shallower drawdown; indiscriminate waiting forfeits runners.
    use_entry_model: bool = False
    entry_wait_min_bps: float = 12.0
    entry_limit_ttl_minutes: float = 240.0
    #: Ask this multiple of the predicted retracement. The deployed artifact
    #: is a CONSERVATIVE quantile (alpha=0.35 -- a level the path likely
    #: reaches); asking 1.3x it was the best risk-adjusted point of the OOS
    #: sweep (net $14,643 at 4.0% maxDD vs 4.5% for the median-quantile ask).
    entry_ask_fraction: float = 1.0
    #: The trained 4-class exit model (hold 5m/30m/2h/6h): when its artifact
    #: exists, every fill gets a deadline from it and exits become
    #: min(client's mirror exit, model deadline) -- ride winners to their
    #: sweet spot, amputate wrong-direction trades at theirs.
    use_exit_model: bool = True
    #: ROOT-CAUSE FIX (Sep 2026): the backtest books an INVERT's profit as
    #: -(client's whole round-trip P&L), which we only realise if we hold until
    #: the client themselves exits. The exit model -- trained on the COPY
    #: objective of riding winners -- was cutting inverts short of the client's
    #: exit, so live inverts captured a mismatched slice of an asymmetric P&L
    #: distribution (frequent small client losses, rare large client wins) and
    #: turned the backtest's +$8/trade edge into the leg that is ~94% of the
    #: live loss. Inverts now mirror-exit ONLY (no model early-cut), which is
    #: exactly what the backtest assumed. Copies keep the exit model.
    invert_mirror_only: bool = True
    #: Netting closes the oldest opposite position -- but only a PROFITABLE
    #: one when this is set: closing a loser to net realises its loss AND
    #: kills its signal's remaining mirror alpha; hedging keeps both signal
    #: relationships alive (the account is a hedging account, it can).
    net_only_profitable: bool = True
    #: Keep the whole risk budget working: scale entries into budget headroom
    #: and, at full budget, harvest the most profitable positions to admit
    #: new signals. Utilisation is measured as the book's one-day one-sigma
    #: dollar move against risk_sigma_share of the budget.
    risk_scaling: bool = False
    risk_sigma_share: float = 0.5
    risk_boost_max: float = 4.0
    #: Sizing multiplier backed out EMPIRICALLY from the deployed-logic
    #: replay: the largest m at which replay drawdown stays inside the
    #: budget, with the leverage wall as the physical bound. Applied on top
    #: of k so the account operates AT its risk budget, not far under it.
    dd_calibration: float = 1.0
    #: STRATEGY 2 ("Fade-15"): a market-only model of the 15-minute outcome
    #: fades client entries the market context says will lose, holding for
    #: exactly 15 minutes. Fade-only -- every horizon test put the money on
    #: the fade side. Runs beside the mirror strategy inside the same wall
    #: and symbol-share limits.
    #: LIVE EQUITY CIRCUIT BREAKER. The replay calibrates drawdown on CLOSED
    #: daily P&L and cannot see the intraday floating book that accumulates
    #: between mirror exits. This guard watches the real thing: below
    #: equity_floor_frac of balance it trims the worst floaters until margin
    #: is safe and pauses NEW entries; new entries resume above resume_frac.
    equity_floor_frac: float = 0.75
    resume_frac: float = 0.85
    safe_margin_level: float = 400.0
    #: Strategy 1 (Mirror) master switch. On by default -- this is the copy/
    #: invert engine. When OFF the loop still SCORES every trade and streams the
    #: decision, but opens no new mirror entries; existing mirror positions are
    #: still managed to their exits. Lets S1 and S2 run independently.
    strategy1: bool = True
    strategy2: bool = False
    strategy2_fade_ceiling: float = 0.10
    strategy2_lots: float = 0.01          # per $5k equity, equal-size
    strategy2_hold_seconds: float = 900.0
    strategy2_max_positions: int = 60
    #: HOW lots are chosen. "client": proportional to the client's size
    #: (k x multiplier x dd_calibration). "magnitude": proportional to the
    #: magnitude model's predicted |move|/cost -- the sizing policy that won
    #: the real-flow shootout (+$13,417/day at $428 maxDD vs +$340 for
    #: client-proportional). Falls back to client sizing per-trade whenever
    #: the magnitude prediction is unavailable.
    sizing_policy: str = "client"
    #: "client" policy scale: OUR lots per CLIENT lot on a 5k-equity account
    #: (1.0 = a 0.01-lot client trade is copied at 0.01 lots on 5k equity,
    #: 0.20 lots on 100k). lots = client_lots x contract adjust x
    #: (equity / 5000) x this, then bounded by the drawdown budget, the
    #: leverage wall, the venue margin envelope and max_lots_per_trade.
    client_lots_per_5k: float = 1.0
    #: Lots per unit of predicted magnitude, equity-scaled from the replay's
    #: winning point ($5k account, factor 0.002).
    magnitude_lot_factor: float = 0.002
    #: Operator override of the backtest symbol multipliers, keyed by CANONICAL
    #: symbol (e.g. {"XAUUSD": 0.0} to exclude gold, 0.3 to quarter it). The
    #: backtest multiplier ignores our execution cost, so this is where live
    #: evidence (a symbol that bleeds after spread+commission) is applied.
    symbol_overrides: dict = field(default_factory=dict)
    #: Canonical symbol -> net open lots. While OUR net open position on that
    #: symbol (|long - short| across its venue variants, positions only) is at
    #: or under the figure, signals take the mirror (market) entry; above it
    #: the entry model's limits apply. {} = entry model always.
    mirror_entry_net_lots: dict = field(default_factory=dict)
    #: Measure that line on the SAME-SIDE net (our net exposure in the new
    #: trade's direction) instead of |net|: a signal that reduces the book
    #: always goes to market, one that adds to an already-loaded side waits.
    #: OOS event-driven study 14 Sep 2026: +3.7% P&L at identical drawdown.
    mirror_entry_same_side: bool = False
    #: Start the engine with the web server. A restart then means a restart of
    #: the WHOLE system -- the previous pattern of new server + engine left
    #: off is how open positions ended up unmanaged for hours.
    auto_start: bool = False

    #: A trade older than this is history, not a signal.
    max_signal_age_minutes: float = 15.0
    #: Consume the UAT kafka events store for DECISIONS. Off by default: the
    #: UAT cluster is incomplete and stale next to the production MySQL feed
    #: (~2.5s lag), and an incomplete feed makes silently wrong features.
    #: Quotes fallback and bar seeding still read the store either way.
    use_kafka_events: bool = False
    #: Poll the production MySQL databases alongside Kafka. The UAT Kafka
    #: cluster is mostly idle; the MySQL reporting proxy carries the real
    #: client flow seconds-fresh (measured: 115 new openings in 10 seconds).
    #: Identity de-duplication means the two sources can never double-fire,
    #: so this simply makes the engine source-agnostic.
    use_db_feed: bool = True
    #: Brackets. Zero disables -- see the module docstring for why that is the
    #: default rather than a timid choice.
    stop_multiple: float = 0.0
    target_multiple: float = 0.0

    poll_seconds: float = 5.0


def load_config() -> VantageConfig:
    import yaml
    if not CONFIG_PATH.exists():
        return VantageConfig()
    try:
        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        known = {f: raw[f] for f in VantageConfig.__dataclass_fields__ if f in raw}
        return VantageConfig(**known)
    except Exception:
        return VantageConfig()


def save_config(config: VantageConfig) -> None:
    import yaml
    CONFIG_PATH.write_text(yaml.safe_dump(asdict(config), sort_keys=False),
                           encoding="utf-8")


def write_example() -> None:
    if not EXAMPLE_PATH.exists():
        EXAMPLE_PATH.write_text(
            "# Copy to vantage.yaml and fill in. MT5 terminal must be installed.\n"
            "login: 12345678\n"
            'password: "CHANGE-ME"\n'
            'server: "VantageInternational-Demo"\n'
            'mode: "paper"\n', encoding="utf-8")


# ---------------------------------------------------------------- state
@dataclass
class EngineState:
    running: bool = False
    kill_switch: bool = False
    connected: bool = False
    last_error: str = ""
    started_at: float = 0.0
    scored: int = 0
    stale_skipped: int = 0
    acted: int = 0
    filled: int = 0
    blocked: int = 0
    scale_factor: float = 0.0
    strategy_dd_1x: float = 0.0
    log: list = field(default_factory=list)
    signals: list = field(default_factory=list)


_STATE = EngineState()
_LOCK = threading.Lock()
_THREAD: threading.Thread | None = None
_SCORES: list[float] = []          # rolling window for quantile routing
_SEEN: set = set()                 # deal identities already decided
_PAPER: dict[int, dict] = {}       # virtual positions
_PAPER_SEQ = [0]
_PAPER_PNL = [0.0]

#: A "start from scratch" line in the sand. When the account is re-funded, the
#: operator resets stats: every performance view then counts ONLY deals that
#: closed at or after this epoch, so a fresh $1000 balance shows a fresh record
#: instead of dragging in the previous run's P&L. Persisted so it survives
#: restarts; 0 means "since the beginning of the window".
_STATS_RESET_FILE = ARTIFACTS / "stats_reset.txt"


def stats_reset_at() -> float:
    try:
        return float(_STATS_RESET_FILE.read_text(encoding="utf-8").strip() or 0.0)
    except Exception:
        return 0.0


def reset_stats() -> dict:
    """Draw the line at NOW: persist the epoch, wipe the order log and the live
    counters, and clear the perf cache so the strategy panels start from zero."""
    epoch = time.time()
    try:
        _STATS_RESET_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATS_RESET_FILE.write_text(str(epoch), encoding="utf-8")
    except Exception:
        pass
    # In-memory engine counters and paper P&L.
    with _LOCK:
        _STATE.scored = _STATE.acted = _STATE.filled = 0
        _STATE.blocked = _STATE.stale_skipped = 0
        _STATE.signals = []
        _PAPER_PNL[0] = 0.0
    # Perf/reconcile caches so the new cutoff takes effect immediately.
    for cache in ("_PERF_CACHE", "_RECONCILE_CACHE"):
        obj = globals().get(cache)
        if isinstance(obj, dict):
            obj.clear()
    # The order log: archive the old rows out of the active table so the log and
    # every ticket-join start clean (kept in vantage_orders_archived for audit).
    try:
        import sqlite3
        with sqlite3.connect(ROOT / "app.db") as cx:
            cx.execute("CREATE TABLE IF NOT EXISTS vantage_orders_archived "
                       "AS SELECT * FROM vantage_orders WHERE 0")
            cx.execute("INSERT INTO vantage_orders_archived SELECT * FROM vantage_orders")
            cx.execute("DELETE FROM vantage_orders")
    except Exception:
        pass
    return {"reset_at": epoch}
_PENDING: dict[int, dict] = {}     # resting rescue limits (paper simulation)


def _log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')}  {message}"
    with _LOCK:
        _STATE.log.append(line)
        _STATE.log = _STATE.log[-200:]
    # Durable copy: the in-memory log dies with the process, and an engine
    # that failed at startup used to leave nothing to diagnose.
    try:
        with open(ARTIFACTS / "vantage_engine.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def state() -> EngineState:
    with _LOCK:
        return EngineState(**asdict(_STATE))


def set_kill_switch(engaged: bool) -> None:
    with _LOCK:
        _STATE.kill_switch = engaged
    _log("KILL SWITCH " + ("ENGAGED" if engaged else "released"))


# ---------------------------------------------------------------- MT5
def _mt5():
    import MetaTrader5 as mt5
    return mt5


#: Short-lived cache for MT5 reads. The terminal IPC holds the interpreter
#: while it waits; with the engine loop, the sizer's margin queries, the tape
#: and the dashboard all asking every few hundred milliseconds the web server
#: starved (242 threads, 2.4 cores, dead pages on 14 Sep 2026). Reads that
#: cannot change within a second share one answer.
_MT5_CACHE: dict = {}


def _mt5_cached(key: str, ttl: float, fetch):
    now = time.time()
    hit = _MT5_CACHE.get(key)
    if hit is not None and now - hit[0] < ttl:
        return hit[1]
    value = fetch()
    _MT5_CACHE[key] = (now, value)
    return value


def _mt5_cache_clear() -> None:
    """Drop every cached venue read the moment an order is accepted. Inside
    one loop tick a burst of signals otherwise all see the SAME free margin
    and the same strategy-margin figure (1-2 s caches), so each one sizes
    against room the previous order already took: 14 Sep 2026 a restart
    burst put 3 market buys + 8 limits through in one second and drove the
    margin level from 167% to 104%."""
    _MT5_CACHE.clear()


def connect(config: VantageConfig) -> tuple[bool, str]:
    try:
        mt5 = _mt5()
    except ImportError:
        return False, "MetaTrader5 package not installed"
    kwargs = {"path": config.terminal_path} if config.terminal_path else {}
    if not mt5.initialize(login=int(config.login or 0), password=config.password,
                          server=config.server, **kwargs):
        return False, f"MT5 login failed: {mt5.last_error()}"
    info = mt5.account_info()
    if info is None:
        return False, f"no account info: {mt5.last_error()}"
    # A symbol map resolved before this login (a paper-mode run, or a terminal
    # not yet attached) holds no venue names; left in place it blocked every
    # live order as "not listed on this account". Re-resolve against this login.
    global _VENUE_SYMBOLS
    _VENUE_SYMBOLS = None
    _SYMBOL_CACHE.clear()
    with _LOCK:
        _STATE.connected = True
    return True, f"connected {info.login} @ {info.server} ({info.company})"


def account_snapshot() -> dict:
    try:
        mt5 = _mt5()
        info = _mt5_cached("account_info", 1.0, mt5.account_info)
        if info is None:
            return {"available": False, "error": "not connected"}
        return {"available": True, "login": info.login, "server": info.server,
                "balance": float(info.balance), "equity": float(info.equity),
                "margin": float(info.margin), "free_margin": float(info.margin_free),
                "currency": info.currency, "profit": float(info.profit),
                "leverage": float(getattr(info, "leverage", 0) or 0)}
    except Exception as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}


def live_positions() -> list[dict]:
    try:
        mt5 = _mt5()
        raw = _mt5_cached("positions", 0.5, lambda: mt5.positions_get() or ())
        return [{"ticket": p.ticket, "symbol": p.symbol,
                 "direction": 1 if p.type == mt5.POSITION_TYPE_BUY else -1,
                 "lots": float(p.volume), "open_price": float(p.price_open),
                 "sl": float(p.sl), "tp": float(p.tp), "profit": float(p.profit),
                 "comment": p.comment,
                 "source": _REASSIGN.get(p.ticket) or _source_of(p.comment)}
                for p in raw]
    except Exception:
        return []


#: The comment IS the position's identity papers: it survives engine restarts,
#: process crashes, and the terminal's own view, so it must carry the FULL
#: source (server + login), not a tail. One letter per server keeps the whole
#: encoding inside MT5's comment budget: zx + stance + server + login + sNN
#: (live score at entry) is at most ~16 chars, e.g. `zxce9755616s87`.
_SERVER_CODES = {"mt4_live01": "a", "mt4_live02": "b", "mt4_live03": "c",
                 "mt4_live04": "d", "mt5_live01": "e", "mt5_dubai_live01": "f"}
_CODE_SERVERS = {v: k for k, v in _SERVER_CODES.items()}

#: Recycled positions: ticket -> the NEW source account now backing it. MT5
#: comments are immutable after open, so reassignment lives here, persisted in
#: sqlite and reloaded at engine start.
_REASSIGN: dict[int, str] = {}


def _comment_for(stance: str, account: str, value: float | None,
                 multiple: float | None = None) -> str:
    """Comment v2 -- the SORTABLE key, strategy digit first.

    `1c e 9755616 s87 m12` (no spaces): strategy 1, copy, mt5_live01 login
    9755616, score 0.87, predicted magnitude 12x cost. Strategy-2 comments
    start `2f`. MT5 truncates around 27 chars, so the comment carries the
    key and the orders ledger carries the full pipe-separated detail.
    """
    server, _, login = str(account).rpartition(":")
    code = _SERVER_CODES.get(server)
    score_part = f"s{int(round((value or 0.5) * 100)):02d}"
    magnitude_part = (f"m{min(int(round(multiple)), 99)}"
                      if multiple is not None and multiple > 0 else "")
    if code and login.isdigit():
        return f"1{stance[0]}{code}{login}{score_part}{magnitude_part}"[:27]
    return f"1{stance[0]}?{str(account)[-8:]}{score_part}"[:27]


def _source_of(comment: str) -> str:
    """Full account_key from a strategy-1 comment (v2 `1c...` or legacy
    `zx...`/`zfx-...`). Strategy-2 comments deliberately return "" -- their
    source is in the ledger, and a parsed source here would let mirror
    exits close a fade position early."""
    text = comment or ""
    if text[:1] == "2" or text.startswith("z2f"):
        return ""
    if text[:1] == "1" and len(text) >= 5 and text[1] in "ci":
        server = _CODE_SERVERS.get(text[2])
        digits = ""
        for character in text[3:]:
            if character.isdigit():
                digits += character
            else:
                break
        if server and digits:
            return f"{server}:{digits}"
        return ""
    if text.startswith("zx") and len(text) >= 5:
        server = _CODE_SERVERS.get(text[3])
        login = text[4:].split("s", 1)[0]
        if server and login.isdigit():
            return f"{server}:{login}"
    parts = text.split("-")
    return parts[-1] if len(parts) >= 3 and parts[0] == "zfx" else ""


def _load_reassignments() -> None:
    with sqlite3.connect(DB) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS vantage_reassign "
            "(ticket INTEGER PRIMARY KEY, source TEXT, created REAL)")
        _REASSIGN.update({int(t): str(s) for t, s in connection.execute(
            "SELECT ticket, source FROM vantage_reassign").fetchall()})


def _save_reassignment(ticket: int, source: str) -> None:
    _REASSIGN[int(ticket)] = source
    with sqlite3.connect(DB) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS vantage_reassign "
            "(ticket INTEGER PRIMARY KEY, source TEXT, created REAL)")
        connection.execute(
            "INSERT OR REPLACE INTO vantage_reassign VALUES (?,?,?)",
            (int(ticket), source, time.time()))


def _prune_reassignments(open_tickets: set[int]) -> None:
    dead = [t for t in _REASSIGN if t not in open_tickets]
    for ticket in dead:
        _REASSIGN.pop(ticket, None)
    if dead:
        with sqlite3.connect(DB) as connection:
            connection.executemany(
                "DELETE FROM vantage_reassign WHERE ticket = ?",
                [(t,) for t in dead])


# -- symbol resolution ------------------------------------------------------
#: A source server's ticker is not necessarily the venue's. Ordering a symbol
#: the account does not list simply errors, which is why nothing reached MT5.
_SYMBOL_CACHE: dict[str, str | None] = {}
_VENUE_SYMBOLS: dict[str, str] | None = None


def _venue_symbols() -> dict[str, str]:
    """CANONICAL instrument -> this venue's tradeable name.

    Both sides go through the SAME canonicaliser the whole pipeline uses
    (trade_feed._canonical, which strips broker suffixes across MT4 and MT5
    stream symbols), so a source server's `XAUUSD.r` and the venue's
    `XAUUSD+` meet at the same key instead of relying on ad-hoc root
    stripping. Shortest venue name wins per canonical -- the plain
    instrument over its suffixed variants."""
    global _VENUE_SYMBOLS
    if _VENUE_SYMBOLS is not None:
        return _VENUE_SYMBOLS
    from webapp.trade_feed import _canonical
    mapping: dict[str, tuple] = {}   # key -> (untradeable, name_len, name)
    try:
        for info in (_mt5().symbols_get() or []):
            root = "".join(c for c in info.name.upper() if c.isalnum())
            canon = _canonical(info.name) or root
            # TRADEABLE FIRST, then shortest. This venue lists DISABLED
            # display twins under the plain names (EURUSD) beside the
            # tradeable `+` variants (EURUSD+); shortest-name-wins routed
            # every fx order to the dead twin while gold worked only
            # because no plain XAUUSD exists.
            SYMBOL_TRADE_MODE_FULL = 4
            candidate = (info.trade_mode != SYMBOL_TRADE_MODE_FULL,
                         len(info.name), info.name)
            for key in {canon, root}:
                if key and (key not in mapping or candidate < mapping[key]):
                    mapping[key] = candidate
    except Exception:
        pass
    resolved = {key: value[2] for key, value in mapping.items()}
    # Cross-naming aliases: instruments the sources and this venue simply
    # name differently (the canonicaliser cannot equate US30 with DJ30).
    for source_name, venue_name in (("US30", "DJ30"), ("US30", "DOW")):
        if source_name not in resolved and venue_name in resolved:
            resolved[source_name] = resolved[venue_name]
    if not resolved:
        # No terminal attached (or no symbols returned): do not cache an empty
        # listing -- the next call, once connected, resolves the real one.
        return resolved
    _VENUE_SYMBOLS = resolved
    # Add every tradeable instrument to Market Watch up front, so quotes stream
    # and orders never bounce because a symbol was not selected on this account.
    selected = 0
    try:
        for name in set(resolved.values()):
            if _mt5().symbol_select(name, True):
                selected += 1
        _log(f"market watch: selected {selected}/{len(set(resolved.values()))} "
             f"tradeable symbols")
    except Exception:
        pass
    return _VENUE_SYMBOLS


def resolve_symbol(symbol: str) -> str | None:
    """The tradeable name on THIS account, or None if it is not listed."""
    if symbol in _SYMBOL_CACHE:
        return _SYMBOL_CACHE[symbol]
    from webapp.trade_feed import _canonical
    venue = _venue_symbols()
    root = "".join(c for c in str(symbol).upper() if c.isalnum())
    resolved = venue.get(_canonical(symbol) or root) or venue.get(root)
    if resolved is None:
        for strip in ("MIN", "MICRO", "247", "PRO", "E", "X", "C", "M"):
            if root.endswith(strip) and len(root) > len(strip) + 3:
                resolved = venue.get(root[:-len(strip)])
                if resolved:
                    break
    if resolved:
        try:
            _mt5().symbol_select(resolved, True)
        except Exception:
            pass
    if venue:                     # "not listed" is only an answer against a real listing
        _SYMBOL_CACHE[symbol] = resolved
    return resolved


def current_quote(symbol: str) -> tuple[float, float] | None:
    """(bid, ask) from the terminal, or the live quote stream as a fallback."""
    try:
        tick = _mt5().symbol_info_tick(symbol)
        if tick and tick.bid and tick.ask:
            return float(tick.bid), float(tick.ask)
    except Exception:
        pass
    try:
        from webapp.kafka_service import STORE, _canonical
        if not STORE.exists():
            return None
        # Timestamps in the store are naive UTC; DuckDB's NOW() is LOCAL time,
        # so comparing against it shifts every window by the machine's UTC
        # offset -- freshness checks silently failed by an hour on this box.
        # Cutoffs are therefore computed in Python (UTC) and passed in.
        cutoff = datetime.utcnow() - timedelta(minutes=10)
        with _store() as connection:
            row = connection.execute("""
                SELECT bid, ask FROM quotes
                WHERE (symbol = ? OR canonical = ?) AND event_time > ?
                ORDER BY event_time DESC LIMIT 1
            """, [symbol, _canonical(symbol) or symbol, cutoff]).fetchone()
        return (float(row[0]), float(row[1])) if row and row[0] and row[1] else None
    except Exception:
        return None


# ---------------------------------------------------------------- store
def _store():
    """A cursor onto the process-wide live-stream DuckDB instance. Opening the
    file separately here created a SECOND instance racing the materialiser for
    the OS lock -- with quotes streaming, every read failed 'store busy'.
    Cursors of the one shared instance interleave with writes natively."""
    from webapp.kafka_service import shared_cursor
    return shared_cursor()


#: Schema-level discriminators. MT5 deals carry `entry` (ENTRY_IN opens,
#: ENTRY_OUT closes); MT4 discriminates by envelope (`orderCreated` opens).
#: Rows stored before those columns existed fall back to the profit heuristic.
_OPENING = """
  AND action IN ('DEAL_BUY','DEAL_SELL') AND login IS NOT NULL
  AND symbol IS NOT NULL AND volume > 0
  AND (entry IN ('ENTRY_IN','ENTRY_INOUT') OR event_kind = 'orderCreated'
       OR ((entry IS NULL OR entry = '') AND event_kind IS NULL
           AND (profit IS NULL OR profit = 0)))
"""
_CLOSING = """
  AND action IN ('DEAL_BUY','DEAL_SELL') AND login IS NOT NULL AND symbol IS NOT NULL
  AND (entry IN ('ENTRY_OUT','ENTRY_OUT_BY','ENTRY_INOUT')
       OR (event_kind IN ('orderUpdated','orderClosedBy') AND profit IS NOT NULL AND profit <> 0)
       OR ((entry IS NULL OR entry = '') AND event_kind IS NULL
           AND profit IS NOT NULL AND profit <> 0))
"""
# ENTRY_INOUT (netting-account REVERSAL) is deliberately in BOTH streams: the
# one deal closes the client's old side AND opens the flip, so it must
# mirror-close our copy of the old side and still open the new one.


def _prepare(frame: pd.DataFrame, deflate_cents: bool = True) -> pd.DataFrame:
    if frame.empty:
        return frame
    lots = pd.to_numeric(frame["volume"], errors="coerce")
    frame["lots"] = np.where(lots > RAW_VOLUME_THRESHOLD, lots / 10000.0, lots)
    frame["direction"] = np.where(frame["action"] == "DEAL_BUY", 1, -1)
    frame["account_key"] = (frame["server"].astype(str) + ":"
                            + frame["login"].astype("int64").astype(str))
    # CENT DEFLATION for the kafka-store path ONLY: those rows arrive raw (a
    # cent client's 0.01 lots read as 1.00). The MySQL feed's row builders
    # (trade_feed._rows_mt4/_rows_mt5) already divide lots AND profit by 100
    # for cent logins, so that caller passes deflate_cents=False -- until
    # 14 Sep 2026 it was deflated twice here, and every cent client's lots
    # and P&L reached the sizer, the live form and the tape 100x too small.
    if not deflate_cents:
        return frame
    try:
        from webapp.trade_feed import cent_logins
        for server in frame["server"].astype(str).unique():
            cents = cent_logins(server)
            if cents:
                mask = ((frame["server"].astype(str) == server)
                        & frame["login"].astype("int64").isin(cents))
                if mask.any():
                    frame.loc[mask, "lots"] = frame.loc[mask, "lots"] / 100.0
                    if "profit" in frame.columns:
                        frame.loc[mask, "profit"] = pd.to_numeric(
                            frame.loc[mask, "profit"], errors="coerce") / 100.0
    except Exception:
        pass
    return frame


def poll_openings(since: datetime, limit: int = 500) -> pd.DataFrame:
    with _store() as connection:
        frame = connection.execute(f"""
            SELECT DISTINCT server, login, symbol, action, volume, price,
                   event_time, ingested_at
            FROM events WHERE ingested_at > ? {_OPENING}
            ORDER BY ingested_at LIMIT ?
        """, [since, limit]).df()
    return _prepare(frame)


def poll_closings(since: datetime) -> pd.DataFrame:
    with _store() as connection:
        frame = connection.execute(f"""
            SELECT DISTINCT server, login, symbol, action, volume, profit,
                   entry, event_time, ingested_at
            FROM events WHERE ingested_at > ? {_CLOSING}
            ORDER BY ingested_at
        """, [since]).df()
    return _prepare(frame)


def recent_openings(limit: int = 10) -> pd.DataFrame:
    """The newest TRADES (by trade time), for the stream panel."""
    with _store() as connection:
        frame = connection.execute(f"""
            SELECT DISTINCT server, login, symbol, action, volume, price,
                   event_time, ingested_at
            FROM events WHERE 1=1 {_OPENING}
            ORDER BY event_time DESC LIMIT ?
        """, [limit * 6]).df()
    frame = _prepare(frame)
    if frame.empty:
        return frame
    return frame.drop_duplicates(
        ["account_key", "symbol", "direction", "lots", "price", "event_time"]).head(limit)


def stream_health() -> dict:
    """Quotes are the heartbeat: on a quiet venue deals can be hours apart."""
    from webapp.kafka_service import STORE
    if not STORE.exists():
        return {"available": False, "reason": "consumer has never run"}
    try:
        # UTC cutoffs computed here, never SQL NOW(): DuckDB's NOW() is local
        # time while the store holds naive UTC, and that one-hour skew is why
        # every "last N minutes" counter read zero against a flowing feed.
        ten_min = datetime.utcnow() - timedelta(minutes=10)
        one_min = datetime.utcnow() - timedelta(minutes=1)
        with _store() as connection:
            deal_last, recent = connection.execute("""
                SELECT MAX(event_time),
                       COUNT(*) FILTER (WHERE ingested_at > ?)
                FROM events""", [ten_min]).fetchone()
            try:
                # The quotes table has no ingested_at column -- its event_time
                # IS effectively arrival for a live tick (broker timestamp).
                # Referencing the missing column threw, was swallowed, and the
                # heartbeat read "0 quotes/min" against a visibly live feed.
                quote_arrival, quote_count = connection.execute("""
                    SELECT MAX(event_time),
                           COUNT(*) FILTER (WHERE event_time > ?)
                    FROM quotes""", [one_min]).fetchone()
            except Exception:
                quote_arrival, quote_count = None, 0

        def age(value):
            if value is None:
                return None
            return (datetime.utcnow() - pd.Timestamp(value).to_pydatetime()).total_seconds()

        return {"available": True, "last_trade": str(deal_last),
                "trade_age_seconds": age(deal_last),
                "quote_age_seconds": age(quote_arrival),
                "quotes_last_minute": int(quote_count or 0),
                "events_last_10min": int(recent or 0)}
    except Exception as error:
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}


# ---------------------------------------------------------------- live bars
#: Training builds market context from the trade tape itself -- per-symbol
#: hourly bars, statistics through the previous COMPLETED hour. The engine
#: sees the same tape live, so it maintains the identical bars here; without
#: this every ctx feature went to the model as NaN and its most important
#: inputs were blind at fill time.
from collections import deque

_BAR_STATE: dict[str, dict] = {}                 # symbol -> current (partial) hour
_BAR_HISTORY: dict[str, deque] = {}              # symbol -> completed bars


def _note_trade(symbol: str, price: float, direction: int, when: datetime) -> None:
    if not price or price <= 0:
        return
    from webapp.trade_feed import _canonical
    symbol = _canonical(symbol)      # one bar series per instrument, not per suffix
    hour = when.replace(minute=0, second=0, microsecond=0)
    state = _BAR_STATE.get(symbol)
    if state is None or state["hour"] != hour:
        if state is not None and state["prices"]:
            _BAR_HISTORY.setdefault(symbol, deque(maxlen=200)).append(
                (state["hour"], float(np.median(state["prices"])),
                 float(np.mean(state["flows"]))))
        _BAR_STATE[symbol] = state = {"hour": hour, "prices": [], "flows": []}
    state["prices"].append(float(price))
    state["flows"].append(1.0 if direction > 0 else -1.0)


def _ctx_features(symbol: str, direction: int) -> dict:
    """Market context + normalised technicals from completed bars only --
    the live mirror of the training builder's shift(1)."""
    from webapp.trade_feed import _canonical
    history = _BAR_HISTORY.get(_canonical(symbol))
    if not history or len(history) < 2:
        return {}
    prices = np.array([bar[1] for bar in history], dtype="float64")
    out: dict = {"ctx_flow_prev": history[-1][2]}

    def ret(window: int) -> float:
        return (prices[-1] / prices[-1 - window] - 1.0
                if len(prices) > window else np.nan)

    r1, r4, r24 = ret(1), ret(4), ret(24)
    out["ctx_return_1h"], out["ctx_return_4h"], out["ctx_return_24h"] = r1, r4, r24
    returns = np.diff(prices) / prices[:-1]
    vol = float(np.std(returns[-24:], ddof=1)) if len(returns) >= 4 else np.nan
    out["ctx_vol_24h"] = vol
    window = prices[-24:]
    if len(window) >= 6:
        std = float(np.std(window, ddof=1))
        out["zscore_24h"] = ((prices[-1] - float(np.mean(window))) / std
                             if std > 0 else np.nan)
        span = float(window.max() - window.min())
        out["range_pos_24h"] = ((prices[-1] - float(window.min())) / span
                                if span > 0 else np.nan)
    if np.isfinite(vol) and vol > 0:
        if np.isfinite(r1):
            out["mom_1h_vol"] = float(np.clip(r1 / vol, -20, 20))
        if np.isfinite(r4):
            out["mom_4h_vol"] = float(np.clip(r4 / (vol * 2.0), -20, 20))
            out["with_momentum"] = direction * out["mom_4h_vol"]
        if np.isfinite(r24):
            out["mom_24h_vol"] = float(np.clip(r24 / (vol * 4.9), -20, 20))
        if len(returns) >= 48:
            vols = np.array([np.std(returns[max(0, t - 24):t], ddof=1)
                             for t in range(24, len(returns) + 1)])
            baseline = float(np.mean(vols[-168:]))
            if baseline > 0:
                out["vol_regime"] = vol / baseline
    out["momentum_align"] = direction * (
        np.sign(np.nan_to_num(r1)) + np.sign(np.nan_to_num(r4))
        + np.sign(np.nan_to_num(r24))) / 3.0
    return out


def _seed_bars() -> None:
    """Warm the bar history from the local events store, so context features
    are real from the first minute instead of after a day of live flow."""
    try:
        cutoff = datetime.utcnow() - timedelta(hours=250)
        with _store() as connection:
            frame = connection.execute(f"""
                SELECT symbol, date_trunc('hour', event_time) AS bar_hour,
                       median(price) AS price,
                       avg(CASE WHEN action = 'DEAL_BUY' THEN 1.0 ELSE -1.0 END) AS flow
                FROM events WHERE event_time > ? {_OPENING}
                GROUP BY 1, 2 HAVING median(price) > 0 ORDER BY 1, 2
            """, [cutoff]).df()
        if len(frame):
            # Merge suffix variants (XAUUSD, XAUUSDmin, XAUUSD+) into ONE
            # canonical series -- the seed reads the kafka store's spellings,
            # which differ from the MySQL feed's, and an unmerged seed leaves
            # every "+" symbol cold for ~24h after a restart.
            from webapp.trade_feed import _canonical
            frame["canon"] = frame["symbol"].map(lambda s: _canonical(str(s)))
            agg = (frame.groupby(["canon", "bar_hour"], as_index=False)
                   .agg(price=("price", "median"), flow=("flow", "mean"))
                   .sort_values(["canon", "bar_hour"]))
            for canon, group in agg.groupby("canon"):
                history = _BAR_HISTORY.setdefault(str(canon), deque(maxlen=200))
                for _, row in group.iterrows():
                    history.append((row["bar_hour"], float(row["price"]),
                                    float(row["flow"])))
            _log(f"bar history seeded: {len(agg):,} hourly bars, "
                 f"{agg['canon'].nunique()} canonical symbols")
    except Exception as error:
        _log(f"bar seed skipped: {type(error).__name__}: {error}")


_SYMBOL_CODES: dict[str, float] | None = None


def _symbol_codes() -> dict[str, float]:
    """The TRAINING frame's category mapping, persisted at train time."""
    global _SYMBOL_CODES
    if _SYMBOL_CODES is None:
        try:
            lines = (ARTIFACTS / "quant_symbols.txt").read_text(
                encoding="utf-8").splitlines()
            codes = {s: float(i) for i, s in enumerate(lines)}
            # ALSO key by canonical, so a live "XAUUSD+" / "EURUSD+" resolves to
            # the training code of its base instead of NaN. First variant (the
            # base spelling comes first in the list) wins the canonical slot.
            from webapp.trade_feed import _canonical
            for i, s in enumerate(lines):
                codes.setdefault(_canonical(s), float(i))
            _SYMBOL_CODES = codes
        except Exception:
            _SYMBOL_CODES = {}
    return _SYMBOL_CODES


_FUNDING: pd.DataFrame | None = None
_FUNDING_COLUMNS = ("deposits_to_date", "withdrawals_to_date",
                    "net_funding_to_date", "deposit_count_to_date",
                    "withdrawal_count_to_date", "days_since_deposit",
                    "days_since_withdrawal", "withdrawal_ratio", "funding_churn")


def _funding_snapshot() -> pd.DataFrame:
    """Every client's funding state as of today, via the SAME attach the
    training pipeline uses, so live and trained features agree by construction.
    Disk-cached per calendar day: the inputs only change with a backfill."""
    global _FUNDING
    if _FUNDING is not None:
        return _FUNDING
    today = str(pd.Timestamp.utcnow().tz_localize(None).floor("D").date())
    cache = ARTIFACTS / "quant_funding_cache.parquet"
    with _HEAVY_LOCK:
        if _FUNDING is not None:
            return _FUNDING
        if _warm_meta().get("funding_day") == today and cache.exists():
            try:
                _FUNDING = pd.read_parquet(cache)
                return _FUNDING
            except Exception:
                pass
        try:
            from webapp import cashflow_store
            movements = cashflow_store.read_cashflows()
            if movements.empty:
                _FUNDING = pd.DataFrame()
                return _FUNDING
            daily = cashflow_store.daily_features(movements)
            probe = pd.DataFrame(
                {"account_key": daily["account_key"].astype(str).unique()})
            probe["day"] = pd.Timestamp.utcnow().tz_localize(None).floor("D")
            probe = cashflow_store.attach(probe, daily)
            _FUNDING = probe.set_index("account_key")
            try:
                _FUNDING.to_parquet(cache)
                _warm_update(funding_day=today)
            except Exception:
                pass
        except Exception:
            _FUNDING = pd.DataFrame()
    return _FUNDING


#: LIVE client form, straight from the closings stream (~2.5s behind the
#: venue). The artifact's history aggregates are a training-time snapshot --
#: for mt4_live02:2860909 they said 2 trades while MySQL held 15,850 and the
#: client was 60-for-61 on the day. These rolling windows put the client's
#: CURRENT form into the features, weighted by sample count so a hot streak
#: overwhelms a thin snapshot within minutes.
_LIVE_CLIENT: dict[str, deque] = {}
_SEEN_CLOSES: set = set()


def _note_client_close(account: str, profit: float) -> None:
    window = _LIVE_CLIENT.get(account)
    if window is None:
        if len(_LIVE_CLIENT) > 150_000:
            _LIVE_CLIENT.clear()
        window = _LIVE_CLIENT[account] = deque(maxlen=200)
    window.append(float(profit))
    # A close changes the account's history: the cached parity state and tape
    # rebuild lazily at the account's next score.
    try:
        invalidate_parity(account)
    except Exception:
        pass


def _live_stats(account: str) -> dict | None:
    window = _LIVE_CLIENT.get(account)
    if not window:
        return None
    values = list(window)
    return {"n": len(values), "wins": sum(1 for v in values if v > 0),
            "sum": sum(values), "abs": sum(abs(v) for v in values),
            "last5": sum(values[-5:])}


_AD_SNAPSHOT: pd.DataFrame | None = None


def _account_day_snapshot() -> pd.DataFrame:
    """Each account's LATEST row from the account-day corpus -- the same
    cached frame training joins, so live and trained values agree."""
    global _AD_SNAPSHOT
    if _AD_SNAPSHOT is not None:
        return _AD_SNAPSHOT
    from webapp.trade_features import _AD_DIR, AD_COLUMNS, AD_FEATURES
    source = _AD_DIR / "model_frame.parquet"
    cache = ARTIFACTS / "quant_ad_cache.parquet"
    with _HEAVY_LOCK:
        if _AD_SNAPSHOT is not None:
            return _AD_SNAPSHOT
        try:
            stamp = source.stat().st_mtime
            if cache.exists() and _warm_meta().get("ad_mtime") == stamp:
                _AD_SNAPSHOT = pd.read_parquet(cache)
                return _AD_SNAPSHOT
            corpus = pd.read_parquet(
                source, columns=["account_key", "decision_day"] + AD_COLUMNS)
            corpus["decision_day"] = pd.to_datetime(corpus["decision_day"])
            latest = (corpus.sort_values("decision_day", kind="mergesort")
                      .groupby("account_key", observed=True).last()
                      .rename(columns=dict(zip(AD_COLUMNS, AD_FEATURES))))
            for name in AD_FEATURES:
                latest[name] = pd.to_numeric(
                    latest[name], errors="coerce").astype("float32")
            _AD_SNAPSHOT = latest[AD_FEATURES]
            try:
                _AD_SNAPSHOT.to_parquet(cache)
                _warm_update(ad_mtime=stamp)
            except Exception:
                pass
        except Exception:
            _AD_SNAPSHOT = pd.DataFrame()
    return _AD_SNAPSHOT


_ACTIVITY: dict[str, tuple] = {}   # account -> (last open, date, count today)


def _activity_features(account: str, when: datetime, update: bool) -> dict:
    """trades_today / hours_since_last from PRIOR state (training's cumcount
    counts earlier trades only), then optionally note this trade."""
    prior = _ACTIVITY.get(account)
    out: dict = {}
    if prior is not None:
        last, day_key, count = prior
        out["hours_since_last"] = min(
            max((when - last).total_seconds(), 0.0) / 3600.0, 24.0 * 14)
        out["trades_today"] = float(count if day_key == when.date() else 0)
    if update:
        if prior is not None and prior[1] == when.date():
            _ACTIVITY[account] = (when, when.date(), prior[2] + 1)
        else:
            _ACTIVITY[account] = (when, when.date(), 1)
        if len(_ACTIVITY) > 200_000:
            _ACTIVITY.clear()
    return out


# ---------------------------------------------------------------- model
_BOOSTER = None
_FEATURE_NAMES: list[str] | None = None
_HISTORY: pd.DataFrame | None = None
_HISTORY_STAMP = None


def booster():
    global _BOOSTER, _FEATURE_NAMES
    if _BOOSTER is None:
        path = ARTIFACTS / "quant_model.txt"
        if path.exists():
            import lightgbm as lgb
            _BOOSTER = lgb.Booster(model_file=str(path))
            _FEATURE_NAMES = (ARTIFACTS / "quant_model_features.txt").read_text(
                encoding="utf-8").splitlines()
    return _BOOSTER


# Per-class routing + calibration (fixes gold's 93% skewing the pooled model and
# the anchors being applied to uncalibrated probabilities). Fully backward
# compatible: with no per-class artifacts these fall back to the pooled model
# and the raw score, so behaviour is unchanged until a per-class retrain lands.
_CLASS_BOOSTERS: dict | None = None
_CALIBRATORS: dict | None = None
_MCLASS_MAP = {"gold": "metals", "silver": "metals", "fx": "fx",
               "index/other": "index", "crypto": "crypto"}


def _class_boosters() -> dict:
    global _CLASS_BOOSTERS
    if _CLASS_BOOSTERS is None:
        _CLASS_BOOSTERS = {}
        try:
            import lightgbm as lgb
            for cls in ("metals", "fx", "index", "crypto"):
                p = ARTIFACTS / f"quant_model_{cls}.txt"
                if p.exists():
                    _CLASS_BOOSTERS[cls] = lgb.Booster(model_file=str(p))
        except Exception:
            _CLASS_BOOSTERS = {}
    return _CLASS_BOOSTERS


def _calibrators() -> dict:
    global _CALIBRATORS
    if _CALIBRATORS is None:
        _CALIBRATORS = {}
        p = ARTIFACTS / "quant_calibrators.json"
        if p.exists():
            try:
                import json
                _CALIBRATORS = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                _CALIBRATORS = {}
    return _CALIBRATORS


def _model_class(symbol: str) -> str:
    from webapp.trade_features import symbol_class
    return _MCLASS_MAP.get(symbol_class(symbol), "index")


def _class_enabled(setting, symbol: str) -> bool:
    """True when the symbol's model-class is in a comma-separated class list
    ('metals,index,crypto'); an empty/None setting or '*' means every class.
    Lets the entry and exit models be deployed per class from the OOS study."""
    text = str(setting or "").strip()
    if not text or text == "*":
        return True
    wanted = {t.strip().lower() for t in text.split(",") if t.strip()}
    return _model_class(symbol) in wanted


def _predict_calibrated(symbol: str, vector, pooled) -> float:
    """Route to the symbol's class booster (pooled fallback), then apply the
    class isotonic calibrator (raw fallback). With no per-class artifacts this
    is exactly the pooled raw prediction -- unchanged legacy behaviour."""
    cls = _model_class(symbol)
    mdl = _class_boosters().get(cls) or pooled
    raw = float(mdl.predict(vector)[0])
    cal = _calibrators().get(cls)
    if cal and cal.get("x"):
        return float(np.interp(raw, np.asarray(cal["x"], dtype=float),
                               np.asarray(cal["y"], dtype=float)))
    return raw


# Per-class REGRESSORS (per-lot E, exit FE, entry = MAE q50, MAE q80 tail) --
# the trainer writes quant_<name>_<class>.txt beside each pooled file; route by
# class, fall back to the pooled model, so legacy artifacts behave unchanged.
_CLASS_REGRESSORS: dict = {}
_MAE_TAIL_MODEL = None


def _class_regressor(name: str, cls: str):
    key = (name, cls)
    if key not in _CLASS_REGRESSORS:
        path = ARTIFACTS / f"quant_{name}_{cls}.txt"
        try:
            import lightgbm as lgb
            _CLASS_REGRESSORS[key] = (lgb.Booster(model_file=str(path))
                                      if path.exists() else False)
        except Exception:
            _CLASS_REGRESSORS[key] = False
    return _CLASS_REGRESSORS[key] or None


def _predict_regressor(name: str, symbol: str, vector, pooled) -> float | None:
    """Class booster for `name` if present, else the pooled one; >= 0."""
    model = _class_regressor(name, _model_class(symbol)) or pooled
    if model is None:
        return None
    try:
        return max(0.0, float(model.predict(vector)[0]))
    except Exception:
        return None


def _mae_tail_model():
    """Pooled MAE q80 (tail pre-profit drawdown, adverse bps) regressor."""
    global _MAE_TAIL_MODEL
    if _MAE_TAIL_MODEL is None:
        path = ARTIFACTS / "quant_mae_q80.txt"
        try:
            import lightgbm as lgb
            _MAE_TAIL_MODEL = lgb.Booster(model_file=str(path)) if path.exists() else False
        except Exception:
            _MAE_TAIL_MODEL = False
    return _MAE_TAIL_MODEL or None


# The scorer's latest predicted TAIL drawdown per canonical symbol, read by the
# sizer moments later in the same event's execute() -- keyed by symbol only
# (drawdown depth is volatility-scaled, applied symmetrically to copy/invert).
_MAE_TAIL: dict = {}
_MAE_TAIL_TTL = 120.0


def _note_mae_tail(symbol: str, bps: float | None) -> None:
    if bps is None:
        return
    from webapp.trade_feed import _canonical
    _MAE_TAIL[_canonical(symbol) or symbol] = (float(bps), time.time())


def _predicted_stress(config, venue_symbol: str) -> tuple[float, float | None]:
    """(stress fraction for this trade, predicted tail bps or None).

    The predicted tail adverse move (MAE q80) replaces the fixed
    dd_stress_move only when LARGER: the model may tighten sizing for a trade
    expected to draw down deeply, never loosen the operator's floor."""
    base = max(float(getattr(config, "dd_stress_move", 0.01)), 1e-4)
    if not getattr(config, "mae_sizing", True):
        return base, None
    from webapp.trade_feed import _canonical
    hit = _MAE_TAIL.get(_canonical(venue_symbol) or venue_symbol)
    if not hit or time.time() - hit[1] > _MAE_TAIL_TTL:
        return base, None
    return max(base, hit[0] / 1e4), hit[0]


_S2_MODEL = None
_S2_MAGNITUDE = None
_S2_FEATURES: list[str] | None = None
_S2_CLASSES: set | None = None


_S2_CLASS_MODELS: dict | None = None


def _strategy2_model():
    global _S2_MODEL, _S2_MAGNITUDE, _S2_FEATURES, _S2_CLASS_MODELS
    if _S2_MODEL is None:
        path = ARTIFACTS / "strategy2_model.txt"
        names = ARTIFACTS / "strategy2_features.txt"
        if path.exists() and names.exists():
            try:
                import lightgbm as lgb
                _S2_MODEL = lgb.Booster(model_file=str(path))
                _S2_FEATURES = names.read_text(encoding="utf-8").splitlines()
                magnitude_path = ARTIFACTS / "strategy2_magnitude.txt"
                if magnitude_path.exists():
                    _S2_MAGNITUDE = lgb.Booster(model_file=str(magnitude_path))
                # PER-CLASS models beat the pooled one at the 5m horizon;
                # the pooled model remains the fallback.
                _S2_CLASS_MODELS = {}
                for cls, filename in (("fx", "strategy2_fx.txt"),
                                      ("index/other", "strategy2_index.txt"),
                                      ("crypto", "strategy2_crypto.txt")):
                    class_path = ARTIFACTS / filename
                    if class_path.exists():
                        _S2_CLASS_MODELS[cls] = lgb.Booster(
                            model_file=str(class_path))
            except Exception:
                _S2_MODEL = False
        else:
            _S2_MODEL = False
    return _S2_MODEL or None


def _strategy2_classes() -> set:
    """Symbol classes S2 may trade -- MEASURED viability, written by the
    trainer. Missing/empty file = trade nothing (fail-closed: an unmeasured
    class is an unvalidated class)."""
    global _S2_CLASSES
    if _S2_CLASSES is None:
        try:
            _S2_CLASSES = {line.strip() for line in
                           (ARTIFACTS / "strategy2_classes.txt")
                           .read_text(encoding="utf-8").splitlines()
                           if line.strip()}
        except Exception:
            _S2_CLASSES = set()
    return _S2_CLASSES


_MAGNITUDE = None


def _magnitude_model():
    """The |pnl|/cost regressor trained on the SAME features; None if the
    artifact has not been produced yet."""
    global _MAGNITUDE
    if _MAGNITUDE is None:
        path = ARTIFACTS / "quant_magnitude.txt"
        if path.exists():
            try:
                import lightgbm as lgb
                _MAGNITUDE = lgb.Booster(model_file=str(path))
            except Exception:
                _MAGNITUDE = False
        else:
            _MAGNITUDE = False
    return _MAGNITUDE or None


_EXIT_FE = None


def _exit_fe_model():
    """Favorable-excursion regressor (quantile a=0.35 on the same features):
    predicted bps available in OUR direction before the client's exit. The
    OOS validation (Aug 20-31): TP at 1.0x prediction beat mirror-only
    (PF 3.46 -> 4.01, net +1.4%) -- and live it also frees winning slots
    early on a capacity-bound account."""
    global _EXIT_FE
    if _EXIT_FE is None:
        path = ARTIFACTS / "quant_exit_fe.txt"
        if path.exists():
            try:
                import lightgbm as lgb
                _EXIT_FE = lgb.Booster(model_file=str(path))
            except Exception:
                _EXIT_FE = False
        else:
            _EXIT_FE = False
    return _EXIT_FE or None


_PERLOT = None


def _perlot_model():
    """$ per CANONICAL cent-adjusted lot regressor (target E) on the SAME
    features. The four-arm OOS study (S1/E/S/T, Aug 10-31) had it beating
    the deployed cost-multiple formula by 7-17% $/slot-hour at every
    selectivity level, so the edge calc prefers it when the artifact
    exists; the cost-multiple stays as the fallback."""
    global _PERLOT
    if _PERLOT is None:
        path = ARTIFACTS / "quant_perlot.txt"
        if path.exists():
            try:
                import lightgbm as lgb
                _PERLOT = lgb.Booster(model_file=str(path))
            except Exception:
                _PERLOT = False
        else:
            _PERLOT = False
    return _PERLOT or None


_ENTRY_MODEL = None


def _entry_model():
    """The entry-improvement regressor: predicted bps of better entry
    available between the client's entry and their exit (>= 0), trained on
    the SAME 249 features as the stance model. OOS study (Aug 10-31):
    waiting only when the prediction >= ~12bps beat client-price entries
    +$845k vs +$483k compounded at a $2k start, with a shallower relative
    drawdown. None until the artifact exists."""
    global _ENTRY_MODEL
    if _ENTRY_MODEL is None:
        path = ARTIFACTS / "entry_model.txt"
        if path.exists():
            try:
                import lightgbm as lgb
                _ENTRY_MODEL = lgb.Booster(model_file=str(path))
            except Exception:
                _ENTRY_MODEL = False
        else:
            _ENTRY_MODEL = False
    return _ENTRY_MODEL or None


def account_history() -> pd.DataFrame:
    """Per-account aggregates -- FEATURES for scoring, never a decision."""
    global _HISTORY, _HISTORY_STAMP
    from webapp import model_service as ms
    meta = ms.artifact_meta(ms.VIEW_QUANT) or {}
    stamp = meta.get("trained_at")
    if _HISTORY is not None and _HISTORY_STAMP == stamp:
        return _HISTORY
    with _HEAVY_LOCK:
        if _HISTORY is not None and _HISTORY_STAMP == stamp:
            return _HISTORY
        return _account_history_locked(stamp)


def _account_history_locked(stamp) -> pd.DataFrame:
    global _HISTORY, _HISTORY_STAMP
    cache = ARTIFACTS / "quant_history_cache.parquet"
    if _warm_meta().get("stamp") == stamp and cache.exists():
        try:
            candidate = pd.read_parquet(cache)
            # SCHEMA check, not just stamp: a cache from an older feature
            # era silently serves NaN for every newer feature.
            required = {"wr_with", "wr_against", "mp_with", "mp_against",
                        "wr_2p", "recent_pnl5", "acct_rank_20d"}
            if required.issubset(candidate.columns):
                _HISTORY, _HISTORY_STAMP = candidate, stamp
                return _HISTORY
        except Exception:
            pass
    from webapp import model_service as ms
    frame = ms.load_scores(ms.VIEW_QUANT)
    if frame is None or frame.empty:
        _HISTORY = pd.DataFrame()
        return _HISTORY
    frame = frame.copy()
    frame["day"] = pd.to_datetime(frame["day"])
    recent = frame.loc[frame["day"] >= frame["day"].max() - pd.Timedelta(days=45)]
    grouped = recent.groupby("account_key", observed=True).agg(
        trades=("score", "size"), mean_score=("score", "mean"),
        mean_pnl=("pnl", "mean"), pnl_std=("pnl", "std"),
        mean_abs_pnl=("pnl", lambda s: float(s.abs().mean())),
        win_rate=("pnl", lambda s: float((s > 0).mean())),
        mean_notional=("notional", "mean"),
        cum_pnl=("pnl", "sum"),
        recent_pnl5=("pnl", lambda s: float(s.tail(5).sum())),
        wr_20p=("pnl", lambda s: float((s.tail(max(3, int(len(s) * .20))) > 0).mean())),
        wr_10p=("pnl", lambda s: float((s.tail(max(3, int(len(s) * .10))) > 0).mean())),
        wr_5p=("pnl", lambda s: float((s.tail(max(3, int(len(s) * .05))) > 0).mean())),
        wr_2p=("pnl", lambda s: float((s.tail(max(3, int(len(s) * .02))) > 0).mean())),
        mp_20p=("pnl", lambda s: float(s.tail(max(3, int(len(s) * .20))).mean())),
        mp_10p=("pnl", lambda s: float(s.tail(max(3, int(len(s) * .10))).mean())),
        mp_5p=("pnl", lambda s: float(s.tail(max(3, int(len(s) * .05))).mean())),
        mp_2p=("pnl", lambda s: float(s.tail(max(3, int(len(s) * .02))).mean())),
    )

    # The account-day activity family, as of each account's last completed
    # day -- the live mirror of the training builder's close-day aggregates.
    daily = (frame.groupby(["account_key", "day"], observed=True)
             .agg(day_pnl=("pnl", "sum"), day_trades=("pnl", "size"),
                  day_wins=("pnl", lambda s: float((s > 0).sum())))
             .reset_index().sort_values(["account_key", "day"]))
    by_account = daily.groupby("account_key", observed=True)
    daily["acct_pnl_5d"] = by_account["day_pnl"].transform(
        lambda s: s.rolling(5, min_periods=1).sum())
    daily["acct_pnl_20d"] = by_account["day_pnl"].transform(
        lambda s: s.rolling(20, min_periods=1).sum())
    daily["acct_pnl_60d"] = by_account["day_pnl"].transform(
        lambda s: s.rolling(60, min_periods=1).sum())
    daily["acct_vol_20d"] = by_account["day_pnl"].transform(
        lambda s: s.rolling(20, min_periods=3).std())
    wins20 = by_account["day_wins"].transform(lambda s: s.rolling(20, min_periods=1).sum())
    trades20 = by_account["day_trades"].transform(lambda s: s.rolling(20, min_periods=1).sum())
    daily["acct_winrate_20d"] = wins20 / trades20.replace(0, np.nan)
    daily["acct_trades_5d"] = by_account["day_trades"].transform(
        lambda s: s.rolling(5, min_periods=1).sum())
    daily["acct_trades_20d"] = trades20
    cumulative = by_account["day_pnl"].cumsum()
    peak20 = cumulative.groupby(daily["account_key"], observed=True).transform(
        lambda s: s.rolling(20, min_periods=1).max())
    daily["acct_dd_20d"] = peak20 - cumulative
    daily["acct_best_20d"] = by_account["day_pnl"].transform(
        lambda s: s.rolling(20, min_periods=1).max())
    daily["acct_worst_20d"] = by_account["day_pnl"].transform(
        lambda s: s.rolling(20, min_periods=1).min())
    daily["acct_rank_20d"] = daily.groupby("day", observed=True)[
        "acct_pnl_20d"].rank(pct=True)
    first_day = by_account["day"].transform("min")
    daily["acct_tenure_days"] = (daily["day"] - first_day).dt.days
    # Regime-conditional record for the live scorer (when the scores frame
    # carries with_momentum -- older artifacts simply serve NaN).
    if "with_momentum" in recent.columns:
        recent = recent.assign(
            _with=pd.to_numeric(recent["with_momentum"], errors="coerce") > 0)
        conditional = recent.groupby(["account_key", "_with"], observed=True)[
            "pnl"].agg(wr=lambda s: float((s > 0).mean()), mp="mean").unstack()
        conditional.columns = [f"{stat}_{'with' if flag else 'against'}"
                               for stat, flag in conditional.columns]
        grouped = grouped.join(conditional, how="left")

    last = daily.groupby("account_key", observed=True).last()
    last = last.rename(columns={"day": "last_active_day"})
    grouped = grouped.join(
        last[["last_active_day", "acct_pnl_5d", "acct_pnl_20d", "acct_pnl_60d",
              "acct_vol_20d", "acct_winrate_20d", "acct_trades_5d",
              "acct_trades_20d", "acct_dd_20d", "acct_best_20d",
              "acct_worst_20d", "acct_rank_20d", "acct_tenure_days"]],
        how="left")

    _HISTORY, _HISTORY_STAMP = grouped, stamp
    try:
        grouped.to_parquet(ARTIFACTS / "quant_history_cache.parquet")
        _warm_update(stamp=stamp)
    except Exception:
        pass
    return grouped


#: History-family features that live had been deriving from the 45-day
#: account_history AGGREGATE -- a different quantity from training's expanding/
#: as-of build (proven: the model gates at 95% on training features yet loses
#: live). These are overwritten per trade with the parity build below.
_PARITY_HISTORY_KEYS = [
    "hist_win_rate", "hist_mean_pnl", "hist_pnl_std", "hist_mean_notional",
    "hist_recent_pnl5", "trade_index", "notional_vs_usual",
    "hist_wr_20p", "hist_wr_10p", "hist_wr_5p", "hist_wr_2p",
    "hist_mp_20p", "hist_mp_10p", "hist_mp_5p", "hist_mp_2p",
    "hist_wr_trend", "hist_wr_chop", "hist_mp_trend", "hist_mp_chop",
    "acct_pnl_5d", "acct_pnl_20d", "acct_pnl_60d", "acct_vol_20d",
    "acct_winrate_20d", "acct_trades_5d", "acct_trades_20d", "acct_dd_20d",
    "acct_best_20d", "acct_worst_20d", "acct_tenure_days", "acct_days_since_active",
    "equity_proxy", "notional_to_equity", "underwater_frac",
]

_ACCOUNT_TAPE: dict = {}          # account_key -> raw closed-trade tape (or None)
_TAPE_CON = None


def _tape_source() -> str | None:
    """The training feature cache holds every account's closed trades in the raw
    schema build_trade_features consumes -- the seed for parity live history."""
    from webapp import model_service as ms
    path = ms.SCRATCH / "quant_feature_cache.parquet"
    return str(path) if path.exists() else None


def _recent_closed_for(account: str, since: pd.Timestamp):
    """This account's closed trades AFTER the training cache's end, straight
    from production MySQL -- the append that carries the tape to NOW. MT4
    orders hold open+close in one row; MT5 pairs deals by position id."""
    try:
        from webapp import trade_feed as tfeed
        server, login = account.split(":", 1)
        login = int(login)
        connection = tfeed._connection(server)
        cents = tfeed.cent_logins(server)
        scale = 100.0 if login in cents else 1.0
        rows = []
        if server.startswith("mt4"):
            with connection.cursor() as cursor:
                # Newest first inside the limit: an active account has more
                # closes in the window than the limit, and ascending order
                # returned the OLDEST ones -- the account page then stopped
                # days short of now (mt5_live01:105036289, 15 Sep 2026).
                cursor.execute(
                    "SELECT * FROM (SELECT symbol_name, cmd, volume, open_price, open_ts, "
                    "close_price, close_ts, profit, `order` FROM orders "
                    "WHERE login = %s AND close_ts > %s AND cmd IN (0,1) "
                    "AND open_price > 0 ORDER BY close_ts DESC LIMIT 5000) recent ORDER BY close_ts",
                    [login, int(since.timestamp())])
                for (sym, cmd, vol, op, ots, cp, cts, profit, ticket) in cursor.fetchall():
                    rows.append({
                        "database": server, "account_key": account,
                        "symbol": str(sym), "cmd": "buy" if int(cmd) == 0 else "sell",
                        # MT4 volume is hundredths of a lot (BQ-verified /100)
                        "volume_lots": float(vol) / 100.0 / scale,
                        "open_time": pd.Timestamp(int(ots), unit="s"),
                        "close_time": pd.Timestamp(int(cts), unit="s"),
                        "open_price": float(op), "close_price": float(cp),
                        "sl": np.nan, "tp": np.nan,
                        "net_profit": float(profit) / scale,
                        "order": str(int(ticket)) if ticket is not None else None,
                        "state": "closed", "reason": None})
        else:
            # MT5: entry deals (entry=0) open a position, out deals (1/3) close
            # it; pair by position id when the schema carries one.
            with connection.cursor() as cursor:
                # MT5 `time` is DATETIME: an epoch int matches nothing.
                cursor.execute(
                    "SELECT * FROM (SELECT position_id, symbol, action, volume, price, `time`, "
                    "entry, profit, deal FROM deals WHERE login = %s AND `time` > %s "
                    "AND action IN (0,1) ORDER BY `time` DESC LIMIT 10000) recent ORDER BY `time`",
                    [login, since.to_pydatetime()])
                by_pos: dict = {}

                def _ts(value):
                    return (pd.Timestamp(int(value), unit="s")
                            if isinstance(value, (int, float))
                            else pd.Timestamp(value))

                for (pid, sym, action, vol, prc, when, entry, profit, deal) in cursor.fetchall():
                    rec = by_pos.setdefault(int(pid or 0), {})
                    if int(entry) == 0:
                        rec.update(symbol=str(sym),
                                   cmd="buy" if int(action) == 0 else "sell",
                                   volume_lots=float(vol) / 10000.0 / scale,
                                   open_time=_ts(when),
                                   open_price=float(prc))
                    else:
                        rec.setdefault("profit", 0.0)
                        rec["profit"] = rec.get("profit", 0.0) + float(profit or 0) / scale
                        rec["close_time"] = _ts(when)
                        rec["close_price"] = float(prc)
                        # The exit deal is the row identity the warehouse uses.
                        rec["order"] = str(int(deal)) if deal is not None else None
                for rec in by_pos.values():
                    if "open_time" in rec and "close_time" in rec:
                        rows.append({
                            "database": server, "account_key": account,
                            "symbol": rec["symbol"], "cmd": rec["cmd"],
                            "volume_lots": rec["volume_lots"],
                            "open_time": rec["open_time"],
                            "close_time": rec["close_time"],
                            "open_price": rec["open_price"],
                            "close_price": rec.get("close_price", np.nan),
                            "sl": np.nan, "tp": np.nan,
                            "net_profit": rec.get("profit", 0.0),
                            "order": rec.get("order"),
                            "state": "closed", "reason": None})
        return pd.DataFrame(rows) if rows else None
    except Exception:
        return None


def _account_tape(account: str):
    """The account's closed-trade tape (raw schema): the training cache via
    DuckDB plus this account's MySQL closes since the cache end, so the tape
    reaches NOW. Memoised; invalidated when the feed sees the account close."""
    if account in _ACCOUNT_TAPE:
        return _ACCOUNT_TAPE[account]
    global _TAPE_CON
    tape = None
    try:
        from webapp import trade_features as tfmod
        src = _tape_source()
        if src is not None:
            import duckdb
            if _TAPE_CON is None:
                _TAPE_CON = duckdb.connect()
            cols = ", ".join(tfmod.SCORING_RAW_COLUMNS)
            df = _TAPE_CON.execute(
                f"SELECT {cols} FROM read_parquet(?) WHERE account_key = ? "
                f"ORDER BY open_time", [src, account]).df()
            tape = df if len(df) else None
    except Exception:
        tape = None
    # APPEND to now (the cache ends days ago; without this, recency features
    # freeze at the cache boundary and trades_today reads zero forever).
    try:
        edge = (pd.to_datetime(tape["close_time"]).max() if tape is not None
                else pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=45))
        recent = _recent_closed_for(account, edge)
        if recent is not None and len(recent):
            tape = (pd.concat([tape, recent], ignore_index=True)
                    if tape is not None else recent)
            tape = tape.drop_duplicates(
                ["account_key", "open_time", "symbol", "volume_lots"],
                keep="first").sort_values("open_time").reset_index(drop=True)
    except Exception:
        pass
    if len(_ACCOUNT_TAPE) < 8000:
        _ACCOUNT_TAPE[account] = tape
    return tape


#: account -> (tape_len, dict of training-faithful feature values). One build
#: per account, reused for every trade until the account closes another trade;
#: the per-trade rebuild lagged the loop 5x (scored 460/2min -> 92/2min), which
#: delayed the stream by a minute and shuffled execution order.
_PARITY_STATE: dict = {}

#: Everything the shared builder computes that does NOT depend on the incoming
#: trade's symbol/size: full history family, account-day activity, AD corpus,
#: funding, plus the raw cum/peak needed for per-trade equity ratios.
_PARITY_STATE_KEYS = None      # resolved lazily from TRADE_FEATURES


_PARITY_QUEUE: "queue.Queue[str]" = None      # type: ignore[assignment]
_PARITY_PENDING: set = set()
_PARITY_WORKER: threading.Thread | None = None


def _parity_enqueue(account: str) -> None:
    """Build states OFF the scoring loop: the synchronous build (~0.3-0.5s
    per cold account) lagged the loop minutes behind the feed after restarts."""
    global _PARITY_QUEUE, _PARITY_WORKER
    import queue as _queue
    if _PARITY_QUEUE is None:
        _PARITY_QUEUE = _queue.Queue()
    with _LOCK:
        if account in _PARITY_PENDING:
            return
        _PARITY_PENDING.add(account)
    _PARITY_QUEUE.put(account)
    if _PARITY_WORKER is None or not _PARITY_WORKER.is_alive():
        def _drain():
            while True:
                acct = _PARITY_QUEUE.get()
                try:
                    _build_parity_state(acct, datetime.utcnow())
                except Exception:
                    pass
                finally:
                    with _LOCK:
                        _PARITY_PENDING.discard(acct)
        _PARITY_WORKER = threading.Thread(target=_drain, daemon=True,
                                          name="parity-builder")
        _PARITY_WORKER.start()


def _parity_state(account: str, naive_now) -> dict | None:
    """The account's as-of-now training-faithful feature state. NEVER blocks:
    a warm state is served instantly (even one close stale -- the background
    rebuild refreshes it); a cold account returns None (legacy values stand
    for that one trade) while the worker builds."""
    cached = _PARITY_STATE.get(account)
    if cached is not None:
        return cached[1]
    _parity_enqueue(account)
    return None


def _build_parity_state(account: str, naive_now) -> dict | None:
    """The synchronous build -- runs on the worker thread."""
    global _PARITY_STATE_KEYS
    tape = _account_tape(account)
    if tape is None or not len(tape):
        return None
    try:
        from webapp import trade_features as tfmod
        if _PARITY_STATE_KEYS is None:
            ctx_like = {"ctx_return_1h", "ctx_return_4h", "ctx_return_24h",
                        "ctx_vol_24h", "ctx_flow_prev", "zscore_24h",
                        "range_pos_24h", "vol_regime", "mom_1h_vol", "mom_4h_vol",
                        "mom_24h_vol", "with_momentum", "momentum_align",
                        "sl_dist_vol", "tp_dist_vol"}
            per_trade = {"hour", "weekday", "log_notional", "log_lots",
                         "direction", "has_sl", "has_tp", "sl_distance",
                         "tp_distance", "risk_reward", "is_expert", "symbol_code",
                         "notional_vs_usual", "notional_to_equity",
                         "underwater_frac", "trades_today", "hours_since_last"}
            _PARITY_STATE_KEYS = [f for f in tfmod.TRADE_FEATURES
                                  if f not in ctx_like and f not in per_trade]
        server = account.split(":")[0] if ":" in account else ""
        probe = pd.DataFrame([{
            "database": server, "account_key": account,
            "symbol": str(tape["symbol"].iloc[-1]), "cmd": "buy",
            "volume_lots": 0.01, "open_time": pd.Timestamp(naive_now),
            "close_time": pd.NaT, "open_price": float(tape["open_price"].iloc[-1]),
            "close_price": np.nan, "sl": np.nan, "tp": np.nan,
            "net_profit": np.nan, "state": "open", "reason": None}])
        built = tfmod.build_features_for_scoring(probe, tape)
        if not len(built):
            return None
        row = built.iloc[0]
        state = {}
        for name in _PARITY_STATE_KEYS:
            if name in built.columns:
                candidate = row[name]
                state[name] = float(candidate) if pd.notna(candidate) else np.nan
        # raw aggregates for the per-trade equity ratios
        for name in ("hist_cum_pnl", "hist_peak_pnl", "hist_mean_notional",
                     "equity_proxy"):
            if name in built.columns:
                candidate = row[name]
                state[name] = float(candidate) if pd.notna(candidate) else np.nan
        _PARITY_STATE[account] = (len(tape), state)
        if len(_PARITY_STATE) > 8000:
            _PARITY_STATE.pop(next(iter(_PARITY_STATE)))
        return state
    except Exception as error:
        _log(f"parity state skipped for {account}: "
             f"{type(error).__name__}: {error}")
        return None


def invalidate_parity(account: str) -> None:
    """A new close changes the account's history. Keep serving the last state
    (one close stale beats falling back to the legacy aggregates) and refresh
    it in the background off the loop."""
    _ACCOUNT_TAPE.pop(account, None)
    if account in _PARITY_STATE:
        _parity_enqueue(account)


def score(symbol: str, direction: int, lots: float, price: float,
          account: str, history: pd.DataFrame,
          update_state: bool = False, with_magnitude: bool = False,
          capture: bool = False):
    """This trade's probability the client wins it. None without a model.

    Fills every feature the live moment can know: trade intrinsics, the bar
    context maintained from the same tape training used, the stable symbol
    code, the client's history aggregates, funding state, activity shape and
    the equity-impact family. What genuinely cannot be known at fill time
    (client SL/TP, underwater peak) stays NaN and rides LightGBM's default
    paths -- the same treatment training gave missing values.
    """
    model = booster()
    if model is None:
        return None
    try:
        now = datetime.now(timezone.utc)
        naive_now = datetime.utcnow()
        notional = lots * 100.0 * max(price, 0.0)
        values = {"hour": now.hour, "weekday": now.weekday(),
                  "log_notional": np.log1p(max(notional, 0.0)),
                  "log_lots": np.log1p(max(lots, 0.0)),
                  "direction": float(direction)}
        values.update(_ctx_features(symbol, direction))
        from webapp.trade_feed import _canonical
        code = _symbol_codes().get(symbol)
        if code is None:
            code = _symbol_codes().get(_canonical(symbol))
        if code is not None:
            values["symbol_code"] = code
        values.update(_activity_features(account, naive_now, update_state))
        account_day = _account_day_snapshot()
        if len(account_day) and account in account_day.index:
            row_ad = account_day.loc[account]
            for name, candidate in row_ad.items():
                if candidate is not None and not pd.isna(candidate):
                    values[name] = float(candidate)
        funding = _funding_snapshot()
        if len(funding) and account in funding.index:
            row_funding = funding.loc[account]
            for name in _FUNDING_COLUMNS:
                candidate = row_funding.get(name)
                if candidate is not None:
                    values[name] = float(candidate)
        if account in history.index:
            row = history.loc[account]
            usual = float(row.get("mean_notional") or np.nan)
            values.update({
                "hist_win_rate": float(row.get("win_rate", np.nan)),
                "hist_mean_pnl": float(row.get("mean_pnl", np.nan)),
                "hist_pnl_std": float(row.get("pnl_std", np.nan)),
                "hist_mean_notional": usual,
                "trade_index": float(row.get("trades", np.nan)),
                "notional_vs_usual": notional / usual if usual and usual > 0 else np.nan,
                "hist_recent_pnl5": float(row.get("recent_pnl5", np.nan)),
                "hist_wr_20p": float(row.get("wr_20p", np.nan)),
                "hist_wr_10p": float(row.get("wr_10p", np.nan)),
                "hist_wr_5p": float(row.get("wr_5p", np.nan)),
                "hist_wr_2p": float(row.get("wr_2p", np.nan)),
                "hist_mp_20p": float(row.get("mp_20p", np.nan)),
                "hist_mp_10p": float(row.get("mp_10p", np.nan)),
                "hist_mp_5p": float(row.get("mp_5p", np.nan)),
                "hist_mp_2p": float(row.get("mp_2p", np.nan)),
                "hist_wr_trend": float(row.get("wr_with", np.nan)),
                "hist_wr_chop": float(row.get("wr_against", np.nan)),
                "hist_mp_trend": float(row.get("mp_with", np.nan)),
                "hist_mp_chop": float(row.get("mp_against", np.nan)),
            })
            cum_pnl = float(row.get("cum_pnl", 0.0) or 0.0)
            net_funding = values.get("net_funding_to_date")
            live = _live_stats(account)
            if live and live["n"] >= 3:
                base_n = float(values.get("trade_index") or 0.0)
                total = base_n + live["n"]
                prior_wr = values.get("hist_win_rate")
                prior_mp = values.get("hist_mean_pnl")
                values["hist_win_rate"] = (
                    ((prior_wr if prior_wr is not None
                      and np.isfinite(prior_wr) else 0.5) * base_n
                     + live["wins"]) / total)
                values["hist_mean_pnl"] = (
                    ((prior_mp if prior_mp is not None
                      and np.isfinite(prior_mp) else 0.0) * base_n
                     + live["sum"]) / total)
                values["hist_recent_pnl5"] = live["last5"]
                values["trade_index"] = total
                # The short rungs of the recency ladder are exactly what the
                # live window measures -- recompute them from it.
                window = _LIVE_CLIENT.get(account)
                if window:
                    tail = list(window)
                    for name, fraction in (("hist_wr_2p", 0.02),
                                           ("hist_mp_2p", 0.02),
                                           ("hist_wr_5p", 0.05),
                                           ("hist_mp_5p", 0.05)):
                        width = max(3, int(total * fraction))
                        recent = tail[-min(width, len(tail)):]
                        if len(recent) >= 3:
                            values[name] = (
                                float(np.mean([v > 0 for v in recent]))
                                if name.startswith("hist_wr")
                                else float(np.mean(recent)))
            equity = ((net_funding if net_funding is not None
                       and np.isfinite(net_funding) else 0.0) + cum_pnl)
            values["equity_proxy"] = equity
            if equity > 0:
                values["notional_to_equity"] = min(notional / equity, 1000.0)
            for name in ("acct_pnl_5d", "acct_pnl_20d", "acct_pnl_60d",
                         "acct_vol_20d", "acct_winrate_20d", "acct_trades_5d",
                         "acct_trades_20d", "acct_dd_20d", "acct_best_20d",
                         "acct_worst_20d", "acct_rank_20d", "acct_tenure_days"):
                candidate = row.get(name)
                if candidate is not None:
                    values[name] = float(candidate)
            last_active = row.get("last_active_day")
            if last_active is not None and not pd.isna(last_active):
                values["acct_days_since_active"] = float(min(
                    (naive_now - pd.Timestamp(last_active).to_pydatetime()).days, 90))
        # PARITY: overwrite every tape-derived feature with training-identical
        # values from the SAME build function (cached per account -- one build
        # per account, not per trade, so the loop keeps pace with the feed).
        state = _parity_state(account, naive_now)
        if state is not None:
            values.update(state)
            usual = state.get("hist_mean_notional")
            if usual and np.isfinite(usual) and usual > 0:
                values["notional_vs_usual"] = notional / usual
            equity = state.get("equity_proxy")
            if equity is not None and np.isfinite(equity) and equity > 0:
                values["notional_to_equity"] = min(notional / equity, 1000.0)
                peak = state.get("hist_peak_pnl")
                cum = state.get("hist_cum_pnl")
                if (peak is not None and cum is not None
                        and np.isfinite(peak) and np.isfinite(cum)):
                    values["underwater_frac"] = float(
                        np.clip((peak - cum) / equity, 0, 100))
        vector = np.array([[values.get(name, np.nan) for name in _FEATURE_NAMES]],
                          dtype="float64")
        probability = _predict_calibrated(symbol, vector, model)
        if capture:
            return probability, {name: values.get(name, np.nan)
                                 for name in _FEATURE_NAMES}
        if not with_magnitude:
            return probability
        magnitude_model = _magnitude_model()
        multiple = (max(0.0, float(magnitude_model.predict(vector)[0]))
                    if magnitude_model is not None else None)
        # Per-class routing for the regressors (pooled fallback).
        perlot = _predict_regressor("perlot", symbol, vector, _perlot_model())
        exit_fe = _predict_regressor("exit_fe", symbol, vector, _exit_fe_model())
        # Tail pre-profit drawdown (MAE q80) -> the sizer's stress term and the
        # veto, read moments later in execute() for this symbol.
        _note_mae_tail(symbol, _predict_regressor("mae_q80", symbol, vector,
                                                  _mae_tail_model()))
        s2_score = s2_magnitude = None
        s2_model = _strategy2_model()
        if s2_model is not None and _S2_FEATURES:
            from webapp.trade_features import symbol_class
            chosen = (_S2_CLASS_MODELS or {}).get(symbol_class(symbol),
                                                  s2_model)
            s2_vector = np.array(
                [[values.get(name, np.nan) for name in _S2_FEATURES]],
                dtype="float64")
            s2_score = float(chosen.predict(s2_vector)[0])
            if _S2_MAGNITUDE is not None:
                s2_magnitude = max(0.0, float(
                    _S2_MAGNITUDE.predict(s2_vector)[0]))
        # Entry-improvement prediction off the SAME vector -- exact
        # live/train parity with the stance model's features.
        # entry_model.txt IS the MAE median (typical pre-profit pullback), so
        # the per-class quant_mae_q50_<class> boosters route it when present.
        entry_wait = _predict_regressor("mae_q50", symbol, vector, _entry_model())
        return (probability, multiple, s2_score, s2_magnitude, entry_wait,
                perlot, exit_fe)
    except Exception:
        return (None,) * 7 if with_magnitude else None


#: The artifact's own decile cutoffs -- the STANDARD the backtest selected at.
#: Set once at engine start; the rolling window may only tighten them, never
#: loosen them.
_SCORE_ANCHORS: dict = {}


def calibrate_anchors(config: VantageConfig) -> dict:
    """Copy floor / invert ceiling from the artifact's score distribution.

    SYMMETRIC EDGE: the invert ceiling is the tighter of the artifact's bottom
    decile and (1 - copy floor). The bottom decile alone sits near 0.40 -- a
    20-point edge against copies' 70 -- and live forensics showed exactly what
    that predicts: copies +$10.26 while inverts bled -$16.01 through the same
    spread. An invert must be as convicted as a copy to pay the same toll.
    """
    # Operator overrides win outright when set.
    if config.copy_anchor_override > 0 and config.invert_anchor_override > 0:
        _SCORE_ANCHORS["copy_floor"] = float(config.copy_anchor_override)
        _SCORE_ANCHORS["invert_ceiling"] = float(config.invert_anchor_override)
        return dict(_SCORE_ANCHORS)
    from webapp import model_service as ms
    frame = ms.load_scores(ms.VIEW_QUANT)
    if frame is None or frame.empty:
        return {}
    values = pd.to_numeric(frame["score"], errors="coerce").dropna()
    copy_floor = float(values.quantile(config.copy_quantile))
    _SCORE_ANCHORS["copy_floor"] = copy_floor
    _SCORE_ANCHORS["invert_ceiling"] = min(
        float(values.quantile(config.invert_quantile)), 1.0 - copy_floor)
    return dict(_SCORE_ANCHORS)


def route(value: float | None, config: VantageConfig) -> str:
    """Copy, invert, or stand aside.

    The decision is the backtest's, verbatim: a trade is copied only when it
    clears the ARTIFACT's top-decile cutoff (~0.85), inverted only under the
    bottom-decile one. The rolling window can tighten the bar when live flow
    runs hot, but can never lower the standard below what was tested --
    letting it do so is how 0.62-score gold got copied overnight.
    """
    if value is None:
        return "ignore"
    _SCORES.append(value)
    if len(_SCORES) > 1000:
        del _SCORES[:len(_SCORES) - 1000]
    copy_floor = _SCORE_ANCHORS.get("copy_floor", 0.80)
    invert_ceiling = _SCORE_ANCHORS.get("invert_ceiling", 0.30)
    # When the operator PINS the anchors (both overrides set), respect them
    # EXACTLY -- that is the whole point of pinning the values you backtested.
    # Otherwise the rolling-quantile guard tightens the bar with live flow (it
    # can only ever make the standard STRICTER, never looser -- which is how the
    # invert ceiling had crept far below its tested 0.25 and starved the leg).
    pinned = (getattr(config, "copy_anchor_override", 0) > 0
              and getattr(config, "invert_anchor_override", 0) > 0)
    if not pinned and len(_SCORES) >= 50:
        copy_floor = max(copy_floor,
                         float(np.quantile(_SCORES, config.copy_quantile)))
        invert_ceiling = min(invert_ceiling,
                             float(np.quantile(_SCORES, config.invert_quantile)))
    if value >= copy_floor:
        return "copy"
    if getattr(config, "enable_invert", True) and value <= invert_ceiling:
        return "invert"
    return "ignore"


def _wall_lev(config: VantageConfig) -> float:
    """Effective gross-exposure leverage: the broker's leverage capped by our
    own risk ceiling. The wall = _wall_lev(config) x equity, so exposure is
    bounded to a fixed multiple of equity no matter how much leverage (e.g. a
    500x demo) the broker grants -- and it scales with any account size."""
    cap = getattr(config, "max_gross_leverage", 0.0) or 0.0
    return min(config.leverage, cap) if cap > 0 else config.leverage


def expected_dollars(value: float, stance: str, account: str,
                     history: pd.DataFrame, fallback: float) -> float:
    """Our edge on this trade at 1x client size: P(win) x typical swing."""
    scale = fallback
    if account in history.index:
        candidate = float(history.loc[account].get("mean_abs_pnl", np.nan))
        if np.isfinite(candidate) and candidate > 0:
            scale = candidate
    live = _live_stats(account)
    if live and live["n"] >= 5:
        scale = 0.5 * scale + 0.5 * (live["abs"] / live["n"])
    edge = (2.0 * value - 1.0) if stance == "copy" else (1.0 - 2.0 * value)
    return max(0.0, edge) * scale


# ---------------------------------------------------------------- sizing
_STRATEGY_STATS: dict | None = None
_STRATEGY_STAMP = None

#: One heavy compute at a time. The Vantage tab polls report() continuously;
#: before this lock each poll RECOMPUTED the uncached statistics over the full
#: scores frame, the recomputes piled onto each other, working set hit 10+ GB
#: and the engine thread starved through its own startup.
_HEAVY_LOCK = threading.Lock()

#: The warm-up products (account history, strategy stats, symbol multipliers,
#: funding snapshot) are DERIVED deterministically from the artifact -- so
#: compute each ONCE per artifact and persist it. Every later boot loads in
#: seconds instead of re-scanning 17.8M rows for minutes.
_WARM_META = ARTIFACTS / "quant_warm.json"


def _warm_meta() -> dict:
    try:
        import json
        return json.loads(_WARM_META.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _warm_update(**entries) -> None:
    try:
        import json
        meta = _warm_meta()
        # A NEW artifact stamp invalidates every section cached under the old
        # one -- without this, stats computed pre-retrain would masquerade as
        # belonging to the new model.
        if "stamp" in entries and entries["stamp"] != meta.get("stamp"):
            meta.pop("stats", None)
            meta.pop("multipliers", None)
        meta.update(entries)
        _WARM_META.write_text(json.dumps(meta), encoding="utf-8")
    except Exception:
        pass


def strategy_stats(config: VantageConfig) -> dict:
    """This policy's backtest shape at 1x client size: drawdown AND daily P&L.

    The drawdown is the denominator of the 35% rule; the daily mean, scaled by
    k, is what the account should EXPECT to earn per day -- the number that
    tells you within a week whether live behaviour matches the backtest.
    Cached per artifact: the scan costs seconds and the answer only changes on
    retrain.
    """
    global _STRATEGY_STATS, _STRATEGY_STAMP
    from webapp import model_service as ms
    meta = ms.artifact_meta(ms.VIEW_QUANT) or {}
    stamp = meta.get("trained_at")
    if _STRATEGY_STATS is not None and _STRATEGY_STAMP == stamp:
        return _STRATEGY_STATS

    empty = {"drawdown": 0.0, "daily_mean": 0.0, "daily_std": 0.0, "days": 0}
    with _HEAVY_LOCK:
        if _STRATEGY_STATS is not None and _STRATEGY_STAMP == stamp:
            return _STRATEGY_STATS
        return _strategy_stats_locked(config, stamp, empty)


def _strategy_stats_locked(config: VantageConfig, stamp, empty: dict) -> dict:
    global _STRATEGY_STATS, _STRATEGY_STAMP
    warm = _warm_meta()
    if warm.get("stamp") == stamp and isinstance(warm.get("stats"), dict):
        _STRATEGY_STATS, _STRATEGY_STAMP = dict(warm["stats"]), stamp
        return _STRATEGY_STATS
    from webapp import model_service as ms
    frame = ms.load_scores(ms.VIEW_QUANT)
    if frame is None or frame.empty:
        return empty
    values = pd.to_numeric(frame["score"], errors="coerce")
    pnl = pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0)
    # Reproduce the routing per day, exactly as the engine decides.
    day = pd.to_datetime(frame["day"])
    sign = np.zeros(len(frame))
    for _, index in pd.Series(range(len(frame))).groupby(day.to_numpy()):
        block = values.to_numpy()[index]
        if len(block) < 20:
            continue
        high = np.quantile(block, config.copy_quantile)
        low = np.quantile(block, config.invert_quantile)
        sign[index] = np.where(block >= high, 1.0, np.where(block <= low, -1.0, 0.0))
    ours = pd.Series(sign * pnl.to_numpy(), index=day)
    daily = ours.groupby(ours.index.normalize()).sum().sort_index()
    curve = daily.cumsum()
    _STRATEGY_STATS = {
        "drawdown": float(abs((curve - curve.cummax()).min())),
        "daily_mean": float(daily.mean()),
        "daily_std": float(daily.std()),
        "days": int(len(daily)),
    }
    _STRATEGY_STAMP = stamp
    _warm_update(stamp=stamp, stats=_STRATEGY_STATS)
    return _STRATEGY_STATS


def strategy_drawdown(config: VantageConfig) -> float:
    return strategy_stats(config)["drawdown"]


_SYMBOL_MULT: dict | None = None
_SYMBOL_MULT_STAMP = None


def symbol_multipliers(config: VantageConfig) -> dict[str, float]:
    """Backtest edge-density multipliers with live config overrides applied.

    The base scaling comes from the backtest, which measures the CLIENT tape
    edge and ignores our execution cost -- so a high-cost symbol like gold can
    look strong in backtest yet bleed live once spread+commission are paid.
    `symbol_overrides` in the config is the operator's live-evidence correction:
    a canonical-symbol -> multiplier map (0.0 excludes) layered on top.
    """
    base = _base_symbol_multipliers(config)
    overrides = getattr(config, "symbol_overrides", None) or {}
    if not overrides:
        return base
    from webapp.trade_feed import _canonical
    out = dict(base)
    for symbol, mult in overrides.items():
        try:
            out[_canonical(str(symbol))] = float(mult)
        except (TypeError, ValueError):
            continue
    return out


def _base_symbol_multipliers(config: VantageConfig) -> dict[str, float]:
    """Per-instrument size scaling from the backtest's own edge density.

    Dollars earned per lot already embeds each instrument's volatility and
    contract size, so scaling by it (relative to XAUUSD, the flow's backbone)
    sizes toward where the edge is dense -- measured per-symbol: XAGUSD $643/lot,
    NAS100 $261, XAUUSD $179, FX majors ~$25. Clipped to [0.5, 2.0] so no
    instrument dominates, neutral (1.0) where evidence is thin, and ZERO --
    trades skipped outright -- where the policy measurably loses: HK50
    (-$163k / 5,306 trades), NZDJPY (-$83k), NAS100U6 (-$23k).
    """
    global _SYMBOL_MULT, _SYMBOL_MULT_STAMP
    from webapp import model_service as ms
    meta = ms.artifact_meta(ms.VIEW_QUANT) or {}
    stamp = meta.get("trained_at")
    if _SYMBOL_MULT is not None and _SYMBOL_MULT_STAMP == stamp:
        return _SYMBOL_MULT

    with _HEAVY_LOCK:
        if _SYMBOL_MULT is not None and _SYMBOL_MULT_STAMP == stamp:
            return _SYMBOL_MULT
        return _symbol_multipliers_locked(config, stamp)


def _symbol_multipliers_locked(config: VantageConfig, stamp) -> dict[str, float]:
    global _SYMBOL_MULT, _SYMBOL_MULT_STAMP
    warm = _warm_meta()
    if warm.get("stamp") == stamp and isinstance(warm.get("multipliers"), dict):
        _SYMBOL_MULT, _SYMBOL_MULT_STAMP = dict(warm["multipliers"]), stamp
        return _SYMBOL_MULT
    from webapp import model_service as ms
    frame = ms.load_scores(ms.VIEW_QUANT)
    if frame is None or frame.empty:
        _SYMBOL_MULT = {}
        return _SYMBOL_MULT
    value = pd.to_numeric(frame["score"], errors="coerce")
    pnl = pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0)
    day = pd.to_datetime(frame["day"])
    sign = np.zeros(len(frame))
    for _, index in pd.Series(range(len(frame))).groupby(day.to_numpy()):
        block = value.to_numpy()[index]
        if len(block) < 20:
            continue
        high = np.quantile(block, config.copy_quantile)
        low = np.quantile(block, config.invert_quantile)
        sign[index] = np.where(block >= high, 1.0, np.where(block <= low, -1.0, 0.0))
    acted = frame.loc[sign != 0]
    ours = pd.Series(sign[sign != 0] * pnl[sign != 0].to_numpy(),
                     index=acted.index)
    from webapp.trade_feed import _canonical
    canonical = acted["symbol"].map(_canonical)
    lots = pd.to_numeric(acted["volume_lots"], errors="coerce").fillna(0.0)
    table = pd.DataFrame({"canonical": canonical, "ours": ours, "lots": lots}) \
        .groupby("canonical", observed=True).agg(
            trades=("ours", "size"), pnl=("ours", "sum"), lots=("lots", "sum"))

    baseline = float(table.loc["XAUUSD", "pnl"] / table.loc["XAUUSD", "lots"]) \
        if "XAUUSD" in table.index else 100.0
    multipliers: dict[str, float] = {}
    for name, row in table.iterrows():
        # Confident losers are excluded outright, not merely down-sized.
        if row["pnl"] < 0 and row["trades"] >= 300:
            multipliers[str(name)] = 0.0
        elif row["trades"] < 1000 or row["lots"] <= 0:
            multipliers[str(name)] = 1.0        # thin evidence: neutral
        else:
            density = float(row["pnl"] / row["lots"])
            multipliers[str(name)] = float(np.clip(density / baseline, 0.5, 2.0))
    _SYMBOL_MULT, _SYMBOL_MULT_STAMP = multipliers, stamp
    _warm_update(stamp=stamp, multipliers=multipliers)
    return multipliers


def scale_factor(config: VantageConfig, capital: float, drawdown: float) -> float:
    """k such that k x historical drawdown = risk_budget x capital.

    Capital is EQUITY, not balance: equity is the quantity the 35% drawdown
    contract defends, and compounding on it lets size grow with open winners
    instead of waiting for them to be banked."""
    if drawdown <= 0 or capital <= 0:
        return 0.0
    return (config.risk_budget_fraction * capital) / drawdown


def _lot_economics(symbol: str, price: float) -> tuple[float, float]:
    """(USD per price-unit per lot, USD notional per lot).

    From the venue's OWN contract spec when connected -- tick_value/tick_size
    is the exact dollars-per-price-unit the broker settles at. The old 6-alpha
    heuristic classified XAUUSD as forex, inflating its computed spread cost
    ~1000x (every gold entry failed the cost hurdle) and understating its
    notional ~4x against the leverage wall.
    """
    try:
        info = _mt5().symbol_info(symbol)
        if info and info.trade_tick_size and info.trade_tick_value:
            per_unit = float(info.trade_tick_value) / float(info.trade_tick_size)
            contract = float(getattr(info, "trade_contract_size", 0) or 100.0)
            root = "".join(c for c in symbol.upper() if c.isalnum())
            # Real FX only. BTCUSD/ETHUSD are also 6 alpha chars, so the old
            # length test wrongly tagged crypto as FX -> notional came out as the
            # contract (~1) instead of contract x price (~$80k), hiding crypto
            # exposure from the leverage wall and letting the sizer oversize it.
            from webapp.trade_features import symbol_class
            is_fx = (len(root) == 6 and root.isalpha()
                     and not root.startswith(("XAU", "XAG"))
                     and symbol_class(symbol) not in ("crypto", "index/other"))
            notional = contract if is_fx else contract * max(price, 0.0)
            return per_unit, notional
    except Exception:
        pass
    root = "".join(c for c in str(symbol).upper() if c.isalnum())
    if root.startswith(("XAU", "XAG")):
        return 100.0, 100.0 * max(price, 0.0)      # metals: 100 oz contract
    if len(root) == 6 and root.isalpha():
        return 100_000.0, 100_000.0                # FX: standard lot
    return 100.0, 100.0 * max(price, 0.0)          # CFD-style


def _notional(symbol: str, lots: float, price: float) -> float:
    """USD notional for the leverage wall."""
    return lots * _lot_economics(symbol, price)[1]


def _commission_cost(symbol: str, lots: float, config) -> float:
    """Round-turn commission for this trade. Charged on gold, FX and metals;
    index and crypto CFDs are spread-only here, so zero."""
    from webapp.trade_features import symbol_class
    rate = float(getattr(config, "commission_per_lot", 6.0) or 0.0)
    if symbol_class(symbol) in ("index/other", "crypto"):
        return 0.0
    return rate * float(lots)


_SOURCE_SPECS: dict[str, dict] = {}


def _contract_adjust(server: str, source_symbol: str, venue_symbol: str) -> float:
    """(source contract size / venue contract size), clipped sane.

    Contract size belongs to the RAW ticker, not the canonical instrument: a
    source server's 10x or 100x contract variant would otherwise become a
    10x or 100x position on our side. 1.0 whenever either side is unknown.
    """
    table = _SOURCE_SPECS.get(server)
    if table is None:
        table = {}
        try:
            from webapp.symbol_specs import SPEC_CACHE
            frame = pd.read_parquet(SPEC_CACHE / f"{server}.parquet")
            symbol_col = next((c for c in frame.columns
                               if c.lower() in ("symbol", "symbol_name", "name")), None)
            size_col = next((c for c in frame.columns
                             if "contract" in c.lower()), None)
            if symbol_col and size_col:
                for _, row in frame.iterrows():
                    size = pd.to_numeric(row[size_col], errors="coerce")
                    if size and size > 0:
                        table[str(row[symbol_col]).upper()] = float(size)
        except Exception:
            pass
        _SOURCE_SPECS[server] = table
    source = table.get(str(source_symbol).upper())
    try:
        info = _mt5().symbol_info(venue_symbol)
        venue = float(getattr(info, "trade_contract_size", 0) or 0)
    except Exception:
        venue = 0.0
    if not source or not venue:
        return 1.0
    return float(np.clip(source / venue, 0.001, 1000.0))


def _quantize_lots(symbol: str, lots: float) -> float:
    """Align volume to THIS symbol's min/step/max -- index and stock CFDs
    use 0.1+ steps and rejected every 0.01-grained order with 10014."""
    try:
        info = _mt5().symbol_info(symbol)
        if info and info.volume_step:
            step = float(info.volume_step)
            quantized = round(round(lots / step) * step, 8)
            low = float(info.volume_min or step)
            high = float(info.volume_max or max(quantized, low))
            return max(low, min(high, quantized))
    except Exception:
        pass
    return round(max(0.01, lots), 2)


def _unrealized(position: dict) -> float:
    """Mark-to-market P&L; live positions carry it, paper ones are marked."""
    profit = position.get("profit")
    if profit is not None:
        return float(profit)
    quote = current_quote(position["symbol"])
    if quote is None:
        return 0.0
    mark = quote[0] if position["direction"] > 0 else quote[1]
    move = (mark - position["open_price"]) * position["direction"]
    per_unit, _ = _lot_economics(position["symbol"], position["open_price"])
    return move * per_unit * float(position["lots"])


def _position_risk(position: dict) -> float:
    """One-day one-sigma dollar move of this position -- the unit the risk
    budget is spent in. Volatility from the live bars; a flat 1% of notional
    when a symbol's bars have not warmed yet."""
    price = float(position.get("open_price") or 0.0)
    lots = float(position.get("lots") or 0.0)
    per_unit, notional_per_lot = _lot_economics(position["symbol"], price)
    context = _ctx_features(position["symbol"], 1)
    vol = context.get("ctx_vol_24h")
    if vol and np.isfinite(vol) and vol > 0 and price > 0:
        return lots * per_unit * (vol * np.sqrt(24.0)) * price
    return lots * notional_per_lot * 0.01


def _fit_margin(venue_symbol: str, direction: int, lots: float, price: float,
                share: float = 0.85) -> float:
    """Shrink `lots` so the VENUE's margin for this order fits inside `share`
    of free margin. The leverage wall works off the account's headline
    leverage, but venues margin instruments individually -- IG margins gold
    at ~4% on a 1:200 account (10019 'No money', 14 Sep 2026) -- and resting
    limits reserve margin too. Returns 0.0 when even the minimum lot does not
    fit; the caller blocks instead of sending an order the venue will refuse."""
    try:
        mt5 = _mt5()
        info = _mt5_cached("account_info", 1.0, mt5.account_info)
        free = float(getattr(info, "margin_free", 0.0) or 0.0) if info else 0.0
        if free <= 0 or lots <= 0 or not price or price <= 0:
            return lots
        otype = mt5.ORDER_TYPE_BUY if direction > 0 else mt5.ORDER_TYPE_SELL
        need = mt5.order_calc_margin(otype, venue_symbol, float(lots), float(price))
        if need is None or need <= 0:
            return lots
        if need <= free * share:
            return lots
        per_lot = need / max(lots, 1e-9)
        fit = _quantize_lots(venue_symbol, (free * share) / per_lot)
        sym = mt5.symbol_info(venue_symbol)
        vmin = float(getattr(sym, "volume_min", 0.01) or 0.01) if sym else 0.01
        if fit < vmin:
            return 0.0
        _log(f"margin fit: {venue_symbol} {lots} -> {fit} lots (venue needs ${need:,.0f}, "
             f"free ${free:,.0f})")
        return fit
    except Exception:
        return lots


def _margin_per_lot(venue_symbol: str, direction: int, price: float) -> float:
    """The venue's margin for ONE lot of this symbol, in account currency
    (cached a minute per symbol and side: it moves with price, slowly)."""
    def fetch():
        try:
            mt5 = _mt5()
            otype = mt5.ORDER_TYPE_BUY if direction > 0 else mt5.ORDER_TYPE_SELL
            need = mt5.order_calc_margin(otype, venue_symbol, 1.0, float(price))
            return float(need) if need and need > 0 else 0.0
        except Exception:
            return 0.0
    return _mt5_cached(f"mpl:{venue_symbol}:{1 if direction > 0 else -1}", 60.0, fetch)


def _strategy_margin_used(prefixes: tuple) -> float:
    """Margin the venue holds for OUR positions and resting orders whose
    comment starts with one of `prefixes` ('1' = strategy 1, '2t' = tape):
    per-lot margin (cached) x volume, one shared answer per two seconds."""
    def fetch():
        total = 0.0
        try:
            mt5 = _mt5()
            positions = _mt5_cached("positions", 0.5, lambda: mt5.positions_get() or ())
            orders = _mt5_cached("orders", 0.5, lambda: mt5.orders_get() or ())
            for p in positions:
                if str(p.comment or "").startswith(prefixes):
                    total += _margin_per_lot(p.symbol, 1 if p.type == 0 else -1, float(p.price_open)) * float(p.volume)
            for o in orders:
                if str(o.comment or "").startswith(prefixes):
                    total += _margin_per_lot(o.symbol, 1 if o.type in (0, 2, 4, 6) else -1,
                                             float(o.price_open)) * float(o.volume_current)
        except Exception:
            pass
        return total
    return _mt5_cached(f"smu:{'|'.join(prefixes)}", 2.0, fetch)


def _envelope_cap(config, venue_symbol: str, direction: int, price: float, equity: float,
                  prefixes: tuple, share: float, slots: int) -> float | None:
    """Largest lot size that keeps this strategy inside its margin envelope:
    per-slot share of the envelope AND the room the envelope has left.
    None when the venue gives no margin figure."""
    per_lot = _margin_per_lot(venue_symbol, direction, price)
    if per_lot <= 0 or equity <= 0:
        return None
    envelope = equity * max(float(share), 0.0)
    per_slot = envelope / (per_lot * max(int(slots), 1))
    room = max(0.0, envelope - _strategy_margin_used(prefixes)) / per_lot
    return min(per_slot, room)


def _client_scaled_lots(config, account: str, source_symbol: str, venue_symbol: str,
                        client_lots: float, equity: float) -> float | None:
    """sizing_policy="client": copy the client's size, scaled to our equity.
    A 0.01-lot client trade is 0.01 lots on 5k equity, 0.20 lots on 100k
    (client_lots_per_5k = 1.0), after the source/venue contract adjustment.
    None when the policy is off or the client's lots are unknown."""
    if str(getattr(config, "sizing_policy", "client")).lower() != "client":
        return None
    if client_lots is None or client_lots <= 0 or equity <= 0:
        return None
    scale = float(getattr(config, "client_lots_per_5k", 1.0) or 0.0)
    if scale <= 0:
        return None
    adjust = (_contract_adjust(account.rpartition(":")[0], source_symbol, venue_symbol)
              if config.mode == "live" else 1.0)
    return float(client_lots) * adjust * (equity / 5000.0) * scale


def _size_position(config, venue_symbol: str, entry: float, direction: int,
                   book: list, equity: float, leverage: float,
                   per_slot: bool = True) -> float:
    """THE position sizer -- one rule, no layers.

    `per_slot=False` (client-proportional sizing) drops the per-slot split of
    the drawdown budget and of the margin envelope: the client dictates the
    size, and only the WHOLE budget, the wall, the envelope's room and
    max_lots_per_trade cap it.

    Place the LARGEST position whose worst-case loss keeps the whole book's
    drawdown within risk_budget_fraction x equity, then never above the
    account's own leverage wall. Worst-case loss of a position is
    dd_stress_move x its notional (a holding-period adverse move, not a daily
    sigma -- these are minute-scale copy holds). Risk is measured on NET signed
    exposure per symbol, so a trade that HEDGES an existing opposite position
    (copy vs invert on one symbol) nets exposure down and is allowed LARGER.

    Every term is a function of live equity and the broker's real leverage, so
    the same rule scales from the minimum deposit upward with no per-account
    tuning: as equity grows the budget grows, positions grow, the 35% drawdown
    ceiling holds, and returns compound."""
    if equity <= 0 or leverage <= 0:
        return 0.0
    per_lot_notional = _lot_economics(venue_symbol, entry)[1]
    if per_lot_notional <= 0:
        return 0.0
    stress = max(float(getattr(config, "dd_stress_move", 0.01)), 1e-4)
    # THIS trade's stress: the predicted tail pre-profit drawdown (MAE q80)
    # when it exceeds the operator's floor -- inverse-MAE sizing. The rest of
    # the book keeps the baseline (their own predictions sized them).
    stress_trade, _ = _predicted_stress(config, venue_symbol)
    slots = max(int(getattr(config, "expected_peak_positions", 20)), 1) if per_slot else 1
    dd_budget = config.risk_budget_fraction * equity            # 35% x equity ($ max DD)

    # (1) RESERVATION: each trade takes at most 1/slots of the budget, so every
    #     signal up to the expected peak concurrency executes (not just the first).
    per_trade_lots = (dd_budget / (stress_trade * slots)) / per_lot_notional

    # net signed notional per symbol currently on the book
    net: dict[str, float] = {}
    used_gross = 0.0
    for p in book:
        s = p["symbol"]
        pl = _lot_economics(s, float(p.get("open_price") or entry))[1]
        signed = float(p.get("direction") or 0) * float(p.get("lots") or 0) * pl
        net[s] = net.get(s, 0.0) + signed
        used_gross += abs(signed)

    # (2) HARD SAFETY: total book worst-case drawdown (net signed per symbol,
    #     additive across symbols) must never exceed the budget -- hedging trades
    #     (copy vs invert on one symbol) net exposure down and are allowed larger.
    other_dd = sum(stress * abs(v) for s, v in net.items() if s != venue_symbol)
    sym_dd_budget = dd_budget - other_dd
    if sym_dd_budget <= 0:
        return 0.0
    sym_net_max = sym_dd_budget / stress_trade                  # max |net notional| for the symbol
    sym_net = net.get(venue_symbol, 0.0)
    room = (sym_net_max - sym_net) if direction > 0 else (sym_net_max + sym_net)
    headroom_lots = max(0.0, room) / per_lot_notional

    # (3) the account's own leverage wall (uses the broker's real leverage)
    wall_lots = max(0.0, leverage * equity - used_gross) / per_lot_notional

    # (4) the VENUE'S margin envelope for strategy 1: each position sized so
    #     expected_peak_positions of them fit inside s1_margin_share of equity
    #     at the venue's real margin per lot, and never past the room left.
    #     IG margins gold at 24x on a 1:200 account; the wall cannot see that.
    cap = _envelope_cap(config, venue_symbol, direction, entry, equity, ("1", "zx"),
                        getattr(config, "s1_margin_share", 0.6), slots)
    if cap is not None:
        return min(per_trade_lots, headroom_lots, wall_lots, config.max_lots_per_trade, cap)
    return min(per_trade_lots, headroom_lots, wall_lots, config.max_lots_per_trade)


# ---------------------------------------------------------------- orders
def _ensure_table() -> None:
    with sqlite3.connect(DB) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS vantage_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL,
                source_account TEXT, stance TEXT, symbol TEXT,
                client_direction INTEGER, our_direction INTEGER,
                client_lots REAL, our_lots REAL, expected_usd REAL,
                live_score REAL, sl REAL, tp REAL, mode TEXT, status TEXT,
                ticket INTEGER, fill_price REAL, detail TEXT)""")
        # Migration: client_price (the client's own entry level on the signalled
        # trade) and close_price (where a qualifying order finally closed) were
        # added after the table shipped -- add them if an older DB lacks them.
        cols = {r[1] for r in connection.execute("PRAGMA table_info(vantage_orders)")}
        if "client_price" not in cols:
            connection.execute("ALTER TABLE vantage_orders ADD COLUMN client_price REAL")
        if "close_price" not in cols:
            connection.execute("ALTER TABLE vantage_orders ADD COLUMN close_price REAL")
        # Client closes AS THE DECISION FEED DELIVERS THEM (MySQL + kafka,
        # cent-deflated): the realized-perf join needs the client's actual
        # P&L per close, and the kafka store alone misses most servers. The
        # UNIQUE index makes duplicate delivery a no-op; the first shipped
        # version lacked it (and a cursor), so a dup-bloated table without
        # the index is dropped and rebuilt clean.
        has_uni = any(r[1] == "vcc_uni" for r in connection.execute(
            "PRAGMA index_list(vantage_client_closes)").fetchall()) \
            if connection.execute(
                "SELECT name FROM sqlite_master WHERE name='vantage_client_closes'"
            ).fetchone() else False
        if not has_uni:
            connection.execute("DROP TABLE IF EXISTS vantage_client_closes")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS vantage_client_closes (
                id INTEGER PRIMARY KEY AUTOINCREMENT, login INTEGER,
                server TEXT, canon TEXT, event_utc REAL,
                profit REAL, volume REAL)""")
        connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS vcc_uni ON "
                           "vantage_client_closes(login, canon, event_utc, "
                           "profit, volume)")
        connection.execute("CREATE INDEX IF NOT EXISTS vcc_time ON "
                           "vantage_client_closes(event_utc)")


def _record(**row) -> None:
    _ensure_table()
    with sqlite3.connect(DB) as connection:
        connection.execute(
            """INSERT INTO vantage_orders (created, source_account, stance, symbol,
               client_direction, our_direction, client_lots, our_lots, expected_usd,
               live_score, sl, tp, mode, status, ticket, fill_price, client_price, detail)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (time.time(), row.get("source_account"), row.get("stance"), row.get("symbol"),
             row.get("client_direction"), row.get("our_direction"), row.get("client_lots"),
             row.get("our_lots"), row.get("expected_usd"), row.get("live_score"),
             row.get("sl"), row.get("tp"), row.get("mode"), row.get("status"),
             row.get("ticket"), row.get("fill_price"), row.get("client_price"),
             row.get("detail")))


def _update_close(ticket: int, close_price: float | None) -> None:
    """Stamp the closing price onto the original fill row so the order log can
    show entry -> close for every qualifying (engine-filled) trade. Targets the
    open fill row for this ticket and flips it to 'closed'."""
    if not ticket:
        return
    _ensure_table()
    try:
        with sqlite3.connect(DB) as connection:
            connection.execute(
                "UPDATE vantage_orders SET close_price=?, status='closed' "
                "WHERE ticket=? AND status='filled'",
                (float(close_price) if close_price else None, int(ticket)))
    except Exception:
        pass


def recent_orders(limit: int = 100) -> list[dict]:
    _ensure_table()
    with sqlite3.connect(DB) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(r) for r in connection.execute(
            "SELECT * FROM vantage_orders ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]


#: High-water mark of recorded client closes. The first version re-processed
#: the WHOLE closings window on every 0.3s loop tick (iterrows + inserts, no
#: cursor) -- the loop fell minutes behind the feed and every signal went
#: stale. Now only rows past the cursor are touched, vectorised, deduped by
#: the table's UNIQUE index.
_CLOSE_REC_CURSOR = [0.0]


def _record_client_closes(closings) -> None:
    """Persist NEW client closes from the decision feed (cent-deflated), so
    realized-perf joins run against the same evidence mirror exits acted on."""
    if closings is None or not len(closings):
        return
    try:
        from webapp.trade_feed import _canonical
        ts = (pd.to_datetime(closings["event_time"])
              .astype("datetime64[s]").astype("int64").to_numpy())
        mask = ts > _CLOSE_REC_CURSOR[0]
        if not mask.any():
            return
        sub = closings.loc[mask]
        tss = ts[mask]
        rows = []
        profits = sub["profit"] if "profit" in sub.columns else None
        lots = sub["lots"] if "lots" in sub.columns else None
        for i, (account, symbol) in enumerate(
                zip(sub["account_key"].astype(str), sub["symbol"].astype(str))):
            server, _, login = account.rpartition(":")
            if not login.isdigit():
                continue
            rows.append((int(login), server, _canonical(symbol) or symbol,
                         float(tss[i]),
                         float(profits.iloc[i] or 0.0) if profits is not None else 0.0,
                         float(lots.iloc[i] or 0.0) if lots is not None else 0.0))
        _CLOSE_REC_CURSOR[0] = float(tss.max())
        if not rows:
            return
        with sqlite3.connect(DB) as cx:
            cx.executemany(
                "INSERT OR IGNORE INTO vantage_client_closes "
                "(login, server, canon, event_utc, profit, volume) "
                "VALUES (?,?,?,?,?,?)", rows)
    except Exception:
        pass


def closed_qualifying(limit: int = 60) -> list[dict]:
    """Filled copy/invert trades the engine has since closed, with entry and
    close prices -- the source for the close-price section under the order log."""
    _ensure_table()
    with sqlite3.connect(DB) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(r) for r in connection.execute(
            "SELECT * FROM vantage_orders WHERE status='closed' "
            "AND stance IN ('copy','invert') AND close_price IS NOT NULL "
            "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]


def open_book(config: VantageConfig) -> list[dict]:
    """Everything we are responsible for, live and paper alike."""
    positions = live_positions() if config.mode == "live" else []
    with _LOCK:
        positions += [dict(p) for p in _PAPER.values()]
    return positions


def _brackets(symbol: str, direction: int, lots: float, expected: float,
              entry: float, config: VantageConfig) -> tuple[float, float]:
    """SL/TP prices, or (0, 0) when disabled or tighter than the venue allows."""
    if config.stop_multiple <= 0 and config.target_multiple <= 0:
        return 0.0, 0.0
    try:
        info = _mt5().symbol_info(symbol)
    except Exception:
        return 0.0, 0.0
    if info is None or not info.trade_tick_size or not info.trade_tick_value:
        return 0.0, 0.0
    per_unit = lots * (info.trade_tick_value / info.trade_tick_size)
    if per_unit <= 0 or expected <= 0:
        return 0.0, 0.0
    # The venue rejects levels inside its minimum stop distance -- which is how
    # 571 orders bounced with stops one pip from entry.
    point = getattr(info, "point", 0.0) or 10 ** -info.digits
    spread = max(0.0, getattr(info, "ask", 0.0) - getattr(info, "bid", 0.0))
    floor = max(getattr(info, "trade_stops_level", 0) * point, 3 * spread, 10 * point)
    stop = (expected * config.stop_multiple) / per_unit
    target = (expected * config.target_multiple) / per_unit
    stop = 0.0 if stop < floor else stop
    target = 0.0 if target < floor else target
    digits = info.digits
    if direction > 0:
        return (round(entry - stop, digits) if stop else 0.0,
                round(entry + target, digits) if target else 0.0)
    return (round(entry + stop, digits) if stop else 0.0,
            round(entry - target, digits) if target else 0.0)


def _signed_net_open_lots(config: VantageConfig, source_symbol: str) -> float:
    """long - short lots we hold on the source symbol's CANONICAL name,
    summed across its venue variants (XAUUSD, XAUUSD+, XAUUSDe ... all count
    as XAUUSD). Positions only: resting limits are not exposure."""
    from webapp.trade_feed import _canonical
    canonical = _canonical(source_symbol) or source_symbol
    net = 0.0
    for position in open_book(config):
        symbol = str(position.get("symbol") or "")
        if (_canonical(symbol) or symbol) == canonical:
            net += float(position.get("direction") or 0) * float(position.get("lots") or 0.0)
    return net


def _net_open_lots(config: VantageConfig, source_symbol: str) -> float:
    """|long - short| on the canonical symbol (see _signed_net_open_lots)."""
    return abs(_signed_net_open_lots(config, source_symbol))


def _mirror_entry_threshold(config: VantageConfig, source_symbol: str) -> float | None:
    """The operator's net-lots line for this symbol, or None when unset."""
    from webapp.trade_feed import _canonical
    table = getattr(config, "mirror_entry_net_lots", None) or {}
    if not table:
        return None
    canonical = _canonical(source_symbol) or source_symbol
    for key, value in table.items():
        if (_canonical(str(key)) or str(key)) == canonical:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _recent_range_bps(venue_symbol: str, direction: int, price: float,
                      minutes: int) -> float | None:
    """Distance in bps from `price` to the extreme of the preceding `minutes`
    COMPLETED M1 bars on our limit side: the lowest low for a buy, the highest
    high for a sell (negative when price already sits beyond it). None when
    the venue has no bars, in which case the model's own ask stands."""
    try:
        mt5 = _mt5()
        rates = mt5.copy_rates_from_pos(venue_symbol, mt5.TIMEFRAME_M1, 1, int(minutes))
        if rates is None or len(rates) == 0 or not price or price <= 0:
            return None
        if direction > 0:
            return (price - float(np.min(rates["low"]))) / price * 1e4
        return (float(np.max(rates["high"])) - price) / price * 1e4
    except Exception:
        return None


def execute(config: VantageConfig, account: str, stance: str, source_symbol: str,
            client_direction: int, client_lots: float, client_price: float,
            k: float, expected: float, value: float,
            multiple: float | None = None,
            entry_wait_bps: float | None = None,
            expected_per_lot: float | None = None,
            exit_fe_bps: float | None = None) -> dict:
    """Open one position, subject to every rule. Returns {'filled': bool, ...}."""
    direction = client_direction if stance == "copy" else -client_direction

    def blocked(reason: str, venue: str = "") -> dict:
        with _LOCK:
            _STATE.blocked += 1
        _record(source_account=account, stance=stance, symbol=venue or source_symbol,
                client_direction=client_direction, our_direction=direction,
                client_lots=client_lots, our_lots=0, expected_usd=expected,
                live_score=value, client_price=client_price,
                mode=config.mode, status="blocked", detail=reason)
        return {"filled": False, "reason": reason}

    with _LOCK:
        if _STATE.kill_switch:
            return blocked("kill switch engaged")

    # Trading switched off at the terminal / server: every order fails the
    # same way, so hold new orders for the backoff window instead.
    if _trading_disabled():
        return blocked(f"trading disabled at the venue ({_DISABLED_CODES.get(_TRADE_DISABLED['code'], 'rejected')})")

    # TAPE BIAS GATE: no new mirror entry against an ACTIVE 15-minute tape
    # bias on this symbol (strategy 2's read of where the informed side is
    # positioned). Existing positions run to their mirror exit.
    if getattr(config, "tape_bias_gate", True):
        canon = _tape_canonical(source_symbol)
        bias = _TAPE_GATE_BIAS.get(canon) or _TAPE_BIAS.get(canon)
        if bias and bias.get("active") and int(bias.get("direction") or 0) * direction < 0:
            return blocked(f"against {bias.get('horizon') or bias.get('hold_minutes') or 15}m tape bias "
                           f"{'UP' if bias['direction'] > 0 else 'DOWN'} "
                           f"(p_up {bias['p_up']:.2f}, conf {bias['confidence']:.2f}, "
                           f"until {bias.get('until')})")

    # Symbol must exist on THIS account.
    venue_symbol = resolve_symbol(source_symbol) if config.mode == "live" else source_symbol
    if venue_symbol is None:
        return blocked(f"{source_symbol} not listed on this account")

    # RULE 2: never enter worse than the client.
    quote = current_quote(venue_symbol)
    if quote is None:
        return blocked("no quote to verify entry", venue_symbol)
    bid, ask = quote
    entry = ask if direction > 0 else bid

    # ENTRY-IMPROVEMENT MODEL: when it predicts a retracement worth waiting
    # for, rest a limit that many bps beyond the client's entry instead of
    # going to market. OOS: selective waiting (>= the threshold) both earned
    # more per trade and cut drawdown; below it, market entry wins -- the
    # never-dip runners are the best trades.
    # MIRROR-ENTRY LINE: while our net open book on this symbol is light (at
    # or under the operator's lots figure) the signal goes to market like a
    # plain mirror; only a book already above the line waits for the model's
    # pullback. Gold: 1.0 lot.
    threshold = _mirror_entry_threshold(config, source_symbol)
    light_book = False
    if threshold is not None:
        signed = _signed_net_open_lots(config, source_symbol)
        # Same-side: the exposure this trade ADDS to. A buy against a net
        # short book (or a flat one) reduces exposure and goes to market.
        exposure = (direction * signed if getattr(config, "mirror_entry_same_side", False)
                    else abs(signed))
        light_book = exposure <= threshold
    if (config.use_entry_model and entry_wait_bps is not None
            and client_price > 0 and not light_book
            and _class_enabled(getattr(config, "entry_model_classes", "*"), source_symbol)):
        ask_bps = entry_wait_bps * max(config.entry_ask_fraction, 0.0)
        capped = False
        cap_minutes = int(getattr(config, "entry_range_cap_minutes", 0) or 0)
        if (cap_minutes > 0 and config.mode == "live"
                and _class_enabled(getattr(config, "entry_range_cap_classes", "*"),
                                   source_symbol)):
            range_bps = _recent_range_bps(venue_symbol, direction, client_price,
                                          cap_minutes)
            if range_bps is not None and range_bps < ask_bps:
                ask_bps, capped = max(range_bps, 0.0), True
        if ask_bps >= config.entry_wait_min_bps:
            improved = client_price * (1.0 - direction * ask_bps / 1e4)
            return _place_rescue_limit(
                config, account, stance, venue_symbol, direction, client_lots,
                client_price, k, expected, value, bid, ask, multiple=multiple,
                limit_override=improved,
                ttl_minutes=config.entry_limit_ttl_minutes,
                tag=f"entry model wait {ask_bps:.1f}bps"
                    + (f" (capped at {cap_minutes}m range)" if capped else ""),
                source_symbol=source_symbol)
    if client_price > 0:
        # A symmetric tolerance band, not better-only: entries up to one
        # spread worse than the client's still take the trade at market. The
        # strict rule filtered us into the adversely-selected dipped cohort
        # while the backtest's economics assumed the whole population.
        tolerance = max(0.0, ask - bid) * config.entry_tolerance_spreads
        acceptable = ((entry <= client_price + tolerance) if direction > 0
                      else (entry >= client_price - tolerance))
        if not acceptable:
            # The entry has run away. Instead of losing the signal outright,
            # rest a limit at a price that makes the entry rule true by
            # construction -- pure optionality on signals that were dead.
            if config.rescue_limits:
                return _place_rescue_limit(config, account, stance, venue_symbol,
                                           direction, client_lots, client_price,
                                           k, expected, value, bid, ask,
                                           multiple=multiple,
                                           source_symbol=source_symbol)
            return blocked(f"entry {entry} worse than client {client_price}", venue_symbol)

    # Symbol scaling from backtest edge density; zero means the policy loses
    # money on this instrument and its trades are refused outright.
    from webapp.trade_feed import _canonical
    multiplier = symbol_multipliers(config).get(_canonical(source_symbol), 1.0)
    if multiplier <= 0:
        return blocked(f"{source_symbol} excluded: policy loses on it in backtest",
                       venue_symbol)
    # SIZE. TEMPORARY fixed-lot mode takes every signal at a constant size;
    # otherwise the one drawdown rule: the largest position that keeps the book's
    # worst-case drawdown within 35% of equity, bounded by the real leverage.
    if config.fixed_lot > 0:
        lots = _quantize_lots(venue_symbol, config.fixed_lot) if config.mode == "live" \
            else round(config.fixed_lot, 2)
    else:
        equity_now = float(account_snapshot().get("equity") or 0.0) \
            or (10_000.0 if config.mode != "live" else 0.0)
        # CLIENT-PROPORTIONAL: the client's lots scaled to our equity (0.01 on
        # 5k), bounded by the whole budget / wall / envelope room; otherwise
        # the drawdown rule split across expected_peak_positions slots.
        scaled = _client_scaled_lots(config, account, source_symbol, venue_symbol,
                                     client_lots, equity_now)
        intended = _size_position(config, venue_symbol, entry, direction,
                                  open_book(config), equity_now, config.leverage,
                                  per_slot=scaled is None)
        if scaled is not None:
            intended = min(scaled, intended)
        lots = float(np.floor(intended * 100) / 100.0)      # floor: never exceed budget
        # MAE VETO: if even the MINIMUM lot's predicted tail drawdown exceeds
        # this trade's share of the budget, it cannot be sized within budget at
        # all -- block, rather than floor up to min_lot and overshoot (the
        # stacked-book failure mode).
        stress_trade, tail_bps = _predicted_stress(config, venue_symbol)
        if tail_bps is not None and getattr(config, "mae_veto", True) and equity_now > 0:
            per_lot_notional = _lot_economics(venue_symbol, entry)[1]
            slots = (max(int(getattr(config, "expected_peak_positions", 20)), 1)
                     if scaled is None else 1)
            reservation = config.risk_budget_fraction * equity_now / slots
            worst_at_min = stress_trade * config.min_lot * per_lot_notional
            if per_lot_notional > 0 and worst_at_min > reservation:
                return blocked(f"mae veto: predicted tail drawdown {tail_bps:.0f}bps "
                               f"-> ${worst_at_min:,.0f} at min lot > per-trade "
                               f"budget ${reservation:,.0f}", venue_symbol)
        if lots < config.min_lot:
            # account too small for full drawdown-sizing: take the MINIMUM lot
            # rather than skip the signal (operator's explicit choice; this can
            # push realised drawdown above the 35% target on a small account).
            lots = config.min_lot
        if config.mode == "live":
            lots = _quantize_lots(venue_symbol, lots)
        pass

    # EDGE AT EXECUTED SIZE: expected-$ = per-lot edge x the lots actually
    # being sent, the same basis the hurdle's spread and commission are on.
    if expected_per_lot is not None and expected_per_lot >= 0:
        expected = float(expected_per_lot) * float(lots)

    # COST HURDLE: BOTH tolls, not just the spread. At 0.01 lots the spread
    # rivals the edge, and commission (~$6/lot round-turn on gold/FX/metals)
    # is a second fixed cost that quietly turned "winning" small trades into
    # net losers. No entry unless expected edge clears
    # min_edge_multiple x spread + full commission.
    spread_cost = (ask - bid) * lots * _lot_economics(venue_symbol, entry)[0]
    commission_cost = _commission_cost(venue_symbol, lots, config)
    hurdle = config.min_edge_multiple * spread_cost + commission_cost
    if hurdle > 0 and expected < hurdle:
        return blocked(
            f"edge ${expected:,.2f} below hurdle ${hurdle:,.2f} "
            f"({config.min_edge_multiple:.1f}x spread ${spread_cost:,.2f} + "
            f"commission ${commission_cost:,.2f})", venue_symbol)

    book_now = open_book(config)

    # RISK-BUDGET UTILISATION: keep the whole budget working. Underused
    # budget scales entries up (bounded, spread over the remaining slots so
    # the book stays diversified across many positions); a full budget is
    # made room in by harvesting the MOST PROFITABLE positions first.
    if config.risk_scaling:
        snapshot_now = account_snapshot()
        equity_now = float(snapshot_now.get("equity") or 0.0)
        if equity_now > 0:
            target = (config.risk_budget_fraction * equity_now
                      * config.risk_sigma_share)
            used = sum(_position_risk(p) for p in book_now)
            base_risk = _position_risk({"symbol": venue_symbol, "lots": lots,
                                        "open_price": entry,
                                        "direction": direction})
            headroom = target - used
            if base_risk > 0 and headroom > base_risk:
                slots = max(1, config.max_open_positions - len(book_now))
                scale = float(np.clip((headroom / slots) / base_risk,
                                      1.0, config.risk_boost_max))
                if scale > 1.05:
                    lots = round(min(lots * scale, config.max_lots_per_trade), 2)
                    expected *= scale
            elif base_risk > 0 and headroom < 0:
                if not _harvest(config, book_now, base_risk):
                    return blocked("risk budget fully deployed, nothing "
                                   "profitable to harvest", venue_symbol)

    # NETTING: an opposing signal on a symbol we already hold CLOSES the oldest
    # opposite position instead of opening a hedge. Two opposite same-size
    # positions pay the spread twice for zero net exposure -- a locked loss by
    # construction, which is exactly the pattern that showed up live. With
    # net_only_profitable, a LOSING opposite is left to its own mirror exit
    # (closing it would realise the loss AND kill its signal's remaining
    # alpha) and the new signal opens as a hedge instead.
    if config.net_by_symbol:
        opposite = [p for p in book_now
                    if p["symbol"] == venue_symbol and p["direction"] == -direction]
        if opposite:
            oldest = min(opposite, key=lambda p: p.get("opened_at") or "")
            if (not config.net_only_profitable) or _unrealized(oldest) > 0:
                close_position(config, oldest,
                               f"netted against opposing {stance} signal")
                _record(source_account=account, stance=stance, symbol=venue_symbol,
                        client_direction=client_direction, our_direction=direction,
                        client_lots=client_lots, our_lots=0, expected_usd=expected,
                        live_score=value, client_price=client_price,
                        mode=config.mode, status="netted",
                        detail=f"closed opposite #{oldest['ticket']} instead of hedging")
                return {"filled": False, "netted": True}

    # CONCENTRATION is measured in NET USD EXPOSURE, not position count: ten
    # hedged tickets carry less symbol risk than two stacked one-way, and the
    # count cap blocked exactly backwards. The net-USD test lives in the
    # leverage-wall section below (a trade that REDUCES net is always let in).

    # RULE 3: leverage wall (modelled in paper too, or paper overstates capacity).
    snapshot = account_snapshot()
    equity = float(snapshot.get("equity") or 0.0) or (10_000.0 if config.mode != "live" else 0.0)
    book = open_book(config)

    # HELD-TICKET RETIREMENT: a stop-out-held position closes the moment a
    # newer valid signal on the same symbol+direction has a WORSE (or equal)
    # entry than ours -- whether that signal would fill, pend, or be skipped
    # downstream. We already own the better entry; the new signal is
    # consumed as this position's exit trigger.
    held_now = _held_map()
    if held_now:
        for p in list(book):
            hrow = held_now.get(int(p.get("ticket") or 0))
            if hrow is None or p["symbol"] != venue_symbol \
                    or int(p.get("direction") or 0) != int(direction):
                continue
            he = float(hrow.get("entry") or p.get("open_price") or 0.0)
            worse = (direction > 0 and he <= float(entry)) \
                or (direction < 0 and he >= float(entry))
            if he > 0 and worse:
                if close_position(config, p, "held-replacement: newer signal "
                                             "with worse entry"):
                    _release_held(int(p.get("ticket") or 0))
                    try:
                        book.remove(p)
                    except ValueError:
                        pass
                return blocked("closed HELD position instead (our entry "
                               f"{he:g} beats the new {float(entry):g})",
                               venue_symbol)

    line = [p for p in book if p["symbol"] == venue_symbol
            and int(p.get("direction") or 0) == int(direction)]
    if line and getattr(config, "side_score_guard", False) and len(line) >= 2:
        # NET RISK FOLLOWS THE SCORE DIFFERENTIAL: no stacking a side whose
        # rolling average edge is weaker than the opposite side's.
        try:
            from webapp.trade_feed import _canonical
            canon = _canonical(venue_symbol) or venue_symbol
            same, opp = _side_avg(canon, direction), _side_avg(canon, -direction)
            if opp > same:
                return blocked(
                    f"side-score guard: {venue_symbol} opposite side edge "
                    f"{opp:.3f} > ours {same:.3f} with {len(line)} stacked",
                    venue_symbol)
        except Exception:
            pass
    if line and getattr(config, "avg_entry_guard", False):
        # AVERAGING GUARD: a losing line only ever averages at a BETTER
        # price than its current extreme -- buy lower / sell higher; a
        # winning line adds freely.
        try:
            floating = sum(_unrealized(p) for p in line)
            if floating < 0:
                if direction > 0:
                    extreme = min(float(p.get("open_price") or 0)
                                  for p in line)
                    better = float(entry) < extreme
                else:
                    extreme = max(float(p.get("open_price") or 0)
                                  for p in line)
                    better = float(entry) > extreme
                if not better:
                    return blocked(
                        f"averaging guard: line floating {floating:+,.0f} "
                        f"and entry {float(entry):g} not better than "
                        f"extreme {extreme:g}", venue_symbol)
        except Exception:
            pass
    if equity > 0 and config.leverage > 0:
        def measure(p):
            return _notional(p["symbol"], p["lots"], p["open_price"])
        used = sum(measure(p) for p in book)
        needed = _notional(venue_symbol, lots, entry)
        # DIVERSIFICATION IN THE BINDING UNITS: one instrument may own at
        # most max_symbol_share of the leverage wall's notional. The count
        # cap alone let 16 gold positions eat the entire wall. Harvest is
        # WITHIN the symbol first -- gold profit funds new gold, other
        # instruments keep their untouched share of the wall.
        same_symbol = [p for p in book if p["symbol"] == venue_symbol]
        # NET signed exposure for the symbol: hedges offset, one-way stacks.
        signed_net = sum(measure(p) * (1 if p.get("direction", 0) > 0 else -1)
                         for p in same_symbol)
        reduces_net = (signed_net > 0 and direction < 0) \
            or (signed_net < 0 and direction > 0)
        symbol_used = 0.0 if reduces_net else abs(signed_net)
        symbol_cap_usd = config.max_symbol_share * _wall_lev(config) * equity
        per_lot_notional = _lot_economics(venue_symbol, entry)[1]

        def _fit_to(headroom: float, reason: str):
            """SIZE DOWN into the available capacity rather than losing the
            trade -- a large-notional signal capped to headroom keeps its
            edge; blocking it outright forfeited it entirely."""
            nonlocal lots, needed
            if per_lot_notional <= 0:
                return blocked(reason, venue_symbol)
            fitted = round(min(lots, headroom / per_lot_notional), 2)
            if config.mode == "live" and fitted >= 0.01:
                fitted = _quantize_lots(venue_symbol, fitted)
            if fitted < 0.01 or fitted * per_lot_notional > headroom * 1.001:
                return blocked(reason + " (no headroom for even minimum "
                               "size)", venue_symbol)
            if fitted < lots:
                _log(f"sized down {venue_symbol} {lots} -> {fitted} lots "
                     f"to fit {reason}")
            lots = fitted
            needed = _notional(venue_symbol, lots, entry)
            return None

        if symbol_used + needed > symbol_cap_usd:
            overflow = symbol_used + needed - symbol_cap_usd
            if not (getattr(config, "harvest_enabled", False)
                    and _harvest(config, same_symbol, overflow,
                                 measure=measure, new_expected=expected)):
                result = _fit_to(max(0.0, symbol_cap_usd - symbol_used),
                                 f"{venue_symbol} {config.max_symbol_share:.0%} share")
                if result is not None:
                    return result
        if used + needed > _wall_lev(config) * equity:
            # The wall is the operating boundary, not a dead end: harvest
            # first (locking gains to admit the signal at full size), size
            # down second, block only when even minimum size cannot fit.
            overflow = used + needed - _wall_lev(config) * equity
            if not (getattr(config, "harvest_enabled", False)
                    and _harvest(config, book, overflow, measure=measure,
                                 new_expected=expected)):
                result = _fit_to(max(0.0, _wall_lev(config) * equity - used),
                                 "leverage wall")
                if result is not None:
                    return result

    if len(book) >= config.max_open_positions:
        rotated = _rotate(config, book, expected)
        if not rotated:
            return blocked("max open positions (mirror at capacity; "
                           "slots free via exits/sweep)", venue_symbol)

    if config.mode != "live":
        _PAPER_SEQ[0] += 1
        ticket = _PAPER_SEQ[0]
        with _LOCK:
            hold = predict_hold(config, value or 0.5, direction, lots,
                                _notional(venue_symbol, lots, entry), account, stance)
            _PAPER[ticket] = {"ticket": ticket, "source": account, "stance": stance,
                              "symbol": venue_symbol, "direction": direction,
                              "lots": lots, "client_lots": client_lots,
                              "open_price": entry, "expected_usd": expected,
                              "deadline": (datetime.utcnow() + timedelta(seconds=hold))
                                          if hold else None,
                              "opened_at": datetime.utcnow().isoformat(timespec="seconds")}
            _STATE.filled += 1
        _record(source_account=account, stance=stance, symbol=venue_symbol,
                client_direction=client_direction, our_direction=direction,
                client_lots=client_lots, our_lots=lots, expected_usd=expected,
                live_score=value, client_price=client_price,
                mode="paper", status="filled", ticket=ticket,
                fill_price=entry, detail="paper fill")
        _log(f"paper {stance.upper()} {'BUY' if direction > 0 else 'SELL'} "
             f"{venue_symbol} {lots} @ {entry} (exp ${expected:,.0f})")
        return {"filled": True, "ticket": ticket}

    try:
        mt5 = _mt5()
        sl, tp = _brackets(venue_symbol, direction, lots, expected, entry, config)
        # EXIT-FE TAKE-PROFIT (validated OOS Aug 20-31: TP at 1.0x the
        # predicted favorable excursion lifted PF 3.46 -> 4.01 over
        # mirror-only, and live it frees winning slots early). Mirror exit
        # remains the fallback when the TP is never touched.
        if (exit_fe_bps is not None and exit_fe_bps >= 6.0
                and _class_enabled(getattr(config, "exit_model_classes", "*"), venue_symbol)):
            try:
                info = mt5.symbol_info(venue_symbol)
                digs = info.digits if info else 5
                min_dist = (getattr(info, "trade_stops_level", 0) or 0) \
                    * (getattr(info, "point", 0.0) or 0.0)
                fe_tp = round(entry * (1 + direction * exit_fe_bps / 1e4),
                              digs)
                if abs(fe_tp - entry) > max(min_dist, 0.0):
                    tp = fe_tp
            except Exception:
                pass
        lots = _fit_margin(venue_symbol, direction, lots, entry)
        if lots <= 0:
            return blocked("no free margin at the venue for even the minimum lot", venue_symbol)
        request = {"action": mt5.TRADE_ACTION_DEAL, "symbol": venue_symbol,
                   "volume": lots, "price": entry, "deviation": 20, "magic": 909090,
                   "type": mt5.ORDER_TYPE_BUY if direction > 0 else mt5.ORDER_TYPE_SELL,
                   "comment": _comment_for(stance, account, value, multiple),
                   "type_filling": _filling_for(venue_symbol)}
        if sl:
            request["sl"] = sl
        if tp:
            request["tp"] = tp
        result = mt5.order_send(request)
        ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        if ok:
            _mt5_cache_clear()
        with _LOCK:
            if ok:
                _STATE.filled += 1
            else:
                _STATE.blocked += 1
        if ok:
            hold = predict_hold(config, value or 0.5, direction, lots,
                                _notional(venue_symbol, lots, entry), account, stance)
            if hold:
                _DEADLINES[int(result.order)] = datetime.utcnow() + timedelta(seconds=hold)
        # informational only (cent/contract normalization factor for the log)
        adjust = _contract_adjust(account.rpartition(":")[0], source_symbol,
                                  venue_symbol)
        _record(source_account=account, stance=stance, symbol=venue_symbol,
                client_direction=client_direction, our_direction=direction,
                client_lots=client_lots, our_lots=lots, expected_usd=expected,
                live_score=value, client_price=client_price,
                sl=sl or None, tp=tp or None, mode="live",
                status="filled" if ok else "rejected",
                ticket=getattr(result, "order", None) if ok else None,
                fill_price=getattr(result, "price", None) if ok else None,
                detail=(f"strat=1|stance={stance}|src={account}"
                        f"|score={value:.3f}"
                        f"|mult={f'{multiple:.1f}' if multiple is not None else 'na'}"
                        f"|sizing={config.sizing_policy}|k={k:.5f}"
                        f"|calib={config.dd_calibration:.0f}|adj={adjust:.2f}"
                        f"|lots={lots}|client_lots={client_lots}"
                        f"|edge=${expected:,.2f}") if ok else
                       f"retcode {getattr(result, 'retcode', '?')} "
                       f"{getattr(result, 'comment', '')}")
        if not ok:
            _note_trade_disabled(getattr(result, "retcode", 0), getattr(result, "comment", ""))
        _log(("FILLED " if ok else "REJECTED ")
             + f"{stance.upper()} {'BUY' if direction > 0 else 'SELL'} {venue_symbol} "
             + f"{lots} @ {entry}"
             + ("" if ok else f" [{getattr(result, 'retcode', '?')} "
                              f"{getattr(result, 'comment', '')}]"))
        return {"filled": ok}
    except Exception as error:
        return blocked(f"{type(error).__name__}: {error}", venue_symbol)


_EXIT_MODEL = None
_EXIT_META: tuple[list, list] | None = None
_DEADLINES: dict[int, datetime] = {}     # live ticket -> model deadline


def _exit_model():
    global _EXIT_MODEL, _EXIT_META
    if _EXIT_MODEL is None:
        path = ARTIFACTS / "exit_model.txt"
        meta = ARTIFACTS / "exit_model_meta.txt"
        if path.exists() and meta.exists():
            import lightgbm as lgb
            _EXIT_MODEL = lgb.Booster(model_file=str(path))
            features, names = meta.read_text(encoding="utf-8").split("---")
            _EXIT_META = ([f for f in features.strip().splitlines()],
                          names.strip().split(","))
    return _EXIT_MODEL


_HOLD_SECONDS = {"5m": 300, "30m": 1800, "2h": 7200, "6h": 21600}


def predict_hold(config: VantageConfig, value: float, direction: int,
                 lots: float, notional: float, account: str,
                 stance: str = "copy") -> float | None:
    """Seconds to hold this fill, from the exit classifier. None = mirror only.

    Inverts are mirror-only by default (`invert_mirror_only`): they profit only
    by capturing the client's ENTIRE adverse round-trip, so an early model exit
    breaks the very assumption the backtest priced them on. Copies still ride
    the exit model to their winner sweet spot."""
    if not config.use_exit_model:
        return None
    if stance == "invert" and getattr(config, "invert_mirror_only", True):
        return None
    model = _exit_model()
    if model is None or _EXIT_META is None:
        return None
    try:
        features, names = _EXIT_META
        history = account_history()
        row = history.loc[account] if account in history.index else None
        now = datetime.utcnow()
        usual = float(row.get("mean_notional")) if row is not None else np.nan
        values = {"score": value, "direction": float(direction),
                  "volume_lots": lots, "hour_": now.hour, "weekday_": now.weekday(),
                  "log_notional": float(np.log1p(max(notional, 0.0))),
                  "win_rate": float(row.get("win_rate", np.nan)) if row is not None else np.nan,
                  "mean_pnl": float(row.get("mean_pnl", np.nan)) if row is not None else np.nan,
                  "pnl_std": float(row.get("pnl_std", np.nan)) if row is not None else np.nan,
                  "mean_abs_pnl": float(row.get("mean_abs_pnl", np.nan)) if row is not None else np.nan,
                  "trades": float(row.get("trades", np.nan)) if row is not None else np.nan,
                  "notional_vs_usual": notional / usual if usual and usual > 0 else np.nan,
                  "symbol_code": np.nan}
        vector = np.array([[values.get(f, np.nan) for f in features]], dtype="float64")
        klass = int(np.argmax(model.predict(vector)[0]))
        return float(_HOLD_SECONDS[names[klass]])
    except Exception:
        return None


def _tend_deadlines(config: VantageConfig) -> None:
    """Close positions whose model hold has elapsed (exit = min(mirror, model))."""
    now = datetime.utcnow()
    for position in open_book(config):
        deadline = position.get("deadline") or _DEADLINES.get(position.get("ticket"))
        if deadline and now >= deadline:
            if close_position(config, position, "exit model hold elapsed"):
                _DEADLINES.pop(position.get("ticket"), None)


def _place_rescue_limit(config: VantageConfig, account: str, stance: str,
                        venue_symbol: str, direction: int, client_lots: float,
                        client_price: float, k: float, expected: float,
                        value: float | None, bid: float, ask: float,
                        multiple: float | None = None,
                        limit_override: float | None = None,
                        ttl_minutes: float | None = None,
                        tag: str = "rescue limit",
                        lots_override: float | None = None,
                        source_symbol: str | None = None) -> dict:
    """Rest a limit where the entry rule holds by construction.

    Buy-limit at client entry minus the spread buffer (sell-limit above it):
    a fill can only ever happen at a price better than the client's by at
    least the toll. Short TTL; cancelled if the client exits first.

    `limit_override` pins the resting price directly (the entry-improvement
    model uses it: client price improved by the predicted bps), with
    `ttl_minutes` and `tag` letting that caller keep its own clock and its
    own order-log identity.
    """
    from webapp.trade_feed import _canonical
    multiplier = symbol_multipliers(config).get(_canonical(venue_symbol), 1.0)
    if multiplier <= 0:
        return {"filled": False, "reason": "symbol excluded"}
    adjust = _contract_adjust(account.rpartition(":")[0], venue_symbol,
                              venue_symbol) if config.mode == "live" else 1.0
    _ = adjust                                          # (kept for signature parity)
    if config.fixed_lot > 0:
        lots = _quantize_lots(venue_symbol, config.fixed_lot) if config.mode == "live" \
            else round(config.fixed_lot, 2)
    else:
        equity_now = float(account_snapshot().get("equity") or 0.0) \
            or (10_000.0 if config.mode != "live" else 0.0)
        scaled = _client_scaled_lots(config, account, source_symbol or venue_symbol,
                                     venue_symbol, client_lots, equity_now)
        intended = _size_position(config, venue_symbol, client_price, direction,
                                  open_book(config), equity_now, config.leverage,
                                  per_slot=scaled is None)
        if scaled is not None:
            intended = min(scaled, intended)
        lots = float(np.floor(intended * 100) / 100.0)  # floor: never exceed budget
        # Same MAE veto as market entries: a resting limit must not be the
        # back door for a trade whose predicted tail drawdown cannot fit the
        # per-trade budget even at minimum lot.
        stress_trade, tail_bps = _predicted_stress(config, venue_symbol)
        if tail_bps is not None and getattr(config, "mae_veto", True) and equity_now > 0:
            per_lot_notional = _lot_economics(venue_symbol, client_price)[1]
            slots = (max(int(getattr(config, "expected_peak_positions", 20)), 1)
                     if scaled is None else 1)
            reservation = config.risk_budget_fraction * equity_now / slots
            worst_at_min = stress_trade * config.min_lot * per_lot_notional
            if per_lot_notional > 0 and worst_at_min > reservation:
                reason = (f"mae veto ({tag}): predicted tail drawdown {tail_bps:.0f}bps "
                          f"-> ${worst_at_min:,.0f} at min lot > per-trade budget "
                          f"${reservation:,.0f}")
                with _LOCK:
                    _STATE.blocked += 1
                _record(source_account=account, stance=stance, symbol=venue_symbol,
                        client_direction=direction if stance == "copy" else -direction,
                        our_direction=direction, client_lots=client_lots, our_lots=0,
                        expected_usd=expected, live_score=value,
                        client_price=client_price, mode=config.mode,
                        status="blocked", detail=reason)
                return {"filled": False, "reason": reason}
        if lots < config.min_lot:
            lots = config.min_lot                       # small account: take the floor
    if lots_override is not None and lots_override > 0:
        lots = float(lots_override)        # replacing a known trade: its own size ...
        try:                               # ... inside strategy 1's margin envelope
            equity_env = float(account_snapshot().get("equity") or 0.0)
            cap = _envelope_cap(config, venue_symbol, direction, client_price, equity_env, ("1", "zx"),
                                getattr(config, "s1_margin_share", 0.6),
                                max(int(getattr(config, "expected_peak_positions", 20)), 1))
            if cap is not None and cap < lots:
                lots = float(np.floor(cap * 100) / 100.0)
        except Exception:
            pass
    # The SAME risk gates as market entries. This path used to bypass them
    # all, and at calibrated sizes a single resting limit exceeded the whole
    # leverage wall (ticket 1917898471: 1.07 lots gold = $467k notional on a
    # $250k wall).
    book = open_book(config)
    if len(book) + len(_PENDING) >= config.max_open_positions:
        return {"filled": False, "reason": "max open positions"}
    snapshot = account_snapshot()
    equity = float(snapshot.get("equity") or 0.0)
    if equity > 0 and config.leverage > 0:
        def measure(p):
            return _notional(p["symbol"], p["lots"], p["open_price"])
        used = sum(measure(p) for p in book)
        pending_used = sum(
            _notional(o["symbol"], o["lots"], o.get("limit") or client_price)
            for o in _PENDING.values())
        symbol_used = sum(measure(p) for p in book
                          if p["symbol"] == venue_symbol)
        wall = _wall_lev(config) * equity
        allowed = max(0.0, min(wall - used - pending_used,
                               config.max_symbol_share * wall - symbol_used))
        per_lot = _lot_economics(venue_symbol, client_price)[1]
        if per_lot > 0:
            lots = min(lots, allowed / per_lot)
        lots = round(lots, 2)
        if lots >= 0.01 and config.mode == "live":
            lots = _quantize_lots(venue_symbol, lots)
        if lots < 0.01 or lots * per_lot > allowed * 1.001:
            return {"filled": False, "reason": "no leverage headroom for limit"}
    elif config.mode == "live":
        lots = _quantize_lots(venue_symbol, lots)
    if limit_override is not None and limit_override > 0:
        limit_price = float(limit_override)
    else:
        spread = max(0.0, ask - bid)
        buffer = spread * config.limit_buffer_spreads
        limit_price = (client_price - buffer) if direction > 0 else (client_price + buffer)
    ttl = ttl_minutes if ttl_minutes is not None else config.limit_ttl_minutes
    expires = datetime.utcnow() + timedelta(minutes=ttl)

    if config.mode != "live":
        _PAPER_SEQ[0] += 1
        ticket = _PAPER_SEQ[0]
        with _LOCK:
            _PENDING[ticket] = {"ticket": ticket, "source": account, "stance": stance,
                                "symbol": venue_symbol, "direction": direction,
                                "lots": lots, "client_lots": client_lots,
                                "limit": limit_price, "expected_usd": expected,
                                "expires": expires}
        _record(source_account=account, stance=stance, symbol=venue_symbol,
                client_direction=direction if stance == "copy" else -direction,
                our_direction=direction, client_lots=client_lots, our_lots=lots,
                expected_usd=expected, live_score=value, client_price=client_price,
                mode="paper",
                status="pending", ticket=ticket, fill_price=limit_price,
                detail=f"{tag} @ {limit_price} ttl {ttl:.0f}m")
        _log(f"paper LIMIT ({tag}) {stance.upper()} {'BUY' if direction > 0 else 'SELL'} "
             f"{venue_symbol} {lots} @ {limit_price}")
        return {"filled": False, "pending": ticket}

    try:
        mt5 = _mt5()
        info = mt5.symbol_info(venue_symbol)
        digits = info.digits if info else 5
        # a resting limit reserves margin at the venue like a fill would
        lots = _fit_margin(venue_symbol, direction, lots, limit_price)
        if lots <= 0:
            return {"filled": False, "reason": "no free margin for the limit"}
        # GTC, not broker-side expiry: this venue rejects ORDER_TIME_SPECIFIED
        # outright (retcode 10022 on every attempt). The TTL is enforced by
        # _tend_pending instead, which cancels our own orders past deadline --
        # the same clock paper mode already runs on.
        result = mt5.order_send({
            "action": mt5.TRADE_ACTION_PENDING, "symbol": venue_symbol,
            "volume": lots, "price": round(limit_price, digits),
            "type": (mt5.ORDER_TYPE_BUY_LIMIT if direction > 0
                     else mt5.ORDER_TYPE_SELL_LIMIT),
            "type_time": mt5.ORDER_TIME_GTC,
            "magic": 909090,
            "comment": _comment_for(stance, account, value, multiple),
            "type_filling": mt5.ORDER_FILLING_RETURN})
        ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        if ok:
            _mt5_cache_clear()
            with _LOCK:
                _PENDING[int(result.order)] = {
                    "ticket": int(result.order), "live": True,
                    "source": account, "stance": stance,
                    "symbol": venue_symbol, "direction": direction,
                    "lots": lots, "client_lots": client_lots,
                    "limit": round(limit_price, digits),
                    "expected_usd": expected, "expires": expires}
        _record(source_account=account, stance=stance, symbol=venue_symbol,
                client_direction=direction if stance == "copy" else -direction,
                our_direction=direction, client_lots=client_lots, our_lots=lots,
                expected_usd=expected, live_score=value, client_price=client_price,
                mode="live",
                status="pending" if ok else "rejected",
                ticket=getattr(result, "order", None) if ok else None,
                fill_price=round(limit_price, digits),
                detail=(f"{tag} ttl {ttl:.0f}m" if ok else
                        f"limit rejected {getattr(result, 'retcode', '?')} "
                        f"{getattr(result, 'comment', '')}"))
        if not ok:
            _note_trade_disabled(getattr(result, "retcode", 0), getattr(result, "comment", ""))
        _log((f"LIMIT PLACED ({tag}) {stance.upper()} " if ok else "LIMIT REJECTED ")
             + f"{venue_symbol} {lots} @ {round(limit_price, digits)}")
        return {"filled": False, "pending": ok}
    except Exception as error:
        _log(f"limit error {venue_symbol}: {error}")
        return {"filled": False, "reason": str(error)}


def _tend_pending(config: VantageConfig) -> None:
    """Fill (paper) / detect fills (live), and expire past-TTL limits.

    The venue refuses broker-side expiry, so the TTL is enforced here for live
    orders too: past deadline they are REMOVEd; a live pending that vanished
    from the broker's order list before its deadline was filled (its position
    shows up through the normal book) and is retired from tracking.
    """
    if not _PENDING:
        return
    now = datetime.utcnow()
    with _LOCK:
        pending = list(_PENDING.values())
    for order in pending:
        if order.get("live"):
            try:
                mt5 = _mt5()
                still_open = mt5.orders_get(ticket=order["ticket"])
                if not still_open:
                    with _LOCK:
                        _PENDING.pop(order["ticket"], None)
                        _STATE.filled += 1
                    _log(f"LIMIT FILLED #{order['ticket']} {order['symbol']} "
                         f"{order['lots']} @ {order['limit']}")
                elif now >= order["expires"]:
                    mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE,
                                    "order": order["ticket"]})
                    with _LOCK:
                        _PENDING.pop(order["ticket"], None)
                    _log(f"LIMIT expired #{order['ticket']} {order['symbol']}")
            except Exception:
                pass
            continue
        if now >= order["expires"]:
            with _LOCK:
                _PENDING.pop(order["ticket"], None)
            _log(f"paper LIMIT expired #{order['ticket']} {order['symbol']}")
            continue
        quote = current_quote(order["symbol"])
        if quote is None:
            continue
        bid, ask = quote
        touch = (ask <= order["limit"]) if order["direction"] > 0 else (bid >= order["limit"])
        if touch:
            with _LOCK:
                _PENDING.pop(order["ticket"], None)
                _PAPER[order["ticket"]] = {
                    "ticket": order["ticket"], "source": order["source"],
                    "stance": order["stance"], "symbol": order["symbol"],
                    "direction": order["direction"], "lots": order["lots"],
                    "client_lots": order["client_lots"],
                    "open_price": order["limit"],
                    "expected_usd": order["expected_usd"],
                    "opened_at": now.isoformat(timespec="seconds")}
                _STATE.filled += 1
            _log(f"paper LIMIT FILLED #{order['ticket']} {order['symbol']} "
                 f"{order['lots']} @ {order['limit']}")


def _cancel_pending_for(config: VantageConfig, account: str, venue_symbol: str) -> None:
    """The client exited: any resting rescue limit from them dies with the
    signal -- filling after their exit would open a trade with no thesis."""
    with _LOCK:
        doomed = [t for t, o in _PENDING.items()
                  if o["source"] == account and o["symbol"] == venue_symbol]
        for ticket in doomed:
            _PENDING.pop(ticket, None)
    for ticket in doomed:
        _log(f"paper LIMIT cancelled #{ticket} (client exited)")
    if config.mode == "live":
        try:
            mt5 = _mt5()
            for order in (mt5.orders_get(symbol=venue_symbol) or []):
                source = _source_of(order.comment or "")
                if source and source in (account, account[-8:]):
                    mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE,
                                    "order": order.ticket})
                    _log(f"LIMIT cancelled #{order.ticket} (client exited)")
        except Exception:
            pass


def _harvest(config: VantageConfig, book: list[dict], needed: float,
             measure=None, new_expected: float | None = None) -> bool:
    """Free `needed` units (risk dollars, or notional via `measure`) by
    partially closing profitable positions with the LEAST expected value
    remaining -- the winner closest to exhaustion, never the biggest runner.

    The first day live proved why the old most-profitable-first rule was a
    self-inflicted wound: harvest banked +$1,401 in early-cut winners while
    the positions left for mirror exits netted +$17.60 -- our own capacity
    management was stripping the mirror alpha. Churn guard: when the new
    trade's expected value is known, no position gives up more remaining
    value than new_expected/1.2."""
    measure = measure or _position_risk
    scored = []
    for position in book:
        profit = _unrealized(position)
        if profit <= 0:
            continue
        expected = None
        if position.get("expected_usd"):
            expected = float(position["expected_usd"])
        else:
            try:
                with sqlite3.connect(DB) as connection:
                    row = connection.execute(
                        "SELECT expected_usd FROM vantage_orders WHERE "
                        "ticket = ? ORDER BY id DESC LIMIT 1",
                        (int(position["ticket"]),)).fetchone()
                expected = float(row[0]) if row and row[0] else None
            except Exception:
                expected = None
        remaining = (max(0.0, expected - profit) if expected is not None
                     else profit)      # unknown expectation: treat cautiously
        if new_expected is not None and remaining > new_expected / 1.2:
            continue                   # would give up more than the new trade earns
        scored.append((remaining, position, profit))
    scored.sort(key=lambda item: item[0])
    freed = 0.0
    for remaining, position, profit in scored:
        if freed >= needed:
            break
        risk = measure(position)
        if risk <= 0:
            continue
        fraction = min(1.0, (needed - freed) / risk)
        slice_lots = min(float(position["lots"]),
                         max(0.01, round(float(position["lots"]) * fraction, 2)))
        if config.mode == "live":
            slice_lots = min(float(position["lots"]),
                             _quantize_lots(position["symbol"], slice_lots))
        slice_of = dict(position)
        slice_of["lots"] = slice_lots
        if close_position(config, slice_of,
                          f"harvested (locked {profit:+,.2f}) for new signal"):
            freed += risk * (slice_lots / max(float(position["lots"]), 1e-9))
    return freed >= needed * 0.8


def _rotate(config: VantageConfig, book: list[dict], new_expected: float) -> bool:
    """RETIRED as a winner-cutter, kept as the full-book gate.

    Measured live (21 rotated positions joined to their clients' eventual
    outcomes at our size): rotation locked +$205 and FORFEITED +$4,284 of
    continuation -- the model's 'remaining value' guess undershoots a living
    profitable mirror ~20x, so every churn comparison against a newcomer's
    (equally model-guessed) expectation was a losing trade of real alpha for
    speculative alpha. A full book now means the mirror is at capacity:
    nothing living is cut; slots free through mirror exits and the orphan
    sweep (which already closes/brackets client-dead positions)."""
    return False


def _close_comment(reason: str) -> str:
    """Reason-coded close comments, so P&L-by-logic audits read straight
    from MT5 history instead of requiring log correlation."""
    text = (reason or "").lower()
    if "harvest" in text:
        return "x1hvs"
    if "mirror" in text:
        return "x1mir"
    if "netted" in text or "netting" in text:
        return "x1net"
    if "rotated" in text:
        return "x1rot"
    if "hold elapsed" in text or "exit model" in text:
        return "x2ttl"
    if "retry" in text:
        return "x1rty"
    return "x1cls"


def close_position(config: VantageConfig, position: dict, reason: str) -> bool:
    """Close one position, live or paper."""
    if position["ticket"] in _PAPER:
        quote = current_quote(position["symbol"])
        mark = (quote[0] if position["direction"] > 0 else quote[1]) if quote else None
        with _LOCK:
            _PAPER.pop(position["ticket"], None)
        pnl = 0.0
        if mark:
            move = (mark - position["open_price"]) * position["direction"]
            pnl = move / max(position["open_price"], 1e-9) * _notional(
                position["symbol"], position["lots"], position["open_price"])
            _PAPER_PNL[0] += pnl
        _update_close(position["ticket"], mark)
        _log(f"paper CLOSE #{position['ticket']} {position['symbol']} ({reason}) {pnl:+,.2f}")
        return True
    try:
        mt5 = _mt5()
        quote = current_quote(position["symbol"])
        if quote is None:
            return False
        closing_buy = position["direction"] < 0
        # HEDGING account: `position: ticket` closes THIS ticket. Without it
        # the same order would OPEN an opposing position -- a locked hedge
        # pair paying double spread, never a close.
        result = mt5.order_send({
            "action": mt5.TRADE_ACTION_DEAL, "position": position["ticket"],
            "symbol": position["symbol"], "volume": float(position["lots"]),
            "type": mt5.ORDER_TYPE_BUY if closing_buy else mt5.ORDER_TYPE_SELL,
            "price": quote[1] if closing_buy else quote[0], "deviation": 20,
            "magic": 909090, "comment": _close_comment(reason),
            "type_filling": _filling_for(position["symbol"])})
        ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        if ok:
            close_mark = getattr(result, "price", None) or (
                quote[1] if closing_buy else quote[0])
            _update_close(position["ticket"], close_mark)
        _log(("CLOSED " if ok else "CLOSE FAILED ")
             + f"{position['symbol']} {position['lots']} ({reason})")
        if not ok:
            # GUARANTEE: a rejected close must still end in a close. A tight
            # server-side stop one spread behind the market survives our own
            # process dying, and _tend_closes keeps retrying the market close.
            _guarantee_close(position, quote)
        return ok
    except Exception:
        return False


#: Positions whose close was rejected: ticket -> (position dict, reason).
#: Retried every tick until the venue confirms the position is gone.
_CLOSE_RETRY: dict[int, tuple[dict, str]] = {}


def _guarantee_close(position: dict, quote: tuple[float, float]) -> None:
    try:
        mt5 = _mt5()
        info = mt5.symbol_info(position["symbol"])
        point = (getattr(info, "point", 0.0) or 1e-5) if info else 1e-5
        digits = info.digits if info else 5
        bid, ask = quote
        spread = max(ask - bid, point)
        floor = max((getattr(info, "trade_stops_level", 0) or 0) * point, spread)
        stop = (round(bid - floor, digits) if position["direction"] > 0
                else round(ask + floor, digits))
        mt5.order_send({"action": mt5.TRADE_ACTION_SLTP,
                        "position": position["ticket"],
                        "symbol": position["symbol"], "sl": stop,
                        "tp": float(position.get("tp") or 0.0)})
        _CLOSE_RETRY[int(position["ticket"])] = (dict(position), "close retry")
        _log(f"close guarantee: tight SL {stop} on #{position['ticket']} "
             f"{position['symbol']}")
    except Exception:
        _CLOSE_RETRY[int(position["ticket"])] = (dict(position), "close retry")


def _tend_closes(config: VantageConfig) -> None:
    """Retry rejected closes until the venue confirms the position is gone."""
    if not _CLOSE_RETRY:
        return
    try:
        mt5 = _mt5()
        for ticket in list(_CLOSE_RETRY):
            if not mt5.positions_get(ticket=ticket):
                _CLOSE_RETRY.pop(ticket, None)
                _DEADLINES.pop(ticket, None)
                _log(f"close confirmed #{ticket}")
                continue
            position, reason = _CLOSE_RETRY[ticket]
            quote = current_quote(position["symbol"])
            if quote is None:
                continue
            closing_buy = position["direction"] < 0
            result = mt5.order_send({
                "action": mt5.TRADE_ACTION_DEAL, "position": ticket,
                "symbol": position["symbol"], "volume": float(position["lots"]),
                "type": mt5.ORDER_TYPE_BUY if closing_buy else mt5.ORDER_TYPE_SELL,
                "price": quote[1] if closing_buy else quote[0], "deviation": 30,
                "magic": 909090, "comment": "x1rty",
                "type_filling": _filling_for(position["symbol"])})
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                _CLOSE_RETRY.pop(ticket, None)
                _log(f"close retry succeeded #{ticket} ({reason})")
    except Exception:
        pass


# ------------------------------------------------- strategy 2: TAPE (15m)
#: The engine's own scored opens / closes per canonical symbol (ts, terms),
#: the score of each client's latest open (for its close), the loaded model
#: per symbol, the current bias, and the evaluation clocks.
_TAPE_EVENTS: dict[str, "deque"] = {}
_TAPE_SCORES: dict[tuple, float] = {}
_TAPE_MODELS: dict[str, tuple] = {}         # canonical -> (booster, meta, mtime, checked_at)
_TAPE_BIAS: dict[str, dict] = {}            # trading horizon
_TAPE_GATE_BIAS: dict[str, dict] = {}       # gate horizon (tape_gate_minutes), when distinct
_TAPE_LAST_EVAL: dict[str, float] = {}
_TAPE_LAST_LOG: dict[str, float] = {}
_TAPE_SYMBOLS: dict = {"stamp": 0.0, "symbols": []}
_TAPE_MAGIC = 909091                         # the strategy-2 slot at the venue
#: The buffer starts empty at every engine start: until the longest window
#: that matters (30 min) has filled, a bias is shown but never ACTIVE, so no
#: tape trade and no gate rests on half-empty rolling sums.
_TAPE_STARTED = [0.0]
_TAPE_WARM_MINUTES = 30

#: TRADING-DISABLED BACKOFF. When the venue answers 10027 / 10026 (AutoTrading
#: off at the terminal / server) or 10017 (trade disabled), every order will
#: fail the same way: pause NEW orders for 30 s and say so once a minute,
#: instead of one rejection line per signal (524 lines in 25 minutes on
#: 14 Sep 2026 while the terminal's Algo Trading button was off).
_TRADE_DISABLED = {"until": 0.0, "logged": 0.0, "code": 0}
_DISABLED_CODES = {10027: "AutoTrading is OFF at the MT5 terminal (Algo Trading button)",
                   10026: "AutoTrading disabled by the server",
                   10017: "trading disabled on the account"}


def _note_trade_disabled(retcode, comment="") -> None:
    try:
        code = int(retcode or 0)
    except (TypeError, ValueError):
        return
    if code not in _DISABLED_CODES:
        return
    now = time.time()
    _TRADE_DISABLED.update(until=now + 30.0, code=code)
    if now - _TRADE_DISABLED["logged"] >= 60:
        _TRADE_DISABLED["logged"] = now
        _log(f"TRADING DISABLED: {_DISABLED_CODES[code]} [{code} {comment}] -- new orders paused, "
             f"retrying every 30 s; enable Algo Trading in the terminal to resume")


def _trading_disabled() -> bool:
    return time.time() < _TRADE_DISABLED["until"]


#: Market-order filling mode per symbol, from the venue's own flags. Vantage
#: takes IOC; IG accepts FOK only (10030 'Unsupported filling mode' on IOC or
#: RETURN, 14 Sep 2026). Bit 1 = FOK, bit 2 = IOC; neither -> RETURN.
_FILLING: dict[str, int] = {}


def _filling_for(venue_symbol: str) -> int:
    mt5 = _mt5()
    cached = _FILLING.get(venue_symbol)
    if cached is not None:
        return cached
    mode = mt5.ORDER_FILLING_IOC
    try:
        info = mt5.symbol_info(venue_symbol)
        flags = int(getattr(info, "filling_mode", 0) or 0) if info else 0
        if flags & 2:
            mode = mt5.ORDER_FILLING_IOC
        elif flags & 1:
            mode = mt5.ORDER_FILLING_FOK
        else:
            mode = mt5.ORDER_FILLING_RETURN
    except Exception:
        pass
    _FILLING[venue_symbol] = mode
    return mode


def _tape_canonical(symbol) -> str:
    from webapp.trade_feed import _canonical
    return _canonical(str(symbol)) or str(symbol)


def _tape_horizon(config) -> int:
    from webapp import tape_model as tm
    try:
        return int(getattr(config, "tape_hold_minutes", tm.HORIZON) or tm.HORIZON)
    except (TypeError, ValueError):
        return tm.HORIZON


def _tape_available_symbols(config=None) -> list[str]:
    """Canonical symbols with a tape artifact on disk for the configured
    horizon (rescanned each minute, so a model trained later deploys itself)."""
    now = time.time()
    if now - _TAPE_SYMBOLS["stamp"] < 60 and _TAPE_SYMBOLS.get("horizon") == _tape_horizon(config):
        return _TAPE_SYMBOLS["symbols"]
    found = []
    try:
        from webapp.model_service import ARTIFACTS
        suffix = f"_{_tape_horizon(config)}m.json"
        for path in ARTIFACTS.glob(f"tape_*{suffix}"):
            name = path.name[len("tape_"):-len(suffix)]
            if name:
                found.append(name)
    except Exception:
        pass
    _TAPE_SYMBOLS.update(stamp=now, symbols=sorted(found), horizon=_tape_horizon(config))
    return _TAPE_SYMBOLS["symbols"]


def _tape_model(config: VantageConfig, canonical: str, horizon: int | None = None):
    """(booster, meta) for a symbol at a horizon (default: the trading one),
    reloaded when the artifact changes and refused when its OOS top-half
    accuracy is under the operator's floor."""
    from webapp import tape_model as tm
    from webapp.model_service import ARTIFACTS
    horizon = int(horizon or _tape_horizon(config))
    key = (canonical, horizon)
    now = time.time()
    entry = _TAPE_MODELS.get(key)
    if entry is not None and now - entry[3] < 60:
        return entry[0], entry[1]
    model_path, _ = tm.artifact_paths(ARTIFACTS, canonical, horizon)
    try:
        mtime = model_path.stat().st_mtime
    except OSError:
        _TAPE_MODELS[key] = (None, None, 0.0, now)
        return None, None
    if entry is not None and entry[2] == mtime:
        _TAPE_MODELS[key] = (entry[0], entry[1], mtime, now)
        return entry[0], entry[1]
    booster = meta = None
    try:
        booster, meta = tm.load(ARTIFACTS, canonical, horizon)
    except Exception as error:
        _log(f"tape {canonical} {horizon}m: model load failed: {type(error).__name__}: {error}")
    if meta is not None:
        top50 = float(((meta.get("oos") or {}).get("top50") or {}).get("acc") or 0.0)
        floor = float(getattr(config, "tape_min_top50_acc", 0.65))
        if not meta.get("ok") or top50 < floor:
            _log(f"tape {canonical} {horizon}m: model below the accuracy floor "
                 f"(top-50% {top50:.3f} < {floor:.2f}) -- not used")
            meta = {**meta, "refused": f"OOS top-half accuracy {top50:.1%} is under the {floor:.0%} floor"}
            booster = None
        else:
            _log(f"tape {canonical} {horizon}m: model loaded (trained {meta.get('trained_at')} UTC | OOS acc "
                 f"{meta['oos']['acc']:.3f}, top-50% {top50:.3f} | threshold {meta['threshold']:.3f})")
    _TAPE_MODELS[key] = (booster, meta, mtime, now)
    return booster, meta


def _tape_note_open(config: VantageConfig, event, value) -> None:
    """Every scored client open joins the tape RAW (ts, kind, direction, lots,
    score): the per-event terms are computed at evaluation time by the same
    tape_model functions training used."""
    try:
        from collections import deque
        canonical = _tape_canonical(event["symbol"])
        ts = pd.Timestamp(event["event_time"]).timestamp()
        score = float(value) if value is not None and np.isfinite(float(value)) else 0.5
        _TAPE_EVENTS.setdefault(canonical, deque()).append(
            (ts, 0, float(int(event["direction"])), float(event["lots"] or 0.0), score))
        _TAPE_SCORES[(str(event["account_key"]), canonical)] = score
        if len(_TAPE_SCORES) > 200_000:
            _TAPE_SCORES.clear()
    except Exception:
        pass


def _tape_note_close(config: VantageConfig, event) -> None:
    """A client close joins the tape at the open's score (kind 1)."""
    try:
        from collections import deque
        symbol = event.get("symbol"); direction = event.get("direction")
        if symbol is None or direction is None or pd.isna(direction):
            return
        canonical = _tape_canonical(symbol)
        ts = pd.Timestamp(event["event_time"]).timestamp()
        score = _TAPE_SCORES.get((str(event["account_key"]), canonical), 0.5)
        _TAPE_EVENTS.setdefault(canonical, deque()).append(
            (ts, 1, float(int(direction)), float(event.get("lots") or 0.0), score))
    except Exception:
        pass


def _tape_sums(config: VantageConfig, canonical: str, at_minute) -> dict:
    """Per-window EVENT_COLS sums for the decision minute T -- the SAME
    minute_sums / sums_at that built the training frame, on completed
    minutes [T-w, T-1] only (the forming minute is never counted)."""
    from webapp import tape_model as tm
    k = len(tm.EVENT_COLS)
    dq = _TAPE_EVENTS.get(canonical)
    if not dq:
        return {w: np.zeros((1, k)) for w in tm.WINDOWS}
    now_ts = time.time()
    keep = (max(float(getattr(config, "tape_stale_minutes", 120)), float(tm.WINDOWS[-1])) + 5.0) * 60.0
    while dq and now_ts - dq[0][0] > keep:
        dq.popleft()
    rows = np.array(list(dq), dtype=float)                   # ts, kind, direction, lots, score
    minutes = (rows[:, 0] * 1000).astype("datetime64[ms]").astype("datetime64[m]")
    is_open = rows[:, 1] == 0
    opens = tm.event_terms(rows[is_open, 2], rows[is_open, 3], rows[is_open, 4])
    closes = tm.close_terms(rows[~is_open, 2], rows[~is_open, 3], rows[~is_open, 4])
    n_o, n_c = int(is_open.sum()), int((~is_open).sum())
    terms = {c: np.concatenate([opens[c], np.zeros(n_c)]) for c in opens}
    terms.update({c: np.concatenate([np.zeros(n_o), closes[c]]) for c in closes})
    m, cs = tm.minute_sums(np.concatenate([minutes[is_open], minutes[~is_open]]), terms)
    at = np.array([np.datetime64(at_minute, "m")])
    return {w: tm.sums_at(m, cs, at, w) for w in tm.WINDOWS}


def _tape_price(venue_symbol: str, at_minute: datetime) -> dict:
    """Price context from the venue's last 60 COMPLETED M1 bars (position 1
    onward: the forming bar is excluded, as in training)."""
    from webapp import tape_model as tm
    fallback = {"hour": float(at_minute.hour), "weekday": float(at_minute.weekday())}
    try:
        mt5 = _mt5()
        rates = mt5.copy_rates_from_pos(venue_symbol, mt5.TIMEFRAME_M1, 1, 61)
        if rates is None or len(rates) == 0:
            return fallback
        return tm.price_context(np.asarray(rates["high"], dtype=float),
                                np.asarray(rates["low"], dtype=float), at_minute)
    except Exception:
        return fallback


def _tape_evaluate(config: VantageConfig, canonical: str) -> dict | None:
    """Score the tape now; keep the bias with its start / end stamps."""
    booster, meta = _tape_model(config, canonical)
    if booster is None or meta is None:
        return None
    from webapp import tape_model as tm
    venue = resolve_symbol(canonical) if config.mode == "live" else canonical
    if venue is None:
        return None
    now_utc = datetime.utcnow(); now_ts = time.time()
    # The decision minute T: features come from completed minutes and bars
    # up to T-1, exactly as the training frame was built.
    at_minute = now_utc.replace(second=0, microsecond=0)
    sums = _tape_sums(config, canonical, at_minute)
    vector = tm.features_from_sums(sums, _tape_price(venue, at_minute))
    try:
        p_up = float(booster.predict(vector)[0])
    except Exception as error:
        _log(f"tape {canonical}: predict failed: {type(error).__name__}: {error}")
        return None
    o_n = tm.EVENT_COLS.index("o_n")
    events_15m = int(sums[15][0, o_n]); events_120m = int(sums[120][0, o_n])
    confidence = abs(p_up - 0.5)
    threshold = float(meta.get("threshold") or 0.0)
    warm = float(getattr(config, "tape_warm_minutes", _TAPE_WARM_MINUTES) or 0.0)
    warming = (now_ts - _TAPE_STARTED[0]) < warm * 60.0
    # LIVE TAPE GUARD: enough opens in the window and a fresh newest event;
    # otherwise the vector is price context on zeros and the bias is void.
    dq = _TAPE_EVENTS.get(canonical)
    newest_age = (now_ts - dq[-1][0]) if dq else float("inf")
    tape_live = (events_15m >= int(getattr(config, "tape_min_events_15m", 50) or 0)
                 and newest_age <= float(getattr(config, "tape_feed_stale_seconds", 180.0) or 180.0))
    active = confidence >= threshold and not warming and tape_live
    lean = 1 if p_up > 0.5 else -1
    direction = lean if active else 0
    hold = int(getattr(config, "tape_hold_minutes", tm.HORIZON) or tm.HORIZON)
    stamp = now_utc.strftime("%Y-%m-%d %H:%M:%S")
    prev = _TAPE_BIAS.get(canonical) or {}
    unchanged = prev.get("active") == active and prev.get("direction") == direction
    since = prev.get("since") if unchanged and prev.get("since") else stamp
    bias = {"symbol": canonical, "venue_symbol": venue, "direction": direction, "lean": lean,
            "p_up": p_up, "confidence": confidence, "threshold": threshold, "active": active,
            "since": since, "updated": stamp,
            "until": (now_utc + timedelta(minutes=hold)).strftime("%Y-%m-%d %H:%M:%S") if active else None,
            "hold_minutes": hold, "events_15m": events_15m, "events_120m": events_120m,
            "model_trained_at": meta.get("trained_at"), "oos_acc": float(meta["oos"]["acc"]),
            "oos_top50_acc": float(meta["oos"]["top50"]["acc"]), "loaded": True,
            "warming": warming, "tape_live": tape_live,
            "newest_event_age_s": None if newest_age == float("inf") else round(newest_age, 1)}
    _TAPE_BIAS[canonical] = bias
    # GATE HORIZON: the same feature vector read by the shorter-horizon model
    # (a copy lives ~5 minutes) -- what strategy 1's gate looks at.
    gate_h = int(getattr(config, "tape_gate_minutes", 0) or 0)
    if gate_h and gate_h != hold:
        gate_booster, gate_meta = _tape_model(config, canonical, gate_h)
        if gate_booster is not None and gate_meta is not None:
            try:
                gp = float(gate_booster.predict(vector)[0])
                gc = abs(gp - 0.5); gt = float(gate_meta.get("threshold") or 0.0)
                gactive = gc >= gt and not warming and tape_live
                glean = 1 if gp > 0.5 else -1
                gdir = glean if gactive else 0
                gprev = _TAPE_GATE_BIAS.get(canonical) or {}
                gsame = gprev.get("active") == gactive and gprev.get("direction") == gdir
                _TAPE_GATE_BIAS[canonical] = {
                    "symbol": canonical, "horizon": gate_h, "hold_minutes": gate_h, "direction": gdir,
                    "lean": glean, "p_up": gp, "confidence": gc, "threshold": gt, "active": gactive,
                    "warming": warming, "tape_live": tape_live, "updated": stamp,
                    "since": gprev.get("since") if gsame and gprev.get("since") else stamp,
                    "until": (now_utc + timedelta(minutes=gate_h)).strftime("%Y-%m-%d %H:%M:%S") if gactive else None,
                    "model_trained_at": gate_meta.get("trained_at"),
                    "oos_acc": float(gate_meta["oos"]["acc"]),
                    "oos_top50_acc": float(gate_meta["oos"]["top50"]["acc"]), "loaded": True}
                if not gsame and now_ts - _TAPE_LAST_LOG.get(canonical + ":gate", 0.0) >= 20:
                    _TAPE_LAST_LOG[canonical + ":gate"] = now_ts
                    _log(f"TAPE GATE {canonical} {gate_h}m: {'UP' if glean > 0 else 'DOWN'} "
                         f"{'ACTIVE' if gactive else 'weak'} | p_up {gp:.2f} conf {gc:.2f} (>= {gt:.2f})")
            except Exception as error:
                _log(f"tape gate {canonical}: predict failed: {type(error).__name__}: {error}")
    if not unchanged and now_ts - _TAPE_LAST_LOG.get(canonical, 0.0) >= 20:
        _TAPE_LAST_LOG[canonical] = now_ts
        _log(f"TAPE {canonical}: {'UP' if lean > 0 else 'DOWN'} {'ACTIVE' if active else 'weak'} | "
             f"p_up {p_up:.2f} conf {confidence:.2f} (>= {threshold:.2f}) | "
             f"{bias['events_15m']} opens/15m | until {bias['until'] or '--'}")
    return bias


def _tape_comment(canonical: str, direction: int, bias: dict) -> str:
    """`2t{b|s}{conf}{p_up}{symbol}`: strategy digit first, like `2f...`."""
    return (f"2t{'b' if direction > 0 else 's'}{int(round(bias['confidence'] * 100)):02d}"
            f"{int(round(bias['p_up'] * 100)):02d}{canonical[:8]}")[:27]


def _tape_open(config: VantageConfig, canonical: str, venue: str, direction: int, bias: dict) -> None:
    quote = current_quote(venue)
    if quote is None:
        return
    bid, ask = quote
    price = ask if direction > 0 else bid
    equity = float(account_snapshot().get("equity") or 0.0)
    if equity <= 0:
        return
    intended = float(getattr(config, "tape_lots_per_5k", 1.0)) * (equity / 5000.0)
    if config.max_lots_per_trade and config.max_lots_per_trade > 0:
        intended = min(intended, float(config.max_lots_per_trade))
    lots = _quantize_lots(venue, round(max(0.01, intended), 2))
    # the tape's OWN margin envelope (tape_margin_share of equity at the
    # venue's real margin per lot), so a full mirror book cannot starve it
    # and it cannot starve the mirror book either
    cap = _envelope_cap(config, canonical and venue, direction, price, equity, ("2t",),
                        getattr(config, "tape_margin_share", 0.25), 1)
    if cap is not None and cap < lots:
        fitted = _quantize_lots(venue, round(cap, 2))
        if fitted < 0.01:
            _log(f"S2 TAPE {canonical}: tape margin envelope full -- not opened")
            return
        _log(f"S2 TAPE {canonical}: {lots} -> {fitted} lots (tape margin envelope)")
        lots = fitted
    book = open_book(config)

    def measure(p):
        return _notional(p["symbol"], p["lots"], p["open_price"])
    used = sum(measure(p) for p in book)
    wall = _wall_lev(config) * equity
    per_lot = _lot_economics(venue, price)[1]
    if per_lot > 0 and used + lots * per_lot > wall:
        lots = _quantize_lots(venue, round(max(0.0, wall - used) / per_lot, 2))
        if lots < 0.01:
            _log(f"S2 TAPE {canonical}: no leverage headroom for {intended:.2f} lots -- not opened")
            return
        _log(f"S2 TAPE {canonical}: trimmed to {lots} lots by the leverage wall")
    lots = _fit_margin(venue, direction, lots, price)
    if lots <= 0:
        _log(f"S2 TAPE {canonical}: no free margin at the venue -- not opened")
        return
    hold = int(bias.get("hold_minutes") or getattr(config, "tape_hold_minutes", 15))
    try:
        mt5 = _mt5()
        result = mt5.order_send({
            "action": mt5.TRADE_ACTION_DEAL, "symbol": venue, "volume": lots, "price": price,
            "deviation": 30, "magic": _TAPE_MAGIC,
            "type": mt5.ORDER_TYPE_BUY if direction > 0 else mt5.ORDER_TYPE_SELL,
            "comment": _tape_comment(canonical, direction, bias),
            "type_filling": _filling_for(venue)})
        ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        if ok:
            _mt5_cache_clear()
        detail = (f"strat=2|stance=tape|p_up={bias['p_up']:.3f}|conf={bias['confidence']:.3f}"
                  f"|thr={bias['threshold']:.3f}|since={bias['since']}|until={bias['until']}"
                  f"|lots={lots}|equity=${equity:,.0f}")
        if ok:
            ticket = int(result.order)
            _DEADLINES[ticket] = datetime.utcnow() + timedelta(minutes=hold)
            _record(source_account=f"tape:{canonical}", stance="tape", symbol=venue,
                    client_direction=direction, our_direction=direction, client_lots=0.0,
                    our_lots=lots, expected_usd=0.0, live_score=bias["p_up"], mode="live",
                    status="filled", ticket=ticket,
                    fill_price=float(getattr(result, "price", 0) or price), detail=detail)
            _log(f"S2 TAPE {'BUY' if direction > 0 else 'SELL'} {venue} {lots} @ {price} "
                 f"(p_up {bias['p_up']:.2f}, conf {bias['confidence']:.2f} >= {bias['threshold']:.2f}, "
                 f"until {bias['until']})")
        else:
            code = getattr(result, "retcode", "?"); note = getattr(result, "comment", "")
            _record(source_account=f"tape:{canonical}", stance="tape", symbol=venue,
                    client_direction=direction, our_direction=direction, client_lots=0.0,
                    our_lots=lots, expected_usd=0.0, live_score=bias["p_up"], mode="live",
                    status="rejected", detail=f"tape rejected {code} {note}|{detail}")
            _note_trade_disabled(code, note)
            _log(f"S2 TAPE REJECTED {'BUY' if direction > 0 else 'SELL'} {venue} {lots} [{code} {note}]")
    except Exception as error:
        _log(f"S2 TAPE error {venue}: {type(error).__name__}: {error}")


def _tape_manage(config: VantageConfig, canonical: str, bias: dict, entries_paused: bool) -> None:
    """One position per symbol: open on an active bias, flip when it turns
    (hedging account: close the old side, open the new), renew the hold
    while it stays active, let _tend_deadlines close it when it lapses."""
    venue = bias["venue_symbol"]
    hold = int(bias.get("hold_minutes") or 15)
    now = datetime.utcnow()
    mine = [p for p in open_book(config)
            if str(p.get("comment") or "").startswith("2t") and p.get("symbol") == venue]
    for p in mine:                                   # adopted after a restart: give it a clock
        _DEADLINES.setdefault(int(p["ticket"]), now + timedelta(minutes=hold))
    desired = int(bias.get("direction") or 0)
    if desired == 0:
        return
    for p in mine:
        if int(p.get("direction") or 0) != desired:
            if close_position(config, p, f"tape flip -> {'UP' if desired > 0 else 'DOWN'}"):
                _DEADLINES.pop(int(p["ticket"]), None)
    same = [p for p in mine if int(p.get("direction") or 0) == desired]
    if same:
        for p in same:
            _DEADLINES[int(p["ticket"])] = now + timedelta(minutes=hold)
        return
    if entries_paused or _trading_disabled():
        return
    with _LOCK:
        if _STATE.kill_switch:
            return
    _tape_open(config, canonical, venue, desired, bias)


def _tape_tick(config: VantageConfig, entries_paused: bool) -> None:
    """Every loop: re-score each symbol with a model (rate-limited) and, when
    strategy 2 is on, manage its tape position."""
    try:
        symbols = _tape_available_symbols(config)
        allowed = str(getattr(config, "tape_symbols", "*") or "*").strip()
        if allowed != "*":
            wanted = {s.strip().upper() for s in allowed.split(",") if s.strip()}
            symbols = [s for s in symbols if s.upper() in wanted]
        now = time.time()
        gap = float(getattr(config, "tape_eval_seconds", 2.0) or 2.0)
        s2_on = bool(getattr(config, "strategy2", False))
        # Strategy 2 switched off: nothing manages a tape position any more,
        # so any left on the book (adopted at start, or open when the switch
        # flipped) is closed rather than orphaned. Checked every 10 s.
        if not s2_on and config.mode == "live" and now - _TAPE_LAST_EVAL.get("__off__", 0.0) >= 10.0:
            _TAPE_LAST_EVAL["__off__"] = now
            for p in open_book(config):
                if str(p.get("comment") or "").startswith("2t"):
                    if close_position(config, p, "strategy 2 switched off"):
                        _DEADLINES.pop(int(p["ticket"]), None)
        for canonical in symbols:
            if now - _TAPE_LAST_EVAL.get(canonical, 0.0) < gap:
                continue
            _TAPE_LAST_EVAL[canonical] = now
            bias = _tape_evaluate(config, canonical)
            if bias is None or config.mode != "live" or not s2_on:
                continue
            _tape_manage(config, canonical, bias, entries_paused)
    except Exception as error:
        _log(f"tape tick error: {type(error).__name__}: {error}")


def tape_snapshot() -> list[dict]:
    """What the dashboard shows per symbol: the bias with its stamps, the
    model's provenance, and the tape position if any."""
    out: list[dict] = []
    try:
        config = load_config()
        book = open_book(config) if config.mode == "live" else []
    except Exception:
        config, book = None, []
    for canonical in _tape_available_symbols(config):
        bias = dict(_TAPE_BIAS.get(canonical) or {"symbol": canonical, "direction": 0, "lean": 0,
                                                   "active": False, "p_up": None, "confidence": None,
                                                   "threshold": None, "since": None, "until": None,
                                                   "updated": None, "loaded": False})
        entry = _TAPE_MODELS.get((canonical, _tape_horizon(config)))
        meta = entry[1] if entry else None
        if meta:
            bias.setdefault("model_trained_at", meta.get("trained_at"))
            bias.setdefault("oos_acc", float(meta["oos"]["acc"]))
            bias.setdefault("oos_top50_acc", float(meta["oos"]["top50"]["acc"]))
            bias.setdefault("threshold", float(meta.get("threshold") or 0.0))
            bias["refused"] = meta.get("refused")
            bias["replay"] = meta.get("replay")
            bias["parity"] = meta.get("parity")
            # the replay in account money at the size the engine would trade now
            try:
                venue = bias.get("venue_symbol") or (resolve_symbol(canonical) if config and config.mode == "live" else None)
                quote = current_quote(venue) if venue else None
                equity = float(account_snapshot().get("equity") or 0.0) if config and config.mode == "live" else 0.0
                if quote and equity > 0 and bias.get("replay"):
                    lots = float(getattr(config, "tape_lots_per_5k", 1.0)) * equity / 5000.0
                    per_bp = _lot_economics(venue, (quote[0] + quote[1]) / 2.0)[1] * lots / 1e4
                    rp = bias["replay"]
                    bias["replay_money"] = {"lots": round(lots, 2), "per_bp": round(per_bp, 2),
                                            "per_trade": round(rp["bps_per_trade_net"] * per_bp, 2),
                                            "per_day": round(rp["bps_per_trade_net"] * rp["per_day"] * per_bp, 2),
                                            "worst_run": round(rp["maxdd_bps"] * per_bp, 2)}
            except Exception:
                pass
        bias.setdefault("hold_minutes", _tape_horizon(config))
        warm = float(getattr(config, "tape_warm_minutes", _TAPE_WARM_MINUTES) or 0.0) if config else _TAPE_WARM_MINUTES
        bias.setdefault("warming", (time.time() - _TAPE_STARTED[0]) < warm * 60.0)
        gate_h = int(getattr(config, "tape_gate_minutes", 0) or 0) if config else 0
        bias["gate_minutes"] = gate_h or bias["hold_minutes"]
        gate_bias = _TAPE_GATE_BIAS.get(canonical)
        if gate_bias is None and gate_h and gate_h != bias["hold_minutes"]:
            gentry = _TAPE_MODELS.get((canonical, gate_h))
            gmeta = gentry[1] if gentry else None
            gate_bias = {"symbol": canonical, "horizon": gate_h, "hold_minutes": gate_h, "direction": 0,
                         "lean": 0, "active": False, "p_up": None, "confidence": None,
                         "threshold": float(gmeta.get("threshold") or 0.0) if gmeta else None,
                         "since": None, "until": None, "updated": None, "warming": bias["warming"],
                         "refused": (gmeta.get("refused") if gmeta else
                                     f"no {gate_h}-minute model for this symbol"),
                         "model_trained_at": gmeta.get("trained_at") if gmeta else None,
                         "oos_acc": float(gmeta["oos"]["acc"]) if gmeta else None,
                         "oos_top50_acc": float(gmeta["oos"]["top50"]["acc"]) if gmeta else None,
                         "loaded": bool(gentry and gentry[0] is not None)}
        bias["gate_bias"] = gate_bias
        venue = bias.get("venue_symbol")
        positions = [p for p in book if str(p.get("comment") or "").startswith("2t")
                     and (venue is None or p.get("symbol") == venue)]
        bias["positions"] = [{
            "ticket": p.get("ticket"), "direction": p.get("direction"), "lots": p.get("lots"),
            "open_price": p.get("open_price"), "profit": p.get("profit"),
            "until": (_DEADLINES[int(p["ticket"])].strftime("%Y-%m-%d %H:%M:%S")
                      if int(p.get("ticket") or 0) in _DEADLINES else None)} for p in positions]
        bias["gate"] = bool(getattr(config, "tape_bias_gate", True)) if config else True
        bias["trading"] = bool(getattr(config, "strategy2", False)) if config else False
        out.append(bias)
    return out


# ------------------------------------------------- manual-close replacement
#: Our positions as the venue showed them LAST loop (ticket -> position), so a
#: ticket that is gone this loop can be examined; reset at engine start so a
#: book flattened by hand across a restart is never re-entered.
_LAST_BOOK: dict[int, dict] = {}
#: Vanished tickets whose exit deal has not shown in history yet
#: (ticket -> (position, first seen)); examined once, then remembered.
_VANISHED: dict[int, tuple[dict, float]] = {}
_EXAMINED: set[int] = set()
_MANUAL_TICK = [0.0]
#: ENUM_DEAL_REASON: 0 desktop terminal, 1 mobile, 2 web = a person closed it.
#: 3 is our own API close, 4/5 a venue stop / take-profit, 6 a stop-out.
_MANUAL_DEAL_REASONS = {0, 1, 2}


def _stance_of(comment: str) -> str:
    text = comment or ""
    if text[:1] == "1":
        return "invert" if text[1:2] == "i" else "copy"
    if text.startswith("zx"):
        return "invert" if text[2:3] == "i" else "copy"
    return "copy"


def _position_deals(ticket: int) -> tuple[dict | None, dict | None]:
    """(entry deal, exit deal) of a position from venue history. The exit
    carries the reason and the summed P&L of every OUT deal (a partial close
    followed by the final one)."""
    mt5 = _mt5()
    entry_deal = exit_deal = None
    profit = 0.0
    for d in sorted(mt5.history_deals_get(position=int(ticket)) or [],
                    key=lambda d: (float(getattr(d, "time", 0) or 0), int(getattr(d, "ticket", 0) or 0))):
        raw_reason = getattr(d, "reason", None)
        rec = {"reason": int(raw_reason) if raw_reason is not None else -1,
               "comment": str(getattr(d, "comment", "") or ""),
               "time": float(getattr(d, "time", 0) or 0)}
        kind = int(getattr(d, "entry", 0) or 0)      # 0 in, 1 out, 2 in/out, 3 out-by
        if kind == 0 and entry_deal is None:
            entry_deal = rec
        elif kind in (1, 3):
            profit += (float(getattr(d, "profit", 0.0) or 0.0) + float(getattr(d, "swap", 0.0) or 0.0)
                       + float(getattr(d, "commission", 0.0) or 0.0))
            exit_deal = rec
    if exit_deal is not None:
        exit_deal["profit"] = profit
    return entry_deal, exit_deal


def _order_row_for(ticket: int) -> dict | None:
    """Our ledger row for the order that opened this position (MT5: the
    position id IS the opening order's ticket)."""
    try:
        _ensure_table()
        with sqlite3.connect(DB) as cx:
            cx.row_factory = sqlite3.Row
            row = cx.execute("SELECT stance, client_lots, expected_usd, live_score "
                             "FROM vantage_orders WHERE ticket = ? ORDER BY id DESC LIMIT 1",
                             (int(ticket),)).fetchone()
            return dict(row) if row else None
    except Exception:
        return None


def _tend_manual_closes(config: VantageConfig, k: float) -> None:
    """Replace a trade closed BY HAND in profit while its client is still in.

    The book is re-read from the venue every loop, so a ticket that was
    there last loop and is gone now was closed by something; its exit deal
    says by what (see _MANUAL_DEAL_REASONS). For a manual close in profit
    whose client has not closed, rest a limit of the SAME lots at the
    ORIGINAL entry -- below a market that ran up after a long, above one
    that fell after a short -- tagged to the same client, so the existing
    machinery applies unchanged: cancelled when the client exits, expired by
    the entry-limit TTL, mirrored on fill. Engine closes, venue stops and
    take-profits, losing manual closes and clients already out are all left
    alone. Each ticket is examined once.
    """
    global _LAST_BOOK
    if config.mode != "live" or not getattr(config, "replace_manual_closes", False):
        return
    # every couple of seconds is plenty to notice a vanished ticket
    if time.time() - _MANUAL_TICK[0] < 2.0:
        return
    _MANUAL_TICK[0] = time.time()
    try:
        current = {int(p["ticket"]): p for p in live_positions()}
    except Exception:
        return
    previous, _LAST_BOOK = _LAST_BOOK, current
    now = time.time()
    for ticket, position in previous.items():
        if ticket not in current and ticket not in _EXAMINED and ticket not in _VANISHED:
            _VANISHED[ticket] = (position, now)
    if not _VANISHED:
        return
    for ticket in list(_VANISHED):
        position, first_seen = _VANISHED[ticket]
        source = position.get("source") or ""
        symbol = position["symbol"]
        if not source or ticket in _CLOSE_RETRY:          # not ours to replace / engine close in flight
            _VANISHED.pop(ticket, None); _EXAMINED.add(ticket)
            continue
        try:
            entry_deal, exit_deal = _position_deals(ticket)
        except Exception:
            entry_deal = exit_deal = None
        if exit_deal is None:
            if now - first_seen > 120:                      # history never showed an exit
                _VANISHED.pop(ticket, None); _EXAMINED.add(ticket)
            continue
        _VANISHED.pop(ticket, None); _EXAMINED.add(ticket)
        reason = exit_deal["reason"]
        manual = (reason in _MANUAL_DEAL_REASONS if reason >= 0
                  else not exit_deal["comment"].startswith(("x", "[")))
        if not manual:
            continue
        profit = exit_deal["profit"]
        if profit <= 0:
            _log(f"manual close #{ticket} {symbol} {profit:+.2f}: not in profit, not replaced")
            continue
        open_utc = (datetime.utcfromtimestamp(entry_deal["time"] - _server_offset_seconds())
                    if entry_deal and entry_deal["time"] else datetime.utcnow() - timedelta(hours=4))
        try:
            if _client_has_closed(source, symbol, open_utc):
                _log(f"manual close #{ticket} {symbol} {profit:+.2f}: client already out, not replaced")
                continue
        except Exception:
            pass
        direction = int(position.get("direction") or 0)
        with _LOCK:
            duplicate = any(o.get("source") == source and o.get("symbol") == symbol
                            and int(o.get("direction") or 0) == direction
                            for o in _PENDING.values())
        if duplicate:
            _log(f"manual close #{ticket} {symbol}: a limit for this client already rests, not replaced")
            continue
        quote = current_quote(symbol)
        if quote is None:
            continue
        bid, ask = quote
        entry_price = float(position.get("open_price") or 0.0)
        if entry_price <= 0 or direction == 0:
            continue
        if (direction > 0 and ask <= entry_price) or (direction < 0 and bid >= entry_price):
            _log(f"manual close #{ticket} {symbol}: price already back through the entry "
                 f"{entry_price}, a limit there is not placeable -- not replaced")
            continue
        row = _order_row_for(ticket) or {}
        stance = row.get("stance") or _stance_of(position.get("comment") or "")
        result = _place_rescue_limit(
            config, source, stance, symbol, direction,
            float(row.get("client_lots") or 0.0), entry_price, k,
            float(row.get("expected_usd") or 0.0), row.get("live_score"), bid, ask,
            limit_override=entry_price, ttl_minutes=config.entry_limit_ttl_minutes,
            tag="re-entry after manual close", lots_override=float(position.get("lots") or 0.0))
        _log(f"manual close #{ticket} {symbol} {position.get('lots')} {profit:+.2f} in profit, "
             f"client still in -> " + ("re-entry limit resting @ " + str(entry_price)
                                       if result.get("pending") else
                                       f"re-entry not placed ({result.get('reason', 'rejected')})"))


# ------------------------------------------------- orphan sweeper (5-minute)
#: Tickets already given protective brackets, so the sweep never re-spams SLTP.
_SWEPT: set[int] = set()
_LAST_SWEEP = [0.0]
SWEEP_SECONDS = 300.0
#: Stream watchdog clocks: [last check, last restart]. Quote/deal consumers
#: have died quietly before; the loop notices staleness and bounces them.
_WATCHDOG = [0.0, 0.0]


def _server_offset_seconds() -> float:
    """Venue clock minus UTC (e.g. +3h), half-hour rounded, from the FRESHEST
    tick across symbols that trade around the clock. A single gold tick went
    stale off-session and skewed the offset a rounding step -- which let the
    pre-reset liquidation leak into a 'fresh' perf window."""
    best = 0.0
    try:
        mt5 = _mt5()
        for sym in ("BTCUSD", "ETHUSD", "XAUUSD", "EURUSD"):
            tick = mt5.symbol_info_tick(sym)
            if tick and tick.time:
                best = max(best, float(tick.time))
    except Exception:
        pass
    if best:
        return round((best - time.time()) / 1800.0) * 1800.0
    return 3 * 3600.0


def _atr(symbol: str, periods: int = 14) -> float | None:
    """H1 ATR(14) from the venue's own bars -- the stale-position stop unit."""
    try:
        mt5 = _mt5()
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, periods + 1)
        if rates is None or len(rates) < periods:
            return None
        highs = np.array([r[2] for r in rates], dtype=float)
        lows = np.array([r[3] for r in rates], dtype=float)
        closes = np.array([r[4] for r in rates], dtype=float)
        tr = np.maximum(highs[1:] - lows[1:],
                        np.maximum(np.abs(highs[1:] - closes[:-1]),
                                   np.abs(lows[1:] - closes[:-1])))
        value = float(tr.mean())
        return value if value > 0 else None
    except Exception:
        return None


# ------------------------------------------- per-symbol side-score memory
#: (canonical, our_direction) -> deque[(epoch, edge_value)] over every
#: scored copy/invert intent -- the flow's own opinion of each side.
_SIDE_SCORES: dict = {}


def _record_side_score(canon: str, direction: int, edge: float) -> None:
    from collections import deque
    key = (str(canon), int(direction))
    dq = _SIDE_SCORES.get(key)
    if dq is None:
        dq = _SIDE_SCORES[key] = deque(maxlen=4000)
    dq.append((time.time(), float(edge)))


def _side_avg(canon: str, direction: int, window_s: float = 7200.0) -> float:
    dq = _SIDE_SCORES.get((str(canon), int(direction)))
    if not dq:
        return 0.0
    cutoff = time.time() - window_s
    values = [e for ts, e in dq if ts >= cutoff]
    return (sum(values) / len(values)) if values else 0.0


# --------------------------------------------- stop-out HOLD registry
#: A client STOPPED OUT or LIQUIDATED (known from the close row's comment /
#: reason on the source server) was forced out at a bad price -- their exit
#: carries no alpha. We DO NOT mirror it: the ticket is HELD -- exempt from
#: mirror exits, bracketed (never insta-closed) by the orphan sweep, and
#: retired the moment a newer same-symbol same-direction valid signal with a
#: WORSE entry arrives (we already own the better entry, so the new signal's
#: slot budget pays for realising ours).

def _ensure_held() -> None:
    import sqlite3
    with sqlite3.connect(ROOT / "app.db") as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS vantage_held ("
                   "ticket INTEGER PRIMARY KEY, symbol TEXT, "
                   "direction INTEGER, entry REAL, source TEXT, "
                   "held_at REAL, reason TEXT)")


def _hold_ticket(position: dict, reason: str) -> None:
    import sqlite3
    _ensure_held()
    with sqlite3.connect(ROOT / "app.db") as cx:
        cx.execute("INSERT OR REPLACE INTO vantage_held VALUES (?,?,?,?,?,?,?)",
                   (int(position.get("ticket") or 0),
                    str(position.get("symbol") or ""),
                    int(position.get("direction") or 0),
                    float(position.get("open_price") or 0.0),
                    str(position.get("source") or ""),
                    time.time(), reason))
    _log(f"HOLD #{position.get('ticket')} {position.get('symbol')}: {reason} "
         f"— not mirroring the client's forced exit")


def _held_map() -> dict:
    import sqlite3
    _ensure_held()
    try:
        with sqlite3.connect(ROOT / "app.db") as cx:
            cx.row_factory = sqlite3.Row
            return {int(r["ticket"]): dict(r) for r in cx.execute(
                "SELECT * FROM vantage_held")}
    except Exception:
        return {}


def _release_held(ticket: int) -> None:
    import sqlite3
    _ensure_held()
    with sqlite3.connect(ROOT / "app.db") as cx:
        cx.execute("DELETE FROM vantage_held WHERE ticket = ?", (int(ticket),))


_STOPOUT_CACHE: dict = {}


def _close_was_stopout(account: str, source_symbol: str, event_time) -> bool:
    """Did THIS close happen because the client was stopped out (SL) or the
    account liquidated (SO)? Read from the source server's own close row:
    MT5 deals carry reason (4=SL, 6=stop-out) and the sl/so comment; MT4
    appends [sl] / so: to the order comment. A generous window absorbs the
    server-clock offset; a liquidation closes everything at once, so any
    stop-out-marked row in it is signal. Any error -> False (normal
    mirror exit -- degrade safe)."""
    key = (account, source_symbol, str(event_time))
    if key in _STOPOUT_CACHE:
        return _STOPOUT_CACHE[key]
    out = False
    try:
        from webapp import trade_feed as tfeed
        server, _, login_s = str(account).rpartition(":")
        login = int(login_s)
        ts = pd.Timestamp(event_time)
        connection = tfeed._connection(server)
        if server.startswith("mt4"):
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT comment FROM orders WHERE login = %s "
                    "AND symbol_name = %s AND close_ts BETWEEN %s AND %s "
                    "ORDER BY close_ts DESC LIMIT 6",
                    [login, source_symbol,
                     int(ts.timestamp()) - 600, int(ts.timestamp()) + 14400])
                for (comment,) in cursor.fetchall():
                    text = str(comment or "").lower()
                    if ("[sl]" in text or "[so]" in text or "so:" in text
                            or "stop out" in text or "stopout" in text):
                        out = True
                        break
        else:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT comment, reason FROM deals WHERE login = %s "
                    "AND symbol = %s AND entry IN (1, 3) "
                    "AND `time` BETWEEN %s AND %s "
                    "ORDER BY `time` DESC LIMIT 6",
                    [login, source_symbol,
                     (ts - pd.Timedelta(minutes=10)).to_pydatetime(),
                     (ts + pd.Timedelta(hours=4)).to_pydatetime()])
                for (comment, reason) in cursor.fetchall():
                    text = str(comment or "").lower()
                    if reason is not None and int(reason) in (4, 6):
                        out = True
                        break
                    if (text.startswith("sl ") or "[sl]" in text
                            or text.startswith("so ") or "so:" in text
                            or "stop out" in text or "stopout" in text):
                        out = True
                        break
    except Exception:
        out = False
    _STOPOUT_CACHE[key] = out
    if len(_STOPOUT_CACHE) > 4000:
        _STOPOUT_CACHE.clear()
    return out


def _client_has_closed(source: str, venue_symbol: str, open_utc: datetime) -> bool:
    """Did the source client post a CLOSE on this symbol after our open? A
    missed mirror exit leaves exactly this signature -- their close exists,
    our position still stands.

    AUTHORITATIVE witness: the source server's own MySQL (via
    _recent_closed_for, MT4+MT5 aware) -- the kafka store covers only a
    subset of servers and would leave most orphans invisible. The sqlite
    close-recorder serves as the cheap fallback."""
    from webapp.trade_feed import _canonical
    canon = _canonical(venue_symbol) or venue_symbol
    if ":" in str(source):
        try:
            # Query from BEFORE our open: the MT5 branch pairs entry+exit
            # deals inside its window, and the client's entry precedes ours.
            closed = _recent_closed_for(
                str(source), pd.Timestamp(open_utc) - pd.Timedelta(hours=2))
            if closed is not None and len(closed):
                cutoff = pd.Timestamp(open_utc) - pd.Timedelta(seconds=60)
                hits = closed.loc[
                    pd.to_datetime(closed["close_time"]) >= cutoff, "symbol"]
                for sym in hits.astype(str):
                    if (_canonical(sym) or sym) == canon:
                        return True
                return False
        except Exception:
            pass
    digits = "".join(ch for ch in str(source).rpartition(":")[2] if ch.isdigit())
    if not digits:
        return False
    try:
        _ensure_table()
        with sqlite3.connect(DB) as cx:
            row = cx.execute(
                "SELECT COUNT(*) FROM vantage_client_closes "
                "WHERE canon = ? AND event_utc > ? "
                "AND CAST(login AS TEXT) LIKE ?",
                (canon, pd.Timestamp(open_utc).timestamp(),
                 "%" + digits[-9:])).fetchone()
        return bool(row and row[0])
    except Exception:
        return False


def _sweep_orphans(config: VantageConfig) -> None:
    """Every 5 minutes: find engine positions whose client already closed (a
    missed mirror exit -- restart gaps, feed drops) and stop them drifting.
    Profitable orphans close immediately; losing ones get TP at flat P&L and
    SL one H1-ATR away, so the venue itself bounds the damage. The overnight
    JPY bleed (-$167 of stale 2h+ holds liquidated mid-drawdown) is exactly
    the drift this prevents."""
    if config.mode != "live":
        return
    try:
        mt5 = _mt5()
        offset = _server_offset_seconds()
        raw = {int(p.ticket): p for p in (mt5.positions_get() or [])
               if getattr(p, "magic", 0) == 909090}
        if not raw:
            return
        min_age = 600.0                       # never race a normal mirror exit
        swept = closed = bracketed = 0
        held = _held_map()
        for position in open_book(config):
            ticket = int(position.get("ticket") or 0)
            p = raw.get(ticket)
            source = position.get("source")
            if p is None or not source:
                continue
            open_utc = datetime.utcfromtimestamp(float(p.time) - offset)
            if (datetime.utcnow() - open_utc).total_seconds() < min_age:
                continue
            if ticket in _SWEPT:
                continue
            is_held = ticket in held
            if not is_held and not _client_has_closed(
                    source, position["symbol"], open_utc):
                continue
            swept += 1
            profit = float(position.get("profit") or 0.0)
            # HELD (stop-out) tickets are deliberately kept open: they get
            # the protective brackets as their close rule, NEVER the
            # profitable insta-close -- their exit is the worse-entry
            # replacement or the bracket, whichever comes first.
            if profit >= 0 and not is_held:
                if close_position(config, position,
                                  "orphan sweep: client closed, in profit"):
                    closed += 1
                    _SWEPT.add(ticket)
                continue
            # Losing orphan: breakeven TP, one-ATR stop -- broker-side, so the
            # exit survives our own process dying.
            info = mt5.symbol_info(position["symbol"])
            digits_n = info.digits if info else 5
            atr = _atr(position["symbol"])
            if atr is None:
                atr = abs(float(position["open_price"])) * 0.003  # ~30bps fallback
            entry = float(position["open_price"])
            direction = int(position.get("direction") or (1 if p.type == 0 else -1))
            tp = round(entry, digits_n)
            sl = round(entry - direction * atr, digits_n)
            result = mt5.order_send({"action": mt5.TRADE_ACTION_SLTP,
                                     "position": ticket,
                                     "symbol": position["symbol"],
                                     "sl": sl, "tp": tp})
            ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
            if ok:
                bracketed += 1
                _SWEPT.add(ticket)
                _log(f"orphan sweep: #{ticket} {position['symbol']} losing "
                     f"{profit:+.2f} -> TP flat @ {tp}, SL 1 ATR @ {sl}")
            else:
                _log(f"orphan sweep: SLTP failed #{ticket} "
                     f"{getattr(result, 'retcode', '?')}")
        # Resting engine limits whose client has since closed die too.
        with _LOCK:
            pending = list(_PENDING.items())
        for ticket, order in pending:
            src = order.get("source")
            if not src:
                continue
            placed = order.get("expires")
            anchor = (placed - timedelta(minutes=config.entry_limit_ttl_minutes)
                      if placed else datetime.utcnow() - timedelta(hours=4))
            if _client_has_closed(src, order["symbol"], anchor):
                _cancel_pending_for(config, src, order["symbol"])
                _log(f"orphan sweep: pending #{ticket} cancelled (client closed)")
        # Visible every run, so a silent failure is distinguishable from
        # a genuinely clean book.
        _log(f"orphan sweep: checked {len(raw)} positions | {swept} orphans | "
             f"{closed} closed in profit | {bracketed} bracketed "
             f"(flat TP / 1-ATR SL)")
        try:
            with sqlite3.connect(DB) as cx:      # close-recorder retention
                cx.execute("DELETE FROM vantage_client_closes "
                           "WHERE event_utc < ?", (time.time() - 31 * 86400,))
        except Exception:
            pass
    except Exception as error:
        _log(f"orphan sweep error: {type(error).__name__}: {error}")


def _s2_comment(account: str, s2_score: float) -> str:
    """`2f{server}{login}s{score}h15`: strategy digit first for sorting."""
    server, _, login = str(account).rpartition(":")
    code = _SERVER_CODES.get(server, "?")
    return f"2f{code}{login}s{int(round(s2_score * 100)):02d}h15"[:27]


def _execute_strategy2(config: VantageConfig, event, s2_score,
                       s2_magnitude=None) -> None:
    """Fade-15: when the market-only model says the client's direction loses
    the next 15 minutes, take the OPPOSITE side for exactly that long.
    Symbol classes are gated by MEASURED viability (fail-closed); sizing is
    magnitude-proportional (S1's winning policy at S2's horizon). Shares the
    leverage wall and symbol-share limits with the mirror book; comment `2f`
    keeps mirror exits, netting and recycling away from it."""
    if (s2_score is None or s2_score > config.strategy2_fade_ceiling
            or config.mode != "live"):
        return
    from webapp.trade_features import symbol_class
    if symbol_class(str(event["symbol"])) not in _strategy2_classes():
        return
    with _LOCK:
        if _STATE.kill_switch:
            return
    venue_symbol = resolve_symbol(str(event["symbol"]))
    if venue_symbol is None:
        return
    quote = current_quote(venue_symbol)
    if quote is None:
        return
    direction = -int(event["direction"])
    bid, ask = quote
    price = ask if direction > 0 else bid
    equity = float(account_snapshot().get("equity") or 0.0) or 5000.0
    # Magnitude-proportional (predicted |15m move|/cost), equity-scaled;
    # equal-size fallback when the twin is unavailable.
    if s2_magnitude is not None and s2_magnitude > 0:
        intended = config.strategy2_lots * s2_magnitude * (equity / 5000.0)
    else:
        intended = config.strategy2_lots * (equity / 5000.0)
    lots = _quantize_lots(venue_symbol, round(max(0.01, intended), 2))
    if lots > max(intended, 1e-9) * 2.0:
        return
    book = open_book(config)
    s2_open = sum(1 for p in book
                  if str(p.get("comment") or "").startswith(("2f", "z2f")))
    if s2_open >= config.strategy2_max_positions:
        return

    def measure(p):
        return _notional(p["symbol"], p["lots"], p["open_price"])
    used = sum(measure(p) for p in book)
    needed = _notional(venue_symbol, lots, price)
    wall = _wall_lev(config) * equity
    symbol_used = sum(measure(p) for p in book if p["symbol"] == venue_symbol)
    if used + needed > wall or \
            symbol_used + needed > config.max_symbol_share * wall:
        return
    try:
        mt5 = _mt5()
        result = mt5.order_send({
            "action": mt5.TRADE_ACTION_DEAL, "symbol": venue_symbol,
            "volume": lots, "price": price, "deviation": 20, "magic": 909091,
            "type": (mt5.ORDER_TYPE_BUY if direction > 0
                     else mt5.ORDER_TYPE_SELL),
            "comment": _s2_comment(str(event["account_key"]), s2_score),
            "type_filling": _filling_for(venue_symbol)})
        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
            _DEADLINES[int(result.order)] = datetime.utcnow() + timedelta(
                seconds=config.strategy2_hold_seconds)
            _record(source_account="strategy2", stance="fade15",
                    symbol=venue_symbol,
                    client_direction=int(event["direction"]),
                    our_direction=direction,
                    client_lots=float(event["lots"] or 0), our_lots=lots,
                    expected_usd=0.0, live_score=s2_score, mode="live",
                    status="filled", ticket=int(result.order),
                    fill_price=float(getattr(result, "price", 0) or price),
                    detail=(f"strat=2|stance=fade15"
                            f"|src={event['account_key']}"
                            f"|s2={s2_score:.3f}"
                            f"|s2mag={f'{s2_magnitude:.1f}' if s2_magnitude is not None else 'na'}"
                            f"|hold={config.strategy2_hold_seconds:.0f}s"
                            f"|lots={lots}|equity=${equity:,.0f}"))
            _log(f"S2 FADE {'BUY' if direction > 0 else 'SELL'} {venue_symbol} "
                 f"{lots} @ {price} (s2 {s2_score:.2f}, closes in "
                 f"{config.strategy2_hold_seconds / 60:.0f}m)")
    except Exception:
        pass


def _client_lots_for(ticket) -> float | None:
    """The client lots that backed a live ticket, from the orders ledger --
    the in-memory book loses this across restarts, sqlite does not."""
    try:
        with sqlite3.connect(DB) as connection:
            row = connection.execute(
                "SELECT client_lots FROM vantage_orders WHERE ticket = ? "
                "AND status IN ('filled', 'recycled') ORDER BY id DESC LIMIT 1",
                (int(ticket),)).fetchone()
        return float(row[0]) if row and row[0] else None
    except Exception:
        return None


def mirror_exits(config: VantageConfig, closings: pd.DataFrame) -> int:
    """Follow the client out -- the evidence-backed exit policy."""
    if closings.empty:
        return 0
    book = open_book(config)
    if not book:
        return 0
    closed = 0
    # Dedupe on the CLOSING EVENT's full identity, never on (account, symbol):
    # on a hedging book three exits from one client on one symbol are three
    # distinct events that must close three of our tickets. Only the feed's
    # literal repeats collapse.
    events = closings.drop_duplicates(
        ["account_key", "symbol", "event_time", "profit", "volume"])
    held = _held_map()
    for _, event in events.iterrows():
        account, source_symbol = str(event["account_key"]), str(event["symbol"])
        venue_symbol = (resolve_symbol(source_symbol) if config.mode == "live"
                        else source_symbol) or source_symbol
        # A resting rescue limit dies with the client's exit.
        _cancel_pending_for(config, account, venue_symbol)
        tail = account[-8:]
        # NETTING REVERSAL (ENTRY_INOUT): one deal closed the client's old
        # side and flipped. Only OUR copies of the OLD side may close here --
        # the flip's own new copy (same source+symbol, other client side)
        # must not be devoured by its own opening deal.
        is_inout = str(event.get("entry") or "") == "ENTRY_INOUT"
        event_dir = int(event.get("direction") or 0)
        for position in list(book):
            matches = (position.get("source") == account
                       or position.get("source") == tail)
            if not matches or position["symbol"] != venue_symbol:
                continue
            if is_inout and event_dir:
                our_dir = int(position.get("direction") or 0)
                client_dir = (our_dir if position.get("stance") == "copy"
                              else -our_dir)
                if client_dir != -event_dir:
                    continue
            # HELD tickets no longer follow this client's closes at all.
            if int(position.get("ticket") or 0) in held:
                continue
            # STOP-OUT / LIQUIDATION (from the source server's close row):
            # a forced exit carries no alpha -- HOLD our position instead of
            # mirroring the close. The event is still consumed. (A reversal
            # is the client's own decision, never a stop-out -- skip probing.)
            if not is_inout and _close_was_stopout(account, source_symbol,
                                                   event.get("event_time")):
                _hold_ticket(position, f"client {account} stopped out / "
                                       f"liquidated")
                held[int(position.get("ticket") or 0)] = {
                    "entry": position.get("open_price")}
                break
            client_pnl = float(event.get("profit") or 0)
            if position["ticket"] in _PAPER and position.get("client_lots"):
                # Equal entry and equal exit means our P&L IS the client's,
                # scaled by the lot ratio and sign-flipped when inverted.
                ratio = position["lots"] / position["client_lots"]
                pnl = (1.0 if position["stance"] == "copy" else -1.0) * client_pnl * ratio
                with _LOCK:
                    _PAPER.pop(position["ticket"], None)
                _PAPER_PNL[0] += pnl
                _log(f"paper CLOSE #{position['ticket']} {venue_symbol} mirror of "
                     f"{account} (client {client_pnl:+,.2f}) ours {pnl:+,.2f}")
                book.remove(position)
            else:
                # PARTIAL CLOSES: a client scaling out of a third of their
                # line must scale US out of a third, not flatten the whole
                # ticket -- MT5 partial exits and MT4 partial-close chains
                # both arrive as a closing event smaller than the lots that
                # backed the position. The venue closes the requested volume
                # and keeps the remainder under the same ticket.
                client_lots = (position.get("client_lots")
                               or _client_lots_for(position["ticket"]))
                event_lots = float(event.get("lots") or 0)
                fraction = (event_lots / client_lots
                            if client_lots and event_lots > 0 else 1.0)
                our_lots = float(position["lots"])
                partial_lots = round(our_lots * min(fraction, 1.0), 2)
                if config.mode == "live":
                    partial_lots = min(our_lots, _quantize_lots(
                        position["symbol"], partial_lots))
                if fraction < 0.9 and 0.01 <= partial_lots < our_lots:
                    slice_of = dict(position)
                    slice_of["lots"] = partial_lots
                    close_position(config, slice_of,
                                   f"partial mirror of {account} ({fraction:.0%})")
                    position["lots"] = round(our_lots - partial_lots, 2)
                    if position["lots"] < 0.01:
                        book.remove(position)
                else:
                    close_position(config, position, f"mirror of {account} exit")
                    book.remove(position)
            closed += 1
            # ONE close per closing event. This is a HEDGING account: three
            # copies of a client who holds three XAUUSD positions are three
            # separate tickets, and the client closing ONE of them must close
            # ONE of ours -- closing all three (the old behaviour) collapsed
            # the whole line on the first partial exit.
            break
    return closed


# _trim_to_safe_margin REMOVED: the forced margin-defense liquidation ("x1brk")
# closed the worst-losing positions at the bottom and was ~half the live loss in
# the replay. Positions now run to their mirror exit; margin is handled by the
# entry-pause above and, in extremis, the broker's own stop-out.


def _reconcile(config: VantageConfig, intents: list[dict],
               closings: pd.DataFrame) -> pd.DataFrame:
    """Transaction-cost netting across the batch.

    A client exit whose position a fresh SAME-direction signal can adopt is
    neither closed nor reopened: the ticket changes owner and two spread
    crossings are saved. Both signals are still obeyed exactly -- the exiting
    client's claim on the ticket ends, the entering client's begins, and the
    position's P&L from this moment mirrors the new signal. Economically the
    adoption is a market entry with ZERO spread paid, which dominates the
    close+reopen it replaces. Sizes must be comparable (0.5x-2.0x) or the
    normal close-then-open path runs instead.
    """
    if closings is None or not len(closings) or not intents:
        return closings
    from webapp.trade_feed import _canonical
    book = open_book(config)
    events = closings.drop_duplicates(
        ["account_key", "symbol", "event_time", "profit", "volume"])
    keep = []
    for _, event in events.iterrows():
        account, source_symbol = str(event["account_key"]), str(event["symbol"])
        venue_symbol = (resolve_symbol(source_symbol) if config.mode == "live"
                        else source_symbol) or source_symbol
        tail = account[-8:]
        position = next((p for p in book
                         if p.get("source") in (account, tail)
                         and p["symbol"] == venue_symbol), None)
        matched = None
        if position is not None:
            for intent in intents:
                if intent.get("consumed") or intent["direction"] != position["direction"]:
                    continue
                intent_venue = (resolve_symbol(intent["symbol"])
                                if config.mode == "live"
                                else intent["symbol"]) or intent["symbol"]
                if intent_venue != venue_symbol:
                    continue
                multiplier = symbol_multipliers(config).get(
                    _canonical(intent["symbol"]), 1.0)
                if multiplier <= 0:
                    continue
                est_lots = round(max(0.01, min(
                    intent["client_lots"] * intent["k"] * multiplier,
                    config.max_lots_per_trade)), 2)
                ratio = est_lots / max(float(position["lots"]), 1e-9)
                if 0.5 <= ratio <= 2.0:
                    matched = intent
                    break
        if matched is None:
            keep.append(event)
            continue
        matched["consumed"] = True
        book.remove(position)
        _save_reassignment(int(position["ticket"]), matched["account"])
        # A recycled ticket serves its NEW client's signal: any stop-out
        # hold from the previous owner is released, or the ticket would be
        # exempt from its new owner's mirror exits forever.
        try:
            _release_held(int(position["ticket"]))
        except Exception:
            pass
        if position["ticket"] in _PAPER:
            with _LOCK:
                _PAPER[position["ticket"]]["source"] = matched["account"]
                _PAPER[position["ticket"]]["stance"] = matched["stance"]
        # The exiting client's resting limits die with their signal.
        _cancel_pending_for(config, account, venue_symbol)
        _record(source_account=matched["account"], stance=matched["stance"],
                symbol=venue_symbol, client_direction=matched["client_direction"],
                our_direction=position["direction"],
                client_lots=matched["client_lots"], our_lots=position["lots"],
                expected_usd=matched["expected"], live_score=matched["value"],
                mode=config.mode, status="recycled", ticket=position["ticket"],
                detail=f"adopted from {account}; close+reopen saved")
        _log(f"RECYCLED #{position['ticket']} {venue_symbol}: {account} out, "
             f"{matched['account']} in (two spread crossings saved)")
    if not keep:
        return closings.iloc[0:0]
    return pd.DataFrame(keep)


# ---------------------------------------------------------------- engine
def _identity(event) -> tuple:
    return (str(event["account_key"]), str(event["symbol"]), int(event["direction"]),
            round(float(event["lots"] or 0), 2), round(float(event["price"] or 0), 5),
            str(event["event_time"])[:19])


def _signal(event, value, decision, s2_score=None, s2_decision=None,
            entry_wait=None, min_wait=None, ask_frac=1.0) -> dict:
    # Timestamps go out as ISO-UTC with an explicit Z so the BROWSER renders
    # them in the viewer's own timezone. Printing naive UTC strings showed
    # London users times an hour off their clock the moment storage was
    # (correctly) normalised to UTC -- store in UTC, display in local, always.
    def iso(value):
        text = str(value or "")[:19]
        return text.replace(" ", "T") + "Z" if text else ""
    return {"seen": iso(event.get("ingested_at")),
            "traded": iso(event["event_time"]),
            "account": str(event["account_key"]), "symbol": str(event["symbol"]),
            "side": "BUY" if int(event["direction"]) > 0 else "SELL",
            "lots": round(float(event["lots"]), 2),
            # Client's own entry price on this trade, so the stream shows the
            # exact level the signal fired at (not our fill) for tracking.
            "client_price": (round(float(event.get("price")), 5)
                             if event.get("price") not in (None, "") else None),
            "score": round(float(value), 4) if value is not None else None,
            "decision": decision,
            # Strategy 2's view of the SAME trade, so the one stream reflects
            # every active strategy's decision, not just the mirror's.
            "s2_score": round(float(s2_score), 4) if s2_score is not None else None,
            "s2_decision": s2_decision,
            # Entry-improvement model: predicted bps of better entry available
            # before the client's exit (None until the artifact exists).
            "wait_bps": round(float(entry_wait), 1) if entry_wait is not None else None,
            # The concrete TARGET ENTRY the engine will act on: for a
            # qualifying trade, the client's price improved by the predicted
            # bps in OUR direction when the wait clears the threshold (a
            # resting limit lands exactly there), or the client's price
            # itself when the model says enter immediately.
            **_target_entry(event, decision, entry_wait, min_wait, ask_frac)}


def _target_entry(event, decision, entry_wait, min_wait, ask_frac=1.0) -> dict:
    try:
        price = float(event.get("price") or 0)
        if decision not in ("copy", "invert") or entry_wait is None or price <= 0:
            return {"target_entry": None, "entry_mode": None}
        d = int(event["direction"]) if decision == "copy" else -int(event["direction"])
        ask = float(entry_wait) * max(float(ask_frac or 1.0), 0.0)
        if min_wait is not None and ask >= float(min_wait):
            return {"target_entry": round(price * (1.0 - d * ask / 1e4), 5),
                    "entry_mode": "limit"}
        return {"target_entry": round(price, 5), "entry_mode": "market"}
    except Exception:
        return {"target_entry": None, "entry_mode": None}


def _push(signal: dict) -> None:
    with _LOCK:
        _STATE.signals.insert(0, signal)
        _STATE.signals = _STATE.signals[:15]


def peek(config: VantageConfig, limit: int = 10) -> list[dict]:
    """Score the newest trades on demand, engine running or not."""
    try:
        frame = recent_openings(limit)
    except Exception:
        return []
    if frame.empty:
        return []
    history = account_history()
    cutoff = datetime.utcnow() - timedelta(minutes=config.max_signal_age_minutes)
    out = []
    for _, event in frame.iterrows():
        value = score(str(event["symbol"]), int(event["direction"]), float(event["lots"]),
                      float(event["price"] or 0), str(event["account_key"]), history)
        fresh = pd.Timestamp(event["event_time"]).to_pydatetime() >= cutoff
        out.append(_signal(event, value, route(value, config) if fresh else "stale"))
    return out


def _run() -> None:
    """Crash-proof wrapper: a startup exception in the engine thread used to
    die silently, leaving 'running' true and the log ending at 'connected' --
    undiagnosable. Every failure now lands in the log with its traceback."""
    try:
        _run_inner()
    except Exception as error:
        import traceback
        _log(f"ENGINE CRASH: {type(error).__name__}: {error}")
        for line in traceback.format_exc().splitlines()[-8:]:
            _log(f"  {line}")
        with _LOCK:
            _STATE.running = False
            _STATE.last_error = f"{type(error).__name__}: {error}"


def _run_inner() -> None:
    config = load_config()
    if booster() is None:
        with _LOCK:
            _STATE.running, _STATE.last_error = False, "no trained model"
        _log("no saved model -- train the quant model first")
        return
    if config.mode == "live":
        ok, message = connect(config)
        _log(message)
        if not ok:
            with _LOCK:
                _STATE.running, _STATE.last_error = False, message
            return
        # ALWAYS the account's ACTUAL leverage as the wall -- never a
        # hand-set stand-in. The DD calibration governs risk; the wall is
        # the venue's real margin boundary.
        actual = float(account_snapshot().get("leverage") or 0.0)
        if actual > 0 and actual != config.leverage:
            _log(f"leverage wall: using account's actual 1:{actual:.0f} "
                 f"(config said {config.leverage:.0f}x)")
            config.leverage = actual

    # AUTO LOT CALIBRATION: max DD = risk_budget_fraction x equity, at the
    # actual leverage. dd_calibration <= 0 means derive it from the replay's
    # sweep reference -- the largest multiplier whose replayed drawdown
    # stayed inside the budget at this wall. k already scales with equity,
    # so the reference m remains valid as the account grows or shrinks.
    if config.dd_calibration <= 0:
        deployed = _warm_meta().get("deployed") or {}
        reference = float(deployed.get("calibration") or 1.0)
        dd_ref = float(deployed.get("drawdown") or 0.0)
        budget_ref = float(deployed.get("dd_target") or 0.0)
        config.dd_calibration = max(reference, 1.0)
        _log(f"auto lot calibration: m={config.dd_calibration:.0f} from replay "
             f"(maxDD ${dd_ref:,.0f} vs budget ${budget_ref:,.0f} at "
             f"{deployed.get('wall', config.leverage):.0f}x)")

    # ADOPTION: every open position on the account becomes ours to manage the
    # moment the engine starts -- the comment (or a persisted reassignment)
    # names its source client, so mirror exits, netting, and rotation apply to
    # positions from earlier engine lives instead of leaving orphans to drift.
    _load_reassignments()
    if config.mode == "live":
        adopted = live_positions()
        _prune_reassignments({int(p["ticket"]) for p in adopted})
        tagged = sum(1 for p in adopted if p.get("source"))
        floating = sum(float(p.get("profit") or 0.0) for p in adopted)
        if adopted:
            _log(f"adopted {len(adopted)} open positions ({tagged} source-tagged, "
                 f"floating ${floating:+,.2f}); exits mirror for all tagged")
        # Strategy-2 positions from a previous engine life: their 15-minute
        # window almost certainly elapsed across the restart -- close now.
        stale_s2 = [p for p in adopted
                    if str(p.get("comment") or "").startswith(("2f", "z2f"))]
        for position in stale_s2:
            _DEADLINES[int(position["ticket"])] = datetime.utcnow()
        if stale_s2:
            _log(f"{len(stale_s2)} strategy-2 positions marked for immediate "
                 f"close (hold elapsed across restart)")
        # Resting limits from a previous engine life have no TTL owner and a
        # thesis whose freshness window has certainly expired -- cancel them.
        try:
            mt5 = _mt5()
            for order in (mt5.orders_get() or []):
                if getattr(order, "magic", 0) == 909090:
                    mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE,
                                    "order": order.ticket})
                    _log(f"stale limit cancelled #{order.ticket} "
                         f"(no owner after restart)")
        except Exception:
            pass
        # The pending map is module state and survives an in-process
        # stop/start; left in place, _tend_pending sees every limit just
        # cancelled above as "vanished before its deadline" and logs a
        # phantom LIMIT FILLED for each (85 of them on 2026-09-13).
        with _LOCK:
            _PENDING.clear()
        # A new engine life starts with no memory of the old book: positions
        # closed by hand while the engine was down are not "vanished" trades.
        _LAST_BOOK.clear()
        _VANISHED.clear()
    # The tape buffer refills from now: no ACTIVE bias until it has warmed.
    _TAPE_STARTED[0] = time.time()
    _TAPE_EVENTS.clear()
    _TAPE_BIAS.clear()
    _TAPE_GATE_BIAS.clear()

    _seed_bars()
    _log("warming: funding snapshot")
    _funding_snapshot()
    _log("warming: account-day corpus")
    _account_day_snapshot()
    _log("warming: account history")
    history = account_history()
    _log("warming: strategy stats")
    fallback = float(history["mean_abs_pnl"].median()) if len(history) else 50.0
    drawdown = strategy_drawdown(config)
    snapshot = account_snapshot()
    balance = float(snapshot.get("balance") or 0.0) or 10_000.0
    capital = float(snapshot.get("equity") or 0.0) or balance
    k = scale_factor(config, capital, drawdown)
    with _LOCK:
        _STATE.strategy_dd_1x, _STATE.scale_factor = drawdown, k
    _log(f"ready: {len(history):,} accounts | dd@1x ${drawdown:,.0f} | "
         f"equity ${capital:,.0f} | k={k:.6f} | mode={config.mode}")
    anchors = calibrate_anchors(config)
    _log(f"score anchors from artifact: copy >= {anchors.get('copy_floor', 0.8):.3f}, "
         f"invert <= {anchors.get('invert_ceiling', 0.3):.3f} "
         f"(the tested standard; rolling window may only tighten)")
    if config.use_entry_model:
        _log("entry model: "
             + (f"ACTIVE -- limits rest when predicted retracement >= "
                f"{config.entry_wait_min_bps:.0f}bps (ttl "
                f"{config.entry_limit_ttl_minutes:.0f}m, cancelled on client exit)"
                if _entry_model() is not None
                else "enabled but artifact MISSING -- market entries only"))
    multipliers = symbol_multipliers(config)
    excluded = sorted(s for s, m in multipliers.items() if m <= 0)
    boosted = sorted(((s, m) for s, m in multipliers.items() if m > 1.2),
                     key=lambda x: -x[1])[:6]
    _log(f"symbol scaling: {len(multipliers)} instruments | excluded "
         f"{excluded or 'none'} | boosted "
         f"{[f'{s} x{m:.1f}' for s, m in boosted] or 'none'}")
    # SYMBOL MAP: every canonical instrument the model knows, checked against
    # this venue's listing, so unmappable flow is a named list at startup
    # rather than a stream of silent "not listed" blocks.
    if config.mode == "live":
        venue = _venue_symbols()
        unmapped = sorted(s for s, m in multipliers.items()
                          if m > 0 and s not in venue)
        _log(f"symbol map: {len(multipliers) - len(unmapped)}/{len(multipliers)} "
             f"canonical instruments tradeable here | unmapped: "
             f"{unmapped[:12] if unmapped else 'none'}"
             + (f" (+{len(unmapped) - 12} more)" if len(unmapped) > 12 else ""))

    cursor = datetime.utcnow() - timedelta(seconds=30)
    resize_at = time.time() + 600
    heartbeat_at = time.time() + 120
    breaker_at = 0.0
    entries_paused = False
    feed_opened = feed_closed = 0
    while True:
        with _LOCK:
            if not _STATE.running:
                break
        try:
            # EQUITY CIRCUIT BREAKER -- checked every 15s, ahead of any new
            # entry. Trims the worst floaters to a safe margin level and
            # pauses entries; the replay's daily-grain drawdown cannot see
            # the intraday floating book, so this watches equity directly.
            if config.mode == "live" and time.time() >= breaker_at:
                breaker_at = time.time() + 15
                snap = account_snapshot()
                balance = float(snap.get("balance") or 0.0)
                equity = float(snap.get("equity") or 0.0)
                margin = float(snap.get("margin") or 0.0)
                level = (equity / margin * 100.0) if margin > 0 else 1e9
                if balance > 0 and (equity < balance * config.equity_floor_frac
                                    or level < config.safe_margin_level):
                    # NO forced liquidation: existing positions run to their
                    # mirror exit (the replay proved our own trims/stop-outs,
                    # not the client exits, were the loss). We only PAUSE new
                    # entries; if margin truly runs out the broker stops out.
                    if not entries_paused:
                        _log(f"MARGIN GUARD: equity ${equity:,.0f} / balance "
                             f"${balance:,.0f} (margin level {level:.0f}%) -- "
                             f"pausing NEW entries; open positions run to their "
                             f"mirror exit (no forced liquidation)")
                    entries_paused = True
                elif (entries_paused and equity >= balance * config.resume_frac
                        and level >= config.safe_margin_level):
                    # Resume only once BOTH triggers have cleared: a pause on
                    # margin level used to lift 15 s later on the equity test
                    # alone, with the margin level still where it tripped.
                    entries_paused = False
                    _log(f"circuit breaker cleared: equity ${equity:,.0f} "
                         f">= {config.resume_frac:.0%} of balance, margin level "
                         f"{level:.0f}% >= {config.safe_margin_level:.0f}% -- entries resume")
            if time.time() >= heartbeat_at:
                heartbeat_at = time.time() + 120
                _log(f"heartbeat: feed {feed_opened} open / {feed_closed} close "
                     f"since last | scored {_STATE.scored} | stale "
                     f"{_STATE.stale_skipped} | acted {_STATE.acted} | filled "
                     f"{_STATE.filled} | blocked {_STATE.blocked}")
                feed_opened = feed_closed = 0
            # COMPOUND ON EQUITY: the 35% contract defends equity, so k tracks
            # live equity (open winners grow size immediately, drawdowns shrink
            # it) -- re-derived every 2 minutes; balance-based resizing lagged
            # the compounding by however long winners stayed open.
            if time.time() >= resize_at:
                resize_at = time.time() + 120
                snapshot = account_snapshot()
                fresh_capital = (float(snapshot.get("equity") or 0.0)
                                 or float(snapshot.get("balance") or 0.0))
                if fresh_capital > 0 and drawdown > 0:
                    fresh_k = scale_factor(config, fresh_capital, drawdown)
                    if fresh_k > 0 and abs(fresh_k - k) / max(k, 1e-9) > 0.02:
                        _log(f"resized: equity ${fresh_capital:,.0f} -> "
                             f"k={fresh_k:.6f} (was {k:.6f})")
                        k = fresh_k
                        with _LOCK:
                            _STATE.scale_factor = k
            openings, closings = pd.DataFrame(), pd.DataFrame()
            if config.use_kafka_events:
                # Kafka is the fast feed. Its reads are wrapped: a transient
                # duckdb 'store busy' (the materialiser mid-write) must NOT abort
                # the iteration and skip the trade logic -- it just falls through
                # to the MySQL feed for this tick and retries next tick.
                try:
                    openings = poll_openings(cursor)
                    closings = poll_closings(cursor)
                    marks = [f["ingested_at"].max()
                             for f in (openings, closings) if len(f)]
                    if marks:
                        cursor = max(marks).to_pydatetime()
                except Exception as error:
                    _log(f"kafka feed: {type(error).__name__}: {str(error)[:50]}")
                    openings, closings = pd.DataFrame(), pd.DataFrame()

            # The PRODUCTION feed, straight from the MySQL reporting proxy.
            # Same shape, same identity space -- a trade arriving from both
            # sources collapses in the _SEEN set, so nothing double-fires.
            if config.use_db_feed:
                try:
                    from webapp import trade_feed
                    db_open, db_close = trade_feed.poll()
                    if len(db_open):
                        openings = pd.concat(
                            [openings, _prepare(db_open, deflate_cents=False)],
                            ignore_index=True)
                    if len(db_close):
                        closings = pd.concat(
                            [closings, _prepare(db_close, deflate_cents=False)],
                            ignore_index=True)
                except Exception as error:
                    _log(f"db feed: {type(error).__name__}: {error}")

            feed_opened += len(openings)
            feed_closed += len(closings)

            # Every closing -- fresh or stale -- updates the client's LIVE
            # form before any of this tick's signals are scored.
            for _, event in closings.iterrows():
                profit = event.get("profit")
                if profit is None or pd.isna(profit):
                    continue
                key = (str(event["account_key"]),
                       str(event.get("event_time"))[:19],
                       round(float(profit), 2),
                       round(float(event.get("lots") or 0), 2))
                if key in _SEEN_CLOSES:
                    continue
                _SEEN_CLOSES.add(key)
                if len(_SEEN_CLOSES) > 100_000:
                    _SEEN_CLOSES.clear()
                _note_client_close(str(event["account_key"]), float(profit))
                _tape_note_close(config, event)

            cutoff = datetime.utcnow() - timedelta(minutes=config.max_signal_age_minutes)
            intents: list[dict] = []
            for _, event in openings.iterrows():
                identity = _identity(event)
                if identity in _SEEN:
                    continue
                _SEEN.add(identity)
                if len(_SEEN) > 50_000:
                    _SEEN.clear()

                # Every deduped opening -- fresh or stale -- feeds the bar
                # history: stale events are still real tape and prime context.
                _note_trade(str(event["symbol"]), float(event["price"] or 0),
                            int(event["direction"]),
                            pd.Timestamp(event["event_time"]).to_pydatetime())

                # RULE 1: freshness. History is COUNTED, never traded -- and no
                # longer shown: a reconnect replays the gap at full speed, and
                # pushing every replayed event flooded the panel with a wall of
                # "stale" rows that displaced the live view and read as broken.
                if pd.Timestamp(event["event_time"]).to_pydatetime() < cutoff:
                    with _LOCK:
                        _STATE.stale_skipped += 1
                    continue

                (value, multiple, s2_score, s2_magnitude, entry_wait,
                 perlot, exit_fe) = score(
                    str(event["symbol"]), int(event["direction"]),
                    float(event["lots"]), float(event["price"] or 0),
                    str(event["account_key"]), history,
                    update_state=True, with_magnitude=True)
                # Every scored open joins the tape (strategy 2's input) with
                # the score the engine just gave it.
                _tape_note_open(config, event, value)
                if config.strategy2 and getattr(config, "strategy2_fade", False):
                    _execute_strategy2(config, event, s2_score, s2_magnitude)
                stance = route(value, config)
                # S2's decision on this same trade, for the unified stream.
                s2_decision = None
                if s2_score is not None:
                    s2_decision = ("fade" if (config.strategy2
                                   and s2_score <= config.strategy2_fade_ceiling)
                                   else "hold")
                with _LOCK:
                    _STATE.scored += 1
                _push(_signal(event, value, stance, s2_score, s2_decision,
                              entry_wait, config.entry_wait_min_bps,
                              config.entry_ask_fraction))
                if stance == "ignore":
                    continue
                # Strategy-1 master switch: keep scoring/streaming, but open no
                # new mirror entries when disabled (existing ones still exit).
                if not getattr(config, "strategy1", True):
                    continue
                with _LOCK:
                    _STATE.acted += 1
                account = str(event["account_key"])
                # expected_dollars already speaks in the client's typical
                # per-trade dollars (their size embedded). Scale by k and by
                # how this trade's size compares to THEIR usual -- multiplying
                # by raw lots double-counted size and crushed expected for
                # small-lot clients, which made the cost hurdle reject 98.9%
                # of decile trades in the deployed-logic replay.
                size_factor = 1.0
                if account in history.index:
                    usual = float(history.loc[account].get("mean_notional")
                                  or np.nan)
                    if np.isfinite(usual) and usual > 0:
                        notional_now = (float(event["lots"] or 0) * 100.0
                                        * float(event["price"] or 0))
                        if notional_now > 0:
                            size_factor = float(np.clip(
                                notional_now / usual, 0.25, 4.0))
                # EDGE PER CANONICAL LOT: edge x predicted |pnl|/cost x the
                # round-trip cost of ONE lot of this symbol. Execute() then
                # multiplies by the lots it actually sizes, so expected-$ and
                # the hurdle (spread+commission at those same lots) are on
                # identical footing -- no k/calibration/floor rescaling chain.
                expected_per_lot = None
                from webapp.trade_features import trade_cost
                cost_lot = trade_cost(str(event["symbol"]),
                                      float(event["price"] or 0), 1.0)
                edge_value = ((2.0 * value - 1.0) if stance == "copy"
                              else (1.0 - 2.0 * value))
                if perlot is not None:
                    # MODEL E: direct $ per canonical lot -- the four-arm
                    # study's winner (beats multiple x cost_lot by 7-17%
                    # $/slot-hour at every selectivity level).
                    expected_per_lot = max(0.0, edge_value) * perlot
                elif multiple is not None and cost_lot > 0:
                    expected_per_lot = max(0.0, edge_value) * multiple * cost_lot
                if expected_per_lot is None:
                    # Heuristic fallback: the client's typical |pnl| per THEIR
                    # lot, times our edge -- normalised to one lot.
                    per_client_lot = expected_dollars(
                        value, stance, account, history, fallback) \
                        / max(float(event["lots"] or 0) or 0.01, 0.01)
                    expected_per_lot = per_client_lot
                expected = expected_per_lot * k * max(config.dd_calibration, 1.0)
                direction = (int(event["direction"]) if stance == "copy"
                             else -int(event["direction"]))
                # feed the per-symbol side-score memory (the side-score
                # guard's evidence) with every scored intent.
                try:
                    from webapp.trade_feed import _canonical
                    _record_side_score(
                        _canonical(str(event["symbol"]))
                        or str(event["symbol"]),
                        direction, max(0.0, edge_value))
                except Exception:
                    pass
                intents.append({"account": account, "stance": stance,
                                "symbol": str(event["symbol"]),
                                "client_direction": int(event["direction"]),
                                "client_lots": float(event["lots"]),
                                "client_price": float(event["price"] or 0),
                                "expected": expected, "value": value, "k": k,
                                "expected_per_lot": expected_per_lot,
                                "direction": direction, "multiple": multiple,
                                "entry_wait_bps": entry_wait,
                                "exit_fe_bps": exit_fe})

            _record_client_closes(closings)
            fresh_exits = (closings.loc[
                pd.to_datetime(closings["event_time"]) >= cutoff]
                if len(closings) else closings)
            # Order: recycle what the batch nets out, mirror the remaining
            # exits (freeing book capacity and shedding dead risk FIRST), then
            # open what is still owed.
            leftovers = _reconcile(config, intents, fresh_exits)
            if leftovers is not None and len(leftovers):
                mirror_exits(config, leftovers)
            for intent in intents:
                if intent.get("consumed"):
                    continue
                if entries_paused:
                    break                 # circuit breaker: no new risk
                execute(config, intent["account"], intent["stance"],
                        intent["symbol"], intent["client_direction"],
                        intent["client_lots"], intent["client_price"],
                        k, intent["expected"], intent["value"],
                        multiple=intent.get("multiple"),
                        entry_wait_bps=intent.get("entry_wait_bps"),
                        expected_per_lot=intent.get("expected_per_lot"),
                        exit_fe_bps=intent.get("exit_fe_bps"))
            _tend_pending(config)
            _tend_deadlines(config)
            _tend_closes(config)
            _tend_manual_closes(config, k)
            _tape_tick(config, entries_paused)
            if time.time() - _LAST_SWEEP[0] >= SWEEP_SECONDS:
                _LAST_SWEEP[0] = time.time()
                _sweep_orphans(config)
            # STREAM WATCHDOG: quotes or deals gone stale while the consumer
            # claims to run -> bounce it (threaded: restart blocks on joining
            # wedged threads; rate-limited to one bounce per 10 minutes).
            if time.time() - _WATCHDOG[0] >= 120:
                _WATCHDOG[0] = time.time()
                try:
                    health = stream_health()
                    q_age = float(health.get("quote_age_seconds") or 0)
                    t_age = float(health.get("trade_age_seconds") or 0)
                    if (health.get("available") and config.use_kafka_events
                            and (q_age > 240 or t_age > 900)
                            and time.time() - _WATCHDOG[1] > 600):
                        _WATCHDOG[1] = time.time()
                        _log(f"stream watchdog: quotes {q_age:.0f}s / deals "
                             f"{t_age:.0f}s stale -- restarting consumer")

                        def _bounce():
                            try:
                                from webapp import kafka_service
                                # DEALS ONLY, as at boot: bouncing WITH quotes
                                # re-added five quote consumers per bounce, and
                                # wedged generations never exit -- 13 leaked
                                # consumers, 124 threads, 6 GB and 100-second
                                # page loads on 14 Sep 2026.
                                kafka_service.MATERIALISER.restart(with_quotes=False)
                                _log("stream watchdog: consumer restarted")
                            except Exception as err:
                                _log(f"stream watchdog: restart failed "
                                     f"{type(err).__name__}: {err}")
                        threading.Thread(target=_bounce, daemon=True).start()
                except Exception:
                    pass
        except Exception as error:
            _log(f"loop error: {type(error).__name__}: {error}")
            with _LOCK:
                _STATE.last_error = f"{type(error).__name__}: {error}"
        time.sleep(config.poll_seconds)
    _log("engine stopped")


def start() -> bool:
    global _THREAD
    with _LOCK:
        if _STATE.running:
            return False
        _STATE.running, _STATE.started_at, _STATE.last_error = True, time.time(), ""
    _THREAD = threading.Thread(target=_run, daemon=True, name="vantage")
    _THREAD.start()
    return True


def stop() -> None:
    with _LOCK:
        _STATE.running = False


# ---------------------------------------------------------------- reporting
# ------------------------------------------------- realized per-strategy P&L
_PERF_CACHE: dict = {"stamp": 0.0, "data": None}
_KF_CLOSES_CACHE: dict = {"stamp": 0.0, "frame": None}
_STRATEGY_MAGIC = {909090: "s1", 909091: "s2"}


def _orders_by_ticket() -> dict[int, dict]:
    """Our order-store rows keyed by venue ticket, to join to realized deals
    (stance, expected, client vs our size, entry score). Queries ONLY ticketed
    rows: the busy-hours blocked/rejected spam pushed actual fills out of a
    newest-N window within hours, silently unjoining most closed trades."""
    _ensure_table()
    out: dict[int, dict] = {}
    with sqlite3.connect(DB) as connection:
        connection.row_factory = sqlite3.Row
        for row in connection.execute(
                "SELECT * FROM vantage_orders WHERE ticket IS NOT NULL "
                "ORDER BY id DESC LIMIT 8000"):
            r = dict(row)
            out.setdefault(int(r["ticket"]), r)
    return out


def _empty_perf() -> dict:
    return {"closed": 0, "realized_usd": 0.0, "gross_usd": 0.0, "cost_usd": 0.0,
            "win_rate": 0.0, "wins": 0, "losses": 0, "avg_win": 0.0,
            "avg_loss": 0.0, "profit_factor": 0.0, "max_dd_usd": 0.0,
            "expected_usd": 0.0, "hit_rate": 0.0, "client_equiv_usd": 0.0,
            "edge_vs_client_usd": 0.0, "by_symbol": [], "recent": []}


def _summarise_perf(rows: list[dict]) -> dict:
    """Aggregate a strategy's closed trades into the stats the panel shows:
    realized vs expected (accuracy), realized vs client-equivalent (edge),
    win rate, profit factor, realized-curve drawdown, and a per-symbol
    breakdown so 'where money is made or lost' is legible at a glance."""
    if not rows:
        return _empty_perf()
    rows = sorted(rows, key=lambda r: r["close_time"])
    nets = [r["net"] for r in rows]
    wins = [n for n in nets if n > 0]
    losses = [n for n in nets if n <= 0]
    realized = float(sum(nets))
    expected = float(sum(r["expected"] for r in rows if r["matched"]))
    client_equiv = float(sum(r["client_equiv"] for r in rows))
    matched = [r for r in rows if r["matched"] and r["expected"]]
    hit = (sum(1 for r in matched if (r["net"] > 0) == (r["expected"] > 0))
           / len(matched)) if matched else 0.0
    curve = np.cumsum(nets)
    max_dd = float(np.max(np.maximum.accumulate(curve) - curve)) if len(curve) else 0.0
    sym: dict[str, dict] = {}
    for r in rows:
        s = sym.setdefault(r["symbol"], {"symbol": r["symbol"], "net": 0.0,
                                         "count": 0, "wins": 0})
        s["net"] += r["net"]; s["count"] += 1
        s["wins"] += 1 if r["net"] > 0 else 0
    gross_win, gross_loss = float(sum(wins)), float(-sum(losses))
    return {
        "closed": len(rows), "realized_usd": round(realized, 2),
        "gross_usd": round(float(sum(r["profit"] for r in rows)), 2),
        "cost_usd": round(float(sum(r["cost"] for r in rows)), 2),
        "win_rate": len(wins) / len(rows), "wins": len(wins), "losses": len(losses),
        "avg_win": round(float(np.mean(wins)), 2) if wins else 0.0,
        "avg_loss": round(float(np.mean(losses)), 2) if losses else 0.0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else 0.0,
        "max_dd_usd": round(max_dd, 2), "expected_usd": round(expected, 2),
        "hit_rate": hit, "client_equiv_usd": round(client_equiv, 2),
        # HEADLINE EDGE from rows with the client's REAL close only: mixing in
        # the stance-mirrored fallback (a tautology) polluted the number.
        "edge_vs_client_usd": round(
            float(sum(r["net"] for r in rows if r.get("client_actual"))
                  - sum(r["client_equiv"] for r in rows if r.get("client_actual"))), 2),
        "client_matched": int(sum(1 for r in rows if r.get("client_actual"))),
        # Share of rows whose client_equiv is the client's REAL close event
        # (the rest fall back to the stance-mirrored approximation).
        "client_actual_share": round(
            sum(1 for r in rows if r.get("client_actual")) / len(rows), 3),
        "by_symbol": [{"symbol": s["symbol"], "net": round(s["net"], 2),
                       "count": s["count"], "wins": s["wins"]}
                      for s in sorted(sym.values(), key=lambda x: x["net"])],
        "recent": [{"symbol": r["symbol"], "direction": r["direction"],
                    "net": round(r["net"], 2), "profit": round(r["profit"], 2),
                    "cost": round(r["cost"], 2),
                    "client_equiv": round(r["client_equiv"], 2),
                    "expected": round(r["expected"], 2), "our_lots": r["our_lots"],
                    "score": r["score"], "stance": r["stance"],
                    "close_time": r["close_time"]}
                   for r in sorted(rows, key=lambda r: r["close_time"],
                                   reverse=True)[:40]]}


#: Transient diagnostics for the client-close join (surfaced in perf output).
_JOIN_DEBUG: dict = {}


_PERF_LOCK = threading.Lock()


def strategy_performance(window_days: int = 30, ttl: float = 15.0) -> dict:
    """Per-strategy REALIZED performance from the venue's own closed deals,
    joined to our order store so each close carries the strategy that owned it,
    the model's expectation, and the client's equivalent outcome at our size.

    Cached AND single-flight: the dashboard polls every 5 s and each poll used
    to recompute this in its own worker thread when the cache had lapsed --
    six threads rebuilding the same report at once, 60-second pages (14 Sep
    2026). Now one thread rebuilds while the others serve the last result.
    client_equiv reconstructs the client's directional result at OUR size from
    the venue market P&L: copy -> same sign, invert/fade -> opposite. Costs
    (swap+commission) are separated so the cost drag is visible.
    """
    now = time.time()
    cached = _PERF_CACHE.get("data")
    if cached is not None and now - _PERF_CACHE["stamp"] < ttl:
        return cached
    if not _PERF_LOCK.acquire(blocking=False):
        if cached is not None:
            return cached                       # a rebuild is in flight: last result
        _PERF_LOCK.acquire()                    # first ever: wait for it
        _PERF_LOCK.release()
        return _PERF_CACHE.get("data") or {"available": False, "window_days": window_days,
                                           "s1": _empty_perf(), "s2": _empty_perf()}
    try:
        return _strategy_performance_inner(window_days)
    finally:
        _PERF_LOCK.release()


def _strategy_performance_inner(window_days: int = 30) -> dict:
    now = time.time()
    _JOIN_DEBUG.clear()
    out = {"available": False, "window_days": window_days,
           "s1": _empty_perf(), "s2": _empty_perf()}
    config = load_config()
    if config.mode != "live":
        out["reason"] = "paper mode -- no venue deal history"
        _PERF_CACHE.update(stamp=now, data=out)
        return out
    try:
        import datetime as _dt
        mt5 = _mt5()
        frm = _dt.datetime.now() - _dt.timedelta(days=window_days)
        reset = stats_reset_at()                     # fresh-balance line in sand
        if reset:
            # WIDE fetch, precise cut: the venue interprets this bound on its
            # own clock (UTC+3 here), so trimming to the raw reset epoch
            # silently dropped the first hours after a reset. The exact
            # server-offset-corrected filter below does the real cutting.
            frm = max(frm, _dt.datetime.fromtimestamp(reset)
                      - _dt.timedelta(hours=12))
        deals = mt5.history_deals_get(frm, _dt.datetime.now()) or []
    except Exception as error:
        out["reason"] = f"{type(error).__name__}: {error}"
        _PERF_CACHE.update(stamp=now, data=out)
        return out

    by_pos: dict[int, dict] = {}
    for d in deals:
        pid = int(getattr(d, "position_id", 0) or 0)
        if not pid:
            continue
        entry = int(getattr(d, "entry", 0) or 0)
        rec = by_pos.setdefault(pid, {
            "symbol": getattr(d, "symbol", ""), "magic": 0, "profit": 0.0,
            "swap": 0.0, "commission": 0.0, "close_time": 0,
            "has_out": False, "in_dir": 0})
        rec["profit"] += float(getattr(d, "profit", 0.0) or 0.0)
        rec["swap"] += float(getattr(d, "swap", 0.0) or 0.0)
        rec["commission"] += float(getattr(d, "commission", 0.0) or 0.0)
        if getattr(d, "magic", 0):
            rec["magic"] = int(d.magic)
        if getattr(d, "symbol", ""):
            rec["symbol"] = d.symbol
        if entry == 0:                                   # DEAL_ENTRY_IN
            rec["in_dir"] = 1 if int(getattr(d, "type", 0) or 0) == 0 else -1
        elif entry in (1, 2, 3):                         # OUT / INOUT / OUT_BY
            rec["has_out"] = True
            rec["close_time"] = max(rec["close_time"], int(getattr(d, "time", 0) or 0))

    joined = _orders_by_ticket()

    # REAL client outcomes: join each of our closed positions to the CLIENT's
    # own closing event (login + symbol + time, from the events store) so
    # "client @ our size" is the client's actual P&L scaled to our lots. The
    # old construction mirrored OUR venue P&L back through the stance sign --
    # a tautology that measured nothing but our own costs.
    client_closes: dict[tuple, list] = {}
    cent_cache: dict[str, set] = {}
    # PRIMARY: the closes the decision feed itself delivered (complete across
    # servers, profits already cent-deflated). Tuple: (utc, profit, volume,
    # server, src) with src 'sq' = pre-deflated / 'kf' = raw kafka store.
    try:
        _ensure_table()
        with sqlite3.connect(DB) as cx:
            for login, server, canon, ts, profit, volume in cx.execute(
                    "SELECT login, server, canon, event_utc, profit, volume "
                    "FROM vantage_client_closes WHERE event_utc > ?",
                    [(frm - _dt.timedelta(hours=6)).timestamp()]):
                client_closes.setdefault((int(login), str(canon)), []).append(
                    (float(ts), float(profit or 0.0), float(volume or 0.0),
                     str(server or ""), "sq"))
        _JOIN_DEBUG["sqlite_rows"] = sum(len(v) for v in client_closes.values())
    except Exception as err:
        _JOIN_DEBUG["sqlite_err"] = f"{type(err).__name__}"
    # FALLBACK: the kafka events store (subset of servers, raw cent profits)
    # for closes that predate the sqlite recorder. Cached 5 minutes -- this
    # 70k-row scan on every perf refresh contended with the quote writer.
    try:
        from webapp.trade_feed import _canonical
        if (time.time() - _KF_CLOSES_CACHE["stamp"] > 300
                or _KF_CLOSES_CACHE["frame"] is None):
            with _store() as connection:
                _KF_CLOSES_CACHE["frame"] = connection.execute(
                    f"""SELECT server, login, symbol, canonical, event_time,
                               profit, volume
                        FROM events WHERE 1=1 {_CLOSING}
                        AND profit IS NOT NULL AND event_time > ?""",
                    [_dt.datetime.utcnow() - _dt.timedelta(hours=30)]).df()
            _KF_CLOSES_CACHE["stamp"] = time.time()
        frame = _KF_CLOSES_CACHE["frame"]
        for row in frame.itertuples(index=False):
            canon = (str(row.canonical) if row.canonical
                     else (_canonical(str(row.symbol)) or str(row.symbol)))
            key = (int(row.login), canon)
            epoch = pd.Timestamp(row.event_time).timestamp()
            client_closes.setdefault(key, []).append(
                (epoch, float(row.profit or 0.0), float(row.volume or 0.0),
                 str(row.server), "kf"))
        for lst in client_closes.values():
            lst.sort()
        _JOIN_DEBUG.update(build="ok", close_rows=int(len(frame)),
                           keys=len(client_closes),
                           sample=next(iter(client_closes), None))
    except Exception as err:
        _JOIN_DEBUG.update(build=f"{type(err).__name__}: {err}")
        for lst in client_closes.values():
            lst.sort()
    srv_off = _server_offset_seconds()

    def _bump(name):
        _JOIN_DEBUG[name] = _JOIN_DEBUG.get(name, 0) + 1

    def _client_pnl(order, rec):
        """Client's actual realized P&L on the mirrored trade, cent-deflated
        and scaled to our lots; None when their close event can't be found."""
        if not order or not client_closes:
            _bump("miss_no_order" if not order else "miss_no_closes")
            return None
        src = str(order.get("source_account") or "")
        server, _, login_s = src.rpartition(":")
        if not login_s.isdigit():
            _bump("miss_login_format")
            _JOIN_DEBUG.setdefault("sample_src", src)
            return None
        try:
            from webapp.trade_feed import _canonical, cent_logins
            canon = _canonical(str(rec["symbol"])) or str(rec["symbol"])
            rows = client_closes.get((int(login_s), canon))
            if not rows:
                _bump("miss_no_rows")
                _JOIN_DEBUG.setdefault("sample_miss", (int(login_s), canon))
                return None
            close_utc = float(rec["close_time"]) - srv_off
            best = min(rows, key=lambda r: abs(r[0] - close_utc))
            if abs(best[0] - close_utc) > 900:
                _bump("miss_time_window")
                _JOIN_DEBUG.setdefault("sample_dt", round(best[0] - close_utc))
                return None
            profit = best[1]
            # 'sq' rows arrived through the decision feed pre-deflated; only
            # raw kafka-store rows still need cent detection.
            if best[4] != "sq":
                for srv in (server, best[3]):
                    if not srv:
                        continue
                    if srv not in cent_cache:
                        try:
                            cent_cache[srv] = set(cent_logins(srv))
                        except Exception:
                            cent_cache[srv] = set()
                    if int(login_s) in cent_cache[srv]:
                        profit /= 100.0
                        break
            our_lots = float(order.get("our_lots") or 0.0)
            client_lots = float(order.get("client_lots") or 0.0)
            if our_lots <= 0 or client_lots <= 0:
                return None
            return profit * (our_lots / client_lots)
        except Exception:
            return None

    buckets: dict[str, list] = {"s1": [], "s2": []}
    for pid, rec in by_pos.items():
        if not rec["has_out"]:
            continue                                     # still open
        # Venue deal times are SERVER clock (UTC+3 here); the reset line is a
        # UTC epoch. Comparing raw let ~3h of pre-reset closes (the stale-book
        # liquidation) bleed into a "fresh" window.
        if reset and rec["close_time"] and (rec["close_time"] - srv_off) < reset:
            continue                                     # before the reset line
        order = joined.get(pid)
        strat = _STRATEGY_MAGIC.get(rec["magic"])
        if strat is None:
            strat = "s2" if (order and order.get("stance") == "fade15") else "s1"
        net = rec["profit"] + rec["swap"] + rec["commission"]
        sign = -1 if strat == "s2" else 1
        if order and order.get("client_direction") and order.get("our_direction"):
            sign = 1 if (int(order["client_direction"])
                         * int(order["our_direction"]) > 0) else -1
        actual = _client_pnl(order, rec)
        buckets[strat].append({
            "symbol": rec["symbol"], "net": net, "profit": rec["profit"],
            "cost": rec["swap"] + rec["commission"],
            "client_equiv": (sign * actual) if actual is not None
                            else rec["profit"] * sign,
            "client_actual": actual is not None,
            # HONEST SCALE: stored expected carries the dd_calibration lot
            # multiplier (sizing, not forecast) -- dividing it back makes the
            # panel's Expected a number realized P&L can be held against
            # (~$6.8/trade vs the inflated $236/trade it used to show).
            "expected": (float(order.get("expected_usd") or 0.0)
                         / max(float(getattr(config, "dd_calibration", 1.0) or 1.0), 1.0))
                        if order else 0.0,
            "our_lots": float(order.get("our_lots") or 0.0) if order else 0.0,
            "score": (float(order["live_score"]) if order
                      and order.get("live_score") is not None else None),
            "stance": order.get("stance") if order else None,
            "direction": rec["in_dir"], "close_time": rec["close_time"],
            "matched": order is not None})

    out = {"available": True, "window_days": window_days,
           "s1": _summarise_perf(buckets["s1"]),
           "s2": _summarise_perf(buckets["s2"])}
    _JOIN_DEBUG["positions_seen"] = len(by_pos)
    _JOIN_DEBUG["rows_bucketed"] = sum(len(b) for b in buckets.values())
    out["s1"]["join_debug"] = dict(_JOIN_DEBUG)
    out["s1"]["window_days"] = window_days
    out["s2"]["window_days"] = window_days
    _PERF_CACHE.update(stamp=now, data=out)
    return out


def s1_trades_dump(window_days: int = 45, strat_filter: str = "s1") -> dict:
    """Forensic dump: every closed position for a strategy with its FULL entry
    and exit legs (time + price) joined to the order store, for offline replay.
    Emitted so the mirror-only / slippage / latency study can run standalone
    against MySQL (client orders + ticks) without re-touching MT5."""
    import datetime as _dt
    try:
        mt5 = _mt5()
        frm = _dt.datetime.now() - _dt.timedelta(days=window_days)
        reset = stats_reset_at()
        if reset:
            frm = max(frm, _dt.datetime.fromtimestamp(reset))
        deals = mt5.history_deals_get(frm, _dt.datetime.now()) or []
    except Exception as error:
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}
    by_pos: dict[int, dict] = {}
    for d in deals:
        pid = int(getattr(d, "position_id", 0) or 0)
        if not pid:
            continue
        entry = int(getattr(d, "entry", 0) or 0)
        rec = by_pos.setdefault(pid, {
            "symbol": "", "magic": 0, "profit": 0.0, "swap": 0.0,
            "commission": 0.0, "in_time": 0, "in_price": 0.0, "in_dir": 0,
            "out_time": 0, "out_price": 0.0, "has_out": False,
            "in_comment": "", "out_comment": ""})
        rec["profit"] += float(getattr(d, "profit", 0.0) or 0.0)
        rec["swap"] += float(getattr(d, "swap", 0.0) or 0.0)
        rec["commission"] += float(getattr(d, "commission", 0.0) or 0.0)
        if getattr(d, "magic", 0):
            rec["magic"] = int(d.magic)
        if getattr(d, "symbol", ""):
            rec["symbol"] = d.symbol
        if entry == 0:
            rec["in_time"] = int(getattr(d, "time", 0) or 0)
            rec["in_price"] = float(getattr(d, "price", 0.0) or 0.0)
            rec["in_dir"] = 1 if int(getattr(d, "type", 0) or 0) == 0 else -1
            rec["in_comment"] = str(getattr(d, "comment", "") or "")
            rec["volume"] = float(getattr(d, "volume", 0.0) or 0.0)
        elif entry in (1, 2, 3):
            rec["has_out"] = True
            if int(getattr(d, "time", 0) or 0) >= rec["out_time"]:
                rec["out_time"] = int(getattr(d, "time", 0) or 0)
                rec["out_price"] = float(getattr(d, "price", 0.0) or 0.0)
                rec["out_comment"] = str(getattr(d, "comment", "") or "")
    joined = _orders_by_ticket()
    reset = stats_reset_at()
    rows = []
    for pid, rec in by_pos.items():
        if not rec["has_out"]:
            continue
        if reset and rec["out_time"] and rec["out_time"] < reset:
            continue
        order = joined.get(pid)
        strat = _STRATEGY_MAGIC.get(rec["magic"])
        if strat is None:
            strat = "s2" if (order and order.get("stance") == "fade15") else "s1"
        if strat != strat_filter:
            continue
        o = order or {}
        rows.append({
            "ticket": pid, "symbol": rec["symbol"],
            "in_time": rec["in_time"], "in_price": rec["in_price"],
            "in_dir": rec["in_dir"], "out_time": rec["out_time"],
            "out_price": rec["out_price"], "volume": rec.get("volume", 0.0),
            "profit": round(rec["profit"], 2),
            "swap": round(rec["swap"], 2), "commission": round(rec["commission"], 2),
            "net": round(rec["profit"] + rec["swap"] + rec["commission"], 2),
            "source_account": o.get("source_account"),
            "client_direction": o.get("client_direction"),
            "our_direction": o.get("our_direction"),
            "our_lots": o.get("our_lots"), "client_lots": o.get("client_lots"),
            "live_score": o.get("live_score"), "stance": o.get("stance"),
            "expected_usd": o.get("expected_usd"), "fill_price": o.get("fill_price"),
            "created": o.get("created"), "matched": order is not None,
            "in_comment": rec["in_comment"], "out_comment": rec["out_comment"],
            "source_from_comment": _source_of(rec["in_comment"])})
    rows.sort(key=lambda r: r["out_time"], reverse=True)
    return {"available": True, "window_days": window_days, "count": len(rows),
            "trades": rows}


def account_reconcile(days: float = 30.0) -> dict:
    """GROUND TRUTH: the account's actual realized P&L from every venue deal,
    split by our stance (copy/invert/fade from the comment), plus deposits and
    the live balance -- so 'is the strategy really losing' is answered by the
    money, not by any derived metric. MT5 signs our P&L for our ACTUAL
    direction, so a stance's sum needs no negation -- if inverts show a loss,
    the invert leg genuinely lost money, not a sign artefact."""
    config = load_config()
    out = {"available": False, "days": days}
    if config.mode != "live":
        out["reason"] = "paper mode"
        return out
    try:
        import datetime as _dt
        mt5 = _mt5()
        frm = _dt.datetime.now() - _dt.timedelta(days=days)
        reset = stats_reset_at()                     # fresh-balance line in sand
        if reset:
            frm = max(frm, _dt.datetime.fromtimestamp(reset))
        deals = mt5.history_deals_get(frm, _dt.datetime.now()) or []
    except Exception as error:
        out["reason"] = f"{type(error).__name__}: {error}"
        return out
    trading = deposits = 0.0
    n_trade = n_balance = 0
    # Group by POSITION: the P&L sits on the exit deal, the stance comment on the
    # entry deal -- attributing by raw deal comment put 83% of P&L in "other".
    by_pos: dict[int, dict] = {}
    for d in deals:
        dtype = int(getattr(d, "type", 0) or 0)
        pnl = (float(getattr(d, "profit", 0.0) or 0.0)
               + float(getattr(d, "swap", 0.0) or 0.0)
               + float(getattr(d, "commission", 0.0) or 0.0))
        if dtype == 2:                                   # DEAL_TYPE_BALANCE
            deposits += float(getattr(d, "profit", 0.0) or 0.0)
            n_balance += 1
            continue
        trading += pnl
        n_trade += 1
        pid = int(getattr(d, "position_id", 0) or 0)
        if not pid:
            continue
        rec = by_pos.setdefault(pid, {"pnl": 0.0, "stance": None, "closed": False})
        rec["pnl"] += pnl
        entry = int(getattr(d, "entry", 0) or 0)
        if entry == 0:                                   # ENTRY deal carries stance
            comment = str(getattr(d, "comment", "") or "")
            rec["stance"] = ("copy" if comment.startswith(("1c", "zxc")) else
                             "invert" if comment.startswith(("1i", "zxi")) else
                             "fade" if comment.startswith(("2f", "z2f")) else "other")
        if entry in (1, 2, 3):
            rec["closed"] = True
    by_stance: dict[str, dict] = {}
    for rec in by_pos.values():
        if not rec["closed"]:
            continue
        stance = rec["stance"] or "unknown"
        bucket = by_stance.setdefault(stance, {"pnl": 0.0, "deals": 0, "wins": 0})
        bucket["pnl"] += rec["pnl"]
        bucket["deals"] += 1
        bucket["wins"] += 1 if rec["pnl"] > 0 else 0
    snapshot = account_snapshot()
    out.update({
        "available": True,
        "total_trading_realized": round(trading, 2), "trade_deals": n_trade,
        "deposits_withdrawals": round(deposits, 2), "balance_deals": n_balance,
        "balance": snapshot.get("balance"), "equity": snapshot.get("equity"),
        "by_stance": {k: {"pnl": round(v["pnl"], 2), "deals": v["deals"],
                          "win_rate": round(v["wins"] / v["deals"], 3)
                          if v["deals"] else 0.0}
                      for k, v in by_stance.items()}})
    return out


def window_diag(hours: float = 4.0) -> dict:
    """Realized-vs-expected diagnostic over the last `hours`, straight from the
    venue's closed deals joined to our order store. Separate from the cached
    dashboard perf so it can answer 'why did we underperform expected' without a
    40-row cap or a 30-day window. Also de-inflates expected by dd_calibration,
    since stored expected is computed at the (capped) 35x sizing intent."""
    config = load_config()
    out = {"hours": hours, "available": False,
           "dd_calibration": float(getattr(config, "dd_calibration", 1.0) or 1.0),
           "s1": {}, "s2": {}}
    if config.mode != "live":
        out["reason"] = "paper mode"
        return out
    try:
        import datetime as _dt
        mt5 = _mt5()
        frm = _dt.datetime.now() - _dt.timedelta(hours=hours)
        deals = mt5.history_deals_get(frm, _dt.datetime.now()) or []
    except Exception as error:
        out["reason"] = f"{type(error).__name__}: {error}"
        return out
    by_pos: dict[int, dict] = {}
    for d in deals:
        pid = int(getattr(d, "position_id", 0) or 0)
        if not pid:
            continue
        entry = int(getattr(d, "entry", 0) or 0)
        rec = by_pos.setdefault(pid, {"symbol": getattr(d, "symbol", ""), "magic": 0,
                                      "profit": 0.0, "swap": 0.0, "commission": 0.0,
                                      "has_out": False})
        rec["profit"] += float(getattr(d, "profit", 0.0) or 0.0)
        rec["swap"] += float(getattr(d, "swap", 0.0) or 0.0)
        rec["commission"] += float(getattr(d, "commission", 0.0) or 0.0)
        if getattr(d, "magic", 0):
            rec["magic"] = int(d.magic)
        if getattr(d, "symbol", ""):
            rec["symbol"] = d.symbol
        if entry in (1, 2, 3):
            rec["has_out"] = True
    joined = _orders_by_ticket()
    ddc = out["dd_calibration"] or 1.0
    agg = {"s1": [], "s2": []}
    for pid, rec in by_pos.items():
        if not rec["has_out"]:
            continue
        order = joined.get(pid)
        strat = _STRATEGY_MAGIC.get(rec["magic"])
        if strat is None:
            strat = "s2" if (order and order.get("stance") == "fade15") else "s1"
        net = rec["profit"] + rec["swap"] + rec["commission"]
        exp = float(order.get("expected_usd") or 0.0) if order else 0.0
        sign = -1 if strat == "s2" else 1
        if order and order.get("client_direction") and order.get("our_direction"):
            sign = 1 if (int(order["client_direction"])
                         * int(order["our_direction"]) > 0) else -1
        agg[strat].append({"symbol": rec["symbol"], "net": net, "expected": exp,
                           "cost": rec["swap"] + rec["commission"],
                           "stance": (order.get("stance") if order else None),
                           "client_equiv": rec["profit"] * sign})
    for strat, rows in agg.items():
        if not rows:
            out[strat] = {"closed": 0}
            continue
        net = sum(r["net"] for r in rows)
        exp = sum(r["expected"] for r in rows)
        wins = sum(1 for r in rows if r["net"] > 0)
        sym: dict[str, float] = {}
        for r in rows:
            sym[r["symbol"]] = sym.get(r["symbol"], 0.0) + r["net"]
        worst = sorted(sym.items(), key=lambda kv: kv[1])[:6]
        # per (symbol, stance): where exactly the loss sits -- copy vs invert,
        # and whether we are on the right side of the client (client_equiv).
        ss: dict[tuple, dict] = {}
        for r in rows:
            key = (r["symbol"], r.get("stance") or "?")
            b = ss.setdefault(key, {"symbol": r["symbol"], "stance": key[1],
                                    "net": 0.0, "count": 0, "wins": 0,
                                    "client_equiv": 0.0})
            b["net"] += r["net"]; b["count"] += 1
            b["wins"] += 1 if r["net"] > 0 else 0
            b["client_equiv"] += r["client_equiv"]
        worst_ss = sorted(ss.values(), key=lambda x: x["net"])[:8]
        out[strat] = {
            "closed": len(rows), "realized_usd": round(net, 2),
            "expected_usd": round(exp, 2),
            "expected_honest_usd": round(exp / ddc, 2),
            "gap_usd": round(net - exp, 2),
            "gap_honest_usd": round(net - exp / ddc, 2),
            "win_rate": wins / len(rows), "wins": wins,
            "cost_usd": round(sum(r["cost"] for r in rows), 2),
            "worst_symbols": [{"symbol": s, "net": round(v, 2)} for s, v in worst],
            "worst_symbol_stance": [
                {"symbol": b["symbol"], "stance": b["stance"], "net": round(b["net"], 2),
                 "count": b["count"], "wins": b["wins"],
                 "win_rate": round(b["wins"] / b["count"], 3) if b["count"] else 0.0,
                 "client_equiv": round(b["client_equiv"], 2)} for b in worst_ss]}
    out["available"] = True
    return out


def report() -> dict:
    """Everything the tab shows, in one call."""
    config = load_config()
    snapshot = account_snapshot()
    positions = open_book(config)
    orders = recent_orders(200)
    # Each open position carries the model's expectation for it, so the panel
    # can show predicted alongside actual P&L per trade -- the comparison that
    # says whether losses are within model tolerance or a drift signal.
    expected_by_ticket = {o["ticket"]: float(o.get("expected_usd") or 0.0)
                          for o in orders if o.get("ticket") is not None}
    for position in positions:
        if "expected_usd" not in position or not position.get("expected_usd"):
            position["expected_usd"] = expected_by_ticket.get(position.get("ticket"), 0.0)
    filled = [o for o in orders if o["status"] == "filled"]
    expected_open = sum(float(o.get("expected_usd") or 0.0) for o in filled[:len(positions)])
    balance = float(snapshot.get("balance") or 0.0)
    equity = float(snapshot.get("equity") or 0.0)

    # What THIS balance and configuration should produce, from the backtest
    # shape of the exact routing the engine runs. CACHE-ONLY: report() is
    # polled by the tab and must NEVER trigger the heavy compute itself --
    # that storm starved the engine through its own startup. Until the engine
    # has warmed the cache the panel says "warming" rather than inventing
    # numbers from a placeholder balance.
    deployed = _warm_meta().get("deployed")
    if isinstance(deployed, dict) and deployed.get("days"):
        # From the CONSTRAINED replay of the exact deployed logic. The replay
        # ran at ITS dd_target; the account's budget moves with balance, so
        # every figure is RESCALED to the CURRENT risk budget (35% x balance)
        # -- serving the raw replay numbers made the panel read like a
        # different account whenever balance changed.
        budget_now = (equity or balance) * config.risk_budget_fraction
        budget_ref = float(deployed.get("dd_target") or 0.0)
        scale = (budget_now / budget_ref) if (budget_now > 0 and budget_ref > 0) else 1.0
        expected = {
            # When the replay meta lacks a drawdown figure, the CONTRACT
            # itself is the number: max DD = the current risk budget.
            "max_dd_usd": (float(deployed.get("drawdown") or 0.0) * scale
                           or budget_now),
            "daily_usd": float(deployed.get("daily_mean") or 0.0) * scale,
            "daily_std_usd": float(deployed.get("daily_std") or 0.0) * scale,
            "backtest_days": int(deployed.get("days") or 0),
            "scaled_to_budget": round(budget_now, 2),
            "warming": False,
        }
    else:
        stats = _STRATEGY_STATS or {"drawdown": 0.0, "daily_mean": 0.0,
                                    "daily_std": 0.0, "days": 0}
        warming = _STRATEGY_STATS is None or not _STATE.scale_factor
        k = _STATE.scale_factor or 0.0
        expected = {
            "max_dd_usd": k * stats["drawdown"],
            "daily_usd": k * stats["daily_mean"],
            "daily_std_usd": k * stats["daily_std"],
            "backtest_days": stats["days"],
            "warming": warming,
        }
    # TWO VIEWS: the strategies share an account but never a panel. Split by
    # the comment's leading strategy digit (magic 909090 vs 909091 backs it
    # at the venue level).
    def _is_s2(item) -> bool:
        return str(item.get("comment") or "").startswith(("2f", "z2f", "2t"))
    positions_s2 = [p for p in positions if _is_s2(p)]
    positions_s1 = [p for p in positions if not _is_s2(p)]
    orders_s1 = [o for o in orders if o.get("stance") in ("copy", "invert")][:30]
    orders_s2 = [o for o in orders if (o.get("stance") in ("fade15", "tape"))][:30]
    perf = strategy_performance()
    strategy_views = {
        "perf_available": perf.get("available", False),
        "perf_reason": perf.get("reason"),
        "perf_window_days": perf.get("window_days", 30),
        "s1": {"positions": positions_s1, "orders": orders_s1,
               "floating": sum(float(p.get("profit") or 0) for p in positions_s1),
               "enabled": bool(getattr(config, "strategy1", True)),
               "perf": perf.get("s1", _empty_perf())},
        "s2": {"positions": positions_s2, "orders": orders_s2,
               "floating": sum(float(p.get("profit") or 0) for p in positions_s2),
               "enabled": bool(getattr(config, "strategy2", False)),
               "perf": perf.get("s2", _empty_perf())},
    }

    return {
        "expected": expected,
        "replay": deployed if isinstance(deployed, dict) else None,
        "strategies": strategy_views,
        "tape": tape_snapshot(),
        "snapshot": snapshot,
        "positions": positions,
        "paper_realized": round(_PAPER_PNL[0], 2),
        "expected_open_usd": expected_open,
        "expected_total_usd": sum(float(o.get("expected_usd") or 0.0) for o in filled),
        "actual_floating_usd": float(snapshot.get("profit") or 0.0)
                               if snapshot.get("available") else None,
        "risk_budget_usd": (equity or balance) * config.risk_budget_fraction,
        "current_drawdown_usd": max(0.0, balance - equity),
        "orders": orders[:60],
        # Entry -> close prices for every qualifying (copy/invert) trade the
        # engine has closed, shown in its own section under the order log.
        "closed_prices": closed_qualifying(60),
        "stream": stream_health(),
    }

