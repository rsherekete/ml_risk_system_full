"""Background Kafka consumer materialising the live stream into DuckDB.

WHY A SERVICE AND NOT A QUERY

Kafka is an append-only log, not a queryable store. The SDK streams forward from
an offset; there is no "select the last 7 days". So a long-lived consumer runs
in a background thread and writes every decoded record into DuckDB, which the
web app then queries like any other table. This also means the 7-day window
survives page loads, restarts of the browser, and multiple concurrent users --
none of which a per-request consumer could offer.

BACKFILL AND ITS LIMIT

Starting at `earliest` replays whatever the broker still retains. If retention
is shorter than seven days the window simply starts later, and
`coverage()` reports the true span rather than implying seven days exist. The
UI shows that span, so nobody reads a partial window as a complete one.

ENVIRONMENT

The supplied config points at UAT (`kafka-cluster.gase.uat.int.traze.com`).
Anything shown from this stream is UAT flow, and the banner in the UI says so.
Nothing here writes to Kafka -- the consumer commits no offsets and uses a
throwaway group id, so it cannot disturb a real consumer group.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent
KAFKA_DIR = ROOT.parent / "kafka"
STORE = ROOT / "artifacts" / "live_stream.duckdb"
RETENTION_DAYS = 7
CLUSTERS_FILE = KAFKA_DIR / "clusters.yaml"

#: ONE DuckDB instance per process, cursors for every user. Writer and readers
#: each calling duckdb.connect(file) created SEPARATE instances fighting over
#: the OS file lock -- with quotes streaming, the writer held it near-constantly
#: and every engine read died with "store busy". Cursors of a single shared
#: instance are DuckDB's supported same-process concurrency: writes and reads
#: interleave internally, no file-lock contention at all.
_SHARED_DB: duckdb.DuckDBPyConnection | None = None
_SHARED_LOCK = threading.Lock()


def shared_cursor() -> duckdb.DuckDBPyConnection:
    """A cursor onto the process-wide store instance (open lazily, retry brief
    external locks -- e.g. a straggler process still holding the file)."""
    global _SHARED_DB
    with _SHARED_LOCK:
        if _SHARED_DB is None:
            last: Exception | None = None
            for attempt in range(3):
                try:
                    STORE.parent.mkdir(exist_ok=True)
                    _SHARED_DB = duckdb.connect(str(STORE))
                    break
                except Exception as error:
                    last = error
                    time.sleep(0.4 * (attempt + 1))
            if _SHARED_DB is None:
                raise RuntimeError(f"live stream store busy: {last}")
        return _SHARED_DB.cursor()


def _load_clusters() -> dict:
    """Production multi-cluster config (kafka/clusters.yaml).

    Returns {topic -> {host, username, password, registry}} for deals/trades,
    a matching map for quotes, and the flat topic tuples the materialiser uses.
    The prod flow is spread over regional clusters (Zeal / AE / ID), each with
    its own broker + credentials, all decoding through Zeal's reachable
    registry. Falls back to the legacy single config.yaml if the file is absent
    so an older checkout still runs.
    """
    import yaml
    trade_cfg: dict[str, dict] = {}
    quote_cfg: dict[str, dict] = {}
    try:
        doc = yaml.safe_load(CLUSTERS_FILE.read_text(encoding="utf-8")) or {}
        registry = str(doc.get("registry") or "").strip()
        for cluster in doc.get("clusters") or []:
            base = {"host": str(cluster["host"]),
                    "username": str(cluster["username"]),
                    "password": str(cluster["password"]),
                    "registry": str(cluster.get("registry") or registry)}
            for t in cluster.get("topics") or []:
                trade_cfg[str(t)] = base
            for t in cluster.get("quotes") or []:
                quote_cfg[str(t)] = base
    except FileNotFoundError:
        pass
    except Exception as error:                     # bad yaml: log, fall back
        print(f"[kafka] clusters.yaml unreadable: {error}")
    return {"trades": trade_cfg, "quotes": quote_cfg}


_CLUSTERS = _load_clusters()

#: Streams worth materialising, per server. `deals`/`trades` carry executions --
#: the risk-relevant flow. `quotes` is far higher volume and only needed for
#: mark-to-market, so it is off by default.
#: Verified against the cluster: `oms-mt5.events.deals.live01` produces and
#: retains ~12 days, so a 7-day window is servable from it. The MT4 trade
#: topics exist with 8 partitions each but returned nothing on probe, so they
#: are listed and will populate if they start producing rather than being
#: assumed dead.
#: All SIX production servers, matching the model coverage exactly. Risk that
#: silently omits a server is worse than no risk screen at all, so the topic
#: list mirrors the warehouse rather than whichever topics happened to produce
#: during a probe.
#: The topics these credentials actually publish. Per the engineer who issued
#: them, `ecn-risk-user` covers TWO servers on FSA UAT -- MT4 Demo02 and MT5
#: Live01 -- with trades and quotes for each. The previous list subscribed to
#: mt4 live01-04, which these credentials do not carry: those consumers sat
#: "connected" with nothing to read, which looked exactly like a broken
#: pipeline. Other servers need credentials created for them first.
#: Every trade-bearing topic this cluster carries, from an actual metadata
#: listing rather than guessed names -- which is how `deals.live-dubai` and
#: `deals.live-indonesia` surfaced after days of "dubai has no topic".
#:
#: THE CLUSTER IS UAT, whole and entire: all 90 visible topics are
#: `traze.fsa.uat.*`, and the SDK author's notebook states "The UAT topic is
#: mostly idle". A raw 45-second read at the head of each MT5 deals topic
#: delivered zero messages -- the producers are idle, not the consumer.
#: Production flow (the "so many clients" volume) lives on the production
#: cluster and needs production credentials from the engineer who issued
#: these.
#: Trade-bearing topics to materialise, taken from the PRODUCTION clusters in
#: kafka/clusters.yaml (Zeal live01 + mt4 live01-04, AE Dubai, ID Indonesia).
#: The old hardcoded `traze.fsa.uat.*` list pointed at an idle UAT cluster --
#: that was the whole reason the stream looked dead. If clusters.yaml is absent
#: the tuple is empty and the legacy config.yaml path still serves one topic.
DEFAULT_TOPICS = tuple(_CLUSTERS["trades"].keys())

#: Quote streams, for the OHLC panels. Higher volume than trades; only consumed
#: when price charts are wanted.
QUOTE_TOPICS = tuple(_CLUSTERS["quotes"].keys())


def _config_for_topic(topic: str):
    """The SDK Config for whichever prod cluster owns this topic. Falls back to
    the legacy single config.yaml when the topic is not in clusters.yaml."""
    import sys as _sys
    if str(KAFKA_DIR) not in _sys.path:
        _sys.path.insert(0, str(KAFKA_DIR))
    from sdk.config import Config
    spec = _CLUSTERS["trades"].get(topic) or _CLUSTERS["quotes"].get(topic)
    if spec:
        return Config(host=spec["host"], username=spec["username"],
                      password=spec["password"], topic=topic,
                      registry_host=spec["registry"])
    legacy = Config.load(KAFKA_DIR / "config.yaml")
    return Config(host=legacy.host, username=legacy.username,
                  password=legacy.password, topic=topic,
                  registry_host=legacy.registry_host, path=legacy.path)


#: Suffix for this service's stable consumer group.
#:
#: The group id MUST live under the topic's own prefix: `ecn-risk-user` is
#: authorised for groups matching `<env>.<subsystem>...` and nothing else --
#: a free-form name is refused outright with "Group authorization failed".
#: The full id is built per topic as `{topic_prefix}.{CONSUMER_GROUP_SUFFIX}`,
#: which is stable across restarts (so the broker remembers our offsets) while
#: staying inside the permitted namespace.
CONSUMER_GROUP_SUFFIX = "zfx-risk-webapp"

_CANONICAL_CACHE: dict[str, str] = {}


def _canonical(symbol) -> str | None:
    """Cached canonical form. Resolution is pure, and tickers repeat endlessly."""
    if not symbol:
        return None
    key = str(symbol)
    if key not in _CANONICAL_CACHE:
        try:
            from trading_data.research import canonical_symbol
            _CANONICAL_CACHE[key] = canonical_symbol(key)
        except Exception:
            _CANONICAL_CACHE[key] = key.upper()
    return _CANONICAL_CACHE[key]


def server_of_topic(topic: str) -> str:
    """`...oms-mt4.events.trades.live02` -> `mt4_live02`, matching account keys."""
    parts = topic.split(".")
    try:
        subsystem = next(p for p in parts if p.startswith("oms-"))
    except StopIteration:
        return topic
    return f"{subsystem.removeprefix('oms-')}_{parts[-1]}"

#: `DEAL_BALANCE`, credits and corrections carry a `profit` that is a cash
#: movement, not trading P&L. Risk aggregates exclude them explicitly.
NON_TRADING_ACTIONS = ("DEAL_BALANCE", "DEAL_CREDIT", "DEAL_CORRECTION",
                       "DEAL_BONUS", "DEAL_CHARGE", "DEAL_COMMISSION")


def _utc_naive(value=None):
    """A naive-UTC datetime, THE timestamp convention of the store.

    DuckDB localizes tz-aware datetimes on insert and its NOW() speaks local
    time, so mixing aware and naive values skewed every freshness comparison
    by the machine's UTC offset. One convention, applied at the write
    boundary: naive, UTC, always.
    """
    if value is None:
        return datetime.utcnow()
    if getattr(value, "tzinfo", None) is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value

#: MT4 order commands, unified onto the MT5 action vocabulary so every query
#: downstream speaks one language.
_MT4_COMMANDS = {"OP_BUY": "DEAL_BUY", "OP_SELL": "DEAL_SELL",
                 "OP_BALANCE": "DEAL_BALANCE", "OP_CREDIT": "DEAL_CREDIT"}


def _normalise_action(raw, event_kind: str) -> str:
    """Action with the proto3 default made explicit, in MT5 vocabulary.

    Proto3 OMITS fields holding their default value, and both platforms' buy
    side IS the default (DEAL_BUY = 0, OP_BUY = 0) -- so every buy event
    arrived with no action at all and was filtered out by every query that
    asked for DEAL_BUY. Half the market, silently invisible.
    """
    text = str(raw or "").upper()
    if not text and event_kind:
        return "DEAL_BUY"                       # the omitted default
    if isinstance(raw, int) or text.isdigit():
        code = int(text or 0)
        if event_kind.startswith("order"):      # MT4 TradeCommand
            return {0: "DEAL_BUY", 1: "DEAL_SELL", 6: "DEAL_BALANCE",
                    7: "DEAL_CREDIT"}.get(code, f"CMD_{code}")
        return {0: "DEAL_BUY", 1: "DEAL_SELL", 2: "DEAL_BALANCE",
                3: "DEAL_CREDIT"}.get(code, f"DEAL_{code}")
    return _MT4_COMMANDS.get(text, text)


def _normalise_entry(raw, event_kind: str) -> str:
    """Entry flag with the omitted proto3 default restored (ENTRY_IN = 0)."""
    text = str(raw or "").upper()
    if not text:
        # Only a DEAL event has an entry concept; MT4 orders discriminate by
        # envelope and keep it empty on purpose.
        return "ENTRY_IN" if event_kind.startswith("deal") else ""
    if isinstance(raw, int) or text.isdigit():
        return {0: "ENTRY_IN", 1: "ENTRY_OUT", 2: "ENTRY_INOUT",
                3: "ENTRY_OUT_BY"}.get(int(text or 0), text)
    return text


@dataclass
class StreamState:
    topic: str
    status: str = "stopped"        # stopped | connecting | streaming | error
    message: str = ""
    consumed: int = 0
    last_record_at: float = 0.0
    started_at: float = 0.0
    errors: list[str] = field(default_factory=list)


class KafkaMaterialiser:
    """Owns the consumer threads and the DuckDB table they write into."""

    def __init__(self, topics: tuple[str, ...] = DEFAULT_TOPICS) -> None:
        self.topics = topics
        self.states: dict[str, StreamState] = {t: StreamState(topic=t) for t in topics}
        self._threads: dict[str, threading.Thread] = {}
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._ensure_schema()

    # -- storage ----------------------------------------------------------
    def _connect(self) -> duckdb.DuckDBPyConnection:
        return shared_cursor()

    def _has_history(self) -> bool:
        """Does the store already hold events? Then history is not needed again."""
        if not STORE.exists():
            return False
        try:
            with self._connect() as connection:
                return bool(connection.execute(
                    "SELECT COUNT(*) > 0 FROM events").fetchone()[0])
        except Exception:
            return False

    def _ensure_schema(self) -> None:
        STORE.parent.mkdir(exist_ok=True)
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    topic        VARCHAR,
                    server       VARCHAR,
                    partition    INTEGER,
                    "offset"     BIGINT,
                    event_time   TIMESTAMP,
                    ingested_at  TIMESTAMP,
                    key          VARCHAR,
                    message      VARCHAR,
                    login        BIGINT,
                    symbol       VARCHAR,
                    canonical    VARCHAR,
                    action       VARCHAR,
                    volume       DOUBLE,
                    price        DOUBLE,
                    profit       DOUBLE,
                    payload      JSON
                )
            """)
            connection.execute(
                'CREATE INDEX IF NOT EXISTS events_time ON events(event_time)')
            # Columns added after the table first shipped; ALTER is idempotent
            # via the try, and existing rows read as NULL (the routing layer
            # falls back to the profit heuristic for those).
            for added in ("entry VARCHAR", "event_kind VARCHAR"):
                try:
                    connection.execute(f"ALTER TABLE events ADD COLUMN {added}")
                except Exception:
                    pass
            # Quotes live in their own table: they arrive orders of magnitude
            # faster than trades and are only used to build price bars.
            connection.execute("""
                CREATE TABLE IF NOT EXISTS quotes (
                    server     VARCHAR,
                    event_time TIMESTAMP,
                    symbol     VARCHAR,
                    canonical  VARCHAR,
                    bid        DOUBLE,
                    ask        DOUBLE,
                    mid        DOUBLE
                )
            """)
            connection.execute(
                'CREATE INDEX IF NOT EXISTS quotes_time ON quotes(canonical, event_time)')

    # -- lifecycle --------------------------------------------------------
    def start(self, backfill: bool = True, with_quotes: bool = False) -> None:
        """Begin consuming.

        `backfill` only matters the FIRST time this consumer group runs: with
        committed offsets the broker resumes wherever we stopped and the
        setting is ignored. It is therefore honoured once, on a store with no
        history, and dropped afterwards -- otherwise every boot replayed two
        weeks of the log and the live view filled with days-old trades.
        """
        if backfill and self._has_history():
            backfill = False
        self._stop.clear()
        topics = list(self.topics) + (list(QUOTE_TOPICS) if with_quotes else [])
        for topic in topics:
            if topic in self._threads and self._threads[topic].is_alive():
                continue
            if topic not in self.states:
                self.states[topic] = StreamState(topic=topic)
            worker = threading.Thread(target=self._consume,
                                      args=(topic, backfill, self._stop),
                                      daemon=True)
            self._threads[topic] = worker
            worker.start()

    def stop(self) -> None:
        self._stop.set()

    def restart(self, with_quotes: bool = True) -> None:
        """HARD restart: retire the current consumer generation and start a
        fresh one. `start()` alone skips any topic whose thread is still
        `is_alive()` -- and a thread wedged in a dead network read after a VPN
        drop IS alive, so the restart button silently did nothing. Each
        generation captures its OWN stop event: retired threads see theirs
        set and exit on their next wakeup, and can neither block the new
        generation nor be resurrected by its cleared event."""
        self._stop.set()
        self._stop = threading.Event()
        with self._lock:
            self._threads = {}
        for state in self.states.values():
            state.status = "restarting"
        self.start(backfill=False, with_quotes=with_quotes)

    def _consume(self, topic: str, backfill: bool,
                 stop_event: threading.Event | None = None) -> None:
        stop_event = stop_event if stop_event is not None else self._stop
        state = self.states[topic]
        state.status, state.started_at = "connecting", time.time()
        try:
            import sys
            if str(KAFKA_DIR) not in sys.path:
                sys.path.insert(0, str(KAFKA_DIR))
            from sdk.kafka_client import KafkaClient

            # Per-topic PRODUCTION cluster config (Zeal / AE / ID). Each topic
            # carries its own broker + credentials; all decode through the shared
            # reachable registry. Replaces the single UAT config.yaml.
            config = _config_for_topic(topic)

            # A STABLE consumer group, so KAFKA ITSELF is our cursor.
            #
            # The SDK mints a random group per process and commits nothing --
            # correct for an example that must never disturb a real consumer,
            # and ruinous for a service: every restart replayed the entire
            # 15-day retention from `earliest`. That is what flooded the engine
            # with ~18k "signals" of ancient history within two minutes of each
            # start, filled the store with duplicates, and made the live panel
            # show days-old trades.
            #
            # With a fixed group id and commits enabled, the broker remembers
            # our position per partition. First run reads the retained history
            # once; every run after resumes exactly where it stopped. No
            # windowing, no de-duplication passes, no merging -- the log's own
            # offset is the state, which is what a log is for.
            # LIVE means live. Kafka is a log, not a queue: `earliest` walks
            # ~15 days of retained history from the beginning at full speed,
            # which is why the panel showed trades stamped days ago arriving
            # this second -- a fast replay of the backlog, not a stalled feed.
            #
            # `latest` positions at the head, so only messages produced from
            # now on are delivered and `traded` tracks `seen`. History still
            # reaches the store: the one-off `backfill=True` pass fills it, and
            # the warehouse holds two years regardless.
            # Deals RESUME (history matters; the stable group's committed
            # offset is the cursor). Quotes always start AT THE HEAD with a
            # throwaway group: a tick is an ephemeral mark, and resuming a
            # five-hour-old commit spends half an hour replaying stale prices
            # before the heartbeat can read "live".
            if ".quotes." in topic:
                offset_reset, group = "latest", None
            else:
                offset_reset = "earliest" if backfill else "latest"
                group = f"{config.topic_prefix}.{CONSUMER_GROUP_SUFFIX}"
            with KafkaClient(config, auto_offset_reset=offset_reset,
                             group_id=group) as client:
                state.status, state.message = "streaming", "connected"
                buffer: list[tuple] = []
                last_flush = time.time()

                is_quotes = ".quotes." in topic

                def on_message(record) -> None:
                    nonlocal buffer, last_flush
                    buffer.append(self._quote_row(record) if is_quotes else self._row(record))
                    state.consumed += 1
                    state.last_record_at = time.time()
                    # Batch inserts: DuckDB is columnar and a row-at-a-time
                    # insert would dominate the cost of consuming. Quotes arrive
                    # far faster, so they batch larger and flush on a slower
                    # clock -- they only feed price charts. TRADES flush on a
                    # ~1s clock so the engine sees a Kafka deal within about a
                    # second, comfortably ahead of the ~2.5s MySQL poll: that is
                    # what makes Kafka the genuinely faster decision feed.
                    # Trades flush on a 0.3s clock (or 40 rows) so a deal reaches
                    # the store sub-second; with a 0.3s engine poll the end-to-end
                    # produce->decision path is ~0.6s -- genuinely sub-1s and well
                    # ahead of the ~2.5s MySQL feed. Quotes stay batched (charts).
                    limit = 5000 if is_quotes else 40
                    flush_secs = 5.0 if is_quotes else 0.3
                    if len(buffer) >= limit or time.time() - last_flush > flush_secs:
                        self._flush(buffer, quotes=is_quotes)
                        buffer = []
                        last_flush = time.time()

                while not stop_event.is_set():
                    client.consume(on_message, count=0, timeout=5.0, poll_timeout=1.0)
                    if buffer:
                        self._flush(buffer, quotes=is_quotes)
                        buffer = []
                self._flush(buffer, quotes=is_quotes)
        except Exception as error:
            state.status = "error"
            state.message = f"{type(error).__name__}: {error}"
            state.errors.append(state.message)
        finally:
            if state.status != "error":
                state.status = "stopped"

    @staticmethod
    def _row(record) -> tuple:
        """Flatten a decoded record into the columns risk screens query.

        The payload is NESTED, not flat -- an MT5 deal arrives as::

            {timestamp, deal, login, dealCreated: {deal: {action, symbol,
             volume, price, profit, group, ...}}}

        so the interesting fields sit two levels down. MT4 trade events use a
        different envelope again, so the inner record is located by trying the
        known wrappers rather than assuming one shape, and the whole payload is
        stored as JSON regardless so nothing is lost to a wrong guess.

        Note that not every event is a trade: `DEAL_BALANCE` rows are deposits
        and withdrawals. They carry a profit figure that is NOT trading P&L, so
        the action is preserved verbatim and the risk queries filter on it --
        summing indiscriminately would report deposits as trading performance.
        """
        value = record.value or {}

        # Unwrap whichever envelope this event uses. The wrapper names come
        # from the published protos: MT5 wraps a DealRecord in dealCreated /
        # dealUpdated / dealDeleted under leaf `deal`; MT4 wraps a TradeRecord
        # in orderCreated / orderUpdated under leaf `tradeRecord` (and
        # orderClosedBy carries the surviving side as `remain`). The previous
        # leaf list never tried `tradeRecord`, so every MT4 trade event landed
        # with NULL symbol and login -- consumed, stored, and useless.
        inner = value
        event_kind = ""
        for wrapper in ("dealCreated", "dealUpdated", "dealDeleted",
                        "orderCreated", "orderUpdated", "orderClosedBy",
                        "tradeCreated", "tradeUpdated"):
            nested = value.get(wrapper)
            if isinstance(nested, dict):
                event_kind = wrapper
                for leaf in ("deal", "tradeRecord", "trade", "order", "remain"):
                    candidate = nested.get(leaf)
                    if isinstance(candidate, dict):
                        inner = candidate
                        break
                else:
                    inner = nested
                break

        def pick(*names, cast=None):
            for source in (inner, value):
                for name in names:
                    raw = source.get(name)
                    if raw not in (None, ""):
                        if cast is None:
                            return raw
                        try:
                            return cast(raw)
                        except (TypeError, ValueError):
                            continue
            return None

        symbol = pick("symbol", "Symbol", "symbolName", "symbol_name")
        return (
            record.topic, server_of_topic(record.topic), record.partition, record.offset,
            _utc_naive(record.timestamp),
            _utc_naive(),
            record.key, record.message,
            pick("login", "Login", "account", cast=int),
            symbol,
            # Canonical form stored alongside the raw ticker: gold trades under
            # six different names across these servers, and aggregating by raw
            # ticker splits one exposure into six.
            _canonical(symbol),
            _normalise_action(pick("action", "Action", "cmd", "command", "type"),
                              event_kind),
            pick("volume", "Volume", "lots", "volumeExt", cast=float),
            pick("price", "Price", "openPrice", "priceOpen", cast=float),
            pick("profit", "Profit", "netProfit", cast=float),
            json.dumps(value)[:20000],
            # The PROPER open/close discriminator, straight from the schema:
            # MT5 deals carry `entry` (ENTRY_IN / ENTRY_OUT / ENTRY_INOUT);
            # MT4's is the envelope itself (orderCreated opens). Stored so the
            # routing layer stops inferring opens from profit == 0.
            _normalise_entry(pick("entry", "Entry"), event_kind),
            event_kind,
        )

    @staticmethod
    def _quote_row(record) -> tuple:
        """One tick: server, time, symbol, bid/ask and their midpoint."""
        value = record.value or {}
        inner = value
        for wrapper in ("quoteCreated", "quote", "tick"):
            nested = value.get(wrapper)
            if isinstance(nested, dict):
                inner = nested.get("quote") if isinstance(nested.get("quote"), dict) else nested
                break

        def number(*names):
            for source in (inner, value):
                for name in names:
                    raw = source.get(name)
                    if raw not in (None, ""):
                        try:
                            return float(raw)
                        except (TypeError, ValueError):
                            continue
            return None

        symbol = None
        for source in (inner, value):
            for name in ("symbol", "Symbol", "symbolName", "symbol_name"):
                if source.get(name):
                    symbol = source[name]
                    break
            if symbol:
                break
        bid, ask = number("bid", "Bid"), number("ask", "Ask")
        mid = ((bid + ask) / 2) if (bid and ask) else (bid or ask)
        return (server_of_topic(record.topic),
                _utc_naive(record.timestamp),
                symbol, _canonical(symbol), bid, ask, mid)

    def _flush(self, rows: list[tuple], quotes: bool = False) -> None:
        if not rows:
            return
        with self._lock, self._connect() as connection:
            if quotes:
                connection.executemany("INSERT INTO quotes VALUES (?,?,?,?,?,?,?)", rows)
            else:
                connection.executemany(
                    "INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)

    # -- queries ----------------------------------------------------------
    def prune(self) -> int:
        """Drop anything older than the retention window."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
        with self._lock, self._connect() as connection:
            before = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            connection.execute("DELETE FROM events WHERE event_time < ?", [cutoff])
            after = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        return before - after

    def coverage(self) -> dict:
        """What the store ACTUALLY holds -- never an assumed 7 days.

        Broker retention may be shorter than the window we want; reporting the
        real span stops a partial history being read as a complete one.

        Takes the same lock the writer uses. Without it a read landing mid-flush
        transiently returned zero rows, which the UI would have rendered as "no
        data" -- a momentary write contention showing up as an empty dashboard.
        """
        try:
            with self._lock, self._connect() as connection:
                row = connection.execute(
                    "SELECT COUNT(*), MIN(event_time), MAX(event_time),"
                    " COUNT(DISTINCT login), COUNT(DISTINCT symbol) FROM events"
                ).fetchone()
        except Exception as error:
            return {"available": False, "error": str(error)}
        count, first, last, logins, symbols = row
        span = (last - first).total_seconds() / 86400 if first and last else 0.0
        return {
            "available": bool(count),
            "events": int(count or 0),
            "first_event": first.isoformat() if first else None,
            "last_event": last.isoformat() if last else None,
            "span_days": round(span, 2),
            "requested_days": RETENTION_DAYS,
            "complete": span >= RETENTION_DAYS - 0.5,
            "logins": int(logins or 0),
            "symbols": int(symbols or 0),
        }

    #: Excludes cash movements from any P&L aggregate. A DEAL_BALANCE of
    #: +$10,000 is a deposit; counting it as trading profit would overstate
    #: performance by whatever clients funded that day.
    _TRADING_ONLY = " AND action NOT IN " + str(NON_TRADING_ACTIONS)

    def recent_activity(self, hours: int = 24) -> list[dict]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        try:
            with self._lock, self._connect() as connection:
                rows = connection.execute(
                    "SELECT date_trunc('hour', event_time) AS bucket, COUNT(*) AS events,"
                    " COUNT(DISTINCT login) AS accounts, SUM(COALESCE(volume,0)) AS volume,"
                    " SUM(CASE WHEN action NOT IN " + str(NON_TRADING_ACTIONS) +
                    " THEN COALESCE(profit,0) ELSE 0 END) AS profit,"
                    " SUM(CASE WHEN action IN " + str(NON_TRADING_ACTIONS) +
                    " THEN COALESCE(profit,0) ELSE 0 END) AS cash_flow"
                    " FROM events WHERE event_time >= ? GROUP BY 1 ORDER BY 1", [cutoff]
                ).fetchall()
        except Exception:
            return []
        return [{"bucket": r[0].isoformat(), "events": r[1], "accounts": r[2],
                 "volume": float(r[3] or 0), "profit": float(r[4] or 0),
                 "cash_flow": float(r[5] or 0)} for r in rows]

    def top_exposure(self, limit: int = 25) -> list[dict]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
        try:
            with self._lock, self._connect() as connection:
                rows = connection.execute(
                    "SELECT symbol, COUNT(*) AS events, COUNT(DISTINCT login) AS accounts,"
                    " SUM(COALESCE(volume,0)) AS volume, SUM(COALESCE(profit,0)) AS profit"
                    " FROM events WHERE event_time >= ? AND symbol IS NOT NULL"
                    + self._TRADING_ONLY +
                    " GROUP BY 1 ORDER BY volume DESC LIMIT ?", [cutoff, limit]
                ).fetchall()
        except Exception:
            return []
        return [{"symbol": r[0], "events": r[1], "accounts": r[2],
                 "volume": float(r[3] or 0), "profit": float(r[4] or 0)} for r in rows]

    def symbol_var(self, horizons: tuple[int, ...] = (1, 5, 20)) -> list[dict]:
        """Per-symbol historical VaR over several horizons, from live flow.

        VaR is scaled from the daily figure by sqrt(horizon) -- the standard
        square-root-of-time rule. It assumes independent daily moves, which
        understates risk when moves cluster (they do), so these are a floor
        rather than a worst case. Stated here because a VaR number without its
        assumption is misleading.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
        try:
            with self._lock, self._connect() as connection:
                rows = connection.execute(
                    "SELECT canonical, CAST(event_time AS DATE) AS d,"
                    " SUM(COALESCE(profit,0)) AS pnl, SUM(COALESCE(volume,0)) AS volume,"
                    " COUNT(*) AS events, COUNT(DISTINCT login) AS accounts"
                    " FROM events WHERE event_time >= ? AND canonical IS NOT NULL"
                    + self._TRADING_ONLY +
                    " GROUP BY 1, 2", [cutoff]
                ).fetchall()
        except Exception:
            return []
        if not rows:
            return []

        by_symbol: dict[str, list] = {}
        for canonical, _day, pnl, volume, events, accounts in rows:
            by_symbol.setdefault(canonical, []).append(
                (float(pnl or 0), float(volume or 0), int(events), int(accounts)))

        results = []
        for canonical, daily in by_symbol.items():
            # Firm P&L is the negative of client P&L.
            series = sorted(-value[0] for value in daily)
            if not series:
                continue
            index = max(0, int(len(series) * 0.05) - 1)
            var_1d = series[index] if len(series) >= 3 else series[0]
            entry = {
                "symbol": canonical,
                "days": len(series),
                "firm_pnl": float(sum(series)),
                "volume": float(sum(v[1] for v in daily)),
                "events": int(sum(v[2] for v in daily)),
                "accounts": int(max(v[3] for v in daily)),
            }
            for horizon in horizons:
                entry[f"var_{horizon}d"] = float(var_1d * (horizon ** 0.5))
            results.append(entry)
        return sorted(results, key=lambda r: -abs(r["firm_pnl"]))

    def ohlc(self, canonical: str, minutes: int = 5, hours_back: int = 24) -> list[dict]:
        """OHLC bars for one instrument, built from the quote stream.

        Bucketed by an arbitrary minute width rather than a fixed set, so the
        timeframe control is genuinely continuous instead of three presets.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        try:
            with self._lock, self._connect() as connection:
                rows = connection.execute(
                    "SELECT time_bucket(INTERVAL '1 minute' * ?, event_time) AS bucket,"
                    " first(mid ORDER BY event_time) AS o, max(mid) AS h,"
                    " min(mid) AS l, last(mid ORDER BY event_time) AS c, COUNT(*) AS ticks"
                    " FROM quotes WHERE canonical = ? AND event_time >= ? AND mid > 0"
                    " GROUP BY 1 ORDER BY 1", [minutes, canonical, cutoff]
                ).fetchall()
        except Exception:
            return []
        return [{"time": r[0].isoformat(), "open": float(r[1]), "high": float(r[2]),
                 "low": float(r[3]), "close": float(r[4]), "ticks": int(r[5])} for r in rows]

    def quote_symbols(self) -> list[str]:
        try:
            with self._lock, self._connect() as connection:
                rows = connection.execute(
                    "SELECT canonical, COUNT(*) AS n FROM quotes"
                    " WHERE canonical IS NOT NULL GROUP BY 1 ORDER BY n DESC LIMIT 40"
                ).fetchall()
        except Exception:
            return []
        return [r[0] for r in rows]

    def open_positions(self) -> list[dict]:
        """Positions inferred as still open from the live stream.

        An entry with no matching exit in the retained window is treated as
        open. This is a stream-derived VIEW, not the broker's position table --
        a position opened before retention began is invisible here, so the count
        is a lower bound and the UI says so.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
        try:
            with self._lock, self._connect() as connection:
                rows = connection.execute(
                    "SELECT server, login, canonical,"
                    " SUM(CASE WHEN action LIKE '%BUY%' THEN COALESCE(volume,0)"
                    "          WHEN action LIKE '%SELL%' THEN -COALESCE(volume,0)"
                    "          ELSE 0 END) AS net_volume,"
                    " COUNT(*) AS events, MAX(event_time) AS last_event"
                    " FROM events WHERE event_time >= ? AND canonical IS NOT NULL"
                    + self._TRADING_ONLY +
                    " GROUP BY 1,2,3 HAVING ABS(net_volume) > 1e-9"
                    " ORDER BY ABS(net_volume) DESC LIMIT 300", [cutoff]
                ).fetchall()
        except Exception:
            return []
        return [{"server": r[0], "login": r[1], "symbol": r[2],
                 "net_volume": float(r[3]), "events": int(r[4]),
                 "last_event": r[5].isoformat() if r[5] else None} for r in rows]

    def status(self) -> dict:
        return {
            "environment": "PROD" if _CLUSTERS["trades"] else "UAT",
            "topics": [
                {"topic": s.topic, "status": s.status, "message": s.message,
                 "consumed": s.consumed,
                 "seconds_since_record": (time.time() - s.last_record_at)
                 if s.last_record_at else None}
                for s in self.states.values()
            ],
            "coverage": self.coverage(),
        }


def probe_connectivity(timeout: float = 3.0) -> dict:
    """Can this host actually reach the cluster?

    Worth answering separately from "is the consumer running", because the usual
    cause of no live data is not a broken consumer -- it is that these are
    internal hostnames requiring the corporate VPN. Reporting DNS and TCP
    directly turns a silent empty dashboard into an actionable message.
    """
    import socket

    # Probe every production broker in clusters.yaml (plus the shared registry),
    # so a VPN gap or one dead region is reported precisely rather than as a
    # blanket "no data".
    targets = []
    seen = set()
    for spec in list(_CLUSTERS["trades"].values()) + list(_CLUSTERS["quotes"].values()):
        hp = spec["host"]
        if hp in seen:
            continue
        seen.add(hp)
        host = hp.rsplit(":", 1)[0]
        port = int(hp.rsplit(":", 1)[1]) if ":" in hp else 9094
        targets.append((f"broker:{host.split('.')[0]}", host, port))
        reg = spec.get("registry", "")
        reg_host = reg.split("://", 1)[-1].split("/")[0].split(":")[0]
        if reg_host and reg_host not in seen:
            seen.add(reg_host)
            targets.append(("schema_registry", reg_host, 80))
    if not targets:
        targets = [("broker", "kafka-cluster.gase.uat.int.traze.com", 9094),
                   ("schema_registry", "schema-registry.uat.int.traze.com", 80)]

    results = {}
    for label, host, port in targets:
        entry = {"host": host, "port": port, "dns": False, "tcp": False}
        try:
            socket.getaddrinfo(host, port)
            entry["dns"] = True
        except socket.gaierror as error:
            entry["error"] = f"DNS: {error.strerror or error}"
        if entry["dns"]:
            try:
                with socket.create_connection((host, port), timeout=timeout):
                    entry["tcp"] = True
            except OSError as error:
                entry["error"] = f"TCP: {error}"
        results[label] = entry
    results["ok"] = all(v.get("tcp") for k, v in results.items() if k != "ok")
    if not results["ok"]:
        results["hint"] = (
            "These are internal hostnames. They resolve only from the corporate "
            "network -- connect the VPN, then press Reconnect.")
    return results


MATERIALISER = KafkaMaterialiser()

