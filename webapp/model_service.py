"""Model configuration, background training, and cached scored artefacts.

THE CENTRAL DESIGN RULE: training and viewing are separated.

Filtering by day, account or symbol reads a cached artefact and never refits
anything. Changing a hyper-parameter marks the configuration DIRTY and leaves
the old results on screen until someone presses Retrain. Refitting on every
control change would make the product unusable -- the client model takes ~45s
and the trade-level model takes closer to an hour -- and would also quietly
change the numbers under a user who only meant to narrow a date range.

Each artefact records the exact configuration that produced it, so a screen can
always answer "what settings am I looking at" and show whether they still match
the current ones.
"""

from __future__ import annotations

import gc
import json
import hashlib
import sqlite3
import threading
import time
import traceback
from dataclasses import dataclass, asdict, field, fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def _data_dir() -> Path:
    """Folder holding the session-scale data files: the 90-day BigQuery
    extract, the markout panels, the quant feature cache and the exit-path
    study. NOTEBOOK_DATA_DIR wins; then the session scratchpad the files were
    first built in on the original machine; else webapp/artifacts/ad, which is
    where `tools/github_data_assets.py fetch` places them on a fresh clone."""
    import os
    env = os.environ.get("NOTEBOOK_DATA_DIR", "").strip()
    if env:
        return Path(env)
    legacy = Path(r"C:\Users\ROYVIV~1\AppData\Local\Temp\claude"
                  r"\c--Users-RoyVivasi-Documents-notebook"
                  r"\9951e7b4-740a-496a-a92d-689972573193\scratchpad")
    if legacy.is_dir():
        return legacy
    return ARTIFACTS / "ad"


SCRATCH = _data_dir()

VIEW_TRADING = "trading"
VIEW_QUANT = "quant"


@dataclass
class TrainingConfig:
    """Everything a user may tune, with the defaults that produced our results."""

    # --- labelling -------------------------------------------------------
    #: Forward horizon in ACTIVE days (days the account actually traded).
    horizon_active_days: int = 5
    #: How good the forward window must be, in units of the account's OWN P&L
    #: volatility, to count as a positive. 0.5 was decisively better than 0.0 --
    #: the signal lives in materially good days, not marginally good ones.
    sigma_threshold: float = 0.5

    # --- history ---------------------------------------------------------
    #: Days of history to train on. Two years rather than ninety days: the
    #: 90-day sample gave the walk-forward only ~70 usable days after warm-up,
    #: which is thin for a model with 174 features and made every result
    #: vulnerable to the particular quarter it landed in.
    history_days: int = 730

    #: Count a day as active when the account was EXPOSED -- opened, closed,
    #: carried a position across it, or holds one now -- rather than only when
    #: it closed a trade. A position held Monday to Friday is risk all week.
    use_exposure_days: bool = True

    # --- walk-forward ----------------------------------------------------
    min_train_days: int = 60
    #: Weekly. On two years of history the marginal value of refitting more
    #: often is small, and a weekly cadence is what a desk would actually
    #: operate.
    refit_cadence_days: int = 7

    #: Rolling training window, in days. An EXPANDING window over two years
    #: means the last refits fit 300 trees on ~4.9M rows, and with ~104 weekly
    #: refits the walk-forward projected to 13 hours. A rolling window still
    #: walks across the full history -- every day is still scored out of sample
    #: -- it just stops each refit re-reading years of stale behaviour. Clients
    #: change; a year is generous for this problem.
    #: Set to 0 for the expanding window.
    max_train_days: int = 365

    #: Continue from the previous booster instead of fitting from scratch.
    #: Measured, not assumed: an earlier daily-refit test showed warm start
    #: COSTING 0.16 AUC. Weekly refits over two years are a different regime,
    #: so it is offered and compared rather than switched on by default.
    warm_start: bool = False

    # --- model -----------------------------------------------------------
    #: Capacity raised with the data. 100 trees and 31 leaves were tuned when a
    #: fit saw ~400k rows; two years is roughly eight times that, and the same
    #: capacity would now underfit.
    n_estimators: int = 300
    num_leaves: int = 63
    learning_rate: float = 0.05
    subsample: float = 0.7
    colsample_bytree: float = 0.7
    min_child_samples: int = 50

    # --- routing policy --------------------------------------------------
    #: Ceiling on trade-level training rows. Two years is ~145 million trades,
    #: which at 25 float32 features is ~14.5 GB before a model sees it. Sampling
    #: is stratified BY DAY so every period stays represented -- taking the most
    #: recent N rows instead would silently reduce the history to a few months
    #: while still calling itself a two-year model.
    max_trade_rows: int = 25_000_000

    #: Fraction of account-days (or trades) to hedge. Applied at VIEW time from
    #: the cached scores, so moving this does not require a refit.
    hedge_fraction: float = 0.05
    #: Alternative to a quota: hedge everything above this probability.
    probability_threshold: float = 0.0

    #: Weight training rows by how much the account actually moves, so the loss
    #: reflects dollars rather than row counts. Off reproduces the uniform
    #: objective, which measurably under-fits the largest size decile -- 83% of
    #: firm P&L, and the model's worst AUC.
    economic_weights: bool = True

    #: Hard cap on rows per individual walk-forward fit. Each refit previously
    #: trained on the whole rolling window (up to 25M rows); at 63 leaves the
    #: model saturates around a few million, so the rest was pure compute. Zero
    #: disables the cap.
    max_fit_rows: int = 3_000_000

    def fingerprint(self) -> str:
        """Hash of the settings that actually affect the fitted model.

        `hedge_fraction` and `probability_threshold` are deliberately excluded:
        they are applied to cached scores at view time, so changing them must
        NOT invalidate an artefact.
        """
        material = {f.name: getattr(self, f.name) for f in fields(self)
                    if f.name not in {"hedge_fraction", "probability_threshold"}}
        blob = json.dumps(material, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class JobState:
    status: str = "idle"          # idle | running | done | error
    message: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    progress: float = 0.0
    log: list[str] = field(default_factory=list)
    #: Structured per-day walk-forward records for the live Strategy Lab
    #: dashboard -- {day, date, cum, dd, day_pnl, rows, auc}. Text goes to
    #: `log`; this is the machine-readable stream the chart polls.
    metrics: list[dict] = field(default_factory=list)


_JOBS: dict[str, JobState] = {}
_LOCK = threading.Lock()


def job_state(view: str) -> JobState:
    with _LOCK:
        return _JOBS.setdefault(view, JobState())


def _log(view: str, message: str, progress: float | None = None) -> None:
    with _LOCK:
        state = _JOBS.setdefault(view, JobState())
        state.log.append(f"{time.strftime('%H:%M:%S')}  {message}")
        state.log = state.log[-40:]
        state.message = message
        if progress is not None:
            state.progress = progress


def _emit_metric(view: str, record: dict) -> None:
    """Append one structured walk-forward record for the live Lab chart."""
    with _LOCK:
        state = _JOBS.setdefault(view, JobState())
        state.metrics.append(record)
        state.metrics = state.metrics[-1000:]


def _reset_metrics(view: str) -> None:
    with _LOCK:
        _JOBS.setdefault(view, JobState()).metrics = []


# ---------------------------------------------------------------------------
# configuration persistence
# ---------------------------------------------------------------------------
def _db() -> sqlite3.Connection:
    connection = sqlite3.connect(ROOT / "app.db", check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute(
        "CREATE TABLE IF NOT EXISTS model_config ("
        " view TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at REAL NOT NULL)"
    )
    return connection


def load_config(view: str) -> TrainingConfig:
    with _db() as connection:
        row = connection.execute("SELECT payload FROM model_config WHERE view = ?", (view,)).fetchone()
    if row is None:
        return TrainingConfig()
    stored = json.loads(row["payload"])
    valid = {f.name for f in fields(TrainingConfig)}
    return TrainingConfig(**{k: v for k, v in stored.items() if k in valid})


def save_config(view: str, config: TrainingConfig) -> None:
    with _db() as connection:
        connection.execute(
            "INSERT INTO model_config (view, payload, updated_at) VALUES (?, ?, ?)"
            " ON CONFLICT(view) DO UPDATE SET payload = excluded.payload,"
            " updated_at = excluded.updated_at",
            (view, json.dumps(asdict(config)), time.time()),
        )


# ---------------------------------------------------------------------------
# artefacts
# ---------------------------------------------------------------------------
def artifact_paths(view: str) -> tuple[Path, Path]:
    return ARTIFACTS / f"{view}_scores.parquet", ARTIFACTS / f"{view}_meta.json"


def artifact_meta(view: str) -> dict | None:
    _, meta_path = artifact_paths(view)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


HISTORY = ARTIFACTS / "history"


def model_history(view: str, limit: int = 12) -> list[dict]:
    """Archived completed runs, newest first -- what the Strategy Lab compares
    the current model against. Each entry mirrors the live meta shape."""
    runs: list[dict] = []
    if not HISTORY.exists():
        return runs
    for run_dir in sorted(HISTORY.glob(f"{view}_*"), reverse=True):
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        note_path = run_dir / "note.txt"
        runs.append({
            "dir": run_dir.name, "trained_at": meta.get("trained_at"),
            "rows": meta.get("rows"), "config": meta.get("config") or {},
            "metrics": meta.get("metrics") or {},
            "note": note_path.read_text(encoding="utf-8") if note_path.exists() else "",
        })
    return runs[:limit]


def archive_current(view: str, note: str = "") -> Path | None:
    """Copy the CURRENT meta + model artifacts into history/ so a new run's
    artifacts never silently erase the model they replace. Idempotent per
    trained_at; called before a finished run writes its artifacts."""
    import shutil

    meta_path = artifact_paths(view)[1]
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        stamp = time.strftime("%Y%m%d_%H%M%S",
                              time.localtime(float(meta.get("trained_at") or time.time())))
    except Exception:
        return None
    dest = HISTORY / f"{view}_{stamp}"
    if (dest / "meta.json").exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(meta_path, dest / "meta.json")
    for path in ARTIFACTS.glob(f"{view}_*.txt"):      # every model/feature/symbol file
        if path.is_file():
            shutil.copy2(path, dest / path.name)
    for extra in ("entry_model.txt", f"{view}_calibrators.json"):
        if (ARTIFACTS / extra).exists():
            shutil.copy2(ARTIFACTS / extra, dest / extra)
    if note:
        (dest / "note.txt").write_text(note, encoding="utf-8")
    return dest


def is_stale(view: str) -> bool:
    """True when the live configuration no longer matches the cached artefact.

    Drives the "settings changed -- Retrain to apply" banner. The screen keeps
    showing the old results, which remain valid for the settings that made
    them, rather than blanking or silently refitting.
    """
    meta = artifact_meta(view)
    return meta is None or meta.get("fingerprint") != load_config(view).fingerprint()


#: Loaded artefacts, keyed by path and modification time. Re-reading the parquet
#: on every request cost ~20s for the Quant artefact (19.5M rows) and made the
#: app feel broken. Keying on mtime means a retrain is picked up automatically
#: without a stale frame ever being served.
_FRAME_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
#: Serialises loading. Without it, a request arriving while the startup warm-up
#: is still reading the artefact does the entire 19.5M-row load a second time
#: rather than waiting for the copy already in flight.
_LOAD_LOCK = threading.Lock()


def load_scores(view: str) -> pd.DataFrame | None:
    scores_path, _ = artifact_paths(view)
    if not scores_path.exists():
        return None
    stamp = scores_path.stat().st_mtime
    cached = _FRAME_CACHE.get(view)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    with _LOAD_LOCK:
        # Re-check: another thread may have finished while we waited.
        cached = _FRAME_CACHE.get(view)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        try:
            frame = pd.read_parquet(scores_path)
            # Normalise the day column once -- every screen would otherwise
            # re-derive it, and on 19.5M rows that is itself expensive.
            frame["day"] = pd.to_datetime(frame["day"])
            # Hold the frame to the configured window. Artefacts carry a tail of
            # all-zero padding rows (pnl, score and sigma all 0, every feature
            # NaN) for accounts whose history predates the window -- 253k rows
            # reaching back to 2021 against a 730-day request. They add nothing
            # to P&L but stretch the equity curve with 1,233 empty leading days
            # and make "days of history" read as 1,963 on screens that quote it.
            horizon = getattr(load_config(view), "history_days", 0) or 0
            if horizon and len(frame):
                floor = frame["day"].max() - pd.Timedelta(days=horizon)
                frame = frame.loc[frame["day"] >= floor]

            # Sorted by day so `views.day_slice` can binary-search instead of
            # masking 19.5M rows per request. Done once at load; the sort is
            # the reason a day filter is instant afterwards.
            frame = frame.sort_values("day", kind="mergesort").reset_index(drop=True)
            # Guarantee the columns the screens reference exist. Artefacts
            # written by different model generations carry different features,
            # and a template referencing a missing attribute raises a 500 rather
            # than degrading -- so every expected name is present, as NaN where
            # the data genuinely is not available. NaN renders as "--".
            for column in ("server", "trades", "wins", "gross_notional",
                           "life_win_rate", "life_closes", "life_profit_factor",
                           "scalp_rate", "martingale_rate", "avg_hold_minutes",
                           "live_positions", "carried", "pnl_20d"):
                if column not in frame.columns:
                    frame[column] = (frame["account_key"].str.split(":").str[0]
                                     if column == "server" else np.nan)
            # NOTE: converting account_key/symbol to `category` was tried and
            # REMOVED. It saved ~0.8 GB but cost ~140s on 19.5M object strings,
            # which dominated the first page load. Peak memory is ~2.4 GB
            # either way, so the trade was strictly bad.
            _FRAME_CACHE[view] = (stamp, frame)
            return frame
        except Exception:
            return None


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def start_training(view: str, config: TrainingConfig) -> bool:
    """Kick off a refit in the background. False if one is already running."""
    with _LOCK:
        state = _JOBS.setdefault(view, JobState())
        if state.status == "running":
            return False
        _JOBS[view] = JobState(status="running", message="starting", started_at=time.time(), log=[])
    worker = threading.Thread(target=_run, args=(view, config), daemon=True)
    worker.start()
    return True


def _run(view: str, config: TrainingConfig) -> None:
    try:
        # Preserve the model being replaced BEFORE training overwrites its
        # artifacts (the trainer saves boosters mid-run), so the Strategy Lab
        # can always compare a run against its predecessor.
        try:
            archived = archive_current(view, note="auto-archived before retrain")
            if archived is not None:
                _log(view, f"previous model archived -> history/{archived.name}", 0.02)
        except Exception as error:
            _log(view, f"archive of previous model skipped: {error}", 0.02)
        if view == VIEW_TRADING:
            frame, metrics = _train_client_model(config)
        else:
            frame, metrics = _train_trade_model(config)
        scores_path, meta_path = artifact_paths(view)
        frame.to_parquet(scores_path, index=False)
        meta_path.write_text(json.dumps({
            "fingerprint": config.fingerprint(),
            "config": asdict(config),
            "metrics": metrics,
            "rows": int(len(frame)),
            "trained_at": time.time(),
        }, indent=2))
        with _LOCK:
            state = _JOBS[view]
            state.status, state.finished_at, state.progress = "done", time.time(), 1.0
            state.message = f"trained on {len(frame):,} rows"
    except Exception as error:
        with _LOCK:
            state = _JOBS[view]
            state.status, state.finished_at = "error", time.time()
            state.message = f"{type(error).__name__}: {error}"
            state.log.append(traceback.format_exc()[-1500:])


def _quarter_bounds(start, end) -> list[tuple]:
    """The window split into quarters, as (lower, upper) datetime pairs."""
    lower = pd.Timestamp(start).tz_localize(None).normalize()
    upper = pd.Timestamp(end).tz_localize(None).normalize()
    edges = [lower] + pd.date_range(lower, upper, freq="QS").tolist() + [upper]
    edges = sorted(set(edges))
    return [(a.to_pydatetime().replace(tzinfo=timezone.utc),
             b.to_pydatetime().replace(tzinfo=timezone.utc))
            for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _estimate_warehouse_rows(store_module, servers, start, end) -> int:
    """Rows the warehouse holds in this window, from parquet metadata alone.

    Parquet records its row count in the file footer, so this is exact per file
    and costs no column read. Files are named by month, so months outside the
    window are skipped entirely and only the partial months at each edge are
    over-counted -- which is the safe direction for choosing a sample fraction.
    """
    import pyarrow.parquet as pq

    lo = pd.Timestamp(start).tz_localize(None).to_period("M")
    hi = pd.Timestamp(end).tz_localize(None).to_period("M")
    total = 0
    for server in servers:
        for path in (store_module.WAREHOUSE / server).rglob("*.parquet"):
            try:
                month = pd.Period(path.stem, freq="M")
            except (ValueError, TypeError):
                continue  # not a month-named file; counted only if unparseable
            if lo <= month <= hi:
                try:
                    total += pq.ParquetFile(path).metadata.num_rows
                except Exception:
                    continue
    return total


def _accumulate_trade_chunk(chunk: pd.DataFrame, columns: list, parts: list,
                            view: str, server) -> None:
    """Clean one warehouse chunk and append it in the training schema."""
    # Cent accounts carry lots and P&L inflated 100x; deflate before anything
    # learns from them. Fail-soft when the group lookup is unreachable.
    try:
        from webapp.trade_feed import cent_logins
        cents = cent_logins(server) if server else set()
        if cents and "login" in chunk.columns:
            mask = chunk["login"].astype("int64").isin(cents)
            if mask.any():
                for column in ("volume_lots", "net_profit"):
                    if column in chunk.columns:
                        chunk.loc[mask, column] = pd.to_numeric(
                            chunk.loc[mask, column], errors="coerce") / 100.0
    except Exception:
        pass
    if "account_key" not in chunk.columns:
        chunk["account_key"] = (chunk["database"].astype(str) + ":"
                                + chunk["login"].astype("int64").astype(str))
    chunk = chunk.loc[chunk["open_time"].notna() & chunk["close_time"].notna()
                      & chunk["net_profit"].notna() & (chunk["volume_lots"] > 0)
                      & (chunk["open_price"] > 0)]
    if chunk.empty:
        return
    # Downcast before accumulating: float32 halves the numeric footprint and
    # the models cast to float32 anyway.
    for column in ("volume_lots", "open_price", "close_price", "net_profit", "sl", "tp"):
        if column in chunk.columns:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce").astype("float32")
    for column in columns:
        if column not in chunk.columns:
            chunk[column] = pd.NA
    parts.append(chunk[columns])


def _load_trade_history(config: TrainingConfig, view: str) -> pd.DataFrame:
    """Trades over the configured history, from the local store.

    Reads the MySQL-backed warehouse first -- it is the system of record and
    costs nothing per query. The 90-day BigQuery extract is the fallback for
    when the store has not been backfilled yet, and for `mt5_dubai_live01`,
    which does not exist in this MySQL instance.
    """
    from datetime import timedelta

    from webapp import data_store

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=config.history_days)
    columns = ["database", "account_key", "symbol", "cmd", "volume_lots", "open_time",
               "close_time", "open_price", "close_price", "sl", "tp", "net_profit",
               "state", "reason"]

    # Read PER SERVER and filter immediately. Two years across five servers is
    # ~145 million trades; materialising that as one frame and then filtering
    # exhausted 31 GB of RAM. Filtering inside the loop keeps peak memory to one
    # server's slice rather than the whole history.
    from webapp import data_store as store_module

    # The warehouse stores `database` and `login`; `account_key` is derived from
    # the two and does not exist as a stored column. Asking for it returned ZERO
    # rows without raising, which silently sent this whole view down the 90-day
    # BigQuery fallback -- a paid query standing in for two years of free local
    # history, and the reason the Quant artefact covered 71 days while Trading
    # covered 670. Read the stored names; derive the key afterwards.
    warehouse_columns = [c for c in columns if c not in ("account_key", "reason")]
    if "login" not in warehouse_columns:
        warehouse_columns.append("login")

    servers = sorted(p.name for p in store_module.WAREHOUSE.iterdir() if p.is_dir())

    # Two years of trades is ~183 million rows. The day-stratified cap further
    # down runs AFTER the whole history is in memory, which at this scale never
    # gets there -- the trading path already had to be chunked by quarter for
    # the same reason. So the read is chunked, and thinned as it goes.
    #
    # The fraction is derived from parquet row-count metadata, which is exact
    # and costs no data read. Sampling uniformly inside each chunk preserves the
    # day and server mix in expectation, so the downstream stratified cap has
    # nothing left to correct.
    estimate = _estimate_warehouse_rows(store_module, servers, start, end)
    keep_fraction = 1.0
    if estimate > config.max_trade_rows:
        keep_fraction = config.max_trade_rows / estimate
        _log(view, f"warehouse holds ~{estimate:,} trades; sampling "
                   f"{keep_fraction:.1%} while reading to stay inside memory", 0.05)

    windows = _quarter_bounds(start, end)
    parts, total_read = [], 0
    for server in servers or [None]:
        before = sum(len(p) for p in parts)
        for lower, upper in windows:
            chunk = data_store.read_history(
                databases=(server,) if server else None,
                start=lower, end=upper, columns=warehouse_columns)
            total_read += len(chunk)
            if chunk.empty:
                continue
            if keep_fraction < 1.0:
                chunk = chunk.sample(frac=keep_fraction, random_state=0)
            _accumulate_trade_chunk(chunk, columns, parts, view, server)
            del chunk
            gc.collect()
        _log(view, f"{server}: {sum(len(p) for p in parts) - before:,} trades kept", 0.06)

    if parts:
        stored = pd.concat(parts, ignore_index=True)
        del parts
        gc.collect()
        _log(view, f"warehouse: {len(stored):,} trades of {total_read:,} read "
                   f"over {config.history_days} days", 0.08)
        # The fraction of the real book this frame represents. Every dollar
        # figure computed from a sampled frame is a fraction of the truth, and
        # without this number the Quant view quoted a $151M "flat B-book"
        # against Trading's $1.1bn -- same book, different denominators, and a
        # 10x confusion for anyone comparing the two screens. Day-stratified
        # uniform sampling makes 1/fraction an unbiased scale-up for daily
        # totals, so the display layer can restore full-book units.
        stored.attrs["sample_fraction"] = keep_fraction
        return stored

    # --- fallback: the cached 90-day BigQuery extract ---------------------
    # Falling back is a COST and a downgrade -- a paid 90-day extract replacing
    # two years of free local history -- so it says so, and says whether the
    # warehouse was genuinely absent or merely returned nothing. A silent
    # fallback is how this view spent months on 71 days of BigQuery data.
    detail = ("warehouse has no server directories" if not servers
              else f"warehouse returned 0 usable rows from {total_read:,} read "
                   f"across {len(servers)} servers -- likely a column mismatch")
    _log(view, f"FALLING BACK TO PAID BIGQUERY (90 days): {detail}", 0.05)
    parts = []
    for server in ("mt4_live01", "mt4_live02", "mt4_live03", "mt4_live04"):
        part = pd.read_parquet(SCRATCH / "bq_90d_records.parquet",
                               columns=[c for c in columns if c != "database"],
                               filters=[("database", "==", server)])
        part = part.loc[(part["state"].astype("string") == "closed")
                        & part["open_time"].notna() & part["close_time"].notna()
                        & part["net_profit"].notna() & (part["volume_lots"] > 0)
                        & (part["open_price"] > 0)]
        if len(part):
            part["database"] = server
            parts.append(part)
        del part
        gc.collect()

    from webapp.mt5_pairing import load_paired_mt5
    for server in ("mt5_live01", "mt5_dubai_live01"):
        try:
            paired = load_paired_mt5(SCRATCH / "bq_90d_records.parquet", server)
        except Exception as error:
            _log(view, f"{server}: pairing failed ({error})", 0.06)
            continue
        if len(paired):
            paired["database"] = server
            parts.append(paired)
        del paired
        gc.collect()

    if not parts:
        raise RuntimeError("no trade history available from the warehouse or the extract")
    trades = pd.concat(parts, ignore_index=True)
    for column in columns:
        if column not in trades.columns:
            trades[column] = pd.NA
    return trades[columns].reset_index(drop=True)


def _fit_params(config: TrainingConfig) -> dict:
    return dict(n_estimators=config.n_estimators, num_leaves=config.num_leaves,
                learning_rate=config.learning_rate, subsample=config.subsample,
                subsample_freq=1, colsample_bytree=config.colsample_bytree,
                min_child_samples=config.min_child_samples,
                # 63 histogram bins instead of the default 255. Every feature
                # here is a coarse behavioural aggregate (win rates, rolling
                # P&L, counts) where 63 quantile buckets lose nothing a tree
                # could split on -- and histogram construction is the dominant
                # cost of a fit, so this alone is ~2.5x.
                max_bin=63,
                n_jobs=-1, verbose=-1, random_state=0)


def _fit_subsample(usable: np.ndarray, cap: int) -> np.ndarray:
    """A boolean training mask capped at `cap` rows, sampled uniformly.

    The walk-forward refits ~96 times over two years, and each refit trained on
    the ENTIRE rolling window -- up to 25M rows. At 63 leaves a tree's capacity
    saturates long before that: past a few million rows extra data buys compute,
    not accuracy. Uniform sampling of the window preserves its day mix in
    expectation. This plus max_bin is the difference between a 7-hour training
    run and a ~20-minute one.
    """
    count = int(usable.sum())
    if cap <= 0 or count <= cap:
        return usable
    rng = np.random.default_rng(0)
    indices = np.flatnonzero(usable)
    chosen = rng.choice(indices, size=cap, replace=False)
    mask = np.zeros_like(usable)
    mask[chosen] = True
    return mask


# ---------------------------------------------------------------------------
# per-class trade modelling
# ---------------------------------------------------------------------------
# Gold is ~93% of trade rows, so a single pooled classifier is effectively a
# gold specialist and its RAW probabilities are miscalibrated for every other
# class -- yet the copy/invert anchors are applied to exactly those numbers. We
# fit one booster per model-class and calibrate each class's out-of-sample
# scores separately, with a cheap pooled fallback for any class too thin to fit
# its own on a given refit. `symbol_class` (gold/silver/fx/crypto/index-other)
# is the shared classifier; here we group it into the four model-classes.
_MODEL_CLASSES: tuple[str, ...] = ("metals", "fx", "index", "crypto")
_SYMBOL_CLASS_TO_MODEL = {"gold": "metals", "silver": "metals", "fx": "fx",
                          "index/other": "index", "crypto": "crypto"}
_POOL_KEY = "_pool"


def _model_class_array(symbols) -> np.ndarray:
    """Map each row's symbol to its model-class (metals/fx/index/crypto)."""
    from webapp.trade_features import symbol_class
    symbols = symbols.astype(str)
    mapping = {s: _SYMBOL_CLASS_TO_MODEL.get(symbol_class(s), "index")
               for s in symbols.unique()}
    return symbols.map(mapping).to_numpy()


def _fit_class_models(X, profit, mclass, symbol_weight, usable, config, lgb) -> dict:
    """One classifier per model-class on the usable window, plus a cheap pooled
    fallback for classes too thin to fit their own. Each class is capped at
    max_fit_rows; the pooled model is capped small since it only serves the rare
    thin class. Returns {class: model} (+ _POOL_KEY)."""
    models: dict = {}
    cap = int(getattr(config, "max_fit_rows", 0) or 0)
    pool_mask = _fit_subsample(usable, min(cap or 300000, 300000))
    yp = profit[pool_mask] > 0
    if yp.size and yp.min() != yp.max():
        m = lgb.LGBMClassifier(**_fit_params(config))
        m.fit(X[pool_mask], yp, sample_weight=symbol_weight[pool_mask])
        models[_POOL_KEY] = m
    for cls in _MODEL_CLASSES:
        cmask = usable & (mclass == cls)
        if int(cmask.sum()) < 3000:
            continue
        cmask = _fit_subsample(cmask, cap)
        yc = profit[cmask] > 0
        if yc.size == 0 or yc.min() == yc.max():
            continue
        m = lgb.LGBMClassifier(**_fit_params(config))
        m.fit(X[cmask], yc, sample_weight=symbol_weight[cmask])
        models[cls] = m
    return models


def _fit_calibrators(score, profit, mclass) -> dict:
    """Per-class isotonic calibration of OOS scores -> empirical win-probability.
    Returns {class: {"x": [...], "y": [...]}} for np.interp at score time."""
    from sklearn.isotonic import IsotonicRegression
    calibrators: dict = {}
    valid = np.isfinite(score)
    for cls in _MODEL_CLASSES:
        m = valid & (mclass == cls)
        if int(m.sum()) < 2000:
            continue
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(score[m], (profit[m] > 0).astype(float))
        calibrators[cls] = {"x": np.asarray(ir.X_thresholds_, float).tolist(),
                            "y": np.asarray(ir.y_thresholds_, float).tolist()}
    return calibrators


def _apply_calibrators(score, mclass, calibrators) -> np.ndarray:
    """Map raw scores to calibrated probabilities per class (piecewise-linear)."""
    out = np.array(score, dtype=float, copy=True)
    for cls, cal in (calibrators or {}).items():
        m = (mclass == cls) & np.isfinite(out)
        if m.any() and cal.get("x"):
            out[m] = np.interp(out[m], np.asarray(cal["x"], float),
                               np.asarray(cal["y"], float))
    return out


def _per_class_metrics(output) -> dict:
    """Per model-class: rows, base win-rate, ROC-AUC, and Brier score raw vs
    calibrated. AUC is rank-based so calibration leaves it unchanged; Brier is
    a reliability score, so calibration should LOWER it -- that is the win."""
    from sklearn.metrics import brier_score_loss, roc_auc_score
    out: dict = {}
    for cls, g in output.groupby("model_class", observed=True):
        y = (g["pnl"] > 0).to_numpy()
        rec = {"rows": int(len(g)),
               "base_rate": round(float(y.mean()), 4) if len(g) else 0.0}
        if len(g) >= 500 and y.min() != y.max():
            raw = np.clip(g["score"].to_numpy(), 0, 1)
            cal = np.clip(g["score_cal"].to_numpy(), 0, 1)
            try:
                rec["auc"] = round(float(roc_auc_score(y, raw)), 4)
                rec["brier_raw"] = round(float(brier_score_loss(y, raw)), 5)
                rec["brier_cal"] = round(float(brier_score_loss(y, cal)), 5)
            except Exception:
                pass
        out[str(cls)] = rec
    return out


def _fit_path_regressors(X, target, mclass, valid, tkey, name, lgb, metrics,
                         alpha: float | None = None, objective: str = "quantile",
                         max_rows: int = 600_000, pooled_rows: int = 300_000) -> dict:
    """Per-class + pooled regressors on `target` over rows where `valid`.
    Saves ARTIFACTS/<name>.txt (pooled) and <name>_<class>.txt, records a
    time-ordered 80/20 holdout correlation per model in metrics[name].

    Row caps are deliberate: a 63-leaf tree's capacity saturates well below a
    million rows, and quantile objectives cost more per row than L2, so a
    600k cap per class (300k for the pooled FALLBACK) keeps up to twenty
    regressor fits per run to minutes rather than an hour."""
    params = dict(n_estimators=150, num_leaves=63, learning_rate=0.08, max_bin=63,
                  n_jobs=-1, verbose=-1, random_state=0)
    if objective == "quantile":
        params.update(objective="quantile", alpha=float(alpha))
    else:
        params.update(objective=objective)

    def fit_save(mask, path, cap=max_rows):
        idx = np.flatnonzero(mask)
        if len(idx) < 3000:
            return None
        if len(idx) > cap:        # capacity saturates; extra rows buy compute only
            idx = np.random.default_rng(0).choice(idx, cap, replace=False)
        idx = idx[np.argsort(tkey[idx], kind="stable")]     # time-ordered holdout
        split = int(len(idx) * 0.8)
        model = lgb.LGBMRegressor(**params)
        model.fit(X[idx[:split]], target[idx[:split]])
        held_pred = model.predict(X[idx[split:]])
        held_act = target[idx[split:]]
        corr = (float(np.corrcoef(held_pred, held_act)[0, 1])
                if held_act.std() > 0 and held_pred.std() > 0 else float("nan"))
        model.fit(X[idx], target[idx])
        model.booster_.save_model(str(path))
        return None if np.isnan(corr) else round(corr, 4)

    result: dict = {}
    corr = fit_save(valid, ARTIFACTS / f"{name}.txt", pooled_rows)   # fallback only
    if corr is not None:
        result["pooled"] = corr
    for cls in _MODEL_CLASSES:
        corr = fit_save(valid & (mclass == cls), ARTIFACTS / f"{name}_{cls}.txt")
        if corr is not None:
            result[cls] = corr
    metrics[name] = result
    return result


def _train_path_models(X, trades, mclass, lgb, metrics) -> None:
    """The three path-target models + per-lot edge, per class, all in ONE pass
    on the same feature frame so none can go stale independently again:
      * quant_mae_q50 / q80 -- pre-profit drawdown (adverse bps): the median
        prices the entry limit, the q80 tail drives veto + inverse sizing;
      * entry_model -- the legacy 'bps of better entry available' IS the MAE
        median, so it is the same booster under the legacy filename;
      * quant_exit_fe -- favourable excursion at q0.35 (the validated design);
      * quant_perlot -- $ per lot, L1 (model E).
    Path models train only on trades that actually have a minute path."""
    import shutil

    tkey = pd.to_datetime(trades["open_time"]).to_numpy().astype("int64")
    seen = pd.to_numeric(trades["bars_seen"], errors="coerce").fillna(0).to_numpy() > 0
    mae_raw = pd.to_numeric(trades["mae_bps"], errors="coerce").to_numpy(dtype=float)
    mfe_raw = pd.to_numeric(trades["mfe_bps"], errors="coerce").to_numpy(dtype=float)
    valid_path = seen & np.isfinite(mae_raw) & np.isfinite(mfe_raw)
    metrics["path_coverage"] = round(float(valid_path.mean()), 4) if len(valid_path) else 0.0
    if int(valid_path.sum()) >= 3000:
        depth = np.clip(np.nan_to_num(-mae_raw, nan=0.0), 0.0, 3000.0)   # adverse depth >= 0
        favour = np.clip(np.nan_to_num(mfe_raw, nan=0.0), 0.0, 3000.0)
        _fit_path_regressors(X, depth, mclass, valid_path, tkey, "quant_mae_q50",
                             lgb, metrics, alpha=0.50)
        _fit_path_regressors(X, depth, mclass, valid_path, tkey, "quant_mae_q80",
                             lgb, metrics, alpha=0.80)
        _fit_path_regressors(X, favour, mclass, valid_path, tkey, "quant_exit_fe",
                             lgb, metrics, alpha=0.35)
        try:   # legacy filename the engine already loads for entry placement
            shutil.copyfile(ARTIFACTS / "quant_mae_q50.txt", ARTIFACTS / "entry_model.txt")
        except Exception:
            pass
        _log(VIEW_QUANT,
             f"path models: coverage {metrics['path_coverage']:.0%} | holdout corr "
             f"mae_q50 {metrics.get('quant_mae_q50', {}).get('pooled')} | "
             f"exit_fe {metrics.get('quant_exit_fe', {}).get('pooled')}", 0.995)
    else:
        _log(VIEW_QUANT, "path models skipped: too few trades with a minute path", 0.995)

    lots = pd.to_numeric(trades["volume_lots"], errors="coerce").to_numpy(dtype=float)
    profit = pd.to_numeric(trades["net_profit"], errors="coerce").to_numpy(dtype=float)
    has_lots = np.isfinite(lots) & (lots > 0) & np.isfinite(profit)
    perlot = np.zeros(len(lots))
    perlot[has_lots] = np.clip(profit[has_lots] / lots[has_lots], -5000.0, 5000.0)
    _fit_path_regressors(X, perlot, mclass, has_lots, tkey, "quant_perlot",
                         lgb, metrics, objective="l1")
    _log(VIEW_QUANT,
         f"per-lot model E: holdout corr {metrics.get('quant_perlot', {}).get('pooled')}",
         0.997)


#: Features built on the exposure calendar. Deliberately small and mechanical:
#: with two years of history the model has room to learn, and every one of these
#: is knowable at the start of the day it describes.
EXPOSURE_FEATURES: tuple[str, ...] = (
    "live_positions", "opened", "closed", "carried",
    "live_positions_5d", "live_positions_20d", "carried_5d", "carried_20d",
    "pnl_5d", "pnl_20d", "pnl_60d", "pnl_vol_20d", "pnl_vol_60d",
    "win_rate_20d", "win_rate_60d", "trades_5d", "trades_20d",
    "exposure_days_20d", "exposure_ratio_20d", "tenure_days",
    "pnl_rank_20d", "drawdown_20d", "best_day_20d", "worst_day_20d",
    "days_since_trade", "avg_hold_days_20d",
)


def _build_exposure_frame(config: TrainingConfig) -> tuple[pd.DataFrame, list[str]]:
    """Account-day frame on the EXPOSURE definition of an active day.

    An account is active on any day it opened, closed, or CARRIED a position.
    The previous definition counted only closing days, so a position held for a
    week appeared once -- and the six intervening days of firm exposure were
    invisible to both the features and the target.
    """
    from datetime import timedelta

    from webapp import data_store as store_module
    from webapp.exposure_days import exposure_days

    # Expand to exposure days PER SERVER and discard the trades immediately.
    # Two years is ~145M trades; holding them all and then expanding overran
    # 31 GB. The calendar itself is far smaller -- one row per account-day
    # rather than per trade -- so aggregating early is what makes two years fit.
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=config.history_days)
    servers = sorted(p.name for p in store_module.WAREHOUSE.iterdir() if p.is_dir())
    calendars = []
    # Chunked by QUARTER as well as by server. A whole server at once was not
    # enough: mt4_live04's 46.5M trades expand into a frame that drove resident
    # memory to 10.7 GB and spent 49 minutes swapping rather than computing.
    # Peak memory is now bounded by the busiest quarter of the busiest server,
    # not by the largest server outright.
    quarters = pd.date_range(pd.Timestamp(start).tz_localize(None).normalize(),
                             pd.Timestamp(end).tz_localize(None).normalize(),
                             freq="QS").tolist()
    quarters = [pd.Timestamp(start).tz_localize(None)] + quarters + \
               [pd.Timestamp(end).tz_localize(None)]
    quarters = sorted(set(quarters))

    for server in servers:
        server_rows, server_days = 0, 0
        for lower, upper in zip(quarters[:-1], quarters[1:]):
            chunk = store_module.read_history(
                databases=(server,), start=lower.to_pydatetime(), end=upper.to_pydatetime(),
                columns=["database", "login", "open_time", "close_time", "net_profit"])
            if chunk.empty:
                continue
            # Coerce before filtering. A source that stores P&L as something
            # other than a float -- BigQuery hands back NUMERIC as Decimal --
            # survives ingestion, survives the read, and then fails inside a
            # groupby hours into training with "unsupported operand type(s) for
            # +: 'decimal.Decimal' and 'float'". Cheap here, expensive there.
            chunk["net_profit"] = pd.to_numeric(chunk["net_profit"], errors="coerce")

            # CENT ACCOUNTS denominate in 1/100 units: their lots and P&L sit
            # in the warehouse inflated 100x, distorting every feature and
            # every dollar figure the model learns from. Deflate them here,
            # keyed on the broker's own group naming. Fail-soft: with the VPN
            # down the lookup returns empty and the pass is a no-op -- the
            # next retrain with connectivity applies it.
            try:
                from webapp.trade_feed import cent_logins
                cents = cent_logins(server)
                if cents:
                    mask = chunk["login"].astype("int64").isin(cents)
                    if mask.any():
                        for column in ("volume_lots", "net_profit"):
                            chunk.loc[mask, column] = chunk.loc[mask, column] / 100.0
                        _log(view, f"{server}: deflated {int(mask.sum()):,} "
                                   f"cent-account trades (/100)", 0.07)
            except Exception:
                pass
            chunk = chunk.loc[chunk["open_time"].notna() & chunk["net_profit"].notna()]
            if chunk.empty:
                continue
            chunk["account_key"] = (chunk["database"].astype(str) + ":"
                                    + chunk["login"].astype("int64").astype(str))
            part = exposure_days(
                chunk[["account_key", "open_time", "close_time", "net_profit"]],
                window_start=lower)
            server_rows += len(chunk)
            server_days += len(part)
            calendars.append(part)
            del chunk, part
            gc.collect()
        _log(VIEW_TRADING, f"{server}: {server_rows:,} trades -> {server_days:,} exposure days", 0.20)

    # Quarter boundaries can produce two rows for the same account-day (a
    # position spanning the boundary appears in both). Summing them would double
    # a day's realised P&L, so they are merged rather than concatenated blindly.
    if calendars:
        merged = pd.concat(calendars, ignore_index=True)
        del calendars
        gc.collect()
        calendars = [merged.groupby(["account_key", "day"], observed=True, as_index=False).agg(
            opened=("opened", "sum"), closed=("closed", "sum"),
            live_positions=("live_positions", "sum"),
            realised_pnl=("realised_pnl", "sum"), carried=("carried", "max"))]
        del merged
        gc.collect()

    if not calendars:
        raise RuntimeError("no trade history in the warehouse for that window")
    calendar = pd.concat(calendars, ignore_index=True)
    del calendars
    gc.collect()
    calendar = calendar.sort_values(["account_key", "day"]).reset_index(drop=True)
    _log(VIEW_TRADING, f"exposure calendar: {len(calendar):,} account-days", 0.30)

    group = calendar.groupby("account_key", observed=True)
    # Every rolling statistic is SHIFTED, so a row never sees its own day.
    for window in (5, 20, 60):
        shifted = group["realised_pnl"].shift()
        calendar[f"pnl_{window}d"] = shifted.groupby(
            calendar["account_key"], observed=True).transform(
            lambda s, w=window: s.rolling(w, min_periods=2).sum())
        if window >= 20:
            calendar[f"pnl_vol_{window}d"] = shifted.groupby(
                calendar["account_key"], observed=True).transform(
                lambda s, w=window: s.rolling(w, min_periods=5).std())
            calendar[f"win_rate_{window}d"] = (shifted > 0).groupby(
                calendar["account_key"], observed=True).transform(
                lambda s, w=window: s.rolling(w, min_periods=5).mean())
    for column in ("live_positions", "carried"):
        for window in (5, 20):
            calendar[f"{column}_{window}d"] = group[column].shift().groupby(
                calendar["account_key"], observed=True).transform(
                lambda s, w=window: s.rolling(w, min_periods=2).mean())
    for window in (5, 20):
        calendar[f"trades_{window}d"] = group["closed"].shift().groupby(
            calendar["account_key"], observed=True).transform(
            lambda s, w=window: s.rolling(w, min_periods=2).sum())

    calendar["tenure_days"] = group.cumcount()
    calendar["exposure_days_20d"] = group["day"].shift().groupby(
        calendar["account_key"], observed=True).transform(
        lambda s: s.rolling(20, min_periods=2).count())
    # How continuously the account is exposed: 1.0 means a position open every
    # day, near 0 means sporadic. A structurally different client profile that
    # the closing-day definition could not express at all.
    calendar["exposure_ratio_20d"] = calendar["exposure_days_20d"] / 20.0
    calendar["days_since_trade"] = group["closed"].shift().groupby(
        calendar["account_key"], observed=True).transform(
        lambda s: s.eq(0).groupby(s.ne(0).cumsum()).cumcount())
    calendar["avg_hold_days_20d"] = calendar["live_positions_20d"] / \
        calendar["trades_20d"].replace(0, np.nan)

    cumulative = group["realised_pnl"].shift().groupby(
        calendar["account_key"], observed=True).cumsum()
    peak = cumulative.groupby(calendar["account_key"], observed=True).cummax()
    calendar["drawdown_20d"] = cumulative - peak
    calendar["best_day_20d"] = group["realised_pnl"].shift().groupby(
        calendar["account_key"], observed=True).transform(
        lambda s: s.rolling(20, min_periods=3).max())
    calendar["worst_day_20d"] = group["realised_pnl"].shift().groupby(
        calendar["account_key"], observed=True).transform(
        lambda s: s.rolling(20, min_periods=3).min())
    calendar["pnl_rank_20d"] = calendar.groupby("day", observed=True)["pnl_20d"].rank(pct=True)

    calendar["decision_day"] = calendar["day"]
    calendar["pnl"] = calendar["realised_pnl"]

    # Funding behaviour. Deposits and withdrawals are among the strongest
    # signals for abuse -- a client who deposits, wins and withdraws at once
    # looks nothing like one who funds an account and trades it down -- and none
    # of it previously reached the model.
    #
    # Every one of these is cumulative-to-date and carried forward by a backward
    # merge, so a day sees only movements that had already settled. Attaching
    # lifetime totals instead would tell the model on day one what the client
    # does on day seven hundred: it would score beautifully and be worthless,
    # which is exactly the leak that produced an AUC of 1.0000 earlier here.
    cash_features: list[str] = []
    try:
        from webapp import cashflow_store
        movements = cashflow_store.read_cashflows()
        if not movements.empty:
            daily = cashflow_store.daily_features(movements)
            calendar = cashflow_store.attach(calendar, daily)
            cash_features = [c for c in cashflow_store.CASHFLOW_FEATURES
                             if c in calendar.columns]
            _log(VIEW_TRADING, f"attached {len(cash_features)} cashflow features "
                               f"from {len(movements):,} movements", 0.44)
        else:
            _log(VIEW_TRADING, "no cached cash movements -- cashflow features skipped", 0.44)
    except Exception as error:
        # Funding data is an enrichment, not a dependency. Losing it must not
        # cost the whole training run.
        _log(VIEW_TRADING, f"cashflow features unavailable ({type(error).__name__})", 0.44)

    features = [f for f in (*EXPOSURE_FEATURES, *cash_features) if f in calendar.columns]
    for column in features:
        calendar[column] = pd.to_numeric(calendar[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan).astype("float32")
    _log(VIEW_TRADING, f"built {len(features)} exposure features", 0.45)
    return calendar, features


def _train_client_model(config: TrainingConfig) -> tuple[pd.DataFrame, dict]:
    """Account-day router on the self-relative target.

    Mirrors `trading_data.self_relative_routing`: forward P&L over N ACTIVE days
    expressed in units of the account's own volatility, with exact per-row
    purging so a label that has not finished resolving never enters training.
    """
    import lightgbm as lgb

    if config.use_exposure_days:
        frame, features = _build_exposure_frame(config)
    else:
        _log(VIEW_TRADING, "loading cached account-day frame", 0.05)
        # ONE corpus for training, live scoring and the watchlist model alike:
        # trade_features._AD_DIR (webapp/artifacts/ad), kept at yesterday by
        # webapp/ad_refresh.py. The scratchpad copy was a session-temp orphan.
        from webapp.trade_features import _AD_DIR
        frame = pd.read_parquet(_AD_DIR / "model_frame.parquet")
        features = pd.read_csv(_AD_DIR / "model_features.csv", header=None)[0].tolist()
    leaked = sorted(set(features) & {"pnl", "target_profit", "target_client_wins"})
    if leaked:
        raise RuntimeError(f"label columns in feature set: {leaked}")

    frame = frame.sort_values(["account_key", "decision_day"], kind="mergesort").reset_index(drop=True)
    group = frame.groupby("account_key", observed=True)
    horizon = max(1, int(config.horizon_active_days))
    if horizon == 1:
        forward = group["pnl"].shift(-1)
    else:
        rolled = group["pnl"].transform(
            lambda s: s[::-1].rolling(horizon, min_periods=1).sum()[::-1])
        forward = rolled.groupby(frame["account_key"], observed=True).shift(-1)
    scale = group["pnl"].transform("std").replace(0, np.nan)
    frame["sigma"] = forward / (scale * np.sqrt(horizon))
    frame["label_end"] = group["decision_day"].shift(-horizon).fillna(
        group["decision_day"].transform("max"))

    frame = frame.sort_values("decision_day", kind="mergesort").reset_index(drop=True)
    X = np.ascontiguousarray(frame[features].to_numpy(dtype="float32"))
    pnl = frame["pnl"].to_numpy(dtype="float64")
    day_codes, day_values = pd.factorize(frame["decision_day"], sort=True)
    lookup = {value: index for index, value in enumerate(day_values)}
    label_end = frame["label_end"].map(lookup).to_numpy(dtype="float64")
    sigma = frame["sigma"].to_numpy(dtype="float64")

    # ECONOMIC SAMPLE WEIGHTS.
    #
    # Every account-day previously counted equally in the loss, so the model
    # spent its capacity where the ROWS are rather than where the MONEY is. The
    # measured consequence: the largest size decile holds 83% of all firm P&L
    # and is the one band where the model is weakest -- AUC 0.670 against ~0.714
    # for deciles five to nine. It discriminates worst exactly where the dollars
    # sit, which is also why ranking by expected dollars lost $411M when tried.
    #
    # The weight is the account's own recent P&L volatility: how much it
    # actually moves. It is already computed point-in-time as a feature, so it
    # adds no lookahead, and unlike the row's own realised P&L it is not coupled
    # to the outcome being predicted.
    #
    # Log-scaled and clipped, because raw dollars span six orders of magnitude
    # and weighting linearly would let a handful of whales own the objective --
    # trading one blind spot for another.
    weights = None
    if getattr(config, "economic_weights", True) and "pnl_vol_20d" in frame.columns:
        magnitude = pd.to_numeric(frame["pnl_vol_20d"], errors="coerce").abs()
        magnitude = magnitude.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        positive = magnitude[magnitude > 0]
        reference = float(np.median(positive)) if positive.size else 1.0
        weights = np.log1p(magnitude / max(reference, 1e-9))
        weights = np.clip(weights, 0.1, 10.0)
        _log(VIEW_TRADING, f"economic weights: median {np.median(weights):.2f}, "
                           f"p99 {np.percentile(weights, 99):.2f}", 0.46)

    score = np.full(len(X), np.nan)
    estimator, fitted = None, False
    total = len(day_values)
    for day in range(config.min_train_days, total):
        lo, hi = np.searchsorted(day_codes, day), np.searchsorted(day_codes, day + 1)
        if hi <= lo:
            continue
        if (day - config.min_train_days) % max(1, config.refit_cadence_days) == 0 or not fitted:
            # Exact purge: only labels that finished resolving before today.
            usable = np.isfinite(sigma) & np.isfinite(label_end) & (label_end < day) & (day_codes < day)
            if config.max_train_days:
                # Rolling window: drop history older than the limit.
                usable &= day_codes >= (day - config.max_train_days)
            if usable.sum() >= 200:
                usable = _fit_subsample(usable, getattr(config, "max_fit_rows", 0))
                y = sigma[usable] > config.sigma_threshold
                if y.min() != y.max():
                    previous = estimator.booster_ if (config.warm_start and fitted
                                                      and estimator is not None) else None
                    estimator = lgb.LGBMClassifier(
                        scale_pos_weight=float((~y).sum() / max(1, y.sum())), **_fit_params(config))
                    estimator.fit(X[usable], y, init_model=previous,
                                  sample_weight=weights[usable] if weights is not None else None)
                    fitted = True
            _log(VIEW_TRADING, f"walk-forward day {day}/{total}", 0.05 + 0.9 * day / total)
        if fitted:
            score[lo:hi] = estimator.predict_proba(X[lo:hi])[:, 1]

    output = pd.DataFrame({
        "account_key": frame["account_key"].astype(str),
        "day": frame["decision_day"],
        "pnl": pnl,
        "score": score,
        "sigma": sigma,
    })
    # Server is stored rather than derived: splitting 4.7M account keys on every
    # render cost 103 seconds when it was done lazily.
    output["server"] = output["account_key"].str.split(":").str[0]

    # Descriptive columns the screens expect. The exposure-day frame carries
    # different features from the old account-day frame, and a missing column
    # raises inside a template rather than degrading -- so every expected name
    # is present, filled from the exposure equivalent where one exists and left
    # as NaN where it genuinely does not. NaN renders as "--"; a missing
    # attribute takes the page down.
    equivalents = {
        "trades": "closed",                    # transactions that day
        "wins": None,
        "gross_notional": None,                # not derivable without trade sizes
        "life_win_rate": "win_rate_60d",
        "life_closes": "tenure_days",
        "life_profit_factor": None,
        "scalp_rate": None,
        "martingale_rate": None,
        "avg_hold_minutes": "avg_hold_days_20d",
    }
    for column, source in equivalents.items():
        if column in frame.columns:
            output[column] = pd.to_numeric(frame[column], errors="coerce")
        elif source and source in frame.columns:
            output[column] = pd.to_numeric(frame[source], errors="coerce")
        else:
            output[column] = np.nan
    # Exposure-specific columns, genuinely new and worth surfacing.
    for column in ("live_positions", "carried", "opened", "exposure_ratio_20d",
                   "pnl_20d", "drawdown_20d", "days_since_trade"):
        if column in frame.columns:
            output[column] = pd.to_numeric(frame[column], errors="coerce")
    output = output.loc[np.isfinite(output["score"])].reset_index(drop=True)

    metrics = _routing_metrics(output, config)
    _log(VIEW_TRADING, "done", 1.0)
    return output, metrics


def _train_trade_model(config: TrainingConfig) -> tuple[pd.DataFrame, dict]:
    """Trade-level router: hedge or keep each individual trade.

    Uses close-time purging -- a trade may only enter training once it has
    actually closed, since its label is the realised profit.
    """
    import hashlib
    import json as json_module

    import lightgbm as lgb

    from webapp.trade_features import TRADE_FEATURES, build_trade_features

    # FEATURE CACHE: loading the warehouse and building 245 features costs
    # ~8 minutes; an experiment that only changes the target, anchors or
    # hyperparameters should not pay it. The cache is keyed by the feature
    # list + sampling settings and expires after a day -- a hit also means
    # IDENTICAL data across runs, which is what clean comparisons need.
    cache_path = SCRATCH / "quant_feature_cache.parquet"
    cache_meta_path = SCRATCH / "quant_feature_cache.json"
    signature = hashlib.md5(
        ("|".join(TRADE_FEATURES)
         + f"|{config.history_days}|{config.max_trade_rows}").encode()).hexdigest()
    trades = None
    sample_fraction = 1.0
    try:
        if cache_path.exists() and cache_meta_path.exists():
            cached = json_module.loads(
                cache_meta_path.read_text(encoding="utf-8"))
            age_hours = (time.time() - float(cached.get("built_at", 0))) / 3600
            if cached.get("signature") == signature and age_hours < 24:
                trades = pd.read_parquet(cache_path)
                sample_fraction = float(cached.get("sample_fraction", 1.0))
                _log(VIEW_QUANT,
                     f"feature cache HIT: {len(trades):,} rows, "
                     f"{age_hours:.1f}h old -- warehouse load and feature "
                     f"build skipped", 0.12)
    except Exception:
        trades = None

    if trades is None:
        trades = _load_trade_history(config, VIEW_QUANT)

        # The extract selects on CLOSE date, so trades OPENED long before the
        # window appear only if they stayed open long enough to close inside
        # it -- survivors by construction, 0.2% of the data, and they make
        # the walk-forward iterate over ~460 near-empty days.
        window_start = trades["close_time"].min().floor("D")
        before = len(trades)
        trades = trades.loc[trades["open_time"] >= window_start]
        _log(VIEW_QUANT, f"dropped {before - len(trades):,} pre-window survivor trades", 0.10)

        # Fractions COMPOUND: the loader may have thinned while reading, and
        # this stage may thin again. The artifact records the product so
        # full-book figures can be restored on screen.
        sample_fraction = float(trades.attrs.get("sample_fraction", 1.0) or 1.0)
        if len(trades) > config.max_trade_rows:
            # Stratify by day so the sample spans the whole history rather
            # than collapsing to whichever months happened to be densest.
            fraction = config.max_trade_rows / len(trades)
            before = len(trades)
            trades = (trades.assign(_d=pd.to_datetime(trades["open_time"]).dt.normalize())
                            .groupby("_d", group_keys=False, observed=True)
                            .apply(lambda g: g.sample(frac=fraction, random_state=0))
                            .drop(columns="_d").reset_index(drop=True))
            _log(VIEW_QUANT, f"sampled {len(trades):,} of {before:,} trades "
                             f"({fraction:.1%}, stratified by day)", 0.11)
            sample_fraction *= fraction
            gc.collect()

        _log(VIEW_QUANT, f"building features for {len(trades):,} trades", 0.12)
        trades = build_trade_features(trades)
        try:
            trades.to_parquet(cache_path)
            cache_meta_path.write_text(json_module.dumps(
                {"signature": signature, "built_at": time.time(),
                 "rows": len(trades), "sample_fraction": sample_fraction}),
                encoding="utf-8")
            _log(VIEW_QUANT, "feature frame cached for fast reruns", 0.13)
        except Exception:
            pass

    # PRICE-PATH excursions (MAE / MFE in bps) for every trade from the shared
    # mt4_live01 minute-bar feed -- the targets for the drawdown, entry and exit
    # models. Bars are cached per window; the sweep itself takes seconds. A feed
    # outage must never block the classifier retrain, so failure just leaves
    # the path columns empty and the path models are skipped.
    try:
        from webapp import path_features
        trades = path_features.attach_excursions(
            trades, SCRATCH,
            log=lambda m: _log(VIEW_QUANT, f"path bars: {m}", 0.14))
        _log(VIEW_QUANT,
             f"path excursions: {int((trades['bars_seen'] > 0).sum()):,} of "
             f"{len(trades):,} trades have a minute path", 0.15)
    except Exception as error:
        _log(VIEW_QUANT, f"path excursions skipped: "
                         f"{type(error).__name__}: {error}", 0.15)
        trades["mae_bps"], trades["mfe_bps"], trades["bars_seen"] = np.nan, np.nan, 0

    X = np.ascontiguousarray(trades[TRADE_FEATURES].to_numpy(dtype="float32"))
    np.putmask(X, ~np.isfinite(X), np.nan)
    symbol_counts = trades["symbol"].astype(str).value_counts()
    symbol_weight = 1.0 / np.sqrt(
        trades["symbol"].astype(str).map(symbol_counts).to_numpy(dtype="float64"))
    symbol_weight *= len(symbol_weight) / symbol_weight.sum()
    mclass = _model_class_array(trades["symbol"])   # metals/fx/index/crypto per row
    profit = trades["net_profit"].to_numpy(dtype="float64")
    day_codes, day_values = pd.factorize(trades["day"], sort=True)
    lookup = {value: index for index, value in enumerate(day_values)}
    close_codes = trades["close_time"].dt.floor("D").map(lookup).to_numpy(dtype="float64")
    close_codes = np.where(np.isfinite(close_codes), close_codes, np.inf)

    score = np.full(len(X), np.nan)
    models: dict = {}                  # per-class boosters from the latest refit
    fitted = False
    total = len(day_values)
    _walk_cum = [0.0, 0.0, 0.0]        # cumulative policy P&L, maxDD, peak
    _reset_metrics(VIEW_QUANT)         # fresh live stream for this run
    _walk_records: list[dict] = []     # persisted into meta for idle viewing
    _walk_scale = 1.0 / max(sample_fraction, 1e-9)   # sampled rows -> full book
    for day in range(config.min_train_days, total):
        lo, hi = np.searchsorted(day_codes, day), np.searchsorted(day_codes, day + 1)
        if hi <= lo:
            continue
        if (day - config.min_train_days) % max(1, config.refit_cadence_days) == 0 or not fitted:
            usable = (day_codes < day) & (close_codes < day)   # opened AND closed before today
            if config.max_train_days:
                usable &= day_codes >= (day - config.max_train_days)
            if usable.sum() >= 5000:
                # PER-CLASS refit: one booster per model-class (+ pooled
                # fallback), so gold's 93% share can't dictate the fit or the
                # probability scale for FX / index / crypto. Symbol-frequency
                # weights still balance symbols WITHIN a class.
                models = _fit_class_models(X, profit, mclass, symbol_weight,
                                           usable, config, lgb)
                fitted = bool(models)
        if fitted:
            # ROUTE each row to its class booster; a class with no model this
            # refit (too thin) falls back to the pooled booster.
            slice_classes = mclass[lo:hi]
            for cls in np.unique(slice_classes):
                rows = np.flatnonzero(slice_classes == cls) + lo
                model = models.get(cls) or models.get(_POOL_KEY)
                if model is not None and len(rows):
                    score[rows] = model.predict_proba(X[rows])[:, 1]
            # INTERIM METRICS, printed as the walk advances: the day's own
            # top/bottom-decile policy P&L and the running cumulative, so a
            # two-hour walk narrates its result instead of going dark until
            # the end. Cheap: quantiles over one day's scores.
            day_scores = score[lo:hi]
            day_pnl = profit[lo:hi]
            if len(day_scores) >= 20:
                hi_cut = np.quantile(day_scores, 0.90)
                lo_cut = np.quantile(day_scores, 0.10)
                day_policy = (float(day_pnl[day_scores >= hi_cut].sum())
                              - float(day_pnl[day_scores <= lo_cut].sum()))
                _walk_cum[0] += day_policy
                _walk_cum[1] = min(_walk_cum[1], _walk_cum[0] - _walk_cum[2])
                _walk_cum[2] = max(_walk_cum[2], _walk_cum[0])
                if (day - config.min_train_days) % max(1, config.refit_cadence_days) == 0:
                    _log(VIEW_QUANT,
                         f"walk-forward day {day}/{total} | rows {hi:,} | "
                         f"day policy ${day_policy:+,.0f} | cumulative "
                         f"${_walk_cum[0]:+,.0f} | maxDD ${_walk_cum[1]:,.0f}",
                         0.12 + 0.85 * day / total)
                    record = {
                        "day": int(day), "total": int(total),
                        "date": str(pd.Timestamp(day_values[day]).date()),
                        "cum": round(_walk_cum[0] * _walk_scale, 2),
                        "dd": round(_walk_cum[1] * _walk_scale, 2),
                        "day_pnl": round(day_policy * _walk_scale, 2),
                        "rows": int(hi - lo),
                    }
                    _emit_metric(VIEW_QUANT, record)
                    _walk_records.append(record)

    # PER-CLASS CALIBRATION: fit isotonic maps on the OOS scores so a score is
    # an honest win-probability WITHIN its class -- then the copy/invert anchors
    # mean the same thing for gold as for FX. Monotonic, so ranking (AUC) is
    # unchanged; reliability (Brier) improves.
    calibrators = _fit_calibrators(score, profit, mclass)
    score_cal = _apply_calibrators(score, mclass, calibrators)
    output = pd.DataFrame({
        "account_key": trades["account_key"].astype(str),
        "symbol": trades["symbol"].astype(str),
        "model_class": mclass,
        "day": trades["day"],
        "open_time": trades["open_time"],
        # Close time and entry price make the artifact self-sufficient for
        # path-aware exit studies (holding period, MAE/MFE joins). Without
        # them every exit-policy question needed a warehouse re-read and a
        # fuzzy join back to the scores.
        "close_time": trades["close_time"],
        "open_price": pd.to_numeric(trades["open_price"], errors="coerce"),
        "close_price": pd.to_numeric(trades["close_price"], errors="coerce"),
        "direction": trades["direction"],
        "volume_lots": trades["volume_lots"],
        "notional": trades["notional"],
        "pnl": profit,
        "score": score,
        "score_cal": score_cal,
        "with_momentum": pd.to_numeric(trades["with_momentum"],
                                       errors="coerce"),
    })
    output = output.loc[np.isfinite(output["score"])].reset_index(drop=True)
    metrics = _routing_metrics(output, config, scale=1.0 / max(sample_fraction, 1e-9))
    metrics["sample_fraction"] = sample_fraction
    metrics["walk_curve"] = _walk_records   # per-refit-day stream for the Lab tab
    metrics["per_class"] = _per_class_metrics(output)
    metrics["calibrated_classes"] = sorted(calibrators.keys())
    # Persist the FINAL walk-forward model so live systems can score trades the
    # moment they open. Without this the artifact holds historical scores only,
    # and anything realtime (the Vantage copytrader) has no model to call.
    pooled = None
    if fitted:
        # FINAL deployment models: refit per class on the FULL frame (all
        # history) + a pooled fallback, then persist the per-class calibrators.
        # The live engine routes by class and applies the matching calibrator;
        # the pooled model + raw score stay as the backward-compatible fallback.
        final_models = _fit_class_models(
            X, profit, mclass, symbol_weight,
            np.ones(len(X), dtype=bool), config, lgb)
        pooled = final_models.get(_POOL_KEY)
        if pooled is not None:
            pooled.booster_.save_model(str(ARTIFACTS / "quant_model.txt"))
        for _cls in _MODEL_CLASSES:
            _m = final_models.get(_cls)
            if _m is not None:
                _m.booster_.save_model(str(ARTIFACTS / f"quant_model_{_cls}.txt"))
        (ARTIFACTS / "quant_calibrators.json").write_text(
            json_module.dumps(calibrators), encoding="utf-8")
        (ARTIFACTS / "quant_model_features.txt").write_text(
            "\n".join(TRADE_FEATURES), encoding="utf-8")
        if pooled is not None:
            metrics["feature_importance"] = {
                name: int(value) for name, value in
                sorted(zip(TRADE_FEATURES, pooled.feature_importances_),
                       key=lambda pair: -pair[1])}
        # The symbol_code feature is a category code: its integer mapping is
        # an artefact of THIS training frame's category order. Persist it so
        # live scoring assigns the SAME code to the same symbol instead of
        # sending NaN (or worse, a different frame's code).
        try:
            categories = output["symbol"].astype("category").cat.categories
            (ARTIFACTS / "quant_symbols.txt").write_text(
                "\n".join(str(s) for s in categories), encoding="utf-8")
        except Exception:
            pass
        # MAGNITUDE MODEL: same features, different question. The classifier
        # answers WHO wins; this answers HOW MUCH the trade moves relative to
        # its all-in cost -- target |pnl|/cost censored to 0 below cost, so
        # the model optimises exactly what the cost hurdle checks. Direction
        # stays the classifier's job; the product edge x multiple x cost is
        # the trade's expected dollars.
        try:
            import lightgbm as lgb

            from webapp.trade_features import trade_cost_vector
            cost = trade_cost_vector(trades["symbol"], trades["open_price"],
                                     trades["volume_lots"])
            magnitude = np.where((np.abs(profit) > cost) & (cost > 0),
                                 np.abs(profit) / np.maximum(cost, 1e-9), 0.0)
            magnitude = np.clip(magnitude, 0.0, 50.0)
            order = np.argsort(trades["open_time"].to_numpy(), kind="stable")
            if len(order) > 2_000_000:   # capacity saturates: cap, keep time order
                keep = np.sort(np.random.default_rng(0).choice(
                    len(order), 2_000_000, replace=False))
                order = order[keep]
            split = int(len(order) * 0.8)
            regressor = lgb.LGBMRegressor(
                n_estimators=200, num_leaves=63, learning_rate=0.08,
                max_bin=63, n_jobs=-1, verbose=-1, random_state=0)
            regressor.fit(X[order[:split]], magnitude[order[:split]])
            held_prediction = regressor.predict(X[order[split:]])
            held_actual = magnitude[order[split:]]
            correlation = float(np.corrcoef(held_prediction, held_actual)[0, 1])
            _log(VIEW_QUANT,
                 f"magnitude model: holdout corr {correlation:.3f} | mean "
                 f"target {held_actual.mean():.2f}x cost | share above cost "
                 f"{(held_actual > 0).mean():.0%}", 0.99)
            regressor.fit(X[order], magnitude[order])
            regressor.booster_.save_model(str(ARTIFACTS / "quant_magnitude.txt"))
            metrics["magnitude_holdout_corr"] = correlation
        except Exception as error:
            _log(VIEW_QUANT, f"magnitude model skipped: "
                             f"{type(error).__name__}: {error}", 0.99)
        # PATH MODELS (drawdown / entry / exit) + per-lot E, per class, on the
        # same frame -- rebuilt every run so they can never go stale alone.
        try:
            import lightgbm as lgb
            _train_path_models(X, trades, mclass, lgb, metrics)
        except Exception as error:
            _log(VIEW_QUANT, f"path models skipped: "
                             f"{type(error).__name__}: {error}", 0.997)
    # feature_importance is captured from the pooled final model above (the
    # Research tab reads it); nothing to add here.
    metrics["symbol_breakdown"] = {
        str(symbol): {"trades": int(len(group)),
                      "client_pnl": float(group["pnl"].sum()),
                      "mean_score": float(group["score"].mean())}
        for symbol, group in output.groupby("symbol", observed=True)
        if len(group) >= 50
    }
    _log(VIEW_QUANT, "done", 1.0)
    return output, metrics


def _routing_metrics(frame: pd.DataFrame, config: TrainingConfig,
                     scale: float = 1.0) -> dict:
    """Headline comparison against B-booking everything.

    Every policy is measured through `equity_curves`, i.e. the same day-by-day
    simulation the charts draw, with a point-in-time cutoff over the routable
    population. Scoring the table one way and drawing the curve another would
    let the two disagree, and the table is what people quote.

    `scale` restores full-book units when the frame is a sample: dollar figures
    multiply by it (an unbiased estimator under day-stratified sampling), while
    Sharpe and Calmar are ratios of the same units and stay untouched. Without
    this, Quant's "flat B-book" read $151M against Trading's $1.1bn -- the same
    book through a 17% keyhole, quoted as if it were the whole thing.
    """
    from trading_data.self_relative_routing import routing_metrics

    def rescaled(metrics: dict) -> dict:
        for key in ("total_pnl_usd", "max_drawdown_usd"):
            if key in metrics and metrics[key] is not None:
                metrics[key] = float(metrics[key]) * scale
        return metrics

    baseline = equity_curves(frame, hedge_fraction=1e-9)
    result = {"flat_bbook": rescaled(routing_metrics(baseline.set_index("day")["flat"])),
              "by_fraction": {}}
    for fraction in (0.02, 0.05, 0.10, 0.20):
        curve = equity_curves(frame, hedge_fraction=fraction)
        metrics = rescaled(routing_metrics(curve.set_index("day")["model"]))
        metrics["mean_routable"] = float(curve["routable"].mean())
        metrics["mean_hedged"] = float(curve["hedged_accounts"].mean())
        result["by_fraction"][f"{fraction:.2f}"] = metrics
    try:
        from trading_data.research import _rank_discrimination
        result["roc_auc"] = float(_rank_discrimination(
            frame["score"].to_numpy(dtype="float64"),
            frame["pnl"].to_numpy(dtype="float64") > 0)["roc_auc"])
    except Exception:
        result["roc_auc"] = None
    return result


#: Computed curves, keyed by (artefact fingerprint, policy). The day-by-day
#: simulation walks ~400k rows (23M for Quant) and was re-running on EVERY page
#: render, costing ~7s per view -- long enough that the app felt broken after
#: sign-in. The inputs are a cached artefact and two numbers, so the result is
#: perfectly cacheable.
_CURVE_CACHE: dict[tuple, pd.DataFrame] = {}
_CURVE_CACHE_LIMIT = 32


_DERIVED_CACHE: dict[str, tuple[float, dict]] = {}


def daily_series(view: str, frame: pd.DataFrame) -> pd.Series:
    """Firm P&L per day, cached per artefact.

    Both the summary and the risk monitor derive this, and on the Quant frame
    the groupby alone is seconds. It changes only when the model is retrained.
    """
    facts = frame_facts(view, frame)
    if "daily" not in facts:
        days = pd.to_datetime(frame["day"]).dt.normalize()
        facts["daily"] = (-frame.assign(_d=days).groupby("_d")["pnl"].sum()).sort_index()
    return facts["daily"]


def account_totals(view: str, frame: pd.DataFrame) -> pd.Series:
    """Firm P&L per account, cached per artefact. Drives concentration."""
    facts = frame_facts(view, frame)
    if "accounts" not in facts:
        facts["accounts"] = (-frame.groupby("account_key", observed=True)["pnl"].sum()
                             ).sort_values(ascending=False)
    return facts["accounts"]


def frame_facts(view: str, frame: pd.DataFrame) -> dict:
    """Day list and coverage summary, computed once per artefact.

    Both scan the whole frame. Recomputing them on every render cost seconds on
    the Quant artefact, for values that only change when the model is retrained.
    """
    scores_path, _ = artifact_paths(view)
    stamp = scores_path.stat().st_mtime if scores_path.exists() else 0.0
    cached = _DERIVED_CACHE.get(view)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    from webapp import views as _views
    facts = {
        "days": _views.available_days(frame),
        "coverage": _views.coverage_note(frame),
    }
    _DERIVED_CACHE[view] = (stamp, facts)
    return facts


def trade_equity_curves(frame: pd.DataFrame, hedge_fraction: float,
                        probability_threshold: float = 0.0) -> pd.DataFrame:
    """Day-by-day routing simulation for TRADE-level decisions.

    The account-day version carries each account's last score forward, because a
    routing decision covers an account that may not trade that day. That idea
    does not transfer: a trade IS the decision, and there is nothing to carry
    forward. Here the cutoff is the quantile of that day's own trade scores --
    still strictly point-in-time, and fully vectorised.

    The vectorisation matters as much as the semantics: the account-day
    implementation ran a per-row Python loop, which on 19.5M trades exhausted
    memory and killed the process.
    """
    working = frame[["day", "score", "pnl"]].copy()
    working["day"] = pd.to_datetime(working["day"]).dt.normalize()
    scores = working["score"].to_numpy(dtype="float64")

    if probability_threshold > 0:
        hedged = scores >= probability_threshold
    else:
        # `groupby.transform(lambda)` runs the lambda once per group in Python
        # and broadcasts, which took ~145s on 19.5M rows. A single
        # groupby.quantile() followed by a map is the same computation done in
        # C, and returns in under a second.
        quantile = 1 - max(1e-6, hedge_fraction)
        cutoff_by_day = working.groupby("day")["score"].quantile(quantile)
        hedged = scores >= working["day"].map(cutoff_by_day).to_numpy(dtype="float64")

    pnl = working["pnl"].to_numpy(dtype="float64")
    daily = pd.DataFrame({
        "day": working["day"].to_numpy(),
        "flat": -pnl,
        "model": np.where(hedged, 0.0, -pnl),
        "hedged_accounts": hedged.astype("int32"),
        "active_accounts": np.ones(len(pnl), dtype="int32"),
    }).groupby("day", as_index=False).sum().sort_values("day")
    daily["routable"] = daily["active_accounts"]
    daily["cutoff"] = np.nan
    daily["flat_cum"] = daily["flat"].cumsum()
    daily["model_cum"] = daily["model"].cumsum()
    daily["flat_dd"] = daily["flat_cum"] - daily["flat_cum"].cummax()
    daily["model_dd"] = daily["model_cum"] - daily["model_cum"].cummax()
    return daily.reset_index(drop=True)


def cached_equity_curves(view: str, frame: pd.DataFrame, hedge_fraction: float,
                         probability_threshold: float = 0.0) -> pd.DataFrame:
    meta = artifact_meta(view) or {}
    key = (view, meta.get("fingerprint"), meta.get("trained_at"),
           round(hedge_fraction, 6), round(probability_threshold, 6))
    cached = _CURVE_CACHE.get(key)
    if cached is None:
        builder = trade_equity_curves if view == VIEW_QUANT else equity_curves
        cached = builder(frame, hedge_fraction, probability_threshold)
        # Quant trains on a day-stratified SAMPLE (memory cap), so its raw curve
        # is a fraction of the real book -- $151M of "flat B-book" against
        # Trading's $1.1bn for the same firm. Scale every dollar column back to
        # full-book units; 1/fraction is unbiased for daily totals under
        # uniform-by-day sampling. Count columns stay raw.
        fraction = float((meta.get("metrics") or {}).get("sample_fraction") or 1.0)
        if view == VIEW_QUANT and 0 < fraction < 1:
            cached = cached.copy()
            for column in ("flat", "model", "flat_cum", "model_cum",
                           "flat_dd", "model_dd"):
                if column in cached.columns:
                    cached[column] = cached[column] / fraction
        if len(_CURVE_CACHE) >= _CURVE_CACHE_LIMIT:
            _CURVE_CACHE.clear()
        _CURVE_CACHE[key] = cached
    return cached


def equity_curves(frame: pd.DataFrame, hedge_fraction: float,
                  probability_threshold: float = 0.0,
                  max_stale_days: int = 30) -> pd.DataFrame:
    """Daily firm P&L under flat B-book versus the model, simulated honestly.

    Two corrections over the naive version, both of which flattered the model.

    1. THE THRESHOLD WAS LOOKAHEAD. Taking one quantile over every row in the
       sample sets today's cutoff using scores that had not been produced yet.
       Here the cutoff is recomputed each day from only that day's ROUTABLE
       POPULATION, which is information a desk genuinely holds each morning.

    2. THE POPULATION WAS CIRCULAR. Ranking only accounts that turned out to be
       active uses tomorrow's activity to choose today's candidates. The
       population is now every account with an observation on or before the
       decision day, carried forward from its most recent one -- matching the
       A-book manifest exactly, so the curve and the list can never disagree.

    Still computed entirely from cached scores, so the controls stay instant.
    """
    return _curves_from(_prepare_curve_inputs(frame), hedge_fraction,
                        probability_threshold, max_stale_days)


def _prepare_curve_inputs(frame: pd.DataFrame) -> dict:
    """The per-frame work that does not depend on the hedge fraction.

    Copying, sorting and factorising a 4.3M-row frame costs several seconds. The
    validation screen evaluates twenty-four curves over the same few windows, and
    doing this inside each of them made the page take minutes. Split out, it is
    paid once per window instead of once per fraction.
    """
    working = frame.copy()
    working["day"] = pd.to_datetime(working["day"])
    working = working.sort_values("day", kind="mergesort")
    days = np.sort(working["day"].unique())
    codes, _uniques = pd.factorize(working["account_key"], sort=False)
    day_values = working["day"].to_numpy()
    return {
        "days": days,
        "codes": codes,
        "n_accounts": int(codes.max()) + 1 if len(codes) else 0,
        "starts": np.searchsorted(day_values, days, side="left"),
        "ends": np.searchsorted(day_values, days, side="right"),
        "scores": working["score"].to_numpy(dtype="float64"),
        "pnl": working["pnl"].to_numpy(dtype="float64"),
    }


def _curves_from(prepared: dict, hedge_fraction: float,
                 probability_threshold: float = 0.0,
                 max_stale_days: int = 30) -> pd.DataFrame:
    days = prepared["days"]
    codes = prepared["codes"]
    n_accounts = prepared["n_accounts"]
    starts, ends = prepared["starts"], prepared["ends"]
    all_scores, all_pnl = prepared["scores"], prepared["pnl"]

    # Carried-forward state: each account's most recent score and when it was
    # observed. Walking days in order keeps this point-in-time by construction.
    rows: list[dict] = []
    fraction = max(1e-6, min(1.0, hedge_fraction))

    # The carry-forward state is held in two arrays indexed by account code
    # rather than in dicts keyed by account string. The semantics are identical
    # -- last observation wins, staleness measured in days -- but the cost is
    # not: the dict version rebuilt the routable pool with a Python-level
    # comprehension over every account ever seen, once per day. At 730 days and
    # ~1M accounts that is ~700M interpreted iterations, and combined with the
    # `day == day` full-frame scan below it made this function take minutes and
    # time the Overview page out on a cold cache.
    last_score = np.zeros(n_accounts, dtype="float64")
    last_day = np.zeros(n_accounts, dtype="int64")
    seen = np.zeros(n_accounts, dtype=bool)

    horizon_ns = int(np.timedelta64(max_stale_days, "D") / np.timedelta64(1, "ns"))

    for day, lo, hi in zip(days, starts, ends):
        day_ns = int(pd.Timestamp(day).value)
        # Threshold from the population as it stands BEFORE today's rows are
        # folded in -- the state a desk would have had this morning.
        fresh_mask = seen & ((day_ns - last_day) <= horizon_ns)
        fresh = last_score[fresh_mask]

        scores = all_scores[lo:hi]
        pnl = all_pnl[lo:hi]
        pool = fresh if fresh.size >= 50 else scores
        cutoff = (probability_threshold if probability_threshold > 0
                  else float(np.quantile(pool, 1 - fraction)) if pool.size else np.inf)

        hedged = scores >= cutoff
        rows.append({
            "day": day,
            "flat": float(-pnl.sum()),
            "model": float(-pnl[~hedged].sum()),
            "hedged_accounts": int(hedged.sum()),
            "active_accounts": int(hi - lo),
            "routable": int(fresh.size),
            "cutoff": float(cutoff) if np.isfinite(cutoff) else np.nan,
        })

        # Fancy indexing carries the same last-wins semantics as the dict writes.
        today_codes = codes[lo:hi]
        last_score[today_codes] = scores
        last_day[today_codes] = day_ns
        seen[today_codes] = True

    daily = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
    daily["flat_cum"] = daily["flat"].cumsum()
    daily["model_cum"] = daily["model"].cumsum()
    daily["flat_dd"] = daily["flat_cum"] - daily["flat_cum"].cummax()
    daily["model_dd"] = daily["model_cum"] - daily["model_cum"].cummax()
    return daily
