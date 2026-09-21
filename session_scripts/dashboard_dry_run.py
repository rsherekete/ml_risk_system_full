"""Execute dashboard.py's actual logic against synthetic data with a stubbed streamlit."""
import sys
import types

sys.path.insert(0, r"C:\Users\RoyVivasi\Documents\notebook")

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Build synthetic MT5-shaped trade records and symbol specs (same generator as
# the research/risk smoke test) and a fake client exposing get_trade_records /
# symbol_specs, so `clients_from_yaml()` can be monkeypatched cheaply.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(11)
n = 3000
days = pd.date_range("2026-05-01", periods=90, freq="D")
symbols_raw = ["EURUSD", "EURUSD.raw", "XAUUSD", "GOLD", "BTCUSDMIN", "US30CASH"]
accounts = list(range(1, 41))

timestamps = rng.choice(days, size=n) + pd.to_timedelta(rng.integers(0, 86400, n), unit="s")
symbol_choice = rng.choice(symbols_raw, size=n, p=[0.30, 0.10, 0.20, 0.05, 0.15, 0.20])
cmd_choice = rng.choice(["buy", "sell"], size=n)
account_choice = rng.choice(accounts, size=n)
volume_lots = np.round(rng.uniform(0.01, 5.0, size=n), 2)
base_price = {"EURUSD": 1.08, "EURUSD.raw": 1.08, "XAUUSD": 2350.0, "GOLD": 2350.0, "BTCUSDMIN": 65000.0, "US30CASH": 39000.0}
open_price = np.array([base_price[s] for s in symbol_choice]) * (1 + rng.normal(0, 0.01, n))
close_price = open_price * (1 + rng.normal(0, 0.01, n))
profit = (close_price - open_price) * np.where(cmd_choice == "buy", 1, -1) * volume_lots * 1000
state_choice = rng.choice(["open", "closed"], size=n, p=[0.2, 0.8])

raw_records = pd.DataFrame({
    "timestamp": pd.to_datetime(timestamps),
    "login": account_choice,
    "symbol": symbol_choice,
    "cmd": cmd_choice,
    "volume_lots": volume_lots,
    "open_time": pd.to_datetime(timestamps) - pd.to_timedelta(rng.integers(1, 5000, n), unit="s"),
    "close_time": pd.to_datetime(timestamps),
    "open_price": open_price,
    "close_price": close_price,
    "price": open_price,
    "state": state_choice,
    "reason": "client",
    "profit": np.where(state_choice == "closed", profit, np.nan),
}).sort_values("timestamp").reset_index(drop=True)

specs = pd.DataFrame({
    "symbol": ["EURUSD", "XAUUSD"],
    "contract_size": [100_000.0, 100.0],
    "tick_value": [1.0, 1.0],
    "tick_size": [0.00001, 0.01],
    "currency_base": ["EUR", "XAU"],
    "currency_profit": ["USD", "USD"],
    "currency_margin": ["EUR", "USD"],
})


class FakeClient:
    def get_trade_records(self, start_time, end_time, symbol=None, account=None):
        start = pd.Timestamp(start_time)
        end = pd.Timestamp(end_time)
        return raw_records.loc[(raw_records["timestamp"] >= start) & (raw_records["timestamp"] < end)].copy()

    def symbol_specs(self):
        return specs.copy()

    def query(self, sql, **kwargs):
        # No account-group data in this synthetic fixture -- every login is
        # "unknown", exactly like a real MT4 login with no userinfo snapshot.
        return pd.DataFrame(columns=["login", "acct_group"])


import trading_data.research as research_module
# mt5_live01, not mt5_demo01: demo databases are excluded from the real
# pipeline by default, so a demo-named fixture would yield zero records.
research_module.clients_from_yaml = lambda path=None: {"mt5_live01": FakeClient()}


# ---------------------------------------------------------------------------
# Minimal streamlit stub covering exactly the API surface dashboard.py uses.
# ---------------------------------------------------------------------------
class StopExecution(Exception):
    pass


class _Ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _Col(_Ctx):
    def metric(self, *a, **k): print("metric:", a, {kk: vv for kk, vv in k.items() if kk != "help"})
    def bar_chart(self, data, **k): print("bar_chart rows:", len(data) if hasattr(data, "__len__") else "?")
    def line_chart(self, data, **k): print("line_chart rows:", len(data) if hasattr(data, "__len__") else "?")
    def altair_chart(self, *a, **k): print("  altair_chart (in column) rendered")
    def caption(self, *a, **k): pass
    def markdown(self, *a, **k): pass
    def table(self, *a, **k): pass
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass
    def expander(self, *a, **k): return _Ctx()
    def dataframe(self, *a, **k): pass
    def container(self, *a, **k): return _Col()
    def multiselect(self, label, options, default=None, **k): return default if default is not None else list(options)
    def number_input(self, label, min_value=None, value=0.0, **k): return value
    def text_input(self, label, value="", **k): return value
    def checkbox(self, label, value=False, **k): return value
    def selectbox(self, label, options, index=0, **k): return list(options)[index]


class _Form(_Ctx):
    pass


class _Tab(_Ctx):
    pass


class _SessionState(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _cache_data(*dargs, **dkwargs):
    def decorator(fn):
        return fn
    return decorator


st_stub = types.ModuleType("streamlit")
st_stub.session_state = _SessionState()
st_stub.set_page_config = lambda **k: None
st_stub.title = lambda *a, **k: print("TITLE:", *a)
st_stub.caption = lambda *a, **k: None
st_stub.subheader = lambda *a, **k: print("  subheader:", *a)
st_stub.markdown = lambda *a, **k: None
st_stub.divider = lambda *a, **k: None
st_stub.info = lambda *a, **k: print("INFO:", *a)
st_stub.warning = lambda *a, **k: print("WARNING:", *a)
st_stub.error = lambda *a, **k: print("ERROR:", *a)
st_stub.stop = lambda: (_ for _ in ()).throw(StopExecution())
st_stub.sidebar = _Ctx()
st_stub.form = lambda name: _Form()
st_stub.form_submit_button = lambda *a, **k: True
st_stub.checkbox = lambda label, value=False, **k: (True if "Compare" in label or "fallback" in label else value)
# Force "MySQL only" so the dry run keeps using the synthetic FakeClient and
# never reaches out to real BigQuery.
st_stub.radio = lambda label, options, index=0, **k: "MySQL only"
st_stub.expander = lambda label, **k: _Ctx()
st_stub.slider = lambda label, *a, **k: (k.get("value", a[2] if len(a) > 2 else (a[0] if a else 0)))
st_stub.select_slider = lambda label, options, value=None, **k: (value if value is not None else list(options)[0])
st_stub.selectbox = lambda label, options, index=0, **k: list(options)[index]
st_stub.multiselect = lambda label, options, default=None, **k: (default if default is not None else list(options))
st_stub.number_input = lambda label, min_value=None, value=0.0, **k: value
st_stub.text_input = lambda label, value="", **k: value
st_stub.date_input = lambda label, value=None, **k: value
st_stub.columns = lambda n, **k: [_Col() for _ in range(n if isinstance(n, int) else len(n))]
st_stub.container = lambda *a, **k: _Col()
st_stub.tabs = lambda names: [_Tab() for _ in names]
st_stub.dataframe = lambda *a, **k: None
st_stub.table = lambda *a, **k: print("  table rendered")
st_stub.bar_chart = lambda *a, **k: None
st_stub.line_chart = lambda *a, **k: None
st_stub.altair_chart = lambda *a, **k: print("  altair_chart rendered")
st_stub.download_button = lambda label, data, **k: print(f"  download_button: {label} ({len(data)} bytes)")
st_stub.metric = lambda label, value, delta=None, **k: print("metric:", label, value, delta)
st_stub.cache_data = _cache_data

# `profit_weight`/`drawdown_weight` slider stub needs to actually read/write
# session_state like the real widget does, since dashboard.py drives them via
# key= rather than return value.
def _slider(label, *args, **kwargs):
    key = kwargs.get("key")
    if key is not None:
        if key not in st_stub.session_state:
            st_stub.session_state[key] = kwargs.get("value", 0)
        return st_stub.session_state[key]
    if "value" in kwargs:
        return kwargs["value"]
    return args[2] if len(args) > 2 else (args[0] if args else 0)


st_stub.slider = _slider

sys.modules["streamlit"] = st_stub
sys.modules["altair"] = __import__("altair")

print("Running dashboard.py with stubbed streamlit + synthetic data (compare-day and fallback-only branches forced on)...")
with open(r"C:\Users\RoyVivasi\Documents\notebook\dashboard.py", encoding="utf-8") as fh:
    source = fh.read()
namespace = {"__name__": "__dashboard_dry_run__"}
exec(compile(source, "dashboard.py", "exec"), namespace)
print("\nDASHBOARD DRY RUN COMPLETED WITHOUT ERROR")

print("\n--- direct callback check: linked profit/drawdown sliders ---")
st_stub.session_state["profit_weight"] = 85
namespace["_sync_from_profit"]()
print("profit=85 -> drawdown =", st_stub.session_state["drawdown_weight"], "(expected 15)")
assert st_stub.session_state["drawdown_weight"] == 15

st_stub.session_state["drawdown_weight"] = 90
namespace["_sync_from_drawdown"]()
print("drawdown=90 -> profit =", st_stub.session_state["profit_weight"], "(expected 10)")
assert st_stub.session_state["profit_weight"] == 10
print("Linked-slider callbacks verified OK.")
