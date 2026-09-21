"""Copy-trading engine — STUB.

The real module is proprietary and lives in a separate private repository. It
is not part of the beta profile: every route that touches it (`/quant/...`,
`/vantage/...`) is blocked by `_beta_gate` when `AF_BETA=1`, so nothing in the
three beta sections reaches this code.

It exists only because `webapp/main.py` imports `copytrader` at module scope, so
the application cannot start without *something* by this name. Each entry point
below raises rather than returning a plausible-looking empty value: a silent
no-op here would let a misconfigured deployment appear to run a copy book that
does not exist, which is a far worse failure than an immediate error.

To run the full application, replace this file with the real module.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_MESSAGE = ("The copy-trading engine is not included in this deployment. "
            "This build serves the anti-fraud beta profile only.")


class CopyTraderUnavailable(RuntimeError):
    """Raised when something asks the excluded engine to do real work."""


def _unavailable(*_args, **_kwargs):
    raise CopyTraderUnavailable(_MESSAGE)


@dataclass
class _State:
    """Inert state object. `kill_switch` reads True so that anything which
    inspects it before acting sees a book that is stopped, never a live one."""
    kill_switch: bool = True
    connected: bool = False
    positions: list = field(default_factory=list)


_STATE = _State()


def state() -> _State:
    return _STATE


def load_config() -> dict:
    """Config reads are safe and must not raise: the admin page renders the
    section before anyone asks the engine to trade."""
    return {"enabled": False, "available": False, "note": _MESSAGE}


# Everything that would touch a live book or a broker connection.
connect = _unavailable
submit = _unavailable
save_config = _unavailable
set_kill_switch = _unavailable
account_snapshot = _unavailable
open_positions = _unavailable
order_history = _unavailable
expected_vs_live = _unavailable
