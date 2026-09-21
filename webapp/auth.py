"""Users, sessions, and the admin approval gate.

Accounts are self-registered but inert until an admin approves them: a new user
can sign in and will be told they are pending, but no data route will serve them
anything. That ordering matters -- this application exposes client P&L, routing
decisions and live risk, so "approved by default" would be the wrong failure
mode.

NOTE ON SHARED CREDENTIALS (2026-09-18): the team is currently onboarded by
sharing one account's credentials rather than registering individually. Two
consequences worth knowing. There is no attribution -- every action in the audit
trail is that one username, so "who looked at this client" cannot be answered.
And THE USER TABLE IS SHARED with the full instance (both read the same
`app.db`), so credentials handed out for the beta also sign in to the full app on
its own port, where the route gate does not apply. Narrow that with `views`, or
revoke the account to cut everyone off at once.

Permissions are two independent axes rather than a role hierarchy:

* ``role``  -- ``admin`` may approve users and switch into any user's view;
  ``user`` may not.
* ``views`` -- which of the two halves of the product (Trading, Quant) the user
  may see. A quant analyst and a dealer need different things, and neither
  needs the other's screens by default.

SQLite is deliberate. The user table is small, the app is single-node, and a
file-backed database means no extra service to run before the dashboard opens.
"""

from __future__ import annotations

import secrets
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import bcrypt

DB_PATH = Path(__file__).resolve().parent / "app.db"
SESSION_TTL_SECONDS = 12 * 60 * 60

#: The two halves of the product. A user carries a subset of these.
VIEW_TRADING = "trading"
VIEW_QUANT = "quant"
ALL_VIEWS = (VIEW_TRADING, VIEW_QUANT)


@dataclass(frozen=True)
class User:
    id: int
    username: str
    email: str
    role: str
    views: tuple[str, ...]
    approved: bool
    created_at: float

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def may_see(self, view: str) -> bool:
        # An admin is never locked out of a view -- they are the ones who have
        # to diagnose it when a user reports something wrong.
        return self.is_admin or view in self.views


@contextmanager
def connect():
    """Open a connection, commit, and ALWAYS close it.

    `with sqlite3.connect(...)` is a TRANSACTION context manager, not a closing
    one -- it commits or rolls back but leaves the handle open. Every request
    here used that form, so handles accumulated for the life of the process and
    contended for the write lock, which is a slow, intermittent failure rather
    than an obvious one. WAL mode additionally lets readers proceed while a
    write is in flight, which matters because the Kafka consumer and the web
    requests share this file.
    """
    connection = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialise() -> None:
    """Create the schema and seed the two bootstrap accounts."""
    with connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT NOT NULL UNIQUE,
                email         TEXT NOT NULL DEFAULT '',
                password_hash TEXT NOT NULL,
                role          TEXT NOT NULL DEFAULT 'user',
                views         TEXT NOT NULL DEFAULT 'trading',
                approved      INTEGER NOT NULL DEFAULT 0,
                created_at    REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token      TEXT PRIMARY KEY,
                user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at REAL NOT NULL
            );
            -- Per-user data-source credentials. Kept per user rather than global
            -- so each person connects with their own database login and nobody
            -- inherits someone else's access.
            CREATE TABLE IF NOT EXISTS data_sources (
                user_id     INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                kind        TEXT NOT NULL DEFAULT 'mysql',
                host        TEXT NOT NULL DEFAULT '',
                port        INTEGER NOT NULL DEFAULT 3306,
                username    TEXT NOT NULL DEFAULT '',
                password    TEXT NOT NULL DEFAULT '',
                bq_project  TEXT NOT NULL DEFAULT '',
                updated_at  REAL NOT NULL DEFAULT 0
            );
            -- Small key/value store for settings an admin edits at runtime
            -- (currently the IP allowlist). It lives in app.db rather than a
            -- config file so it is covered by the same git-ignore as the user
            -- table -- an allowlist names internal network ranges and should
            -- not travel with the source.
            CREATE TABLE IF NOT EXISTS settings (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL DEFAULT '',
                updated_at REAL NOT NULL DEFAULT 0,
                updated_by TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id);
            """
        )
        seeded = connection.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
        if not seeded:
            # Bootstrap: one admin, and the shared test account the brief asks
            # for. Both are pre-approved because there would otherwise be nobody
            # able to approve anyone.
            _insert(connection, "admin", "admin@local", "admin", "admin", ALL_VIEWS, True)
            _insert(connection, "test", "test@local", "test", "user", ALL_VIEWS, True)


def _insert(connection, username, email, password, role, views, approved) -> int:
    cursor = connection.execute(
        "INSERT INTO users (username, email, password_hash, role, views, approved, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (username, email, hash_password(password), role, ",".join(views), int(approved), time.time()),
    )
    return int(cursor.lastrowid)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("ascii"))
    except (ValueError, TypeError):
        return False


def _row_to_user(row: sqlite3.Row) -> User:
    return User(
        id=row["id"], username=row["username"], email=row["email"], role=row["role"],
        views=tuple(v for v in row["views"].split(",") if v),
        approved=bool(row["approved"]), created_at=row["created_at"],
    )


def register(username: str, email: str, password: str, views: tuple[str, ...]) -> tuple[bool, str]:
    """Create a PENDING account. Returns (ok, message)."""
    username = username.strip().lower()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    requested = tuple(v for v in views if v in ALL_VIEWS) or (VIEW_TRADING,)
    try:
        with connect() as connection:
            _insert(connection, username, email.strip(), password, "user", requested, False)
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    return True, "Account created. An administrator must approve it before you can sign in."


def authenticate(username: str, password: str) -> User | None:
    with connect() as connection:
        # Compare case-insensitively on BOTH sides. Lowercasing only the input
        # meant any account stored with a capital -- which `register` cannot
        # create but an operator renaming a row certainly can -- could never log
        # in, with the generic "Incorrect username or password" as the only
        # clue. Names keep the casing they were stored with, for display.
        row = connection.execute(
            "SELECT * FROM users WHERE lower(username) = ?", (username.strip().lower(),)
        ).fetchone()
    if row is None or not verify_password(password, row["password_hash"]):
        return None
    return _row_to_user(row)


def start_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with connect() as connection:
        connection.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))
        connection.execute(
            "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
            (token, user_id, time.time() + SESSION_TTL_SECONDS),
        )
    return token


def user_for_session(token: str | None) -> User | None:
    if not token:
        return None
    with connect() as connection:
        row = connection.execute(
            "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id"
            " WHERE s.token = ? AND s.expires_at > ?",
            (token, time.time()),
        ).fetchone()
    return _row_to_user(row) if row else None


def end_session(token: str | None) -> None:
    if token:
        with connect() as connection:
            connection.execute("DELETE FROM sessions WHERE token = ?", (token,))


def list_users() -> list[User]:
    with connect() as connection:
        rows = connection.execute("SELECT * FROM users ORDER BY approved, created_at DESC").fetchall()
    return [_row_to_user(row) for row in rows]


def get_user(user_id: int) -> User | None:
    with connect() as connection:
        row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _row_to_user(row) if row else None


def set_approval(user_id: int, approved: bool) -> None:
    with connect() as connection:
        connection.execute("UPDATE users SET approved = ? WHERE id = ?", (int(approved), user_id))
        if not approved:
            # Revoking access must also drop live sessions, or the user keeps
            # working until their cookie happens to expire.
            connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))


def set_permissions(user_id: int, role: str, views: tuple[str, ...]) -> None:
    role = role if role in ("admin", "user") else "user"
    allowed = ",".join(v for v in views if v in ALL_VIEWS)
    with connect() as connection:
        connection.execute("UPDATE users SET role = ?, views = ? WHERE id = ?", (role, allowed, user_id))


def delete_user(user_id: int) -> None:
    with connect() as connection:
        connection.execute("DELETE FROM users WHERE id = ?", (user_id,))


def session_counts() -> dict[int, int]:
    """Live sessions per user id, for the console. Expired rows are excluded --
    showing a count that includes dead cookies would overstate who is actually
    signed in."""
    try:
        with connect() as connection:
            rows = connection.execute(
                "SELECT user_id, COUNT(*) AS n FROM sessions WHERE expires_at > ?"
                " GROUP BY user_id", (time.time(),)).fetchall()
        return {int(r["user_id"]): int(r["n"]) for r in rows}
    except Exception:
        return {}


def get_setting(key: str, default: str = "") -> str:
    try:
        with connect() as connection:
            row = connection.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row is not None else default
    except Exception:
        # A missing table (an app.db predating this schema) must not take the
        # site down: fall back to the default, which is the env-var behaviour.
        return default


def setting_meta(key: str) -> dict:
    try:
        with connect() as connection:
            row = connection.execute(
                "SELECT value, updated_at, updated_by FROM settings WHERE key = ?",
                (key,)).fetchone()
        return dict(row) if row is not None else {}
    except Exception:
        return {}


def set_setting(key: str, value: str, by: str = "") -> None:
    with connect() as connection:
        connection.execute(
            "INSERT INTO settings (key, value, updated_at, updated_by) VALUES (?, ?, ?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value,"
            " updated_at = excluded.updated_at, updated_by = excluded.updated_by",
            (key, value, time.time(), by))


def revoke_sessions(user_id: int | None = None) -> int:
    """Sign a user out everywhere, or everyone when `user_id` is None.

    Returns how many sessions were dropped, because "signed out 14 people" and
    "signed out nobody, the cookie you are worried about was already gone" need
    to read differently in the console.
    """
    with connect() as connection:
        if user_id is None:
            n = connection.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"]
            connection.execute("DELETE FROM sessions")
        else:
            n = connection.execute(
                "SELECT COUNT(*) AS n FROM sessions WHERE user_id = ?",
                (user_id,)).fetchone()["n"]
            connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
    return int(n)


def set_password(user_id: int, password: str, keep_sessions: bool = False) -> tuple[bool, str]:
    """Set a password and, by default, sign that user out everywhere.

    Dropping the sessions is the default because a password change that leaves
    live cookies working does not actually remove anyone's access -- which is
    the usual reason for changing it. `keep_sessions=True` is for the case where
    an admin is rotating their OWN password and does not want to be logged out
    mid-task.
    """
    if len(password or "") < 6:
        return False, "Password must be at least 6 characters."
    with connect() as connection:
        row = connection.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            return False, "No such user."
        connection.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                           (hash_password(password), user_id))
        dropped = 0
        if not keep_sessions:
            dropped = connection.execute(
                "SELECT COUNT(*) AS n FROM sessions WHERE user_id = ?",
                (user_id,)).fetchone()["n"]
            connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
    if keep_sessions:
        return True, "Password changed; existing sessions kept."
    return True, f"Password changed; {dropped} session(s) signed out."


def rename_user(user_id: int, username: str) -> tuple[bool, str]:
    """Change a username. Case is preserved for display; uniqueness is checked
    case-INSENSITIVELY, because `authenticate` matches that way and two names
    differing only in case would otherwise both resolve to one login."""
    username = (username or "").strip()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    with connect() as connection:
        clash = connection.execute(
            "SELECT id FROM users WHERE lower(username) = ? AND id <> ?",
            (username.lower(), user_id)).fetchone()
        if clash is not None:
            return False, "That username is already taken."
        connection.execute("UPDATE users SET username = ? WHERE id = ?", (username, user_id))
    return True, f"Username changed to {username}."


def save_data_source(user_id: int, **fields) -> None:
    """Store this user's own database credentials. MySQL is the default source."""
    with connect() as connection:
        connection.execute(
            "INSERT INTO data_sources (user_id, kind, host, port, username, password, bq_project, updated_at)"
            " VALUES (:user_id, :kind, :host, :port, :username, :password, :bq_project, :updated_at)"
            " ON CONFLICT(user_id) DO UPDATE SET kind=:kind, host=:host, port=:port,"
            " username=:username, password=:password, bq_project=:bq_project, updated_at=:updated_at",
            {
                "user_id": user_id,
                "kind": fields.get("kind", "mysql"),
                "host": fields.get("host", ""),
                "port": int(fields.get("port") or 3306),
                "username": fields.get("username", ""),
                "password": fields.get("password", ""),
                "bq_project": fields.get("bq_project", ""),
                "updated_at": time.time(),
            },
        )


def get_data_source(user_id: int) -> dict:
    with connect() as connection:
        row = connection.execute("SELECT * FROM data_sources WHERE user_id = ?", (user_id,)).fetchone()
    return dict(row) if row else {"kind": "mysql", "host": "", "port": 3306,
                                  "username": "", "password": "", "bq_project": ""}
