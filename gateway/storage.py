"""SQLite-backed API key registry and usage ledger (stdlib sqlite3, no ORM)."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

KEY_PREFIX = "mm-"

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    id          INTEGER PRIMARY KEY,
    key_hash    TEXT UNIQUE NOT NULL,
    prefix      TEXT NOT NULL,
    name        TEXT NOT NULL,
    ip_allow    TEXT,
    rate_per_minute INTEGER,
    created_at  REAL NOT NULL,
    revoked_at  REAL,
    backends    TEXT
);
CREATE TABLE IF NOT EXISTS usage (
    id          INTEGER PRIMARY KEY,
    ts          REAL NOT NULL,
    request_id  TEXT NOT NULL,
    key_id      INTEGER NOT NULL,
    path        TEXT NOT NULL,
    model_requested TEXT,
    model_actual    TEXT,
    backend     TEXT,
    status      INTEGER NOT NULL,
    stream      INTEGER NOT NULL DEFAULT 0,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    usage_estimated   INTEGER NOT NULL DEFAULT 0,
    latency_ms  REAL NOT NULL,
    ttft_ms     REAL,
    fallback_from TEXT
);
CREATE INDEX IF NOT EXISTS usage_ts ON usage(ts);
CREATE INDEX IF NOT EXISTS usage_key ON usage(key_id, ts);
"""


@dataclass(frozen=True)
class KeyRecord:
    id: int
    prefix: str
    name: str
    ip_allow: tuple[str, ...] | None
    rate_per_minute: int | None
    created_at: float
    revoked_at: float | None
    backends: tuple[str, ...] | None = None  # None = any backend; else only these (no fallback outside)

    @property
    def revoked(self) -> bool:
        return self.revoked_at is not None


@dataclass(frozen=True)
class UsageRow:
    request_id: str
    key_id: int
    path: str
    model_requested: str | None
    model_actual: str | None
    backend: str | None
    status: int
    stream: bool
    prompt_tokens: int
    completion_tokens: int
    usage_estimated: bool
    latency_ms: float
    ttft_ms: float | None
    fallback_from: str | None = None
    ts: float | None = None


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class Storage:
    """Thread-safe (single connection + lock); fine for a gateway's write rate."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(api_keys)")}
        if "backends" not in cols:  # migrate ledgers created before per-key backend pinning
            self._conn.execute("ALTER TABLE api_keys ADD COLUMN backends TEXT")
            self._conn.commit()
        self._lock = threading.Lock()

    def close(self) -> None:
        self._conn.close()

    # ---- keys -------------------------------------------------------------

    def create_key(self, name: str, ip_allow: list[str] | None = None,
                   rate_per_minute: int | None = None,
                   backends: list[str] | None = None) -> tuple[str, KeyRecord]:
        """Create a key. Returns (plaintext_key, record). Plaintext is never stored."""
        plain = KEY_PREFIX + secrets.token_hex(24)
        prefix = plain[: len(KEY_PREFIX) + 8]
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO api_keys(key_hash,prefix,name,ip_allow,rate_per_minute,created_at,backends)"
                " VALUES (?,?,?,?,?,?,?)",
                (_hash(plain), prefix, name,
                 json.dumps(ip_allow) if ip_allow else None, rate_per_minute, time.time(),
                 json.dumps(backends) if backends else None),
            )
            self._conn.commit()
            rec = self.get_key(cur.lastrowid)
        assert rec is not None
        return plain, rec

    def verify_key(self, plain: str) -> KeyRecord | None:
        row = self._conn.execute(
            "SELECT * FROM api_keys WHERE key_hash=?", (_hash(plain),)).fetchone()
        return self._row_to_key(row) if row else None

    def get_key(self, key_id: int) -> KeyRecord | None:
        row = self._conn.execute("SELECT * FROM api_keys WHERE id=?", (key_id,)).fetchone()
        return self._row_to_key(row) if row else None

    def list_keys(self) -> list[KeyRecord]:
        rows = self._conn.execute("SELECT * FROM api_keys ORDER BY id").fetchall()
        return [self._row_to_key(r) for r in rows]

    def set_backends(self, key_id: int, backends: list[str] | None) -> bool:
        """Pin a key to these backends (None/empty = unrestricted)."""
        with self._lock:
            cur = self._conn.execute("UPDATE api_keys SET backends=? WHERE id=?",
                                     (json.dumps(backends) if backends else None, key_id))
            self._conn.commit()
        return cur.rowcount == 1

    def revoke_key(self, key_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE api_keys SET revoked_at=? WHERE id=? AND revoked_at IS NULL",
                (time.time(), key_id))
            self._conn.commit()
        return cur.rowcount == 1

    @staticmethod
    def _row_to_key(row: sqlite3.Row) -> KeyRecord:
        ip = json.loads(row["ip_allow"]) if row["ip_allow"] else None
        be = json.loads(row["backends"]) if row["backends"] else None
        return KeyRecord(
            id=row["id"], prefix=row["prefix"], name=row["name"],
            ip_allow=tuple(ip) if ip else None,
            rate_per_minute=row["rate_per_minute"],
            created_at=row["created_at"], revoked_at=row["revoked_at"],
            backends=tuple(be) if be else None,
        )

    # ---- usage ------------------------------------------------------------

    def record_usage(self, u: UsageRow) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO usage(ts,request_id,key_id,path,model_requested,model_actual,backend,"
                "status,stream,prompt_tokens,completion_tokens,usage_estimated,latency_ms,ttft_ms,fallback_from)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (u.ts or time.time(), u.request_id, u.key_id, u.path, u.model_requested,
                 u.model_actual, u.backend, u.status, int(u.stream), u.prompt_tokens,
                 u.completion_tokens, int(u.usage_estimated), u.latency_ms, u.ttft_ms,
                 u.fallback_from),
            )
            self._conn.commit()

    def usage_report(self, since_ts: float, by: str = "key") -> list[dict]:
        group = {
            "key": "k.name",
            "model": "u.model_actual",
            "backend": "u.backend",
            "day": "date(u.ts,'unixepoch')",
        }[by]
        rows = self._conn.execute(
            f"SELECT {group} AS grp, COUNT(*) AS requests,"
            " SUM(u.prompt_tokens) AS prompt_tokens, SUM(u.completion_tokens) AS completion_tokens,"
            " SUM(u.usage_estimated) AS estimated_rows,"
            " SUM(CASE WHEN u.status>=400 THEN 1 ELSE 0 END) AS errors,"
            " AVG(u.latency_ms) AS avg_latency_ms"
            " FROM usage u LEFT JOIN api_keys k ON k.id=u.key_id"
            " WHERE u.ts>=? GROUP BY grp ORDER BY requests DESC",
            (since_ts,),
        ).fetchall()
        return [dict(r) for r in rows]

    def usage_rows(self, since_ts: float, key_id: int | None = None) -> list[dict]:
        q = "SELECT * FROM usage WHERE ts>=?"
        args: list = [since_ts]
        if key_id is not None:
            q += " AND key_id=?"
            args.append(key_id)
        return [dict(r) for r in self._conn.execute(q + " ORDER BY ts", args).fetchall()]
