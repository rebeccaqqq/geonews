"""SQLite store of trade proposals awaiting approval."""

from __future__ import annotations

import json
import secrets
import sqlite3
import string
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .rebalance import Trade

ALPHABET = string.ascii_uppercase.replace("O", "").replace("I", "") + "23456789"


@dataclass
class Proposal:
    code: str
    created_at: datetime
    expires_at: datetime
    status: str  # pending | approved | rejected | executed | expired | failed
    trades: list[Trade]
    summary: str
    result: str = ""

    @property
    def expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at


class ApprovalStore:
    def __init__(self, state_dir: str | Path):
        self.path = Path(state_dir) / "proposals.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS proposals (
                    code TEXT PRIMARY KEY, created_at TEXT, expires_at TEXT,
                    status TEXT, trades TEXT, summary TEXT, result TEXT DEFAULT '')"""
            )

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, isolation_level="IMMEDIATE")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create(self, trades: list[Trade], summary: str, ttl_minutes: int) -> Proposal:
        code = "".join(secrets.choice(ALPHABET) for _ in range(6))
        now = datetime.now(timezone.utc)
        p = Proposal(code, now, now + timedelta(minutes=ttl_minutes), "pending", trades, summary)
        with self._conn() as c:
            # Only one live proposal at a time: supersede older pending ones.
            c.execute("UPDATE proposals SET status='expired' WHERE status='pending'")
            c.execute(
                "INSERT INTO proposals VALUES (?,?,?,?,?,?,?)",
                (code, now.isoformat(), p.expires_at.isoformat(), p.status,
                 json.dumps([t.to_dict() for t in trades]), summary, ""),
            )
        return p

    def get(self, code: str) -> Proposal | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM proposals WHERE code=?", (code.upper(),)).fetchone()
        if not row:
            return None
        return Proposal(
            row[0], datetime.fromisoformat(row[1]), datetime.fromisoformat(row[2]), row[3],
            [Trade(**t) for t in json.loads(row[4])], row[5], row[6],
        )

    def claim(self, code: str, new_status: str) -> Proposal | None:
        """Atomically move a pending, unexpired proposal to `new_status`.

        Returns the proposal if this caller won the transition, else None.
        Prevents double execution when Slack and SMS approvals race.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as c:
            cur = c.execute(
                "UPDATE proposals SET status=? WHERE code=? AND status='pending' AND expires_at > ?",
                (new_status, code.upper(), now),
            )
            won = cur.rowcount == 1
        return self.get(code) if won else None

    def finish(self, code: str, status: str, result: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE proposals SET status=?, result=? WHERE code=?", (status, result, code.upper()))

    def executed_notional_today(self) -> float:
        today = datetime.now(timezone.utc).date().isoformat()
        with self._conn() as c:
            rows = c.execute(
                "SELECT trades FROM proposals WHERE status='executed' AND substr(created_at,1,10)=?", (today,)
            ).fetchall()
        return sum(t["notional_usd"] for (js,) in rows for t in json.loads(js))
