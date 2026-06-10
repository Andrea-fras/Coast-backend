"""Shared SQLite connection settings for all OMA stores."""

from __future__ import annotations

import sqlite3
from pathlib import Path

# Milliseconds to wait on "database is locked" before failing.
BUSY_TIMEOUT_MS = 5000


def connect_db(db_path: Path | str) -> sqlite3.Connection:
    """Open SQLite with WAL mode and a busy timeout for concurrent writers."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    return conn
