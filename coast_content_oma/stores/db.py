"""SQLite connections with optional atomic multi-store transactions."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import sqlite3

BUSY_TIMEOUT_MS = 5000
_active = ContextVar('oma_transaction', default=None)
_wal_ready = set()

@contextmanager
def connect_db(db_path):
    path = str(Path(db_path).resolve())
    active = _active.get()
    if active and active[0] == path:
        yield active[1]
        return
    conn = sqlite3.connect(path)
    try:
        conn.execute(f'PRAGMA busy_timeout={BUSY_TIMEOUT_MS}')
        conn.execute('PRAGMA synchronous=NORMAL')  # safe under WAL; skips an fsync per commit
        if path not in _wal_ready:  # WAL is a persistent file setting; set it once per process
            conn.execute('PRAGMA journal_mode=WAL')
            _wal_ready.add(path)
        with conn:
            yield conn
    finally:
        conn.close()

@contextmanager
def transaction(db_path):
    path = str(Path(db_path).resolve())
    with connect_db(path) as conn:
        if _active.get():
            raise RuntimeError('Nested OMA projection transaction')
        conn.execute('BEGIN IMMEDIATE')
        token = _active.set((path, conn))
        try:
            yield conn
        finally:
            _active.reset(token)


@contextmanager
def atomic(db_path):
    """Join the caller's OMA transaction if one is open, else start one."""
    path = str(Path(db_path).resolve())
    active = _active.get()
    if active and active[0] == path:
        yield active[1]
        return
    with transaction(path) as conn:
        yield conn
