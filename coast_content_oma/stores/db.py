"""SQLite connections with optional atomic multi-store transactions."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import hashlib
import sqlite3

# Several files are indexed at once, each writing batches of pages and their search entries; a
# writer waits its turn for up to 30 s instead of failing ("database is locked" at 5 s under load).
BUSY_TIMEOUT_MS = 30000
_active = ContextVar('oma_transaction', default=None)
_wal_ready = set()

def fts_rowid(item_id: str) -> int:
    """An item's row number in its full-text table, derived from its id. FTS5 finds a row by
    number at once, but by id only by reading every row: 0.01 ms against 10 ms per lookup at
    50,000 rows, with every course's pages in one table and writers waiting on each other."""
    return int.from_bytes(hashlib.blake2b(item_id.encode(), digest_size=8).digest(), 'big') >> 1


def key_fts_rows(conn, table: str, fts: str) -> None:
    """Once per full-text table: number its existing rows by their ids (rows written before
    fts_rowid had numbers in insertion order). Keeps the last row of any id written twice."""
    conn.execute('CREATE TABLE IF NOT EXISTS fts_keyed (name TEXT PRIMARY KEY)')
    if conn.execute('SELECT 1 FROM fts_keyed WHERE name=?', (fts,)).fetchone():
        return
    conn.create_function('fts_rowid', 1, fts_rowid, deterministic=True)
    conn.execute(f'CREATE TEMP TABLE fts_copy AS SELECT max(rowid), id, namespace, content, entities, tags FROM {fts} GROUP BY id')
    conn.execute(f'DELETE FROM {fts}')
    conn.execute(f'INSERT INTO {fts}(rowid, id, namespace, content, entities, tags) '
                 f'SELECT fts_rowid(id), id, namespace, content, entities, tags FROM fts_copy')
    conn.execute('DROP TABLE fts_copy')
    conn.execute('INSERT OR IGNORE INTO fts_keyed VALUES (?)', (fts,))  # another process may have just done it


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
