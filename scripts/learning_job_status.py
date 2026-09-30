#!/usr/bin/env python3
"""Inspect durable work without importing the app or contacting model providers."""
import argparse
import sqlite3
import time
from contextlib import closing
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--db', required=True, type=Path, help='Application SQLite database (coast.db)')
parser.add_argument('--retry-failed', action='store_true', help='Explicitly requeue failed jobs for the running worker')
args = parser.parse_args()
if not args.db.is_file(): parser.error('Database does not exist')
mode = 'rw' if args.retry_failed else 'ro'
with closing(sqlite3.connect(f'file:{args.db.resolve()}?mode={mode}', uri=True)) as conn:
    conn.execute('PRAGMA busy_timeout=5000')
    if args.retry_failed:
        with conn:
            changed = conn.execute("UPDATE learning_jobs SET status='queued', attempts=0, available_at=0, lease_until=0 WHERE status='failed'").rowcount
        print(f'Requeued {changed} failed jobs. The worker will pick them up.')
    for status, count in conn.execute('SELECT status, count(*) FROM learning_jobs GROUP BY status ORDER BY status'):
        print(f'{status}: {count}')
    expired = conn.execute("SELECT count(*) FROM learning_jobs WHERE status='running' AND lease_until < ?", (time.time(),)).fetchone()[0]
    print(f'expired leases awaiting recovery: {expired}')
