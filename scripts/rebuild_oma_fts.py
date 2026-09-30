#!/usr/bin/env python3
"""Rebuild derived full-text indexes from canonical OMA rows, preserving learning records."""
import argparse
import sqlite3
from contextlib import closing
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--db',type=Path,required=True)
parser.add_argument('--apply',action='store_true',help='Rebuild indexes; default only reports duplicate postings')
args=parser.parse_args()
if not args.db.is_file(): parser.error('Database does not exist')
mode='rw' if args.apply else 'ro'
with closing(sqlite3.connect(f'file:{args.db.resolve()}?mode={mode}',uri=True)) as conn:
    conn.execute('PRAGMA busy_timeout=5000')
    tables={r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for store in ('concept','content','image','episode','pattern','concept_mastery','academic_identity','active_context'):
        table,fts=f'{store}_items',f'{store}_items_fts'
        if table not in tables or fts not in tables: continue
        count,unique=conn.execute(f'SELECT count(*), count(DISTINCT id) FROM {fts}').fetchone()
        print(f'{store}: {count-unique} duplicate postings')
        if args.apply:
            with conn:
                conn.execute(f'DELETE FROM {fts}')
                conn.execute(f'INSERT INTO {fts}(id,namespace,content,entities,tags) SELECT id,namespace,content,entities,tags FROM {table}')
            print(f'{store}: rebuilt')
