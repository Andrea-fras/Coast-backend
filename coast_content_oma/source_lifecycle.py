"""Remove deleted source material from live retrieval without erasing learner episodes."""
from pathlib import Path
from .stores.db import transaction


def remove_source_material(db_path, namespace, source_id):
    path = Path(db_path)
    if not path.exists():
        return
    doc_id = 'doc_' + source_id
    with transaction(path) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'source_page_progress' in tables:
            conn.execute('DELETE FROM source_page_progress WHERE namespace=? AND source_id=?', (namespace, doc_id))
        # Canonical concepts may be shared by other sources; student evidence is independent.
        for store in ('content', 'image'):
            table, fts = f'{store}_items', f'{store}_items_fts'
            if table not in tables:
                continue
            if fts in tables:
                conn.execute(f'DELETE FROM {fts} WHERE id IN (SELECT id FROM {table} WHERE namespace=? AND source_doc_id=?)', (namespace, doc_id))
            conn.execute(f'DELETE FROM {table} WHERE namespace=? AND source_doc_id=?', (namespace, doc_id))
