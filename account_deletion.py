"""A student's own data: download all of it, or delete the account and everything with it.

Deletion removes, in one go: every row of theirs in the app database (courses, sources, uploads,
chats, notes, progress, map, rewards, quizzes, feedback, jobs, verification codes), every Content
and Student OMA item in their namespaces ("u<id>__…": course index, figures, concepts, Pedro's
memory of them), their old search collections, and their files on the disk and in R2 (permanently:
not to the 30-day trash). Kept, without the student: AI usage totals (the user id is cleared) and
the beta code they used (marked used, their email removed). Nightly database backups in R2 still
hold the account until they expire, 30 days later; the privacy policy says so.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)

SHARED = {"papers", "applied_learning_events"}      # no owner
ANONYMISE = {"ai_usage"}                            # totals stay, the student goes
# Not part of a download: internal bookkeeping, derived indexes, secrets and very large text.
EXPORT_SKIP_TABLES = {"ai_usage", "learning_jobs", "source_search_indexes", "map_tile_provenance", "map_snapshots",
                      "email_verifications", "password_resets", "beta_codes", "papers", "applied_learning_events"}
EXPORT_SKIP_COLUMNS = {"password_hash", "google_id", "embedding", "vectors", "passages_json", "raw_text", "lease_token"}
STUDENT_STORES = ("episode", "pattern", "concept_mastery", "academic_identity", "active_context")


def _tables(conn) -> list[str]:
    return [t for (t,) in conn.execute("select name from sqlite_master where type='table' and name not like 'sqlite_%'")]


def _columns(conn, table) -> list[str]:
    return [r[1] for r in conn.execute(f'pragma table_info("{table}")')]


def _owned(prefix_len: int) -> str:
    # substr, not LIKE: "_" is a LIKE wildcard, and "u1__%" would also match user 12's namespaces
    return f"substr(namespace, 1, {prefix_len}) = ?"


def _oma_paths(oma, prefix: str) -> set[Path]:
    paths = set()
    for t in _tables(oma):
        if t.endswith("_items") and "namespace" in _columns(oma, t) and "store_specific" in _columns(oma, t):
            for (raw,) in oma.execute(f'select store_specific from "{t}" where {_owned(len(prefix))}', (prefix,)):
                try:
                    fp = (json.loads(raw or "{}") or {}).get("file_path")
                except ValueError:
                    fp = None
                if fp:
                    paths.add(Path(fp))
    return paths


def delete_account(user_id: int) -> dict:
    """Delete the account and everything held about the student. Returns what was removed."""
    import database
    import file_store
    import oma_provider
    from coast_content_oma.normalized_source import cache_dir

    uid = int(user_id)
    prefix = f"u{uid}__"
    db = sqlite3.connect(database.DB_PATH, timeout=60)
    oma = sqlite3.connect(oma_provider.OMA_DB_PATH, timeout=60)
    removed: dict[str, int] = {}
    try:
        row = db.execute("select email from users where id = ?", (uid,)).fetchone()
        if not row:
            return {"deleted": False}
        email = row[0]
        source_ids = [s for (s,) in db.execute("select source_id from folder_sources where user_id = ?", (uid,))]
        files: set[Path] = set()
        for (fp,) in db.execute("select file_path from folder_sources where user_id = ? and file_path is not null", (uid,)):
            files |= {Path(fp), cache_dir(fp)}
        for (fp,) in db.execute("select image_path from source_images where user_id = ? and image_path is not null", (uid,)):
            files.add(Path(fp))
        files |= _oma_paths(oma, prefix)

        tables = _tables(db)
        ordered = sorted(tables, key=lambda t: (t != "session_answers", t == "users"))  # answers before sessions, users last
        with db:
            for t in ordered:
                cols = set(_columns(db, t))
                if t in SHARED:
                    continue
                if t == "users":
                    n = db.execute("delete from users where id = ?", (uid,)).rowcount
                elif t == "session_answers":
                    n = db.execute("delete from session_answers where session_id in "
                                   "(select id from quiz_sessions where user_id = ?)", (uid,)).rowcount
                elif t == "source_search_indexes":
                    n = db.executemany("delete from source_search_indexes where source_id = ?",
                                       [(s,) for s in source_ids]).rowcount if source_ids else 0
                elif t == "learning_jobs":
                    n = db.execute("delete from learning_jobs where json_extract(payload_json, '$.user_id') = ?", (uid,)).rowcount
                elif t == "beta_codes":
                    n = db.execute("update beta_codes set used_email = '', used_by_user_id = null "
                                   "where used_by_user_id = ?", (uid,)).rowcount
                elif t in ("email_verifications", "password_resets") and "email" in cols:
                    n = db.execute(f'delete from "{t}" where lower(email) = lower(?)', (email,)).rowcount
                elif t in ANONYMISE and "user_id" in cols:
                    n = db.execute(f'update "{t}" set user_id = null where user_id = ?', (uid,)).rowcount
                elif "user_id" in cols:
                    n = db.execute(f'delete from "{t}" where user_id = ?', (uid,)).rowcount
                else:
                    continue
                if n and n > 0:
                    removed[t] = n
        with oma:
            for t in _tables(oma):
                if "namespace" in _columns(oma, t):
                    n = oma.execute(f'delete from "{t}" where {_owned(len(prefix))}', (prefix,)).rowcount
                    if n > 0:
                        removed[f"oma.{t}"] = n
        _drop_chroma(uid)
        file_store.remove(sorted(files), trash=False)
        removed["files"] = len([f for f in files])
        log.info("account %s deleted: %s", uid, removed)
        return {"deleted": True, "removed": removed}
    finally:
        db.close()
        oma.close()


def _drop_chroma(uid: int) -> None:
    """The legacy search index kept one collection per course, named "u<id>_<course>"."""
    try:
        from rag import CHROMA_PATH, _get_chroma
        if not Path(CHROMA_PATH).is_dir():
            return
        client = _get_chroma()
        for col in client.list_collections():
            name = col if isinstance(col, str) else col.name
            if name.startswith(f"u{uid}_"):
                client.delete_collection(name)
    except Exception:
        log.exception("legacy search index: could not drop the collections of user %s", uid)


def export_account(user_id: int) -> dict:
    """Everything held about the student, as data they can read and take elsewhere."""
    import database
    import oma_provider
    uid = int(user_id)
    prefix = f"u{uid}__"
    db = sqlite3.connect(f"file:{database.DB_PATH}?mode=ro", uri=True, timeout=30)
    db.row_factory = sqlite3.Row
    oma = sqlite3.connect(f"file:{oma_provider.OMA_DB_PATH}?mode=ro", uri=True, timeout=30)
    oma.row_factory = sqlite3.Row
    try:
        out: dict = {"about": "Everything Coast holds about you. Your uploaded files are not included: you have them."}
        for t in _tables(db):
            if t in EXPORT_SKIP_TABLES:
                continue
            cols = _columns(db, t)
            if t == "users":
                rows = db.execute("select * from users where id = ?", (uid,)).fetchall()
            elif t == "session_answers":
                rows = db.execute("select * from session_answers where session_id in "
                                  "(select id from quiz_sessions where user_id = ?)", (uid,)).fetchall()
            elif "user_id" in cols:
                rows = db.execute(f'select * from "{t}" where user_id = ?', (uid,)).fetchall()
            else:
                continue
            if rows:
                out[t] = [{k: _plain(r[k]) for k in r.keys() if k not in EXPORT_SKIP_COLUMNS} for r in rows]
        memory = {}
        for store in STUDENT_STORES:
            table = f"{store}_items"
            if table not in _tables(oma):
                continue
            rows = oma.execute(f'select namespace, content, created_at, store_specific from "{table}" '
                               f'where {_owned(len(prefix))} and superseded_by is null', (prefix,)).fetchall()
            if rows:
                memory[store] = [{"course": r["namespace"][len(prefix):], "content": r["content"], "created_at": r["created_at"],
                                  "details": _plain(r["store_specific"])} for r in rows]
        if memory:
            out["what_pedro_remembers"] = memory
        return out
    finally:
        db.close()
        oma.close()


def _plain(value):
    if isinstance(value, bytes):
        return None
    if isinstance(value, str) and value[:1] in "[{":
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value
