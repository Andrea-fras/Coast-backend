"""Give the eval student their own private copy of an existing uploaded course:
outline, sources and Content OMA material (concepts, text chunks, figures).

Every id is renamed so nothing is shared with the original owner — the copy is
exactly what the student would have after uploading those sources themselves.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from database import SessionLocal, CourseOutline, FolderSource, StudyFolder
from coast_content_oma import course_identity
from coast_content_oma.stores import make_namespace
from coast_content_oma.stores.db import connect_db

TABLES = ("concept_items", "content_items", "image_items")


def _rewrite(value, id_map: dict):
    if isinstance(value, str):
        return id_map.get(value, value)
    if isinstance(value, list):
        return [_rewrite(v, id_map) for v in value]
    if isinstance(value, dict):
        return {k: _rewrite(v, id_map) for k, v in value.items()}
    return value


def copy_course(oma_db_path, src_uid: int, src_folder: str, dst_uid: int, dst_folder: str) -> dict:
    suffix = f"__e{dst_uid}"
    with SessionLocal() as db:
        src_key = course_identity.namespace_key(src_uid, src_folder)
        outline = db.query(CourseOutline).filter_by(user_id=src_uid, folder_name=src_folder).one()
        outline_json, outline_minutes = outline.outline_json, outline.estimated_minutes
        sources = [{c: getattr(f, c) for c in ("source_id", "title", "filename", "source_type", "page_count",
                                                "raw_text", "file_path", "oma_ingest_status")}
                   for f in db.query(FolderSource).filter_by(user_id=src_uid, folder_name=src_folder)]
        course_identity.register(db, dst_uid, dst_folder)
        db.commit()
        dst_key = course_identity.namespace_key(dst_uid, dst_folder)
    src_ns, dst_ns = f"u{src_uid}__{src_key}", make_namespace(dst_uid, dst_folder)
    assert dst_ns == f"u{dst_uid}__{dst_key}"

    id_map = {s["source_id"]: f"{s['source_id']}{suffix}"[:50] for s in sources}
    with connect_db(oma_db_path) as conn:
        rows = {t: conn.execute(f"SELECT * FROM {t} WHERE namespace = ?", (src_ns,)).fetchall() for t in TABLES}
        for t in TABLES:
            for r in rows[t]:
                id_map[r[0]] = f"{r[0]}{suffix}"
        counts = {}
        for t in TABLES:
            for r in rows[t]:
                (_id, _ns, content, created, accessed, n_access, importance, source_doc,
                 entities, tags, superseded, store_specific, embedding) = r
                ents = _rewrite(json.loads(entities or "[]"), id_map)
                ss = _rewrite(json.loads(store_specific or "{}"), id_map)
                tag_list = json.loads(tags or "[]")
                conn.execute(
                    f"INSERT OR REPLACE INTO {t} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (id_map[_id], dst_ns, content, created, accessed, 0, importance,
                     id_map.get(source_doc, source_doc), json.dumps(ents), json.dumps(tag_list),
                     id_map.get(superseded, superseded), json.dumps(ss), embedding),
                )
                conn.execute(f"INSERT INTO {t.replace('_items', '_items_fts')} (id, namespace, content, entities, tags) "
                             f"VALUES (?,?,?,?,?)", (id_map[_id], dst_ns, content, " ".join(map(str, ents)),
                                                     " ".join(map(str, tag_list))))
            counts[t] = len(rows[t])

    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        if not db.query(StudyFolder).filter_by(user_id=dst_uid, name=dst_folder).first():
            db.add(StudyFolder(user_id=dst_uid, name=dst_folder, created_at=now))
        for s in sources:
            db.add(FolderSource(user_id=dst_uid, folder_name=dst_folder, created_at=now,
                                **{**s, "source_id": id_map[s["source_id"]]}))
        sections = _rewrite(json.loads(outline_json), id_map)
        db.add(CourseOutline(user_id=dst_uid, folder_name=dst_folder, outline_json=json.dumps(sections),
                             total_sections=len(sections), current_section=0,
                             estimated_minutes=outline_minutes, created_at=now, updated_at=now))
        db.commit()
    return {"namespace": dst_ns, "sections": len(sections), "sources": len(sources), **counts}
