"""Durable upload intentions and a stable source set for roadmap generation."""
import os
import time
import uuid
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import text
from database import SessionLocal, SourceUpload, FolderSource, SavedNotebook, User

UPLOAD_LEASE_SECONDS = 30 * 60
MAX_UPLOAD_MB = int(os.environ.get("COAST_MAX_UPLOAD_MB", "70"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
TOO_LARGE = f"This file is larger than {MAX_UPLOAD_MB} MB. Split it or export a smaller PDF."
# More sources than this make a roadmap with too many sections to follow.
MAX_SOURCES = int(os.environ.get("COAST_MAX_SOURCES_PER_LESSON", "8"))


def _check_room(db, user_id, folder, ids):
    """A lesson holds at most MAX_SOURCES sources: those it has, files still uploading, and these.
    Lessons that already hold more keep them; they only can't take new ones."""
    rows = _rows(db, user_id, folder).all()
    _expire(rows)
    status = {r.upload_id: r.status for r in rows}
    pending = sum(1 for r in rows if r.status in ("queued", "processing") and r.upload_id not in ids)
    incoming = sum(1 for uid in ids if status.get(uid) != "complete")
    have = db.query(FolderSource).filter_by(user_id=user_id, folder_name=folder).count()
    if incoming and have + pending + incoming > MAX_SOURCES:
        room = max(0, MAX_SOURCES - have - pending)
        raise HTTPException(409, f"A lesson can have up to {MAX_SOURCES} sources. "
                                 + (f"There's room for {room} more." if room else "Remove one to add another."))


def _check_allowance(db, user_id, folder, ids):
    """This month's files (see plans.py): those uploaded, those on their way, and these."""
    import plans
    status = {r.upload_id: r.status for r in _rows(db, user_id, folder).filter(SourceUpload.upload_id.in_(ids))}
    incoming = sum(1 for uid in ids if status.get(uid) != "complete")
    plans.check(db, db.get(User, user_id), "uploads", incoming=incoming,
                pending=plans.uploads_on_their_way(db, user_id, set(ids)))


def _rows(db, user_id, folder):
    return db.query(SourceUpload).filter_by(user_id=user_id, folder_name=folder)


def _expire(rows):
    for row in rows:
        if row.status in ("queued", "processing") and row.expires_at < time.time():
            row.status = "failed"
            row.claim = None
            row.error = "Upload interrupted. Retry this file or remove it from the upload list."


def _public(row):
    return {"upload_id": row.upload_id, "filename": row.filename, "size_bytes": row.size_bytes,
            "status": row.status, "source_id": row.source_id, "error": row.error}


def reserve(user_id, folder, files):
    if not files:
        raise HTTPException(400, "Select at least one file.")
    with SessionLocal() as db:
        db.execute(text("BEGIN IMMEDIATE"))
        ids = [f.get("upload_id") for f in files]
        _check_room(db, user_id, folder, ids)
        _check_allowance(db, user_id, folder, ids)
        out = []
        for file in files:
            uid, name, size = file["upload_id"], file["filename"], file["size_bytes"]
            if not uid or len(uid) > 80 or not name or size <= 0 or Path(name).suffix.lower() not in (".pdf", ".pptx"):
                raise HTTPException(400, "Choose a non-empty PDF or PowerPoint (.pptx) file.")
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(413, TOO_LARGE)
            row = _rows(db, user_id, folder).filter_by(upload_id=uid).first()
            if row and (row.filename != name or row.size_bytes != size):
                raise HTTPException(409, "This upload belongs to a different file. Select the file again.")
            if row:
                _expire([row])
            if not row:
                row = SourceUpload(user_id=user_id, folder_name=folder, upload_id=uid,
                                   filename=name, size_bytes=size, status="queued", expires_at=0)
                db.add(row)
            if row.status in ("queued", "failed", "cancelled"):
                row.status, row.error, row.claim = "queued", None, None
                row.expires_at = time.time() + UPLOAD_LEASE_SECONDS
            out.append(_public(row))
        db.commit()
        return out


def list_uploads(user_id, folder):
    with SessionLocal() as db:
        rows = _rows(db, user_id, folder).all()
        _expire(rows)
        result = [_public(r) for r in rows]
        db.commit()
        return result


def begin(user_id, folder, uid, filename, size):
    with SessionLocal() as db:
        db.execute(text("BEGIN IMMEDIATE"))
        row = _rows(db, user_id, folder).filter_by(upload_id=uid).first()
        if not row:
            raise HTTPException(409, "Register this file before uploading it.")
        if row.filename != filename or row.size_bytes != size:
            raise HTTPException(409, "The selected file does not match this upload.")
        _expire([row])
        if row.status == "complete":
            source = db.query(FolderSource).filter_by(source_id=row.source_id, user_id=user_id).first()
            if not source:
                raise HTTPException(409, "This source was removed. Select the file as a new upload.")
            return None, {"source_id": source.source_id, "title": source.title,
                          "page_count": source.page_count, "filename": source.filename}
        if row.status != "queued":
            raise HTTPException(409, "This file is still being processed or needs to be retried.")
        row.status, row.claim = "processing", uuid.uuid4().hex
        row.expires_at = time.time() + UPLOAD_LEASE_SECONDS
        claim = row.claim
        db.commit()
        return claim, None


def finish(db, user_id, folder, uid, claim, source_id):
    # In the same transaction as FolderSource: an abandoned request cannot publish later.
    changed = _rows(db, user_id, folder).filter_by(upload_id=uid, status="processing", claim=claim).update(
        {"status": "complete", "source_id": source_id, "claim": None, "error": None})
    if changed != 1:
        raise HTTPException(409, "This upload was removed or replaced. Its source was not added.")
    import plans
    plans.record(db, user_id, "uploads", source_id)


def fail(user_id, folder, uid, claim, message):
    with SessionLocal() as db:
        _rows(db, user_id, folder).filter_by(upload_id=uid, status="processing", claim=claim).update(
            {"status": "failed", "claim": None, "error": str(message)[:500]})
        db.commit()


def cancel(user_id, folder, uid):
    with SessionLocal() as db:
        db.execute(text("BEGIN IMMEDIATE"))
        row = _rows(db, user_id, folder).filter_by(upload_id=uid).first()
        if not row:
            raise HTTPException(404, "Upload not found.")
        if row.status == "complete":
            raise HTTPException(409, "This file has finished uploading. Remove it from Sources instead.")
        row.status, row.claim = "cancelled", None
        db.commit()


def assert_ready(db, user_id, folder):
    rows = _rows(db, user_id, folder).all()
    _expire(rows)
    if any(r.status not in ("complete", "cancelled") for r in rows):
        raise HTTPException(409, "Finish or remove the pending uploads before generating your roadmap.")


def source_signature(db, user_id, folder):
    documents = db.query(FolderSource.source_id).filter_by(user_id=user_id, folder_name=folder).all()
    notebooks = db.query(SavedNotebook.notebook_id).filter_by(user_id=user_id, folder=folder).filter(
        SavedNotebook.deleted_at == None).all()
    return tuple(sorted(r[0] for r in documents)), tuple(sorted(r[0] for r in notebooks))
