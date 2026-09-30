"""Per-source OMA ingest status on folder_sources (coast.db).

Prevents duplicate concurrent ingests and gives ensure_oma_ready_for_outline
a authoritative readiness signal beyond page-count heuristics.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

STATUS_PENDING = "PENDING"
STATUS_INGESTING = "INGESTING"
STATUS_CONTENT_INDEXED = "CONTENT_INDEXED"
STATUS_READY = "READY_FOR_ROADMAP"
STATUS_COMPLETE = "COMPLETE"
STATUS_FAILED = "FAILED"

TERMINAL_OK = frozenset({STATUS_READY, STATUS_COMPLETE})
CONTENT_DONE = frozenset({STATUS_CONTENT_INDEXED, *TERMINAL_OK})


def get_status(source_id: str) -> str:
    from database import FolderSource, SessionLocal

    db = SessionLocal()
    try:
        row = db.query(FolderSource).filter(FolderSource.source_id == source_id).first()
        if not row:
            return STATUS_PENDING
        return (row.oma_ingest_status or STATUS_PENDING).strip().upper()
    finally:
        db.close()


def set_status(source_id: str, status: str, *, error: Optional[str] = None) -> None:
    from database import FolderSource, SessionLocal

    db = SessionLocal()
    try:
        row = db.query(FolderSource).filter(FolderSource.source_id == source_id).first()
        if not row:
            logger.warning("set_status: unknown source_id %s", source_id)
            return
        row.oma_ingest_status = status
        row.oma_ingest_error = (error or "")[:2000] if error else None
        db.commit()
    finally:
        db.close()


def try_claim(source_id: str) -> bool:
    """Atomically claim ingest for source_id. Returns False if already ingesting/ready."""
    from sqlalchemy import text
    from database import SessionLocal

    db = SessionLocal()
    try:
        result = db.execute(
            text(
                "UPDATE folder_sources SET oma_ingest_status = :ingesting, oma_ingest_error = NULL "
                "WHERE source_id = :sid AND ("
                "oma_ingest_status IN (:pending, :failed) OR oma_ingest_status IS NULL"
                ")"
            ),
            {
                "ingesting": STATUS_INGESTING,
                "sid": source_id,
                "pending": STATUS_PENDING,
                "failed": STATUS_FAILED,
            },
        )
        db.commit()
        claimed = result.rowcount > 0
        if claimed:
            logger.info("ingest claimed source_id=%s", source_id)
        return claimed
    finally:
        db.close()


def should_queue(source_id: str) -> bool:
    """True when this source still needs an OMA ingest run."""
    st = get_status(source_id)
    return st in (STATUS_PENDING, STATUS_FAILED)


def folder_sources_ready(pdf_sources: list[dict]) -> bool:
    """All listed sources reached READY_FOR_ROADMAP (or COMPLETE)."""
    if not pdf_sources:
        return False
    for src in pdf_sources:
        st = (src.get("oma_ingest_status") or STATUS_PENDING).strip().upper()
        if st not in TERMINAL_OK:
            return False
    return True
