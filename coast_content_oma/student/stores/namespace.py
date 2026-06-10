"""Namespace helpers for student stores.

Per-course stores use: u{user_id}__student__{folder_slug}
Cross-course identity uses: u{user_id}__identity

Keeping these as plain functions so the shape is grep-able everywhere.
"""

from __future__ import annotations

import re


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def course_namespace(user_id: int | str, folder_name: str) -> str:
    """Per-(student, course) namespace for ActiveContext, ConceptMastery,
    Episode and Pattern stores."""
    return f"u{user_id}__student__{_slug(folder_name)}"


def identity_namespace(user_id: int | str) -> str:
    """Cross-course namespace for AcademicIdentity store."""
    return f"u{user_id}__identity"


def parse_course_namespace(ns: str) -> tuple[str | None, str | None]:
    """Reverse of course_namespace — returns (user_id, folder_slug) or
    (None, None) if not a course namespace."""
    m = re.match(r"^u([^_]+)__student__(.+)$", ns)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def list_course_namespaces(db_path, user_id: int | str) -> list[str]:
    """Return every per-course Student OMA namespace that has data for
    this user (episodes and/or mastery rows)."""
    import sqlite3
    from pathlib import Path

    from ...stores.db import connect_db

    prefix = f"u{user_id}__student__"
    found: set[str] = set()
    db = Path(db_path)
    if not db.exists():
        return []
    with connect_db(db) as conn:
        for table in ("episode_items", "concept_mastery_items"):
            try:
                for (ns,) in conn.execute(
                    f"SELECT DISTINCT namespace FROM {table} WHERE namespace LIKE ?",
                    (prefix + "%",),
                ):
                    found.add(ns)
            except sqlite3.OperationalError:
                pass
    return sorted(found)
