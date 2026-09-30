"""ConceptAliasStore — append-only ledger for concept identity merges.

Maps alias_id (superseded concept) → canonical_id. Reversible via reversed_at.
Never delete rows; resolve at read time.
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from .db import connect_db
from .base import now_iso


# Alias maps are read on every profile build but change only on concept merges.
# Process-wide cache, cleared by any write here; the short TTL covers writes made
# by another process (maintenance scripts).
_MAP_TTL_SEC = 5.0
_map_cache: dict[tuple[str, str], tuple[float, dict[str, str]]] = {}
_map_lock = threading.Lock()


def _invalidate(db_path: Path, namespace: str) -> None:
    with _map_lock:
        _map_cache.pop((str(db_path), namespace), None)


class ConceptAliasStore:
    TABLE = "concept_alias_items"

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with connect_db(self.db_path) as conn:
            statements = f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE} (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    alias_id TEXT NOT NULL,
                    canonical_id TEXT NOT NULL,
                    merged_at TEXT NOT NULL,
                    confidence REAL,
                    reason TEXT,
                    reversed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_{self.TABLE}_ns_alias
                    ON {self.TABLE}(namespace, alias_id);
                CREATE INDEX IF NOT EXISTS idx_{self.TABLE}_ns_canonical
                    ON {self.TABLE}(namespace, canonical_id);
            """
            for statement in statements.split(";"):
                if statement.strip():
                    conn.execute(statement)

    def active_map(self, namespace: str) -> dict[str, str]:
        """alias_id → canonical_id for active (non-reversed) entries."""
        key = (str(self.db_path), namespace)
        with _map_lock:
            hit = _map_cache.get(key)
        if hit and hit[0] > time.monotonic():
            return hit[1]
        with connect_db(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT alias_id, canonical_id FROM {self.TABLE} "
                f"WHERE namespace = ? AND reversed_at IS NULL",
                (namespace,),
            ).fetchall()
        amap = {a: c for a, c in rows}
        with _map_lock:
            _map_cache[key] = (time.monotonic() + _MAP_TTL_SEC, amap)
        return amap

    def resolve(self, namespace: str, concept_id: str, *, max_hops: int = 16) -> str:
        """Follow alias chain to terminal canonical ID. Cycle-safe."""
        amap = self.active_map(namespace)
        cur = concept_id
        seen: set[str] = {cur}
        for _ in range(max_hops):
            nxt = amap.get(cur)
            if not nxt or nxt == cur:
                return cur
            if nxt in seen:
                return nxt
            seen.add(nxt)
            cur = nxt
        return cur

    def resolve_many(self, namespace: str, concept_ids: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for cid in concept_ids:
            canon = self.resolve(namespace, cid)
            if canon not in seen:
                seen.add(canon)
                out.append(canon)
        return out

    def ids_for_canonical(self, namespace: str, canonical_id: str) -> list[str]:
        """All IDs that resolve to canonical_id (including canonical itself)."""
        terminal = self.resolve(namespace, canonical_id)
        ids = {terminal}
        amap = self.active_map(namespace)
        for alias, canon in amap.items():
            if self.resolve(namespace, alias) == terminal:
                ids.add(alias)
        return sorted(ids)

    def append_merge(
        self,
        namespace: str,
        alias_id: str,
        canonical_id: str,
        *,
        confidence: float = 1.0,
        reason: str = "dedup",
    ) -> bool:
        """Record alias_id → canonical_id. Returns False if cycle or duplicate."""
        try:
            return self._append_merge(namespace, alias_id, canonical_id, confidence=confidence, reason=reason)
        finally:
            _invalidate(self.db_path, namespace)

    def _append_merge(self, namespace, alias_id, canonical_id, *, confidence, reason) -> bool:
        if alias_id == canonical_id:
            return False
        terminal = self.resolve(namespace, canonical_id)
        if terminal == alias_id:
            return False
        if self.resolve(namespace, alias_id) == terminal:
            return True

        with connect_db(self.db_path) as conn:
            existing = conn.execute(
                f"SELECT id FROM {self.TABLE} WHERE namespace = ? AND alias_id = ? "
                f"AND reversed_at IS NULL",
                (namespace, alias_id),
            ).fetchone()
            if existing:
                conn.execute(
                    f"UPDATE {self.TABLE} SET canonical_id = ?, merged_at = ?, "
                    f"confidence = ?, reason = ? WHERE id = ?",
                    (terminal, now_iso(), confidence, reason, existing[0]),
                )
                return True
            conn.execute(
                f"INSERT INTO {self.TABLE} "
                f"(id, namespace, alias_id, canonical_id, merged_at, confidence, reason) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    f"alias_{uuid.uuid4().hex[:12]}",
                    namespace,
                    alias_id,
                    terminal,
                    now_iso(),
                    confidence,
                    reason,
                ),
            )
        return True

    def reverse(self, namespace: str, alias_id: str) -> bool:
        """Undo a merge — alias_id becomes its own identity again."""
        try:
            return self._reverse(namespace, alias_id)
        finally:
            _invalidate(self.db_path, namespace)

    def _reverse(self, namespace: str, alias_id: str) -> bool:
        with connect_db(self.db_path) as conn:
            row = conn.execute(
                f"SELECT id FROM {self.TABLE} WHERE namespace = ? AND alias_id = ? "
                f"AND reversed_at IS NULL",
                (namespace, alias_id),
            ).fetchone()
            if not row:
                return False
            conn.execute(
                f"UPDATE {self.TABLE} SET reversed_at = ? WHERE id = ?",
                (now_iso(), row[0]),
            )
        return True
