"""Namespaced semantic-style store.

Single SQLite file shared across all namespaces (user_id, folder).
Hybrid retrieval: OpenAI embeddings (cosine via numpy) + FTS5 (BM25),
merged with reciprocal rank fusion.

This is the base for ConceptStore / ContentStore / ImageStore. Each
content store is a thin wrapper that adds store-specific helpers and
fixes the store name.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import struct
from pathlib import Path
from typing import Optional

from .base import MemoryItem, new_item_id, now_iso
from .db import connect_db

logger = logging.getLogger(__name__)

EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = 1536


class SemanticStoreBase:
    """Per-store table inside a shared SQLite database.

    Subclasses set `STORE_NAME` (e.g. "concept", "content", "image") which
    becomes the table prefix. The FTS shadow table mirrors the same.
    """

    STORE_NAME: str = "items"

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._init_db()

    @property
    def table(self) -> str:
        return f"{self.STORE_NAME}_items"

    @property
    def fts_table(self) -> str:
        return f"{self.STORE_NAME}_items_fts"

    def _init_db(self) -> None:
        with connect_db(self.db_path) as conn:
            conn.executescript(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    importance REAL DEFAULT 0.5,
                    source_doc_id TEXT,
                    entities TEXT,
                    tags TEXT,
                    superseded_by TEXT,
                    store_specific TEXT,
                    embedding BLOB
                );
                CREATE INDEX IF NOT EXISTS idx_{self.table}_ns ON {self.table}(namespace);
                CREATE INDEX IF NOT EXISTS idx_{self.table}_created ON {self.table}(created_at);
                CREATE INDEX IF NOT EXISTS idx_{self.table}_importance ON {self.table}(importance);
                CREATE VIRTUAL TABLE IF NOT EXISTS {self.fts_table}
                    USING fts5(id UNINDEXED, namespace UNINDEXED, content, entities, tags);
            """)

    # ── Write ─────────────────────────────────────────────────────

    def write(
        self,
        namespace: str,
        content: str,
        entities: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        importance: float = 0.5,
        source_doc_id: Optional[str] = None,
        store_specific: Optional[dict] = None,
    ) -> Optional[MemoryItem]:
        content = (content or "").strip()
        if not content:
            return None
        item = MemoryItem(
            id=new_item_id(self.STORE_NAME),
            namespace=namespace,
            store=self.STORE_NAME,
            content=content,
            entities=list(entities or []),
            tags=list(tags or []),
            importance=importance,
            source_doc_id=source_doc_id,
            store_specific=dict(store_specific or {}),
        )
        self._insert(item)
        return item

    def write_item(self, item: MemoryItem) -> None:
        if item.store != self.STORE_NAME:
            item.store = self.STORE_NAME
        self._insert(item)

    def write_items_bulk(self, items: list[MemoryItem], batch_size: int = 200) -> None:
        """Bulk insert with batched embedding. Much cheaper than per-item."""
        if not items:
            return
        id_to_emb: dict[str, list[float]] = {}
        if os.environ.get("OPENAI_API_KEY"):
            try:
                if self._client is None:
                    from openai import OpenAI
                    self._client = OpenAI()
                for start in range(0, len(items), batch_size):
                    chunk = items[start : start + batch_size]
                    texts = [it.content[:8000] for it in chunk]
                    resp = self._client.embeddings.create(model=EMBED_MODEL, input=texts)
                    for it, data in zip(chunk, resp.data):
                        id_to_emb[it.id] = list(data.embedding)
            except Exception as e:
                logger.warning(f"bulk embedding failed ({e}); inserting without embeddings.")
                id_to_emb = {}

        with connect_db(self.db_path) as conn:
            for it in items:
                if it.store != self.STORE_NAME:
                    it.store = self.STORE_NAME
                emb = id_to_emb.get(it.id)
                conn.execute(
                    f"""INSERT OR REPLACE INTO {self.table}
                        (id, namespace, content, created_at, last_accessed, access_count,
                         importance, source_doc_id, entities, tags,
                         superseded_by, store_specific, embedding)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        it.id, it.namespace, it.content, it.created_at, it.last_accessed,
                        it.access_count, it.importance, it.source_doc_id,
                        json.dumps(it.entities), json.dumps(it.tags),
                        it.superseded_by, json.dumps(it.store_specific),
                        _pack(emb) if emb else None,
                    ),
                )
                conn.execute(
                    f"INSERT OR REPLACE INTO {self.fts_table} (id, namespace, content, entities, tags) VALUES (?,?,?,?,?)",
                    (it.id, it.namespace, it.content, " ".join(it.entities), " ".join(it.tags)),
                )

    def _insert(self, item: MemoryItem) -> None:
        emb = self._embed(item.content)
        with connect_db(self.db_path) as conn:
            conn.execute(
                f"""INSERT OR REPLACE INTO {self.table}
                    (id, namespace, content, created_at, last_accessed, access_count,
                     importance, source_doc_id, entities, tags,
                     superseded_by, store_specific, embedding)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    item.id, item.namespace, item.content, item.created_at, item.last_accessed,
                    item.access_count, item.importance, item.source_doc_id,
                    json.dumps(item.entities), json.dumps(item.tags),
                    item.superseded_by, json.dumps(item.store_specific),
                    _pack(emb) if emb else None,
                ),
            )
            conn.execute(
                f"INSERT OR REPLACE INTO {self.fts_table} (id, namespace, content, entities, tags) VALUES (?,?,?,?,?)",
                (item.id, item.namespace, item.content, " ".join(item.entities), " ".join(item.tags)),
            )

    def supersede(self, old_id: str, new_id: str) -> None:
        with connect_db(self.db_path) as conn:
            conn.execute(f"UPDATE {self.table} SET superseded_by = ? WHERE id = ?", (new_id, old_id))

    def update_store_specific(self, item_id: str, patch: dict) -> None:
        """Merge patch into store_specific JSON for an existing item."""
        with connect_db(self.db_path) as conn:
            row = conn.execute(
                f"SELECT store_specific FROM {self.table} WHERE id = ?",
                (item_id,),
            ).fetchone()
            if not row:
                return
            current = json.loads(row[0]) if row[0] else {}
            current.update(patch)
            conn.execute(
                f"UPDATE {self.table} SET store_specific = ? WHERE id = ?",
                (json.dumps(current), item_id),
            )

    # ── Read ──────────────────────────────────────────────────────

    def get(self, item_id: str) -> Optional[MemoryItem]:
        with connect_db(self.db_path) as conn:
            row = conn.execute(f"SELECT * FROM {self.table} WHERE id = ?", (item_id,)).fetchone()
        return _row_to_item(row, self.STORE_NAME) if row else None

    def get_many(self, item_ids: list[str]) -> list[MemoryItem]:
        if not item_ids:
            return []
        placeholders = ",".join("?" * len(item_ids))
        with connect_db(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT * FROM {self.table} WHERE id IN ({placeholders})",
                item_ids,
            ).fetchall()
        return [_row_to_item(r, self.STORE_NAME) for r in rows if r]

    def all(self, namespace: str, include_superseded: bool = False) -> list[MemoryItem]:
        sql = f"SELECT * FROM {self.table} WHERE namespace = ?"
        if not include_superseded:
            sql += " AND superseded_by IS NULL"
        sql += " ORDER BY created_at"
        with connect_db(self.db_path) as conn:
            rows = conn.execute(sql, (namespace,)).fetchall()
        return [_row_to_item(r, self.STORE_NAME) for r in rows if r]

    def count(self, namespace: str) -> int:
        with connect_db(self.db_path) as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM {self.table} WHERE namespace = ? AND superseded_by IS NULL",
                (namespace,),
            ).fetchone()
        return row[0] if row else 0

    def search(
        self,
        namespace: str,
        query: str,
        max_results: int = 8,
        tag_filter: Optional[list[str]] = None,
        entity_filter: Optional[list[str]] = None,
        include_superseded: bool = False,
    ) -> list[MemoryItem]:
        """Hybrid search: vector + FTS, merged via reciprocal rank fusion.

        Filters tag_filter (any) and entity_filter (any) post-fetch so we
        keep the FTS/vector pipeline simple.
        """
        query = (query or "").strip()
        if not query:
            return []

        semantic_hits: list[tuple[str, float]] = []
        query_emb = self._embed(query)
        if query_emb:
            semantic_hits = self._vector_search(namespace, query_emb, top_k=max_results * 3)
        keyword_hits = self._keyword_search(namespace, query, top_k=max_results * 3)

        scores: dict[str, float] = {}
        for rank, (fid, _) in enumerate(semantic_hits):
            scores[fid] = scores.get(fid, 0.0) + 1.0 / (rank + 1)
        for rank, fid in enumerate(keyword_hits):
            scores[fid] = scores.get(fid, 0.0) + 0.5 / (rank + 1)

        if not scores:
            return []

        ids = list(scores.keys())
        placeholders = ",".join("?" * len(ids))
        sql = f"SELECT * FROM {self.table} WHERE id IN ({placeholders}) AND namespace = ?"
        if not include_superseded:
            sql += " AND superseded_by IS NULL"
        with connect_db(self.db_path) as conn:
            rows = conn.execute(sql, [*ids, namespace]).fetchall()

        items = [_row_to_item(r, self.STORE_NAME) for r in rows if r]
        if tag_filter:
            tag_set = {t.lower() for t in tag_filter}
            items = [it for it in items if any(t.lower() in tag_set for t in it.tags)]
        if entity_filter:
            ent_set = {e.lower() for e in entity_filter}
            items = [it for it in items if any(e.lower() in ent_set for e in it.entities)]

        items.sort(key=lambda it: scores.get(it.id, 0), reverse=True)
        if items:
            self._touch([it.id for it in items[:max_results]])
        return items[:max_results]

    def find_by_tag(
        self,
        namespace: str,
        tag: str,
        max_results: int = 50,
    ) -> list[MemoryItem]:
        """Exact tag match (e.g. content_type='example')."""
        items = self.all(namespace)
        tag_l = tag.lower()
        out = [it for it in items if any(t.lower() == tag_l for t in it.tags)]
        return out[:max_results]

    def find_by_entity(
        self,
        namespace: str,
        entity: str,
        max_results: int = 50,
    ) -> list[MemoryItem]:
        """Exact entity match (entities here are concept_ids or canonical concept names)."""
        items = self.all(namespace)
        ent_l = entity.lower()
        out = [it for it in items if any(e.lower() == ent_l for e in it.entities)]
        return out[:max_results]

    def _keyword_search(self, namespace: str, query: str, top_k: int) -> list[str]:
        stop = {
            "the", "a", "an", "is", "was", "were", "of", "to", "in", "on", "at",
            "and", "or", "it", "with", "for", "by", "as", "be", "do", "does",
            "did", "have", "has", "had", "i", "you", "we", "they", "what", "who",
            "where", "when", "why", "how", "that", "this",
        }
        tokens = [t.strip(".,?!:;'\"").lower() for t in query.split()]
        terms = [t for t in tokens if t and t not in stop and len(t) > 1]
        if not terms:
            terms = [t for t in tokens if t and len(t) > 1]
        if not terms:
            return []
        fts_query = " OR ".join(f'"{t}"' for t in terms)
        with connect_db(self.db_path) as conn:
            try:
                rows = conn.execute(
                    f"SELECT id FROM {self.fts_table} WHERE {self.fts_table} MATCH ? AND namespace = ? ORDER BY rank LIMIT ?",
                    (fts_query, namespace, top_k),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [r[0] for r in rows]

    def _vector_search(self, namespace: str, query_emb: list[float], top_k: int) -> list[tuple[str, float]]:
        with connect_db(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT id, embedding FROM {self.table} WHERE embedding IS NOT NULL AND namespace = ? AND superseded_by IS NULL",
                (namespace,),
            ).fetchall()
        if not rows:
            return []
        try:
            import numpy as np
        except ImportError:
            scored = []
            for fid, blob in rows:
                emb = _unpack(blob)
                if not emb:
                    continue
                scored.append((fid, _cosine(query_emb, emb)))
            scored.sort(key=lambda kv: kv[1], reverse=True)
            return scored[:top_k]

        ids = [fid for fid, _ in rows]
        D = len(query_emb)
        mat = np.empty((len(rows), D), dtype=np.float32)
        for i, (_fid, blob) in enumerate(rows):
            if blob and len(blob) // 4 == D:
                mat[i] = np.frombuffer(blob, dtype=np.float32)
            else:
                mat[i] = 0.0
        q = np.asarray(query_emb, dtype=np.float32)
        mat_norms = np.linalg.norm(mat, axis=1) + 1e-12
        q_norm = np.linalg.norm(q) + 1e-12
        scores = (mat @ q) / (mat_norms * q_norm)
        k = min(top_k, len(scores))
        if k <= 0:
            return []
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(ids[i], float(scores[i])) for i in idx]

    def _touch(self, ids: list[str]) -> None:
        if not ids:
            return
        ts = now_iso()
        with connect_db(self.db_path) as conn:
            placeholders = ",".join("?" * len(ids))
            conn.execute(
                f"UPDATE {self.table} SET last_accessed = ?, access_count = access_count + 1 WHERE id IN ({placeholders})",
                (ts, *ids),
            )

    def stats(self, namespace: Optional[str] = None) -> dict:
        with connect_db(self.db_path) as conn:
            if namespace:
                total = conn.execute(f"SELECT COUNT(*) FROM {self.table} WHERE namespace = ?", (namespace,)).fetchone()[0]
                active = conn.execute(
                    f"SELECT COUNT(*) FROM {self.table} WHERE namespace = ? AND superseded_by IS NULL",
                    (namespace,),
                ).fetchone()[0]
                with_emb = conn.execute(
                    f"SELECT COUNT(*) FROM {self.table} WHERE namespace = ? AND embedding IS NOT NULL",
                    (namespace,),
                ).fetchone()[0]
            else:
                total = conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]
                active = conn.execute(f"SELECT COUNT(*) FROM {self.table} WHERE superseded_by IS NULL").fetchone()[0]
                with_emb = conn.execute(f"SELECT COUNT(*) FROM {self.table} WHERE embedding IS NOT NULL").fetchone()[0]
        return {
            "store": self.STORE_NAME,
            "namespace": namespace,
            "total": total,
            "active": active,
            "superseded": total - active,
            "with_embeddings": with_emb,
        }

    def delete(self, item_id: str) -> None:
        with connect_db(self.db_path) as conn:
            conn.execute(f"DELETE FROM {self.table} WHERE id = ?", (item_id,))
            conn.execute(f"DELETE FROM {self.fts_table} WHERE id = ?", (item_id,))

    def delete_namespace(self, namespace: str) -> int:
        """Wipe all items in a namespace. Used when a folder is deleted or re-ingested."""
        with connect_db(self.db_path) as conn:
            n = conn.execute(f"DELETE FROM {self.table} WHERE namespace = ?", (namespace,)).rowcount
            conn.execute(f"DELETE FROM {self.fts_table} WHERE namespace = ?", (namespace,))
        return n

    def _embed(self, text: str) -> Optional[list[float]]:
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        try:
            if self._client is None:
                from openai import OpenAI
                self._client = OpenAI()
            resp = self._client.embeddings.create(model=EMBED_MODEL, input=text[:8000])
            return resp.data[0].embedding
        except Exception as e:
            logger.debug(f"embedding failed: {e}")
            return None


# ── Helpers ─────────────────────────────────────────────────────────

def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack(blob: bytes) -> list[float]:
    if not blob:
        return []
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def _row_to_item(row: tuple, store_name: str) -> MemoryItem:
    (
        _id, namespace, content, created_at, last_accessed, access_count,
        importance, source_doc_id, entities_json, tags_json,
        superseded_by, store_specific_json, _embedding,
    ) = row
    return MemoryItem(
        id=_id,
        namespace=namespace,
        store=store_name,
        content=content,
        created_at=created_at,
        last_accessed=last_accessed,
        access_count=access_count or 0,
        importance=importance or 0.5,
        source_doc_id=source_doc_id,
        entities=json.loads(entities_json) if entities_json else [],
        tags=json.loads(tags_json) if tags_json else [],
        superseded_by=superseded_by,
        store_specific=json.loads(store_specific_json) if store_specific_json else {},
    )
