"""ConceptStore — canonical concepts extracted from course material.

Each item is a distinct concept (e.g. "eigenvalues") with:
  - content: the concept's canonical definition (short, one paragraph)
  - entities: aliases / surface forms found in lectures
  - store_specific:
      name: canonical name (= entities[0])
      aliases: alternative names found in lectures
      prerequisite_concept_ids: list of concept_ids this depends on
      related_concept_ids: list of concept_ids that co-occur but aren't strict prereqs
      lecture_sources: list of source_doc_ids where this concept appears
      first_appearance: source_doc_id of the lecture where this concept is introduced
"""

from __future__ import annotations

from typing import Optional

from ._semantic_base import SemanticStoreBase
from .base import MemoryItem


class ConceptStore(SemanticStoreBase):
    STORE_NAME = "concept"

    def find_by_name(self, namespace: str, name: str) -> Optional[MemoryItem]:
        """Exact (case-insensitive) match on canonical name, alias or entity — in SQL,
        so a lookup does not load the whole concept graph."""
        n = name.lower().strip()
        if not n:
            return None
        from .db import connect_db
        from ._semantic_base import _row_to_item
        with connect_db(self.db_path) as conn:
            row = conn.execute(
                f"""SELECT * FROM {self.table} WHERE namespace = ? AND superseded_by IS NULL AND (
                        lower(json_extract(store_specific, '$.name')) = ?
                        OR EXISTS (SELECT 1 FROM json_each(store_specific, '$.aliases') WHERE lower(value) = ?)
                        OR EXISTS (SELECT 1 FROM json_each(entities) WHERE lower(value) = ?))
                    ORDER BY lower(json_extract(store_specific, '$.name')) = ? DESC LIMIT 1""",
                (namespace, n, n, n, n),
            ).fetchone()
        return _row_to_item(row, self.STORE_NAME) if row else None

    def find_candidates(self, namespace: str, name: str, max_results: int = 5) -> list[MemoryItem]:
        """Fuzzy lookup via hybrid search (handles unknown phrasings)."""
        return self.search(namespace, name, max_results=max_results)

    def prerequisites_of(
        self,
        namespace: str,
        concept_id: str,
        resolver=None,
    ) -> list[MemoryItem]:
        rid = resolver.resolve(namespace, concept_id) if resolver else concept_id
        it = self.get(rid) or self.get(concept_id)
        if not it:
            return []
        pre_ids = (it.store_specific or {}).get("prerequisite_concept_ids") or []
        if resolver:
            pre_ids = resolver.resolve_edge_ids(namespace, pre_ids)
        return [c for c in self.get_many(pre_ids) if c and c.namespace == namespace]

    def traverse_prereq_chain(
        self,
        namespace: str,
        concept_id: str,
        max_depth: int = 4,
        resolver=None,
    ) -> list[MemoryItem]:
        """Walk the prerequisite chain breadth-first; returns ordered prerequisites
        (closest first). Caps depth to avoid cycles in a malformed graph."""
        root = resolver.resolve(namespace, concept_id) if resolver else concept_id
        out: list[MemoryItem] = []
        seen: set[str] = {root}
        frontier = [root]
        for _ in range(max_depth):
            next_frontier: list[str] = []
            for cid in frontier:
                for p in self.prerequisites_of(namespace, cid, resolver=resolver):
                    pid = resolver.resolve(namespace, p.id) if resolver else p.id
                    if pid in seen:
                        continue
                    seen.add(pid)
                    out.append(p)
                    next_frontier.append(p.id)
            if not next_frontier:
                break
            frontier = next_frontier
        return out

    def related_to(self, namespace: str, concept_id: str) -> list[MemoryItem]:
        it = self.get(concept_id)
        if not it:
            return []
        rel_ids = (it.store_specific or {}).get("related_concept_ids") or []
        return [c for c in self.get_many(rel_ids) if c and c.namespace == namespace]

    def all_concepts(self, namespace: str) -> list[MemoryItem]:
        return self.all(namespace)

    # ── Dedup support ─────────────────────────────────────────────

    def find_similar(
        self,
        namespace: str,
        text: str,
        threshold: float = 0.90,
        exclude_ids: Optional[set[str]] = None,
    ) -> list[tuple[MemoryItem, float]]:
        """Embedding-similarity lookup for near-duplicate concepts.

        Returns [(item, cosine)] above threshold, best first. Empty when
        embeddings are unavailable (no API key) — callers must treat that
        as 'no match', not an error."""
        emb = self._embed(text)
        if not emb:
            return []
        hits = self._vector_search(namespace, emb, top_k=8)
        out: list[tuple[MemoryItem, float]] = []
        for item_id, score in hits:
            if score < threshold:
                continue
            if exclude_ids and item_id in exclude_ids:
                continue
            it = self.get(item_id)
            if it is not None:
                out.append((it, score))
        return out

    def embeddings_for_namespace(self, namespace: str) -> list[tuple[str, list[float]]]:
        """(concept_id, embedding) for every active concept that has one."""
        import sqlite3
        from ..db import connect_db
        from ._semantic_base import _unpack
        with connect_db(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT id, embedding FROM {self.table} "
                "WHERE namespace = ? AND superseded_by IS NULL AND embedding IS NOT NULL",
                (namespace,),
            ).fetchall()
        return [(rid, _unpack(blob)) for rid, blob in rows if blob]

    def merge_into(
        self,
        namespace: str,
        src_id: str,
        dst_id: str,
        *,
        alias_store=None,
        use_ledger: bool = False,
        merge_confidence: float = 1.0,
        merge_reason: str = "dedup",
    ) -> bool:
        """Merge concept src into dst.

        When use_ledger=True, append an alias ledger entry and skip
        destructive edge rewrites — reads resolve through the ledger.
        """
        if src_id == dst_id:
            return False
        src = self.get(src_id)
        dst = self.get(dst_id)
        if not src or not dst or src.namespace != namespace or dst.namespace != namespace:
            return False

        sss = src.store_specific or {}
        dss = dict(dst.store_specific or {})
        dst_name = (dss.get("name") or "").lower().strip()

        merged_aliases = list(dict.fromkeys(
            (dss.get("aliases") or [])
            + [sss.get("name") or ""]
            + (sss.get("aliases") or [])
        ))
        dss["aliases"] = [
            a for a in merged_aliases
            if a and a.lower().strip() != dst_name
        ]

        if use_ledger:
            for key in ("prerequisite_concept_ids", "related_concept_ids", "lecture_sources"):
                merged = list(dict.fromkeys((dss.get(key) or []) + (sss.get(key) or [])))
                dss[key] = [v for v in merged if v not in (src_id, dst_id)]
        else:
            for key in ("prerequisite_concept_ids", "related_concept_ids", "lecture_sources"):
                merged = list(dict.fromkeys((dss.get(key) or []) + (sss.get(key) or [])))
                dss[key] = [v for v in merged if v not in (src_id, dst_id)]

        if not dss.get("definition") and sss.get("definition"):
            dss["definition"] = sss["definition"]
            dst.content = sss["definition"]

        dst.store_specific = dss
        dst.entities = [dss.get("name") or dst_id] + dss["aliases"]
        self._insert(dst)

        if use_ledger and alias_store is not None:
            if not alias_store.append_merge(
                namespace, src_id, dst_id,
                confidence=merge_confidence, reason=merge_reason,
            ):
                return False
            self.supersede(src_id, dst_id)
            return True

        # Legacy destructive path — re-point edges in every other concept.
        for other in self.all(namespace):
            if other.id in (src_id, dst_id):
                continue
            oss = other.store_specific or {}
            changed = False
            for key in ("prerequisite_concept_ids", "related_concept_ids"):
                ids = oss.get(key) or []
                if src_id in ids:
                    new_ids = list(dict.fromkeys(
                        dst_id if v == src_id else v for v in ids
                    ))
                    new_ids = [v for v in new_ids if v != other.id]
                    oss[key] = new_ids
                    changed = True
            if changed:
                self.update_store_specific(other.id, oss)

        self.supersede(src_id, dst_id)
        return True
