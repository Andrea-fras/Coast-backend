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
        """Exact (case-insensitive) match on canonical name or any alias."""
        n = name.lower().strip()
        if not n:
            return None
        for it in self.all(namespace):
            ss = it.store_specific or {}
            if (ss.get("name") or "").lower() == n:
                return it
            aliases = [a.lower() for a in (ss.get("aliases") or [])]
            if n in aliases:
                return it
            if n in (e.lower() for e in it.entities):
                return it
        return None

    def find_candidates(self, namespace: str, name: str, max_results: int = 5) -> list[MemoryItem]:
        """Fuzzy lookup via hybrid search (handles unknown phrasings)."""
        return self.search(namespace, name, max_results=max_results)

    def prerequisites_of(self, namespace: str, concept_id: str) -> list[MemoryItem]:
        it = self.get(concept_id)
        if not it:
            return []
        pre_ids = (it.store_specific or {}).get("prerequisite_concept_ids") or []
        return [c for c in self.get_many(pre_ids) if c and c.namespace == namespace]

    def traverse_prereq_chain(
        self,
        namespace: str,
        concept_id: str,
        max_depth: int = 4,
    ) -> list[MemoryItem]:
        """Walk the prerequisite chain breadth-first; returns ordered prerequisites
        (closest first). Caps depth to avoid cycles in a malformed graph."""
        out: list[MemoryItem] = []
        seen: set[str] = {concept_id}
        frontier = [concept_id]
        for _ in range(max_depth):
            next_frontier: list[str] = []
            for cid in frontier:
                for p in self.prerequisites_of(namespace, cid):
                    if p.id in seen:
                        continue
                    seen.add(p.id)
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

    def merge_into(self, namespace: str, src_id: str, dst_id: str) -> bool:
        """Merge concept src into dst: union aliases / edges / sources,
        re-point every other concept's edges from src to dst, then mark
        src superseded. Reads of the namespace skip superseded rows."""
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

        for key in ("prerequisite_concept_ids", "related_concept_ids", "lecture_sources"):
            merged = list(dict.fromkeys((dss.get(key) or []) + (sss.get(key) or [])))
            dss[key] = [v for v in merged if v not in (src_id, dst_id)]

        if not dss.get("definition") and sss.get("definition"):
            dss["definition"] = sss["definition"]
            dst.content = sss["definition"]

        dst.store_specific = dss
        dst.entities = [dss.get("name") or dst_id] + dss["aliases"]
        self._insert(dst)

        # Re-point edges in every other concept of the namespace.
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
