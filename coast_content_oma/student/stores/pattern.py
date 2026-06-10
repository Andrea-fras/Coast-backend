"""PatternStore — derived insights about a student in one course.

Populated by the CourseConsolidator. Inspectable and queryable as a
structured record (not just embedding).

store_specific schema:
  - pattern_type        one of PatternType
  - confidence          0.0 — 1.0
  - evidence_count      how many episodes back this up
  - first_observed      ISO timestamp
  - last_confirmed      ISO timestamp
  - related_concept_ids list of concept ids this pattern is about (if any)
  - derivation          short text describing how this was inferred
"""

from __future__ import annotations

from typing import Optional

from ...stores._semantic_base import SemanticStoreBase
from ...stores.base import MemoryItem, new_item_id, now_iso


PatternType = str


CANONICAL_PATTERN_TYPES = (
    # Content/format preferences
    "prefers_examples",
    "prefers_definitions_first",
    "prefers_diagrams",
    "asks_followups",
    "prefers_short_answers",
    "prefers_step_by_step",
    # Struggle / misconception
    "struggle_cluster",       # group of concepts repeatedly missed together
    "misconception",          # specific identified false belief
    "frequent_giveup",        # gives up on hard problems
    # Pace / scheduling
    "drops_off_after_n_min",  # avg session length signal
    "studies_in_bursts",      # episodic vs steady
    "needs_review_after_n_days",  # forgetting curve insight
    # Conceptual strengths
    "strong_in_topic",        # area of strength
    "weak_in_topic",          # area of weakness
)


class PatternStore(SemanticStoreBase):
    STORE_NAME = "pattern"

    def upsert(
        self,
        namespace: str,
        pattern_type: PatternType,
        description: str,
        confidence: float,
        evidence_count: int,
        related_concept_ids: Optional[list[str]] = None,
        derivation: str = "",
        dedupe_key: Optional[str] = None,
    ) -> MemoryItem:
        """Insert a new pattern, or update an existing one identified by
        (pattern_type, dedupe_key) — typically dedupe_key is the concept
        the pattern is about, or a normalized version of the description."""
        existing = self._find_existing(namespace, pattern_type, dedupe_key)
        ts = now_iso()

        ss = {
            "pattern_type": pattern_type,
            "confidence": float(max(0.0, min(1.0, confidence))),
            "evidence_count": int(evidence_count),
            "first_observed": existing.store_specific["first_observed"] if existing else ts,
            "last_confirmed": ts,
            "related_concept_ids": list(related_concept_ids or []),
            "derivation": derivation,
            "dedupe_key": dedupe_key,
        }
        if existing:
            existing.store_specific = ss
            existing.content = description
            existing.tags = [pattern_type]
            existing.entities = list(related_concept_ids or [])
            existing.importance = ss["confidence"]
            self._insert(existing)
            return existing

        item = MemoryItem(
            id=new_item_id("pat"),
            namespace=namespace,
            store=self.STORE_NAME,
            content=description,
            tags=[pattern_type],
            entities=list(related_concept_ids or []),
            importance=ss["confidence"],
            store_specific=ss,
        )
        self._insert(item)
        return item

    def _find_existing(
        self,
        namespace: str,
        pattern_type: PatternType,
        dedupe_key: Optional[str],
    ) -> Optional[MemoryItem]:
        for it in self.all(namespace):
            ss = it.store_specific or {}
            if ss.get("pattern_type") != pattern_type:
                continue
            if (dedupe_key is not None) and ss.get("dedupe_key") == dedupe_key:
                return it
            if dedupe_key is None and ss.get("dedupe_key") is None:
                return it
        return None

    def by_type(self, namespace: str, pattern_type: PatternType) -> list[MemoryItem]:
        return [it for it in self.all(namespace) if pattern_type in it.tags]

    def top_confidence(self, namespace: str, k: int = 8) -> list[MemoryItem]:
        items = self.all(namespace)
        items.sort(key=lambda it: (it.store_specific or {}).get("confidence", 0.0), reverse=True)
        return items[:k]
