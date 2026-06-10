"""ActiveContextStore — "what's happening right now in this course".

Per-(student, course) namespace. A handful of small rows (usually <10),
each tagged with a fragment_type. When a fragment is updated, the old
row is marked superseded so:
  - Pedro always sees the current state via .current()
  - The teacher view can replay historical focus shifts via .history()

Fragment types:
  - current_focus       what topic the student is actively working on
  - current_goal        broader goal (e.g. "OpenMP midterm in 2 weeks")
  - last_unresolved     question/exercise that wasn't fully resolved
  - last_session        summary of the most recent session
  - open_question       student-raised question awaiting a follow-up
  - recent_topic        topics touched in the last few sessions
"""

from __future__ import annotations

from typing import Optional

from ...stores._semantic_base import SemanticStoreBase
from ...stores.base import MemoryItem, new_item_id


ContextFragmentType = str  # see module docstring for the canonical set


CANONICAL_FRAGMENT_TYPES = (
    "current_focus",
    "current_goal",
    "last_unresolved",
    "last_session",
    "open_question",
    "recent_topic",
)

# Fragment types where a new write SUPERSEDES the previous active row.
# Others (open_question, recent_topic) are append-only and can have
# multiple active rows at once.
SINGLETON_TYPES = {"current_focus", "current_goal", "last_unresolved", "last_session"}


class ActiveContextStore(SemanticStoreBase):
    STORE_NAME = "active_context"

    # ── Write ─────────────────────────────────────────────────────

    def set_fragment(
        self,
        namespace: str,
        fragment_type: ContextFragmentType,
        content: str,
        concept_ids: Optional[list[str]] = None,
        lesson_id: Optional[str] = None,
        meta: Optional[dict] = None,
    ) -> MemoryItem:
        """Write a context fragment. For singleton types, the previous
        active row of the same type is automatically superseded so
        .current() always returns at most one row per singleton type."""
        meta = dict(meta or {})
        meta.setdefault("fragment_type", fragment_type)
        if lesson_id:
            meta["lesson_id"] = lesson_id

        if fragment_type in SINGLETON_TYPES:
            for existing in self.current(namespace, fragment_type=fragment_type):
                self.supersede(existing.id, "(replaced)")

        item = MemoryItem(
            id=new_item_id("ctx"),
            namespace=namespace,
            store=self.STORE_NAME,
            content=content,
            tags=[fragment_type],
            entities=list(concept_ids or []),
            importance=0.8,
            store_specific=meta,
        )
        self._insert(item)
        return item

    def add_open_question(
        self,
        namespace: str,
        question: str,
        concept_ids: Optional[list[str]] = None,
    ) -> MemoryItem:
        return self.set_fragment(
            namespace,
            "open_question",
            question,
            concept_ids=concept_ids,
        )

    def add_recent_topic(
        self,
        namespace: str,
        topic: str,
        concept_ids: Optional[list[str]] = None,
    ) -> MemoryItem:
        # Cap to ~10 recent topics — older ones get superseded as a sliding window.
        existing = self.current(namespace, fragment_type="recent_topic")
        # Sort newest first by created_at; keep top 9, supersede the rest.
        existing.sort(key=lambda it: it.created_at, reverse=True)
        for old in existing[9:]:
            self.supersede(old.id, "(window-evicted)")
        return self.set_fragment(namespace, "recent_topic", topic, concept_ids=concept_ids)

    def resolve_question(self, item_id: str) -> None:
        """Mark an open_question as resolved (supersede)."""
        self.supersede(item_id, "(resolved)")

    # ── Read ──────────────────────────────────────────────────────

    def current(
        self,
        namespace: str,
        fragment_type: Optional[ContextFragmentType] = None,
    ) -> list[MemoryItem]:
        """All non-superseded fragments, optionally filtered by type."""
        items = self.all(namespace, include_superseded=False)
        if fragment_type:
            items = [it for it in items if fragment_type in it.tags]
        items.sort(key=lambda it: it.created_at, reverse=True)
        return items

    def snapshot(self, namespace: str) -> dict:
        """Single dict summarizing the full current active context.
        Suitable for direct inspection or as raw input to a prompt builder."""
        out: dict = {
            "current_focus": None,
            "current_goal": None,
            "last_unresolved": None,
            "last_session": None,
            "open_questions": [],
            "recent_topics": [],
        }
        for it in self.current(namespace):
            tag = (it.tags or [None])[0]
            ss = it.store_specific or {}
            if tag in ("current_focus", "current_goal", "last_unresolved", "last_session"):
                out[tag] = {
                    "id": it.id,
                    "text": it.content,
                    "concept_ids": it.entities,
                    "lesson_id": ss.get("lesson_id"),
                    "as_of": it.created_at,
                }
            elif tag == "open_question":
                out["open_questions"].append({
                    "id": it.id,
                    "text": it.content,
                    "concept_ids": it.entities,
                    "as_of": it.created_at,
                })
            elif tag == "recent_topic":
                out["recent_topics"].append({
                    "id": it.id,
                    "text": it.content,
                    "concept_ids": it.entities,
                    "as_of": it.created_at,
                })
        return out
