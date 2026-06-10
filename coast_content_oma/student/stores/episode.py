"""EpisodeStore — immutable chronological event log.

Every meaningful interaction becomes one episode. The store is
append-only — episodes are never edited or superseded, so it remains
the ground-truth log from which Patterns and ConceptMastery can be
re-derived.

store_specific schema:
  - episode_type        one of EpisodeType
  - outcome             one of EpisodeOutcome
  - user_message        truncated student text (if applicable)
  - assistant_response  truncated Pedro text (if applicable)
  - concept_ids         list of Content OMA concept ids involved
  - lesson_id           optional lesson reference
  - duration_sec        optional time spent on this interaction
  - signals             dict of detected behavioural signals
                        (asked_for_example, expressed_confusion,
                         asked_followup, gave_up, requested_easier, ...)
  - source              ui surface: "chat" | "exercise" | "lesson_player"
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from ...stores._semantic_base import SemanticStoreBase, _row_to_item
from ...stores.base import MemoryItem, new_item_id, now_iso
from ...stores.db import connect_db


EpisodeType = str  # see ALLOWED_TYPES
EpisodeOutcome = str  # success | struggle | neutral


ALLOWED_TYPES = (
    "qa",                 # student asked, Pedro answered
    "exercise_attempt",   # student tried an exercise
    "lesson_started",     # student opened a lesson
    "lesson_completed",   # student finished a lesson
    "section_completed",  # student finished one lesson section
    "lesson_dropoff",     # student left a lesson before finishing
    "concept_reviewed",   # student revisited a known concept
    "self_assessment",    # student rated own understanding
    "external_event",     # anything else worth logging
)


ALLOWED_OUTCOMES = ("success", "struggle", "neutral", "mistake")


class EpisodeStore(SemanticStoreBase):
    STORE_NAME = "episode"

    # ── Write ─────────────────────────────────────────────────────

    def record(
        self,
        namespace: str,
        episode_type: EpisodeType,
        summary: str,
        outcome: EpisodeOutcome = "neutral",
        concept_ids: Optional[list[str]] = None,
        lesson_id: Optional[str] = None,
        user_message: Optional[str] = None,
        assistant_response: Optional[str] = None,
        duration_sec: Optional[int] = None,
        signals: Optional[dict] = None,
        source: str = "chat",
        section_title: Optional[str] = None,
        section_index: Optional[int] = None,
    ) -> MemoryItem:
        if episode_type not in ALLOWED_TYPES:
            # We don't reject — we tag it for review.
            episode_type = "external_event"
        if outcome not in ALLOWED_OUTCOMES:
            outcome = "neutral"

        ss = {
            "episode_type": episode_type,
            "outcome": outcome,
            "user_message": (user_message or "")[:1500],
            "assistant_response": (assistant_response or "")[:1500],
            "concept_ids": list(concept_ids or []),
            "lesson_id": lesson_id,
            "duration_sec": duration_sec,
            "signals": dict(signals or {}),
            "source": source,
        }
        if section_title:
            ss["section_title"] = section_title
        if section_index is not None:
            ss["section_index"] = int(section_index)
        item = MemoryItem(
            id=new_item_id("ep"),
            namespace=namespace,
            store=self.STORE_NAME,
            content=summary,
            entities=list(concept_ids or []),
            tags=[episode_type, outcome, source],
            importance=0.5 if outcome == "neutral" else 0.7,
            store_specific=ss,
        )
        self._insert(item)
        return item

    # ── Read / query ──────────────────────────────────────────────

    def recent(self, namespace: str, limit: int = 20) -> list[MemoryItem]:
        with connect_db(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT * FROM {self.table} WHERE namespace = ? "
                f"AND superseded_by IS NULL ORDER BY created_at DESC LIMIT ?",
                (namespace, limit),
            ).fetchall()
        return [_row_to_item(r, self.STORE_NAME) for r in rows if r]

    def since(self, namespace: str, days: float) -> list[MemoryItem]:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        with connect_db(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT * FROM {self.table} WHERE namespace = ? "
                f"AND superseded_by IS NULL AND created_at >= ? ORDER BY created_at",
                (namespace, cutoff),
            ).fetchall()
        return [_row_to_item(r, self.STORE_NAME) for r in rows if r]

    def by_outcome(self, namespace: str, outcome: EpisodeOutcome, days: Optional[float] = None) -> list[MemoryItem]:
        pool = self.since(namespace, days) if days else self.all(namespace)
        return [it for it in pool if (it.store_specific or {}).get("outcome") == outcome]

    def by_type(self, namespace: str, episode_type: EpisodeType, days: Optional[float] = None) -> list[MemoryItem]:
        pool = self.since(namespace, days) if days else self.all(namespace)
        return [it for it in pool if (it.store_specific or {}).get("episode_type") == episode_type]

    def for_concept(self, namespace: str, concept_id: str, days: Optional[float] = None) -> list[MemoryItem]:
        pool = self.since(namespace, days) if days else self.all(namespace)
        return [it for it in pool if concept_id in (it.store_specific or {}).get("concept_ids", [])]

    def for_section(self, namespace: str, section_index: int) -> list[MemoryItem]:
        """Episodes tagged to a specific lesson section index."""
        return [
            it for it in self.all(namespace)
            if (it.store_specific or {}).get("section_index") == section_index
        ]

    def signal_counts(self, namespace: str, days: float = 30.0) -> dict[str, int]:
        """Aggregate behavioural signals over recent episodes."""
        out: dict[str, int] = {}
        for it in self.since(namespace, days):
            for k, v in ((it.store_specific or {}).get("signals") or {}).items():
                if v:
                    out[k] = out.get(k, 0) + 1
        return out

    def time_on_task(self, namespace: str, days: float = 7.0) -> int:
        """Total recorded duration in seconds over the last N days."""
        total = 0
        for it in self.since(namespace, days):
            d = (it.store_specific or {}).get("duration_sec")
            if isinstance(d, (int, float)):
                total += int(d)
        return total

    def session_summary(self, namespace: str, days: float = 1.0) -> dict:
        """Quick stats on the most recent N-day window."""
        ep = self.since(namespace, days)
        out_by = {"success": 0, "struggle": 0, "neutral": 0, "mistake": 0}
        for it in ep:
            o = (it.store_specific or {}).get("outcome", "neutral")
            out_by[o] = out_by.get(o, 0) + 1
        return {
            "window_days": days,
            "n_episodes": len(ep),
            "outcomes": out_by,
            "time_on_task_sec": self.time_on_task(namespace, days=days),
        }

    def compact(
        self,
        namespace: str,
        *,
        before_days: float = 180.0,
        keep_types: Optional[tuple[str, ...]] = None,
    ) -> int:
        """Delete low-value episodes older than before_days.

        Preserves section completions, lesson milestones, and any episode
        with a non-neutral outcome. Returns the number of rows deleted."""
        keep_types = keep_types or (
            "section_completed",
            "lesson_completed",
            "lesson_started",
            "lesson_dropoff",
            "exercise_attempt",
        )
        cutoff = (datetime.now() - timedelta(days=before_days)).isoformat(timespec="seconds")
        deleted = 0
        for ep in self.all(namespace):
            if ep.created_at >= cutoff:
                continue
            ss = ep.store_specific or {}
            etype = ss.get("episode_type", "")
            outcome = ss.get("outcome", "neutral")
            if etype in keep_types or outcome != "neutral":
                continue
            self.delete(ep.id)
            deleted += 1
        return deleted
