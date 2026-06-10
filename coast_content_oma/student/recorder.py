"""StudentRecorder — single entry point Coast calls to record student events.

Each call to record_episode():
  1. Writes a row to EpisodeStore.
  2. Bayesian-updates ConceptMastery for every concept referenced.
  3. Refreshes ActiveContextStore (current_focus, last_unresolved, etc.)
     based on signals in the event.

This keeps the four per-course stores in sync without requiring callers
to know about each one.
"""

from __future__ import annotations

import logging
from typing import Optional

from .stores import (
    ActiveContextStore,
    ConceptMasteryStore,
    EpisodeStore,
    course_namespace,
)
from .stores.episode import ALLOWED_TYPES, ALLOWED_OUTCOMES

logger = logging.getLogger(__name__)


EpisodeOutcome = str  # success | struggle | neutral | mistake


class StudentRecorder:
    """Thin orchestrator over the four per-course stores."""

    def __init__(
        self,
        active_context: ActiveContextStore,
        concept_mastery: ConceptMasteryStore,
        episodes: EpisodeStore,
    ):
        self.active = active_context
        self.mastery = concept_mastery
        self.episodes = episodes

    # ── Public API ────────────────────────────────────────────────

    def record_episode(
        self,
        user_id: int | str,
        folder: str,
        episode_type: str,
        summary: str,
        outcome: EpisodeOutcome = "neutral",
        concept_refs: Optional[list[dict]] = None,
        lesson_id: Optional[str] = None,
        user_message: Optional[str] = None,
        assistant_response: Optional[str] = None,
        duration_sec: Optional[int] = None,
        signals: Optional[dict] = None,
        source: str = "chat",
        focus_text: Optional[str] = None,
        unresolved_text: Optional[str] = None,
        open_question: Optional[str] = None,
        section_title: Optional[str] = None,
        section_index: Optional[int] = None,
    ) -> dict:
        """One call writes Episode + updates Mastery + refreshes ActiveContext.

        concept_refs: list of {"concept_id": ..., "concept_name": ...} —
            we need both because ConceptMastery caches the name for
            inspectability without joining back to Content OMA.

        focus_text / unresolved_text / open_question: optional inputs the
            caller can pass to update ActiveContext explicitly. If omitted,
            we infer reasonable defaults from the episode."""
        ns = course_namespace(user_id, folder)
        concept_refs = concept_refs or []
        concept_ids = [c.get("concept_id") for c in concept_refs if c.get("concept_id")]

        # 1. Episode
        episode = self.episodes.record(
            ns,
            episode_type=episode_type,
            summary=summary,
            outcome=outcome,
            concept_ids=concept_ids,
            lesson_id=lesson_id,
            user_message=user_message,
            assistant_response=assistant_response,
            duration_sec=duration_sec,
            signals=signals,
            source=source,
            section_title=section_title,
            section_index=section_index,
        )

        # 2. Mastery — one update per concept involved (skip neutral —
        # those are intros, transitions, or ambiguous turns that should
        # not drag scores down).
        mastery_updates = []
        if outcome in ("success", "struggle"):
            for c in concept_refs:
                cid = c.get("concept_id")
                cname = c.get("concept_name") or cid
                if not cid:
                    continue
                item = self.mastery.record_evidence(
                    ns,
                    concept_id=cid,
                    concept_name=cname,
                    outcome=outcome,
                    lesson_id=lesson_id,
                )
                mastery_updates.append(item.id)

        # 3. ActiveContext refresh
        ctx_updates = []
        if focus_text:
            ctx_updates.append(self.active.set_fragment(
                ns, "current_focus", focus_text,
                concept_ids=concept_ids, lesson_id=lesson_id,
            ).id)
        if unresolved_text:
            ctx_updates.append(self.active.set_fragment(
                ns, "last_unresolved", unresolved_text,
                concept_ids=concept_ids, lesson_id=lesson_id,
            ).id)
        elif outcome == "success" and not unresolved_text:
            # Implicitly clear last_unresolved when the student succeeds
            # on the thing they were stuck on.
            for prev in self.active.current(ns, fragment_type="last_unresolved"):
                if set(prev.entities) & set(concept_ids):
                    self.active.resolve_question(prev.id)
        if open_question:
            ctx_updates.append(self.active.add_open_question(
                ns, open_question, concept_ids=concept_ids,
            ).id)
        # Always log a recent_topic when concepts are touched.
        if concept_refs and episode_type in ("qa", "lesson_started", "lesson_completed", "exercise_attempt"):
            topic_label = ", ".join(c.get("concept_name") for c in concept_refs[:3] if c.get("concept_name"))
            if topic_label:
                ctx_updates.append(self.active.add_recent_topic(
                    ns, topic_label, concept_ids=concept_ids,
                ).id)

        return {
            "episode_id": episode.id,
            "mastery_updates": mastery_updates,
            "context_updates": ctx_updates,
        }

    # ── Convenience wrappers ──────────────────────────────────────

    def record_qa(self, user_id, folder, *, user_message: str, assistant_response: str,
                  concept_refs: list[dict], outcome: EpisodeOutcome = "neutral",
                  lesson_id: Optional[str] = None, signals: Optional[dict] = None,
                  duration_sec: Optional[int] = None,
                  section_index: Optional[int] = None) -> dict:
        return self.record_episode(
            user_id, folder, "qa",
            summary=user_message[:200],
            outcome=outcome,
            concept_refs=concept_refs,
            lesson_id=lesson_id,
            user_message=user_message,
            assistant_response=assistant_response,
            duration_sec=duration_sec,
            signals=signals,
            section_index=section_index,
        )

    def record_exercise(self, user_id, folder, *, summary: str,
                        concept_refs: list[dict], outcome: EpisodeOutcome,
                        lesson_id: Optional[str] = None,
                        duration_sec: Optional[int] = None) -> dict:
        return self.record_episode(
            user_id, folder, "exercise_attempt",
            summary=summary,
            outcome=outcome,
            concept_refs=concept_refs,
            lesson_id=lesson_id,
            duration_sec=duration_sec,
            source="exercise",
        )

    def record_lesson_completed(self, user_id, folder, *, lesson_id: str,
                                 lesson_title: str,
                                 concept_refs: list[dict],
                                 duration_sec: Optional[int] = None) -> dict:
        return self.record_episode(
            user_id, folder, "lesson_completed",
            summary=f"Completed lesson: {lesson_title}",
            outcome="success",
            concept_refs=concept_refs,
            lesson_id=lesson_id,
            duration_sec=duration_sec,
            source="lesson_player",
            focus_text=f"Just completed: {lesson_title}",
        )

    def record_section_completed(
        self,
        user_id,
        folder,
        *,
        section_title: str,
        section_index: Optional[int] = None,
        lesson_id: Optional[str] = None,
    ) -> dict:
        """Log that the student finished a lesson section (SECTION_COMPLETE)."""
        return self.record_episode(
            user_id, folder, "section_completed",
            summary=f"Completed section: {section_title}",
            outcome="success",
            concept_refs=[],
            lesson_id=lesson_id,
            user_message=None,
            assistant_response=None,
            source="lesson_player",
            focus_text=f"Just completed section: {section_title}",
            section_title=section_title,
            section_index=section_index,
        )

    def record_lesson_dropoff(self, user_id, folder, *, lesson_id: str,
                               lesson_title: str,
                               concept_refs: list[dict],
                               at_section: str,
                               duration_sec: Optional[int] = None) -> dict:
        return self.record_episode(
            user_id, folder, "lesson_dropoff",
            summary=f"Dropped off lesson {lesson_title} at section: {at_section}",
            outcome="struggle",
            concept_refs=concept_refs,
            lesson_id=lesson_id,
            duration_sec=duration_sec,
            source="lesson_player",
            unresolved_text=f"Left {lesson_title} at section: {at_section}",
        )
