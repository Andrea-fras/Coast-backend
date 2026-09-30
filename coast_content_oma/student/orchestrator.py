"""StudentOrchestrator — assembles the personalized profile block for Pedro.

Given a (user_id, folder) and an optional current query, returns:
  1. A structured dict (for programmatic consumption / teacher dashboard)
  2. A compact text block suitable for injection into Pedro's system prompt

The text block is designed to be small (<800 chars typical) so we can
afford to attach it to every Pedro request without dominating the
context window — bulk content material still comes from Content OMA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..stores import make_namespace
from ..stores._semantic_base import SemanticStoreBase  # noqa: F401 (forces import)
from .accomplishments import summarize_accomplishments
from .stores.concept_mastery import days_since_evidence, effective_mastery
from .stores.academic_identity import effective_trait_confidence
from ..concept_resolve import ConceptResolver, resolver_for_db
from .stores import (
    ActiveContextStore,
    AcademicIdentityStore,
    ConceptMasteryStore,
    EpisodeStore,
    PatternStore,
    course_namespace,
    identity_namespace,
    list_course_namespaces,
    parse_course_namespace,
)


class StudentOrchestrator:
    def __init__(
        self,
        active: ActiveContextStore,
        mastery: ConceptMasteryStore,
        episodes: EpisodeStore,
        patterns: PatternStore,
        identity: AcademicIdentityStore,
        resolver: Optional[ConceptResolver] = None,
    ):
        self.active = active
        self.mastery = mastery
        self.episodes = episodes
        self.patterns = patterns
        self.identity = identity
        self.resolver = resolver

    # ── Profile bundle (structured) ───────────────────────────────

    def build_profile(
        self,
        user_id: int | str,
        folder: str,
        current_concept_ids: Optional[list[str]] = None,
    ) -> dict:
        course_ns = course_namespace(user_id, folder)
        from ..course_identity import content_namespace_for_student
        content_ns = content_namespace_for_student(user_id, folder)
        identity_ns = identity_namespace(user_id)
        current_concept_ids = list(dict.fromkeys(current_concept_ids or []))
        if self.resolver:
            current_concept_ids = self.resolver.resolve_many(content_ns, current_concept_ids)

        active_snapshot = self.active.snapshot(course_ns)
        mastery_overview = self.mastery.overview(course_ns)
        recent = self.episodes.session_summary(course_ns, days=7.0)

        # Weakest + strongest concepts (course-wide, alias-resolved)
        if self.resolver:
            weakest = [
                self._mastery_card_from_agg(a)
                for a in self.mastery.weakest_resolved(course_ns, content_ns, self.resolver, k=5)
            ]
            strongest = [
                self._mastery_card_from_agg(a)
                for a in self.mastery.strongest_resolved(course_ns, content_ns, self.resolver, k=3)
            ]
        else:
            weakest = [self._mastery_card(it) for it in self.mastery.weakest(course_ns, k=5)]
            strongest = [self._mastery_card(it) for it in self.mastery.strongest(course_ns, k=3)]

        # Concept-specific: if we know which concepts the current query
        # touches, surface the student's state on exactly those.
        focused_mastery: list[dict] = []
        for cid in current_concept_ids:
            if self.resolver:
                agg = self.mastery.aggregate_for_concept(
                    course_ns, content_ns, cid, self.resolver,
                )
                if agg:
                    focused_mastery.append(self._mastery_card_from_agg(agg))
                    continue
            it = self.mastery.for_concept(course_ns, cid)
            if it:
                focused_mastery.append(self._mastery_card(it))

        # Patterns — keep the top few by confidence.
        course_patterns = [
            {
                "type": (it.store_specific or {}).get("pattern_type"),
                "text": it.content,
                "confidence": (it.store_specific or {}).get("confidence", 0.0),
            }
            for it in self.patterns.top_confidence(course_ns, k=6)
        ]
        golden_moments = [
            {
                "type": "golden_moment",
                "text": it.content,
                "concept_ids": (it.store_specific or {}).get("related_concept_ids") or [],
                "confidence": (it.store_specific or {}).get("confidence", 0.0),
            }
            for it in self.patterns.all(course_ns)
            if (it.store_specific or {}).get("pattern_type") == "golden_moment"
        ]
        for moment in golden_moments:
            if self.resolver:
                moment["concept_ids"] = self.resolver.resolve_many(content_ns, moment["concept_ids"])
        if current_concept_ids:
            focus_ids = set(current_concept_ids)
            golden_moments = [g for g in golden_moments if focus_ids.intersection(g["concept_ids"])]
        # Select relevant evidence before bounding the profile, including old aliases.
        golden_moments = sorted(golden_moments, key=lambda g: g["confidence"], reverse=True)[:5]
        identity_traits = [
            {
                "type": (it.store_specific or {}).get("trait_type"),
                "text": it.content,
                "quote": (it.store_specific or {}).get("evidence_quote"),
                "confidence": effective_trait_confidence(it.store_specific or {}),
            }
            for it in self.identity.all_traits(identity_ns)
        ]

        accomplishments = summarize_accomplishments(
            self.episodes, course_ns, mastery=self.mastery,
        )

        # Spaced repetition: solid concepts whose effective mastery decayed.
        due_for_review = [
            self._mastery_card_from_agg(a)
            for a in (
                self.mastery.due_for_review_resolved(course_ns, content_ns, self.resolver, k=5)
                if self.resolver
                else []
            )
        ] or [
            self._mastery_card(it)
            for it in self.mastery.due_for_review(course_ns, k=5)
        ]

        # Autobiographical cross-course history — what Pedro taught this
        # student in other courses, so a returning student is never a stranger.
        other_courses = self._other_courses_summary(user_id, folder)

        return {
            "user_id": str(user_id),
            "folder": folder,
            "course_namespace": course_ns,
            "identity_namespace": identity_ns,
            "active_context": active_snapshot,
            "mastery_overview": mastery_overview,
            "weakest_concepts": weakest,
            "strongest_concepts": strongest,
            "focused_mastery": focused_mastery,
            "requested_concept_count": len(set(current_concept_ids)),
            "requested_concept_ids": list(dict.fromkeys(current_concept_ids)),
            "due_for_review": due_for_review,
            "other_courses": other_courses,
            "recent_window": recent,
            "course_patterns": course_patterns,
            "golden_moments": golden_moments,
            "identity_traits": identity_traits,
            "accomplishments": accomplishments,
        }

    def _other_courses_summary(
        self,
        user_id: int | str,
        current_folder: str,
        max_courses: int = 4,
    ) -> list[dict]:
        """Compact per-course summaries for every OTHER course this student
        has history in. Read straight from the per-course namespaces — no
        denormalized copy to drift out of sync."""
        current_ns = course_namespace(user_id, current_folder)
        out: list[dict] = []
        for ns in list_course_namespaces(self.episodes.db_path, user_id):
            if ns == current_ns:
                continue
            _, folder_slug = parse_course_namespace(ns)
            if not folder_slug:
                continue
            overview = self.mastery.overview(ns)
            acc = summarize_accomplishments(
                self.episodes, ns, mastery=self.mastery, window_days=365.0,
            )
            sections_done = acc.get("sections_completed") or []
            if not overview.get("n_concepts") and not sections_done:
                continue
            confident = [
                self._mastery_card(it)["name"]
                for it in self.mastery.strongest(ns, k=4)
                if effective_mastery(it.store_specific or {}) >= 0.7
            ]
            out.append({
                "folder": folder_slug,
                "n_concepts": overview.get("n_concepts", 0),
                "avg_mastery": overview.get("avg_mastery", 0.0),
                "n_fading": overview.get("n_fading", 0),
                "sections_completed": len(sections_done),
                "recent_section_titles": sections_done[-3:],
                "confident_in": [c for c in confident if c],
            })
            if len(out) >= max_courses:
                break
        return out

    # ── Cross-course profile (for global chat) ─────────────────

    def build_global_profile(self, user_id: int | str) -> dict:
        """Aggregate Student OMA data across every course folder the
        student has interacted with."""
        identity_ns = identity_namespace(user_id)
        courses: list[dict] = []

        for ns in list_course_namespaces(self.episodes.db_path, user_id):
            _, folder_slug = parse_course_namespace(ns)
            if not folder_slug:
                continue
            overview = self.mastery.overview(ns)
            recent = self.episodes.session_summary(ns, days=30.0)
            accomplishments = summarize_accomplishments(
                self.episodes, ns, mastery=self.mastery,
            )
            if not overview.get("n_concepts") and not recent.get("n_episodes") and not accomplishments.get("narrative_lines"):
                continue

            strongest = [self._mastery_card(it) for it in self.mastery.strongest(ns, k=5)]
            weakest_raw = [self._mastery_card(it) for it in self.mastery.weakest(ns, k=5)]
            struggled = [
                w for w in weakest_raw
                if w.get("score", 1.0) <= 0.35 or w.get("struggles", 0) >= 1
            ]
            ac = self.active.snapshot(ns)
            focus = (ac.get("current_focus") or {}).get("text") or ""
            if not focus and ac.get("recent_topics"):
                focus = ac["recent_topics"][0].get("text", "")

            courses.append({
                "folder": folder_slug,
                "namespace": ns,
                "mastery_overview": overview,
                "recent_window": recent,
                "strongest_concepts": strongest,
                "struggled_concepts": struggled,
                "current_focus": focus,
                "patterns": [
                    {
                        "type": (it.store_specific or {}).get("pattern_type"),
                        "text": it.content,
                        "confidence": (it.store_specific or {}).get("confidence", 0.0),
                    }
                    for it in self.patterns.top_confidence(ns, k=3)
                ],
                "accomplishments": accomplishments,
            })

        identity_traits = [
            {
                "type": (it.store_specific or {}).get("trait_type"),
                "text": it.content,
                "quote": (it.store_specific or {}).get("evidence_quote"),
                "confidence": effective_trait_confidence(it.store_specific or {}),
            }
            for it in self.identity.all_traits(identity_ns)
        ]

        return {
            "user_id": str(user_id),
            "courses": courses,
            "identity_traits": identity_traits,
        }

    def to_global_prompt_block(self, profile: dict, max_chars: int = 1600) -> str:
        from .prompt_budget import global_profile
        return global_profile(profile, max_chars)

    def to_prompt_block(self, profile: dict, max_chars: int = 1200) -> str:
        from .prompt_budget import course_profile
        return course_profile(profile, max_chars)

    # ── Helpers ───────────────────────────────────────────────────

    def _mastery_card_from_agg(self, agg: dict) -> dict:
        days = days_since_evidence(agg)
        return {
            "concept_id": agg.get("concept_id"),
            "name": agg.get("concept_name"),
            "score": float(agg.get("mastery_score", 0.0)),
            "effective_score": float(agg.get("effective_score", agg.get("mastery_score", 0.0))),
            "days_since_practiced": round(days, 1) if days is not None else None,
            "confidence": float(agg.get("confidence", 0.0)),
            "successes": agg.get("successes", 0),
            "struggles": agg.get("struggles", 0),
            "last_seen": agg.get("last_seen"),
            "last_eval_state": agg.get("last_eval_state"),
            "misconception_state": agg.get("misconception_state"),
            "misconception_type": agg.get("misconception_type"),
        }

    def _mastery_card(self, item) -> dict:
        ss = item.store_specific or {}
        days = days_since_evidence(ss)
        return {
            "concept_id": ss.get("concept_id"),
            "name": ss.get("concept_name"),
            "score": float(ss.get("mastery_score", 0.0)),
            "effective_score": effective_mastery(ss),
            "days_since_practiced": round(days, 1) if days is not None else None,
            "confidence": float(ss.get("confidence", 0.0)),
            "successes": ss.get("successes", 0),
            "struggles": ss.get("struggles", 0),
            "last_seen": ss.get("last_seen"),
            "last_eval_state": ss.get("last_eval_state"),
            "misconception_state": ss.get("misconception_state"),
            "misconception_type": ss.get("misconception_type"),
        }


def build_student_orchestrator(db_path: Path) -> StudentOrchestrator:
    """Factory — one orchestrator with all five stores wired to a
    shared SQLite database."""
    resolver = resolver_for_db(db_path)
    return StudentOrchestrator(
        active=ActiveContextStore(db_path),
        mastery=ConceptMasteryStore(db_path),
        episodes=EpisodeStore(db_path),
        patterns=PatternStore(db_path),
        identity=AcademicIdentityStore(db_path),
        resolver=resolver,
    )
