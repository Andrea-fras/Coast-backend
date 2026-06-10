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

from ..stores._semantic_base import SemanticStoreBase  # noqa: F401 (forces import)
from .accomplishments import summarize_accomplishments
from .stores.concept_mastery import days_since_evidence, effective_mastery
from .stores.academic_identity import effective_trait_confidence
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
    ):
        self.active = active
        self.mastery = mastery
        self.episodes = episodes
        self.patterns = patterns
        self.identity = identity

    # ── Profile bundle (structured) ───────────────────────────────

    def build_profile(
        self,
        user_id: int | str,
        folder: str,
        current_concept_ids: Optional[list[str]] = None,
    ) -> dict:
        course_ns = course_namespace(user_id, folder)
        identity_ns = identity_namespace(user_id)
        current_concept_ids = current_concept_ids or []

        active_snapshot = self.active.snapshot(course_ns)
        mastery_overview = self.mastery.overview(course_ns)
        recent = self.episodes.session_summary(course_ns, days=7.0)

        # Weakest + strongest concepts (course-wide)
        weakest = [self._mastery_card(it) for it in self.mastery.weakest(course_ns, k=5)]
        strongest = [self._mastery_card(it) for it in self.mastery.strongest(course_ns, k=3)]

        # Concept-specific: if we know which concepts the current query
        # touches, surface the student's state on exactly those.
        focused_mastery: list[dict] = []
        for cid in current_concept_ids:
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
        identity_traits = [
            {
                "type": (it.store_specific or {}).get("trait_type"),
                "text": it.content,
                "confidence": effective_trait_confidence(it.store_specific or {}),
            }
            for it in self.identity.all_traits(identity_ns)
        ]

        accomplishments = summarize_accomplishments(
            self.episodes, course_ns, mastery=self.mastery,
        )

        # Spaced repetition: solid concepts whose effective mastery decayed.
        due_for_review = [
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
            "due_for_review": due_for_review,
            "other_courses": other_courses,
            "recent_window": recent,
            "course_patterns": course_patterns,
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
        """Compact cross-course block for global Pedro chat."""
        courses = profile.get("courses") or []
        if not courses:
            return ""

        out: list[str] = []
        out.append("--- STUDENT PROFILE (cross-course learning history) ---")
        out.append(
            "The student may ask what they have studied, mastered, or struggled with. "
            "Answer ONLY from the course summaries below — do not invent topics."
        )

        for c in courses:
            folder = c.get("folder", "?")
            acc_lines = (c.get("accomplishments") or {}).get("narrative_lines") or []
            if acc_lines:
                out.append(f"Course '{folder}' — what they have done:")
                for line in acc_lines[:4]:
                    out.append(f"  • {line}")
            else:
                rw = c.get("recent_window") or {}
                o = rw.get("outcomes") or {}
                n = rw.get("n_episodes", 0)
                line = f"Course '{folder}': {n} interactions in last 30 days"
                if o:
                    line += f" ({o.get('success', 0)} ok / {o.get('struggle', 0)} struggled)"
                out.append(line)

            focus = (c.get("current_focus") or "").strip()
            if focus:
                out.append(f"  Recent focus: {focus[:200]}")

            strong = c.get("strongest_concepts") or []
            mastered = [s for s in strong if s.get("score", 0) >= 0.75 and s.get("struggles", 0) == 0]
            if mastered:
                names = ", ".join(s["name"] for s in mastered[:6])
                out.append(f"  Confident in: {names}")

            struggled = c.get("struggled_concepts") or []
            real_struggles = [s for s in struggled if s.get("struggles", 0) >= 1]
            if real_struggles:
                names = ", ".join(
                    f"{s['name']} ({s['struggles']} correction(s))" for s in real_struggles[:3]
                )
                out.append(f"  Pedro corrected them on: {names}")

            pats = c.get("patterns") or []
            if pats:
                out.append("  Patterns: " + "; ".join(p["text"][:80] for p in pats[:2]))

        if profile.get("identity_traits"):
            out.append("Cross-course traits:")
            for t in profile["identity_traits"][:3]:
                out.append(f"  - {t['text']}")

        out.append("--- END STUDENT PROFILE ---")
        block = "\n".join(out)
        if len(block) > max_chars:
            block = block[:max_chars] + "\n[... truncated ...]"
        return block

    # ── Prompt block (compact text for Pedro) ────────────────────

    def to_prompt_block(self, profile: dict, max_chars: int = 1200) -> str:
        out: list[str] = []
        out.append("--- STUDENT PROFILE (use this to personalize your response) ---")

        # Authoritative progress from CourseOutline (100% accurate).
        ledger = profile.get("progress_ledger") or {}
        completed = ledger.get("completed_sections") or []
        if completed:
            out.append("Lesson progress (authoritative):")
            for sec in completed:
                title = sec.get("title") or f"Section {sec.get('index', 0) + 1}"
                topics = sec.get("key_topics") or []
                if topics:
                    out.append(f"  • Finished: {title} — topics: {', '.join(topics[:6])}")
                else:
                    out.append(f"  • Finished: {title}")
            cs = ledger.get("current_section")
            total = ledger.get("total_sections")
            if cs is not None and total:
                if ledger.get("is_complete"):
                    out.append(f"  • Lesson complete ({total}/{total} sections)")
                elif ledger.get("current_section_title"):
                    out.append(
                        f"  • Currently on section {cs + 1}/{total}: "
                        f"{ledger['current_section_title']}"
                    )

        # One-off practice mistakes (Pedro re-teaches — not "struggling").
        mistakes = profile.get("section_mistakes") or []
        if mistakes:
            out.append("Practice mistakes (re-taught, usually corrected after):")
            by_sec: dict[int, list] = {}
            for m in mistakes:
                idx = m.get("section_index")
                if idx is None:
                    continue
                by_sec.setdefault(int(idx), []).append(m.get("user_message") or "")
            for idx in sorted(by_sec):
                answers = by_sec[idx]
                out.append(
                    f"  • Section {idx + 1}: "
                    + "; ".join(f'"{a[:60]}"' for a in answers[:3])
                )

        # Persistent struggle only — repeated wrong or >50% wrong ratio.
        struggling = profile.get("struggling_topics") or []
        if struggling:
            out.append("Struggling with (persistent — not one-off mistakes):")
            for t in struggling[:4]:
                out.append(
                    f"  • {t.get('name')}: {t.get('mistakes', 0)} wrong / "
                    f"{t.get('successes', 0)} right"
                )

        # Narrative from accomplishments (section completions, etc.).
        acc = profile.get("accomplishments") or {}
        narrative = acc.get("narrative_lines") or []
        extra_narrative = [
            line for line in narrative
            if not line.startswith("Completed lesson section:")
            and not line.startswith("Demonstrated strong")
            and not line.startswith("Mostly competent")
            and not line.startswith("Worked through")
            and not line.startswith("Still building")
        ]
        if extra_narrative:
            out.append("Other notes:")
            for line in extra_narrative[:4]:
                out.append(f"  • {line}")

        ac = profile.get("active_context") or {}
        if ac.get("current_focus"):
            out.append(f"Currently working on: {ac['current_focus']['text']}")
        if ac.get("current_goal"):
            out.append(f"Goal: {ac['current_goal']['text']}")
        if ac.get("last_unresolved"):
            out.append(f"Unresolved from last time: {ac['last_unresolved']['text']}")
        if ac.get("open_questions"):
            qs = ac["open_questions"][:2]
            out.append("Open questions: " + " | ".join(q["text"] for q in qs))

        # Focused mastery wins over generic weakest/strongest — Pedro should
        # know exactly where the student stands on what's being discussed.
        if profile.get("focused_mastery"):
            out.append("On the concepts in this query:")
            for m in profile["focused_mastery"]:
                out.append(f"  - {m['name']}: mastery {m['score']:.2f} "
                           f"({m['successes']} ok)")
        else:
            strongest = profile.get("strongest_concepts") or []
            if strongest:
                mastered = [s for s in strongest if s.get("score", 0) >= 0.75]
                if mastered:
                    names = ", ".join(s["name"] for s in mastered[:6])
                    out.append(f"Confident in: {names}")

            struggling = profile.get("struggling_topics") or []
            if struggling:
                out.append(
                    "Struggling with: "
                    + ", ".join(
                        f"{t['name']} ({t['mistakes']} wrong / {t['successes']} right)"
                        for t in struggling[:3]
                    )
                )

        # Top course patterns — keep terse.
        if profile.get("course_patterns"):
            top = profile["course_patterns"][:3]
            out.append("Course patterns:")
            for p in top:
                out.append(f"  - {p['text']}  [{p['type']}, conf {p['confidence']:.2f}]")

        # Spaced repetition — concepts they once knew that have faded.
        due = profile.get("due_for_review") or []
        if due:
            out.append("Fading mastery (knew well, not practiced recently):")
            for m in due[:4]:
                days = m.get("days_since_practiced")
                ago = f", last practiced {int(days)}d ago" if days is not None else ""
                out.append(
                    f"  • {m['name']}: was {m['score']:.2f}, "
                    f"now ~{m.get('effective_score', m['score']):.2f}{ago}"
                )
            out.append(
                "  → When there's a natural opening, offer a quick refresher "
                "on one of these before moving on."
            )

        # Autobiographical history from other courses with Pedro.
        others = profile.get("other_courses") or []
        if others:
            out.append("History from their other courses (you have taught them before):")
            for c in others:
                line = f"  • '{c['folder']}': {c['sections_completed']} sections completed"
                if c.get("n_concepts"):
                    line += f", {c['n_concepts']} concepts (avg mastery {c['avg_mastery']:.2f})"
                out.append(line)
                if c.get("confident_in"):
                    out.append("    Confident in: " + ", ".join(c["confident_in"][:4]))

        # Cross-course identity (these are gold for new courses / first messages).
        if profile.get("identity_traits"):
            top = profile["identity_traits"][:3]
            out.append("Cross-course traits:")
            for t in top:
                out.append(f"  - {t['text']}  [{t['type']}, conf {t['confidence']:.2f}]")

        rw = profile.get("recent_window") or {}
        if rw.get("n_episodes"):
            o = rw.get("outcomes") or {}
            n_mistakes = o.get("mistake", 0)
            line = (
                f"Last 7 days: {rw['n_episodes']} logged events, "
                f"{rw.get('time_on_task_sec', 0) // 60} min total"
            )
            if n_mistakes:
                line += f", {n_mistakes} practice mistake(s)"
            out.append(line)

        out.append("--- END STUDENT PROFILE ---")
        out.append(
            "PROFILE RULES: Lead with what the student has DONE (above), not "
            "invented weaknesses. A single practice mistake does NOT mean they "
            "are struggling — Pedro re-teaches and they usually get it right. "
            "Only 'Struggling with' reflects persistent difficulty (>50% wrong "
            "or repeated mistakes on the same topic). Never say they found "
            "something 'tricky' unless Open questions say so. If accomplishments "
            "show strong competence in a topic, treat it as mastered. If 'Fading "
            "mastery' lists concepts, you MAY proactively offer a short refresher "
            "— frame it as 'it's been a while since we covered X', never as a "
            "weakness. Use other-course history to connect new material to "
            "things they already learned with you."
        )
        block = "\n".join(out)
        if len(block) > max_chars:
            block = block[:max_chars] + "\n[... truncated ...]"
        return block

    # ── Helpers ───────────────────────────────────────────────────

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
        }


def build_student_orchestrator(db_path: Path) -> StudentOrchestrator:
    """Factory — one orchestrator with all five stores wired to a
    shared SQLite database."""
    return StudentOrchestrator(
        active=ActiveContextStore(db_path),
        mastery=ConceptMasteryStore(db_path),
        episodes=EpisodeStore(db_path),
        patterns=PatternStore(db_path),
        identity=AcademicIdentityStore(db_path),
    )
