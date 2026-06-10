"""Consolidators — derive Patterns from Episodes (course-level), and
roll Patterns across all of a student's courses into AcademicIdentity.

Designed to run periodically (end of session, or on a cron). All
inferences are explicit rules — no LLM needed at this layer so it stays
fast, cheap, and inspectable. Each inference records its derivation so
the teacher view can show "why was this pattern flagged?"
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Optional

from .stores import (
    ActiveContextStore,
    AcademicIdentityStore,
    ConceptMasteryStore,
    EpisodeStore,
    PatternStore,
    course_namespace,
    identity_namespace,
)

logger = logging.getLogger(__name__)


# ── Course-level consolidator ────────────────────────────────────────

class CourseConsolidator:
    """Reads the per-course Episode + Mastery stores, writes Patterns."""

    def __init__(self, episodes: EpisodeStore, mastery: ConceptMasteryStore, patterns: PatternStore):
        self.episodes = episodes
        self.mastery = mastery
        self.patterns = patterns

    def run(self, namespace: str, window_days: float = 30.0) -> dict:
        """Run every detector and write patterns. Returns a summary."""
        out: dict[str, int] = {}
        out["preference_patterns"] = self._infer_format_preferences(namespace, window_days)
        out["struggle_patterns"] = self._infer_struggle_clusters(namespace, window_days)
        out["pace_patterns"] = self._infer_session_pace(namespace, window_days)
        out["giveup_patterns"] = self._infer_giveup_pattern(namespace, window_days)
        out["topic_strength"] = self._infer_topic_strengths(namespace)
        return out

    # ── Detectors ─────────────────────────────────────────────────

    def _infer_format_preferences(self, ns: str, days: float) -> int:
        """Detect whether the student asks for examples, definitions
        first, diagrams, step-by-step etc. Based on signal frequencies."""
        signals = self.episodes.signal_counts(ns, days=days)
        total = sum(signals.values()) or 1
        n_written = 0

        mapping = [
            ("asked_for_example", "prefers_examples",
             "Often asks for worked examples"),
            ("asked_for_diagram", "prefers_diagrams",
             "Often asks to see diagrams or visuals"),
            ("asked_for_definition_first", "prefers_definitions_first",
             "Tends to ask for the definition before exploring further"),
            ("asked_followup", "asks_followups",
             "Frequently asks follow-up questions"),
            ("asked_step_by_step", "prefers_step_by_step",
             "Prefers step-by-step explanations"),
            ("asked_for_shorter", "prefers_short_answers",
             "Often asks for shorter / more concise answers"),
        ]
        for signal_key, pattern_type, description in mapping:
            count = signals.get(signal_key, 0)
            ratio = count / total
            if count >= 3 and ratio >= 0.20:
                conf = min(1.0, ratio + count * 0.02)
                self.patterns.upsert(
                    ns, pattern_type, description,
                    confidence=conf, evidence_count=count,
                    derivation=f"signal '{signal_key}' fired in {count}/{total} recent episodes",
                    dedupe_key=signal_key,
                )
                n_written += 1
        return n_written

    def _infer_struggle_clusters(self, ns: str, days: float) -> int:
        """Concepts the student has struggled on >=2 times in window get
        flagged. Concepts struggled together (same episode) get grouped
        into a struggle_cluster pattern."""
        n_written = 0
        # One-off mistakes don't count — only persistent struggle patterns.
        struggle_episodes = self.episodes.by_outcome(ns, "struggle", days=days)
        mistake_episodes = self.episodes.by_outcome(ns, "mistake", days=days)
        per_concept: Counter[str] = Counter()
        co_struggle: dict[frozenset, int] = defaultdict(int)
        for ep in list(struggle_episodes) + list(mistake_episodes):
            cids = (ep.store_specific or {}).get("concept_ids") or []
            for c in cids:
                per_concept[c] += 1
            if len(cids) >= 2:
                co_struggle[frozenset(cids)] += 1

        # Lookup concept names from the mastery store (cached there).
        name_by_id = {}
        successes_by_id: dict[str, int] = {}
        for it in self.mastery.all(ns):
            ss = it.store_specific or {}
            cid = ss.get("concept_id")
            if cid:
                name_by_id[cid] = ss.get("concept_name") or cid
                successes_by_id[cid] = int(ss.get("successes", 0) or 0)

        from .mastery_tier import is_topic_struggling
        for cid, n in per_concept.items():
            successes = successes_by_id.get(cid, 0)
            if not is_topic_struggling(successes, n):
                continue
            cname = name_by_id.get(cid, cid)
            self.patterns.upsert(
                ns, "weak_in_topic",
                f"Struggling with {cname} ({n} wrong / {successes} right in last {int(days)}d)",
                confidence=min(1.0, 0.4 + 0.15 * n),
                evidence_count=n,
                related_concept_ids=[cid],
                derivation=f"{n} struggle-outcome episodes mentioning concept {cid}",
                dedupe_key=cid,
            )
            n_written += 1

        for cluster, n in co_struggle.items():
            if n < 2 or len(cluster) > 4:
                continue
            names = [name_by_id.get(c, c) for c in cluster]
            self.patterns.upsert(
                ns, "struggle_cluster",
                f"Concepts often struggled with together: {', '.join(names)}",
                confidence=min(1.0, 0.3 + 0.2 * n),
                evidence_count=n,
                related_concept_ids=sorted(cluster),
                derivation=f"these concepts co-occurred in {n} struggle episodes",
                dedupe_key="|".join(sorted(cluster)),
            )
            n_written += 1
        return n_written

    def _infer_session_pace(self, ns: str, days: float) -> int:
        """Average session length, dropoff rate. Buckets the student into
        rough pace patterns."""
        eps = self.episodes.since(ns, days)
        if len(eps) < 5:
            return 0
        durations = [
            (e.store_specific or {}).get("duration_sec")
            for e in eps
            if (e.store_specific or {}).get("duration_sec")
        ]
        n_written = 0
        if durations:
            avg = sum(durations) / len(durations)
            bucket_min = int(avg // 60)
            if bucket_min <= 5:
                desc = "Brief interactions — typically under 5 minutes"
            elif bucket_min <= 20:
                desc = f"Focused short sessions — typically around {bucket_min} minutes"
            elif bucket_min <= 60:
                desc = f"Steady sessions — typically around {bucket_min} minutes"
            else:
                desc = "Long deep sessions — over an hour at a time"
            self.patterns.upsert(
                ns, "drops_off_after_n_min", desc,
                confidence=min(1.0, len(durations) / 10),
                evidence_count=len(durations),
                derivation=f"avg duration across {len(durations)} recent episodes = {bucket_min} min",
                dedupe_key="session_length",
            )
            n_written += 1

        dropoffs = self.episodes.by_type(ns, "lesson_dropoff", days=days)
        if len(dropoffs) >= 2:
            self.patterns.upsert(
                ns, "drops_off_after_n_min",
                f"Has dropped off mid-lesson {len(dropoffs)} times recently",
                confidence=min(1.0, 0.4 + 0.1 * len(dropoffs)),
                evidence_count=len(dropoffs),
                derivation=f"{len(dropoffs)} lesson_dropoff episodes in last {int(days)}d",
                dedupe_key="lesson_dropoffs",
            )
            n_written += 1
        return n_written

    def _infer_giveup_pattern(self, ns: str, days: float) -> int:
        signals = self.episodes.signal_counts(ns, days=days)
        n_give = signals.get("gave_up", 0)
        n_easy = signals.get("requested_easier", 0)
        if n_give >= 2 or n_easy >= 2:
            self.patterns.upsert(
                ns, "frequent_giveup",
                f"Has given up or asked for an easier path {n_give + n_easy} times recently",
                confidence=min(1.0, 0.4 + 0.15 * (n_give + n_easy)),
                evidence_count=n_give + n_easy,
                derivation=f"signals: gave_up={n_give}, requested_easier={n_easy}",
                dedupe_key="giveup",
            )
            return 1
        return 0

    def _infer_topic_strengths(self, ns: str) -> int:
        """Surface strong/weak concepts via mastery store directly."""
        n = 0
        for it in self.mastery.strongest(ns, k=3, min_n=3):
            ss = it.store_specific or {}
            cid = ss.get("concept_id")
            cname = ss.get("concept_name") or cid
            self.patterns.upsert(
                ns, "strong_in_topic",
                f"Confident in {cname} (mastery {ss.get('mastery_score', 0):.2f})",
                confidence=min(1.0, float(ss.get("mastery_score", 0))),
                evidence_count=ss.get("successes", 0),
                related_concept_ids=[cid] if cid else [],
                derivation=f"mastery_score={ss.get('mastery_score'):.2f} over {ss.get('successes', 0)} successes",
                dedupe_key=cid,
            )
            n += 1
        return n


# ── Cross-course consolidator ────────────────────────────────────────

class IdentityConsolidator:
    """Roll patterns across all of a student's per-course PatternStores
    into the cross-course AcademicIdentityStore."""

    def __init__(
        self,
        pattern_store: PatternStore,
        identity_store: AcademicIdentityStore,
        course_namespaces: list[str],
    ):
        self.pattern_store = pattern_store
        self.identity_store = identity_store
        self.course_namespaces = course_namespaces

    def run(self, user_id: int | str) -> dict:
        identity_ns = identity_namespace(user_id)
        # Bucket course-level patterns by pattern_type.
        by_type: dict[str, list[tuple[str, "MemoryItem"]]] = defaultdict(list)  # type: ignore
        for course_ns in self.course_namespaces:
            for it in self.pattern_store.all(course_ns):
                pt = (it.store_specific or {}).get("pattern_type", "")
                by_type[pt].append((course_ns, it))

        n_written = 0
        # Learning style — any preference pattern present in >=2 courses
        # becomes an identity trait.
        STYLE_MAP = {
            "prefers_examples": ("learning_style", "example-led learner"),
            "prefers_diagrams": ("learning_style", "visual learner — benefits from diagrams"),
            "prefers_definitions_first": ("learning_style", "definition-first learner"),
            "asks_followups": ("engagement_pattern", "engaged — frequently asks follow-ups"),
            "prefers_short_answers": ("learning_style", "prefers concise answers"),
            "prefers_step_by_step": ("learning_style", "prefers step-by-step explanations"),
        }
        for pat_type, (trait_type, desc) in STYLE_MAP.items():
            entries = by_type.get(pat_type, [])
            distinct_courses = {ns for ns, _ in entries}
            if len(distinct_courses) >= 2:
                avg_conf = sum((it.store_specific or {}).get("confidence", 0.0) for _, it in entries) / len(entries)
                self.identity_store.upsert_trait(
                    identity_ns,
                    trait_type=trait_type,
                    description=desc,
                    confidence=avg_conf,
                    evidence_courses=sorted(distinct_courses),
                    derivation=f"present in {len(distinct_courses)} courses",
                    dedupe_key=pat_type,
                )
                n_written += 1

        # Session pace
        pace_entries = by_type.get("drops_off_after_n_min", [])
        if pace_entries:
            descs = [it.content for _, it in pace_entries if it.content]
            if descs:
                self.identity_store.upsert_trait(
                    identity_ns,
                    trait_type="session_pattern",
                    description=f"Across courses: {descs[0]}",
                    confidence=min(1.0, 0.5 + 0.1 * len(descs)),
                    evidence_courses=sorted({ns for ns, _ in pace_entries}),
                    derivation=f"derived from {len(descs)} course session-length patterns",
                    dedupe_key="session_pattern",
                )
                n_written += 1

        # Giveup pattern across courses
        giveup_entries = by_type.get("frequent_giveup", [])
        if len(giveup_entries) >= 2:
            self.identity_store.upsert_trait(
                identity_ns,
                trait_type="motivation_pattern",
                description="Tends to disengage or request easier paths when blocked",
                confidence=min(1.0, 0.5 + 0.15 * len(giveup_entries)),
                evidence_courses=sorted({ns for ns, _ in giveup_entries}),
                derivation=f"frequent_giveup pattern in {len(giveup_entries)} courses",
                dedupe_key="giveup_motivation",
            )
            n_written += 1

        # Strengths / weaknesses summarised by topic
        for src_pat_type, trait_type, label in [
            ("strong_in_topic", "general_strength", "Consistently strong in"),
            ("weak_in_topic", "general_weakness", "Repeatedly weak in"),
        ]:
            entries = by_type.get(src_pat_type, [])
            topic_count: Counter[str] = Counter()
            for _, it in entries:
                for cid in it.entities:
                    topic_count[cid] += 1
            for cid, n in topic_count.items():
                if n < 1:
                    continue
                self.identity_store.upsert_trait(
                    identity_ns,
                    trait_type=trait_type,
                    description=f"{label}: {cid}",
                    confidence=min(1.0, 0.4 + 0.15 * n),
                    evidence_courses=sorted({ns for ns, _ in entries}),
                    derivation=f"{src_pat_type} for concept {cid} appeared in {n} course patterns",
                    dedupe_key=f"{src_pat_type}:{cid}",
                )
                n_written += 1

        return {"identity_traits_written": n_written}
