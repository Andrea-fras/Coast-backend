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
  - concept_label       Pedro's own name for the concept a graded answer tested
  - chat_message_ids    [student_msg_id, pedro_msg_id] in coast.db chat_messages —
                        the canonical transcript; episodes index it, never replace it
  - hinted              graded success that needed a hint / worked step
"""

from __future__ import annotations

import time
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
    "section_evaluation", # post-section cognitive assessment (authoritative)
    "workshop_artifact",  # what the student built in a workshop milestone (evaluator summary)
    "lesson_dropoff",     # student left a lesson before finishing
    "concept_reviewed",   # student revisited a known concept
    "self_assessment",    # student rated own understanding
    "external_event",     # anything else worth logging
)


ALLOWED_OUTCOMES = ("success", "struggle", "neutral", "mistake")


class EpisodeStore(SemanticStoreBase):
    STORE_NAME = "episode"

    def _init_db(self) -> None:
        super()._init_db()
        with connect_db(self.db_path) as conn:
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table}_type "
                f"ON {self.table}(namespace, (json_extract(store_specific, '$.episode_type')), created_at)"
            )

    # ── Write ─────────────────────────────────────────────────────

    def record(
        self,
        namespace: str,
        episode_type: EpisodeType,
        summary: str,
        outcome: EpisodeOutcome = "neutral",
        concept_ids: Optional[list[str]] = None,
        matched_concept_ids: Optional[list[str]] = None,
        lesson_id: Optional[str] = None,
        user_message: Optional[str] = None,
        assistant_response: Optional[str] = None,
        duration_sec: Optional[int] = None,
        signals: Optional[dict] = None,
        source: str = "chat",
        section_title: Optional[str] = None,
        section_index: Optional[int] = None,
        chat_message_ids: Optional[list[int]] = None,
        hinted: bool = False,
        concept_label: Optional[str] = None,
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
            "seq": time.time_ns(),  # strict order within a second; survives in-place rewrites
        }
        if matched_concept_ids:
            ss["matched_concept_ids"] = list(matched_concept_ids)
        if chat_message_ids:
            ss["chat_message_ids"] = [int(i) for i in chat_message_ids if i is not None]
        if hinted:
            ss["hinted"] = True
        if concept_label:
            ss["concept_label"] = concept_label[:120]
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
        return self.by_types(namespace, (episode_type,), days=days)

    def by_types(self, namespace: str, episode_types, days: Optional[float] = None,
                 section_index: Optional[int] = None) -> list[MemoryItem]:
        """Indexed filter by episode type (oldest first) — avoids loading a
        student's whole multi-year log to find a handful of graded attempts."""
        types = list(episode_types)
        sql = (f"SELECT * FROM {self.table} WHERE namespace = ? AND superseded_by IS NULL "
               f"AND json_extract(store_specific, '$.episode_type') IN ({','.join('?' * len(types))})")
        params: list = [namespace, *types]
        if days:
            sql += " AND created_at >= ?"
            params.append((datetime.now() - timedelta(days=days)).isoformat(timespec="seconds"))
        if section_index is not None:
            sql += " AND json_extract(store_specific, '$.section_index') = ?"
            params.append(int(section_index))
        with connect_db(self.db_path) as conn:
            rows = conn.execute(sql + " ORDER BY created_at, COALESCE(json_extract(store_specific, '$.seq'), 0), rowid", params).fetchall()
        return [_row_to_item(r, self.STORE_NAME) for r in rows if r]

    def for_concept(self, namespace: str, concept_id: str, days: Optional[float] = None) -> list[MemoryItem]:
        pool = self.since(namespace, days) if days else self.all(namespace)
        return [
            it for it in pool
            if concept_id in (it.store_specific or {}).get("concept_ids", [])
            or concept_id in (it.store_specific or {}).get("matched_concept_ids", [])
        ]

    def has_message(self, namespace: str, chat_message_id: int) -> bool:
        """True if a turn referencing this chat message is already recorded (idempotent backfills)."""
        with connect_db(self.db_path) as conn:
            row = conn.execute(
                f"SELECT 1 FROM {self.table}, json_each({self.table}.store_specific, '$.chat_message_ids') j "
                f"WHERE namespace = ? AND j.value = ? LIMIT 1",
                (namespace, int(chat_message_id)),
            ).fetchone()
        return row is not None

    def for_section(self, namespace: str, section_index: int) -> list[MemoryItem]:
        """Episodes tagged to a specific lesson section index."""
        with connect_db(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT * FROM {self.table} WHERE namespace = ? AND superseded_by IS NULL "
                f"AND json_extract(store_specific, '$.section_index') = ? "
                f"ORDER BY created_at, COALESCE(json_extract(store_specific, '$.seq'), 0), rowid",
                (namespace, int(section_index)),
            ).fetchall()
        return [_row_to_item(r, self.STORE_NAME) for r in rows if r]

    def mark_mistakes_resolved(
        self,
        namespace: str,
        section_index: int,
        concept_ids: list[str],
        through_message_id: Optional[int] = None,
        names: Optional[list[str]] = None,
    ) -> int:
        """Mark provisional mistake episodes resolved after section evaluation: only
        mistakes on the evaluated concepts (by id, or by Pedro's label against `names`),
        and only ones the evaluator actually saw (up to through_message_id)."""
        from ..grading import same_concept
        if not concept_ids:
            return 0
        cid_set = set(concept_ids)
        n = 0
        for ep in self.for_section(namespace, section_index):
            ss = ep.store_specific or {}
            if ss.get("episode_type") != "exercise_attempt":
                continue
            if ss.get("outcome") not in ("mistake", "struggle"):
                continue
            ep_cids = set(ss.get("concept_ids") or []) | set(ss.get("matched_concept_ids") or [])
            if not (ep_cids & cid_set) and not any(same_concept(ss.get("concept_label"), name) for name in names or ()):
                continue
            msg_ids = ss.get("chat_message_ids") or []
            if through_message_id is not None and (not msg_ids or max(msg_ids) > int(through_message_id)):
                continue  # after the evaluated transcript: the evaluator never saw it
            signals = dict(ss.get("signals") or {})
            if signals.get("resolved_by_evaluation"):
                continue
            signals["resolved_by_evaluation"] = True
            signals["provisional"] = True
            ss = dict(ss)
            ss["signals"] = signals
            ep.store_specific = ss
            self._insert(ep)
            n += 1
        return n

    def mark_tutor_error(self, namespace: str, section_index: int, label: Optional[str] = None,
                         concept_ids: Optional[list[str]] = None, within: int = 3) -> Optional[list[str]]:
        """Pedro corrected himself: flag a mistake in this section as his error, not the
        student's. A named correction looks at the section's last `within` graded
        attempts on that concept (by Pedro's label, or by concept id for older episodes),
        however many other concepts were graded in between; an unnamed one looks at the
        section's last `within` attempts. The latest unflagged mistake among them is
        flagged. The lesson record (pedro_context._reconciled) applies the same rule.
        Returns its concept ids, or None when there was nothing to flag."""
        from ..grading import same_concept
        wanted = set(concept_ids or ())

        def about(ep) -> bool:
            ss = ep.store_specific or {}
            ids = set(ss.get("concept_ids") or ()) | set(ss.get("matched_concept_ids") or ())
            return same_concept(label, ss.get("concept_label")) or bool(wanted & ids)

        attempts = self.by_types(namespace, ("exercise_attempt",), section_index=section_index)
        pool = [ep for ep in attempts if about(ep)] if label else attempts
        for ep in reversed(pool[-within:]):
            ss = dict(ep.store_specific or {})
            signals = dict(ss.get("signals") or {})
            if ss.get("outcome") not in ("mistake", "struggle") or signals.get("tutor_error"):
                continue
            signals.update(tutor_error=True, resolved_by_evaluation=True)
            ss["signals"] = signals
            ep.store_specific = ss
            self._insert(ep)
            return list(ss.get("concept_ids") or [])
        return None

    def valid_attempts(self, namespace: str, concept_id: str) -> list[dict]:
        """The evidence behind a concept's mastery, oldest first: every episode on it with
        an outcome that moves mastery, except ones flagged as Pedro's own error."""
        out = []
        for ep in self.for_concept(namespace, concept_id):
            ss = ep.store_specific or {}
            if ss.get("outcome") not in ("success", "struggle", "mistake") or (ss.get("signals") or {}).get("tutor_error"):
                continue
            out.append({"outcome": ss["outcome"], "hinted": bool(ss.get("hinted")), "at": ep.created_at,
                        "seq": ss.get("seq") or 0})
        return sorted(out, key=lambda a: (a["at"] or "", a["seq"]))

    def signal_counts(self, namespace: str, days: float = 30.0) -> dict[str, int]:
        """Aggregate behavioural signals over recent episodes."""
        out: dict[str, int] = {}
        for it in self.since(namespace, days):
            for k, v in ((it.store_specific or {}).get("signals") or {}).items():
                if v is True:
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
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        out_by = {"success": 0, "struggle": 0, "neutral": 0, "mistake": 0}
        with connect_db(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT json_extract(store_specific, '$.outcome'), COUNT(*), "
                f"SUM(COALESCE(json_extract(store_specific, '$.duration_sec'), 0)) FROM {self.table} "
                f"WHERE namespace = ? AND superseded_by IS NULL AND created_at >= ? GROUP BY 1",
                (namespace, cutoff),
            ).fetchall()
        for outcome, n, _ in rows:
            out_by[outcome or "neutral"] = out_by.get(outcome or "neutral", 0) + n
        return {
            "window_days": days,
            "n_episodes": sum(r[1] for r in rows),
            "outcomes": out_by,
            "time_on_task_sec": int(sum(r[2] or 0 for r in rows)),
        }

    def compact(
        self,
        namespace: str,
        *,
        before_days: float = 180.0,
        keep_types: Optional[tuple[str, ...]] = None,
    ) -> int:
        """Delete low-value episodes older than before_days.

        Preserves section completions, lesson milestones, conversation turns
        (qa — they index the transcript for multi-year recall) and any episode
        with a non-neutral outcome. Returns the number of rows deleted."""
        keep_types = keep_types or (
            "qa",
            "section_evaluation",
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
