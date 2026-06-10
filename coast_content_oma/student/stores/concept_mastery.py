"""ConceptMasteryStore — per-(student, course, concept_id) mastery record.

One row per concept the student has touched. The row is updated in place
as new evidence arrives. Schema (in store_specific):
  - concept_id           the Content OMA concept_id (canonical)
  - concept_name         cached for inspectability without a join
  - mastery_score        0.0 (no understanding) — 1.0 (mastered)
  - confidence           0.0 (1 sample) — 1.0 (>=10 samples)
  - successes            int
  - struggles            int
  - neutral_touches      int
  - first_seen           ISO timestamp
  - last_seen            ISO timestamp
  - last_strengthened    ISO timestamp (last success)
  - last_struggle        ISO timestamp (last failure)
  - related_lesson_ids   list of lesson ids that touched this concept
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

from ...stores._semantic_base import SemanticStoreBase, _row_to_item
from ...stores.base import MemoryItem, new_item_id, now_iso
from ...stores.db import connect_db
from ..mastery_tier import compute_mastery_tier, sync_mastery_tier


# Exponential moving average rate for mastery updates.
EMA_ALPHA = 0.30

# Confidence saturates after this many interactions.
CONFIDENCE_CAP_N = 10

# ── Time decay ──────────────────────────────────────────────────────
# Without fresh evidence, effective mastery decays exponentially toward
# a retention floor (you don't fully forget — but you're no longer 100%).
# Half-life is scaled by confidence: well-practiced concepts fade slower.
DECAY_HALF_LIFE_DAYS = float(os.environ.get("OMA_MASTERY_HALF_LIFE_DAYS", "45"))
# Fraction of the raw score retained as t → ∞.
RETENTION_FLOOR = float(os.environ.get("OMA_MASTERY_RETENTION_FLOOR", "0.30"))
# A concept whose raw score was solid but whose effective score dropped
# below this is "due for review" (spaced-repetition resurfacing).
REVIEW_THRESHOLD = float(os.environ.get("OMA_MASTERY_REVIEW_THRESHOLD", "0.55"))


def days_since_evidence(ss: dict, now: Optional[datetime] = None) -> Optional[float]:
    """Days since the last recorded evidence for this concept (any touch)."""
    last = ss.get("last_seen") or ss.get("first_seen")
    if not last:
        return None
    try:
        last_dt = datetime.fromisoformat(last)
    except (ValueError, TypeError):
        return None
    delta = (now or datetime.now()) - last_dt
    return max(0.0, delta.total_seconds() / 86400.0)


def effective_mastery(ss: dict, now: Optional[datetime] = None) -> float:
    """Raw mastery_score with exponential time-decay applied.

    decay = 0.5 ** (days / half_life); half_life grows with confidence
    (0.75x at conf=0 → 1.5x at conf=1). Score decays toward
    raw * RETENTION_FLOOR, never to zero.
    """
    raw = float(ss.get("mastery_score", 0.0) or 0.0)
    if raw <= 0.0:
        return 0.0
    days = days_since_evidence(ss, now)
    if days is None or days <= 0.0:
        return raw
    confidence = float(ss.get("confidence", 0.0) or 0.0)
    half_life = DECAY_HALF_LIFE_DAYS * (0.75 + 0.75 * confidence)
    decay = 0.5 ** (days / half_life)
    floor = raw * RETENTION_FLOOR
    return floor + (raw - floor) * decay


def _outcome_value(outcome: str) -> float:
    return {
        "success": 1.0,
        "struggle": 0.0,
        "neutral": 0.5,
    }.get(outcome, 0.5)


class ConceptMasteryStore(SemanticStoreBase):
    STORE_NAME = "concept_mastery"

    def _init_db(self) -> None:
        super()._init_db()
        with connect_db(self.db_path) as conn:
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table}_concept_id "
                f"ON {self.table}(namespace, (json_extract(store_specific, '$.concept_id')))"
            )

    # ── Write / update ────────────────────────────────────────────

    def record_evidence(
        self,
        namespace: str,
        concept_id: str,
        concept_name: str,
        outcome: str,  # success | struggle | neutral
        lesson_id: Optional[str] = None,
    ) -> MemoryItem:
        """Apply a single piece of evidence to this concept's mastery row.
        Creates the row on first touch, updates in place thereafter."""
        existing = self._find_by_concept(namespace, concept_id)
        ts = now_iso()
        outcome_val = _outcome_value(outcome)

        if existing is None:
            ss = {
                "concept_id": concept_id,
                "concept_name": concept_name,
                "mastery_score": outcome_val,  # first point IS the score
                "confidence": 1 / CONFIDENCE_CAP_N,
                "successes": 1 if outcome == "success" else 0,
                "struggles": 1 if outcome == "struggle" else 0,
                "neutral_touches": 1 if outcome == "neutral" else 0,
                "first_seen": ts,
                "last_seen": ts,
                "last_strengthened": ts if outcome == "success" else None,
                "last_struggle": ts if outcome == "struggle" else None,
                "related_lesson_ids": [lesson_id] if lesson_id else [],
            }
            sync_mastery_tier(ss)
            item = MemoryItem(
                id=new_item_id("mast"),
                namespace=namespace,
                store=self.STORE_NAME,
                content=self._summary_text(concept_name, ss),
                entities=[concept_id, concept_name],
                tags=[ss["mastery_tier"], self._mastery_tag(ss["mastery_score"])],
                importance=0.6,
                store_specific=ss,
            )
            self._insert(item)
            return item

        ss = dict(existing.store_specific or {})
        old_score = float(ss.get("mastery_score", 0.5))
        new_score = EMA_ALPHA * outcome_val + (1 - EMA_ALPHA) * old_score
        n = ss.get("successes", 0) + ss.get("struggles", 0) + ss.get("neutral_touches", 0) + 1

        ss["mastery_score"] = max(0.0, min(1.0, new_score))
        ss["confidence"] = min(1.0, n / CONFIDENCE_CAP_N)
        ss["successes"] = ss.get("successes", 0) + (1 if outcome == "success" else 0)
        ss["struggles"] = ss.get("struggles", 0) + (1 if outcome == "struggle" else 0)
        ss["neutral_touches"] = ss.get("neutral_touches", 0) + (1 if outcome == "neutral" else 0)
        ss["last_seen"] = ts
        if outcome == "success":
            ss["last_strengthened"] = ts
        elif outcome == "struggle":
            ss["last_struggle"] = ts
        if lesson_id and lesson_id not in (ss.get("related_lesson_ids") or []):
            ss.setdefault("related_lesson_ids", []).append(lesson_id)

        if outcome == "struggle":
            ss["last_misconception"] = True
        elif outcome == "success" and new_score >= 0.6:
            ss.pop("last_misconception", None)

        sync_mastery_tier(ss)
        existing.store_specific = ss
        existing.content = self._summary_text(concept_name, ss)
        existing.tags = [ss["mastery_tier"], self._mastery_tag(ss["mastery_score"])]
        existing.last_accessed = ts
        self._insert(existing)  # INSERT OR REPLACE
        return existing

    # ── Read ──────────────────────────────────────────────────────

    def for_concept(self, namespace: str, concept_id: str) -> Optional[MemoryItem]:
        return self._find_by_concept(namespace, concept_id)

    def weakest(self, namespace: str, k: int = 5, min_n: int = 2) -> list[MemoryItem]:
        """Return the k weakest concepts (lowest effective mastery), among
        those with at least min_n interactions so we don't flag a
        single-attempt flop as 'struggling'."""
        items = self.all(namespace)
        filtered = []
        for it in items:
            ss = it.store_specific or {}
            n = ss.get("successes", 0) + ss.get("struggles", 0) + ss.get("neutral_touches", 0)
            if n >= min_n:
                filtered.append((effective_mastery(ss), it))
        filtered.sort(key=lambda kv: kv[0])
        return [it for _, it in filtered[:k]]

    def strongest(self, namespace: str, k: int = 5, min_n: int = 2) -> list[MemoryItem]:
        items = self.all(namespace)
        filtered = []
        for it in items:
            ss = it.store_specific or {}
            n = ss.get("successes", 0) + ss.get("struggles", 0) + ss.get("neutral_touches", 0)
            if n >= min_n:
                filtered.append((effective_mastery(ss), it))
        filtered.sort(key=lambda kv: kv[0], reverse=True)
        return [it for _, it in filtered[:k]]

    def stale(self, namespace: str, days: float = 14.0) -> list[MemoryItem]:
        """Concepts not touched in the last N days — candidates for review."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        items = self.all(namespace)
        return [
            it for it in items
            if (it.store_specific or {}).get("last_seen", "") < cutoff
        ]

    def due_for_review(
        self,
        namespace: str,
        k: int = 5,
        min_raw: float = 0.6,
        now: Optional[datetime] = None,
    ) -> list[MemoryItem]:
        """Concepts the student once knew well (raw >= min_raw) whose
        effective mastery has decayed below REVIEW_THRESHOLD — the
        spaced-repetition resurfacing list, biggest drop first."""
        out: list[tuple[float, MemoryItem]] = []
        for it in self.all(namespace):
            ss = it.store_specific or {}
            raw = float(ss.get("mastery_score", 0.0) or 0.0)
            if raw < min_raw:
                continue
            eff = effective_mastery(ss, now)
            if eff >= REVIEW_THRESHOLD:
                continue
            out.append((raw - eff, it))
        out.sort(key=lambda kv: kv[0], reverse=True)
        return [it for _, it in out[:k]]

    def overview(self, namespace: str) -> dict:
        """Stats summary for inspection / teacher dashboard.

        avg/mastered/struggling use decayed (effective) scores; n_fading
        counts concepts that were solid but have decayed below review
        threshold."""
        items = self.all(namespace)
        if not items:
            return {"n_concepts": 0}
        now = datetime.now()
        effs = []
        n_fading = 0
        for it in items:
            ss = it.store_specific or {}
            raw = float(ss.get("mastery_score", 0.0) or 0.0)
            eff = effective_mastery(ss, now)
            effs.append(eff)
            if raw >= 0.6 and eff < REVIEW_THRESHOLD:
                n_fading += 1
        return {
            "n_concepts": len(items),
            "avg_mastery": sum(effs) / len(effs),
            "n_mastered": sum(1 for s in effs if s >= 0.75),
            "n_struggling": sum(1 for s in effs if s <= 0.35),
            "n_borderline": sum(1 for s in effs if 0.35 < s < 0.75),
            "n_fading": n_fading,
        }

    # ── Merge (concept dedup support) ─────────────────────────────

    def merge_concepts(
        self,
        namespace: str,
        src_concept_id: str,
        dst_concept_id: str,
        dst_concept_name: Optional[str] = None,
    ) -> bool:
        """Fold the mastery row for src_concept_id into dst_concept_id.

        Used when Content OMA merges two duplicate concept nodes so the
        student's evidence isn't fragmented. Combines counters, recomputes
        the score as an evidence-weighted average, keeps the earliest
        first_seen / latest last_seen."""
        src = self._find_by_concept(namespace, src_concept_id)
        if src is None:
            return False
        dst = self._find_by_concept(namespace, dst_concept_id)

        if dst is None:
            # No row for the canonical concept yet — relabel src in place.
            ss = dict(src.store_specific or {})
            ss["concept_id"] = dst_concept_id
            if dst_concept_name:
                ss["concept_name"] = dst_concept_name
            sync_mastery_tier(ss)
            src.store_specific = ss
            src.entities = [dst_concept_id, ss.get("concept_name") or dst_concept_id]
            src.content = self._summary_text(ss.get("concept_name") or dst_concept_id, ss)
            self._insert(src)
            return True

        sss = src.store_specific or {}
        dss = dict(dst.store_specific or {})
        n_src = sss.get("successes", 0) + sss.get("struggles", 0) + sss.get("neutral_touches", 0)
        n_dst = dss.get("successes", 0) + dss.get("struggles", 0) + dss.get("neutral_touches", 0)
        total = max(1, n_src + n_dst)

        src_score = float(sss.get("mastery_score", 0.0) or 0.0)
        dst_score = float(dss.get("mastery_score", 0.0) or 0.0)
        dss["mastery_score"] = max(0.0, min(1.0, (src_score * n_src + dst_score * n_dst) / total))
        dss["confidence"] = min(1.0, (n_src + n_dst) / CONFIDENCE_CAP_N)
        for key in ("successes", "struggles", "neutral_touches"):
            dss[key] = (dss.get(key, 0) or 0) + (sss.get(key, 0) or 0)

        def _min_ts(a, b):
            return min(x for x in (a, b) if x) if (a or b) else None

        def _max_ts(a, b):
            return max(x for x in (a, b) if x) if (a or b) else None

        dss["first_seen"] = _min_ts(sss.get("first_seen"), dss.get("first_seen"))
        dss["last_seen"] = _max_ts(sss.get("last_seen"), dss.get("last_seen"))
        dss["last_strengthened"] = _max_ts(sss.get("last_strengthened"), dss.get("last_strengthened"))
        dss["last_struggle"] = _max_ts(sss.get("last_struggle"), dss.get("last_struggle"))
        if sss.get("last_misconception"):
            dss["last_misconception"] = True

        lessons = list(dict.fromkeys(
            (dss.get("related_lesson_ids") or []) + (sss.get("related_lesson_ids") or [])
        ))
        dss["related_lesson_ids"] = lessons
        if dst_concept_name:
            dss["concept_name"] = dst_concept_name

        sync_mastery_tier(dss)
        name = dss.get("concept_name") or dst_concept_id
        dst.store_specific = dss
        dst.content = self._summary_text(name, dss)
        dst.tags = [dss["mastery_tier"], self._mastery_tag(dss["mastery_score"])]
        self._insert(dst)
        self.delete(src.id)
        return True

    # ── Internals ─────────────────────────────────────────────────

    def _find_by_concept(self, namespace: str, concept_id: str) -> Optional[MemoryItem]:
        with connect_db(self.db_path) as conn:
            row = conn.execute(
                f"SELECT * FROM {self.table} WHERE namespace = ? "
                f"AND superseded_by IS NULL "
                f"AND json_extract(store_specific, '$.concept_id') = ?",
                (namespace, concept_id),
            ).fetchone()
        return _row_to_item(row, self.STORE_NAME) if row else None

    def _summary_text(self, concept_name: str, ss: dict) -> str:
        score = ss.get("mastery_score", 0.0)
        succ = ss.get("successes", 0)
        strug = ss.get("struggles", 0)
        return (
            f"{concept_name}: mastery {score:.2f} "
            f"({succ} successes, {strug} struggles)"
        )

    def _mastery_tag(self, score: float) -> str:
        if score >= 0.75:
            return "mastered"
        if score <= 0.35:
            return "struggling"
        return "developing"
