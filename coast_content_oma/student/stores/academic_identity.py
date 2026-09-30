"""AcademicIdentityStore — cross-course traits about a student.

This is the ONLY store that lives in a single per-student namespace
(u{user_id}__identity) regardless of how many courses the student takes.
It captures traits that should transfer to a new course on day one:
  - learning style ("prefers visual/example-led")
  - cognitive strengths and weaknesses across topics
  - working habits (session length, time of day, week vs weekend)
  - meta-traits ("asks lots of follow-ups", "rarely revisits review material")

Populated by the IdentityConsolidator which periodically reads each of
the student's course-level PatternStores and rolls them up.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Optional

from ...stores._semantic_base import SemanticStoreBase
from ...stores.base import MemoryItem, new_item_id, now_iso


IdentityTraitType = str

# Traits fade when not re-confirmed by IdentityConsolidator.
TRAIT_HALF_LIFE_DAYS = float(os.environ.get("OMA_IDENTITY_HALF_LIFE_DAYS", "120"))
TRAIT_MIN_CONFIDENCE = float(os.environ.get("OMA_IDENTITY_MIN_CONFIDENCE", "0.25"))


def days_since_confirmed(ss: dict, now: Optional[datetime] = None) -> Optional[float]:
    last = ss.get("last_confirmed") or ss.get("first_observed")
    if not last:
        return None
    try:
        last_dt = datetime.fromisoformat(last)
    except (ValueError, TypeError):
        return None
    delta = (now or datetime.now()) - last_dt
    return max(0.0, delta.total_seconds() / 86400.0)


_RAW_ID_RE = re.compile(r"\b(?:con|ep|pat|mast|ctx|idn)_\d{8}_\d{6}_[0-9a-f]{6}\b")

# Derivations written by Pedro / onboarding — never auto-retired by the consolidator.
OBSERVED_DERIVATIONS = ("pedro_remember_tag", "Pedro onboarding conversation")


def is_displayable_trait(text: str) -> bool:
    """False for traits that leak internal ids or are empty — never show those to Pedro."""
    text = (text or "").strip()
    return bool(text) and not _RAW_ID_RE.search(text)


def effective_trait_confidence(ss: dict, now: Optional[datetime] = None) -> float:
    """Stored confidence with exponential decay since last confirmation."""
    raw = float(ss.get("confidence", 0.0) or 0.0)
    if raw <= 0.0:
        return 0.0
    days = days_since_confirmed(ss, now)
    if days is None or days <= 0.0:
        return raw
    decay = 0.5 ** (days / TRAIT_HALF_LIFE_DAYS)
    return raw * decay


CANONICAL_TRAIT_TYPES = (
    "learning_style",         # e.g. "visual-led", "example-led", "proof-first"
    "session_pattern",        # e.g. "20-min focused sessions", "evening learner"
    "engagement_pattern",     # e.g. "frequently asks follow-ups"
    "general_strength",       # e.g. "strong in implementation"
    "general_weakness",       # e.g. "weak in abstract proofs"
    "motivation_pattern",     # e.g. "deadline-driven", "consistent steady pace"
)


class AcademicIdentityStore(SemanticStoreBase):
    STORE_NAME = "academic_identity"

    def upsert_trait(
        self,
        identity_namespace: str,
        trait_type: IdentityTraitType,
        description: str,
        confidence: float,
        evidence_courses: list[str],
        derivation: str = "",
        dedupe_key: Optional[str] = None,
        evidence_quote: Optional[str] = None,
        observed_at: Optional[str] = None,
    ) -> MemoryItem:
        """Insert or update an identity trait. dedupe_key (optional) lets
        the consolidator say "this is the same trait as before" (e.g. by
        normalized topic name) — otherwise (trait_type, content) is the
        identity. observed_at is when the evidence behind a derived trait was
        last seen: recomputing an inference from old patterns is not fresh
        evidence, so the trait keeps decaying from that date."""
        existing = self._find_existing(identity_namespace, trait_type, dedupe_key, description)
        ts = now_iso()
        seen = observed_at or ts
        ss = {
            "trait_type": trait_type,
            "confidence": float(max(0.0, min(1.0, confidence))),
            "evidence_courses": list(evidence_courses),
            "first_observed": existing.store_specific["first_observed"] if existing else seen,
            "last_confirmed": seen,
            "derivation": derivation,
            "dedupe_key": dedupe_key,
        }
        # The student's own words behind the trait — keeps paraphrases honest.
        quote = evidence_quote or ((existing.store_specific or {}).get("evidence_quote") if existing else None)
        if quote:
            ss["evidence_quote"] = quote[:300]
        if existing:
            existing.store_specific = ss
            existing.content = description
            existing.tags = [trait_type]
            existing.importance = ss["confidence"]
            self._insert(existing)
            return existing
        item = MemoryItem(
            id=new_item_id("idn"),
            namespace=identity_namespace,
            store=self.STORE_NAME,
            content=description,
            tags=[trait_type],
            importance=ss["confidence"],
            store_specific=ss,
        )
        self._insert(item)
        return item

    def _find_existing(
        self,
        identity_namespace: str,
        trait_type: IdentityTraitType,
        dedupe_key: Optional[str],
        description: str,
    ) -> Optional[MemoryItem]:
        for it in self.all(identity_namespace):
            ss = it.store_specific or {}
            if ss.get("trait_type") != trait_type:
                continue
            if dedupe_key is not None and ss.get("dedupe_key") == dedupe_key:
                return it
            if dedupe_key is None and ss.get("dedupe_key") is None and it.content.strip() == description.strip():
                return it
        return None

    def all_traits(
        self,
        identity_namespace: str,
        min_confidence: float = TRAIT_MIN_CONFIDENCE,
    ) -> list[MemoryItem]:
        now = datetime.now()
        scored: list[tuple[float, MemoryItem]] = []
        for it in self.all(identity_namespace):
            if not is_displayable_trait(it.content):
                continue
            ss = it.store_specific or {}
            eff = effective_trait_confidence(ss, now)
            if eff >= min_confidence:
                scored.append((eff, it))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        return [it for _, it in scored]

    def retire_derived(self, identity_namespace: str, trait_types: tuple, keep_keys: set) -> int:
        """Remove consolidator-derived traits of these types that are no longer
        supported, plus any malformed trait. Observed traits (Pedro tags,
        onboarding) are kept."""
        n = 0
        for it in self.all(identity_namespace):
            ss = it.store_specific or {}
            malformed = not is_displayable_trait(it.content)
            derived = ss.get("derivation") not in OBSERVED_DERIVATIONS
            stale = ss.get("trait_type") in trait_types and derived and ss.get("dedupe_key") not in keep_keys
            if malformed or stale:
                self.delete(it.id)
                n += 1
        return n
