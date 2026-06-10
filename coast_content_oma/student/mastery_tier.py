"""Four-tier mastery states for the lesson constellation map.

Red    — not attempted, or fundamental misconception
Orange — shallow / inconsistent understanding
Yellow — understood but not pressure-tested
Green  — demonstrated solid understanding (transfer/challenge in later phases)
"""

from __future__ import annotations

from typing import Optional

TIERS = ("red", "orange", "yellow", "green")


def is_topic_struggling(successes: int, mistakes: int) -> bool:
    """Persistent difficulty — not a one-off wrong answer Pedro re-taught.

    Struggling when the student got the same topic wrong multiple times,
    or wrong/(wrong+right) exceeds 50%.
    """
    if mistakes >= 2:
        return True
    total = successes + mistakes
    if total < 2:
        return False
    return mistakes / total > 0.5


def compute_mastery_tier(store_specific: Optional[dict]) -> str:
    if not store_specific:
        return "red"

    ss = store_specific
    if ss.get("transfer_passed"):
        return "green"

    score = float(ss.get("mastery_score", 0.0))
    struggles = int(ss.get("struggles", 0) or 0)
    successes = int(ss.get("successes", 0) or 0)
    neutral = int(ss.get("neutral_touches", 0) or 0)
    touches = successes + struggles + neutral

    if touches == 0:
        return "red"

    if ss.get("last_misconception") or (struggles >= 2 and score <= 0.35):
        return "red"

    if score >= 0.75 and successes > struggles and struggles == 0:
        return "green"

    if score >= 0.5 and struggles <= successes:
        return "yellow"

    if score >= 0.45 and struggles == 0 and successes >= 1:
        return "yellow"

    return "orange"


def sync_mastery_tier(store_specific: dict) -> dict:
    store_specific["mastery_tier"] = compute_mastery_tier(store_specific)
    return store_specific


def edge_link_state(source_tier: str) -> str:
    """Prerequisite link is solid only when the source concept is yellow or green."""
    if source_tier in ("yellow", "green"):
        return "solid"
    return "broken"
