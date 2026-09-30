"""Four-tier mastery states for the lesson constellation map.

Red    — not attempted, or fundamental misconception
Orange — shallow / inconsistent understanding
Yellow — understood but not pressure-tested
Green  — demonstrated solid understanding (transfer/challenge in later phases)
"""

from __future__ import annotations

from typing import Optional

TIERS = ("red", "orange", "yellow", "green")


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

    # The section evaluator judged it mastered: green needs an answer of their own behind
    # that judgment, and their latest answer on it right. Help-only success stays yellow.
    independent = successes - int(ss.get("hinted_successes", 0) or 0)
    latest_right = (ss.get("last_strengthened") or "") >= (ss.get("last_struggle") or "")
    if ss.get("last_eval_state") == "mastered" and independent >= 1 and latest_right and score >= 0.6:
        return "green"
    # It judged them still shaky: at most orange, until an answer after that judgment says otherwise.
    if ss.get("last_eval_state") == "struggling" and (ss.get("last_strengthened") or "") <= (ss.get("last_eval_at") or ""):
        return "orange"

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
