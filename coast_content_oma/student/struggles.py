"""Which concepts is the student *currently* struggling with?

Judged from the ordered log of graded answers, weighted to the most recent ones,
so a student who recovers stops being labelled — and one who is stuck now is
flagged even if they did well long ago.
"""
from __future__ import annotations

from typing import Callable, Optional

RECENT_WINDOW = 5


def is_struggling(outcomes: list[bool], legacy_successes: int = 0) -> bool:
    """outcomes: graded answers oldest → newest (True = correct).

    Struggling = at least two wrong among the last RECENT_WINDOW answers and
    wrong answers are at least half of them. legacy_successes are successes
    recorded before answers were logged in order; they can only clear a flag.
    """
    recent = outcomes[-RECENT_WINDOW:]
    wrong = recent.count(False)
    if wrong < 2 or wrong / len(recent) < 0.5:
        return False
    if legacy_successes:
        total_wrong = outcomes.count(False)
        if total_wrong / (len(outcomes) + legacy_successes) <= 0.5:
            return False
    return True


def graded_history(episodes, namespace: str, days: Optional[float] = None) -> dict[str, list[bool]]:
    """{concept_id: [correct?, ...]} from exercise_attempt episodes, oldest first.
    Mistakes the section evaluator later marked resolved are left out."""
    out: dict[str, list[bool]] = {}
    for ep in episodes.by_types(namespace, ("exercise_attempt",), days=days):
        ss = ep.store_specific or {}
        outcome = ss.get("outcome")
        if outcome not in ("success", "mistake", "struggle"):
            continue
        if outcome != "success" and (ss.get("signals") or {}).get("resolved_by_evaluation"):
            continue
        for cid in ss.get("concept_ids") or []:
            if cid:
                out.setdefault(cid, []).append(outcome == "success")
    return out


def struggling_concepts(episodes, mastery, namespace: str, *, days: Optional[float] = None,
                        name_for: Optional[Callable[[str], str]] = None) -> list[dict]:
    rows = {(it.store_specific or {}).get("concept_id"): it.store_specific or {} for it in mastery.all(namespace)}
    out = []
    for cid, outcomes in graded_history(episodes, namespace, days=days).items():
        row = rows.get(cid) or {}
        legacy = max(0, int(row.get("successes", 0) or 0) - outcomes.count(True))
        if not is_struggling(outcomes, legacy_successes=legacy):
            continue
        name = row.get("concept_name") or (name_for(cid) if name_for else None) or cid
        out.append({"concept_id": cid, "name": name, "mistakes": outcomes.count(False),
                    "successes": outcomes.count(True) + legacy})
    out.sort(key=lambda t: (-t["mistakes"], t["name"] or ""))
    return out
