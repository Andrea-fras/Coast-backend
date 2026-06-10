"""Derive narrative accomplishments from the episode log.

Mastery scores are a secondary signal. What Pedro and the teacher
dashboard need first is: what sections/topics did the student work
through, and did they demonstrate competence there?

This module reads the immutable EpisodeStore and produces human-readable
accomplishment lines — no LLM required. Topic grouping is derived from
actual concept practice in the course, not hardcoded demo buckets.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Optional

from .stores.episode import EpisodeStore

if TYPE_CHECKING:
    from .stores.concept_mastery import ConceptMasteryStore


def summarize_accomplishments(
    episodes: EpisodeStore,
    namespace: str,
    *,
    mastery: Optional["ConceptMasteryStore"] = None,
    window_days: float = 60.0,
) -> dict:
    """Return structured + narrative summary of what the student has done."""
    concept_stats: dict[str, dict] = defaultdict(
        lambda: {"name": "", "success": 0, "struggle": 0, "neutral": 0}
    )
    sections_completed: list[str] = []
    perfect_rounds: list[str] = []
    lines: list[str] = []

    for ep in episodes.since(namespace, window_days):
        ss = ep.store_specific or {}
        etype = ss.get("episode_type", "")

        if etype == "section_completed":
            title = (ss.get("section_title") or ep.content or "").strip()
            if title:
                sections_completed.append(title)
            continue

        outcome = ss.get("outcome", "neutral")
        if outcome not in ("success", "struggle", "neutral", "mistake"):
            outcome = "neutral"
        if outcome == "mistake":
            outcome = "struggle"

        cids = ss.get("concept_ids") or ep.entities or []
        if not cids:
            continue

        for cid in cids:
            stats = concept_stats[cid]
            if not stats["name"]:
                if mastery is not None:
                    row = mastery.for_concept(namespace, cid)
                    if row:
                        stats["name"] = (row.store_specific or {}).get("concept_name") or cid
                if not stats["name"]:
                    stats["name"] = cid
            stats[outcome if outcome in stats else "neutral"] += 1

        if etype == "exercise_attempt" and outcome == "success" and len(cids) == 1:
            name = concept_stats[cids[0]]["name"] or cids[0]
            if name not in perfect_rounds:
                perfect_rounds.append(name)

    for _cid, stats in sorted(
        concept_stats.items(),
        key=lambda kv: kv[1]["success"],
        reverse=True,
    ):
        name = stats["name"] or _cid
        succ = stats["success"]
        strug = stats["struggle"]
        if succ >= 3 and strug == 0:
            lines.append(f"Demonstrated strong competence in {name} ({succ} correct practice)")
        elif succ >= 2 and strug <= 1:
            lines.append(f"Practiced {name} successfully ({succ} correct)")
        elif succ >= 1 and strug == 0:
            lines.append(f"Worked through {name} successfully")

    for title in sections_completed[-6:]:
        lines.append(f"Completed lesson section: {title}")

    bucket_stats = {
        stats["name"] or cid: {
            "success": stats["success"],
            "struggle": stats["struggle"],
            "neutral": stats["neutral"],
        }
        for cid, stats in concept_stats.items()
        if stats["success"] or stats["struggle"]
    }

    return {
        "bucket_stats": bucket_stats,
        "sections_completed": sections_completed,
        "perfect_rounds": perfect_rounds,
        "narrative_lines": lines,
    }
