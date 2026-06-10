"""Remap Content OMA concept ids after recanonicalization or dedup.

When concept nodes are replaced (new ids, same canonical names), student
evidence must follow or mastery fragments and Pedro loses history.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..stores.base import MemoryItem

if TYPE_CHECKING:
    from .orchestrator import StudentOrchestrator


def build_concept_id_map(
    old_concepts: list[MemoryItem],
    name_to_id: dict[str, str],
) -> dict[str, str]:
    """Map old concept ids → new ids by matching canonical name or alias."""
    id_map: dict[str, str] = {}
    for it in old_concepts:
        ss = it.store_specific or {}
        new_id: Optional[str] = None
        for n in [ss.get("name")] + (ss.get("aliases") or []):
            key = (n or "").lower().strip()
            if key and key in name_to_id:
                new_id = name_to_id[key]
                break
        if new_id and new_id != it.id:
            id_map[it.id] = new_id
    return id_map


def remap_student_concept_ids(
    orch: "StudentOrchestrator",
    course_namespace: str,
    id_map: dict[str, str],
) -> dict[str, int]:
    """Rewrite student-store references from old concept ids to new ones."""
    if not id_map:
        return {"mastery": 0, "episodes": 0, "patterns": 0, "active": 0}

    stats = {"mastery": 0, "episodes": 0, "patterns": 0, "active": 0}

    # Mastery rows: merge when both old and new exist, else relabel.
    for old_id, new_id in id_map.items():
        src = orch.mastery.for_concept(course_namespace, old_id)
        if src is None:
            continue
        name = (src.store_specific or {}).get("concept_name")
        if orch.mastery.merge_concepts(course_namespace, old_id, new_id, dst_concept_name=name):
            stats["mastery"] += 1

    for ep in orch.episodes.all(course_namespace):
        ss = ep.store_specific or {}
        cids = ss.get("concept_ids") or []
        if not any(c in id_map for c in cids):
            continue
        ss["concept_ids"] = list(dict.fromkeys(id_map.get(c, c) for c in cids))
        ep.store_specific = ss
        ep.entities = ss["concept_ids"]
        orch.episodes._insert(ep)
        stats["episodes"] += 1

    for pat in orch.patterns.all(course_namespace):
        ss = pat.store_specific or {}
        rel = ss.get("related_concept_ids") or []
        if not any(c in id_map for c in rel):
            continue
        ss["related_concept_ids"] = list(dict.fromkeys(id_map.get(c, c) for c in rel))
        pat.store_specific = ss
        if pat.entities:
            pat.entities = list(dict.fromkeys(id_map.get(e, e) for e in pat.entities))
        orch.patterns._insert(pat)
        stats["patterns"] += 1

    for frag in orch.active.all(course_namespace):
        if not frag.entities or not any(e in id_map for e in frag.entities):
            continue
        frag.entities = list(dict.fromkeys(id_map.get(e, e) for e in frag.entities))
        orch.active._insert(frag)
        stats["active"] += 1

    return stats
