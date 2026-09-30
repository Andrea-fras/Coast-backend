"""What has this student already learned, in OTHER courses, that the concepts
being taught now build on?

Concepts are matched by meaning when both sides have stored embeddings, and by
name otherwise. Only concepts the student has actual evidence on are used, so a
link is always something they really did — never an assumption.
"""
from __future__ import annotations

import re
import struct

from ..stores.db import connect_db
from .stores import course_namespace, list_course_namespaces, parse_course_namespace

SIMILARITY = 0.58


def _vectors(db_path, ids: list[str]) -> dict[str, list[float]]:
    if not ids:
        return {}
    with connect_db(db_path) as conn:
        rows = conn.execute(
            f"SELECT id, embedding FROM concept_items WHERE id IN ({','.join('?' * len(ids))}) AND embedding IS NOT NULL",
            ids,
        ).fetchall()
    return {i: list(struct.unpack(f"{len(b) // 4}f", b)) for i, b in rows if b}


def _cos(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _tokens(name: str) -> set[str]:
    return {t[:5] for t in re.findall(r"[a-z0-9]+", (name or "").lower()) if len(t) > 2}


def _name_match(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def related_prior_learning(orch, user_id, folder: str, current: list[dict], limit: int = 3) -> list[dict]:
    """current: [{concept_id, concept_name}] being taught now.
    Returns [{course, concept, state, when, relates_to, similarity}] best first."""
    if not current:
        return []
    db = orch.episodes.db_path
    here = course_namespace(user_id, folder)
    cur_vecs = _vectors(db, [c["concept_id"] for c in current if c.get("concept_id")])
    pairs = []
    for ns in list_course_namespaces(db, user_id):
        if ns == here:
            continue
        _, other = parse_course_namespace(ns)
        rows = [it.store_specific or {} for it in orch.mastery.all(ns)]
        rows = [r for r in rows if r.get("concept_id") and (r.get("successes") or r.get("last_eval_state"))]
        if not rows:
            continue
        prior_vecs = _vectors(db, [r["concept_id"] for r in rows])
        for r in rows:
            for c in current:
                a, b = prior_vecs.get(r["concept_id"]), cur_vecs.get(c.get("concept_id"))
                sim = _cos(a, b) if a and b else _name_match(r.get("concept_name"), c.get("concept_name")) * 0.8
                if sim >= SIMILARITY:
                    pairs.append((sim, other, r, c))
    pairs.sort(key=lambda p: -p[0])
    out, seen = [], set()
    for sim, other, r, c in pairs:
        key = (other, r.get("concept_name"))
        if key in seen:
            continue
        seen.add(key)
        state = r.get("last_eval_state") or ("solid" if float(r.get("mastery_score") or 0) >= 0.75 else "practised")
        out.append({"course": other, "concept": r.get("concept_name"), "state": state,
                    "when": (r.get("last_seen") or "")[:10], "relates_to": c.get("concept_name"),
                    "similarity": round(sim, 2)})
        if len(out) >= limit:
            break
    return out
