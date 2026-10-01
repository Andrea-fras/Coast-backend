"""What has this student already learned, in OTHER courses, that the concepts
being taught now build on?

Concepts are matched by meaning when both sides have stored embeddings, and by
name otherwise. Only concepts the student has actual evidence on are used, so a
link is always something they really did — never an assumption.

Meaning alone over-links: concepts from one field sit close together, so "dense
graph" scores 0.70 against "rdf graph" while "directed graph" and "directed
edge-labelled graph" score 0.65. A link therefore also needs a specific shared
word (not "graph" or "network"), unless the meanings are near-identical; and when
the section's topics are given, only concepts the section actually teaches count.
"""
from __future__ import annotations

import re
import struct

from ..stores.db import connect_db
from .stores import course_namespace, list_course_namespaces, parse_course_namespace

SIMILARITY = 0.65  # cosine of the concepts' stored embeddings
SAME_IDEA = 0.9    # near-identical meaning needs no shared word


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


# Words a whole field shares; sharing only these is not the same idea.
GENERIC = {"graph", "netwo", "theor", "repre", "data", "model", "syste", "conce", "defin", "basic", "gener",
           "forma", "notat", "struc", "type", "types", "analy", "metho", "intro", "overv", "real", "world",
           "compl", "simpl", "and", "the", "for", "with"}


def _specific(a: str, b: str) -> set[str]:
    return (_tokens(a) & _tokens(b)) - GENERIC


def same_idea(prior: str, current: str, similarity: float) -> bool:
    """Is the concept they learned elsewhere the one being taught now?"""
    return similarity >= SAME_IDEA or (similarity >= SIMILARITY and bool(_specific(prior, current)))


def _name_match(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def related_prior_learning(orch, user_id, folder: str, current: list[dict], limit: int = 3,
                           topics: str = "") -> list[dict]:
    """current: [{concept_id, concept_name}] being taught now; topics: the section's title and
    key topics, when known, to keep only the concepts it actually teaches.
    Returns [{course, concept, state, when, relates_to, similarity, own, hinted, wrong}] best
    first, at most one link per concept on either side."""
    if topics:
        wanted = _tokens(topics) - GENERIC
        current = [c for c in current if (_tokens(c.get("concept_name")) - GENERIC) & wanted]
    if not current:
        return []
    import numpy as np
    db = orch.episodes.db_path
    here = course_namespace(user_id, folder)

    def unit(vecs):
        return {k: (lambda a: a / (np.linalg.norm(a) or 1.0))(np.asarray(v, dtype=np.float32)) for k, v in vecs.items()}

    cur_vecs = unit(_vectors(db, [c["concept_id"] for c in current if c.get("concept_id")]))
    pairs = []
    for ns in list_course_namespaces(db, user_id):
        if ns == here:
            continue
        _, other = parse_course_namespace(ns)
        rows = [it.store_specific or {} for it in orch.mastery.all(ns)]
        # Mistakes are evidence too: "they got this wrong in network science" is worth knowing.
        rows = [r for r in rows if r.get("concept_id")
                and (r.get("successes") or r.get("struggles") or r.get("last_eval_state"))]
        if not rows:
            continue
        prior_vecs = unit(_vectors(db, [r["concept_id"] for r in rows]))
        for r in rows:
            for c in current:
                a, b = prior_vecs.get(r["concept_id"]), cur_vecs.get(c.get("concept_id"))
                sim = float(a @ b) if a is not None and b is not None else _name_match(r.get("concept_name"), c.get("concept_name")) * 0.8
                if same_idea(r.get("concept_name") or "", c.get("concept_name") or "", sim):
                    pairs.append((sim, other, r, c))
    pairs.sort(key=lambda p: -p[0])
    out, used_prior, used_current = [], set(), set()
    for sim, other, r, c in pairs:
        prior = (other, re.sub(r"\s*\(.*\)", "", r.get("concept_name") or ""))  # "path" and "path (graph theory)"
        if prior in used_prior or c.get("concept_name") in used_current:
            continue
        used_prior.add(prior)
        used_current.add(c.get("concept_name"))
        hinted = int(r.get("hinted_successes") or 0)
        state = r.get("last_eval_state") or ("solid" if float(r.get("mastery_score") or 0) >= 0.75 else "practised")
        out.append({"course": other, "concept": r.get("concept_name"), "state": state,
                    "when": (r.get("last_seen") or "")[:10], "relates_to": c.get("concept_name"),
                    "similarity": round(sim, 2), "own": max(0, int(r.get("successes") or 0) - hinted),
                    "hinted": hinted, "wrong": int(r.get("struggles") or 0)})
        if len(out) >= limit:
            break
    return out
