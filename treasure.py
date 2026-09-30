"""Map treasure chests — Content OMA concept challenge with typed-answer verification."""

from __future__ import annotations

import provider_capacity

import hashlib
import json
import os
import random
import re
from pathlib import Path
from sqlalchemy import text
from sqlalchemy.dialects.sqlite import insert
from datetime import datetime, timezone

from database import CourseOutline, SessionLocal, TreasureChestOpen, TreasureChallenge, UserMapState, SectionVerification, SectionRewardClaim

# Exported from the same world generators the browser renders; ids carry the
# world ("lumen:x,y", "neon:x,y") so they stay unique across levels.
def _load_chests() -> list[dict]:
    import map_world
    return [{**c, 'level': no} for no, lv in sorted(map_world.LEVELS.items()) for c in lv.chests]


TREASURE_CHESTS = _load_chests()

XP_REWARD = 40
MIN_SCORE_TO_PASS = 50


GRADING_PROMPT = """You are a university tutor grading a short recall answer.

QUESTION: {question_text}

MODEL ANSWER (from course material):
{model_answer}

KEY CONCEPT: {concept_name}

STUDENT ANSWER:
{student_answer}

INSTRUCTIONS:
- Score 0–100 based on whether the student captures the core idea of the concept.
- Accept paraphrasing and different wording if the meaning is right.
- A concise correct answer deserves a high score — do not penalise brevity.
- Partial understanding: 40–69. Clear correct understanding: 70+.

Respond with ONLY valid JSON (no markdown fences):
{{
  "score": <0-100>,
  "is_correct": <true if score >= {pass_threshold} else false>,
  "feedback": "<1-2 sentence constructive feedback>"
}}"""


def _chest_by_id(chest_id: str) -> dict | None:
    for c in TREASURE_CHESTS:
        if c["id"] == chest_id:
            return c
    return None


def _opened_ids(user_id: int) -> set[str]:
    db = SessionLocal()
    try:
        rows = db.query(TreasureChestOpen).filter(
            TreasureChestOpen.user_id == user_id,
        ).all()
        return {r.chest_id for r in rows}
    finally:
        db.close()


def get_treasure_state(user_id: int) -> dict:
    opened = _opened_ids(user_id)
    return {
        "chests": [
            {**c, "opened": c["id"] in opened}
            for c in TREASURE_CHESTS
        ],
        "opened_ids": sorted(opened),
    }


def _quiz_rng(user_id: int, chest_id: str) -> random.Random:
    seed = int(hashlib.sha256(f"{user_id}:{chest_id}".encode()).hexdigest()[:12], 16)
    return random.Random(seed)


def _concept_pool(user_id: int) -> list[dict]:
    """Random concept candidates from Content OMA across completed lesson sections."""
    from curated_config import curated_source_uid as _curated_uid
    import lesson
    import oma_provider

    if not oma_provider.is_oma_enabled():
        return []

    db = SessionLocal()
    try:
        outlines = db.query(CourseOutline).filter(
            CourseOutline.user_id == user_id,
        ).all()
        pool: list[dict] = []
        seen: set[str] = set()
        for outline in outlines:
            folder = outline.folder_name
            src_uid = _curated_uid(folder) if _curated_uid(folder) is not None else user_id
            sections = json.loads(outline.outline_json or "[]")
            cs = int(outline.current_section or 0)
            completed = {i for i in range(min(cs, len(sections)))}
            completed.update(r.section_index for r in db.query(SectionVerification).filter_by(user_id=user_id, folder_name=folder, is_active=True))
            completed.update(r.section_index for r in db.query(SectionRewardClaim).filter_by(user_id=user_id, folder_name=folder))
            for i in sorted(completed):
                if not 0 <= i < len(sections):
                    continue
                sec = sections[i]
                sec_title = sec.get("title") or f"Section {i + 1}"
                refs = lesson.get_section_concept_refs(
                    user_id, folder, i, source_user_id=src_uid,
                )
                for ref in refs:
                    dedupe = f"{folder}:{ref['concept_id']}"
                    if dedupe in seen:
                        continue
                    seen.add(dedupe)
                    pool.append({
                        **ref,
                        "folder": folder,
                        "section_index": i,
                        "section": sec_title,
                        "source_user_id": src_uid,
                    })
        return pool
    finally:
        db.close()


def _concept_model_answer(src_uid: int, folder: str, concept_id: str, concept_name: str) -> str | None:
    import oma_provider
    from coast_content_oma.stores import make_namespace

    ns = make_namespace(src_uid, folder)
    orch = oma_provider._content_orchestrator()
    parts: list[str] = []

    concept = orch.concept.get(concept_id)
    if concept and concept.namespace == ns:
        ss = concept.store_specific or {}
        definition = (ss.get("definition") or concept.content or "").strip()
        if definition:
            parts.append(definition)

    for item in orch.content.find_definitions_of(ns, concept_id, max_results=2):
        text = (item.content or "").strip()
        if text and text not in parts:
            parts.append(text)

    block, _ = oma_provider.get_folder_context(
        src_uid,
        folder,
        f"define {concept_name}",
        max_chars=3500,
        max_content=3,
        max_images=0,
    )
    if block and block.strip():
        parts.append(block.strip())

    merged = "\n\n".join(parts).strip()
    if len(merged) < 20:
        return None
    return merged[:2400]


def _build_challenge(user_id: int, chest_id: str) -> dict:
    chest = _chest_by_id(chest_id)
    if not chest:
        return {"error": "Unknown treasure chest"}

    if chest_id in _opened_ids(user_id):
        return {
            "error": "already_opened",
            "chest_id": chest_id,
            "chest_name": chest["name"],
            "already_opened": True,
        }

    pool = _concept_pool(user_id)
    if not pool:
        return {
            "error": "not_enough_topics",
            "message": "Complete more lesson sections with uploaded course material before opening treasure chests.",
            "available": 0,
            "required": 1,
        }

    rng = _quiz_rng(user_id, chest_id)
    picked = rng.choice(pool)
    model_answer = _concept_model_answer(
        picked["source_user_id"],
        picked["folder"],
        picked["concept_id"],
        picked["concept_name"],
    )
    if not model_answer:
        return {
            "error": "not_enough_topics",
            "message": "No Content OMA material found for your completed lessons yet.",
            "available": len(pool),
            "required": 1,
        }

    concept_name = picked["concept_name"]
    question = f'Explain "{concept_name}" in your own words.'
    challenge_id = f"{picked['folder']}:{picked['concept_id']}"

    return {
        "chest_id": chest_id,
        "chest_name": chest["name"],
        "challenge_id": challenge_id,
        "concept_id": picked["concept_id"],
        "concept_name": concept_name,
        "folder": picked["folder"],
        "section": picked["section"],
        "question": question,
        "model_answer": model_answer,
    }


def _grade_answer(question: str, model_answer: str, student_answer: str, concept_name: str) -> dict:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return {
            "score": 0,
            "is_correct": False,
            "feedback": "Answer grading is not configured on the server.",
            "error": "grading_unavailable",
        }

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    prompt = GRADING_PROMPT.format(
        question_text=question,
        model_answer=model_answer[:1800],
        concept_name=concept_name,
        student_answer=student_answer[:2000],
        pass_threshold=MIN_SCORE_TO_PASS,
    )

    try:
        response = provider_capacity.call('openai', lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.2,
        ), priority='interactive')
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        result = json.loads(raw)
        score = min(max(int(result.get("score", 0)), 0), 100)
        is_correct = score >= MIN_SCORE_TO_PASS
        return {
            "score": score,
            "is_correct": is_correct,
            "feedback": str(result.get("feedback") or "").strip(),
        }
    except Exception:
        return {
            "score": 0,
            "is_correct": False,
            "feedback": "Could not grade your answer — please try again.",
            "error": "grading_unavailable",
        }


def build_quiz(user_id: int, chest_id: str) -> dict:
    chest = _chest_by_id(chest_id)
    if not chest:
        return {'error': 'Unknown treasure chest'}
    if chest_id in _opened_ids(user_id):
        return {'error': 'already_opened', 'already_opened': True}
    import map_world
    if not map_world.is_charted(user_id, int(chest.get('level') or 1), chest['x'], chest['y']):
        return {'error': 'Explore this tile before opening its treasure'}
    with SessionLocal() as db:
        row = db.get(TreasureChallenge, (user_id, chest_id))
        challenge = json.loads(row.challenge_json) if row else None
    if challenge is None:
        challenge = _build_challenge(user_id, chest_id)
        if challenge.get('error'):
            return challenge
        with SessionLocal() as db:
            db.execute(insert(TreasureChallenge).values(user_id=user_id, chest_id=chest_id,
                challenge_json=json.dumps(challenge)).on_conflict_do_nothing())
            db.commit()
            challenge = json.loads(db.get(TreasureChallenge, (user_id, chest_id)).challenge_json)
    return {key: challenge[key] for key in ('chest_id', 'chest_name', 'challenge_id',
        'concept_name', 'folder', 'section', 'question')} | {'xp_reward': XP_REWARD}


def complete_treasure(user_id: int, chest_id: str, answer: str) -> dict:
    """Grade the issued question and consume a valid attempt once, whether correct or not."""
    chest = _chest_by_id(chest_id)
    if not chest:
        return {"error": "Unknown treasure chest"}

    if chest_id in _opened_ids(user_id):
        return {"error": "already_opened", "already_opened": True}

    student_answer = re.sub(r"\s+", " ", str(answer or "").strip())
    if len(student_answer) < 8:
        return {
            "ok": False,
            "message": "Write a short answer (at least a sentence or two).",
        }

    with SessionLocal() as db:
        row = db.get(TreasureChallenge, (user_id, chest_id))
        if row is None:
            return {'error': 'Open the chest question before submitting an answer'}
        challenge = json.loads(row.challenge_json)

    grade = _grade_answer(
        challenge["question"],
        challenge["model_answer"],
        student_answer,
        challenge["concept_name"],
    )

    if grade.get('error'):
        return {'error': grade['error'], 'message': grade['feedback']}
    xp = XP_REWARD if grade['is_correct'] else 0

    db = SessionLocal()
    try:
        db.execute(text('BEGIN IMMEDIATE'))
        existing = db.query(TreasureChestOpen).filter(
            TreasureChestOpen.user_id == user_id,
            TreasureChestOpen.chest_id == chest_id,
        ).first()
        if existing:
            return {"error": "already_opened", "already_opened": True}

        row = db.query(UserMapState).filter(UserMapState.user_id == user_id).first()
        if not row:
            row = UserMapState(user_id=user_id)
            db.add(row)
            db.flush()

        row.total_xp = int(row.total_xp or 0) + xp
        db.add(TreasureChestOpen(
            user_id=user_id,
            chest_id=chest_id,
            xp_gained=xp,
            correct_count=int(grade['is_correct']),
            opened_at=datetime.now(timezone.utc),
        ))
        db.commit()
        total_xp = int(row.total_xp)
    finally:
        db.close()

    import map_world
    state = map_world.get_map_state(user_id)
    return {
        "ok": True,
        "consumed": True,
        "correct": int(grade['is_correct']),
        "total_cards": 1,
        "score": grade["score"],
        "feedback": grade["feedback"],
        "xp_gained": xp,
        "total_xp": total_xp,
        "chest_id": chest_id,
        "chest_name": chest["name"],
        "concept_name": challenge["concept_name"],
        "opened": True,
        "level": state.get("level"),
        "xp": state.get("xp"),
        "xp_max": state.get("xp_max"),
    }
