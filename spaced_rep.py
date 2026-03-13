"""Spaced Repetition Engine — SM-2 algorithm, concept extraction, and helpers."""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any

from dotenv import load_dotenv
from google import genai
from openai import OpenAI

load_dotenv()

from database import ReviewCard, ReviewHistory, SessionLocal, SavedNotebook

# ---------------------------------------------------------------------------
# SM-2 Algorithm
# ---------------------------------------------------------------------------

def sm2_update(card: ReviewCard, quality: int) -> ReviewCard:
    """Apply the SM-2 algorithm to a review card.

    quality: 0-5 scale
        0 = complete blackout
        1 = wrong, but recognised after seeing answer
        2 = wrong, but answer felt familiar
        3 = correct with serious difficulty
        4 = correct with some hesitation
        5 = perfect recall
    """
    quality = max(0, min(5, quality))

    card.ease_factor = max(
        1.3,
        card.ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)),
    )

    if quality < 3:
        card.repetitions = 0
        card.interval = 1.0
    else:
        if card.repetitions == 0:
            card.interval = 1.0
        elif card.repetitions == 1:
            card.interval = 6.0
        else:
            card.interval = round(card.interval * card.ease_factor, 1)
        card.repetitions += 1

    now = datetime.now(timezone.utc)
    card.last_review = now
    card.next_review = now + timedelta(days=card.interval)
    return card


# ---------------------------------------------------------------------------
# Concept Extraction via LLM
# ---------------------------------------------------------------------------

CONCEPT_EXTRACTION_PROMPT = """You are an educational concept extractor. Given notebook content, extract the key reviewable concepts.

For each section, extract 3-6 concepts that a student should be able to recall and explain. Each concept should be:
- A specific term, principle, or technique (not too broad)
- Something worth testing via active recall
- Accompanied by a concise 1-sentence summary the student should know

Return a JSON array of objects with these fields:
- "section_title": the section this concept belongs to
- "concept": the term or principle name (short, 2-8 words)
- "summary": a one-sentence explanation (what the student should know)

Example output:
[
  {"section_title": "Derivatives (The Rate Detective)", "concept": "Power Rule", "summary": "For x^n, the derivative is n*x^(n-1) — drop the exponent as a multiplier and reduce by 1."},
  {"section_title": "Derivatives (The Rate Detective)", "concept": "Marginal Cost", "summary": "The derivative of total cost gives the cost of producing one additional unit."}
]

Return ONLY the JSON array, no markdown fences or extra text."""


def extract_concepts(notebook_json: dict) -> list[dict]:
    """Use an LLM to extract reviewable concepts from notebook content."""
    sections = notebook_json.get("sections") or []
    if not sections:
        return []

    content_parts = []
    for sec in sections:
        title = sec.get("title", "Untitled")
        tags = sec.get("tags", [])
        body = sec.get("content", "")
        subs = sec.get("subsections") or []

        sub_text = ""
        for sub in subs:
            sub_text += f"\n  - {sub.get('title', '')}: {sub.get('content', '')}"
            for b in (sub.get("bullets") or []):
                sub_text += f"\n    * {b}"

        content_parts.append(
            f"## {title}\nTags: {', '.join(tags)}\n{body}{sub_text}"
        )

    notebook_text = "\n\n".join(content_parts)
    if len(notebook_text) > 8000:
        notebook_text = notebook_text[:8000] + "\n...[truncated]"

    messages = [
        {"role": "system", "content": CONCEPT_EXTRACTION_PROMPT},
        {"role": "user", "content": f"Extract concepts from this notebook:\n\n{notebook_text}"},
    ]

    try:
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if gemini_key:
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model="gemini-3.1-pro-preview",
                contents=notebook_text + "\n\nExtract key reviewable concepts as JSON.",
                config={
                    "system_instruction": CONCEPT_EXTRACTION_PROMPT,
                    "max_output_tokens": 2048,
                    "temperature": 0.3,
                },
            )
            text = ""
            if response.candidates:
                for part in (response.candidates[0].content.parts or []):
                    if hasattr(part, "text") and part.text:
                        text += part.text
        else:
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
            resp = openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                messages=messages,
                max_tokens=2048,
                temperature=0.3,
            )
            text = resp.choices[0].message.content or ""

        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        concepts = json.loads(cleaned)
        if not isinstance(concepts, list):
            return []
        return concepts

    except Exception:
        traceback.print_exc()
        return _fallback_extract(sections)


def _fallback_extract(sections: list[dict]) -> list[dict]:
    """Simple keyword-based fallback if the LLM call fails."""
    concepts = []
    for sec in sections:
        title = sec.get("title", "Untitled")
        for sub in (sec.get("subsections") or []):
            sub_title = sub.get("title", "")
            sub_content = sub.get("content", "")
            if sub_title:
                concepts.append({
                    "section_title": title,
                    "concept": sub_title,
                    "summary": (sub_content[:120] + "...") if len(sub_content) > 120 else sub_content,
                })
        for tag in (sec.get("tags") or [])[:2]:
            concepts.append({
                "section_title": title,
                "concept": tag.title(),
                "summary": f"Key concept from {title}.",
            })
    return concepts


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def create_cards_for_notebook(user_id: int, notebook_id: str, notebook_json: dict) -> int:
    """Extract concepts and create review cards. Returns count of cards created."""
    db = SessionLocal()
    try:
        existing = (
            db.query(ReviewCard)
            .filter(ReviewCard.user_id == user_id, ReviewCard.notebook_id == notebook_id)
            .count()
        )
        if existing > 0:
            return 0

        concepts = extract_concepts(notebook_json)
        now = datetime.now(timezone.utc)
        created = 0

        for c in concepts:
            card = ReviewCard(
                user_id=user_id,
                notebook_id=notebook_id,
                section_title=c.get("section_title", ""),
                concept=c.get("concept", ""),
                concept_summary=c.get("summary", ""),
                interval=1.0,
                ease_factor=2.5,
                repetitions=0,
                next_review=now,
                created_at=now,
            )
            db.add(card)
            created += 1

        db.commit()
        return created
    except Exception:
        db.rollback()
        traceback.print_exc()
        return 0
    finally:
        db.close()


def get_due_cards(user_id: int, limit: int = 20) -> list[dict]:
    """Get review cards due now, ordered by most overdue first."""
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        cards = (
            db.query(ReviewCard)
            .filter(ReviewCard.user_id == user_id, ReviewCard.next_review <= now)
            .order_by(ReviewCard.next_review.asc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": c.id,
                "notebook_id": c.notebook_id,
                "section_title": c.section_title,
                "concept": c.concept,
                "concept_summary": c.concept_summary,
                "interval": c.interval,
                "ease_factor": round(c.ease_factor, 2),
                "repetitions": c.repetitions,
                "last_review": c.last_review.isoformat() if c.last_review else None,
                "next_review": c.next_review.isoformat() if c.next_review else None,
            }
            for c in cards
        ]
    finally:
        db.close()


def submit_review(card_id: int, user_id: int, quality: int) -> dict:
    """Apply SM-2 update and log the review. Returns updated card info."""
    db = SessionLocal()
    try:
        card = (
            db.query(ReviewCard)
            .filter(ReviewCard.id == card_id, ReviewCard.user_id == user_id)
            .first()
        )
        if not card:
            return {"error": "Card not found"}

        sm2_update(card, quality)

        history = ReviewHistory(
            card_id=card.id,
            user_id=user_id,
            quality=quality,
            reviewed_at=datetime.now(timezone.utc),
        )
        db.add(history)
        db.commit()

        return {
            "id": card.id,
            "concept": card.concept,
            "interval": card.interval,
            "ease_factor": round(card.ease_factor, 2),
            "repetitions": card.repetitions,
            "next_review": card.next_review.isoformat() if card.next_review else None,
        }
    except Exception:
        db.rollback()
        traceback.print_exc()
        return {"error": "Failed to submit review"}
    finally:
        db.close()


def get_review_stats(user_id: int) -> dict:
    """Aggregate stats for the dashboard."""
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        total = db.query(ReviewCard).filter(ReviewCard.user_id == user_id).count()
        due = (
            db.query(ReviewCard)
            .filter(ReviewCard.user_id == user_id, ReviewCard.next_review <= now)
            .count()
        )
        mastered = (
            db.query(ReviewCard)
            .filter(ReviewCard.user_id == user_id, ReviewCard.interval > 21)
            .count()
        )
        struggling = (
            db.query(ReviewCard)
            .filter(ReviewCard.user_id == user_id, ReviewCard.ease_factor < 1.8)
            .count()
        )
        return {
            "total_cards": total,
            "due_today": due,
            "mastered": mastered,
            "struggling": struggling,
        }
    finally:
        db.close()


def backfill_cards_for_user(user_id: int) -> int:
    """Extract concepts from any notebooks that don't have review cards yet.
    Also marks never-reviewed cards as immediately due.
    """
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        never_reviewed = (
            db.query(ReviewCard)
            .filter(
                ReviewCard.user_id == user_id,
                ReviewCard.last_review == None,
                ReviewCard.next_review > now,
            )
            .all()
        )
        for card in never_reviewed:
            card.next_review = now
        if never_reviewed:
            db.commit()

        notebooks = db.query(SavedNotebook).filter(SavedNotebook.user_id == user_id, SavedNotebook.deleted_at == None).all()
        total_created = 0
        for nb in notebooks:
            existing = db.query(ReviewCard).filter(
                ReviewCard.user_id == user_id,
                ReviewCard.notebook_id == nb.notebook_id,
            ).count()
            if existing > 0:
                continue
            try:
                notebook_json = json.loads(nb.notebook_json)
                count = create_cards_for_notebook(user_id, nb.notebook_id, notebook_json)
                total_created += count
            except Exception:
                traceback.print_exc()
        return total_created
    finally:
        db.close()


def generate_briefing(user_id: int, user_name: str) -> str:
    """Generate Pedro's daily briefing message using LLM."""
    from database import SkillProfile, QuizSession

    backfill_cards_for_user(user_id)

    db = SessionLocal()
    try:
        stats = get_review_stats(user_id)

        skill = db.query(SkillProfile).filter(SkillProfile.user_id == user_id).first()
        profile = json.loads(skill.profile_json) if skill and skill.profile_json else {}
        weak = sorted(profile.items(), key=lambda x: x[1])[:3]

        sessions = (
            db.query(QuizSession)
            .filter(QuizSession.user_id == user_id, QuizSession.completed == True)
            .order_by(QuizSession.completed_at.desc())
            .limit(3)
            .all()
        )
        recent_activity = [
            f"{s.paper_title}: {s.score}/{s.total}"
            for s in sessions if s.paper_title
        ]

        due_cards_sample = get_due_cards(user_id, limit=5)
        due_concepts = [c["concept"] for c in due_cards_sample]

        context = (
            f"Student name: {user_name}\n"
            f"Cards due for review: {stats['due_today']}\n"
            f"Total cards: {stats['total_cards']}\n"
            f"Mastered cards: {stats['mastered']}\n"
            f"Struggling cards: {stats['struggling']}\n"
            f"Weak topics: {', '.join(f'{t}: {s}/100' for t, s in weak) if weak else 'None yet'}\n"
            f"Due concepts: {', '.join(due_concepts) if due_concepts else 'None right now'}\n"
            f"Recent quiz scores: {', '.join(recent_activity) if recent_activity else 'No quizzes yet'}\n"
        )

        system = (
            "You are Pedro, a warm study buddy. Write a SHORT daily briefing (2-4 sentences) for the student. "
            "Be encouraging and specific. Mention what they should review today, any weak areas to focus on, "
            "and celebrate progress. Keep it conversational — like a friend texting before a study session. "
            "Do NOT use bullet points or headers. Just flowing text."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": context},
        ]

        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if gemini_key:
            client = genai.Client(api_key=gemini_key)
            try:
                response = client.models.generate_content(
                    model="gemini-3.1-pro-preview",
                    contents=context,
                    config={
                        "system_instruction": system,
                        "max_output_tokens": 600,
                        "temperature": 0.8,
                        "thinking_config": {"thinking_budget": 1024},
                    },
                )
                text = ""
                if response.candidates and response.candidates[0].content:
                    for part in (response.candidates[0].content.parts or []):
                        if hasattr(part, "text") and part.text:
                            text += part.text
                if text.strip():
                    return text.strip()
            except Exception:
                traceback.print_exc()
        
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key:
            openai_client = OpenAI(api_key=openai_key)
            resp = openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                messages=messages,
                max_tokens=500,
                temperature=0.8,
            )
            text = resp.choices[0].message.content or ""
            if text.strip():
                return text.strip()

        return _fallback_briefing(user_name, stats)

    except Exception:
        traceback.print_exc()
        return _fallback_briefing(user_name, stats if 'stats' in dir() else get_review_stats(user_id))
    finally:
        db.close()


def _fallback_briefing(name: str, stats: dict) -> str:
    """Non-LLM fallback briefing."""
    first = name.split()[0] if name else "there"
    due = stats.get("due_today", 0)
    if due > 0:
        return (
            f"Hey {first}! You have {due} concept{'s' if due != 1 else ''} due for review today. "
            f"A quick 5-minute session will keep everything fresh. Let's go!"
        )
    total = stats.get("total_cards", 0)
    if total > 0:
        mastered = stats.get("mastered", 0)
        return (
            f"Nice work, {first}! Nothing due for review right now — you've mastered {mastered} out of {total} concepts. "
            f"Keep building your notebook library to add more."
        )
    return (
        f"Welcome, {first}! Upload some lecture notes or explore a notebook to start building your review deck. "
        f"Pedro will help you remember everything."
    )
