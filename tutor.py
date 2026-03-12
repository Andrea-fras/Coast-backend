"""Pedro – Adaptive Socratic AI Tutor engine.

Assembles context, builds system prompts, and calls GPT-4o for
conversational tutoring that references the student's notebooks,
skill profile, and past interactions.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from google import genai

load_dotenv()

from database import (
    ChatMessage,
    QuizSession,
    SavedNotebook,
    SessionAnswer,
    SessionLocal,
    SkillProfile,
    TutorMemo,
    User,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
KIMI_API_KEY = os.getenv("KIMI_API_KEY", "")
MEMO_UPDATE_INTERVAL = 5  # Update memo every N new messages
MAX_HISTORY_MESSAGES = 20

# Provider config for Pedro chat vs memo updates
TUTOR_PROVIDERS = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
        "base_url": None,
    },
    "kimi": {
        "api_key_env": "KIMI_API_KEY",
        "model": "moonshotai/kimi-k2.5",
        "base_url": "https://integrate.api.nvidia.com/v1",
    },
    "gemini": {
        "model": "gemini-3.1-pro-preview",
    },
}

# Which provider to use for chat responses (switch here)
CHAT_PROVIDER = os.getenv("PEDRO_PROVIDER", "gemini")
# Memo updates always use OpenAI (cheaper, reliable, not user-facing)
MEMO_PROVIDER = "openai"

# ---------------------------------------------------------------------------
# System Prompt Templates
# ---------------------------------------------------------------------------

PEDRO_IDENTITY = """You are Pedro, a warm and encouraging Socratic tutor for university students.

CORE RULES:
1. NEVER give the direct answer upfront. Ask a leading question to guide them.
2. Wait for the student to respond before continuing.
3. When you have notes to reference, ONLY use content from the provided notes — never invent facts or equations. When no notes are available, you may use general knowledge but be clear about it.
4. Keep responses concise (2-4 sentences). Students lose focus on walls of text.
5. Use analogies and real-world examples when explaining abstract concepts.
6. Address the student by name when it feels natural.
7. When recommending study actions, be specific (which topic, which notebook section).

PROGRESSION RULES (CRITICAL — avoid repetitive loops):
8. When the student answers correctly, DO NOT just rephrase their answer back as a question. Instead, either:
   a) Introduce a DEEPER concept or nuance they haven't covered yet (from the notes),
   b) Give them a concrete mini-challenge or example problem to test their understanding,
   c) Connect the topic to a DIFFERENT related concept from their notes, or
   d) Acknowledge mastery and suggest what to study next.
9. If the student has answered correctly 2-3 times in a row on the same topic, they understand it. Move on. Say something like "You've got this down — want to explore [next topic] or try a practice problem?"
10. NEVER ask "How might this apply to X?" or "How does this help when Y?" more than once per topic. Variety is key.
11. If the student seems confused, give a smaller hint. If they seem frustrated, simplify and be encouraging.
12. Add VALUE with each response — share a fact, connection, edge case, or insight from the notes that the student hasn't mentioned yet. Don't just echo what they said.

ANTI-REPETITION RULES (CRITICAL):
13. NEVER use the same question pattern twice in a row. Rotate between these approaches:
    - Mini-problem: "Try this: if X, what happens to Y?"
    - Connection: "This actually links to [other topic] because..."
    - Edge case: "But what if [unusual scenario]?"
    - Deeper why: "Why do you think this works rather than [alternative]?"
    - Prediction: "Given this, what would you expect to happen if we changed [variable]?"
    - Comparison: "How does this differ from [related concept]?"
14. Do NOT always start with a short definition. If the student asks about a topic, vary your opening:
    - Sometimes start with a provocative question
    - Sometimes start with a surprising fact or counterexample
    - Sometimes start with a scenario/problem FIRST, then explain after
    - Sometimes start with what makes the topic tricky or commonly misunderstood
15. Do NOT ask "Can you think of a real-world example?" more than once per conversation. If the student's memo says they like real-world analogies, YOU provide the analogy instead of always asking them to think of one.
16. When the student asks "what am I weakest in" or similar, don't start from scratch with basics. Jump to the level they're at — give them a targeted challenge problem for their weak area, then teach based on their response.

NOTEBOOK NUDGE RULES:
13. If you are in a general chat and NO notebook content is available for the topic the student is asking about, mention ONCE (in your first reply on that topic) that uploading their lecture notes would let you give much more specific, course-tailored explanations. Keep it brief and natural, e.g. "I can help with the basics here — but if you upload your lecture slides on this topic, I can give you explanations tailored exactly to your course!"
14. Do NOT repeat the notebook upload suggestion if the student continues asking about the same topic. They heard you — just help them as best you can with general knowledge.
15. If the student changes to a DIFFERENT topic that also has no notebook, you may suggest uploading once more for that new topic.

PROACTIVE STUDY RECOMMENDATIONS:
16. If the student's skill profile shows weak areas (score < 50), look for natural moments to suggest reviewing those topics. Don't force it — weave it in when relevant, e.g. "By the way, your quiz results show derivatives might need some attention — want to work through a few problems together?"
17. When the student starts a new conversation with no specific question, consider proactively suggesting they work on their weakest area. But only do this at the START of a conversation, not mid-discussion.
18. Be specific with recommendations: name the topic, the score if helpful, and suggest a concrete action (review a notebook section, try practice problems, etc.)."""

MEMO_UPDATE_PROMPT = """You are maintaining a structured memo about a student for their AI tutor Pedro.
The memo has THREE sections with different retention rules. You MUST output all three sections.

Here is the current memo:
---
{current_memo}
---

Here are the student's most recent messages and Pedro's replies:
---
{recent_messages}
---

Update the memo using EXACTLY this format:

## PERMANENT
(Facts that should NEVER be removed — learning style, background, accessibility needs, major breakthroughs, core personality traits. Only add here if you're confident it's a lasting trait observed multiple times. Keep this section short — max 5 bullets.)

## PATTERNS
(Long-term trends — recurring struggles, improving areas, consistent behaviours. Only remove a pattern if it's clearly superseded by a newer one. Max 6 bullets.)

## ACTIVE
(What they're currently working on, recent struggles, latest recommendations, emotional state. This section rotates freely — drop old items to make room for new ones. Max 8 bullets.)

RULES:
- NEVER delete items from PERMANENT unless they're proven wrong.
- NEVER delete items from PATTERNS unless a newer pattern contradicts them.
- Freely rotate ACTIVE items — newest info wins.
- Total memo must stay under 400 words.
- Use concise bullet points (one line each).
- Output ONLY the memo (all three sections), nothing else."""

MEMO_MAX_CHARS = 2500  # Hard cap — truncate if LLM exceeds this

SUMMARIZE_THRESHOLD = 12  # Summarize when history exceeds this many messages
KEEP_RECENT = 6          # Keep this many recent messages in full after summarizing

CONVERSATION_SUMMARY_PROMPT = """Summarize the following tutoring conversation between a student and Pedro (AI tutor) in 3-5 concise bullet points.
Focus on:
- What topics were discussed
- What the student understood vs struggled with
- Any key explanations or analogies Pedro gave
- Where the conversation left off

Conversation:
---
{conversation}
---

Output ONLY the bullet-point summary, nothing else."""


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def _get_client(provider: str = "openai") -> tuple[OpenAI, str]:
    """Return (client, model_name) for the given provider. For Gemini, returns None."""
    cfg = TUTOR_PROVIDERS.get(provider, TUTOR_PROVIDERS["openai"])
    if provider == "gemini":
        return None, cfg["model"]
    api_key = os.getenv(cfg["api_key_env"], "")
    kwargs = {"api_key": api_key}
    if cfg.get("base_url"):
        kwargs["base_url"] = cfg["base_url"]
    return OpenAI(**kwargs), cfg["model"]


def _call_gemini(messages: list[dict], max_tokens: int = 500, temperature: float = 0.7) -> str:
    """Call Gemini 3.1 Pro with an OpenAI-style messages list."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
    model = TUTOR_PROVIDERS["gemini"]["model"]

    parts = []
    system_text = ""
    for m in messages:
        if m["role"] == "system":
            system_text += m["content"] + "\n"
        elif m["role"] == "user":
            parts.append({"role": "user", "parts": [{"text": m["content"]}]})
        elif m["role"] == "assistant":
            parts.append({"role": "model", "parts": [{"text": m["content"]}]})

    config = {
        "temperature": temperature,
        "max_output_tokens": 4096,
        "thinking_config": {"thinking_budget": 1024},
    }
    if system_text:
        config["system_instruction"] = system_text.strip()

    response = client.models.generate_content(
        model=model,
        contents=parts,
        config=config,
    )

    # Extract only text parts, skip thought_signature parts
    text_parts = []
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)
    return " ".join(text_parts).strip() if text_parts else response.text.strip()


def _summarize_old_messages(messages: list[ChatMessage]) -> Optional[str]:
    """Compress older messages into a short summary to preserve context in long conversations."""
    if len(messages) <= SUMMARIZE_THRESHOLD:
        return None

    old_msgs = messages[:-KEEP_RECENT]
    conversation_text = "\n".join(
        f"{'Student' if m.role == 'user' else 'Pedro'}: {m.content}" for m in old_msgs
    )

    try:
        client, model = _get_client(MEMO_PROVIDER)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": CONVERSATION_SUMMARY_PROMPT.format(conversation=conversation_text)}],
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Pedro] Conversation summarization failed: {e}")
        return None


def build_system_prompt(
    user: User,
    context_type: str,
    notebook_content: Optional[str] = None,
    tutor_memo: Optional[str] = None,
    skill_profile: Optional[dict] = None,
    session_context: Optional[str] = None,
    explicit_notebook_ref: bool = False,
) -> str:
    """Construct the full system prompt with all available context."""
    parts = [PEDRO_IDENTITY]

    # Student info
    parts.append(f"\nThe student's name is {user.name}.")
    if user.course:
        parts.append(f"They are studying {user.course}.")

    # Skill profile — with actionable guidance
    if skill_profile:
        weak = {k: v for k, v in skill_profile.items() if v < 50}
        medium = {k: v for k, v in skill_profile.items() if 50 <= v < 70}
        strong = {k: v for k, v in skill_profile.items() if v >= 70}
        if weak:
            parts.append(
                f"\nWeak areas needing attention (score < 50): {', '.join(f'{k} ({v}%)' for k, v in sorted(weak.items(), key=lambda x: x[1]))}"
            )
            weakest = min(weak.items(), key=lambda x: x[1])
            parts.append(
                f"Their weakest topic is '{weakest[0]}' at {weakest[1]}%. Consider proactively suggesting they work on this."
            )
        if medium:
            parts.append(
                f"Developing areas (50-69): {', '.join(f'{k} ({v}%)' for k, v in sorted(medium.items(), key=lambda x: x[1]))}"
            )
        if strong:
            parts.append(
                f"Strong areas (score >= 70): {', '.join(f'{k} ({v}%)' for k, v in sorted(strong.items(), key=lambda x: x[1], reverse=True))}"
            )
        if not weak and not medium and not strong:
            parts.append("\nNo quiz data yet — the student hasn't completed any practice sessions.")

    # Tutor memo
    if tutor_memo:
        parts.append(f"\nYour notes about this student from past sessions:\n{tutor_memo}")

    # Context-specific content
    if context_type == "notebook" and notebook_content:
        parts.append(
            "\n--- NOTEBOOK CONTENT (use ONLY this for factual references) ---\n"
            + notebook_content[:12000]
        )
    elif context_type == "session" and session_context:
        parts.append(
            "\n--- SESSION RESULTS (the student just completed a quiz) ---\n"
            + session_context
        )
    elif context_type == "global":
        if notebook_content and explicit_notebook_ref:
            parts.append(
                "\nThis is a general conversation. The student explicitly referenced the following notebook(s) — "
                "use them as your PRIMARY source for factual references, just like in a notebook-specific chat."
                "\n--- REFERENCED NOTEBOOK CONTENT ---\n"
                + notebook_content[:12000]
                + "\n--- END NOTEBOOK CONTENT ---"
            )
        elif notebook_content:
            parts.append(
                "\nThis is a general conversation. You found some relevant content from the student's notebooks:"
                "\n--- RELEVANT NOTEBOOK SNIPPETS ---\n"
                + notebook_content
                + "\n--- END SNIPPETS ---"
                "\nReference these when answering. If the student asks about a topic NOT covered in these snippets, "
                "follow the NOTEBOOK NUDGE RULES (suggest uploading once, then continue helping)."
            )
        else:
            parts.append(
                "\nThis is a general conversation. You have NO notebook content for the student's current question. "
                "Follow the NOTEBOOK NUDGE RULES: gently suggest uploading lecture notes for this topic (ONCE), "
                "but continue helping with general knowledge if they keep asking. "
                "Use your knowledge of their skill profile and past interactions to give personalised advice."
            )

    return "\n".join(parts)


def _load_notebook_text(notebook_id: str, user_id: int) -> Optional[str]:
    """Load notebook content as plain text for the system prompt."""
    db = SessionLocal()
    try:
        nb = (
            db.query(SavedNotebook)
            .filter(
                SavedNotebook.user_id == user_id,
                SavedNotebook.notebook_id == notebook_id,
                SavedNotebook.deleted_at == None,
            )
            .first()
        )
        if not nb:
            nb = (
                db.query(SavedNotebook)
                .filter(
                    SavedNotebook.notebook_id == notebook_id,
                    SavedNotebook.is_premade == True,
                    SavedNotebook.deleted_at == None,
                )
                .first()
            )
        if not nb:
            return None

        data = json.loads(nb.notebook_json)
        parts = [f"Title: {data.get('title', '')}"]
        for section in data.get("sections", []):
            parts.append(f"\n## {section.get('title', '')}")
            parts.append(section.get("content", ""))
            for sub in section.get("subsections", []):
                parts.append(f"### {sub.get('title', '')}")
                parts.append(sub.get("content", ""))
                for bullet in sub.get("bullets", []):
                    parts.append(f"  - {bullet}")
        return "\n".join(parts)
    finally:
        db.close()


def _load_session_context(session_id: int, user_id: int) -> Optional[str]:
    """Build a text summary of a quiz session's wrong answers."""
    db = SessionLocal()
    try:
        session = (
            db.query(QuizSession)
            .filter(QuizSession.id == session_id, QuizSession.user_id == user_id)
            .first()
        )
        if not session:
            return None

        answers = db.query(SessionAnswer).filter(SessionAnswer.session_id == session_id).all()
        lines = [
            f"Quiz: {session.paper_title or session.paper_id}",
            f"Score: {session.score}/{session.total}",
            "",
            "Wrong answers:",
        ]
        for a in answers:
            if not a.is_correct:
                lines.append(f"- Q: {a.question_text}")
                lines.append(f"  Student answered: {a.user_answer}")
                lines.append(f"  Correct answer: {a.correct_answer}")
                lines.append("")

        correct_count = sum(1 for a in answers if a.is_correct)
        lines.append(f"\nCorrect answers: {correct_count}/{len(answers)}")
        return "\n".join(lines)
    finally:
        db.close()


def _get_relevant_notebook_snippets(user_id: int, message: str, max_chars: int = 4000) -> str:
    """For global chat: find the most relevant notebook content based on the user's message."""
    db = SessionLocal()
    try:
        notebooks = (
            db.query(SavedNotebook)
            .filter(
                (SavedNotebook.user_id == user_id) | (SavedNotebook.is_premade == True),
                SavedNotebook.deleted_at == None,
            )
            .all()
        )
        if not notebooks:
            return ""

        keywords = set(message.lower().split())
        scored = []
        for nb in notebooks:
            data = json.loads(nb.notebook_json)
            text = json.dumps(data).lower()
            score = sum(1 for kw in keywords if kw in text and len(kw) > 3)
            if score > 0:
                scored.append((score, data))

        scored.sort(key=lambda x: x[0], reverse=True)

        result_parts = []
        total = 0
        for _, data in scored[:3]:
            snippet = f"From '{data.get('title', '')}': "
            for sec in data.get("sections", [])[:3]:
                snippet += f"\n{sec.get('title', '')}: {sec.get('content', '')[:300]}"
            if total + len(snippet) > max_chars:
                break
            result_parts.append(snippet)
            total += len(snippet)

        return "\n\n".join(result_parts)
    finally:
        db.close()


def send_message(
    user_id: int,
    message: str,
    conversation_id: Optional[str],
    context_type: str,
    context_id: Optional[str] = None,
    notebook_ids: Optional[list[str]] = None,
) -> dict:
    """Process a student message and return Pedro's response.

    Returns: { reply, conversation_id, message_id }
    """
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError("User not found")

        # Generate conversation_id if new conversation
        if not conversation_id:
            conversation_id = f"conv_{uuid.uuid4().hex[:12]}"

        # Load tutor memo
        memo_row = db.query(TutorMemo).filter(TutorMemo.user_id == user_id).first()
        memo_text = memo_row.memo_text if memo_row else ""

        # Load skill profile
        skill_row = db.query(SkillProfile).filter(SkillProfile.user_id == user_id).first()
        skill_data = json.loads(skill_row.profile_json) if skill_row else {}

        # Load context-specific content
        notebook_content = None
        session_context = None
        explicit_notebook_ref = False

        if context_type == "notebook" and context_id:
            notebook_content = _load_notebook_text(context_id, user_id)
        elif context_type == "session" and context_id:
            try:
                session_context = _load_session_context(int(context_id), user_id)
            except (ValueError, TypeError):
                pass
        elif context_type == "global":
            if notebook_ids:
                parts = []
                for nb_id in notebook_ids[:3]:
                    text = _load_notebook_text(nb_id, user_id)
                    if text:
                        parts.append(text)
                if parts:
                    notebook_content = "\n\n--- NEXT NOTEBOOK ---\n\n".join(parts)
                    explicit_notebook_ref = True
            if not notebook_content:
                snippets = _get_relevant_notebook_snippets(user_id, message)
                if snippets:
                    notebook_content = snippets

        # Build system prompt
        system_prompt = build_system_prompt(
            user=user,
            context_type=context_type,
            notebook_content=notebook_content,
            tutor_memo=memo_text,
            skill_profile=skill_data,
            session_context=session_context,
            explicit_notebook_ref=explicit_notebook_ref,
        )

        # Load conversation history
        history = (
            db.query(ChatMessage)
            .filter(ChatMessage.conversation_id == conversation_id)
            .order_by(ChatMessage.created_at.asc())
            .all()
        )

        # Build messages array — summarize old messages if the conversation is long
        messages = [{"role": "system", "content": system_prompt}]

        summary = _summarize_old_messages(history)
        if summary:
            messages.append({
                "role": "system",
                "content": f"Summary of earlier conversation:\n{summary}",
            })
            recent_history = history[-KEEP_RECENT:]
        else:
            recent_history = history[-MAX_HISTORY_MESSAGES:]

        for msg in recent_history:
            role = "assistant" if msg.role == "pedro" else "user"
            messages.append({"role": role, "content": msg.content})
        messages.append({"role": "user", "content": message})

        # Call LLM (uses CHAT_PROVIDER for student-facing responses)
        if CHAT_PROVIDER == "gemini":
            reply = _call_gemini(messages, max_tokens=500, temperature=0.7)
        else:
            client, model = _get_client(CHAT_PROVIDER)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=500,
                temperature=0.7,
            )
            reply = response.choices[0].message.content.strip()

        # Save user message
        user_msg = ChatMessage(
            user_id=user_id,
            conversation_id=conversation_id,
            role="user",
            content=message,
            context_type=context_type,
            context_id=context_id,
        )
        db.add(user_msg)

        # Save Pedro's reply
        pedro_msg = ChatMessage(
            user_id=user_id,
            conversation_id=conversation_id,
            role="pedro",
            content=reply,
            context_type=context_type,
            context_id=context_id,
        )
        db.add(pedro_msg)
        db.commit()
        db.refresh(pedro_msg)

        # Check if memo needs updating
        if memo_row:
            memo_row.message_count_since_update += 2  # user + pedro
            if memo_row.message_count_since_update >= MEMO_UPDATE_INTERVAL:
                _trigger_memo_update(db, user_id, conversation_id, memo_row)
        else:
            # Create initial memo after first conversation
            new_memo = TutorMemo(
                user_id=user_id,
                memo_text="",
                message_count_since_update=2,
            )
            db.add(new_memo)

        db.commit()

        return {
            "reply": reply,
            "conversation_id": conversation_id,
            "message_id": pedro_msg.id,
        }
    finally:
        db.close()


def _trigger_memo_update(db, user_id: int, conversation_id: str, memo_row: TutorMemo):
    """Update the tutor memo with recent interaction observations."""
    try:
        recent = (
            db.query(ChatMessage)
            .filter(ChatMessage.user_id == user_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(10)
            .all()
        )
        recent.reverse()

        recent_text = "\n".join(
            f"{'Student' if m.role == 'user' else 'Pedro'}: {m.content}" for m in recent
        )

        prompt = MEMO_UPDATE_PROMPT.format(
            current_memo=memo_row.memo_text or "(No memo yet — this is the first interaction.)",
            recent_messages=recent_text,
        )

        # Memo updates always use OpenAI (cheap, reliable, not user-facing)
        client, model = _get_client(MEMO_PROVIDER)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.5,
        )
        new_memo = response.choices[0].message.content.strip()

        # Hard cap — truncate to last complete bullet if too long
        if len(new_memo) > MEMO_MAX_CHARS:
            truncated = new_memo[:MEMO_MAX_CHARS]
            last_newline = truncated.rfind('\n')
            if last_newline > MEMO_MAX_CHARS // 2:
                new_memo = truncated[:last_newline]
            else:
                new_memo = truncated

        memo_row.memo_text = new_memo
        memo_row.message_count_since_update = 0
        memo_row.updated_at = datetime.now(timezone.utc)
    except Exception as e:
        print(f"[Pedro] Memo update failed: {e}")


def update_skill_profile(user_id: int):
    """Recalculate the student's skill profile from quiz session data.

    Looks at all completed sessions, extracts topic tags from questions,
    and computes accuracy per topic.
    """
    db = SessionLocal()
    try:
        sessions = (
            db.query(QuizSession)
            .filter(QuizSession.user_id == user_id, QuizSession.completed == True)
            .all()
        )

        topic_stats: dict[str, dict] = {}  # topic -> {correct: int, total: int}

        for session in sessions:
            answers = db.query(SessionAnswer).filter(SessionAnswer.session_id == session.id).all()
            for ans in answers:
                # Extract topics from question text (simple keyword extraction)
                tags = _extract_topic_tags(ans.question_text, ans.correct_answer)
                for tag in tags:
                    if tag not in topic_stats:
                        topic_stats[tag] = {"correct": 0, "total": 0}
                    topic_stats[tag]["total"] += 1
                    if ans.is_correct:
                        topic_stats[tag]["correct"] += 1

        # Convert to proficiency scores (0-100)
        profile = {}
        for topic, stats in topic_stats.items():
            if stats["total"] > 0:
                profile[topic] = round(stats["correct"] / stats["total"] * 100)

        # Upsert skill profile
        existing = db.query(SkillProfile).filter(SkillProfile.user_id == user_id).first()
        if existing:
            existing.profile_json = json.dumps(profile)
            existing.updated_at = datetime.now(timezone.utc)
        else:
            new_profile = SkillProfile(
                user_id=user_id,
                profile_json=json.dumps(profile),
            )
            db.add(new_profile)

        db.commit()
        return profile
    finally:
        db.close()


def _extract_topic_tags(question_text: str, correct_answer: str) -> list[str]:
    """Extract topic tags from question text using keyword matching.

    Simple heuristic: looks for common academic keywords.
    """
    text = (question_text + " " + correct_answer).lower()
    tags = []

    topic_keywords = {
        "derivatives": ["derivative", "differentiate", "d/dx", "power rule", "chain rule"],
        "integrals": ["integral", "integrate", "antiderivative", "area under"],
        "linear equations": ["linear equation", "solve for x", "2x +", "3x -"],
        "quadratics": ["quadratic", "x²", "x^2", "parabola", "factoring"],
        "percentages": ["percent", "%", "proportion"],
        "statistics": ["mean", "median", "standard deviation", "variance", "probability"],
        "elasticity": ["elasticity", "elastic", "inelastic"],
        "supply and demand": ["supply", "demand", "equilibrium", "market"],
        "geometry": ["area", "perimeter", "circle", "triangle", "radius"],
        "functions": ["function", "f(x)", "domain", "range", "substitut"],
        "matrices": ["matrix", "matrices", "determinant", "eigenvalue"],
        "regression": ["regression", "correlation", "r-squared", "least squares"],
    }

    for topic, keywords in topic_keywords.items():
        if any(kw in text for kw in keywords):
            tags.append(topic)

    if not tags:
        # Fallback: use first few significant words
        words = [w for w in text.split() if len(w) > 4][:2]
        if words:
            tags.append(" ".join(words))

    return tags


def get_chat_history(conversation_id: str, user_id: int) -> list[dict]:
    """Get all messages for a conversation."""
    db = SessionLocal()
    try:
        messages = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.conversation_id == conversation_id,
                ChatMessage.user_id == user_id,
            )
            .order_by(ChatMessage.created_at.asc())
            .all()
        )
        return [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in messages
        ]
    finally:
        db.close()


def get_conversations(user_id: int) -> list[dict]:
    """List all conversations for a user, with last message preview."""
    db = SessionLocal()
    try:
        # Get distinct conversation IDs with their latest message
        from sqlalchemy import func

        convos = (
            db.query(
                ChatMessage.conversation_id,
                ChatMessage.context_type,
                ChatMessage.context_id,
                func.max(ChatMessage.created_at).label("last_at"),
            )
            .filter(ChatMessage.user_id == user_id)
            .group_by(ChatMessage.conversation_id)
            .order_by(func.max(ChatMessage.created_at).desc())
            .all()
        )

        results = []
        for conv_id, ctx_type, ctx_id, last_at in convos:
            # Get the last message for preview
            last_msg = (
                db.query(ChatMessage)
                .filter(ChatMessage.conversation_id == conv_id)
                .order_by(ChatMessage.created_at.desc())
                .first()
            )
            results.append({
                "conversation_id": conv_id,
                "context_type": ctx_type,
                "context_id": ctx_id,
                "last_message": last_msg.content[:100] if last_msg else "",
                "last_role": last_msg.role if last_msg else "",
                "updated_at": last_at.isoformat() if last_at else None,
            })

        return results
    finally:
        db.close()


def get_skill_profile(user_id: int) -> dict:
    """Get the user's skill profile."""
    db = SessionLocal()
    try:
        row = db.query(SkillProfile).filter(SkillProfile.user_id == user_id).first()
        if not row:
            return {"topics": {}, "updated_at": None}
        return {
            "topics": json.loads(row.profile_json),
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }
    finally:
        db.close()


def get_tutor_memo(user_id: int) -> dict:
    """Get the tutor's memo about a user."""
    db = SessionLocal()
    try:
        row = db.query(TutorMemo).filter(TutorMemo.user_id == user_id).first()
        if not row:
            return {"memo": "", "updated_at": None}
        return {
            "memo": row.memo_text,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }
    finally:
        db.close()


NOTE_CONDENSE_PROMPT = """You are converting a tutor's explanation into a concise study note
to be inserted into a student's notebook.

The tutor said:
---
{pedro_message}
---

Convert this into a SHORT study note (2-4 bullet points) suitable for inserting into lecture notes.
Rules:
- Use clear, factual language (not conversational)
- Keep each bullet to 1-2 sentences max
- Include any key definitions, formulas, or examples mentioned
- Use HTML formatting: wrap in <ul><li>...</li></ul>
- Do NOT include greetings, encouragement, or questions — only the knowledge
- Output ONLY the HTML, nothing else"""


def generate_note_for_notebook(pedro_message: str) -> str:
    """Condense a Pedro chat message into a concise HTML note for the notebook."""
    client, model = _get_client(MEMO_PROVIDER)
    prompt = NOTE_CONDENSE_PROMPT.format(pedro_message=pedro_message)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3,
    )
    html = response.choices[0].message.content.strip()

    # Strip markdown fences if the LLM wraps it
    if html.startswith("```"):
        lines = html.split("\n")
        html = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    return html
