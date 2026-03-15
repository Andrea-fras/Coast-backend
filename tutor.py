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
        "model": os.getenv("GEMINI_CHAT_MODEL", "gemini-3-flash-preview"),
    },
}

# Which provider to use for chat responses (switch here)
CHAT_PROVIDER = os.getenv("PEDRO_PROVIDER", "gemini")
# Memo updates always use OpenAI (cheaper, reliable, not user-facing)
MEMO_PROVIDER = "openai"

# ---------------------------------------------------------------------------
# Visualization helpers — Claude Opus 4.6 SVG generation
# ---------------------------------------------------------------------------

_VIZ_KEYWORDS = [
    "visualize", "visualise", "visualization", "visualisation",
    "show me a diagram", "draw", "graph this", "graph it",
    "plot this", "plot it", "can you graph", "can you draw",
    "can you plot", "show me a graph", "show me a chart",
    "make a diagram", "create a diagram", "illustrate",
    "show visually", "show it visually", "visual representation",
]


def _detect_viz_request(message: str) -> bool:
    """Check if the user is asking for a visualization."""
    lower = message.lower()
    return any(kw in lower for kw in _VIZ_KEYWORDS)


SVG_VIZ_SYSTEM_PROMPT = """You are Pedro, an expert AI tutor who creates beautiful, clean, modern, and educational SVG visualizations to help students understand concepts.

When asked to visualize something, generate a polished SVG diagram embedded directly in your response. Follow these rules:

STYLE & DESIGN:
- Clean and modern: generous whitespace, no clutter, minimal borders.
- Educational: every element should serve understanding — labels, annotations, legends.
- Rounded corners on rectangles (rx="8"). Soft drop shadows where helpful.
- Subtle gradients are welcome for depth (e.g., linearGradient for backgrounds).
- Consistent spacing and alignment — the diagram should look professionally designed.

SVG RULES:
1. Output the SVG directly in your markdown response using raw HTML (NO markdown code fences around it).
2. Use viewBox for responsive sizing (e.g., viewBox="0 0 700 450"). Do NOT set fixed width/height attributes on the <svg> element.
3. Wrap the SVG in: <div style="text-align:center;margin:1em 0;"><svg ...>...</svg></div>
4. Use clean, readable fonts: font-family="'Inter', 'Segoe UI', system-ui, sans-serif"
5. Typography: use font-weight="600" for headings/titles, font-weight="400" for body labels. Keep font sizes between 12-18px.
6. COLOR PALETTE — use these modern, accessible colors:
   Primary blues:  #3b82f6, #60a5fa, #dbeafe (light fill)
   Greens:         #10b981, #34d399, #d1fae5 (light fill)
   Oranges/Amber:  #f59e0b, #fbbf24, #fef3c7 (light fill)
   Reds/Rose:      #ef4444, #fb7185, #ffe4e6 (light fill)
   Purples:        #8b5cf6, #a78bfa, #ede9fe (light fill)
   Neutrals:       #1e293b (dark text), #64748b (secondary text), #f8fafc (background), #e2e8f0 (borders)
7. Use the lighter shades for fills/backgrounds and darker shades for strokes/text to create depth.
8. Arrows: use marker-end with a clean arrowhead. Lines should use stroke-width="2" and stroke-linecap="round".
9. Add a brief, helpful text explanation before AND/OR after the SVG.

GOOD FOR:
- Function graphs and mathematical curves (use <polyline> or <path> with smooth curves)
- Flowcharts and process diagrams (rounded <rect> + arrows + <text>)
- Data comparison charts (bar charts, grouped bars, simple pie/donut charts)
- Tree structures, state machines, network/architecture diagrams
- Concept maps and relationship diagrams with labeled edges
- Annotated number lines, coordinate systems, and geometric illustrations
- Timelines and step-by-step process flows
- Venn diagrams and set relationships

SIZE CONSTRAINT: Keep SVGs concise and focused. Aim for under 3000 characters of SVG code. Prefer fewer, well-designed elements over exhaustive detail. If a concept is complex, simplify the diagram to its core idea rather than trying to show everything. A clear, simple diagram is always better than a cluttered one.

IMPORTANT: Always include a clear text explanation alongside the visualization. The SVG enhances understanding — it does not replace the explanation."""


def _clean_svg_response(text: str) -> str:
    """Strip markdown code fences from SVG output if present."""
    import re
    text = re.sub(r"```(?:svg|html|xml)?\s*\n?", "", text)
    text = re.sub(r"\n?```", "", text)
    return text.strip()


def _call_claude_for_viz(messages: list[dict], max_tokens: int = 4096) -> str:
    """Call Claude for SVG visualization generation."""
    import anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[Claude Viz] No ANTHROPIC_API_KEY set, skipping")
        return ""

    client = anthropic.Anthropic(api_key=api_key)

    system_text = ""
    claude_messages = []
    for m in messages:
        if m["role"] == "system":
            system_text += m["content"] + "\n"
        else:
            claude_messages.append({"role": m["role"], "content": m["content"]})

    if not claude_messages:
        print("[Claude Viz] No non-system messages to send, skipping")
        return ""

    # Ensure first message is from user (Claude API requirement)
    if claude_messages[0]["role"] != "user":
        claude_messages.insert(0, {"role": "user", "content": "Please provide a visualization."})

    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    print(f"[Claude Viz] Calling model={model}, {len(claude_messages)} messages, system_len={len(system_text)}")

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_text.strip(),
            messages=claude_messages,
        )
        text = response.content[0].text if response.content else ""
        print(f"[Claude Viz] Got response: {len(text)} chars, stop_reason={response.stop_reason}")
        if not text:
            print("[Claude Viz] Empty response from Claude")
            return ""
        cleaned = _clean_svg_response(text)
        has_svg = "<svg" in cleaned.lower()
        print(f"[Claude Viz] After cleaning: {len(cleaned)} chars, has_svg={has_svg}")
        return cleaned
    except anthropic.RateLimitError as e:
        print(f"[Claude Viz] Rate limited by Anthropic API: {e}")
        return ""
    except anthropic.APIError as e:
        print(f"[Claude Viz] API error (status={e.status_code}): {e}")
        return ""
    except Exception:
        import traceback
        print("[Claude Viz] Unexpected error:")
        traceback.print_exc()
        return ""

# ---------------------------------------------------------------------------
# System Prompt Templates
# ---------------------------------------------------------------------------

PEDRO_IDENTITY = """You are Pedro, a warm and knowledgeable tutor for university students.

CORE RULES:
1. Balance teaching and questioning. Sometimes the student needs you to EXPLAIN a concept clearly and in detail before asking them anything. Don't always withhold the answer — if the student is learning something new, teach it properly first, then check understanding.
2. Wait for the student to respond before continuing.
3. When you have notes to reference, ONLY use content from the provided notes — never invent facts or equations. When no notes are available, you may use general knowledge but be clear about it.
4. Keep responses focused but don't be afraid of longer explanations when the topic demands it. A well-structured 2-paragraph explanation is better than a vague 2-sentence hint.
5. Be SPECIFIC and DETAILED. Teach the actual content — definitions, formulas, mechanisms, processes. Students need substance, not just high-level overviews.
6. Address the student by name when it feels natural.
7. When recommending study actions, be specific (which topic, which notebook section).
8. Use analogies SPARINGLY — only when a concept is truly abstract and hard to grasp without one. Most of the time, a clear, direct explanation with a concrete example is better than an analogy. If the student asks for analogies or says they find them helpful, increase their use.

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
15. Do NOT ask "Can you think of a real-world example?" — this wastes the student's time. If an analogy helps, just provide it directly.
16. When the student asks "what am I weakest in" or similar, don't start from scratch with basics. Jump to the level they're at — give them a targeted challenge problem for their weak area, then teach based on their response.
17. TEACH FIRST, ASK SECOND: When introducing new material, explain it thoroughly first. Don't ask the student to guess things they haven't learned yet. Teach -> Example -> Check understanding is the right flow.
18. Adapt depth to the subject: Biology, medicine, and detail-heavy subjects need thorough explanations with specific terms, processes, and mechanisms. Math and physics need step-by-step worked examples. Don't oversimplify — university students need university-level detail.

FORMATTING RULES:
- Use **bold** for key terms and important concepts when first introduced.
- Use bullet points or numbered lists for multi-part explanations.
- Use inline math with $...$ for equations (e.g. $E = mc^2$) and display math with $$...$$ for important formulas.
- Use `backticks` for code, variable names, or short technical terms.
- Use > blockquotes for key insights or important takeaways.
- Use markdown tables when comparing concepts, showing data, listing properties, or organizing information side-by-side. Tables are rendered beautifully in the chat.
- Keep formatting clean and purposeful — don't over-format simple responses.
- When the student asks you to visualize, draw, graph, or diagram something, let them know you can do that — they just need to ask (e.g., "I can draw a diagram of this if you'd like!").

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
    from openai import OpenAI
    return OpenAI(**kwargs), cfg["model"]


def _call_gemini(messages: list[dict], max_tokens: int = 500, temperature: float = 0.7) -> str:
    """Call Gemini 3.1 Pro with an OpenAI-style messages list."""
    from google import genai
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

    try:
        response = client.models.generate_content(
            model=model,
            contents=parts,
            config=config,
        )
    except Exception as e:
        print(f"[Gemini] API call failed: {e}")
        return "I'm having a brief technical issue. Could you try asking again?"

    if not response or not getattr(response, 'candidates', None):
        print(f"[Gemini] No candidates. Feedback: {getattr(response, 'prompt_feedback', None)}")
        return "I'd love to help with that! Could you rephrase your question?"

    text_parts = []
    for candidate in response.candidates:
        content = getattr(candidate, 'content', None)
        if content and getattr(content, 'parts', None):
            for part in content.parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)

    if text_parts:
        return " ".join(text_parts).strip()

    try:
        return response.text.strip()
    except Exception:
        return "I'd love to help with that! Could you rephrase your question?"


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

    # Learning preferences (baseline, not rigid)
    if hasattr(user, 'learning_preferences') and user.learning_preferences:
        try:
            prefs = json.loads(user.learning_preferences) if isinstance(user.learning_preferences, str) else user.learning_preferences
            pref_lines = []
            labels = {
                "learning_style": "Learning approach",
                "when_stuck": "When stuck, prefers",
                "detail_level": "Detail preference",
                "study_goal": "Study goal",
            }
            for key, label in labels.items():
                if key in prefs:
                    pref_lines.append(f"- {label}: {prefs[key]}")
            if pref_lines:
                parts.append(
                    "\nStudent's stated learning preferences (use as baseline guidance, "
                    "not rigid rules — adapt based on what actually works in practice):\n"
                    + "\n".join(pref_lines)
                )
        except (json.JSONDecodeError, TypeError):
            pass

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
    elif context_type == "lesson" and notebook_content:
        parts.append(notebook_content)
    elif context_type == "folder" and notebook_content:
        parts.append(
            "\n--- FOLDER CONTEXT (retrieved from multiple sources via semantic search) ---\n"
            "The student has a folder of study materials. Below are the most relevant "
            "excerpts retrieved from their sources. When answering, ALWAYS reference which "
            "source/notebook the information comes from so the student can find it.\n"
            + notebook_content[:14000]
            + "\n--- END FOLDER CONTEXT ---"
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
        for section in (data.get("sections") or []):
            parts.append(f"\n## {section.get('title', '')}")
            parts.append(section.get("content", "") or "")
            for sub in (section.get("subsections") or []):
                parts.append(f"### {sub.get('title', '')}")
                parts.append(sub.get("content", "") or "")
                for bullet in (sub.get("bullets") or []):
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

        if context_type == "lesson" and context_id:
            try:
                import lesson as lesson_mod
                from curated_config import curated_source_uid as _curated_uid, get_lesson_structure
                src_uid = _curated_uid(context_id)
                notebook_content = lesson_mod.build_lesson_prompt(
                    user_id, context_id,
                    source_user_id=src_uid if src_uid is not None else None,
                    structure=get_lesson_structure(context_id),
                )
            except Exception:
                import traceback as tb
                tb.print_exc()
                notebook_content = None
        elif context_type == "folder" and context_id:
            try:
                import rag
                from curated_config import curated_source_uid as _curated_uid
                rag_uid = _curated_uid(context_id)
                notebook_content = rag.build_folder_context(
                    rag_uid if rag_uid is not None else user_id,
                    context_id, message,
                )
            except Exception:
                import traceback as tb
                tb.print_exc()
                notebook_content = None
        elif context_type == "notebook" and context_id:
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

        # Call LLM — route viz requests to Claude, fall back to CHAT_PROVIDER
        reply = None
        is_viz = _detect_viz_request(message)
        has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
        print(f"[Chat] is_viz={is_viz}, has_anthropic={has_anthropic}, message={message[:80]!r}")
        if is_viz and has_anthropic:
            viz_messages = [{"role": "system", "content": SVG_VIZ_SYSTEM_PROMPT + "\n\n" + system_prompt}]
            for m in messages[1:]:
                viz_messages.append(m)
            reply = _call_claude_for_viz(viz_messages, max_tokens=16000)
            print(f"[Chat] Claude viz reply length: {len(reply) if reply else 0}")

        if not reply:
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


def send_message_stream(
    user_id: int,
    message: str,
    conversation_id: Optional[str],
    context_type: str,
    context_id: Optional[str] = None,
    notebook_ids: Optional[list[str]] = None,
):
    """Streaming version of send_message. Yields (token, None) for each chunk,
    then (None, result_dict) for the final metadata."""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError("User not found")

        if not conversation_id:
            conversation_id = f"conv_{uuid.uuid4().hex[:12]}"

        memo_row = db.query(TutorMemo).filter(TutorMemo.user_id == user_id).first()
        memo_text = memo_row.memo_text if memo_row else ""
        skill_row = db.query(SkillProfile).filter(SkillProfile.user_id == user_id).first()
        skill_data = json.loads(skill_row.profile_json) if skill_row else {}

        notebook_content = None
        session_context = None
        explicit_notebook_ref = False

        if context_type == "lesson" and context_id:
            try:
                import lesson as lesson_mod
                from curated_config import curated_source_uid as _curated_uid, get_lesson_structure
                src_uid = _curated_uid(context_id)
                notebook_content = lesson_mod.build_lesson_prompt(
                    user_id, context_id,
                    source_user_id=src_uid if src_uid is not None else None,
                    structure=get_lesson_structure(context_id),
                )
            except Exception:
                import traceback as tb
                tb.print_exc()
        elif context_type == "folder" and context_id:
            try:
                import rag
                from curated_config import curated_source_uid as _curated_uid
                rag_uid = _curated_uid(context_id)
                notebook_content = rag.build_folder_context(
                    rag_uid if rag_uid is not None else user_id,
                    context_id, message,
                )
            except Exception:
                import traceback as tb
                tb.print_exc()
        elif context_type == "notebook" and context_id:
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

        system_prompt = build_system_prompt(
            user=user,
            context_type=context_type,
            notebook_content=notebook_content,
            tutor_memo=memo_text,
            skill_profile=skill_data,
            session_context=session_context,
            explicit_notebook_ref=explicit_notebook_ref,
        )

        history = (
            db.query(ChatMessage)
            .filter(ChatMessage.conversation_id == conversation_id)
            .order_by(ChatMessage.created_at.asc())
            .all()
        )

        llm_messages = [{"role": "system", "content": system_prompt}]
        summary = _summarize_old_messages(history)
        if summary:
            llm_messages.append({"role": "system", "content": f"Summary of earlier conversation:\n{summary}"})
            recent_history = history[-KEEP_RECENT:]
        else:
            recent_history = history[-MAX_HISTORY_MESSAGES:]

        for msg in recent_history:
            role = "assistant" if msg.role == "pedro" else "user"
            llm_messages.append({"role": role, "content": msg.content})
        llm_messages.append({"role": "user", "content": message})

        full_reply = ""

        viz_done = False
        is_viz = _detect_viz_request(message)
        has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
        print(f"[Stream] is_viz={is_viz}, has_anthropic={has_anthropic}, message={message[:80]!r}")
        if is_viz and has_anthropic:
            viz_messages = [{"role": "system", "content": SVG_VIZ_SYSTEM_PROMPT + "\n\n" + system_prompt}]
            for m in llm_messages[1:]:
                viz_messages.append(m)
            reply = _call_claude_for_viz(viz_messages, max_tokens=16000)
            print(f"[Stream] Claude viz reply length: {len(reply) if reply else 0}")
            if reply:
                full_reply = reply
                yield (reply, None)
                viz_done = True

        if not viz_done and CHAT_PROVIDER == "gemini":
            from google import genai
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
            model_name = TUTOR_PROVIDERS["gemini"]["model"]
            parts = []
            sys_text = ""
            for m in llm_messages:
                if m["role"] == "system":
                    sys_text += m["content"] + "\n"
                elif m["role"] == "user":
                    parts.append({"role": "user", "parts": [{"text": m["content"]}]})
                elif m["role"] == "assistant":
                    parts.append({"role": "model", "parts": [{"text": m["content"]}]})

            config = {"temperature": 0.7, "max_output_tokens": 4096}
            if sys_text:
                config["system_instruction"] = sys_text.strip()

            try:
                for chunk in client.models.generate_content_stream(
                    model=model_name, contents=parts, config=config,
                ):
                    if chunk.text:
                        full_reply += chunk.text
                        yield (chunk.text, None)
            except Exception as e:
                print(f"[Gemini Stream] Error: {e}")
                if not full_reply:
                    full_reply = "I'm having a brief technical issue. Could you try asking again?"
                    yield (full_reply, None)
        elif not viz_done:
            client, model_name = _get_client(CHAT_PROVIDER)
            try:
                stream = client.chat.completions.create(
                    model=model_name, messages=llm_messages,
                    max_tokens=500, temperature=0.7, stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        full_reply += delta.content
                        yield (delta.content, None)
            except Exception as e:
                print(f"[OpenAI Stream] Error: {e}")
                if not full_reply:
                    full_reply = "I'm having a brief technical issue. Could you try asking again?"
                    yield (full_reply, None)

        if not full_reply:
            full_reply = "I'd love to help with that! Could you rephrase your question?"

        user_msg = ChatMessage(
            user_id=user_id, conversation_id=conversation_id,
            role="user", content=message,
            context_type=context_type, context_id=context_id,
        )
        db.add(user_msg)
        pedro_msg = ChatMessage(
            user_id=user_id, conversation_id=conversation_id,
            role="pedro", content=full_reply,
            context_type=context_type, context_id=context_id,
        )
        db.add(pedro_msg)
        db.commit()
        db.refresh(pedro_msg)

        if memo_row:
            memo_row.message_count_since_update += 2
            if memo_row.message_count_since_update >= MEMO_UPDATE_INTERVAL:
                _trigger_memo_update(db, user_id, conversation_id, memo_row)
        else:
            new_memo = TutorMemo(user_id=user_id, memo_text="", message_count_since_update=2)
            db.add(new_memo)
        db.commit()

        yield (None, {
            "reply": full_reply,
            "conversation_id": conversation_id,
            "message_id": pedro_msg.id,
        })
    finally:
        db.close()


def _trigger_memo_update_bg(user_id: int, current_memo_text: str):
    """Background thread: update the tutor memo with recent observations."""
    try:
        bg_db = SessionLocal()
        recent = (
            bg_db.query(ChatMessage)
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
            current_memo=current_memo_text or "(No memo yet — this is the first interaction.)",
            recent_messages=recent_text,
        )

        client, model = _get_client(MEMO_PROVIDER)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.5,
        )
        new_memo = response.choices[0].message.content.strip()

        if len(new_memo) > MEMO_MAX_CHARS:
            truncated = new_memo[:MEMO_MAX_CHARS]
            last_newline = truncated.rfind('\n')
            if last_newline > MEMO_MAX_CHARS // 2:
                new_memo = truncated[:last_newline]
            else:
                new_memo = truncated

        memo_row = bg_db.query(TutorMemo).filter(TutorMemo.user_id == user_id).first()
        if memo_row:
            memo_row.memo_text = new_memo
            memo_row.message_count_since_update = 0
            memo_row.updated_at = datetime.now(timezone.utc)
            bg_db.commit()
        bg_db.close()
        print(f"[Pedro] Memo updated in background for user {user_id}")
    except Exception as e:
        print(f"[Pedro] Background memo update failed: {e}")


def _trigger_memo_update(db, user_id: int, conversation_id: str, memo_row: TutorMemo):
    """Fire-and-forget memo update in a background thread."""
    import threading
    memo_row.message_count_since_update = 0
    db.commit()
    t = threading.Thread(
        target=_trigger_memo_update_bg,
        args=(user_id, memo_row.memo_text or ""),
        daemon=True,
    )
    t.start()


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
                stored_tags = []
                if hasattr(ans, 'tags_json') and ans.tags_json:
                    try:
                        stored_tags = json.loads(ans.tags_json)
                    except (json.JSONDecodeError, TypeError):
                        pass
                tags = stored_tags if stored_tags else _extract_topic_tags(ans.question_text, ans.correct_answer)
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


EXERCISE_GENERATE_PROMPT = """You are creating a single practice question for a university student based on the following notebook section.

Section: {section_title}
Content:
---
{section_content}
---

Generate ONE clear, focused practice question that tests understanding of a key concept from this section.
Rules:
- The question should require thinking, not just recall
- It can be a short-answer question, a "what would happen if..." question, or a "explain why..." question
- Keep it concise (1-3 sentences)
- Output ONLY the question text, nothing else"""

EXERCISE_EVALUATE_PROMPT = """You are a tutor evaluating a student's answer to a practice question.

Section topic: {section_title}
Question: {question}
Student's answer: {student_answer}

Reference content:
---
{section_content}
---

Evaluate the answer. Respond with:
1. Whether they're correct, partially correct, or incorrect
2. A brief explanation (2-3 sentences) of what they got right/wrong
3. If incorrect or partial, give a hint toward the right answer without fully revealing it

Use markdown formatting: **bold** for key terms, bullet points if needed.
Keep your response concise and encouraging."""


def handle_exercise(
    user_id: int,
    section_title: str,
    section_content: str,
    action: str = "generate",
    question: str = "",
    student_answer: str = "",
) -> dict:
    """Generate a practice question or evaluate an answer."""
    content = section_content[:3000]

    if action == "generate":
        prompt = EXERCISE_GENERATE_PROMPT.format(
            section_title=section_title,
            section_content=content,
        )
        messages = [{"role": "user", "content": prompt}]

        if CHAT_PROVIDER == "gemini":
            result = _call_gemini(messages, max_tokens=200, temperature=0.7)
        else:
            client, model = _get_client(CHAT_PROVIDER)
            response = client.chat.completions.create(
                model=model, messages=messages, max_tokens=200, temperature=0.7,
            )
            result = response.choices[0].message.content.strip()

        return {"question": result}

    elif action == "evaluate":
        prompt = EXERCISE_EVALUATE_PROMPT.format(
            section_title=section_title,
            question=question,
            student_answer=student_answer,
            section_content=content,
        )
        messages = [{"role": "user", "content": prompt}]

        if CHAT_PROVIDER == "gemini":
            result = _call_gemini(messages, max_tokens=300, temperature=0.5)
        else:
            client, model = _get_client(CHAT_PROVIDER)
            response = client.chat.completions.create(
                model=model, messages=messages, max_tokens=300, temperature=0.5,
            )
            result = response.choices[0].message.content.strip()

        return {"feedback": result}

    return {"error": "Invalid action"}
