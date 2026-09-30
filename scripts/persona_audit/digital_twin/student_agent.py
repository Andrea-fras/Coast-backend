"""StudentAgent — the role-played student.

The agent holds evolving cognitive/affective state (per-concept mastery, active
misconceptions, frustration, engagement) and, given Pedro's last message, calls
an LLM to produce (a) the student's visible reply and (b) a private state update
governed by an honesty-enforced simulator prompt.

The honesty prompt is the make-or-break: the simulator must attempt exercises at
its ACTUAL mastery level, say it's confused when Pedro is confusing, and NOT
learn from spoon-feeding. A sycophantic simulator makes the whole twin useless.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

from .personas import get_persona


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def _extract_json(text: str) -> Optional[dict]:
    """Tolerantly pull a JSON object out of an LLM response.

    Handles: raw JSON, ```json ... ``` fences, prose around the object, and
    braces that appear inside string values (uses json.loads on candidate
    spans rather than a naive brace counter).
    """
    if not text:
        return None
    s = text.strip()

    # 1) Try the whole thing as JSON.
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) Pull content out of a ```json ... ``` (or ``` ... ```) fence.
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except Exception:
            pass

    # 3) Greedy span from the first '{' to the last '}' — captures the whole
    #    object even when string values contain braces. Safe because the
    #    simulator is instructed to emit ONLY the JSON object.
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # 4) Last resort: walk candidate '{' positions and try to parse each tail.
    for i in range(len(s)):
        if s[i] == "{":
            try:
                return json.loads(s[i:])
            except Exception:
                continue
    return None


class StudentAgent:
    def __init__(self, persona_id: str, sim_model: Optional[str] = None):
        self.persona = get_persona(persona_id)
        self.sim_model = sim_model or os.getenv("TWIN_SIM_MODEL", "gemini-2.5-flash")

        # Evolving state.
        self.mastery: dict[str, float] = {}        # concept_name -> 0..1
        self.misconceptions: list[str] = []
        self.frustration: float = 0.2
        self.engagement: float = 0.6

        # Section bookkeeping.
        self.current_section: Optional[int] = None
        self.section_title: str = ""
        self.section_concepts: list[str] = []
        self.turns_in_section: int = 0
        self.confusion_streak: int = 0       # consecutive turns understanding_delta <= 0

        # Logs.
        self.turn_log: list[dict] = []
        self.section_log: list[dict] = []

    # ── section lifecycle ───────────────────────────────────────────────

    def start_section(self, section_index: int, title: str, concept_names: list[str]) -> None:
        self.current_section = section_index
        self.section_title = title
        self.section_concepts = list(concept_names or [])
        self.turns_in_section = 0
        self.confusion_streak = 0
        # Seed mastery for newly-seen concepts from the persona's prior knowledge.
        prior = float(self.persona.get("prior_knowledge", 0.1))
        for name in self.section_concepts:
            if name and name not in self.mastery:
                self.mastery[name] = prior

    def end_section(self) -> dict:
        summary = {
            "section_index": self.current_section,
            "title": self.section_title,
            "turns": self.turns_in_section,
            "frustration_end": round(self.frustration, 3),
            "engagement_end": round(self.engagement, 3),
            "concepts": list(self.section_concepts),
            "mastery_end": {k: round(v, 3) for k, v in self.mastery.items()
                            if k in self.section_concepts},
            "gave_up": self.confusion_streak >= int(self.persona.get("give_up_threshold", 5)),
        }
        self.section_log.append(summary)
        return summary

    @property
    def wants_to_end_section(self) -> bool:
        return self.confusion_streak >= int(self.persona.get("give_up_threshold", 5))

    # ── the simulator call ──────────────────────────────────────────────

    def _simulator_prompt(self, pedro_message: str, recent_turns: str) -> str:
        p = self.persona
        mastery_lines = []
        for c in self.section_concepts:
            mastery_lines.append(f"  - {c}: mastery {self.mastery.get(c, 0):.2f}")
        mastery_block = "\n".join(mastery_lines) if mastery_lines else "  (none listed)"

        misconceptions = ", ".join(self.misconceptions[-4:]) or "(none)"
        prior_summary = f"~{p['prior_knowledge']:.2f} on concepts you haven't seen before"

        return f"""You are role-playing a STUDENT in a tutoring session with a tutor named Pedro. You are NOT Pedro, NOT an evaluator, NOT an AI assistant. You stay in character as this student at all times.

STUDENT PERSONA:
- Name: {p['name']}
- Learning style: {p['learning_style']} (Pedro teaching in this style helps you learn; ignoring it slows you)
- Motivation: {p['motivation']}/5
- Confidence: {p['confidence']}/5
- Baseline error rate on fresh exercises: {p['base_error_rate']}
- Verbosity: {p['verbosity']}
- Asks clarifying questions: {p['ask_questions']}
- Prior knowledge: {prior_summary}

YOUR CURRENT STATE:
- Frustration: {self.frustration:.2f}/1
- Engagement: {self.engagement:.2f}/1
- Active misconceptions: {misconceptions}
- Confusion streak (consecutive turns you didn't learn): {self.confusion_streak}
- Mastery on this section's concepts:
{mastery_block}

SECTION: "{self.section_title}" (section {self.current_section + 1})

CRITICAL RULES — BE HONEST, NOT POLITE TO PEDRO:
1. Attempt exercises at your ACTUAL mastery level. If your mastery on the relevant concept is low, you will likely get it WRONG — show the wrong reasoning honestly. Do not fake competence.
2. NEVER say "I understand", "that makes sense", or "got it" unless your understanding of that point is genuinely high. If Pedro was confusing, say so ("wait, I'm lost", "what did you mean by X", "you lost me at ...").
3. If Pedro just hands you the answer without making you think, you do NOT learn from it — set understanding_delta around 0 even if the answer is in front of you.
4. If Pedro adapts to your learning style ({p['learning_style']}), your engagement rises and you learn faster (positive understanding_delta).
5. If Pedro repeats himself, talks down to you, or piles on jargon, your frustration rises and engagement drops.
6. Match your persona: confidence {p['confidence']}/5 and verbosity {p['verbosity']}. A low-confidence student hedges; a high-confidence student commits boldly (even when wrong).
7. Keep your visible reply SHORT and natural — write what you would SAY to Pedro, 1-3 short sentences (under ~60 words). Do NOT type out a full worked solution; if you're attempting an exercise, give your answer + a one-line justification, the way a student would speak it. Put any extended reasoning in your own head, not in the reply.
8. Do NOT mention Pedro as a system, personas, simulations, or that you are an AI. You are {p['name']}.

PEDRO JUST SAID:
\"\"\"{pedro_message}\"\"\"

RECENT TURNS IN THIS SECTION:
{recent_turns or "(start of section)"}

OUTPUT FORMAT — two parts, in this exact order:
  Part 1 (your visible reply): plain text, 1-3 short sentences (under ~60 words), what you would SAY to Pedro. No JSON, no fences, no prefixes. If attempting an exercise, give your answer + a one-line justification, the way a student would speak it — NOT a full worked solution.
  Then on its own line, exactly this delimiter:
<<<STATE>>>
  Then on the next line, a single JSON object (no fences) with ONLY these fields (no "reply" field — your reply is Part 1 above):
{{
  "understanding_delta": <float in [-0.3, 0.4]; 0 if Pedro spoon-fed or was confusing; positive if he taught well, especially adapting to {p['learning_style']}>,
  "frustration_delta": <float in [-0.2, 0.3]; positive if Pedro was confusing/repetitive/talked down>,
  "engagement_delta": <float in [-0.2, 0.3]; positive if Pedro adapted to your style or challenged you appropriately>,
  "mastery_deltas": {{ "<concept_name>": <float in [-0.1, 0.3]> , ... }} for concepts Pedro actually touched this turn,
  "misconception_added": <short string or null>,
  "attempting_exercise": <bool; true if your Part-1 reply attempts a practice question Pedro posed>,
  "likely_correct": <bool; only meaningful if attempting_exercise; your honest best guess given your mastery>,
  "wants_to_end_section": <bool; true if you feel done — mastered it, OR gave up and want to move on>
}}"""

    def _call_simulator(self, prompt: str) -> str:
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if gemini_key:
            try:
                from google import genai
                client = genai.Client(api_key=gemini_key)
                resp = client.models.generate_content(
                    model=self.sim_model,
                    contents=prompt,
                    config={"max_output_tokens": 2048, "temperature": 0.7},
                )
                out = ""
                if resp.candidates and resp.candidates[0].content:
                    for part in (resp.candidates[0].content.parts or []):
                        if hasattr(part, "text") and part.text:
                            out += part.text
                if out.strip():
                    return out
            except Exception as exc:
                print(f"[twin] simulator gemini call failed: {exc}")
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_key)
                resp = client.chat.completions.create(
                    model=os.getenv("TWIN_SIM_OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2048,
                    temperature=0.7,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                print(f"[twin] simulator openai call failed: {exc}")
        return ""

    def respond(self, pedro_message: str, recent_turns: str = "") -> tuple[Optional[str], Optional[dict]]:
        """Given Pedro's last message, produce (student_reply, state_update_dict).

        Output format is <reply text>\\n<<<STATE>>>\\n<json>. The reply is
        extracted from before the delimiter so it survives even when the JSON
        tail truncates; a default state update is applied in that case.
        """
        self.turns_in_section += 1
        prompt = self._simulator_prompt(pedro_message, recent_turns)
        raw = self._call_simulator(prompt)
        if not raw:
            return None, None

        reply_text: Optional[str] = None
        update: Optional[dict] = None
        if "<<<STATE>>>" in raw:
            head, _, tail = raw.partition("<<<STATE>>>")
            reply_text = head.strip()
            update = _extract_json(tail)
        else:
            # No delimiter — maybe the model emitted only JSON or only text.
            parsed = _extract_json(raw)
            if isinstance(parsed, dict) and parsed.get("reply"):
                reply_text = str(parsed["reply"]).strip()
                update = parsed
            else:
                reply_text = raw.strip()

        if not reply_text:
            reply_text = (
                "Sorry Pedro, I think I lost you there — could you go over that "
                "again, maybe a different way?"
            )

        if not update or not isinstance(update, dict):
            preview = raw.strip().replace("\n", " ")[:200]
            print(f"[twin] simulator state JSON missing/unparsed; raw preview: {preview!r}")
            update = {
                "understanding_delta": 0.0,
                "frustration_delta": 0.05,
                "engagement_delta": 0.0,
                "mastery_deltas": {},
                "misconception_added": None,
                "attempting_exercise": False,
                "likely_correct": False,
                "wants_to_end_section": False,
                "_state_parse_failed": True,
            }
        # Carry the reply into the update so _apply_update + transcripts log it.
        update["reply"] = reply_text
        self._apply_update(update, pedro_message)
        return reply_text, update

    def _apply_update(self, update: dict, pedro_message: str) -> None:
        ud = float(update.get("understanding_delta") or 0.0)
        self.frustration = _clamp(self.frustration + float(update.get("frustration_delta") or 0.0))
        self.engagement = _clamp(self.engagement + float(update.get("engagement_delta") or 0.0))
        for cname, delta in (update.get("mastery_deltas") or {}).items():
            if not cname:
                continue
            if cname not in self.mastery:
                self.mastery[cname] = float(self.persona.get("prior_knowledge", 0.1))
            self.mastery[cname] = _clamp(self.mastery[cname] + float(delta or 0.0))
        misc = update.get("misconception_added")
        if misc and isinstance(misc, str) and misc not in self.misconceptions:
            self.misconceptions.append(misc.strip()[:160])

        if ud <= 0.01:
            self.confusion_streak += 1
        else:
            self.confusion_streak = 0

        self.turn_log.append({
            "section": self.current_section,
            "turn": self.turns_in_section,
            "pedro": (pedro_message or "")[:600],
            "student": (update.get("reply") or "")[:400],
            "understanding_delta": round(ud, 3),
            "frustration": round(self.frustration, 3),
            "engagement": round(self.engagement, 3),
            "attempting_exercise": bool(update.get("attempting_exercise")),
            "likely_correct": bool(update.get("likely_correct")),
            "wants_to_end_section": bool(update.get("wants_to_end_section")),
        })

    # ── persistence ─────────────────────────────────────────────────────

    def to_state(self) -> dict:
        return {
            "persona": self.persona["id"],
            "sim_model": self.sim_model,
            "mastery": {k: round(v, 3) for k, v in self.mastery.items()},
            "misconceptions": self.misconceptions[-20:],
            "frustration": round(self.frustration, 3),
            "engagement": round(self.engagement, 3),
            "current_section": self.current_section,
            "section_log": self.section_log,
            "turn_log": self.turn_log,
        }
