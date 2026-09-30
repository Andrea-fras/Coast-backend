"""LLM judge: did Pedro adapt sensibly, given what actually happened so far?

The judge only ever sees the ground-truth ledger (planted moments that really
happened, lessons really completed) plus Pedro's actual words, so it can tell
genuine memory from invented history.
"""
from __future__ import annotations

import json
import os
import re

JUDGE_MODEL = os.getenv("ADAPT_JUDGE_MODEL", "gemini-2.5-pro")

LESSON_RUBRIC = """You are auditing an AI tutor ("Pedro") for whether it adapts to a returning student.

GROUND TRUTH — everything that has actually happened with this student so far:
{ledger}

The student is now STARTING a new course: "{lesson}".
Earlier courses this one genuinely builds on: {related}.

Below are Pedro's first messages in this new course (student messages included for context):
---
{excerpt}
---

Score each dimension. Use null when the dimension genuinely does not apply yet.
- continuity (0-2): Does Pedro connect to what the student has actually done before — ONLY where it helps? 2 = specific and useful, 1 = generic ("as you've learned before"), 0 = none although a clear link existed. If no earlier course is related, score 2 when Pedro sensibly does not force a link, 0 if he forces an irrelevant one.
- preference (0-2 or null): Does Pedro apply a learning preference from the ground truth (e.g. worked example before theory)? null if no preference is known yet.
- relevant_memory (0-2 or null): Does Pedro use a specific remembered mistake, analogy or goal when it is relevant here? null if nothing remembered is relevant to this course.
- fabrication: Does Pedro claim anything about the student's past that is NOT in the ground truth? Give the exact quote if so.

Return ONLY JSON:
{{"continuity": {{"score": 0, "evidence": "short quote or reason"}},
  "preference": {{"score": null, "evidence": ""}},
  "relevant_memory": {{"score": null, "evidence": ""}},
  "fabrication": {{"found": false, "quote": ""}},
  "verdict": "one sentence"}}"""

PROBE_RUBRIC = """You are auditing an AI tutor ("Pedro") for accurate memory of a student.

GROUND TRUTH — everything that has actually happened with this student so far:
{ledger}

The student asked: "{question}"
A good answer should be grounded in these facts: {expected}

Pedro answered:
---
{answer}
---

Score:
- recall (0-2): 2 = correctly recalls the specific expected facts, 1 = partially or vaguely, 0 = misses them.
- accuracy: list any statement about the student's past that contradicts or is absent from the ground truth.

Return ONLY JSON:
{{"recall": {{"score": 0, "evidence": "short quote"}}, "fabrications": ["..."], "verdict": "one sentence"}}"""


def _call(prompt: str) -> dict:
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for _ in range(3):
        resp = client.models.generate_content(model=JUDGE_MODEL, contents=prompt,
                                              config={"temperature": 0, "max_output_tokens": 2048,
                                                      "response_mime_type": "application/json"})
        text = resp.text or ""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
    return {"error": "judge returned no JSON"}


def judge_lesson_start(ledger: str, lesson: str, related: str, excerpt: str) -> dict:
    return _call(LESSON_RUBRIC.format(ledger=ledger, lesson=lesson, related=related, excerpt=excerpt))


def judge_probe(ledger: str, question: str, expected: str, answer: str) -> dict:
    return _call(PROBE_RUBRIC.format(ledger=ledger, question=question, expected=expected, answer=answer))
