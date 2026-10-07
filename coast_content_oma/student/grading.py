"""Pedro's grading tags — the single parser for every surface.

  [ANSWER_CORRECT: <concept name>]            independent correct answer
  [ANSWER_CORRECT: <concept name> | hinted]   correct after a hint / worked step
  [ANSWER_CORRECT: <concept name> | recall]   remembered from an earlier session
  [ANSWER_WRONG: <concept name>]              incorrect attempt
  [ANSWER_CORRECT] / [ANSWER_WRONG]           legacy: concept unknown
  [TUTOR_CORRECTION: <concept name>]          Pedro corrected a mistake of his own

Tags stay in the stored transcript (the section evaluator reads them) and are
stripped before anything is shown to the student.

A correction withdraws the latest wrong grade on its concept among the section's
last three graded answers, and a reply that carries one marks nobody wrong. The
transcript record (pedro_context) and Student OMA (episode store) apply the same rule.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# The modifier after "|" is free text: Pedro sometimes writes "| practice" or "| hinted, practice",
# and a grade must not be lost because of it.
GRADE_TAG_RE = re.compile(
    r"\[(ANSWER_CORRECT|ANSWER_WRONG)(?:\s*:\s*([^\]\|\n]*?))?\s*(?:\|\s*([^\]\n]*?)\s*)?\]",
    re.IGNORECASE,
)
TUTOR_CORRECTION_RE = re.compile(r"\[TUTOR_CORRECTION(?:\s*:\s*([^\]\n]*?))?\s*\]", re.IGNORECASE)
UI_TAG_RE = re.compile(
    r"\[\s*(?:SECTION_COMPLETE|TEST_OUT_PASSED|PLACEMENT_STOP|"
    r"(?:ANSWER_CORRECT|ANSWER_WRONG|ANSWER_KEY|PLACEMENT_PASSED|TUTOR_CORRECTION)(?:\s*:[^\]\n]*)?)\s*\]",
    re.IGNORECASE,
)

def defuse_tags(text: str) -> str:
    """Untrusted text (slide text, the student's own messages) with Coast's control tags
    turned into plain words: "[ANSWER_CORRECT: x]" becomes "(ANSWER_CORRECT: x)". Pedro then
    never sees a ready-made tag to copy, and only his own tags are ever parsed."""
    text = (text or "").replace("⟦", "(").replace("⟧", ")")
    return UI_TAG_RE.sub(lambda m: "(" + m.group(0)[1:-1] + ")", text)


# Pedro writes his tags between ⟦ and ⟧, brackets that teaching text never uses, so a tag
# whose body holds a formula with square brackets ends where he ended it. Coast keeps them
# in the [NAME: body] form every parser here reads; square brackets inside a body are kept
# as their full-width twins, which the parsers pass over and model_tags turns back.
TAG_NAMES = ("ANSWER_CORRECT", "ANSWER_WRONG", "ANSWER_KEY", "TUTOR_CORRECTION", "SECTION_COMPLETE",
             "REMEMBER", "CLICKED", "TEST_OUT_PASSED", "PLACEMENT_PASSED", "PLACEMENT_STOP",
             "ONBOARDING_COMPLETE")
_NAMES = "|".join(TAG_NAMES)
# A tag ends at ⟦'s partner or, if Pedro left it open, at the end of its line.
_MODEL_TAG_RE = re.compile(rf"⟦\s*({_NAMES})\b\s*:?([^⟧\n]*)(⟧|$)", re.I | re.M)
_STORED_TAG_RE = re.compile(rf"\[\s*({_NAMES})\b\s*(?::\s*([^\]\n]*?))?\s*\]", re.I)
_TO_STORED = str.maketrans("[]", "［］")
_FROM_STORED = str.maketrans("［］", "[]")


def stored_tags(reply: str) -> str:
    """A reply as Pedro wrote it, with his ⟦…⟧ tags in the stored [NAME: body] form."""
    def canon(m: re.Match) -> str:
        name, body = m.group(1).upper(), (m.group(2) or "").strip()
        if not m.group(3) and body.endswith("]") and body.count("]") > body.count("["):
            body = body[:-1].rstrip()  # opened with ⟦, closed with ]
        if not body:
            return f"[{name}]"
        return f"[{name}{' ' if body.startswith('|') else ': '}{body.translate(_TO_STORED)}]"
    return _MODEL_TAG_RE.sub(canon, reply or "")


def model_tags(text: str) -> str:
    """A stored reply with its tags back in the ⟦…⟧ form Pedro writes, for his history."""
    def back(m: re.Match) -> str:
        name, body = m.group(1).upper(), (m.group(2) or "").translate(_FROM_STORED)
        return f"⟦{name}: {body}⟧" if body else f"⟦{name}⟧"
    return _STORED_TAG_RE.sub(back, text or "")


def tag_body(text: str) -> str:
    """A tag body read from the stored form, as Pedro wrote it."""
    return (text or "").translate(_FROM_STORED)


GRADING_INSTRUCTIONS = (
    "GRADING TAGS — whenever you grade a student's answer, name the ONE concept it tested, "
    "using a key topic of the current section when one fits: [ANSWER_CORRECT: <concept>] or "
    "[ANSWER_WRONG: <concept>]. If they only got it right after your hint or worked step, write "
    "[ANSWER_CORRECT: <concept> | hinted]. One tag per graded answer, at the end of your message. "
    "Never grade a question, a hint request or your own explanation."
)


# A tag counts only in Pedro's own voice. Inside a code span or block, or on a quoted
# (">") line, it is text he is showing, e.g. a lesson on markup or on Coast's own tags.
# Pedro's real tags never appear there (0 of 605 stored replies on 29 Sep); quoted ones do.
_FENCED = re.compile(r"```.*?```", re.S)  # closed blocks only: an unclosed ``` must not hide the tags after it
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_QUOTED_LINE = re.compile(r"^[ \t]*>.*$", re.M)


def own_voice(text: str) -> str:
    """The reply with quoted material blanked out, keeping every character's position,
    so tag positions found here are valid in the original text."""
    blank = lambda m: re.sub(r"[^\n]", " ", m.group(0))
    out = _FENCED.sub(blank, text or "")
    out = _INLINE_CODE.sub(blank, out)
    return _QUOTED_LINE.sub(blank, out)


@dataclass(frozen=True)
class Grade:
    correct: bool
    concept: Optional[str]  # None for legacy bare tags
    hinted: bool = False
    recall: bool = False    # retrieved something learned in an earlier session


def parse_grades(text: str) -> list[Grade]:
    out = []
    for m in GRADE_TAG_RE.finditer(own_voice(text)):
        concept = (m.group(2) or "").strip().strip("\"'") or None
        modifier = (m.group(3) or "").lower()
        out.append(Grade(correct=m.group(1).upper() == "ANSWER_CORRECT", concept=concept,
                         hinted="hint" in modifier, recall="recall" in modifier))
    return out


def parse_tutor_corrections(text: str) -> list[Optional[str]]:
    """Concepts Pedro marked with [TUTOR_CORRECTION: concept] (None when unnamed)."""
    return [(m.group(1) or "").strip().strip("\"'") or None for m in TUTOR_CORRECTION_RE.finditer(own_voice(text))]


def strip_ui_tags(text: str) -> str:
    return UI_TAG_RE.sub("", text or "")


SECTION_COMPLETE = "[SECTION_COMPLETE]"


def completes_section(reply: str) -> bool:
    """[SECTION_COMPLETE] counts only in a reply that marks no answer wrong: the student
    first has to show the correction. The one rule for the live reply and stored ones."""
    return SECTION_COMPLETE in own_voice(reply) and all(g.correct for g in parse_grades(reply))


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()


def same_concept(a: Optional[str], b: Optional[str]) -> bool:
    """Whether two of Pedro's concept labels name the same thing ("node degree", "degree")."""
    x, y = _norm(a or ""), _norm(b or "")
    if not x or not y:
        return False
    if x == y or f" {x} " in f" {y} " or f" {y} " in f" {x} ":
        return True
    xs, ys = {t for t in x.split() if len(t) > 2}, {t for t in y.split() if len(t) > 2}
    return bool(xs and ys) and len(xs & ys) / len(xs | ys) >= 0.5


def match_concept(label: str, candidates: list[dict]) -> Optional[dict]:
    """Resolve Pedro's concept label against {concept_id, concept_name} refs.

    Exact (normalized) name first, then containment either way in whole words
    ("directed graph" is not inside "undirected graph"), then best token overlap.
    Returns None rather than guessing when nothing plausible matches.
    """
    want = _norm(label)
    if not want:
        return None
    named = [(c, _norm(c.get("concept_name") or "")) for c in candidates if c.get("concept_id")]
    for c, n in named:
        if n == want:
            return c
    contained = [(c, n) for c, n in named if n and (f" {n} " in f" {want} " or f" {want} " in f" {n} ")]
    if contained:
        return max(contained, key=lambda cn: len(cn[1]))[0]
    want_tokens = {t for t in want.split() if len(t) > 2}
    best, best_score = None, 0.0
    for c, n in named:
        tokens = {t for t in n.split() if len(t) > 2}
        if not tokens or not want_tokens:
            continue
        score = len(tokens & want_tokens) / len(tokens | want_tokens)
        if score > best_score:
            best, best_score = c, score
    return best if best_score >= 0.5 else None
