"""Plan steps 2 & 3: the profile Pedro sees is accurate — no inflated mastery,
no stale struggles, no junk traits, nothing invented."""
from harness import (
    CONCEPT_ID_RE, chat, consolidate, course_block, global_block, identity_namespace,
    identity_traits, intro_block, make_course, make_student, mastery, mastery_rows, seed_answers, student,
)

OPS = [
    {"title": "Queueing basics", "key_topics": ["Little's Law", "utilisation", "holding time", "arrival rate"]},
    {"title": "Linear programming", "key_topics": ["simplex method", "basic feasible solution"]},
]


def _grade(uid, folder, section, concept, correct, hinted=False):
    tag = "ANSWER_CORRECT" if correct else "ANSWER_WRONG"
    suffix = " | hinted" if hinted else ""
    chat(uid, f"my answer about {concept}", f"{'Right' if correct else 'Not quite'}. [{tag}: {concept}{suffix}]",
         context_id=folder, section_index=section)


# ── Step 3: mastery reflects what was actually demonstrated ─────────

def test_one_correct_answer_is_not_mastery():
    """Step 3. A single correct answer is weak evidence, not 'mastered'/green."""
    uid = make_student()
    ids = make_course(uid, "Operations", OPS)
    _grade(uid, "Operations", 0, "Little's Law", True)
    row = mastery(uid, "Operations", ids["little's law"])
    assert row, "correct answer produced no mastery evidence"
    assert row["mastery_tier"] != "green", row
    assert row["mastery_score"] < 0.75, row


def test_named_grading_tag_credits_only_that_concept():
    """Step 3. [ANSWER_CORRECT: X] credits X only — not every concept in the section."""
    uid = make_student()
    make_course(uid, "Operations", OPS)
    _grade(uid, "Operations", 0, "Little's Law", True)
    rows = mastery_rows(uid, "Operations")
    assert "little's law" in rows, rows.keys()
    others = set(rows) - {"little's law"}
    assert not others, f"credit leaked to {others}"


def test_bare_grading_tag_does_not_credit_whole_section():
    """Step 3. A legacy bare tag in a multi-concept section must not credit all of them."""
    uid = make_student()
    make_course(uid, "Operations", OPS)
    chat(uid, "L = 8", "Right! [ANSWER_CORRECT]", context_id="Operations", section_index=0)
    credited = [n for n, r in mastery_rows(uid, "Operations").items() if r.get("successes")]
    assert len(credited) <= 1, f"one answer credited {credited}"


def test_wrong_answer_counts_as_evidence():
    """Step 3. Wrong answers are recorded as evidence on the tested concept."""
    uid = make_student()
    ids = make_course(uid, "Operations", OPS)
    _grade(uid, "Operations", 0, "utilisation", False)
    row = mastery(uid, "Operations", ids["utilisation"])
    assert row and row.get("struggles") == 1, row


def test_hinted_success_weighs_less_than_independent():
    """Step 3. Success after hints is weaker evidence than an independent answer."""
    a, b = make_student("Ind"), make_student("Hint")
    ids_a = make_course(a, "Operations", OPS)
    ids_b = make_course(b, "Operations", OPS)
    for _ in range(3):
        _grade(a, "Operations", 0, "Little's Law", True)
        _grade(b, "Operations", 0, "Little's Law", True, hinted=True)
    ra, rb = mastery(a, "Operations", ids_a["little's law"]), mastery(b, "Operations", ids_b["little's law"])
    assert ra and rb and rb["mastery_score"] < ra["mastery_score"], (ra, rb)


def test_recovery_clears_struggling():
    """Step 2. Two early mistakes followed by consistent success is not 'struggling'."""
    uid = make_student()
    ids = make_course(uid, "Operations", OPS)
    seed_answers(uid, "Operations", "holding time", ids["holding time"], [False, False, True, True, True, True, True])
    block = course_block(uid, "Operations", query="holding time")
    assert "Repeated difficulty: holding time" not in block, block


def test_mostly_right_is_not_repeated_difficulty():
    """Step 2. 3 wrong / 9 right (real case from your DB) must not be shown as a difficulty."""
    uid = make_student()
    ids = make_course(uid, "Operations", OPS)
    seed_answers(uid, "Operations", "basic feasible solution", ids["basic feasible solution"],
                 [False] * 3 + [True] * 9, section_index=1)
    block = course_block(uid, "Operations")
    assert "Repeated difficulty: basic feasible solution" not in block, block


def test_recent_repeated_failure_is_still_flagged():
    """Step 2. Guard against over-correcting: genuinely stuck students are still flagged."""
    uid = make_student()
    ids = make_course(uid, "Operations", OPS)
    seed_answers(uid, "Operations", "arrival rate", ids["arrival rate"], [True, False, False, False])
    block = course_block(uid, "Operations", query="arrival rate")
    assert "arrival rate" in block.lower(), block


# ── Step 2: identity traits are real, readable, cross-course ────────

def test_single_course_strength_is_not_an_identity_trait():
    """Step 2. 'Consistently strong' requires evidence from 2+ courses."""
    uid = make_student()
    ids = make_course(uid, "Operations", OPS)
    seed_answers(uid, "Operations", "simplex method", ids["simplex method"], [True] * 5, section_index=1)
    consolidate(uid, "Operations")
    assert not [t for t in identity_traits(uid) if "strong" in t.lower()], identity_traits(uid)


def test_identity_traits_never_contain_concept_ids():
    uid = make_student()
    ids = make_course(uid, "Operations", OPS)
    seed_answers(uid, "Operations", "simplex method", ids["simplex method"], [True] * 5, section_index=1)
    consolidate(uid, "Operations")
    bad = [t for t in identity_traits(uid) if CONCEPT_ID_RE.search(t)]
    assert not bad, bad
    assert not [t for t in identity_traits(uid) if "con_" in t], identity_traits(uid)


def test_cross_course_strength_uses_concept_names():
    """Step 2. The same topic mastered in two courses becomes a named identity trait."""
    uid = make_student()
    for folder in ("Operations", "Machine Learning"):
        ids = make_course(uid, folder, [{"title": "Foundations", "key_topics": ["linear algebra"]}])
        seed_answers(uid, folder, "linear algebra", ids["linear algebra"], [True] * 5)
        consolidate(uid, folder)
    assert any("linear algebra" in t.lower() for t in identity_traits(uid)), identity_traits(uid)


def test_prompt_blocks_hide_malformed_traits():
    """Step 2. Even if junk already exists in the DB, Pedro never sees raw concept ids."""
    uid = make_student()
    ids = make_course(uid, "Operations", OPS)
    _grade(uid, "Operations", 0, "Little's Law", True)
    junk = f"Consistently strong in: {ids['simplex method']}"
    student().identity.upsert_trait(identity_namespace(uid), "general_strength", junk, confidence=0.9,
                                    evidence_courses=["x"], dedupe_key="junk")
    for name, block in (("course", course_block(uid, "Operations")), ("global", global_block(uid)),
                        ("intro", intro_block(uid, "Operations"))):
        assert not CONCEPT_ID_RE.search(block or ""), f"{name} block leaks ids:\n{block}"


def test_new_student_gets_no_invented_history():
    """Step 2. No evidence -> no profile block, nothing for Pedro to embellish."""
    uid = make_student()
    make_course(uid, "Operations", OPS)
    assert course_block(uid, "Operations") == ""
    assert global_block(uid) == ""
