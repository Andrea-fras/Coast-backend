"""Plan step 1/3: comprehension gating — students advance only after Pedro verifies
understanding, and the evidence behind that verification lands in Student OMA."""
from harness import (
    chat, episodes, lesson, make_course, make_student, mastery, run_learning_jobs, stub_evaluator,
)

OPS = [
    {"title": "Queueing basics", "key_topics": ["Little's Law"]},
    {"title": "Linear programming", "key_topics": ["simplex method"]},
]


def test_cannot_advance_without_verification():
    uid = make_student()
    make_course(uid, "Operations", OPS)
    chat(uid, "next", "You still need to answer the check question first.", context_id="Operations", section_index=0)
    assert not lesson.can_advance_from_section(uid, "Operations", 0)


def test_section_complete_unlocks_advance():
    uid = make_student()
    make_course(uid, "Operations", OPS)
    res = chat(uid, "L = 8", "Correct. [ANSWER_CORRECT: Little's Law] [SECTION_COMPLETE]",
               context_id="Operations", section_index=0)
    assert res.get("section_verified") is True
    assert lesson.can_advance_from_section(uid, "Operations", 0)


def test_gate_answers_are_kept_as_evidence():
    """The verification answers the student gave are stored with their concept."""
    uid = make_student()
    ids = make_course(uid, "Operations", OPS)
    chat(uid, "L = lambda * W = 8", "Correct. [ANSWER_CORRECT: Little's Law]", context_id="Operations", section_index=0)
    graded = [e.store_specific for e in episodes(uid, "Operations")
              if (e.store_specific or {}).get("episode_type") == "exercise_attempt"]
    assert graded, "verification answer left no episode"
    assert ids["little's law"] in graded[-1].get("concept_ids", []), graded[-1]


def test_verified_section_updates_mastery_from_evaluator():
    """After the gate, the evaluator's verdict (from the whole transcript) sets mastery."""
    uid = make_student()
    ids = make_course(uid, "Operations", OPS)
    stub_evaluator({"little's law": "mastered"})
    chat(uid, "L = 8", "Correct. [ANSWER_CORRECT: Little's Law] [SECTION_COMPLETE]", context_id="Operations", section_index=0)
    run_learning_jobs()
    row = mastery(uid, "Operations", ids["little's law"])
    assert row and row.get("last_eval_state") == "mastered", row


def test_evaluator_can_disagree_with_the_gate():
    """If the transcript shows the student is still shaky, memory says so even though they advanced."""
    uid = make_student()
    ids = make_course(uid, "Operations", OPS)
    stub_evaluator({"little's law": "struggling"})
    chat(uid, "L = 8?", "Correct. [ANSWER_CORRECT: Little's Law] [SECTION_COMPLETE]", context_id="Operations", section_index=0)
    run_learning_jobs()
    row = mastery(uid, "Operations", ids["little's law"])
    assert row and row.get("last_eval_state") == "struggling", row
    assert row["mastery_tier"] in ("red", "orange"), row
