"""Plan steps 1 & 4: every Pedro conversation is recorded, reliably, with references
back to the original chat messages."""
from harness import (
    SessionLocal, chat, consolidate, episodes, job_rows, make_course, make_student,
    run_learning_jobs, stub_evaluator, student, course_namespace,
)

OPS = [
    {"title": "Queueing basics", "key_topics": ["Little's Law", "utilisation"]},
    {"title": "Linear programming", "key_topics": ["simplex method", "basic feasible solution"]},
]


def _types(eps):
    return [(e.store_specific or {}).get("episode_type") for e in eps]


# ── Step 1: completion + evaluation pipeline ─────────────────────────

def test_evaluator_accepts_open_questions():
    """Step 1. Evaluator output with an open question must not crash apply_evaluation."""
    import evaluator
    uid = make_student()
    make_course(uid, "Operations", OPS)
    evaluation = evaluator._normalize_evaluation(
        {"section_summary": "ok", "concepts": [], "golden_moments": [],
         "open_questions": ["Why can utilisation not exceed 1?"]}, [])
    evaluator.apply_evaluation(uid, "Operations", 0, "Queueing basics", evaluation)
    snap = student().active.snapshot(course_namespace(uid, "Operations"))
    assert any("utilisation" in q["text"] for q in snap["open_questions"]), snap


def test_section_completion_is_recorded_with_evaluation():
    """Step 1. [SECTION_COMPLETE] -> durable job -> completion + evaluation + open question in OMA."""
    uid = make_student()
    make_course(uid, "Operations", OPS)
    stub_evaluator({"little's law": "mastered"}, open_questions=["Does Little's Law need a stable queue?"])
    chat(uid, "L = lambda W, so 4 * 2 = 8 customers.", "Correct! [ANSWER_CORRECT: Little's Law] [SECTION_COMPLETE]",
         context_id="Operations", section_index=0)
    run_learning_jobs()
    types = _types(episodes(uid, "Operations"))
    assert "section_completed" in types, f"no completion episode; jobs={job_rows()}"
    assert "section_evaluation" in types, f"no evaluation episode; jobs={job_rows()}"
    snap = student().active.snapshot(course_namespace(uid, "Operations"))
    assert snap["open_questions"], "open question from evaluator was lost"


def test_completion_replay_does_not_duplicate():
    """Step 1. Running the same completion job twice must not duplicate memory."""
    import learning_jobs
    from database import LearningJob
    import json
    uid = make_student()
    make_course(uid, "Operations", OPS)
    stub_evaluator({"little's law": "mastered"})
    chat(uid, "8 customers", "Yes. [ANSWER_CORRECT: Little's Law] [SECTION_COMPLETE]",
         context_id="Operations", section_index=0)
    run_learning_jobs()
    with SessionLocal() as db:
        job = next(j for j in db.query(LearningJob).all() if json.loads(j.payload_json).get("user_id") == uid)
        job_id, payload = job.id, json.loads(job.payload_json)
    learning_jobs.project(job_id, payload)
    assert _types(episodes(uid, "Operations")).count("section_completed") == 1
    assert _types(episodes(uid, "Operations")).count("section_evaluation") == 1


def test_failed_completions_can_be_retried():
    """Step 1. After fixing a bug, failed completion jobs can be requeued and replayed."""
    import learning_jobs
    from database import LearningJob
    import json
    uid = make_student()
    make_course(uid, "Operations", OPS)
    stub_evaluator({"little's law": "mastered"})
    chat(uid, "8 customers", "Yes. [SECTION_COMPLETE]", context_id="Operations", section_index=0)
    with SessionLocal() as db:
        for j in db.query(LearningJob).all():
            if json.loads(j.payload_json).get("user_id") == uid:
                j.status, j.attempts, j.last_error = "failed", 8, "TypeError: importance"
        db.commit()
    assert hasattr(learning_jobs, "retry_failed_completions"), "need learning_jobs.retry_failed_completions(user_id=None)"
    learning_jobs.retry_failed_completions(user_id=uid)
    run_learning_jobs()
    assert "section_completed" in _types(episodes(uid, "Operations"))


# ── Step 4: every conversation updates the profile ──────────────────

def test_correct_answer_writes_episode():
    """Step 4. Correct answers are evidence too; mastery must be rebuildable from the log."""
    uid = make_student()
    make_course(uid, "Operations", OPS)
    chat(uid, "L = 8", "Exactly right. [ANSWER_CORRECT: Little's Law]", context_id="Operations", section_index=0)
    graded = [e for e in episodes(uid, "Operations")
              if (e.store_specific or {}).get("episode_type") == "exercise_attempt"]
    assert graded and graded[-1].store_specific["outcome"] == "success", _types(episodes(uid, "Operations"))


def test_every_lesson_turn_is_recorded_with_message_refs():
    """Step 4. Ungraded lesson Q&A is recorded and points at the original chat messages."""
    uid = make_student()
    make_course(uid, "Operations", OPS)
    res = chat(uid, "Why does the queue explode as utilisation approaches 1?",
               "Because waiting time grows like 1/(1-rho)...", context_id="Operations", section_index=0)
    eps = episodes(uid, "Operations")
    assert eps, "ungraded lesson turn left no trace in Student OMA"
    refs = (eps[-1].store_specific or {}).get("chat_message_ids") or []
    assert res["message_id"] in refs, f"episode must reference Pedro's message id; got {refs}"
    assert len(refs) == 2, "episode should reference both the student and Pedro message"


def test_folder_chat_is_recorded():
    """Step 4. Folder (ask-your-sources) chat updates the profile."""
    uid = make_student()
    make_course(uid, "Operations", OPS)
    chat(uid, "Can you explain the simplex method?", "Sure — the simplex method walks vertices...",
         context_type="folder", context_id="Operations")
    assert episodes(uid, "Operations"), "folder chat left no trace in Student OMA"


def test_general_chat_is_recorded():
    """Step 4. General chat with no course still updates the student's profile."""
    uid = make_student()
    chat(uid, "I have my exams in three weeks and I'm stressed about stats.",
         "That's a lot — let's make a plan...", context_type="global")
    assert episodes(uid), "general chat left no trace in Student OMA"


def test_general_chat_about_a_course_lands_in_that_course():
    """Step 4. General chat that is clearly about a course is filed under that course."""
    uid = make_student()
    make_course(uid, "Operations", OPS)
    chat(uid, "In Operations, is the simplex method always finite?",
         "With anti-cycling rules, yes...", context_type="global")
    assert episodes(uid, "Operations"), "course-specific general chat was not filed under the course"


def test_behaviour_signals_become_patterns():
    """Step 4. Repeated behaviour (asking for examples) becomes a course pattern."""
    uid = make_student()
    make_course(uid, "Operations", OPS)
    for q in ("Can you give me an example of Little's Law?",
              "Show me a worked example for utilisation",
              "Another example please, with numbers",
              "Could you give an example of a basic feasible solution?"):
        chat(uid, q, "Here's an example: ...", context_type="folder", context_id="Operations")
    consolidate(uid, "Operations")
    kinds = {(p.store_specific or {}).get("pattern_type") for p in student().patterns.all(course_namespace(uid, "Operations"))}
    assert "prefers_examples" in kinds, kinds
