"""Plan step 5: Pedro can find exactly what happened in any earlier course/lecture —
what was taught, what the student wrote, when — and stays quiet when there's no evidence."""
from datetime import datetime, timezone

from harness import (
    SessionLocal, ChatMessage, chat, course_block, make_course, make_student,
    run_learning_jobs, stub_evaluator, student, course_namespace,
)
from coast_content_oma.stores.db import connect_db

OPS = [
    {"title": "Queueing basics", "key_topics": ["Little's Law", "utilisation"]},
    {"title": "Linear programming", "key_topics": ["simplex method"]},
    {"title": "Network flows", "key_topics": ["max flow"]},
]
QUOTE = "I thought rho could go above 1 as long as the server is really fast"


def _filler(uid, folder, section, n):
    """Lots of later conversation so the early lecture is not 'recent'."""
    with SessionLocal() as db:
        for i in range(n):
            db.add(ChatMessage(user_id=uid, conversation_id=f"fill{section}", role="user" if i % 2 == 0 else "pedro",
                               content=f"filler turn {i} about section {section}", context_type="lesson",
                               context_id=folder, section_index=section))
        db.commit()


def _backdate(uid, folder, section, when: datetime):
    with SessionLocal() as db:
        for m in db.query(ChatMessage).filter_by(user_id=uid, context_id=folder, section_index=section):
            m.created_at = when
        db.commit()
    with connect_db(student().episodes.db_path) as conn:
        conn.execute("UPDATE episode_items SET created_at=? WHERE namespace=? AND json_extract(store_specific,'$.section_index')=?",
                     (when.isoformat(timespec="seconds"), course_namespace(uid, folder), section))


def _took_queueing_lecture(uid):
    make_course(uid, "Operations", OPS)
    chat(uid, QUOTE, "Not quite — rho must stay below 1 or the queue grows without bound. [ANSWER_WRONG: utilisation]",
         context_id="Operations", section_index=0)
    _backdate(uid, "Operations", 0, datetime(2026, 3, 14, 10, 0, tzinfo=timezone.utc))
    _filler(uid, "Operations", 1, 50)
    _filler(uid, "Operations", 2, 50)


def test_general_chat_recalls_a_specific_old_lecture():
    """Step 5. 'What did we do in the queueing lecture?' finds THAT lecture, not the last 60 messages."""
    uid = make_student()
    _took_queueing_lecture(uid)
    res = chat(uid, "What did we do in the queueing lecture in Operations, and what did I get wrong?",
               "In Queueing basics you...", context_type="global")
    prompt = res["system_prompt"]
    assert QUOTE in prompt, "student's own words from that lecture were not retrieved"
    assert "Queueing basics" in prompt


def test_recall_includes_when_it_happened():
    """Step 5. Recall carries the context of that time (date of the session)."""
    uid = make_student()
    _took_queueing_lecture(uid)
    res = chat(uid, "Remind me what happened in the Operations queueing lecture",
               "Back in March...", context_type="global")
    assert "2026-03-14" in res["system_prompt"] or "14 Mar" in res["system_prompt"], "no date context in recall"


def test_lesson_chat_can_recall_another_course():
    """Step 5. From inside a different course's lesson, Pedro can pull the other course's history."""
    uid = make_student()
    _took_queueing_lecture(uid)
    make_course(uid, "Statistics", [{"title": "Poisson processes", "key_topics": ["Poisson process"]}])
    res = chat(uid, "This feels like the utilisation thing from Operations — what did I say back then?",
               "You said...", context_id="Statistics", section_index=0)
    assert QUOTE in res["system_prompt"], "cross-course recall missing from lesson prompt"


def test_recall_finds_it_without_trigger_words():
    """Step 5. Recall should not depend on phrases like 'recap' or 'what did we do'."""
    uid = make_student()
    _took_queueing_lecture(uid)
    res = chat(uid, "I remember getting rho wrong in Operations ages ago, why was that?",
               "Because...", context_type="global")
    assert QUOTE in res["system_prompt"]


def test_golden_moment_from_chat_is_reused():
    """Step 5. An analogy that clicked ([CLICKED]) comes back when the same concept returns."""
    uid = make_student()
    make_course(uid, "Operations", OPS)
    chat(uid, "Oh! The supermarket checkout analogy made Little's Law click.",
         "Great! [CLICKED: Little's Law as a supermarket checkout: people in line = arrival rate x wait]",
         context_id="Operations", section_index=0)
    res = chat(uid, "Can we go over Little's Law again?", "Sure...", context_id="Operations", section_index=0)
    assert "supermarket" in res["system_prompt"], "golden moment not offered back to Pedro"


def test_golden_moment_from_evaluator_is_reused():
    """Step 5. Evaluator-captured analogies reach later sessions on that concept."""
    uid = make_student()
    ids = make_course(uid, "Operations", OPS)
    stub_evaluator({"little's law": "resolved"}, golden=[{
        "concept_id": ids["little's law"], "concept_name": "Little's Law", "moment_type": "analogy",
        "description": "Coffee-shop queue made L = lambda W intuitive", "reuse_hint": "Start from the coffee shop"}])
    chat(uid, "8 customers", "Yes! [ANSWER_CORRECT: Little's Law] [SECTION_COMPLETE]", context_id="Operations", section_index=0)
    run_learning_jobs()
    block = course_block(uid, "Operations", query="Little's Law", concept_ids=[ids["little's law"]])
    assert "Coffee-shop" in block, block


def test_unresolved_mistake_is_raised_then_dropped_after_recovery():
    """Step 5. A past mistake is surfaced while relevant, and disappears once resolved."""
    uid = make_student()
    ids = make_course(uid, "Operations", OPS)
    chat(uid, QUOTE, "Not quite. [ANSWER_WRONG: utilisation]", context_id="Operations", section_index=0)
    before = chat(uid, "Let's continue", "OK", context_id="Operations", section_index=0)["system_prompt"]
    assert "rho could go above 1" in before, "unresolved mistake not shown to Pedro"
    stub_evaluator({"utilisation": "resolved"})
    chat(uid, "rho < 1 for stability", "Exactly. [ANSWER_CORRECT: utilisation] [SECTION_COMPLETE]",
         context_id="Operations", section_index=0)
    run_learning_jobs()
    after = course_block(uid, "Operations", query="utilisation", concept_ids=[ids["utilisation"]])
    assert "rho could go above 1" not in after, "resolved mistake still presented as open"


def test_no_recall_for_a_course_never_taken():
    """Step 5. Asking about a course with no history yields no invented memories."""
    uid = make_student()
    _took_queueing_lecture(uid)
    res = chat(uid, "What did we cover in my Biology lecture last week?", "I don't have a record of that.",
               context_type="global")
    assert QUOTE not in res["system_prompt"], "unrelated course history was injected"
