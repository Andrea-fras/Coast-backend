"""Adaptive placement: check skipped sections one at a time, place the student at
their first gap, credit what they demonstrated — and never trust a tag alone."""
import uuid

from harness import SessionLocal, chat, episodes, lesson, make_course, make_student

SECTIONS = [{"title": f"Topic {c}", "key_topics": [f"concept {c}"]} for c in "ABCD"]


def _start(uid, target, folder="Algorithms"):
    conv = f"place_{uuid.uuid4().hex[:8]}"
    chat(uid, "I think I already know this.", "Let's check. First question on Topic A…",
         context_type="test_out", context_id=folder, section_index=target, conversation_id=conv)
    return conv


def _say(uid, conv, reply, target, folder="Algorithms", msg="my answer"):
    return chat(uid, msg, reply, context_type="test_out", context_id=folder, section_index=target,
                conversation_id=conv)["placement"]


def _current(uid, folder="Algorithms"):
    from database import CourseOutline
    with SessionLocal() as db:
        return db.query(CourseOutline).filter_by(user_id=uid, folder_name=folder).one().current_section


def test_places_student_at_first_gap_and_credits_passed_sections():
    uid = make_student()
    make_course(uid, "Algorithms", SECTIONS)
    conv = _start(uid, 3)
    _say(uid, conv, "Right. [ANSWER_CORRECT: concept A] [PLACEMENT_PASSED: 1] Now Topic B…", 3)
    _say(uid, conv, "Yes. [ANSWER_CORRECT: concept B] [PLACEMENT_PASSED: 2] Now Topic C…", 3)
    _say(uid, conv, "Not quite. [ANSWER_WRONG: concept C] One more…", 3)
    st = _say(uid, conv, "Let's start you there. [ANSWER_WRONG: concept C] [PLACEMENT_STOP]", 3)
    assert st["done"] and st["passed_count"] == 2 and st["place_at"] == 2, st
    out = lesson.apply_test_out(uid, "Algorithms", 3, conv)
    assert out.get("current_section") == 2, out
    assert out["skipped_sections"] == [0, 1]
    assert lesson.can_advance_from_section(uid, "Algorithms", 1)


def test_pass_tag_before_any_answer_is_ignored():
    uid = make_student()
    make_course(uid, "Algorithms", SECTIONS)
    conv = f"place_{uuid.uuid4().hex[:8]}"
    st = chat(uid, "I think I already know this.", "Great, you know it. [PLACEMENT_PASSED: 1]",
              context_type="test_out", context_id="Algorithms", section_index=3, conversation_id=conv)["placement"]
    assert st["passed_count"] == 0, st


def test_a_forgotten_correct_tag_passes_once_per_reply():
    """Pedro sometimes confirms a right answer with only [PLACEMENT_PASSED]; that counts, once."""
    uid = make_student()
    make_course(uid, "Algorithms", SECTIONS)
    conv = _start(uid, 3)
    st = _say(uid, conv, "Right. [PLACEMENT_PASSED: 1] [PLACEMENT_PASSED: 2]", 3)
    assert st["passed_count"] == 1, st


def test_hinted_answer_does_not_pass_a_section():
    uid = make_student()
    make_course(uid, "Algorithms", SECTIONS)
    conv = _start(uid, 3)
    st = _say(uid, conv, "With that hint, yes. [ANSWER_CORRECT: concept A | hinted] [PLACEMENT_PASSED: 1]", 3)
    assert st["passed_count"] == 0, st


def test_cannot_pass_a_section_out_of_order():
    uid = make_student()
    make_course(uid, "Algorithms", SECTIONS)
    conv = _start(uid, 3)
    st = _say(uid, conv, "Correct. [ANSWER_CORRECT: concept C] [PLACEMENT_PASSED: 3]", 3)
    assert st["passed_count"] == 0 and st["checking_section"] == 0, st


def test_repeated_wrong_answers_end_the_check():
    uid = make_student()
    make_course(uid, "Algorithms", SECTIONS)
    conv = _start(uid, 3)
    for _ in range(3):
        st = _say(uid, conv, "Not quite. [ANSWER_WRONG: concept A]", 3)
    assert st["done"] and st["passed_count"] == 0 and not st["can_apply"], st


def test_can_test_out_of_the_whole_course():
    uid = make_student()
    make_course(uid, "Algorithms", SECTIONS)
    conv = _start(uid, len(SECTIONS))
    for n, c in enumerate("ABCD", start=1):
        st = _say(uid, conv, f"Yes. [ANSWER_CORRECT: concept {c}] [PLACEMENT_PASSED: {n}]", len(SECTIONS))
    assert st["done"] and st["passed_count"] == 4, st
    out = lesson.apply_test_out(uid, "Algorithms", len(SECTIONS), conv)
    assert out.get("is_complete") is True, out


def test_placement_answers_are_filed_under_the_section_being_checked():
    uid = make_student()
    make_course(uid, "Algorithms", SECTIONS)
    conv = _start(uid, 3)
    _say(uid, conv, "Right. [ANSWER_CORRECT: concept A] [PLACEMENT_PASSED: 1]", 3, msg="A is about …")
    _say(uid, conv, "Not quite. [ANSWER_WRONG: concept B]", 3, msg="B is about …")
    graded = {(e.store_specific or {}).get("user_message"): (e.store_specific or {}).get("section_index")
              for e in episodes(uid, "Algorithms") if (e.store_specific or {}).get("episode_type") == "exercise_attempt"}
    assert graded.get("A is about …") == 0 and graded.get("B is about …") == 1, graded


def test_placement_tags_never_reach_the_student():
    import tutor
    uid = make_student()
    make_course(uid, "Algorithms", SECTIONS)
    conv = _start(uid, 3)
    _say(uid, conv, "Right. [ANSWER_CORRECT: concept A] [PLACEMENT_PASSED: 1] Next question.", 3)
    shown = " ".join(m["content"] for m in tutor.get_chat_history(conv, uid))
    assert "PLACEMENT" not in shown and "ANSWER_" not in shown, shown
