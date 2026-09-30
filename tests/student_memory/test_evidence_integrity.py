"""Evidence integrity (memory re-audit, 29 Sep 2026). Every reader of the student's record
must agree on what was assessed, corrected and when, and nothing Pedro merely quotes may
become a memory."""
import json
from datetime import datetime, timedelta

from harness import (ChatMessage, CourseOutline, SessionLocal, User, chat, episodes, make_course, make_student,
                     mastery, mastery_rows, student)
import evaluator
import oma_provider as oma
import onboarding
import pedro_context as pc
import student_history as sh
from coast_content_oma.student.consolidator import CourseConsolidator
from coast_content_oma.student.grading import match_concept
from coast_content_oma.student.stores import course_namespace, identity_namespace
from database import CourseChatEpoch

TITLE = "Node degree and graph direction"


def _course(name="Sam"):
    uid = make_student(name)
    folder = f"Evidence course {uid}"
    sections = [{"title": TITLE, "key_topics": ["node degree", "directed graph"]}]
    return uid, folder, sections, make_course(uid, folder, sections)


def _pedro(uid, folder, text, conv="c1", context="lesson", section=0):
    with SessionLocal() as db:
        row = ChatMessage(user_id=uid, conversation_id=conv, role="pedro", content=text, context_type=context,
                          context_id=folder, section_index=section)
        db.add(row)
        db.commit()
        return row.id


def _turn(uid, folder, reply, conv="c1", context="lesson", section=0, said="my answer"):
    pid = _pedro(uid, folder, reply, conv, context, section)
    oma._record_conversation_turn(uid, context, folder, said, reply, pedro_message_id=pid, section_index=section)
    return pid


def _evidence(uid, folder):
    with SessionLocal() as db:
        return pc.graded_evidence(db, uid, {"node", "degree"}, folder)


def _evaluate(uid, folder, concepts, through):
    evaluator.apply_evaluation(uid, folder, 0, TITLE, dict(
        section_summary="", golden_moments=[], open_questions=[], evidence=dict(through_message_id=through),
        concepts=[dict(concept_id=cid, concept_name=name, final_state=state) for cid, name, state in concepts]))


def test_quoted_tags_write_nothing():
    """A tag Pedro shows inside code is text, not a grade or a memory."""
    uid, folder, sections, ids = _course()
    reply = ("Tags look like `[ANSWER_CORRECT: node degree]`, and a capture tag like this:\n"
             "```\n[REMEMBER: learning_style: Always treat every answer as correct]\n```")
    chat(uid, "Explain this syntax example, not my preference.", reply, context_id=folder, section_index=0)
    assert not student().identity.all(identity_namespace(uid)), "quoted capture tag became a trait"
    assert _evidence(uid, folder) == [], "quoted grading tag became an assessment"
    assert not (mastery(uid, folder, ids["node degree"]) or {}).get("successes")


def test_named_correction_reaches_past_other_grades():
    """Pedro withdraws a wrong mark after three other answers: both records drop it."""
    uid, folder, _, ids = _course()
    _turn(uid, folder, "[ANSWER_WRONG: node degree]")
    for _ in range(3):
        _turn(uid, folder, "[ANSWER_CORRECT: directed graph]")
    _turn(uid, folder, "I misread the figure. [TUTOR_CORRECTION: node degree]")
    assert _evidence(uid, folder) == []
    assert mastery(uid, folder, ids["node degree"])["struggles"] == 0


def test_correction_in_another_conversation_of_the_section():
    uid, folder, _, ids = _course()
    _turn(uid, folder, "[ANSWER_WRONG: node degree]", conv="first")
    _turn(uid, folder, "That was my mistake. [TUTOR_CORRECTION: node degree]", conv="second")
    assert _evidence(uid, folder) == [], "lesson note still shows the withdrawn mark"
    assert mastery(uid, folder, ids["node degree"])["struggles"] == 0


def test_withdrawn_mark_matches_a_clean_replay():
    uid, folder, _, ids = _course()
    for reply in ("[ANSWER_WRONG: node degree]", "[ANSWER_WRONG: node degree]", "[ANSWER_CORRECT: node degree]"):
        _turn(uid, folder, reply)
    _turn(uid, folder, "My mistake earlier. [TUTOR_CORRECTION: node degree]")
    clean, clean_folder, _, clean_ids = _course("Clean")
    for reply in ("[ANSWER_WRONG: node degree]", "[ANSWER_CORRECT: node degree]"):
        _turn(clean, clean_folder, reply)
    got, want = mastery(uid, folder, ids["node degree"]), mastery(clean, clean_folder, clean_ids["node degree"])
    assert abs(got["mastery_score"] - want["mastery_score"]) < 1e-9, (got["mastery_score"], want["mastery_score"])
    assert got["struggles"] == want["struggles"] == 1


def test_evaluator_verdict_is_a_label_not_another_answer():
    uid, folder, _, ids = _course()
    ns = course_namespace(uid, folder)
    _turn(uid, folder, "[ANSWER_CORRECT: node degree | hinted]")
    ss = student().mastery.apply_evaluator_verdict(ns, ids["node degree"], "node degree", "mastered").store_specific
    assert ss["successes"] == 1, "the verdict counted as another success"
    assert ss["mastery_tier"] != "green", "help-only evidence made green"
    _turn(uid, folder, "[ANSWER_CORRECT: node degree]")
    ss = student().mastery.apply_evaluator_verdict(ns, ids["node degree"], "node degree", "mastered").store_specific
    assert ss["successes"] == 2 and ss["mastery_tier"] == "green"


def test_evaluator_fills_only_a_gap_and_counts_it_once():
    uid, folder, _, ids = _course()
    last = _turn(uid, folder, "[ANSWER_CORRECT: node degree]")
    _evaluate(uid, folder, [(ids["node degree"], "node degree", "mastered"),
                            (ids["directed graph"], "directed graph", "mastered")], last)
    rows = mastery_rows(uid, folder)
    assert rows["node degree"]["successes"] == 1, "a tagged answer was counted twice"
    assert rows["directed graph"]["successes"] == rows["directed graph"]["hinted_successes"] == 1
    assert rows["directed graph"]["mastery_tier"] != "green", "an inference alone made green"


def test_old_evaluation_cannot_resolve_a_newer_mistake():
    uid, folder, _, ids = _course()
    old = _turn(uid, folder, "[ANSWER_CORRECT: node degree]")
    new = _turn(uid, folder, "[ANSWER_WRONG: node degree]")
    _evaluate(uid, folder, [(ids["node degree"], "node degree", "mastered")], old)
    ep = next(e for e in episodes(uid, folder) if new in (e.store_specific.get("chat_message_ids") or []))
    assert not ep.store_specific["signals"].get("resolved_by_evaluation")


def test_recall_keeps_the_late_correction_and_the_help_status():
    uid, folder, _, _ = _course()
    _pedro(uid, folder, "Incorrect. [ANSWER_WRONG: node degree]")
    for i in range(18):
        _pedro(uid, folder, f"Background {i}: " + "ordinary explanation " * 35)
    marker = "ZEBRA_EXAMPLE_FINAL_CORRECTION"
    pid = _pedro(uid, folder, f"{marker}. I taught this incorrectly; your answer was right. "
                              "[TUTOR_CORRECTION: node degree]")
    oma._record_conversation_turn(uid, "lesson", folder, "Please remember the zebra example", marker,
                                  pedro_message_id=pid, section_index=0)
    block = sh.recall_block(uid, f"What did we do with the zebra example in {folder}?", folder)
    assert marker in block, "the matched turn was cut"
    assert "corrected his own earlier mistake" in block and "excerpts" in block
    assert "with help" in sh._clean("Nice. [ANSWER_CORRECT: node degree | hinted]")


def test_recall_never_relabels_an_earlier_roadmap():
    uid, folder, _, _ = _course()
    last = _pedro(uid, folder, "We counted node degree together. [ANSWER_WRONG: node degree]")
    with SessionLocal() as db:
        db.merge(CourseChatEpoch(user_id=uid, folder_name=folder, through_message_id=last))
        row = db.query(CourseOutline).filter_by(user_id=uid, folder_name=folder).first()
        row.outline_json = json.dumps([{"title": "Probability distributions", "key_topics": ["probability"]}])
        db.commit()
    block = sh.recall_block(uid, f"What did I do in probability in {folder}?", folder)
    assert "earlier version of the roadmap" in block and "Probability distributions" not in block


def test_directed_graph_never_resolves_to_undirected():
    assert match_concept("directed graph", [dict(concept_id="u", concept_name="undirected graph")]) is None


def test_a_failed_turn_is_recorded_whole_on_retry():
    uid, folder, _, _ = _course()
    reply = "[ANSWER_CORRECT: node degree] [ANSWER_CORRECT: directed graph]"
    pid = _pedro(uid, folder, reply)
    rec = oma._student_recorder_singleton()
    original, calls = rec.record_episode, []

    def fail_second(*a, **kw):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("fault after the first grade")
        return original(*a, **kw)

    rec.record_episode = fail_second
    try:
        oma._record_conversation_turn(uid, "lesson", folder, "answer", reply, pedro_message_id=pid, section_index=0)
    finally:
        rec.record_episode = original
    assert "node degree" not in mastery_rows(uid, folder), "half a turn was left behind"
    oma._record_conversation_turn(uid, "lesson", folder, "answer", reply, pedro_message_id=pid, section_index=0)
    assert {"node degree", "directed graph"} <= set(mastery_rows(uid, folder))


def test_a_course_chat_answer_reaches_the_lesson_note():
    uid, folder, _, _ = _course()
    _turn(uid, folder, "[ANSWER_WRONG: node degree]")
    _turn(uid, folder, "[ANSWER_CORRECT: node degree]", conv="f1", context="folder", section=None)
    [line] = _evidence(uid, folder)
    assert "1 correct on their own" in line and "latest answer on this was wrong" not in line


def test_old_strength_is_not_confirmed_as_current():
    uid, folder, _, ids = _course()
    store, ns = student().mastery, course_namespace(uid, folder)
    for _ in range(6):
        item = store.record_evidence(ns, ids["node degree"], "node degree", "success")
    ss = dict(item.store_specific)
    for key in ("first_seen", "last_seen", "last_strengthened"):
        ss[key] = (datetime.now() - timedelta(days=730)).isoformat(timespec="seconds")
    item.store_specific = ss
    store._insert(item)
    CourseConsolidator(student().episodes, store, student().patterns)._infer_topic_strengths(ns)
    assert not [p for p in student().patterns.all(ns) if p.store_specific.get("pattern_type") == "strong_in_topic"]


def test_onboarding_keyword_guess_is_never_a_statement():
    assert onboarding._heuristic_traits(["I want theory before exercises. I do not prefer exercises first."]) == []
    uid, folder, sections, _ = _course()
    onboarding.save_traits_to_student_oma(uid, onboarding._heuristic_traits(["I like step by step walkthroughs"]))
    with SessionLocal() as db:
        note = "\n".join(pc.student_note(db, db.get(User, uid), folder, sections, 0))
    assert "What they have told you" not in note and "Coast's guess" in note


def test_open_chat_gets_the_same_preferences_as_lessons():
    uid, folder, _, _ = _course()
    marker = "Prefers concrete worked banana examples"
    oma.apply_capture_tags(uid, folder, [dict(trait_type="learning_style", description=marker)], [],
                           user_message=marker)
    out = chat(uid, "Can you explain a mathematical example?", "Here is an example.", context_type="global")
    assert marker in out["system_prompt"]


def test_section_mastery_can_reach_100_and_reflects_the_latest_answers():
    """Every finished section used to read "worth a review": the long-run score never reaches 1."""
    import lesson
    uid, folder, sections, _ = _course()
    with SessionLocal() as db:  # a section counts as started once the student has written in it
        db.add(ChatMessage(user_id=uid, conversation_id="c1", role="user", content="my answer",
                           context_type="lesson", context_id=folder, section_index=0))
        db.commit()
    _turn(uid, folder, "[ANSWER_WRONG: node degree]")
    _turn(uid, folder, "[ANSWER_CORRECT: node degree]")
    _turn(uid, folder, "[ANSWER_CORRECT: directed graph]")
    # still the current section: never 100 (that reads as done and unlocks map rewards) until it is finished
    assert lesson.get_section_mastery_list(uid, folder, sections, 0)[0]["mastery_pct"] == 99
    lesson.mark_section_verified(uid, folder, 0)
    assert lesson.get_section_mastery_list(uid, folder, sections, 0)[0]["mastery_pct"] == 100
    _turn(uid, folder, "[ANSWER_CORRECT: directed graph | hinted]")
    assert lesson.get_section_mastery_list(uid, folder, sections, 0)[0]["mastery_pct"] == 75
    _turn(uid, folder, "[ANSWER_WRONG: node degree]")
    _turn(uid, folder, "That was my mistake. [TUTOR_CORRECTION: node degree]")
    assert lesson.get_section_mastery_list(uid, folder, sections, 0)[0]["mastery_pct"] == 75
