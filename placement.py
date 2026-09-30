"""Adaptive placement ("test out"). Client tags never authorize a section jump.

Pedro checks the skipped sections in order, one at a time. A section counts as
passed only when Pedro marks exactly the section being checked with
[PLACEMENT_PASSED: <n>] after an independent correct answer: an [ANSWER_CORRECT: …]
without "| hinted" since the last pass, or, when he forgot that tag, a pass in a reply
to the student's answer that grades nothing as wrong or helped (one such pass per reply). A section the student cannot clear within
MAX_GRADED_PER_SECTION graded answers ends the placement. The student is placed
at the first section they did not demonstrate; every section before it is credited.
"""
import hashlib
import json
import re

from database import SessionLocal, CourseOutline, PlacementTestSession
from workshops import decorate_sections, is_workshop

MAX_GRADED_PER_SECTION = 2
_PASSED_RE = re.compile(r"\[PLACEMENT_PASSED:\s*(\d+)\s*\]", re.I)
_STOP_RE = re.compile(r"\[PLACEMENT_STOP\]", re.I)
LEGACY_PASS = "[TEST_OUT_PASSED]"


def outline_digest(outline):
    return hashlib.sha256(outline.outline_json.encode('utf-8')).hexdigest()


def begin(user_id, folder, target, conversation_id):
    """target: the section to reach; total_sections means the whole course."""
    with SessionLocal() as db:
        outline = db.query(CourseOutline).filter_by(user_id=user_id, folder_name=folder).first()
        if not outline or target <= outline.current_section or target > outline.total_sections:
            raise ValueError("Choose a locked section in the current lesson")
        if is_workshop(decorate_sections(folder, json.loads(outline.outline_json))):
            raise ValueError("Workshops are built milestone by milestone, so there's no skipping ahead.")
        row = db.get(PlacementTestSession, (user_id, conversation_id))
        digest = outline_digest(outline)
        if row:
            if (row.folder_name != folder or row.target_section != target
                    or row.start_section != outline.current_section or row.outline_digest != digest or row.consumed):
                raise ValueError("This placement session no longer matches the lesson. Start a new test.")
        else:
            db.add(PlacementTestSession(user_id=user_id, conversation_id=conversation_id,
                                        folder_name=folder, target_section=target,
                                        start_section=outline.current_section, outline_digest=digest))
            db.commit()


def _state(row):
    if row is None:
        return None
    passed = row.passed_count or 0
    checking = row.start_section + passed
    finished = bool(row.done) or checking >= row.target_section
    return {
        "start_section": row.start_section,
        "target_section": row.target_section,
        "passed_count": passed,
        "checking_section": None if finished else checking,
        "place_at": checking,
        "done": finished,
        "can_apply": passed > 0 and not row.consumed,
    }


def state(user_id, conversation_id):
    with SessionLocal() as db:
        return _state(db.get(PlacementTestSession, (user_id, conversation_id)))


def record_turn(user_id, conversation_id, reply):
    """Apply one Pedro reply to the placement. Returns the new state."""
    from coast_content_oma.student.grading import own_voice, parse_grades
    with SessionLocal() as db:
        row = db.get(PlacementTestSession, (user_id, conversation_id))
        if not row or row.consumed or row.done:
            return _state(row)
        outline = db.query(CourseOutline).filter_by(user_id=user_id, folder_name=row.folder_name).first()
        if not outline or outline_digest(outline) != row.outline_digest or outline.current_section != row.start_section:
            return _state(row)
        grades = parse_grades(reply)
        row.graded_since = (row.graded_since or 0) + len(grades)
        row.correct_since = (row.correct_since or 0) + sum(1 for g in grades if g.correct and not g.hinted)
        checking = row.start_section + (row.passed_count or 0)
        # A pass without its [ANSWER_CORRECT] tag still counts once per reply, but never on
        # Pedro's opening message and never when this reply grades an answer wrong or helped.
        from database import ChatMessage
        answered = db.query(ChatMessage).filter_by(user_id=user_id, conversation_id=conversation_id,
                                                   role="user").count() > 1
        implied = answered and not any(not g.correct or g.hinted for g in grades)
        for m in _PASSED_RE.finditer(own_voice(reply)):
            # Pedro numbers sections from 1; only the section being checked can pass,
            # and only after an independent correct answer since the last pass.
            if int(m.group(1)) - 1 == checking and (row.correct_since > 0 or implied):
                implied = implied and row.correct_since > 0  # the forgotten-tag allowance is spent
                row.passed_count = (row.passed_count or 0) + 1
                row.correct_since = row.graded_since = 0
                checking += 1
        if (_STOP_RE.search(own_voice(reply)) or row.graded_since > MAX_GRADED_PER_SECTION
                or checking >= row.target_section):
            row.done = True
        row.passed = (row.passed_count or 0) > 0
        db.commit()
        return _state(row)


def record_pass(user_id, conversation_id):
    """Legacy all-at-once pass ([TEST_OUT_PASSED]); kept for older clients and tests."""
    with SessionLocal() as db:
        row = db.get(PlacementTestSession, (user_id, conversation_id))
        if not row or row.consumed:
            return False
        outline = db.query(CourseOutline).filter_by(user_id=user_id, folder_name=row.folder_name).first()
        if not outline or outline_digest(outline) != row.outline_digest or outline.current_section != row.start_section:
            return False
        row.passed = True
        row.passed_count = row.target_section - row.start_section
        row.done = True
        db.commit()
        return True


def authorize(db, user_id, folder, target, conversation_id, outline):
    row = db.get(PlacementTestSession, (user_id, conversation_id)) if conversation_id else None
    if (not row or not row.passed or not row.passed_count or row.folder_name != folder
            or row.target_section != target or row.outline_digest != outline_digest(outline)):
        raise ValueError("Pass a placement test for this lesson and target section first")
    if not row.consumed and row.start_section != outline.current_section:
        raise ValueError("Lesson progress changed. Start a new placement test.")
    return row
