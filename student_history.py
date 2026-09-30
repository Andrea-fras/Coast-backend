"""Recall of past sessions for Pedro ("what did we do in that lecture?").

Finds the course(s) and section(s) the student is asking about, then quotes
excerpts of the real transcript from chat_messages (the canonical record) with
dates: the turns around what matched, every exchange where Pedro corrected
himself, and how each answer was graded. Student OMA episodes are the searchable
index that points back at those messages. Long messages are shortened and gaps
are marked, so the block never claims to be the whole conversation.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Optional

from database import SessionLocal, ChatMessage, CourseOutline, StudyFolder

MEMORY_CUES = re.compile(
    r"\b(remember|recall|remind me|back then|earlier|previous(?:ly)?|ago|last (?:time|week|month|term|semester|year)|"
    r"what (?:did|have) (?:i|we)|did (?:i|we)|we (?:did|covered|learned|learnt|studied)|"
    r"i (?:said|wrote|picked|chose|answered|got|thought)|recap|catch me up)\b",
    re.I,
)
_STOP = set("""a an and are as at be been but by can could did do does doing for from get got had has have how i if in
into is it its just like me my of on or our so than that the their them then there these they thing things this to
too up us was we were what when where which who why will with would you your about again also back before earlier
ago remember recall remind previous previously last time week weeks month term semester year lecture lectures lesson
lessons class section sections course courses session sessions cover covered learn learned learnt studied study
say said wrote write think thought feel feels tell told know okay ok yes no please help explain happened wrong right""".split())

MAX_SECTIONS = 2
PER_SECTION_CHARS = 2600
STUDENT_MSG_CHARS = 400
PEDRO_MSG_CHARS = 260


def _terms(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return list(dict.fromkeys(w for w in words if len(w) > 2 and w not in _STOP))


def _stem(word: str) -> str:
    return word[:5] if len(word) >= 5 else word


def _overlap(terms: list[str], text: str) -> int:
    tokens = {_stem(t) for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2}
    return sum(1 for t in terms if _stem(t) in tokens)


def _student_courses(user_id: int) -> dict[str, list[dict]]:
    """{course name: outline sections (possibly empty)} for everything the student has touched."""
    courses: dict[str, list[dict]] = {}
    with SessionLocal() as db:
        for o in db.query(CourseOutline).filter(CourseOutline.user_id == user_id):
            try:
                courses[o.folder_name] = json.loads(o.outline_json) or []
            except (TypeError, ValueError):
                courses[o.folder_name] = []
        for f in db.query(StudyFolder).filter(StudyFolder.user_id == user_id):
            courses.setdefault(f.name, [])
        for (name,) in (db.query(ChatMessage.context_id)
                        .filter(ChatMessage.user_id == user_id,
                                ChatMessage.context_type.in_(("lesson", "folder", "test_out")))
                        .distinct()):
            if name:
                courses.setdefault(name, [])
    return courses


def _mentions(message: str, course: str) -> bool:
    msg = (message or "").lower()
    name = course.lower().strip()
    if name and re.search(r"\b" + re.escape(name) + r"\b", msg):
        return True
    tokens = [t for t in re.findall(r"[a-z0-9]+", name) if len(t) > 2 and t not in _STOP]
    return bool(tokens) and all(re.search(r"\b" + re.escape(t) + r"\b", msg) for t in tokens)


def _episode_hits(user_id: int, folder: Optional[str], terms: list[str]) -> dict:
    """{section_index or None: [chat message ids]} for this student's episodes matching the terms."""
    import oma_provider
    from coast_content_oma.stores.db import connect_db
    from coast_content_oma.student.stores import course_namespace

    if not terms:
        return {}
    ns = course_namespace(user_id, folder) if folder else oma_provider.general_namespace(user_id)
    match = " OR ".join(f'"{t}"*' if len(t) >= 4 else f'"{t}"' for t in terms[:12])
    hits: dict = defaultdict(list)
    with connect_db(oma_provider._student_orchestrator().episodes.db_path) as conn:
        try:
            rows = conn.execute(
                "SELECT e.store_specific FROM episode_items_fts f JOIN episode_items e ON e.id = f.id "
                "WHERE episode_items_fts MATCH ? AND f.namespace = ? ORDER BY rank LIMIT 40",
                (match, ns),
            ).fetchall()
        except Exception:
            return {}
    for (raw,) in rows:
        ss = json.loads(raw or "{}")
        hits[ss.get("section_index")].extend(ss.get("chat_message_ids") or [])
    return hits


def _marks(text: str) -> str:
    """How Pedro graded this reply, in words, keeping "with help", "remembered" and his own
    corrections: the tags sit at the end of a reply, where shortening would cut them."""
    from coast_content_oma.student.grading import parse_grades, parse_tutor_corrections
    marks = []
    for g in parse_grades(text):
        verdict = ("wrong" if not g.correct else "correct with help" if g.hinted
                   else "correct, remembered from an earlier session" if g.recall else "correct")
        marks.append(f"Pedro marked the answer {verdict}" + (f": {g.concept}" if g.concept else ""))
    for concept in parse_tutor_corrections(text):
        marks.append("Pedro corrected his own earlier mistake" + (f": {concept}" if concept else ""))
    return f" ({'; '.join(marks)})" if marks else ""


def _plain(text: str) -> str:
    import oma_provider
    return re.sub(r"\s+", " ", oma_provider.strip_pedro_tags(text or "")).strip()


def _clean(text: str) -> str:
    return _plain(text) + _marks(text)


def _line(m) -> str:
    if m.role == "user":
        body = _plain(m.content)[:STUDENT_MSG_CHARS]
        return f'[{m.created_at:%Y-%m-%d}] Student: "{body}"' if m.created_at else f'Student: "{body}"'
    body = _plain(m.content)
    body = body if len(body) <= PEDRO_MSG_CHARS else body[:PEDRO_MSG_CHARS].rsplit(" ", 1)[0] + " …"
    return f"Pedro: {body}{_marks(m.content)}"


def _render(messages: list, header: str, focus_ids=()) -> str:
    """The turns that matter within the budget, in order, with gaps marked: the matched
    turns and their neighbours (without a match, the start and the end) and every
    exchange where Pedro corrected himself."""
    from coast_content_oma.student.grading import parse_tutor_corrections
    if not messages:
        return ""
    focus = set(focus_ids or ())
    matched = [i for i, m in enumerate(messages) if m.id in focus]
    last = len(messages) - 1
    order = ([j for i in matched for j in (i, i + 1, i - 1)] if matched
             else [0, last, 1, last - 1, last - 2, 2, last - 3])
    order += [j for i, m in enumerate(messages)
              if m.role == "pedro" and parse_tutor_corrections(m.content) for j in (i, i - 1)]
    lines = {i: _line(m) for i, m in enumerate(messages)}
    keep, used = set(), 0
    for j in order:
        if 0 <= j <= last and j not in keep and used + len(lines[j]) <= PER_SECTION_CHARS:
            keep.add(j)
            used += len(lines[j])
    if not keep:
        return ""
    out, prev = [], -1
    for j in sorted(keep):
        if j != prev + 1:
            out.append("…")
        out.append(lines[j])
        prev = j
    if prev != last:
        out.append("…")
    dates = sorted({messages[j].created_at.strftime("%Y-%m-%d") for j in keep if messages[j].created_at})
    when = dates[0] if len(dates) == 1 else f"{dates[0]} to {dates[-1]}" if dates else "date unknown"
    return f"{header} — {when}\n" + "\n".join(out)


_OPENER_TITLE = re.compile(r'learn about ["“]([^"”]+)["”]', re.I)


def _epoch(user_id: int, folder: str) -> int:
    """Lesson messages up to this id belong to an earlier version of the course's roadmap."""
    from database import CourseChatEpoch
    with SessionLocal() as db:
        row = db.get(CourseChatEpoch, (int(user_id), folder))
        return int(row.through_message_id) if row else 0


def _section_conversation(user_id: int, folder: str, section_index: int, focus_ids=()) -> tuple[list, Optional[str], bool]:
    """(messages, title at the time, from an earlier roadmap) for the conversation in this
    section that matched, else the latest one. A regenerated roadmap reuses section
    numbers, so the title comes from the conversation's own opener, never from today's
    roadmap for a conversation held under an earlier one."""
    import lesson
    with SessionLocal() as db:
        rows = (db.query(ChatMessage)
                .filter(ChatMessage.user_id == user_id, ChatMessage.context_id == folder,
                        ChatMessage.section_index == section_index,
                        ChatMessage.context_type.in_(("lesson", "folder", "test_out")))
                .order_by(ChatMessage.id).all())
    if not rows:
        return [], None, False
    conversations: dict = defaultdict(list)
    for m in rows:
        conversations[m.conversation_id].append(m)
    focus = set(focus_ids or ())
    chosen = next((c for c in reversed(list(conversations.values())) if any(m.id in focus for m in c)),
                  list(conversations.values())[-1])
    opener = next((m for m in chosen if m.role == "user" and lesson._SECTION_OPENER.match((m.content or "").strip())), None)
    title = _OPENER_TITLE.search(opener.content).group(1) if opener and _OPENER_TITLE.search(opener.content or "") else None
    earlier = chosen[-1].id <= _epoch(user_id, folder)
    body = [m for m in chosen if m is not opener and not (m.role == "user" and lesson._SECTION_OPENER.match((m.content or "").strip()))]
    return body, title, earlier


def _messages_by_ids(user_id: int, ids: list[int]) -> list:
    """The referenced turns (student message + Pedro reply), oldest first."""
    if not ids:
        return []
    with SessionLocal() as db:
        return (db.query(ChatMessage)
                .filter(ChatMessage.user_id == user_id, ChatMessage.id.in_(set(ids)))
                .order_by(ChatMessage.id).all())


def recall_block(user_id: int, message: str, current_folder: Optional[str] = None, max_chars: int = 6000) -> str:
    """Verbatim history for the course/section the student is asking about, or ''."""
    if not message:
        return ""
    courses = _student_courses(int(user_id))
    mentioned = [c for c in courses if _mentions(message, c)]
    others = [c for c in mentioned if c != current_folder]
    cue = bool(MEMORY_CUES.search(message))
    if not others and not cue:
        return ""

    terms = _terms(message)
    targets = mentioned or list(courses)
    candidates = []  # (score, folder, section_index, section title)
    hits_by_folder: dict = {}
    for folder in targets:
        sections = courses.get(folder) or []
        hits = hits_by_folder[folder] = _episode_hits(int(user_id), folder, terms)
        name_terms = {_stem(t) for t in _terms(folder)}
        topic_terms = [t for t in terms if _stem(t) not in name_terms]
        for idx, sec in enumerate(sections):
            score = 3 * _overlap(topic_terms, sec.get("title", ""))
            score += 2 * _overlap(topic_terms, " ".join(map(str, sec.get("key_topics") or [])))
            score += min(3, len(hits.get(idx, [])) // 2)
            if score:
                candidates.append((score + (1 if folder in mentioned else 0), folder, idx, sec.get("title")))
        if hits.get(None):
            candidates.append((min(3, len(hits[None]) // 2) + (1 if folder in mentioned else 0),
                               folder, None, hits[None]))
        if folder in mentioned and not any(c[1] == folder for c in candidates) and sections:
            with SessionLocal() as db:
                last = (db.query(ChatMessage.section_index)
                        .filter(ChatMessage.user_id == user_id, ChatMessage.context_id == folder,
                                ChatMessage.section_index.isnot(None))
                        .order_by(ChatMessage.id.desc()).first())
            if last and last[0] is not None and last[0] < len(sections):
                candidates.append((1, folder, last[0], sections[last[0]].get("title")))
    if cue and not mentioned:
        hits = _episode_hits(int(user_id), None, terms).get(None)
        if hits:
            candidates.append((min(3, len(hits) // 2), None, None, hits))

    blocks = []
    for _score, folder, idx, extra in sorted(candidates, key=lambda c: -c[0])[:MAX_SECTIONS]:
        if idx is None:
            msgs = _messages_by_ids(int(user_id), extra)
            header = f"{folder} — ask-your-sources chat" if folder else "General chat with Pedro"
            block = _render(msgs, header, extra)
        else:
            focus = hits_by_folder.get(folder, {}).get(idx, [])
            msgs, title, earlier = _section_conversation(int(user_id), folder, idx, focus)
            if earlier:
                header = (f'{folder} — Section {idx + 1} of an earlier version of the roadmap'
                          + (f' ("{title}")' if title else ""))
            else:
                header = f'{folder} — Section {idx + 1} "{title or extra}"'
            block = _render(msgs, header, focus)
        if block:
            blocks.append(block)
    if not blocks:
        return ""
    body = "\n\n".join(blocks)[:max_chars]
    return (
        "--- STUDENT HISTORY RECALL (excerpts from past sessions) ---\n"
        "Selected turns from what actually happened, with dates; long messages are shortened and … marks "
        "turns left out. Answer questions about the past from it: say when it was, quote the student's own "
        "words where useful, and never add events that are not shown here. If what they ask about isn't "
        "shown, say you don't have that part in front of you.\n"
        f"{body}\n"
        "--- END STUDENT HISTORY RECALL ---"
    )
