#!/usr/bin/env python3
"""Bring existing students' Student OMA up to the current memory model.

Steps (each idempotent; safe to re-run):
  1. backfill   Replay chat_messages through the live recording path so every past
                turn (lesson, workshop, folder, general chat) is indexed with its
                chat message ids and original timestamp. Legacy mistake episodes
                are re-recorded under current attribution rules (never duplicated).
  2. mastery    Rebuild every concept_mastery row from the ordered episode log
                (graded answers + section-evaluator verdicts) with the evidence
                model — replaces inflated legacy scores.
  3. traits     Delete malformed identity traits (raw concept ids etc.).
  4. consolidate  Re-derive patterns and cross-course identity for each course.

Dry run by default. --apply backs up oma.db first.

  python3 scripts/rebuild_student_memory.py                 # dry run, all students
  python3 scripts/rebuild_student_memory.py --user-id 14 --apply
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("OMA_CONSOLIDATE_EVERY_N_TURNS", "1000000000")  # consolidate once, at the end

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import oma_provider  # noqa: E402
from database import SessionLocal, ChatMessage  # noqa: E402
from coast_content_oma.stores.db import connect_db  # noqa: E402
from coast_content_oma.student.stores import course_namespace, identity_namespace, list_course_namespaces  # noqa: E402
from coast_content_oma.student.stores.academic_identity import is_displayable_trait  # noqa: E402

COURSE_CONTEXTS = ("lesson", "folder", "test_out")
RECORDED_CONTEXTS = COURSE_CONTEXTS + ("global", "notebook", "session")


def _pairs(user_id: int):
    """(student message, Pedro reply) pairs in order, per conversation."""
    with SessionLocal() as db:
        rows = (db.query(ChatMessage)
                .filter(ChatMessage.user_id == user_id, ChatMessage.context_type.in_(RECORDED_CONTEXTS))
                .order_by(ChatMessage.id).all())
        db.expunge_all()
    pending = {}
    for m in rows:
        if m.role == "user":
            pending[m.conversation_id] = m
        elif m.role == "pedro" and m.conversation_id in pending:
            yield pending.pop(m.conversation_id), m


def _take_legacy(orch, ns, user_msg) -> dict | None:
    """Remove a legacy (pre-message-id) mistake episode for this turn so it can be
    re-recorded under current attribution rules; returns its evaluator flags."""
    text = (user_msg.content or "")[:200]
    for ep in orch.episodes.by_types(ns, ("exercise_attempt",), section_index=user_msg.section_index):
        ss = ep.store_specific or {}
        if ss.get("chat_message_ids") or (ss.get("user_message") or "")[:200] != text:
            continue
        orch.episodes.delete(ep.id)
        return {k: v for k, v in (ss.get("signals") or {}).items() if k == "resolved_by_evaluation"}
    return None


def _restore_flags(orch, ns, pedro_id, flags) -> None:
    if not flags:
        return
    for ep in orch.episodes.by_types(ns, ("exercise_attempt",)):
        ss = ep.store_specific or {}
        if pedro_id in (ss.get("chat_message_ids") or []):
            ss["signals"] = {**(ss.get("signals") or {}), **flags}
            ep.store_specific = ss
            orch.episodes._insert(ep)


def _set_time(orch, ns, pedro_id, when) -> None:
    if not when:
        return
    with connect_db(orch.episodes.db_path) as conn:
        conn.execute(
            "UPDATE episode_items SET created_at = ? WHERE namespace = ? AND id IN ("
            " SELECT e.id FROM episode_items e, json_each(e.store_specific, '$.chat_message_ids') j"
            " WHERE e.namespace = ? AND j.value = ?)",
            (when.replace(tzinfo=None).isoformat(timespec="seconds"), ns, ns, int(pedro_id)),
        )


def backfill(user_id: int, apply: bool, stats: Counter) -> set:
    orch = oma_provider._student_orchestrator()
    folders = set()
    for user_msg, pedro_msg in _pairs(user_id):
        folder = user_msg.context_id if user_msg.context_type in COURSE_CONTEXTS else None
        ns = course_namespace(user_id, folder) if folder else oma_provider.general_namespace(user_id)
        if folder:
            folders.add(folder)
        if oma_provider._is_lesson_intro(user_msg.content or ""):
            stats["skipped_openers"] += 1  # automatic section openers are never student work
            continue
        if orch.episodes.has_message(ns, pedro_msg.id):
            stats["already_recorded"] += 1
            continue
        if not apply:
            stats["would_record"] += 1
            continue
        flags = _take_legacy(orch, ns, user_msg) if folder else None
        oma_provider._record_conversation_turn(
            user_id, user_msg.context_type, folder, user_msg.content or "", pedro_msg.content or "",
            user_message_id=user_msg.id, pedro_message_id=pedro_msg.id, section_index=user_msg.section_index,
        )
        _set_time(orch, ns, pedro_msg.id, pedro_msg.created_at)
        _restore_flags(orch, ns, pedro_msg.id, flags)
        stats["replaced_legacy" if flags is not None else "recorded"] += 1
    return folders


def rebuild_mastery(user_id: int, ns: str, apply: bool, stats: Counter) -> None:
    orch = oma_provider._student_orchestrator()
    names = {(it.store_specific or {}).get("concept_id"): (it.store_specific or {}).get("concept_name")
             for it in orch.mastery.all(ns)}
    log = orch.episodes.by_types(ns, ("exercise_attempt", "section_evaluation"))
    stats["mastery_rows_before"] += len(names)
    if not apply:
        return
    for it in orch.mastery.all(ns):
        orch.mastery.delete(it.id)
    for ep in log:
        ss = ep.store_specific or {}
        if ss.get("episode_type") == "exercise_attempt":
            if ss.get("outcome") not in ("success", "mistake", "struggle"):
                continue
            for cid in ss.get("concept_ids") or []:
                orch.mastery.record_evidence(ns, cid, names.get(cid) or oma_provider._lookup_concept_name(
                    user_id, _folder_of(ns), cid), ss["outcome"], hinted=bool(ss.get("hinted")))
        else:
            for c in (ss.get("evaluation") or {}).get("concepts") or []:
                if c.get("concept_id") and c.get("final_state") not in (None, "not_touched"):
                    orch.mastery.apply_evaluator_verdict(ns, c["concept_id"], c.get("concept_name") or c["concept_id"],
                                                         c["final_state"], section_index=ss.get("section_index"))
    stats["mastery_rows_after"] += orch.mastery.count(ns)


def _folder_of(ns: str) -> str:
    from coast_content_oma.student.stores import parse_course_namespace
    return parse_course_namespace(ns)[1] or ""


def clean_traits(user_id: int, apply: bool, stats: Counter) -> None:
    orch = oma_provider._student_orchestrator()
    for it in orch.identity.all(identity_namespace(user_id)):
        if not is_displayable_trait(it.content):
            stats["malformed_traits"] += 1
            if apply:
                orch.identity.delete(it.id)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user-id", type=int)
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = ap.parse_args()

    if not oma_provider.is_student_enabled():
        print("Student OMA is disabled (STUDENT_OMA_ENABLED / RAG_PROVIDER). Nothing to do.")
        return 1
    db_path = Path(oma_provider.OMA_DB_PATH)
    if args.apply and db_path.exists():
        backup = db_path.with_name(f"{db_path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        with connect_db(db_path) as conn:
            conn.execute("PRAGMA wal_checkpoint(FULL)")
        shutil.copy2(db_path, backup)
        print(f"Backed up {db_path} -> {backup}")

    with SessionLocal() as db:
        users = [args.user_id] if args.user_id else [
            r[0] for r in db.query(ChatMessage.user_id).distinct().order_by(ChatMessage.user_id)]

    total = Counter()
    for uid in users:
        stats = Counter()
        folders = backfill(uid, args.apply, stats)
        namespaces = set(list_course_namespaces(db_path, uid)) | {course_namespace(uid, f) for f in folders}
        for ns in sorted(namespaces):
            rebuild_mastery(uid, ns, args.apply, stats)
        clean_traits(uid, args.apply, stats)
        if args.apply:
            for f in sorted(folders):
                oma_provider._run_course_consolidation(uid, f)
        print(f"user {uid}: {dict(stats)}")
        total.update(stats)
    print(f"\n{'APPLIED' if args.apply else 'DRY RUN'} total: {dict(total)}")
    if not args.apply:
        print("Re-run with --apply to write (oma.db is backed up first).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
