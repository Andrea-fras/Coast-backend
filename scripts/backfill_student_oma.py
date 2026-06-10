"""Backfill Student OMA from coast.db using the tag-based architecture.

Records only:
  - exercise_attempt episodes (from [ANSWER_WRONG]/[ANSWER_CORRECT] tags, or
    heuristic fallback for legacy chat before tags existed)
  - section_completed episodes from CourseOutline (authoritative progress)

Does NOT log teaching chat — that was the main source of junk.

Usage:
  python3 scripts/backfill_student_oma.py --user-id 13 --folder prisma --wipe
  python3 scripts/backfill_student_oma.py --user-id 13 --folder prisma --dry-run
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv()

import oma_provider  # noqa: E402
import lesson  # noqa: E402
from coast_content_oma.student.stores import course_namespace  # noqa: E402


def _wipe_course(user_id: int, folder: str) -> None:
    ns = course_namespace(user_id, folder)
    db_path = Path(os.environ.get("OMA_DB_PATH", "./oma_data/oma.db"))
    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for table in ("episode_items", "concept_mastery_items", "pattern_items"):
            if table not in tables:
                continue
            n = conn.execute(
                f"DELETE FROM {table} WHERE namespace = ?", (ns,)
            ).rowcount
            print(f"Cleared {n} rows from {table} for {ns}")


def _pair_messages(rows: list[tuple]) -> list[tuple[dict, dict]]:
    pairs = []
    pending_user_by_conv: dict[str, dict] = {}
    for r in rows:
        msg = {
            "id": r[0], "conv": r[1], "role": r[2], "ctx_type": r[3],
            "ctx_id": r[4], "content": r[5], "created_at": r[6],
            "section_index": r[7],
        }
        if msg["role"] == "user":
            pending_user_by_conv[msg["conv"]] = msg
        elif msg["role"] == "pedro":
            user_msg = pending_user_by_conv.pop(msg["conv"], None)
            if user_msg:
                pairs.append((user_msg, msg))
    return pairs


def _resolve_graded_outcome(user_content: str, pedro_content: str) -> str | None:
    tagged = oma_provider._answer_outcome_from_tags(pedro_content)
    if tagged == "struggle":
        return "struggle"
    if tagged == "success":
        return None
    # Legacy chat (pre-tags): heuristic fallback for migration only.
    outcome, _signals = oma_provider._classify_outcome_and_signals(
        user_content, pedro_content,
    )
    if outcome == "struggle":
        return "struggle"
    return None


def _backfill_sections(user_id: int, folder: str, dry_run: bool) -> int:
    progress = lesson.get_authoritative_progress(user_id, folder)
    if not progress:
        return 0
    n = 0
    for sec in progress.get("completed_sections") or []:
        idx = sec.get("index")
        title = sec.get("title") or f"Section {(idx or 0) + 1}"
        print(f"  [section {idx}] completed: {title!r}")
        if dry_run:
            n += 1
            continue
        oma_provider.record_section_completed_authoritative(
            user_id, folder, section_index=int(idx), section_title=title,
        )
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", type=int, required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--since", help="optional YYYY-MM-DD lower bound on created_at")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--wipe", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(ROOT / "coast.db")
    q = (
        "SELECT id, conversation_id, role, context_type, context_id, content, "
        "created_at, section_index "
        "FROM chat_messages WHERE user_id=? AND context_id=? "
        "AND context_type IN ('folder','lesson') "
    )
    params: list = [args.user_id, args.folder]
    if args.since:
        q += "AND date(created_at) >= date(?) "
        params.append(args.since)
    q += "ORDER BY id ASC"

    rows = conn.execute(q, params).fetchall()
    pairs = _pair_messages(rows)
    print(f"Found {len(rows)} messages, {len(pairs)} user/pedro pairs.")

    if not oma_provider.is_student_enabled():
        print("Student OMA is not enabled. Aborting.")
        return

    if args.wipe and not args.dry_run:
        _wipe_course(args.user_id, args.folder)

    counts = {"struggle": 0, "skipped_intro": 0, "skipped_teaching": 0, "skipped_correct": 0}

    for i, (umsg, pmsg) in enumerate(pairs, 1):
        content = umsg["content"] or ""
        pedro = pmsg["content"] or ""
        if oma_provider._is_lesson_intro(content):
            counts["skipped_intro"] += 1
            continue

        outcome = _resolve_graded_outcome(content, pedro)
        if outcome is None:
            counts["skipped_teaching"] += 1
            continue

        counts[outcome] = counts.get(outcome, 0) + 1
        preview = content.replace("\n", " ")[:60]
        tag = "tagged" if oma_provider._answer_outcome_from_tags(pedro) else "heuristic"
        print(
            f"  [#{i:>3}] {outcome:>8} ({tag}) s{umsg.get('section_index')} "
            f"user={preview!r}"
        )

        if args.dry_run:
            continue

        oma_provider._record_graded_mistake(
            args.user_id, args.folder, content, pedro,
            section_index=umsg.get("section_index"),
        )

    print()
    print(f"Mistakes recorded: {counts}")
    print("Backfilling section progress from CourseOutline...")
    n_sec = _backfill_sections(args.user_id, args.folder, args.dry_run)
    print(f"Section completions recorded: {n_sec}")
    print("Done." if not args.dry_run else "Dry run — nothing written.")


if __name__ == "__main__":
    main()
