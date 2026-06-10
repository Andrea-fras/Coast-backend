#!/usr/bin/env python3
"""Compare Content OMA vs flat RAG for a folder query.

Examples:
  python3 scripts/compare_retrieval.py --folder ttttesst --user-id 11 \\
    --query "OpenMP worksharing parallel for directive"

  # Use the current lesson section's topics as the query:
  python3 scripts/compare_retrieval.py --folder ttttesst --user-id 11 --from-lesson
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import rag  # noqa: E402
import oma_provider  # noqa: E402
from database import CourseOutline, SessionLocal  # noqa: E402


def _section_query(user_id: int, folder: str) -> str:
    db = SessionLocal()
    try:
        outline = (
            db.query(CourseOutline)
            .filter(CourseOutline.user_id == user_id, CourseOutline.folder_name == folder)
            .first()
        )
        if not outline:
            raise SystemExit(f"No lesson outline for user {user_id} folder {folder!r}")
        sections = json.loads(outline.outline_json)
        idx = outline.current_section
        sec = sections[idx]
        parts = [sec.get("title", "")] + sec.get("key_topics", []) + sec.get("learning_objectives", [])
        return " ".join(p for p in parts if p)
    finally:
        db.close()


def _banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    p = argparse.ArgumentParser(description="Compare OMA vs flat RAG retrieval")
    p.add_argument("--folder", required=True)
    p.add_argument("--user-id", type=int, default=11)
    p.add_argument("--query", help="Search query (omit with --from-lesson)")
    p.add_argument("--from-lesson", action="store_true", help="Build query from current lesson section")
    p.add_argument("--max-chars", type=int, default=14000)
    args = p.parse_args()

    if args.from_lesson:
        query = _section_query(args.user_id, args.folder)
    elif args.query:
        query = args.query
    else:
        p.error("Provide --query or --from-lesson")

    print(f"RAG_PROVIDER={oma_provider.RAG_PROVIDER}")
    print(f"Folder={args.folder!r}  user={args.user_id}")
    print(f"Query: {query}")

    t0 = time.time()
    oma_block, concept_ids = oma_provider.get_folder_context(
        args.user_id, args.folder, query, max_chars=args.max_chars,
        max_content=12, max_images=6,
    )
    oma_ms = (time.time() - t0) * 1000

    t0 = time.time()
    rag_block = rag.build_folder_context(
        args.user_id, args.folder, query, max_chars=args.max_chars,
    )
    rag_ms = (time.time() - t0) * 1000

    if oma_provider.is_oma_enabled() and oma_block:
        lesson_uses = "content_oma"
    elif rag_block:
        lesson_uses = "flat_rag"
    else:
        lesson_uses = "fallback_raw_text"

    print(f"\nLesson chat would use: {lesson_uses}")

    _banner(f"FLAT RAG ({len(rag_block)} chars, {rag_ms:.0f}ms)")
    print(rag_block[:3000] or "(empty)")
    if len(rag_block) > 3000:
        print(f"\n... [{len(rag_block) - 3000} more chars]")

    _banner(f"CONTENT OMA ({len(oma_block)} chars, {oma_ms:.0f}ms, {len(concept_ids)} concepts)")
    print(oma_block[:3000] or "(empty)")
    if len(oma_block) > 3000:
        print(f"\n... [{len(oma_block) - 3000} more chars]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
