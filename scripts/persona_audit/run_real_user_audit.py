#!/usr/bin/env python3
"""Run personalization audit against a REAL user + folder (populated Content + Student OMA).

Uses the same message shapes the frontend sends (LessonView section opener, etc.).

Usage:
  RAG_PROVIDER=oma STUDENT_OMA_ENABLED=true python3 scripts/persona_audit/run_real_user_audit.py \\
    --user-id 14 --folder Operations
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import oma_provider
import lesson as lesson_mod
from database import SessionLocal, CourseOutline, User
from tutor import send_message
from scripts.persona_audit.run_audit import _llm_relevance_judge, _profile_body


def _current_section(user_id: int, folder: str) -> tuple[int, str, list]:
    db = SessionLocal()
    try:
        outline = (
            db.query(CourseOutline)
            .filter(CourseOutline.user_id == user_id, CourseOutline.folder_name == folder)
            .first()
        )
        if not outline:
            return 0, "Section 1", []
        import json as _json
        sections = _json.loads(outline.outline_json)
        idx = int(outline.current_section)
        if idx >= len(sections):
            idx = max(0, len(sections) - 1)
        title = sections[idx].get("title") or f"Section {idx + 1}"
        return idx, title, sections
    finally:
        db.close()


def build_real_scenarios(user_id: int, folder: str) -> list[dict]:
    """Mirror the 5 audit scenario types with real course context."""
    sec_idx, sec_title, sections = _current_section(user_id, folder)
    concept_refs = lesson_mod.get_section_concept_refs(user_id, folder, sec_idx)
    concept_ids = [r["concept_id"] for r in concept_refs if r.get("concept_id")][:6]

    # Prior section title for bridge expectations
    prior_title = ""
    if sec_idx > 0 and sec_idx - 1 < len(sections):
        prior_title = sections[sec_idx - 1].get("title") or ""

    return [
        {
            "id": "profile_injection_baseline",
            "description": "Review ask mid-course (lesson chat, current section)",
            "context_type": "lesson",
            "message": "What should I review before we continue?",
            "section_index": sec_idx,
            "concept_ids": concept_ids,
            "relevance_topics": ["prior sections completed", "what to review next"],
        },
        {
            "id": "concept_intuition",
            "description": "Struggling concept intuition (folder chat — student-initiated)",
            "context_type": "folder",
            "message": "Explain the birth-death process to me — I keep losing the intuition.",
            "section_index": None,
            "concept_ids": concept_ids,
            "relevance_topics": ["birth-death process", "student struggle history if any"],
        },
        {
            "id": "no_false_struggle",
            "description": "Self-check on a topic with mixed mastery",
            "context_type": "folder",
            "message": "Quick check: am I still bad at holding times in queueing?",
            "section_index": None,
            "concept_ids": concept_ids,
            "relevance_topics": ["holding time", "accurate not harsh about past mistakes"],
        },
        {
            "id": "queueing_stability",
            "description": "Classic queueing stability question",
            "context_type": "folder",
            "message": "In an M/M/1 queue, can ρ be 1.2 if arrivals are heavy?",
            "section_index": None,
            "concept_ids": concept_ids,
            "relevance_topics": ["utilization rho", "stability condition"],
        },
        {
            "id": "lesson_section_intro",
            "description": "Exact LessonView.jsx section opener",
            "context_type": "lesson",
            "message": f'I\'m ready to learn about "{sec_title}". Please teach me this section.',
            "section_index": sec_idx,
            "concept_ids": concept_ids,
            "relevance_topics": [
                f"section intro for {sec_title}",
                f"bridge from {prior_title}" if prior_title else "course intro",
            ],
        },
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", type=int, required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    os.environ.setdefault("RAG_PROVIDER", "oma")
    os.environ.setdefault("STUDENT_OMA_ENABLED", "true")

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == args.user_id).first()
        if not user:
            print(f"User {args.user_id} not found")
            return 1
        user_name = user.name
    finally:
        db.close()

    sec_idx, sec_title, _ = _current_section(args.user_id, args.folder)
    scenarios = build_real_scenarios(args.user_id, args.folder)

    print(f"Real-user audit: {user_name} (id={args.user_id}) folder={args.folder!r}")
    print(f"Current section: {sec_idx} — {sec_title!r}")
    print(f"Scenarios: {len(scenarios)}\n")

    rows = []
    for sc in scenarios:
        block = oma_provider.get_student_profile_block(
            args.user_id,
            args.folder,
            current_concept_ids=sc.get("concept_ids"),
            max_chars=5000,
        ) or "(empty)"
        profile = _profile_body(block).strip()

        conv = f"real_{args.user_id}_{sc['id']}_{uuid.uuid4().hex[:6]}"
        t0 = time.perf_counter()
        try:
            result = send_message(
                user_id=args.user_id,
                message=sc["message"],
                conversation_id=conv,
                context_type=sc["context_type"],
                context_id=args.folder,
                section_index=sc.get("section_index"),
            )
            reply = result.get("reply") or ""
            err = None
        except Exception as exc:
            reply = ""
            err = str(exc)
        elapsed = round(time.perf_counter() - t0, 1)

        judge_score = None
        if args.judge and reply and sc.get("relevance_topics"):
            judge_score = _llm_relevance_judge(
                sc["message"], reply, sc["relevance_topics"],
            )

        row = {
            "user_id": args.user_id,
            "user_name": user_name,
            "folder": args.folder,
            "scenario_id": sc["id"],
            "description": sc["description"],
            "context_type": sc["context_type"],
            "section_index": sc.get("section_index"),
            "question": sc["message"],
            "profile_used": profile,
            "profile_chars": len(profile),
            "reply": reply,
            "reply_chars": len(reply),
            "elapsed_sec": elapsed,
            "judge_score": judge_score,
            "error": err,
        }
        rows.append(row)
        js = f" judge={judge_score}/5" if judge_score else ""
        print(f"  [{sc['id']}] {elapsed}s reply={len(reply)} chars{js}")

    out_path = Path(args.json_out) if args.json_out else (
        ROOT / "scripts/persona_audit/last_real_user_report.json"
    )
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
