#!/usr/bin/env python3
"""Verify post-section evaluator (Phase 2 write path).

Usage:
  RAG_PROVIDER=oma STUDENT_OMA_ENABLED=true python3 scripts/verify_section_evaluator.py \\
    --user-id 17 --folder letsodit --section-index 0

  # Heuristic-only (no LLM):
  python3 scripts/verify_section_evaluator.py --user-id 17 --folder letsodit --section-index 0 --heuristic
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv()

import evaluator as evaluator_mod  # noqa: E402
import oma_provider  # noqa: E402
import lesson as lesson_mod  # noqa: E402
from coast_content_oma.student.stores import course_namespace  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", type=int, required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--section-index", type=int, default=0)
    ap.add_argument("--section-title", default="")
    ap.add_argument("--heuristic", action="store_true", help="Skip LLM; use tag heuristic only")
    ap.add_argument("--force", action="store_true", help="Re-run even if already evaluated")
    args = ap.parse_args()

    if not oma_provider.is_student_enabled():
        print("FAIL: STUDENT_OMA_ENABLED is off")
        return 1

    uid, folder, idx = args.user_id, args.folder, args.section_index
    ns = course_namespace(uid, folder)

    messages, transcript = evaluator_mod.fetch_section_transcript(uid, folder, idx)
    print(f"Transcript: {len(messages)} messages, {len(transcript)} chars")
    if not messages:
        print("FAIL: no section chat found")
        return 1

    concept_refs = lesson_mod.get_section_concept_refs(uid, folder, idx)
    print(f"Section concepts: {len(concept_refs)}")

    if args.heuristic:
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("OPENAI_API_KEY", None)

    if args.force:
        result = evaluator_mod.run_section_evaluation(
            uid, folder, idx, args.section_title, force=True,
        )
    else:
        result = evaluator_mod.run_section_evaluation(
            uid, folder, idx, args.section_title,
        )

    if not result:
        print("No evaluation applied (already done or empty)")
        return 0

    print("\n## Evaluation result")
    print(json.dumps(result, indent=2))

    orch = oma_provider._student_orchestrator()
    eval_eps = [
        ep for ep in orch.episodes.for_section(ns, idx)
        if (ep.store_specific or {}).get("episode_type") == "section_evaluation"
    ]
    print(f"\nsection_evaluation episodes: {len(eval_eps)}")
    if eval_eps:
        ev = (eval_eps[-1].store_specific or {}).get("evaluation") or {}
        print(json.dumps(ev, indent=2)[:2500])

    golden = [
        p for p in orch.patterns.all(ns)
        if (p.store_specific or {}).get("pattern_type") == "golden_moment"
    ]
    print(f"\ngolden_moment patterns: {len(golden)}")
    for g in golden[:3]:
        print(f"  - {g.content[:120]}")

    block = oma_provider.get_student_profile_block(uid, folder)
    if block and "Golden moments" in block:
        print("\nOK: profile block includes golden moments")
    elif golden:
        print("\nWARN: golden patterns exist but not in profile block snippet")

    mistakes = oma_provider._load_section_mistakes(uid, folder)
    unresolved = [m for m in mistakes if m.get("section_index") == idx]
    print(f"\nUnresolved mistakes in section {idx}: {len(unresolved)}")

    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
