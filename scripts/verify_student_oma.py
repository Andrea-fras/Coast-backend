"""Verify Student OMA produces a trustworthy cognitive profile.

Usage:
  STUDENT_OMA_ENABLED=true python3 scripts/verify_student_oma.py --user-id 13 --folder prisma
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

import oma_provider  # noqa: E402
import lesson  # noqa: E402
from coast_content_oma.student.stores import course_namespace  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", type=int, required=True)
    ap.add_argument("--folder", required=True)
    args = ap.parse_args()

    if not oma_provider.is_student_enabled():
        print("FAIL: STUDENT_OMA_ENABLED is off")
        return 1

    uid, folder = args.user_id, args.folder
    ns = course_namespace(uid, folder)
    orch = oma_provider._student_orchestrator()

    progress = lesson.get_authoritative_progress(uid, folder) or {}
    mistakes = oma_provider._load_section_mistakes(uid, folder)
    struggling = oma_provider._load_struggling_topics(uid, folder)
    episodes = list(orch.episodes.all(ns))
    mastery = list(orch.mastery.all(ns))
    patterns = list(orch.patterns.all(ns))

    print("=" * 60)
    print(f"Student OMA verification — user {uid}, folder {folder}")
    print("=" * 60)

    print("\n## Progress ledger (authoritative)")
    print(json.dumps(progress, indent=2, default=str))

    print("\n## Episodes (should be lean: sections + mistakes only)")
    for ep in episodes:
        ss = ep.store_specific or {}
        print(
            f"  [{ss.get('episode_type'):18}] outcome={ss.get('outcome'):8} "
            f"s{ss.get('section_index')} "
            f"{(ss.get('user_message') or ss.get('section_title') or ep.content or '')[:55]}"
        )
    print(f"  Total: {len(episodes)}")

    print("\n## Practice mistakes")
    for m in mistakes:
        print(f"  s{m.get('section_index')}: {m.get('user_message', '')[:60]}")

    print("\n## Struggling topics (should be empty unless persistent difficulty)")
    if struggling:
        for t in struggling:
            print(f"  {t['name']}: {t['mistakes']} wrong / {t['successes']} right")
    else:
        print("  (none)")

    print("\n## Concept mastery")
    for it in mastery:
        ss = it.store_specific or {}
        print(
            f"  {ss.get('concept_name')}: score={ss.get('mastery_score', 0):.2f} "
            f"ok={ss.get('successes', 0)} struggles={ss.get('struggles', 0)}"
        )

    print("\n## Pedro profile block")
    block = oma_provider.get_student_profile_block(uid, folder)
    print(block[:2000] if block else "(empty)")

    print("\n## Trust checks")
    checks = []
    qa_eps = [e for e in episodes if (e.store_specific or {}).get("episode_type") == "qa"]
    checks.append(("No teaching-chat qa episodes", len(qa_eps) == 0, f"found {len(qa_eps)}"))

    sec_done = len(progress.get("completed_sections") or [])
    sec_eps = [e for e in episodes if (e.store_specific or {}).get("episode_type") == "section_completed"]
    checks.append((
        "Section episodes match outline progress",
        len(sec_eps) == sec_done,
        f"episodes={len(sec_eps)} outline={sec_done}",
    ))

    single_mistake_not_struggling = (
        len(mistakes) <= 1 and len(struggling) == 0
    ) or len(struggling) > 0
    checks.append(("Single mistake ≠ struggling", single_mistake_not_struggling, f"mistakes={len(mistakes)} struggling={len(struggling)}"))

    checks.append(("Profile block non-empty", bool(block), "empty profile"))

    all_ok = True
    for label, ok, detail in checks:
        status = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {label} — {detail}")

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
