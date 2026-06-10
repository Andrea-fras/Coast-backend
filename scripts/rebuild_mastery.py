"""Rebuild ConceptMasteryStore from the episode log using current rules.

Wipes per-course mastery rows and replays episodes chronologically:
  - neutral episodes do not touch mastery
  - concept refs inferred from turn text only (no bulk OMA ids)
  - outcomes reclassified with current tutor heuristics

Usage:
  python3 scripts/rebuild_mastery.py --user-id 11 --folder pop
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
from coast_content_oma.student.stores import course_namespace  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", type=int, required=True)
    ap.add_argument("--folder", required=True)
    args = ap.parse_args()

    ns = course_namespace(args.user_id, args.folder)
    db = Path(os.environ.get("OMA_DB_PATH", "./oma_data/oma.db"))

    orch = oma_provider._student_orchestrator()
    rec = oma_provider._student_recorder_singleton()

    # Wipe mastery for this course.
    with sqlite3.connect(db) as conn:
        n = conn.execute(
            "DELETE FROM concept_mastery_items WHERE namespace = ?", (ns,)
        ).rowcount
    print(f"Cleared {n} mastery rows for {ns}")

    # Replay episodes oldest-first.
    episodes = list(orch.episodes.all(ns))
    episodes.sort(key=lambda it: it.created_at)
    print(f"Replaying {len(episodes)} episodes...")

    updates = 0
    for ep in episodes:
        ss = ep.store_specific or {}
        etype = ss.get("episode_type", "")
        if etype not in ("qa", "exercise_attempt", "section_completed"):
            continue
        um = ss.get("user_message") or ""
        ar = ss.get("assistant_response") or ""
        if oma_provider._is_lesson_intro(um):
            continue

        if etype == "section_completed":
            continue  # logged for accomplishments, no mastery

        outcome, _signals = oma_provider._classify_outcome_and_signals(um, ar)
        if outcome not in ("success", "struggle"):
            continue

        refs = oma_provider._infer_concept_refs_from_text(
            args.user_id, args.folder, um, ar, outcome,
        )
        for c in refs:
            rec.mastery.record_evidence(
                ns,
                concept_id=c["concept_id"],
                concept_name=c["concept_name"],
                outcome=outcome,
            )
            updates += 1

    print(f"Applied {updates} mastery evidence updates.")
    overview = orch.mastery.overview(ns)
    print("Overview:", overview)
    profile = orch.build_profile(args.user_id, args.folder)
    print()
    print(orch.to_prompt_block(profile))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
