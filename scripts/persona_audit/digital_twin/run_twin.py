#!/usr/bin/env python3
"""Run a digital-twin student through real Coast lessons.

Drives the real backend (tutor.send_message + lesson.advance_section + the
post-section evaluator/consolidator) as a role-played student with evolving
mastery/frustration/engagement state. Writes real Student OMA interaction data
across multiple sections, exactly like a real student, then scores the arc.

Usage:
  RAG_PROVIDER=oma STUDENT_OMA_ENABLED=true python3 -m scripts.persona_audit.digital_twin.run_twin \\
    --user-id 14 --folder Operations --persona anxious_step_by_step \\
    --sections 3 --max-turns 8

Personas: anxious_step_by_step, overconfident_speedrunner, visual_learner, foundational_struggler

The run record (transcript + agent state) is written to
  scripts/persona_audit/digital_twin/last_twin_run.json
and a human-readable metrics summary is printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

os.environ.setdefault("RAG_PROVIDER", "oma")
os.environ.setdefault("STUDENT_OMA_ENABLED", "true")

from scripts.persona_audit.digital_twin.twin_runner import run_twin
from scripts.persona_audit.digital_twin.scoring import score_run, format_metrics
from scripts.persona_audit.digital_twin.personas import list_personas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", type=int, required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--persona", required=True, choices=list_personas())
    ap.add_argument("--sections", type=int, default=3, help="number of sections to run from the current section")
    ap.add_argument("--max-turns", type=int, default=8, help="max student turns per section")
    ap.add_argument("--sim-model", default="", help="simulator LLM model (default: gemini-2.5-flash via TWIN_SIM_MODEL)")
    ap.add_argument("--no-force-advance", action="store_true", help="don't force-advance sections Pedro didn't complete")
    ap.add_argument("--out", default="", help="path to write the run record JSON")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else (
        ROOT / "scripts" / "persona_audit" / "digital_twin" / "last_twin_run.json"
    )

    record = run_twin(
        user_id=args.user_id,
        folder=args.folder,
        persona_id=args.persona,
        sections=args.sections,
        max_turns=args.max_turns,
        sim_model=args.sim_model or None,
        force_advance=not args.no_force_advance,
        out_path=out_path,
        verbose=not args.quiet,
    )

    metrics = score_run(record)
    print("\n" + "=" * 70)
    print(format_metrics(metrics))
    print("=" * 70)

    metrics_path = out_path.with_name(out_path.stem + "_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nRun record: {out_path}\nMetrics:     {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
