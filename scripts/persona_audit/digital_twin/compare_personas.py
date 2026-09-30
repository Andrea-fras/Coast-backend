#!/usr/bin/env python3
"""Run multiple personas on the SAME section back-to-back and render a
side-by-side comparison of Pedro's responses.

Drives the real tutor.send_message (real system prompt, real OMA retrieval,
real Student OMA profile) but does NOT advance the course or trigger the
post-section pipeline — so it's safe and non-destructive: current_section is
untouched. Each persona gets its own conversation_id on the chosen section.

This is the fastest way to SEE how Pedro adapts to different students. The
profile block is dominated by the account's long history, so openers will be
similar; the divergence shows up in the turns as Pedro reacts to each
student's actual utterances.

Usage:
  RAG_PROVIDER=oma STUDENT_OMA_ENABLED=true \\
  python3 -m scripts.persona_audit.digital_twin.compare_personas \\
    --user-id 14 --folder Operations --section 5 \\
    --personas anxious_step_by_step,overconfident_speedrunner --turns 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
os.environ.setdefault("RAG_PROVIDER", "oma")
os.environ.setdefault("STUDENT_OMA_ENABLED", "true")

from scripts.persona_audit.digital_twin.student_agent import StudentAgent
from scripts.persona_audit.digital_twin.personas import list_personas, get_persona


def _section_info(user_id: int, folder: str, section_index: int) -> tuple[str, list[str], list[str]]:
    import json as _json
    import lesson as lesson_mod
    from database import SessionLocal, CourseOutline
    db = SessionLocal()
    try:
        outline = (
            db.query(CourseOutline)
            .filter(CourseOutline.user_id == user_id, CourseOutline.folder_name == folder)
            .first()
        )
        if not outline:
            raise RuntimeError(f"No outline for user={user_id} folder={folder!r}")
        sections = _json.loads(outline.outline_json)
        if section_index < 0 or section_index >= len(sections):
            raise RuntimeError(f"section {section_index} out of range (0..{len(sections) - 1})")
        title = sections[section_index].get("title") or f"Section {section_index + 1}"
    finally:
        db.close()
    refs = lesson_mod.get_section_concept_refs(user_id, folder, section_index)
    names, ids = [], []
    for r in refs:
        cid = r.get("concept_id")
        name = r.get("concept_name") or r.get("name") or (cid or "")
        if cid:
            ids.append(cid)
        if name:
            names.append(name)
    return title, names, ids


def run_persona_on_section(user_id, folder, section_index, persona_id, turns, sim_model=None) -> dict:
    import oma_provider
    from tutor import send_message

    title, concept_names, concept_ids = _section_info(user_id, folder, section_index)
    primary_concept_id = concept_ids[0] if concept_ids else None
    agent = StudentAgent(persona_id, sim_model=sim_model)
    agent.start_section(section_index, title, concept_names)
    persona = get_persona(persona_id)
    conv_id = f"cmp_{persona_id}_s{section_index}_{uuid.uuid4().hex[:6]}"

    transcript: list[dict] = []
    opener = f'I\'m ready to learn about "{title}". Please teach me this section.'
    res = send_message(user_id=user_id, message=opener, conversation_id=conv_id,
                       context_type="lesson", context_id=folder, section_index=section_index)
    pedro_reply = res.get("reply") or ""
    pedro_clean = oma_provider.strip_pedro_tags(pedro_reply)
    transcript.append({"turn": 0, "role": "student", "content": opener})
    transcript.append({"turn": 0, "role": "pedro", "content": pedro_clean})

    for t in range(1, turns + 1):
        recent = "\n".join(
            f"Pedro: {x['content'][:300]}" if x["role"] == "pedro"
            else f"{persona['name']}: {x['content'][:200]}"
            for x in transcript[-4:]
        )
        student_reply, update = agent.respond(pedro_clean, recent_turns=recent)
        if not student_reply or not student_reply.strip():
            break
        transcript.append({"turn": t, "role": "student", "content": student_reply, "state": update})
        res = send_message(user_id=user_id, message=student_reply, conversation_id=conv_id,
                           context_type="lesson", context_id=folder, section_index=section_index,
                           concept_id=primary_concept_id)
        pedro_reply = res.get("reply") or ""
        pedro_clean = oma_provider.strip_pedro_tags(pedro_reply)
        transcript.append({"turn": t, "role": "pedro", "content": pedro_clean})

    return {
        "persona": persona_id, "persona_name": persona["name"],
        "section_index": section_index, "section_title": title,
        "concepts": concept_names, "transcript": transcript,
        "agent_state": agent.to_state(),
    }


def render_side_by_side(runs: list[dict]) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append(f"SIDE-BY-SIDE — section {runs[0]['section_index']}: {runs[0]['section_title']!r}")
    lines.append(f"concepts: {', '.join(runs[0]['concepts'][:6])}")
    lines.append("=" * 78)
    max_turns = max(len([m for m in r["transcript"] if m["role"] == "student"]) for r in runs)
    for turn in range(0, max_turns + 1):
        for r in runs:
            persona_label = f"{r['persona_name']} ({r['persona']})"
            student_msgs = [m for m in r["transcript"] if m["role"] == "student" and m["turn"] == turn]
            pedro_msgs = [m for m in r["transcript"] if m["role"] == "pedro" and m["turn"] == turn]
            if not student_msgs and not pedro_msgs:
                continue
            lines.append("")
            lines.append(f"--- [{persona_label}]  turn {turn} ---")
            if student_msgs:
                lines.append(f"{r['persona_name']}: {student_msgs[0]['content']}")
            if pedro_msgs:
                pc = pedro_msgs[0]["content"]
                lines.append(f"Pedro: {pc[:900]}{'…' if len(pc) > 900 else ''}")
        lines.append("")
        lines.append("-" * 78)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", type=int, required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--section", type=int, required=True, help="section index to run all personas on")
    ap.add_argument("--personas", required=True, help="comma-separated persona ids")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--sim-model", default="")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    persona_ids = [p.strip() for p in args.personas.split(",") if p.strip()]
    for pid in persona_ids:
        if pid not in list_personas():
            print(f"unknown persona {pid!r}; known: {list_personas()}")
            return 1

    out_dir = Path(args.out_dir) if args.out_dir else (
        ROOT / "scripts" / "persona_audit" / "digital_twin"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for pid in persona_ids:
        print(f"\n>>> running persona {pid} on section {args.section}...")
        run = run_persona_on_section(
            args.user_id, args.folder, args.section, pid, args.turns,
            sim_model=args.sim_model or None,
        )
        runs.append(run)
        per_path = out_dir / f"compare_{pid}_s{args.section}.json"
        per_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
        print(f"    wrote {per_path}")

    print()
    print(render_side_by_side(runs))

    cmp_path = out_dir / f"compare_s{args.section}_{'_'.join(persona_ids)}.json"
    cmp_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"\nCombined: {cmp_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
