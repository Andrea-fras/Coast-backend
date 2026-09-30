"""Twin runner — drives the real Coast backend as a role-played student across
multiple sections of a folder.

For each section:
  1. Send the LessonView.jsx section opener ("I'm ready to learn about ...").
  2. Loop turns: the StudentAgent replies to Pedro; Pedro replies via the real
     tutor.send_message (real system prompt, real retrieval, real Student OMA
     writes, real capture-tag parsing).
  3. Stop when Pedro emits [SECTION_COMPLETE], the agent gives up, or the turn
     cap is hit.
  4. Advance the section via lesson.advance_section (triggers the real
     post-section evaluator + course/identity consolidation) and wait for that
     pipeline to settle so the NEXT section's profile reflects this one.

Output: a JSON run record (transcript + agent state + section summaries) for
scoring.py to turn into metrics.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Optional

from .student_agent import StudentAgent


def _load_sections(user_id: int, folder: str) -> tuple[int, list[dict]]:
    from database import SessionLocal, CourseOutline
    db = SessionLocal()
    try:
        outline = (
            db.query(CourseOutline)
            .filter(CourseOutline.user_id == user_id, CourseOutline.folder_name == folder)
            .first()
        )
        if not outline:
            return 0, []
        sections = json.loads(outline.outline_json)
        idx = int(outline.current_section)
        if idx >= len(sections):
            idx = max(0, len(sections) - 1)
        return idx, sections
    finally:
        db.close()


def _section_concepts(user_id: int, folder: str, section_index: int) -> tuple[list[str], list[str]]:
    """Return (concept_names, concept_ids) for a section."""
    import lesson as lesson_mod
    refs = lesson_mod.get_section_concept_refs(user_id, folder, section_index)
    names, ids = [], []
    for r in refs:
        cid = r.get("concept_id")
        name = r.get("concept_name") or r.get("name") or (cid or "")
        if cid:
            ids.append(cid)
        if name:
            names.append(name)
    return names, ids


def _wait_for_post_section_pipeline(user_id: int, folder: str, timeout: float = 90.0) -> bool:
    """Block until the async post-section evaluator + consolidator finish."""
    import oma_provider
    key = oma_provider._ingest_key(user_id, folder)
    time.sleep(1.0)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if key not in oma_provider._consolidation_active:
            return True
        time.sleep(1.5)
    return False


def _recent_turns_text(agent: StudentAgent, n: int = 4) -> str:
    tail = agent.turn_log[-n:]
    lines = []
    for t in tail:
        lines.append(f"Pedro: {t.get('pedro','')[:300]}")
        lines.append(f"{agent.persona['name']}: {t.get('student','')[:200]}")
    return "\n".join(lines)


def run_twin(
    user_id: int,
    folder: str,
    persona_id: str,
    sections: int = 3,
    max_turns: int = 8,
    sim_model: Optional[str] = None,
    force_advance: bool = True,
    out_path: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """Run a persona through `sections` sections of `folder` on the real backend."""
    import lesson as lesson_mod
    import oma_provider
    from tutor import send_message

    if not oma_provider.is_student_enabled():
        raise RuntimeError("STUDENT_OMA_ENABLED must be true for the twin to write real history.")

    agent = StudentAgent(persona_id, sim_model=sim_model)
    current_idx, all_sections = _load_sections(user_id, folder)
    if not all_sections:
        raise RuntimeError(f"No outline found for user={user_id} folder={folder!r}")

    run_id = f"twin_{user_id}_{persona_id}_{uuid.uuid4().hex[:6]}"
    transcript: list[dict] = []
    end_idx = min(len(all_sections), current_idx + max(1, sections))

    if verbose:
        print(f"[twin] run_id={run_id} persona={persona_id} user={user_id} folder={folder!r}")
        print(f"[twin] sections {current_idx}..{end_idx - 1} of {len(all_sections)}  max_turns={max_turns}")

    for section_index in range(current_idx, end_idx):
        sec = all_sections[section_index]
        title = sec.get("title") or f"Section {section_index + 1}"
        concept_names, concept_ids = _section_concepts(user_id, folder, section_index)
        primary_concept_id = concept_ids[0] if concept_ids else None
        agent.start_section(section_index, title, concept_names)

        if verbose:
            print(f"\n[twin] === Section {section_index}: {title!r} ===")
            print(f"[twin] concepts: {concept_names[:6]}")

        # 1. Section opener — exact LessonView.jsx shape.
        conv_id = f"{run_id}_s{section_index}"
        opener = f'I\'m ready to learn about "{title}". Please teach me this section.'
        t0 = time.perf_counter()
        try:
            result = send_message(
                user_id=user_id, message=opener, conversation_id=conv_id,
                context_type="lesson", context_id=folder, section_index=section_index,
            )
            pedro_reply = result.get("reply") or ""
            err = None
        except Exception as exc:
            pedro_reply, err = "", str(exc)
        elapsed = round(time.perf_counter() - t0, 1)
        pedro_clean = oma_provider.strip_pedro_tags(pedro_reply)
        transcript.append({
            "section": section_index, "turn": 0, "role": "student",
            "content": opener, "elapsed_sec": elapsed, "error": err,
        })
        transcript.append({
            "section": section_index, "turn": 0, "role": "pedro",
            "content": pedro_clean, "content_raw": pedro_reply, "elapsed_sec": elapsed,
            "section_complete_tag": oma_provider.TAG_SECTION_COMPLETE in pedro_reply,
        })
        if verbose:
            print(f"[twin] opener -> pedro {len(pedro_clean)} chars ({elapsed}s)")

        section_done = oma_provider.TAG_SECTION_COMPLETE in pedro_reply

        # 2. Turn loop.
        for turn in range(1, max_turns + 1):
            if section_done or agent.wants_to_end_section:
                break
            student_reply, update = agent.respond(
                pedro_clean, recent_turns=_recent_turns_text(agent),
            )
            if not student_reply or not student_reply.strip():
                if verbose:
                    print(f"[twin]   turn {turn}: simulator produced no reply, stopping section")
                break
            transcript.append({
                "section": section_index, "turn": turn, "role": "student",
                "content": student_reply, "state": update,
            })
            if verbose:
                ud = (update or {}).get("understanding_delta", 0)
                att = (update or {}).get("attempting_exercise")
                print(f"[twin]   turn {turn}: student -> {len(student_reply)} chars "
                      f"(ud={ud}, attempt={att})")

            t0 = time.perf_counter()
            try:
                result = send_message(
                    user_id=user_id, message=student_reply, conversation_id=conv_id,
                    context_type="lesson", context_id=folder, section_index=section_index,
                    concept_id=primary_concept_id,
                )
                pedro_reply = result.get("reply") or ""
                err = None
            except Exception as exc:
                pedro_reply, err = "", str(exc)
            elapsed = round(time.perf_counter() - t0, 1)
            pedro_clean = oma_provider.strip_pedro_tags(pedro_reply)
            transcript.append({
                "section": section_index, "turn": turn, "role": "pedro",
                "content": pedro_clean, "content_raw": pedro_reply, "elapsed_sec": elapsed,
                "error": err,
                "section_complete_tag": oma_provider.TAG_SECTION_COMPLETE in pedro_reply,
            })
            if verbose:
                print(f"[twin]   turn {turn}: pedro -> {len(pedro_clean)} chars ({elapsed}s)")
            if oma_provider.TAG_SECTION_COMPLETE in pedro_reply:
                section_done = True
                if verbose:
                    print(f"[twin]   Pedro emitted [SECTION_COMPLETE]")
                break

        # 3. Advance the section (real path triggers post-section pipeline).
        try:
            if not lesson_mod.can_advance_from_section(user_id, folder, section_index):
                if force_advance:
                    lesson_mod.mark_section_verified(user_id, folder, section_index)
                else:
                    if verbose:
                        print(f"[twin] section {section_index} not advanceable (no [SECTION_COMPLETE]); skipping advance")
            advanced = lesson_mod.advance_section(user_id, folder)
            if verbose:
                ap = "ok" if "current_section" in advanced else f"err={advanced.get('error')}"
                print(f"[twin] advance_section -> {ap}")
        except Exception as exc:
            if verbose:
                print(f"[twin] advance failed: {exc}")

        # 4. Wait for the post-section evaluator + consolidator to settle.
        if oma_provider.is_student_enabled():
            settled = _wait_for_post_section_pipeline(user_id, folder)
            if verbose and not settled:
                print(f"[twin] post-section pipeline still running after timeout; continuing")

        agent.end_section()

    # Persist run record.
    record = {
        "run_id": run_id,
        "user_id": user_id,
        "folder": folder,
        "persona": persona_id,
        "sim_model": agent.sim_model,
        "started_section": current_idx,
        "sections_run": end_idx - current_idx,
        "agent_state": agent.to_state(),
        "transcript": transcript,
    }
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        if verbose:
            print(f"\n[twin] wrote {out_path}")
    return record
