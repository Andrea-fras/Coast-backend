"""Score a digital-twin run record.

Metrics focus on the ARC, not single turns:
  - Mastery trajectory across sections (did learning happen? did early mastery
    hold up later?).
  - Frustration / engagement trajectories (does Pedro keep the student
    comfortable and engaged as difficulty rises?).
  - Give-ups per section.
  - Exercise attempt rate + simulator-claimed correctness.
  - Pedro-vs-student talk ratio (a tutor that lectures too much is a smell).
  - Section completion via [SECTION_COMPLETE] vs forced advance (did Pedro
    judge the student ready, or did we force it?).
  - Personalization markers in Pedro's replies (cross-section recall phrases).
  - Capture pipeline actually fired: count [REMEMBER] traits and [CLICKED]
    golden moments written to the Student OMA stores for this user/course.

No single number is "the score" — read the trajectories. Relative comparisons
(across personas, or with/without the profile) are more reliable than absolutes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

_RECALL_PHRASES = [
    r"\bback in section\b",
    r"\bin section \d+\b",
    r"\bearlier (we|you|in)\b",
    r"\blast time\b",
    r"\byou mentioned\b",
    r"\bas you said\b",
    r"\bwhen we covered\b",
    r"\bremember (when|how|the)\b",
    r"\byou (struggled with|got|mastered)\b",
]
_RECALL_RE = re.compile("|".join(_RECALL_PHRASES), re.IGNORECASE)


def _load_record(record_or_path) -> dict:
    if isinstance(record_or_path, dict):
        return record_or_path
    return json.loads(Path(record_or_path).read_text(encoding="utf-8"))


def _capture_counts(user_id, folder: str) -> dict:
    """Count capture-tag-derived writes in the real Student OMA stores."""
    out = {"remember_traits": 0, "clicked_golden_moments": 0, "available": False}
    try:
        import oma_provider
        if not oma_provider.is_student_enabled():
            return out
        from coast_content_oma.student.stores import course_namespace, identity_namespace
        orch = oma_provider._student_orchestrator()

        id_ns = identity_namespace(user_id)
        n_rem = 0
        for it in orch.identity.all_traits(id_ns, min_confidence=0.0):
            if (it.store_specific or {}).get("derivation") == "pedro_remember_tag":
                n_rem += 1
        out["remember_traits"] = n_rem

        course_ns = course_namespace(user_id, folder)
        n_click = 0
        for it in orch.patterns.all(course_ns):
            ss = it.store_specific or {}
            if ss.get("pattern_type") == "golden_moment" and ss.get("derivation") == "pedro_clicked_tag":
                n_click += 1
        out["clicked_golden_moments"] = n_click
        out["available"] = True
    except Exception as exc:
        out["error"] = str(exc)
    return out


def score_run(record_or_path) -> dict:
    rec = _load_record(record_or_path)
    agent = rec.get("agent_state") or {}
    section_log = agent.get("section_log") or []
    turn_log = agent.get("turn_log") or []
    transcript = rec.get("transcript") or []
    persona_id = rec.get("persona")
    user_id = rec.get("user_id")
    folder = rec.get("folder")

    # Per-section trajectories.
    per_section = []
    for s in section_log:
        mastery_end = s.get("mastery_end") or {}
        avg_mastery = (sum(mastery_end.values()) / len(mastery_end)) if mastery_end else None
        per_section.append({
            "section_index": s.get("section_index"),
            "title": s.get("title"),
            "turns": s.get("turns"),
            "frustration_end": s.get("frustration_end"),
            "engagement_end": s.get("engagement_end"),
            "avg_mastery_end": round(avg_mastery, 3) if avg_mastery is not None else None,
            "gave_up": s.get("gave_up"),
            "n_concepts": len(s.get("concepts") or []),
        })

    mastery_traj = [ps["avg_mastery_end"] for ps in per_section if ps["avg_mastery_end"] is not None]
    frustration_traj = [ps["frustration_end"] for ps in per_section]
    engagement_traj = [ps["engagement_end"] for ps in per_section]

    # Exercise attempts + claimed correctness (simulator self-report).
    attempts = [t for t in turn_log if t.get("attempting_exercise")]
    claimed_correct = sum(1 for t in attempts if t.get("likely_correct"))

    # Talk ratio.
    pedro_chars = [m["content"] for m in transcript if m.get("role") == "pedro" and m.get("content")]
    student_chars = [m["content"] for m in transcript if m.get("role") == "student" and m.get("content")]
    pedro_avg = (sum(len(c) for c in pedro_chars) / len(pedro_chars)) if pedro_chars else 0
    student_avg = (sum(len(c) for c in student_chars) / len(student_chars)) if student_chars else 0
    talk_ratio = round(pedro_avg / student_avg, 2) if student_avg else None

    # Section completion: Pedro-emitted [SECTION_COMPLETE] vs forced.
    pedro_msgs = [m for m in transcript if m.get("role") == "pedro"]
    pedro_emitted_complete = sum(1 for m in pedro_msgs if m.get("section_complete_tag"))
    sections_run = rec.get("sections_run") or 0

    # Personalization recall markers in Pedro replies.
    pedro_text = "\n".join(pedro_chars)
    recall_hits = len(_RECALL_RE.findall(pedro_text))

    # Capture pipeline writes.
    captures = _capture_counts(user_id, folder) if (user_id and folder) else {"available": False}

    metrics = {
        "run_id": rec.get("run_id"),
        "persona": persona_id,
        "user_id": user_id,
        "folder": folder,
        "sections_run": sections_run,
        "total_turns": len(turn_log),
        "mastery_trajectory": mastery_traj,
        "frustration_trajectory": frustration_traj,
        "engagement_trajectory": engagement_traj,
        "give_ups": sum(1 for ps in per_section if ps.get("gave_up")),
        "exercise_attempts": len(attempts),
        "claimed_correct_rate": round(claimed_correct / len(attempts), 3) if attempts else None,
        "pedro_avg_chars": round(pedro_avg, 1),
        "student_avg_chars": round(student_avg, 1),
        "talk_ratio_pedro_per_student": talk_ratio,
        "pedro_emitted_section_complete": pedro_emitted_complete,
        "forced_advances": max(0, sections_run - pedro_emitted_complete),
        "recall_markers_in_pedro_replies": recall_hits,
        "capture_writes": captures,
        "per_section": per_section,
    }
    return metrics


def format_metrics(m: dict) -> str:
    lines = []
    lines.append(f"Run {m.get('run_id')}  persona={m.get('persona')}  user={m.get('user_id')}  folder={m.get('folder')!r}")
    lines.append(f"sections_run={m.get('sections_run')}  total_turns={m.get('total_turns')}")
    lines.append("")
    lines.append("Mastery trajectory (avg mastery_end per section):")
    lines.append("  " + (" -> ".join(f"{v:.2f}" for v in (m.get("mastery_trajectory") or [])) or "(none)"))
    lines.append("Frustration trajectory (per section, lower is better):")
    lines.append("  " + (" -> ".join(f"{v:.2f}" for v in (m.get("frustration_trajectory") or [])) or "(none)"))
    lines.append("Engagement trajectory (per section, higher is better):")
    lines.append("  " + (" -> ".join(f"{v:.2f}" for v in (m.get("engagement_trajectory") or [])) or "(none)"))
    lines.append("")
    lines.append(f"give_ups={m.get('give_ups')}  exercise_attempts={m.get('exercise_attempts')}  "
                 f"claimed_correct_rate={m.get('claimed_correct_rate')}")
    lines.append(f"talk_ratio (pedro/student chars)={m.get('talk_ratio_pedro_per_student')}  "
                 f"(high = Pedro lectures a lot)")
    lines.append(f"pedro_emitted_section_complete={m.get('pedro_emitted_section_complete')}  "
                 f"forced_advances={m.get('forced_advances')}")
    lines.append(f"recall_markers_in_pedro_replies={m.get('recall_markers_in_pedro_replies')}  "
                 f"(cross-section / past-reference phrases)")
    cap = m.get("capture_writes") or {}
    lines.append(f"capture_writes: remember_traits={cap.get('remember_traits')}  "
                 f"clicked_golden_moments={cap.get('clicked_golden_moments')}  "
                 f"available={cap.get('available')}")
    lines.append("")
    lines.append("Per section:")
    for ps in m.get("per_section") or []:
        lines.append(f"  s{ps.get('section_index')} {ps.get('title')!r}: turns={ps.get('turns')} "
                     f"frust={ps.get('frustration_end')} engage={ps.get('engagement_end')} "
                     f"mastery={ps.get('avg_mastery_end')} gave_up={ps.get('gave_up')}")
    return "\n".join(lines)
