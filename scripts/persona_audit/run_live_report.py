#!/usr/bin/env python3
"""Run live audit and write detailed JSON report."""

from __future__ import annotations

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
from tutor import send_message
from scripts.persona_audit import PERSONA_IDS, load_course, load_persona, load_scenarios
from scripts.persona_audit.run_audit import (
    _contains_any,
    _contains_none,
    _llm_relevance_judge,
    _profile_body,
)


def main() -> int:
    os.environ.setdefault("RAG_PROVIDER", "oma")
    os.environ.setdefault("STUDENT_OMA_ENABLED", "true")

    course = load_course()
    folder = course["folder"]
    scenarios = load_scenarios()["scenarios"]
    rows = []

    for pid in PERSONA_IDS:
        persona = load_persona(pid)
        uid = persona["user_id"]
        for sc in scenarios:
            concept_ids = sc.get("concept_ids")
            block = oma_provider.get_student_profile_block(
                uid, folder, current_concept_ids=concept_ids, max_chars=4000,
            ) or "(empty)"
            body = _profile_body(block).strip()
            question = sc["message"]
            live = (sc.get("live") or {}).get(pid) or {}
            checks = (sc.get("checks") or {}).get(pid) or {}

            row = {
                "persona": pid,
                "scenario_id": sc["id"],
                "description": sc.get("description", ""),
                "question": question,
                "profile_used": body,
                "reply": None,
                "block_pass": True,
                "live_pass": None,
                "keyword_score": None,
                "judge_score": None,
                "overall_score": None,
                "elapsed_sec": None,
                "notes": [],
            }

            inc_any = checks.get("profile_must_include_any") or []
            for needle in checks.get("profile_must_exclude") or []:
                if needle.lower() in body.lower():
                    row["block_pass"] = False
                    row["notes"].append(f"profile forbidden: {needle}")

            if inc_any and not _contains_any(body, inc_any):
                row["block_pass"] = False
                row["notes"].append(f"profile missing: {inc_any}")

            if not live:
                row["live_pass"] = True
                row["overall_score"] = 1.0 if row["block_pass"] else 0.0
                rows.append(row)
                print(f"{pid}/{sc['id']}: skip live")
                continue

            conv = f"audit_{pid}_{sc['id']}_{uuid.uuid4().hex[:6]}"
            t0 = time.perf_counter()
            try:
                result = send_message(
                    user_id=uid,
                    message=question,
                    conversation_id=conv,
                    context_type=sc.get("context_type", "folder"),
                    context_id=folder,
                    section_index=sc.get("section_index"),
                    concept_id=(concept_ids or [None])[0],
                )
                reply = result.get("reply") or ""
            except Exception as exc:
                reply = ""
                row["notes"].append(str(exc))
            row["elapsed_sec"] = round(time.perf_counter() - t0, 1)
            row["reply"] = reply

            must_inc = live.get("must_include_any") or []
            must_exc = live.get("must_exclude") or []
            topics = live.get("relevance_topics") or []
            score_parts: list[float] = []
            live_ok = True

            if must_inc:
                hit = _contains_any(reply, must_inc)
                score_parts.append(1.0 if hit else 0.0)
                if not hit:
                    live_ok = False
                    row["notes"].append(f"reply missing: {must_inc}")
            if must_exc and not _contains_none(reply, must_exc):
                score_parts.append(0.0)
                live_ok = False
                row["notes"].append(f"reply forbidden hit: {must_exc}")
            elif must_exc:
                score_parts.append(1.0)

            row["keyword_score"] = round(sum(score_parts) / len(score_parts), 2) if score_parts else None
            row["live_pass"] = live_ok

            if topics and reply and (os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")):
                js = _llm_relevance_judge(question, reply, topics)
                row["judge_score"] = js
                if js < 3:
                    row["live_pass"] = False
                    row["notes"].append(f"judge {js}/5")

            parts = [p for p in [
                row["keyword_score"],
                row["judge_score"] / 5.0 if row["judge_score"] else None,
            ] if p is not None]
            overall = sum(parts) / len(parts) if parts else (1.0 if live_ok else 0.0)
            if not row["block_pass"]:
                overall *= 0.5
            row["overall_score"] = round(overall, 2)
            rows.append(row)
            print(f"{pid}/{sc['id']}: score={row['overall_score']} judge={row['judge_score']}")

    out = ROOT / "scripts/persona_audit/last_live_report.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
