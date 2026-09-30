#!/usr/bin/env python3
"""Run Pedro personalization audit against sparse / medium / rich personas.

Usage:
  # 1. Seed personas first
  RAG_PROVIDER=oma STUDENT_OMA_ENABLED=true python3 scripts/persona_audit/seed.py

  # 2. Block-mode (fast, no LLM) — profile injection checks
  python3 scripts/persona_audit/run_audit.py --block-only

  # 3. Live Pedro responses (requires GEMINI_API_KEY or OPENAI_API_KEY)
  python3 scripts/persona_audit/run_audit.py --live

  # Single persona / scenario
  python3 scripts/persona_audit/run_audit.py --live --persona rich --scenario queueing_misconception_block
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv()

import oma_provider  # noqa: E402
from scripts.persona_audit import PERSONA_IDS, load_course, load_persona, load_scenarios  # noqa: E402


@dataclass
class CheckResult:
    scenario_id: str
    persona_id: str
    mode: str
    passed: bool
    detail: str
    score: float = 0.0


@dataclass
class AuditReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, **kwargs) -> None:
        self.results.append(CheckResult(**kwargs))

    def summary(self) -> str:
        lines = ["", "=" * 72, "PERSONA AUDIT SUMMARY", "=" * 72]
        by_persona: dict[str, list[CheckResult]] = {}
        for r in self.results:
            by_persona.setdefault(r.persona_id, []).append(r)

        for pid in PERSONA_IDS:
            rows = by_persona.get(pid, [])
            if not rows:
                continue
            passed = sum(1 for r in rows if r.passed)
            lines.append(f"\n## {pid.upper()} — {passed}/{len(rows)} passed")
            for r in rows:
                mark = "PASS" if r.passed else "FAIL"
                sc = f" ({r.score:.0%})" if r.mode == "live" else ""
                lines.append(f"  [{mark}] {r.mode:5} {r.scenario_id}: {r.detail}{sc}")

        total = len(self.results)
        ok = sum(1 for r in self.results if r.passed)
        lines.append(f"\nTOTAL: {ok}/{total} passed ({ok/total:.0%})" if total else "\nNo results.")
        return "\n".join(lines)


def _contains_any(haystack: str, needles: list[str]) -> bool:
    h = haystack.lower()
    return any(n.lower() in h for n in needles if n)


def _contains_none(haystack: str, needles: list[str]) -> bool:
    h = haystack.lower()
    return not any(n.lower() in h for n in needles if n)


def _profile_body(block: str) -> str:
    """Profile facts only — exclude Pedro meta RULES footer."""
    if "PROFILE RULES" in block:
        block = block.split("PROFILE RULES", 1)[0]
    if "--- END STUDENT PROFILE ---" in block:
        block = block.split("--- END STUDENT PROFILE ---", 1)[0]
    return block


def run_block_checks(
    persona_id: str,
    user_id: int,
    folder: str,
    scenario: dict,
    report: AuditReport,
) -> str:
    checks = (scenario.get("checks") or {}).get(persona_id) or {}
    concept_ids = scenario.get("concept_ids") or None

    block = oma_provider.get_student_profile_block(
        user_id, folder, current_concept_ids=concept_ids, max_chars=4000,
    ) or ""
    body = _profile_body(block)

    inc_any = checks.get("profile_must_include_any") or []
    inc_all = checks.get("profile_must_include") or []
    exc = checks.get("profile_must_exclude") or []

    ok = True
    details: list[str] = []

    if inc_any and not _contains_any(body, inc_any):
        ok = False
        details.append(f"missing any of {inc_any!r}")
    for needle in inc_all:
        if needle.lower() not in body.lower():
            ok = False
            details.append(f"missing {needle!r}")
    for needle in exc:
        if needle.lower() in body.lower():
            ok = False
            details.append(f"should not contain {needle!r}")

    if ok:
        detail = "profile block OK"
    else:
        detail = "; ".join(details) or "check failed"
        detail += f" [block len={len(block)}]"

    report.add(
        scenario_id=scenario["id"],
        persona_id=persona_id,
        mode="block",
        passed=ok,
        detail=detail,
    )
    return block


def run_live_checks(
    persona_id: str,
    user_id: int,
    folder: str,
    scenario: dict,
    report: AuditReport,
    *,
    use_judge: bool = False,
) -> None:
    live = (scenario.get("live") or {}).get(persona_id)
    if not live:
        report.add(
            scenario_id=scenario["id"],
            persona_id=persona_id,
            mode="live",
            passed=True,
            detail="no live expectations",
        )
        return

    from tutor import send_message

    conv_id = f"audit_{persona_id}_{scenario['id']}_{uuid.uuid4().hex[:8]}"
    t0 = time.perf_counter()
    try:
        result = send_message(
            user_id=user_id,
            message=scenario["message"],
            conversation_id=conv_id,
            context_type=scenario.get("context_type", "folder"),
            context_id=folder,
            section_index=scenario.get("section_index"),
            concept_id=(scenario.get("concept_ids") or [None])[0],
        )
        reply = result.get("reply") or ""
    except Exception as exc:
        report.add(
            scenario_id=scenario["id"],
            persona_id=persona_id,
            mode="live",
            passed=False,
            detail=f"send_message failed: {exc}",
        )
        return
    elapsed = time.perf_counter() - t0

    must_inc = live.get("must_include_any") or []
    must_exc = live.get("must_exclude") or []
    topics = live.get("relevance_topics") or []

    ok = True
    details: list[str] = []
    score_parts: list[float] = []

    if must_inc:
        hit = _contains_any(reply, must_inc)
        score_parts.append(1.0 if hit else 0.0)
        if not hit:
            ok = False
            details.append(f"response missing any of {must_inc!r}")

    if must_exc and not _contains_none(reply, must_exc):
        ok = False
        score_parts.append(0.0)
        details.append(f"response contains forbidden {must_exc!r}")
    elif must_exc:
        score_parts.append(1.0)

    judge_score = None
    if use_judge and topics and reply:
        judge_score = _llm_relevance_judge(scenario["message"], reply, topics)
        score_parts.append(judge_score / 5.0)
        if judge_score < 3:
            ok = False
            details.append(f"judge relevance {judge_score}/5 for {topics}")

    score = sum(score_parts) / len(score_parts) if score_parts else (1.0 if ok else 0.0)
    detail = "; ".join(details) if details else f"response OK ({len(reply)} chars, {elapsed:.1f}s)"
    if judge_score is not None:
        detail += f"; judge={judge_score}/5"

    report.add(
        scenario_id=scenario["id"],
        persona_id=persona_id,
        mode="live",
        passed=ok,
        detail=detail,
        score=score,
    )


def _llm_relevance_judge(question: str, reply: str, topics: list[str]) -> int:
    """1-5 score: how relevant is Pedro's reply to expected personalization topics."""
    prompt = (
        "Rate tutor personalization from 1 (generic) to 5 (clearly tailored to this student).\n"
        f"Student question: {question[:500]}\n"
        f"Look for: {', '.join(topics)}\n"
        f"Tutor reply: {reply[:2000]}\n"
        "Reply with ONLY one digit 1-5. No other text."
    )
    eval_model = os.getenv("GEMINI_EVAL_MODEL", "gemini-2.5-flash")
    text = ""
    last_err: Exception | None = None

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model=eval_model,
                contents=prompt,
                config={"max_output_tokens": 128, "temperature": 0.0},
            )
            if response.candidates and response.candidates[0].content:
                for part in (response.candidates[0].content.parts or []):
                    if hasattr(part, "text") and part.text:
                        text += part.text
        except Exception as exc:
            last_err = exc

    if not text.strip():
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_key)
                resp = client.chat.completions.create(
                    model=os.getenv("OPENAI_EVAL_MODEL", "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=16,
                    temperature=0.0,
                )
                text = resp.choices[0].message.content or ""
            except Exception as exc:
                last_err = exc

    m = re.search(r"\b([1-5])\b", text.strip())
    if m:
        return int(m.group(1))
    if last_err:
        print(f"[audit judge] LLM failed ({last_err}); defaulting to 3. raw={text!r}")
    elif text.strip():
        print(f"[audit judge] could not parse score; raw={text!r}")
    return 3


def main() -> int:
    ap = argparse.ArgumentParser(description="Pedro persona personalization audit")
    ap.add_argument("--block-only", action="store_true", help="Profile block checks only (no LLM)")
    ap.add_argument("--live", action="store_true", help="Also call Pedro (requires API key)")
    ap.add_argument("--judge", action="store_true", help="LLM relevance judge on live replies")
    ap.add_argument("--persona", choices=PERSONA_IDS, help="Run one persona")
    ap.add_argument("--scenario", help="Run one scenario id")
    ap.add_argument("--json-out", help="Write JSON report to path")
    args = ap.parse_args()

    if not args.block_only and not args.live:
        args.block_only = True

    os.environ.setdefault("RAG_PROVIDER", "oma")
    os.environ.setdefault("STUDENT_OMA_ENABLED", "true")

    if not oma_provider.is_student_enabled():
        print("FAIL: STUDENT_OMA_ENABLED / RAG_PROVIDER not set for Student OMA")
        return 1

    course = load_course()
    scenarios_data = load_scenarios()
    folder = course["folder"]
    scenarios = scenarios_data.get("scenarios") or []
    if args.scenario:
        scenarios = [s for s in scenarios if s["id"] == args.scenario]
        if not scenarios:
            print(f"Unknown scenario {args.scenario!r}")
            return 1

    personas = [args.persona] if args.persona else list(PERSONA_IDS)
    report = AuditReport()

    print("Pedro Persona Audit")
    print(f"Folder: {folder} | personas: {', '.join(personas)} | scenarios: {len(scenarios)}")

    for pid in personas:
        persona = load_persona(pid)
        uid = persona["user_id"]
        print(f"\n--- Persona: {pid} (user {uid}) ---")

        block = oma_provider.get_student_profile_block(uid, folder, max_chars=500)
        if not block and pid != "sparse":
            print(f"  WARN: empty profile for {pid} — run seed.py first")

        for sc in scenarios:
            print(f"  scenario: {sc['id']}")
            if args.block_only or args.live:
                run_block_checks(pid, uid, folder, sc, report)
            if args.live:
                run_live_checks(pid, uid, folder, sc, report, use_judge=args.judge)

    summary = report.summary()
    print(summary)

    if args.json_out:
        out = {
            "results": [r.__dict__ for r in report.results],
            "passed": sum(1 for r in report.results if r.passed),
            "total": len(report.results),
        }
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")

    failed = sum(1 for r in report.results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
