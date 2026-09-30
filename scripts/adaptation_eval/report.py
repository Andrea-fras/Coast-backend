#!/usr/bin/env python3
"""Turn eval_runs/<name>/state.json into a readable scorecard (report.md).

  python3 scripts/adaptation_eval/report.py --name ten_lessons
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scenario import BEATS, LESSONS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _score(j, key):
    v = (j or {}).get(key) or {}
    return "–" if v.get("score") is None else str(v.get("score"))


def _clip(text, n=220):
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def build(state: dict) -> str:
    out = ["# Pedro adaptation eval — 10 lessons", ""]
    lessons = [state["lessons"][k] for k in sorted(state["lessons"], key=int)]
    done = [L for L in lessons if L.get("done")]
    gates = sum(L["sections"] for L in done)
    passed = sum(L["gates_passed"] for L in done)
    judged = [L["start_judgement"] for L in lessons if L.get("start_judgement")]
    fabs = [L for L in lessons if (L.get("start_judgement") or {}).get("fabrication", {}).get("found")]

    def avg(key):
        vals = [j.get(key, {}).get("score") for j in judged if isinstance(j.get(key, {}).get("score"), (int, float))]
        return f"{sum(vals) / len(vals):.2f} / 2 ({len(vals)} scored)" if vals else "n/a"

    probes = state.get("probes", [])
    recall = [p["judgement"].get("recall", {}).get("score") for p in probes
              if isinstance(p["judgement"].get("recall", {}).get("score"), (int, float))]
    out += ["## Summary", "",
            f"- Lessons completed: **{len(done)} / {len(LESSONS)}**; comprehension gates: **{gates}**, "
            f"passed by the student **{passed}**, forced after the turn cap **{gates - passed}**",
            f"- Continuity at the start of each lesson: **{avg('continuity')}**",
            f"- Learning preference applied: **{avg('preference')}**",
            f"- Relevant memory used (mistakes, analogies, goals): **{avg('relevant_memory')}**",
            f"- Lesson starts with invented history: **{len(fabs)}**",
            f"- Recall questions: **{sum(recall) / len(recall):.2f} / 2** over {len(recall)} questions" if recall
            else "- Recall questions: none yet",
            f"- Recall answers with invented history: **{sum(1 for p in probes if p['judgement'].get('fabrications'))}**",
            ""]

    out += ["## Per lesson", "",
            "| # | Lesson | Gates passed | Continuity | Preference | Memory used | Invented? | Judge's verdict |",
            "|---|---|---|---|---|---|---|---|"]
    for L in lessons:
        j = L.get("start_judgement") or {}
        gp = f"{L.get('gates_passed', '…')}/{L.get('sections', len(L.get('sections_log', [])))}"
        out.append(f"| {L['n']} | {L['name']} | {gp} | {_score(j, 'continuity')} | {_score(j, 'preference')} | "
                   f"{_score(j, 'relevant_memory')} | {'**yes**' if j.get('fabrication', {}).get('found') else 'no'} | "
                   f"{_clip(j.get('verdict'), 160)} |")
    out.append("")

    out += ["## Planted memories — stored and used?", "",
            "| Moment | Planted | Student said | In memory after each lesson |", "|---|---|---|---|"]
    for b in BEATS:
        got = state["beats"].get(b["id"])
        if not got:
            out.append(f"| {b['id']} | not performed | – | – |")
            continue
        seen = [f"L{k}:{'✓' if v['planted_in_memory'].get(b['id']) else '✗'}"
                for k, v in sorted(state.get("snapshots", {}).items(), key=lambda kv: int(kv[0]))
                if b["id"] in v.get("planted_in_memory", {})]
        out.append(f"| {b['id']} | lesson {got['lesson']}, section {got['section'] + 1} | "
                   f"\"{_clip(got['quote'], 140)}\" | {' '.join(seen)} |")
    out.append("")

    out += ["## Recall questions", ""]
    for p in probes:
        j = p["judgement"]
        out += [f"**After lesson {p['after']}: \"{p['question']}\"** — recall {j.get('recall', {}).get('score')}/2",
                "", f"> {_clip(p['answer'], 900)}", "",
                f"Judge: {_clip(j.get('verdict'), 300)}"
                + (f" Invented: {j.get('fabrications')}" if j.get("fabrications") else ""), ""]

    out += ["## Evidence at the start of each lesson", ""]
    for L in lessons:
        j = L.get("start_judgement")
        if not j:
            continue
        opener = next((t["text"] for t in L["sections_log"][0]["turns"] if t["role"] == "pedro"), "")
        given = (L.get("pedro_was_given") or {})
        out += [f"### Lesson {L['n']}: {L['name']}", "",
                f"- Continuity: {_clip(j.get('continuity', {}).get('evidence'), 300)}",
                f"- Preference: {_clip(j.get('preference', {}).get('evidence'), 300)}",
                f"- Memory: {_clip(j.get('relevant_memory', {}).get('evidence'), 300)}",
                "", "Pedro's opening:", "", f"> {_clip(opener, 700)}", "",
                "<details><summary>What Student OMA gave Pedro</summary>", "", "```",
                (given.get("profile") or "(no profile block)")[:2500],
                *[blk[:1500] for blk in given.get("section_blocks") or []], "```", "</details>", ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    run_dir = ROOT / "eval_runs" / args.name
    state = json.loads((run_dir / "state.json").read_text())
    report = build(state)
    (run_dir / "report.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
