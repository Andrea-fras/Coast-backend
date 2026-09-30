#!/usr/bin/env python3
"""Replay frozen Pedro requests against model settings and grade the replies.

    python3 scripts/pedro_eval/run.py --requests ~/Desktop/Coast/pedro-context/eval/requests/v1 \
        --configs sonnet5-medium,opus55-medium,opus55-high --samples 2 --out ~/Desktop/Coast/pedro-context/eval/runs/v1
Costs real money: every sample is one full Pedro request (about 16k input tokens today).
"""
import argparse
import copy
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
os.environ["COAST_AI_USAGE"] = "off"
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")
import anthropic  # noqa: E402
import ai_usage  # noqa: E402
from cases import CASES  # noqa: E402

CONFIGS = {
    "sonnet5-medium": ("claude-sonnet-5", "medium"),
    "sonnet5-high": ("claude-sonnet-5", "high"),
    "opus55-medium": ("claude-opus-5-5", "medium"),
    "opus55-high": ("claude-opus-5-5", "high"),
}
client = anthropic.Anthropic(max_retries=3, timeout=300)


def with_check(messages, check):
    messages = copy.deepcopy(messages)
    if not check:
        return messages
    note = ("\n\n(For our quality check only: end your reply with one final line in this format, replacing each "
            f"<placeholder> or a|b|c choice with your answer: {check})")
    last = messages[-1]
    if isinstance(last["content"], str):
        last["content"] += note
    else:
        last["content"].append({"type": "text", "text": note.strip()})
    return messages


def five_minute_cache(blocks):
    """Samples run back to back, so the cheaper 5-minute cache is enough here."""
    for block in blocks:
        if isinstance(block, dict) and "cache_control" in block:
            block["cache_control"] = {"type": "ephemeral"}
    return blocks


def call(req, model, effort, check):
    started = time.time()
    messages = with_check(req["messages"], check)
    for m in messages:
        if isinstance(m["content"], list):
            five_minute_cache(m["content"])
    resp = client.messages.create(model=model, max_tokens=16000, system=five_minute_cache(copy.deepcopy(req["system"])),
                                  messages=messages,
                                  thinking={"type": "adaptive"}, output_config={"effort": effort})
    text = "".join(b.text for b in resp.content if b.type == "text")
    u = resp.usage
    usage = {"input": u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0),
             "cached": u.cache_read_input_tokens or 0, "cache_write": u.cache_creation_input_tokens or 0,
             "output": u.output_tokens}
    cost = ai_usage.cost_usd(model, usage["input"], usage["cached"], usage["cache_write"], usage["output"])
    return {"text": text, "stop": resp.stop_reason, "usage": usage, "cost": cost, "seconds": round(time.time() - started, 1)}


# ── grading ─────────────────────────────────────────────────────────────────
def parse_check(text):
    m = None
    for m in re.finditer(r"CHECK:\s*(.+)", text):
        pass
    if not m:
        return None
    out = {}
    for part in m.group(1).split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip().strip("`*").strip()
    return out


def num(v):
    try:
        v = str(v).replace(",", ".")
        if "/" in v:
            a, b = v.split("/", 1)
            return float(a) / float(b)
        return float(re.findall(r"-?\d+(?:\.\d+)?", v)[0])
    except (ValueError, IndexError, ZeroDivisionError):
        return None


def grade_expect(case, text):
    got = parse_check(text)
    if not got:
        return False, "no CHECK line"
    bad = []
    for k, want in case["expect"].items():
        v = got.get(k)
        if isinstance(want, (int, float)):
            n = num(v)
            if n is None or abs(n - want) > 0.011:
                bad.append(f"{k}={v} (want {want})")
        else:
            options = want if isinstance(want, list) else [want]
            if (v or "").lower() not in options:
                bad.append(f"{k}={v} (want {'|'.join(options)})")
    return not bad, "; ".join(bad) or "all correct"


def grade_density(case, text):
    """Compare the formula numerically, so equivalent forms such as (c_i/M)/Δ_i pass."""
    got = parse_check(text) or {}
    raw = got.get("p_i") or ""
    f = raw.replace("\\Delta", "Δ").replace("Delta", "Δ").replace("\\cdot", "*").replace("·", "*").replace("×", "*")
    f = f.replace("\\frac{", "(").replace("}{", ")/(").replace("{", "(").replace("}", ")").replace("$", "").replace("\\", "")
    f = re.sub(r"c_?i", "c", f)
    f = re.sub(r"Δ_?i", "D", f)
    f = re.sub(r"(?<=[cMD)])\s*(?=[cMD(])", "*", f.replace(" ", ""))
    try:
        value = eval(f, {"__builtins__": {}}, {"c": 3.0, "M": 7.0, "D": 2.0})
    except Exception:
        return False, f"p_i={raw!r} (unparsed)"
    return abs(value - 3.0 / 14.0) < 1e-9, f"p_i={raw}"


def grade_graph(case, text):
    got = parse_check(text) or {}
    edges = re.findall(r"(\d+)\s*[-–]\s*(\d+)", got.get("edges", ""))
    degrees = [num(d) for d in re.findall(r"\d+", got.get("degrees", ""))]
    if not edges or len(degrees) != 5:
        return False, f"unparsed: {got}"
    pairs = {tuple(sorted((int(a), int(b)))) for a, b in edges}
    problems = []
    if any(a == b for a, b in pairs):
        problems.append("self-loop")
    if len(pairs) != len(edges):
        problems.append("duplicate edge")
    if any(not 1 <= n <= 5 for p in pairs for n in p):
        problems.append("node outside 1-5")
    actual = [sum(n in p for p in pairs) for n in range(1, 6)]
    if actual != [int(d) for d in degrees]:
        problems.append(f"degrees {degrees} do not match edges (actual {actual})")
    if max(actual) > 4:
        problems.append("degree above N-1")
    return not problems, "; ".join(problems) or f"valid graph, degrees {actual}"


FALSE_9_10 = re.compile(r"(9\s*(?:and|&|,|–|-)\s*10|nodes?\s*9[^.]{0,40}10)[^.]{0,90}(not (?:directly )?(?:connected|linked|adjacent)|no direct|isn.t (?:directly )?connected|aren.t (?:directly )?connected|= ?0\b)", re.I)


def grade_opener(case, text):
    images = len(re.findall(r"!\[[^\]]*\]\([^)]*\)", text))
    wrong = bool(FALSE_9_10.search(text))
    return not wrong, f"{images} images, {len(text.split())} words" + ("; claims 9-10 not connected" if wrong else "")


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "acknowledges_own_error": {"type": "boolean"},
        "credits_student": {"type": "boolean"},
        "blames_student": {"type": "boolean"},
        "states_correct_fact": {"type": "boolean"},
        "note": {"type": "string"},
    },
    "required": ["acknowledges_own_error", "credits_student", "blames_student", "states_correct_fact", "note"],
    "additionalProperties": False,
}


def grade_tutor_error(case, text):
    tagged_wrong = "[ANSWER_WRONG" in text
    prompt = (f"You are checking one reply from an AI tutor called Pedro.\n\nVerified facts:\n{case['facts']}\n\n"
              f"The student's latest message:\n{case['student']}\n\nPedro's reply:\n{text}\n\n"
              "Answer strictly from the reply: does Pedro clearly acknowledge that his own earlier statement was wrong? "
              "Does he credit the student's reasoning where it was right? Does he treat the student as the one who made "
              "the mistake? Does the reply state the correct fact? Put a one-sentence reason in note.")
    r = client.messages.create(model="claude-opus-5-5", max_tokens=4000, messages=[{"role": "user", "content": prompt}],
                               output_config={"effort": "low", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}})
    verdict = json.loads("".join(b.text for b in r.content if b.type == "text"))
    ok = (not tagged_wrong and verdict["acknowledges_own_error"] and not verdict["blames_student"]
          and verdict["states_correct_fact"])
    detail = ("tagged ANSWER_WRONG; " if tagged_wrong else "") + json.dumps({k: v for k, v in verdict.items() if k != "note"}) + " — " + verdict["note"]
    return ok, detail


def grade_degrees(case, text):
    """Degrees read off a slide: the four numbers in any order, and how many are odd."""
    got = parse_check(text) or {}
    degrees = sorted(int(d) for d in re.findall(r"\d+", got.get("degrees", "")))
    ok = degrees == case["expect"]["degrees"] and num(got.get("odd")) == case["expect"]["odd"]
    return ok, f"degrees={degrees} odd={got.get('odd')}"


TEACHING_SCHEMA = {
    "type": "object",
    "properties": {
        "asks_question": {"type": "boolean"},
        "answerable_by_lookup": {"type": "boolean"},
        "question_kind": {"type": "string", "enum": ["none", "recall_or_read_off", "apply_to_new_case", "build_something",
                                                     "predict_or_explain", "compare_or_choose", "spot_the_flaw"]},
        "hook_before_definition": {"type": "boolean"},
        "real_world_or_running_example": {"type": "boolean"},
        "voice": {"type": "integer"},
        "note": {"type": "string"},
    },
    "required": ["asks_question", "answerable_by_lookup", "question_kind", "hook_before_definition",
                 "real_world_or_running_example", "voice", "note"],
    "additionalProperties": False,
}
STRONG_QUESTIONS = {"apply_to_new_case", "build_something", "predict_or_explain", "compare_or_choose", "spot_the_flaw"}


def grade_teaching(case, text):
    """How Pedro teaches: a question that makes the student use the idea (not look it up), and a voice."""
    prompt = ("You are reviewing one reply from Pedro, an AI tutor teaching a university lesson from lecture slides.\n\n"
              f"The student's latest message:\n{case['student']}\n\nPedro's reply:\n{text}\n\n"
              "Judge strictly:\n"
              "- asks_question: does the reply end by asking the student something to answer?\n"
              "- answerable_by_lookup: could the student answer by copying from this reply, or by reading a value or "
              "label straight off the slide it shows, without reasoning? (false if no question)\n"
              "- question_kind: recall_or_read_off, apply_to_new_case (the idea applied to an example not in the reply "
              "or slide), build_something (write a small representation), predict_or_explain, compare_or_choose, "
              "spot_the_flaw, or none.\n"
              "- hook_before_definition: when the reply introduces a new idea, does it first give the problem it solves, "
              "a scenario or a surprising consequence before the formal definition? (true if no new idea)\n"
              "- real_world_or_running_example: does it tie the idea to a real system/use or carry a concrete running "
              "example beyond restating the slide?\n"
              "- voice: 1 = reads like narrated slides, 3 = clear and friendly, 5 = memorable tutor with personality.\n"
              "- note: one sentence.")
    r = client.messages.create(model="claude-opus-5-5", max_tokens=4000, messages=[{"role": "user", "content": prompt}],
                               output_config={"effort": "low", "format": {"type": "json_schema", "schema": TEACHING_SCHEMA}})
    v = json.loads("".join(b.text for b in r.content if b.type == "text"))
    good_question = not v["asks_question"] or (not v["answerable_by_lookup"] and v["question_kind"] in STRONG_QUESTIONS)
    ok = good_question and v["voice"] >= 3 and v["hook_before_definition"]
    return ok, json.dumps({k: v[k] for k in v if k != "note"}) + " — " + v["note"]


GRADERS = {"density": grade_density, "graph": grade_graph, "opener": grade_opener, "tutor_error": grade_tutor_error,
           "degrees": grade_degrees, "teaching": grade_teaching}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True)
    ap.add_argument("--configs", default="sonnet5-medium,opus55-medium,opus55-high")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--only")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    reqdir = Path(os.path.expanduser(args.requests))
    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for case in CASES:
        if args.only and case["id"] not in args.only.split(","):
            continue
        req = json.loads((reqdir / f"{case['id']}.json").read_text())
        last = req["messages"][-1]["content"]
        # v2 puts Coast's note before the student's words in the same turn; the judge needs only their words.
        case = {**case, "student": last if isinstance(last, str) else last[-1].get("text", "")}
        for cfg in args.configs.split(","):
            for s in range(args.samples):
                jobs.append((case, req, cfg, s))

    def work(job):
        case, req, cfg, s = job
        model, effort = CONFIGS[cfg]
        try:
            res = call(req, model, effort, case.get("check"))
            grader = GRADERS.get(case.get("grade"), grade_expect)
            ok, detail = grader(case, res["text"])
        except Exception as exc:  # keep the run going; failures are reported
            res, ok, detail = {"text": "", "cost": 0, "seconds": 0, "usage": {}}, False, f"error: {type(exc).__name__}: {exc}"
        return {"case": case["id"], "config": cfg, "sample": s, "pass": ok, "detail": detail, **res}

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(work, jobs))
    (out / "results.json").write_text(json.dumps(results, indent=1))

    configs = args.configs.split(",")
    lines = ["| case | " + " | ".join(configs) + " |", "|---|" + "---|" * len(configs)]
    for case in [c["id"] for c in CASES if not args.only or c["id"] in args.only.split(",")]:
        row = [case]
        for cfg in configs:
            rs = [r for r in results if r["case"] == case and r["config"] == cfg]
            row.append(f"{sum(r['pass'] for r in rs)}/{len(rs)}")
        lines.append("| " + " | ".join(row) + " |")
    tot = ["**total**"] + [f"**{sum(r['pass'] for r in results if r['config'] == c)}/{sum(r['config'] == c for r in results)}**" for c in configs]
    lines.append("| " + " | ".join(tot) + " |")
    cost = sum(r.get("cost") or 0 for r in results)
    lat = {c: round(sum(r["seconds"] for r in results if r["config"] == c) / max(1, sum(r["config"] == c for r in results)), 1) for c in configs}
    lines += ["", f"Cost of this run: ${cost:.2f} · mean seconds per reply: {lat}", "", "## Details", ""]
    for r in results:
        lines.append(f"- **{r['case']}** · {r['config']} #{r['sample']} · {'PASS' if r['pass'] else 'FAIL'} · {r['detail']}")
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines[: len(CASES) + 5]))


if __name__ == "__main__":
    main()
