#!/usr/bin/env python3
"""Freeze the exact Pedro request for every case, for one context version.

    python3 scripts/pedro_eval/capture.py --version v1 --out ~/Desktop/Coast/pedro-context/eval/requests
PEDRO_CONTEXT=<version> selects the context builder in the app (v1 = current prompts).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from cases import CASES, ROUTING_CASES, USER, FOLDER  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--version", default="v1", help="folder name for this capture")
ap.add_argument("--context", help="PEDRO_CONTEXT to build with (default: --version)")
ap.add_argument("--out", required=True)
ap.add_argument("--only")
args = ap.parse_args()
out = Path(os.path.expanduser(args.out)) / args.version
out.mkdir(parents=True, exist_ok=True)
env = {**os.environ, "PEDRO_CONTEXT": args.context or args.version}
dump = [sys.executable, str(ROOT / "scripts" / "dump_pedro_context.py")]

for case in CASES:
    if args.only and case["id"] not in args.only.split(","):
        continue
    cmd = dump + ["--user", str(case.get("user", USER)), "--folder", case.get("folder", FOLDER),
                  "--out", str(out / f"{case['id']}.md"), "--json", str(out / f"{case['id']}.json")]
    if "until" in case:
        cmd += ["--until", str(case["until"])]
        if "section" in case:
            cmd += ["--section", str(case["section"])]
    else:
        cmd += ["--section", str(case["section"]), "--message", case["message"]]
        if case.get("history"):
            history = out / f"{case['id']}.history.json"
            history.write_text(json.dumps(case["history"]))
            cmd += ["--history", str(history)]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    print(case["id"], "·", (r.stdout.strip().splitlines() or ["(no output)"])[-1] if r.returncode == 0 else "FAILED\n" + r.stderr[-800:])
    heading = next((line for line in (out / f"{case['id']}.md").read_text().splitlines() if "# Current section:" in line), "")
    if case.get("section_title") and case["section_title"].lower() not in heading.lower():
        print(f"   WARNING: expected a section about {case['section_title']!r}, got {heading!r}")

for case in ROUTING_CASES:
    if args.only and case["id"] not in args.only.split(","):
        continue
    cmd = dump + ["--user", str(USER), "--global", "--message", case["message"], "--out", str(out / f"{case['id']}.md"), "--json", str(out / f"{case['id']}.json")]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode:
        print(case["id"], "FAILED", r.stderr[-800:])
        continue
    req = json.loads((out / f"{case['id']}.json").read_text())
    text = json.dumps(req)
    attached = [c for c in ("Network Science", "The Polya Method", "Memory Palace")
                if f"COURSE MATERIAL: {c}" in text or f"about the course: {c}." in text]
    ok = attached == [case["expect_course"]]
    print(case["id"], "·", "PASS" if ok else "FAIL", "· course material attached:", attached or "none")
