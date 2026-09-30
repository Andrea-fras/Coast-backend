#!/usr/bin/env python3
"""Run the Student OMA / Pedro memory suite locally (no pytest, no network).

  python3 tests/student_memory/run.py              # everything
  python3 tests/student_memory/run.py recall       # files/tests matching "recall"
  python3 tests/student_memory/run.py -v           # show failure tracebacks

Each test file is also pytest-compatible if you install pytest later.
"""
from __future__ import annotations

import importlib
import io
import logging
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness  # noqa: E402,F401  (must load before Coast modules)

logging.disable(logging.CRITICAL)
MODULES = ["test_recording", "test_profile_accuracy", "test_recall", "test_gating", "test_adaptation", "test_providers", "test_placement", "test_workshops", "test_evidence_integrity"]
# Known gaps we have consciously scheduled for later. Reported, never hidden.
DEFERRED: dict[str, str] = {}


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    verbose = "-v" in sys.argv
    results: list[tuple[str, str, str]] = []
    for mod_name in MODULES:
        mod = importlib.import_module(mod_name)
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            full = f"{mod_name}::{name}"
            if args and not any(a in full for a in args):
                continue
            fn = getattr(mod, name)
            if full in DEFERRED:
                results.append(("DEFER", full, DEFERRED[full]))
                continue
            sink = io.StringIO()
            try:
                with redirect_stdout(sink), redirect_stderr(sink):
                    fn()
                results.append(("PASS", full, ""))
            except AssertionError as e:
                results.append(("FAIL", full, str(e) or traceback.format_exc(limit=2)))
            except Exception:
                results.append(("ERROR", full, traceback.format_exc(limit=6)))

    width = max((len(r[1]) for r in results), default=10)
    for status, full, detail in results:
        first = detail.strip().splitlines()[-1] if detail.strip() else ""
        print(f"{status:5}  {full:<{width}}  {first[:140] if status != 'PASS' else ''}")
        if verbose and status not in ("PASS", "DEFER"):
            print("       " + detail.strip().replace("\n", "\n       "))
    counts = {s: sum(1 for r in results if r[0] == s) for s in ("PASS", "FAIL", "ERROR", "DEFER")}
    print(f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['ERROR']} errors, "
          f"{counts['DEFER']} deferred  (data in {harness.TMP})")
    return 0 if not counts["FAIL"] and not counts["ERROR"] else 1


if __name__ == "__main__":
    sys.exit(main())
