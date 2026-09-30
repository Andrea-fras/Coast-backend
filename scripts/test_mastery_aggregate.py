#!/usr/bin/env python3
"""Unit tests for virtual mastery aggregation (alias ledger read path)."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coast_content_oma.student.mastery_aggregate import (
    aggregate_mastery_rows,
    eval_severity,
)


def _row(**kwargs) -> dict:
    base = {
        "concept_id": "con_a",
        "concept_name": "Alpha",
        "mastery_score": 0.5,
        "successes": 2,
        "struggles": 1,
        "neutral_touches": 0,
        "first_seen": "2026-01-01T00:00:00",
        "last_seen": "2026-02-01T00:00:00",
    }
    base.update(kwargs)
    return base


def test_pessimistic_misconception_wins():
    rows = [
        _row(concept_id="con_47", last_eval_state="mastered", last_misconception=False),
        _row(concept_id="con_12", last_eval_state="misconception", last_misconception=True),
    ]
    agg = aggregate_mastery_rows(rows, canonical_id="con_12", canonical_name="Canonical")
    assert agg["last_eval_state"] == "misconception"
    assert agg["last_misconception"] is True
    assert eval_severity({"last_eval_state": "misconception"}) > eval_severity(
        {"last_eval_state": "mastered"}
    )


def test_weighted_score_by_evidence():
    rows = [
        _row(mastery_score=0.9, successes=8, struggles=0),
        _row(mastery_score=0.2, successes=1, struggles=4),
    ]
    agg = aggregate_mastery_rows(rows, canonical_id="con_x")
    assert 0.35 < agg["mastery_score"] < 0.75
    assert agg["successes"] == 9
    assert agg["struggles"] == 4


def test_empty_returns_none():
    assert aggregate_mastery_rows([], canonical_id="con_x") is None


def test_decay_applied_per_row():
    old = (datetime.now() - timedelta(days=90)).isoformat(timespec="seconds")
    rows = [
        _row(mastery_score=0.85, successes=5, last_seen=old, confidence=0.5),
        _row(mastery_score=0.85, successes=5, last_seen=datetime.now().isoformat(timespec="seconds")),
    ]
    agg = aggregate_mastery_rows(rows, canonical_id="con_x", now=datetime.now())
    assert agg["effective_score"] < agg["mastery_score"]


def main() -> int:
    tests = [
        test_pessimistic_misconception_wins,
        test_weighted_score_by_evidence,
        test_empty_returns_none,
        test_decay_applied_per_row,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  OK  {t.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
