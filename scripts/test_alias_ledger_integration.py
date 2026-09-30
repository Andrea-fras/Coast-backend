#!/usr/bin/env python3
"""Integration test: merge → profile sees combined evidence → reverse → split.

End-to-end arc against a fixture DB (no live ingest required):
  1. Seed mastery on alias + canonical concept IDs
  2. Append ledger merge
  3. Student orchestrator focused mastery aggregates both
  4. Reverse merge
  5. Canonical view excludes alias evidence again
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coast_content_oma.stores import make_namespace
from coast_content_oma.stores.concept_alias import ConceptAliasStore
from coast_content_oma.student.orchestrator import StudentOrchestrator
from coast_content_oma.student.stores import (
    ActiveContextStore,
    AcademicIdentityStore,
    ConceptMasteryStore,
    EpisodeStore,
    PatternStore,
    course_namespace,
)
from coast_content_oma.concept_resolve import resolver_for_db


def _build_orch(db: Path) -> StudentOrchestrator:
    resolver = resolver_for_db(db)
    return StudentOrchestrator(
        active=ActiveContextStore(db),
        mastery=ConceptMasteryStore(db),
        episodes=EpisodeStore(db),
        patterns=PatternStore(db),
        identity=AcademicIdentityStore(db),
        resolver=resolver,
    )


def run_integration() -> None:
    fd, path = tempfile.mkstemp(suffix=".db")
    import os
    os.close(fd)
    db = Path(path)

    user_id, folder = 99, "IntegrationTest"
    content_ns = make_namespace(user_id, folder)
    course_ns = course_namespace(user_id, folder)
    alias = ConceptAliasStore(db)
    orch = _build_orch(db)

    # Seed: student mastered canonical, struggled on alias (simulates pre-merge history)
    orch.mastery.record_evidence(course_ns, "con_canon", "Birth-death process", "success")
    orch.mastery.record_evidence(course_ns, "con_alias", "Birth-death process (old id)", "struggle")

    # Golden moment pattern tied to alias concept id (what Pedro should still find via aggregation)
    orch.patterns.upsert(
        course_ns,
        "golden_moment",
        "traffic-flow analogy for birth-death rates",
        confidence=0.9,
        evidence_count=1,
        related_concept_ids=["con_alias"],
        derivation="test_fixture",
        dedupe_key="golden_traffic",
    )

    def focused_score() -> float | None:
        profile = orch.build_profile(user_id, folder, current_concept_ids=["con_canon"])
        fm = profile.get("focused_mastery") or []
        return fm[0]["score"] if fm else None

    def golden_texts() -> list[str]:
        profile = orch.build_profile(user_id, folder, current_concept_ids=["con_canon"])
        return [g["text"] for g in (profile.get("golden_moments") or [])]

    pre = focused_score()
    assert pre is not None and pre > 0.5, f"pre-merge focused score unexpected: {pre}"

    # Merge alias → canonical via ledger
    assert alias.append_merge(content_ns, "con_alias", "con_canon", reason="integration_test")
    merged = focused_score()
    assert merged is not None
    # Combined: canonical high + alias low → between the two
    assert merged < pre, f"merged view should reflect alias struggle: {merged} vs {pre}"

    # Golden moment still visible (pattern on alias id; profile filters by focused concept)
    gold = golden_texts()
    assert any("traffic-flow" in t for t in gold), f"golden moment missing after merge: {gold}"

    # Reverse — alias evidence leaves canonical aggregate
    assert alias.reverse(content_ns, "con_alias")
    post = focused_score()
    assert post is not None and post >= pre - 0.05, f"post-reverse should match canonical-only: {post} vs {pre}"

    print("INTEGRATION OK")
    print(f"  pre-merge score={pre:.2f}  merged={merged:.2f}  post-reverse={post:.2f}")
    print(f"  golden moments after merge: {gold}")


if __name__ == "__main__":
    run_integration()
