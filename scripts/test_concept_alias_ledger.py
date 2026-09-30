#!/usr/bin/env python3
"""Unit tests for ConceptAliasStore + cross-namespace mastery aggregation."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coast_content_oma.concept_resolve import ConceptResolver, resolver_for_db
from coast_content_oma.stores import make_namespace
from coast_content_oma.stores.concept_alias import ConceptAliasStore
from coast_content_oma.student.mastery_aggregate import aggregate_mastery_rows
from coast_content_oma.student.stores import course_namespace
from coast_content_oma.student.stores.concept_mastery import ConceptMasteryStore


def _tmp_db() -> Path:
    fd, path = tempfile.mkstemp(suffix=".db")
    import os
    os.close(fd)
    return Path(path)


def test_append_merge_rejects_cycle():
    db = _tmp_db()
    ns = "u1__test_course"
    store = ConceptAliasStore(db)
    assert store.append_merge(ns, "con_a", "con_b")
    assert not store.append_merge(ns, "con_b", "con_a"), "B→A must fail when A→B exists"
    assert store.resolve(ns, "con_a") == "con_b"
    print("  OK  test_append_merge_rejects_cycle")


def test_resolve_three_deep_chain():
    db = _tmp_db()
    ns = "u1__chain"
    store = ConceptAliasStore(db)
    store.append_merge(ns, "con_47", "con_12")
    store.append_merge(ns, "con_12", "con_3")
    assert store.resolve(ns, "con_47") == "con_3"
    assert store.resolve(ns, "con_12") == "con_3"
    assert store.resolve(ns, "con_3") == "con_3"
    print("  OK  test_resolve_three_deep_chain")


def test_reverse_middle_of_chain():
    db = _tmp_db()
    ns = "u1__chain_rev"
    store = ConceptAliasStore(db)
    store.append_merge(ns, "con_47", "con_12")
    store.append_merge(ns, "con_12", "con_3")
    assert store.reverse(ns, "con_12")
    # con_47 → con_12 still active; con_12 no longer → con_3
    assert store.resolve(ns, "con_47") == "con_12"
    assert store.resolve(ns, "con_12") == "con_12"
    assert store.resolve(ns, "con_3") == "con_3"
    print("  OK  test_reverse_middle_of_chain")


def test_ids_for_canonical_excludes_reversed_alias():
    db = _tmp_db()
    ns = "u1__rev_ids"
    store = ConceptAliasStore(db)
    resolver = ConceptResolver(store)
    store.append_merge(ns, "con_47", "con_12")
    assert "con_47" in resolver.ids_for_canonical(ns, "con_12")
    store.reverse(ns, "con_47")
    ids = resolver.ids_for_canonical(ns, "con_12")
    assert "con_47" not in ids
    assert "con_12" in ids
    print("  OK  test_ids_for_canonical_excludes_reversed_alias")


def test_aggregate_canonical_plus_two_aliases():
    rows = [
        {"concept_id": "con_12", "concept_name": "Canonical", "mastery_score": 0.8,
         "successes": 4, "struggles": 0, "neutral_touches": 0,
         "first_seen": "2026-01-01", "last_seen": "2026-02-01"},
        {"concept_id": "con_47", "concept_name": "Alias A", "mastery_score": 0.4,
         "successes": 1, "struggles": 2, "neutral_touches": 0,
         "first_seen": "2026-01-02", "last_seen": "2026-02-02"},
        {"concept_id": "con_99", "concept_name": "Alias B", "mastery_score": 0.6,
         "successes": 2, "struggles": 0, "neutral_touches": 0,
         "first_seen": "2026-01-03", "last_seen": "2026-02-03"},
    ]
    agg = aggregate_mastery_rows(rows, canonical_id="con_12", canonical_name="Canonical")
    assert agg["successes"] == 7
    assert agg["struggles"] == 2
    assert 0.5 < agg["mastery_score"] < 0.75
    print("  OK  test_aggregate_canonical_plus_two_aliases")


def test_aggregate_after_reverse_drops_alias_evidence():
    db = _tmp_db()
    content_ns = make_namespace(1, "Ops")
    course_ns = course_namespace(1, "Ops")
    resolver = resolver_for_db(db)
    alias = ConceptAliasStore(db)
    mastery = ConceptMasteryStore(db)

    mastery.record_evidence(course_ns, "con_12", "Canonical", "success")
    mastery.record_evidence(course_ns, "con_47", "Alias", "struggle")

    alias.append_merge(content_ns, "con_47", "con_12")
    agg_merged = mastery.aggregate_for_concept(course_ns, content_ns, "con_12", resolver)
    assert agg_merged["successes"] >= 1 and agg_merged["struggles"] >= 1

    alias.reverse(content_ns, "con_47")
    agg_split = mastery.aggregate_for_concept(course_ns, content_ns, "con_12", resolver)
    assert agg_split["struggles"] == 0, "alias struggle must leave canonical view after reverse"
    assert agg_split["successes"] == 1
    print("  OK  test_aggregate_after_reverse_drops_alias_evidence")


def test_cross_namespace_resolve_isolated():
    db = _tmp_db()
    content_a = make_namespace(1, "CourseA")
    content_b = make_namespace(1, "CourseB")
    store = ConceptAliasStore(db)
    store.append_merge(content_a, "con_x", "con_y")
    resolver = ConceptResolver(store)
    assert resolver.resolve(content_a, "con_x") == "con_y"
    assert resolver.resolve(content_b, "con_x") == "con_x", "aliases must not leak across namespaces"
    print("  OK  test_cross_namespace_resolve_isolated")


def main() -> int:
    tests = [
        test_append_merge_rejects_cycle,
        test_resolve_three_deep_chain,
        test_reverse_middle_of_chain,
        test_ids_for_canonical_excludes_reversed_alias,
        test_aggregate_canonical_plus_two_aliases,
        test_aggregate_after_reverse_drops_alias_evidence,
        test_cross_namespace_resolve_isolated,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as exc:
            failed += 1
            print(f"  FAIL {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
