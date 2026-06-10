#!/usr/bin/env python3
"""End-to-end test: Content OMA retrieval + Student OMA storage + Pedro personalization.

Three layers:
  1. Content OMA  — material is indexed and retrievable for a query
  2. Student OMA  — interactions are stored, consolidated, profile survives scale
  3. Personalization — profile block contains the signals Pedro needs

Quick mode (no PDFs, isolated temp DB):
  cd coast-local-oma
  python3 scripts/e2e_oma_test.py --quick

Live mode (real user + folder on your coast.db):
  STUDENT_OMA_ENABLED=true RAG_PROVIDER=oma \\
    python3 scripts/e2e_oma_test.py --user-id 13 --folder "Linear Algebra"

Years-scale stress (synthetic, ~3yr history, thousands of episodes):
  python3 scripts/e2e_oma_test.py --quick --years 3 --episodes-per-month 40
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "coast-content-oma" / "src"))
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv()

from coast_content_oma.stores.base import MemoryItem, make_namespace, new_item_id
from coast_content_oma.stores.concept import ConceptStore
from coast_content_oma.stores.content import ContentStore
from coast_content_oma.student.stores import (
    course_namespace,
    identity_namespace,
)
from coast_content_oma.student.consolidator import CourseConsolidator, IdentityConsolidator
from coast_content_oma.student.orchestrator import build_student_orchestrator
from coast_content_oma.student.stores import (
    AcademicIdentityStore,
    ConceptMasteryStore,
    EpisodeStore,
    PatternStore,
)


# ── Helpers ──────────────────────────────────────────────────────────

def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def fail(msg: str, errors: list[str]) -> None:
    print(f"  ✗ {msg}")
    errors.append(msg)


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ── Content OMA (synthetic seed) ─────────────────────────────────────

def seed_content_oma(db: Path, user_id: str, folder: str) -> dict[str, str]:
    """Write a tiny course graph + content chunks — no PDF ingest needed."""
    ns = make_namespace(user_id, folder)
    concepts = ConceptStore(db)
    content = ContentStore(db)

    eigen = MemoryItem(
        id="con_eigenvalue",
        namespace=ns,
        store="concept",
        content="An eigenvalue λ satisfies Av = λv for non-zero vector v.",
        entities=["eigenvalue"],
        store_specific={
            "name": "eigenvalue",
            "aliases": ["eigenvalues"],
            "definition": "Scalar λ where Av = λv.",
            "prerequisite_concept_ids": [],
            "related_concept_ids": [],
            "lecture_sources": ["lec1"],
        },
    )
    svd = MemoryItem(
        id="con_svd",
        namespace=ns,
        store="concept",
        content="Singular value decomposition factors A = UΣVᵀ.",
        entities=["singular value decomposition", "svd"],
        store_specific={
            "name": "singular value decomposition",
            "aliases": ["svd"],
            "definition": "A = UΣVᵀ factorization.",
            "prerequisite_concept_ids": ["con_eigenvalue"],
            "related_concept_ids": ["con_eigenvalue"],
            "lecture_sources": ["lec2"],
        },
    )
    concepts.write_item(eigen)
    concepts.write_item(svd)

    chunk = MemoryItem(
        id="cnt_eigen_def",
        namespace=ns,
        store="content",
        content=(
            "Definition: An eigenvalue of matrix A is a scalar λ such that "
            "there exists a non-zero vector v with Av = λv. The vector v is "
            "called an eigenvector."
        ),
        entities=["con_eigenvalue"],
        tags=["definition"],
        importance=0.9,
        store_specific={
            "content_type": "definition",
            "concept_mentions_raw": ["eigenvalue"],
            "source_doc_id": "lec1",
            "page_number": 3,
        },
    )
    content.write_item(chunk)
    return {"eigenvalue": "con_eigenvalue", "svd": "con_svd"}


def reset_oma_singletons(db: Path) -> None:
    """Point oma_provider at an isolated DB (clears cached orchestrators)."""
    import oma_provider
    oma_provider.OMA_DB_PATH = db
    oma_provider._content_orch = None
    oma_provider._student_orch = None
    oma_provider._student_recorder = None


def test_content_retrieval(db: Path, user_id: str, folder: str, errors: list[str]) -> list[str]:
    """Content OMA should return the eigenvalue definition for a targeted query."""
    reset_oma_singletons(db)
    import oma_provider

    t0 = time.perf_counter()
    block, source, concept_ids = oma_provider.resolve_folder_content(
        user_id, folder,
        query="What is an eigenvalue? Give the definition.",
        max_chars=4000,
    )
    elapsed = time.perf_counter() - t0

    if not block:
        fail("Content OMA returned a non-empty context block", errors)
    else:
        ok(f"Content OMA retrieved {len(block)} chars via {source} in {elapsed:.2f}s")

    if "Av = λv" in block or "Av = λv" in block.replace("λ", "lambda"):
        ok("Retrieved chunk contains eigenvalue definition (Av = λv)")
    elif "eigenvalue" in block.lower() and "non-zero" in block.lower():
        ok("Retrieved chunk contains eigenvalue definition (paraphrase)")
    else:
        fail(f"Expected eigenvalue definition in context block; got snippet: {block[:200]!r}", errors)

    if "con_eigenvalue" in concept_ids:
        ok("Concept id con_eigenvalue surfaced for downstream mastery focus")
    else:
        fail(f"Expected con_eigenvalue in concept_ids; got {concept_ids}", errors)

    return concept_ids


# ── Student OMA (multi-year synthetic history) ───────────────────────

CONCEPTS_A = {
    "con_eigenvalue": "eigenvalue",
    "con_svd": "singular value decomposition",
    "con_det": "determinant",
}
CONCEPTS_B = {
    "con_bayes": "Bayes theorem",
    "con_var": "variance",
}


def _backdate_item(store, item: MemoryItem, created_at: str) -> None:
    item.created_at = created_at
    item.last_accessed = created_at
    store._insert(item)


def simulate_years(
    db: Path,
    user_id: str,
    folder_a: str,
    folder_b: str,
    *,
    years: float,
    episodes_per_month: int,
) -> dict:
    """Simulate long-running student history with backdated timestamps."""
    episode_store = EpisodeStore(db)
    mastery = ConceptMasteryStore(db)
    ns_a = course_namespace(user_id, folder_a)
    ns_b = course_namespace(user_id, folder_b)

    # Bulk episode writes skip embeddings — otherwise 500+ OpenAI calls.
    saved_key = os.environ.pop("OPENAI_API_KEY", None)
    try:
        now = datetime.now()
        start = now - timedelta(days=int(years * 365.25))
        total_days = (now - start).days
        months = max(1, total_days // 30)
        n_created = 0

        concept_cycle_a = list(CONCEPTS_A.items())
        concept_cycle_b = list(CONCEPTS_B.items())

        for month in range(months):
            month_start = start + timedelta(days=month * 30)
            for i in range(episodes_per_month):
                ts = (month_start + timedelta(hours=i * (720 // max(1, episodes_per_month)))).isoformat(timespec="seconds")
                cid_a, name_a = concept_cycle_a[(month + i) % len(concept_cycle_a)]
                outcome_a = "success" if cid_a != "con_svd" else ("struggle" if i % 5 == 0 else "success")

                ep = episode_store.record(
                    ns_a,
                    episode_type="exercise_attempt",
                    summary=f"Practiced {name_a}",
                    outcome=outcome_a,
                    concept_ids=[cid_a],
                    source="exercise",
                )
                _backdate_item(episode_store, ep, ts)
                n_created += 1

                if i == 0 and month % 3 == 0:
                    sec = episode_store.record(
                        ns_a,
                        episode_type="section_completed",
                        summary=f"Completed section {month}",
                        outcome="neutral",
                        section_title=f"Lecture {month + 1}: {name_a}",
                        section_index=month,
                    )
                    _backdate_item(episode_store, sec, ts)

            if month % 2 == 0:
                cid_b, name_b = concept_cycle_b[month % len(concept_cycle_b)]
                ts_b = month_start.isoformat(timespec="seconds")
                ep_b = episode_store.record(
                    ns_b,
                    episode_type="exercise_attempt",
                    summary=f"Practiced {name_b}",
                    outcome="success",
                    concept_ids=[cid_b],
                    source="exercise",
                )
                _backdate_item(episode_store, ep_b, ts_b)
                n_created += 1

        # Seed mastery rows (representative of years of practice).
        for cid, name in CONCEPTS_A.items():
            n_ok = 8 if cid != "con_svd" else 4
            n_bad = 0 if cid != "con_svd" else 3
            for _ in range(n_ok):
                mastery.record_evidence(ns_a, cid, name, "success")
            for _ in range(n_bad):
                mastery.record_evidence(ns_a, cid, name, "struggle")
        for cid, name in CONCEPTS_B.items():
            for _ in range(5):
                mastery.record_evidence(ns_b, cid, name, "success")
    finally:
        if saved_key:
            os.environ["OPENAI_API_KEY"] = saved_key

    # Well-practiced eigenvalue from 2 years ago → should appear as fading.
    old_ts = (datetime.now() - timedelta(days=730)).isoformat(timespec="seconds")
    row = mastery.for_concept(ns_a, "con_eigenvalue")
    if row:
        ss = dict(row.store_specific or {})
        ss["last_seen"] = old_ts
        ss["last_strengthened"] = old_ts
        ss["mastery_score"] = 0.95
        ss["confidence"] = 0.9
        row.store_specific = ss
        mastery._insert(row)

    # Consolidate both courses + identity
    for folder in (folder_a, folder_b):
        CourseConsolidator(
            episodes=EpisodeStore(db),
            mastery=ConceptMasteryStore(db),
            patterns=PatternStore(db),
        ).run(course_namespace(user_id, folder), window_days=365 * max(1, int(years)))

    IdentityConsolidator(
        pattern_store=PatternStore(db),
        identity_store=AcademicIdentityStore(db),
        course_namespaces=[ns_a, ns_b],
    ).run(user_id)

    return {
        "episodes_created": n_created,
        "months": months,
        "episode_count_a": episode_store.count(ns_a),
        "episode_count_b": episode_store.count(ns_b),
    }


def test_student_profile(
    db: Path,
    user_id: str,
    folder_a: str,
    folder_b: str,
    errors: list[str],
    *,
    max_build_seconds: float = 3.0,
) -> str:
    reset_oma_singletons(db)
    import oma_provider
    orch = oma_provider._student_orchestrator()
    t0 = time.perf_counter()
    profile = orch.build_profile(user_id, folder_a, current_concept_ids=["con_eigenvalue"])
    build_ms = (time.perf_counter() - t0) * 1000

    overview = profile.get("mastery_overview") or {}
    if overview.get("n_concepts", 0) >= 2:
        ok(f"Mastery overview: {overview['n_concepts']} concepts, avg={overview.get('avg_mastery', 0):.2f}")
    else:
        fail(f"Expected ≥2 mastery concepts; got {overview}", errors)

    if build_ms <= max_build_seconds * 1000:
        ok(f"Profile built in {build_ms:.0f}ms (budget {max_build_seconds * 1000:.0f}ms)")
    else:
        fail(f"Profile build too slow: {build_ms:.0f}ms > {max_build_seconds * 1000:.0f}ms", errors)

    others = profile.get("other_courses") or []
    if others:
        ok(f"Cross-course memory: {len(others)} other course(s) — {[c['folder'] for c in others]}")
    else:
        fail("Expected other_courses block for multi-course student", errors)

    due = profile.get("due_for_review") or []
    eigen_due = any(d.get("concept_id") == "con_eigenvalue" for d in due)
    if eigen_due:
        ok("Fading mastery: eigenvalue flagged for spaced-repetition review")
    else:
        fail("Expected con_eigenvalue in due_for_review (2yr stale, high raw score)", errors)

    focused = profile.get("focused_mastery") or []
    if focused and focused[0].get("concept_id") == "con_eigenvalue":
        ok("Focused mastery surfaces the concept Pedro is about to teach")
    else:
        fail(f"Expected focused mastery on con_eigenvalue; got {focused}", errors)

    acc_lines = (profile.get("accomplishments") or {}).get("narrative_lines") or []
    if any("eigenvalue" in l.lower() for l in acc_lines):
        ok("Accomplishments mention eigenvalue practice history")
    else:
        fail(f"Expected eigenvalue in accomplishments; got {acc_lines[:3]}", errors)

    block = oma_provider.get_student_profile_block(
        user_id, folder_a, current_concept_ids=["con_eigenvalue"], max_chars=4000,
    )
    if not block:
        fail("Pedro profile block is empty", errors)
    else:
        ok(f"Pedro profile block: {len(block)} chars")

    checks = [
        ("eigenvalue", "weak or strong concept reference"),
        ("PROFILE RULES", "Pedro guardrails"),
        ("History from their other courses", "cross-course autobiographical memory"),
    ]
    for needle, label in checks:
        if needle.lower() in block.lower():
            ok(f"Profile block contains {label!r}")
        else:
            fail(f"Profile block missing {label!r} (searched {needle!r})", errors)

    if "Fading mastery" in block or "fading" in block.lower():
        ok("Profile block surfaces fading/spaced-repetition signal")
    else:
        fail("Profile block should mention fading mastery for stale eigenvalue", errors)

    return block


def test_live(user_id: int, folder: str, errors: list[str]) -> None:
    import oma_provider
    import lesson

    if not oma_provider.is_oma_enabled():
        fail("RAG_PROVIDER is not 'oma'", errors)
        return
    if not oma_provider.is_student_enabled():
        fail("STUDENT_OMA_ENABLED is off", errors)
        return

    ns = course_namespace(user_id, folder)
    orch = oma_provider._student_orchestrator()
    n_episodes = orch.episodes.count(ns)
    n_mastery = orch.mastery.count(ns)
    ok(f"Live DB: {n_episodes} episodes, {n_mastery} mastery rows in {ns}")

    block, source, cids = oma_provider.resolve_folder_content(
        user_id, folder, "Explain the main concepts from this course.",
    )
    if block:
        ok(f"Content retrieval via {source}: {len(block)} chars, {len(cids)} concepts")
    else:
        fail("No content retrieved — ingest PDFs first (oma ingest <user> <folder>)", errors)

    progress = lesson.get_authoritative_progress(user_id, folder) or {}
    profile_block = oma_provider.get_student_profile_block(user_id, folder)
    if profile_block:
        ok(f"Pedro profile block: {len(profile_block)} chars")
        print("\n--- Pedro profile block (first 1200 chars) ---")
        print(profile_block[:1200])
    elif n_mastery or progress.get("completed_sections"):
        fail("Student has history but profile block is empty", errors)
    else:
        ok("No student history yet — profile correctly empty (complete a section first)")


def run_quick(args: argparse.Namespace) -> int:
    errors: list[str] = []
    user_id = 9999
    folder_a = "Linear Algebra"
    folder_b = "Statistics"

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "e2e_oma.db"
        os.environ["OMA_DB_PATH"] = str(db)
        os.environ["RAG_PROVIDER"] = "oma"
        os.environ["STUDENT_OMA_ENABLED"] = "true"

        section("1 · Content OMA — seed + retrieval")
        seed_content_oma(db, user_id, folder_a)
        test_content_retrieval(db, user_id, folder_a, errors)

        section(f"2 · Student OMA — {args.years}yr synthetic history")
        stats = simulate_years(
            db, user_id, folder_a, folder_b,
            years=args.years,
            episodes_per_month=args.episodes_per_month,
        )
        ok(f"Created {stats['episodes_created']} episodes across {stats['months']} months")
        ok(f"Stored: course A={stats['episode_count_a']} episodes, course B={stats['episode_count_b']} episodes")

        section("3 · Personalization — profile + Pedro block")
        block = test_student_profile(db, user_id, folder_a, folder_b, errors)
        print("\n--- Pedro profile block (excerpt) ---")
        print(block[:1500])

    section("Result")
    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        print(f"\nFAILED — {len(errors)} check(s)")
        return 1
    print("ALL CHECKS PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="E2E test for Content OMA + Student OMA + Pedro personalization")
    ap.add_argument("--quick", action="store_true", help="Synthetic test in isolated temp DB (recommended first)")
    ap.add_argument("--user-id", type=int, help="Live test against coast.db for this user")
    ap.add_argument("--folder", help="Live test folder name")
    ap.add_argument("--years", type=float, default=3.0, help="Years of synthetic history (--quick)")
    ap.add_argument("--episodes-per-month", type=int, default=12, help="Episodes per month (--quick)")
    args = ap.parse_args()

    if args.user_id and args.folder:
        errors: list[str] = []
        section(f"Live verification — user {args.user_id}, folder {args.folder!r}")
        test_live(args.user_id, args.folder, errors)
        section("Result")
        if errors:
            for e in errors:
                print(f"  ✗ {e}")
            return 1
        print("ALL CHECKS PASSED")
        return 0

    if args.quick or not (args.user_id or args.folder):
        return run_quick(args)

    print("Provide --quick OR both --user-id and --folder")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
