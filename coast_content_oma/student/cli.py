"""Standalone CLI for inspecting Student OMA without Coast integration.

Commands:
  profile       <user_id> <folder>           — show personalized profile
  block         <user_id> <folder>           — show the Pedro prompt block
  episodes      <user_id> <folder> [--limit] — list recent episodes
  mastery       <user_id> <folder>           — list concept mastery
  consolidate   <user_id> <folder>           — run course-level consolidator
  identity      <user_id>                    — show cross-course identity
  identity-run  <user_id> <folder> [<folder>...] — run identity consolidator
  wipe          <user_id> <folder>           — wipe all per-course stores
  wipe-identity <user_id>                    — wipe identity store

The database lives at ./content_oma.db by default — same SQLite file
as Content OMA, just different tables.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .consolidator import CourseConsolidator, IdentityConsolidator
from .orchestrator import build_student_orchestrator
from .recorder import StudentRecorder
from .stores import (
    ActiveContextStore,
    AcademicIdentityStore,
    ConceptMasteryStore,
    EpisodeStore,
    PatternStore,
    course_namespace,
    identity_namespace,
)


def _db_path() -> Path:
    return Path(os.environ.get("OMA_DB_PATH", "./content_oma.db")).resolve()


def cmd_profile(args: list[str]) -> int:
    if len(args) < 2:
        print("Usage: profile <user_id> <folder>")
        return 2
    user_id, folder = args[0], args[1]
    orch = build_student_orchestrator(_db_path())
    profile = orch.build_profile(user_id, folder)
    print(json.dumps(profile, indent=2, default=str))
    return 0


def cmd_block(args: list[str]) -> int:
    if len(args) < 2:
        print("Usage: block <user_id> <folder> [<concept_id>...]")
        return 2
    user_id, folder, *cids = args
    orch = build_student_orchestrator(_db_path())
    profile = orch.build_profile(user_id, folder, current_concept_ids=cids or None)
    print(orch.to_prompt_block(profile))
    return 0


def cmd_episodes(args: list[str]) -> int:
    if len(args) < 2:
        print("Usage: episodes <user_id> <folder> [--limit N]")
        return 2
    user_id, folder = args[0], args[1]
    limit = 20
    for i, a in enumerate(args):
        if a == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])
    ns = course_namespace(user_id, folder)
    store = EpisodeStore(_db_path())
    for it in store.recent(ns, limit=limit):
        ss = it.store_specific or {}
        print(f"[{it.created_at}] {ss.get('episode_type'):20} "
              f"outcome={ss.get('outcome'):8} "
              f"concepts={len(ss.get('concept_ids') or [])} "
              f"— {it.content[:80]}")
    return 0


def cmd_mastery(args: list[str]) -> int:
    if len(args) < 2:
        print("Usage: mastery <user_id> <folder>")
        return 2
    user_id, folder = args[0], args[1]
    ns = course_namespace(user_id, folder)
    store = ConceptMasteryStore(_db_path())
    items = store.all(ns)
    items.sort(key=lambda it: (it.store_specific or {}).get("mastery_score", 0.0))
    print(f"{'Concept':<40} {'Score':>7} {'Conf':>6} {'OK':>4} {'STR':>4} {'Last seen':>20}")
    for it in items:
        ss = it.store_specific or {}
        print(f"{(ss.get('concept_name') or '')[:40]:<40} "
              f"{ss.get('mastery_score', 0):>7.2f} "
              f"{ss.get('confidence', 0):>6.2f} "
              f"{ss.get('successes', 0):>4} "
              f"{ss.get('struggles', 0):>4} "
              f"{(ss.get('last_seen') or '')[:19]:>20}")
    print()
    print("Overview:", json.dumps(store.overview(ns), indent=2))
    return 0


def cmd_consolidate(args: list[str]) -> int:
    if len(args) < 2:
        print("Usage: consolidate <user_id> <folder> [--days N]")
        return 2
    user_id, folder = args[0], args[1]
    days = 30.0
    for i, a in enumerate(args):
        if a == "--days" and i + 1 < len(args):
            days = float(args[i + 1])
    ns = course_namespace(user_id, folder)
    cons = CourseConsolidator(
        episodes=EpisodeStore(_db_path()),
        mastery=ConceptMasteryStore(_db_path()),
        patterns=PatternStore(_db_path()),
    )
    out = cons.run(ns, window_days=days)
    print(json.dumps(out, indent=2))
    # also show top patterns now in the store
    pstore = PatternStore(_db_path())
    print(f"\nPatterns now in store ({len(pstore.all(ns))} total):")
    for p in pstore.top_confidence(ns, k=10):
        ss = p.store_specific or {}
        print(f"  [{ss.get('pattern_type'):25}] conf={ss.get('confidence', 0):.2f} — {p.content}")
    return 0


def cmd_identity(args: list[str]) -> int:
    if len(args) < 1:
        print("Usage: identity <user_id>")
        return 2
    user_id = args[0]
    ns = identity_namespace(user_id)
    store = AcademicIdentityStore(_db_path())
    items = store.all_traits(ns)
    if not items:
        print(f"No identity traits for {ns}")
        return 0
    print(f"Identity traits for {ns}:")
    for it in items:
        ss = it.store_specific or {}
        print(f"  [{ss.get('trait_type'):20}] conf={ss.get('confidence', 0):.2f} "
              f"courses={len(ss.get('evidence_courses') or [])} — {it.content}")
    return 0


def cmd_identity_run(args: list[str]) -> int:
    if len(args) < 2:
        print("Usage: identity-run <user_id> <folder> [<folder>...]")
        return 2
    user_id, *folders = args
    course_nss = [course_namespace(user_id, f) for f in folders]
    cons = IdentityConsolidator(
        pattern_store=PatternStore(_db_path()),
        identity_store=AcademicIdentityStore(_db_path()),
        course_namespaces=course_nss,
    )
    out = cons.run(user_id)
    print(json.dumps(out, indent=2))
    return 0


def cmd_wipe(args: list[str]) -> int:
    if len(args) < 2:
        print("Usage: wipe <user_id> <folder>")
        return 2
    user_id, folder = args[0], args[1]
    ns = course_namespace(user_id, folder)
    counts = {}
    for store in (
        ActiveContextStore(_db_path()),
        ConceptMasteryStore(_db_path()),
        EpisodeStore(_db_path()),
        PatternStore(_db_path()),
    ):
        counts[store.STORE_NAME] = store.delete_namespace(ns)
    print(json.dumps(counts, indent=2))
    return 0


def cmd_wipe_identity(args: list[str]) -> int:
    if len(args) < 1:
        print("Usage: wipe-identity <user_id>")
        return 2
    user_id = args[0]
    ns = identity_namespace(user_id)
    store = AcademicIdentityStore(_db_path())
    n = store.delete_namespace(ns)
    print(f"Deleted {n} identity rows")
    return 0


def main(argv: list[str]) -> int:
    load_dotenv()
    if not argv:
        print(__doc__)
        return 0
    cmd, *rest = argv
    handlers = {
        "profile": cmd_profile,
        "block": cmd_block,
        "episodes": cmd_episodes,
        "mastery": cmd_mastery,
        "consolidate": cmd_consolidate,
        "identity": cmd_identity,
        "identity-run": cmd_identity_run,
        "wipe": cmd_wipe,
        "wipe-identity": cmd_wipe_identity,
    }
    if cmd not in handlers:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        return 2
    return handlers[cmd](rest)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
