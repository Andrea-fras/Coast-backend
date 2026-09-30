#!/usr/bin/env python3
"""Seed curated audit personas (sparse / medium / rich) into coast.db + oma.db.

Usage:
  cd OCR
  RAG_PROVIDER=oma STUDENT_OMA_ENABLED=true python3 scripts/persona_audit/seed.py
  python3 scripts/persona_audit/seed.py --persona medium
  python3 scripts/persona_audit/seed.py --wipe-only
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
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
from coast_content_oma.student.stores import (
    AcademicIdentityStore,
    ConceptMasteryStore,
    EpisodeStore,
    PatternStore,
)
from coast_content_oma.student.mastery_tier import sync_mastery_tier

from scripts.persona_audit import (
    FIXTURES_DIR,
    PERSONA_IDS,
    load_course,
    load_persona,
)

AUDIT_TABLES = (
    "episode_items",
    "concept_mastery_items",
    "pattern_items",
    "active_context_items",
)


def _oma_db_path() -> Path:
    return Path(os.environ.get("OMA_DB_PATH", str(ROOT / "oma_data" / "oma.db")))


def _wipe_namespace(db_path: Path, namespace: str) -> None:
    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for table in AUDIT_TABLES:
            if table not in tables:
                continue
            n = conn.execute(
                f"DELETE FROM {table} WHERE namespace = ?", (namespace,)
            ).rowcount
            if n:
                print(f"  cleared {n} from {table} ({namespace})")


def _wipe_identity(db_path: Path, user_id: int) -> None:
    ns = identity_namespace(user_id)
    _wipe_namespace(db_path, ns)


def seed_content_course(db_path: Path, user_id: int, course: dict) -> None:
    """Shared Content OMA for audit folder (same material for all personas)."""
    folder = course["folder"]
    ns = make_namespace(user_id, course["folder"])
    concepts = ConceptStore(db_path)
    content = ContentStore(db_path)

    for c in course["concepts"]:
        item = MemoryItem(
            id=c["id"],
            namespace=ns,
            store="concept",
            content=c.get("definition") or c["name"],
            entities=[c["name"]] + list(c.get("aliases") or []),
            store_specific={
                "name": c["name"],
                "aliases": list(c.get("aliases") or []),
                "definition": c.get("definition") or "",
                "prerequisite_concept_ids": list(c.get("prerequisite_concept_ids") or []),
                "related_concept_ids": [],
                "lecture_sources": ["audit_lec"],
            },
        )
        concepts.write_item(item)

    for ch in course.get("content_chunks") or []:
        item = MemoryItem(
            id=ch["id"],
            namespace=ns,
            store="content",
            content=ch["text"],
            entities=[ch["concept_id"]],
            tags=["definition"],
            importance=0.85,
            store_specific={
                "content_type": "definition",
                "concept_mentions_raw": [ch["concept_id"]],
                "source_doc_id": "audit_lec",
                "page_number": 1,
            },
        )
        content.write_item(item)

    print(f"  content OMA seeded: {folder} ({len(course['concepts'])} concepts)")


def seed_prior_course(db_path: Path, user_id: int, prior: dict) -> None:
    folder = prior["folder"]
    ns = course_namespace(user_id, folder)
    mastery = ConceptMasteryStore(db_path)
    episodes = EpisodeStore(db_path)

    for m in prior.get("mastery") or []:
        cid = m["concept_id"]
        cname = m["concept_name"]
        item = mastery.record_evidence(ns, cid, cname, "success")
        ss = dict(item.store_specific or {})
        ss["mastery_score"] = float(m.get("mastery_score", 0.85))
        ss["successes"] = int(m.get("successes", 3))
        ss["struggles"] = int(m.get("struggles", 0))
        sync_mastery_tier(ss)
        item.store_specific = ss
        item.content = mastery._summary_text(cname, ss)
        mastery._insert(item)

    for i, title in enumerate(prior.get("sections_completed") or []):
        episodes.record(
            ns,
            episode_type="section_completed",
            summary=f"Completed section: {title}",
            outcome="success",
            section_title=title,
            section_index=i,
            source="audit_seed",
        )
    print(f"  prior course: {folder}")


def seed_student_oma(db_path: Path, persona: dict, course: dict) -> None:
    uid = persona["user_id"]
    folder = course["folder"]
    ns = course_namespace(uid, folder)
    identity_ns = identity_namespace(uid)

    mastery_store = ConceptMasteryStore(db_path)
    episode_store = EpisodeStore(db_path)
    pattern_store = PatternStore(db_path)
    identity_store = AcademicIdentityStore(db_path)

    _wipe_namespace(db_path, ns)
    _wipe_identity(db_path, uid)

    for trait in persona.get("identity_traits") or []:
        identity_store.upsert_trait(
            identity_ns,
            trait_type=trait["trait_type"],
            description=trait["description"],
            confidence=float(trait.get("confidence", 0.8)),
            evidence_courses=["onboarding"],
            derivation=trait.get("derivation", "Pedro onboarding conversation"),
            dedupe_key=trait["trait_type"],
        )

    now = datetime.now()
    for m in persona.get("mastery") or []:
        cid = m["concept_id"]
        cname = m["concept_name"]
        item = mastery_store.record_evidence(ns, cid, cname, "success")
        ss = dict(item.store_specific or {})
        ss["mastery_score"] = float(m.get("mastery_score", 0.5))
        ss["successes"] = int(m.get("successes", 1))
        ss["struggles"] = int(m.get("struggles", 0))
        if m.get("last_misconception"):
            ss["last_misconception"] = True
        if m.get("last_eval_state"):
            ss["last_eval_state"] = m["last_eval_state"]
        days_ago = m.get("last_seen_days_ago")
        if days_ago is not None:
            ts = (now - timedelta(days=float(days_ago))).isoformat(timespec="seconds")
            ss["last_seen"] = ts
            ss["last_strengthened"] = ts
        sync_mastery_tier(ss)
        item.store_specific = ss
        item.content = mastery_store._summary_text(cname, ss)
        item.tags = [ss["mastery_tier"], mastery_store._mastery_tag(ss["mastery_score"])]
        mastery_store._insert(item)

    for ep in persona.get("episodes") or []:
        ep_item = episode_store.record(
            ns,
            episode_type=ep.get("episode_type", "external_event"),
            summary=ep.get("summary") or ep.get("user_message") or ep.get("section_title") or "audit episode",
            outcome=ep.get("outcome", "neutral"),
            concept_ids=list(ep.get("concept_ids") or []),
            user_message=ep.get("user_message"),
            signals=dict(ep.get("signals") or {}),
            source="audit_seed",
            section_title=ep.get("section_title"),
            section_index=ep.get("section_index"),
        )
        if ep.get("evaluation"):
            ss = dict(ep_item.store_specific or {})
            ss["evaluation"] = ep["evaluation"]
            ep_item.store_specific = ss
            episode_store._insert(ep_item)

    for p in persona.get("patterns") or []:
        pattern_store.upsert(
            ns,
            p["pattern_type"],
            p["description"],
            confidence=float(p.get("confidence", 0.75)),
            evidence_count=int(p.get("evidence_count", 1)),
            related_concept_ids=list(p.get("related_concept_ids") or []),
            derivation="audit persona seed",
            dedupe_key=p.get("dedupe_key") or p["pattern_type"],
        )

    for gm in persona.get("golden_moments") or []:
        cid = gm.get("concept_id") or ""
        desc = gm.get("description") or ""
        pattern_store.upsert(
            ns,
            "golden_moment",
            desc[:400],
            confidence=0.85,
            evidence_count=1,
            related_concept_ids=[cid] if cid else [],
            derivation="audit persona seed",
            dedupe_key=f"golden_{cid or persona['id']}_{hash(desc[:40]) % 9999}",
        )

    if persona.get("prior_course"):
        _wipe_namespace(db_path, course_namespace(uid, persona["prior_course"]["folder"]))
        seed_prior_course(db_path, uid, persona["prior_course"])

    print(f"  student OMA: {persona['id']} (user {uid})")


def ensure_coast_user(persona: dict, course: dict) -> None:
    from database import CourseOutline, SessionLocal, User

    db = SessionLocal()
    try:
        user = db.query(User).filter(
            (User.id == persona["user_id"]) | (User.email == persona["email"])
        ).first()
        prefs = json.dumps(persona.get("learning_preferences") or {})
        if not user:
            user = User(
                id=persona["user_id"],
                email=persona["email"],
                name=persona["name"],
                password_hash="",
                course="Audit",
                learning_preferences=prefs,
                onboarding_completed=True,
            )
            db.add(user)
        else:
            user.email = persona["email"]
            user.name = persona["name"]
            user.learning_preferences = prefs
            user.onboarding_completed = True

        folder = course["folder"]
        outline = (
            db.query(CourseOutline)
            .filter(CourseOutline.user_id == persona["user_id"], CourseOutline.folder_name == folder)
            .first()
        )
        sections = course.get("outline_sections") or []
        progress = persona.get("outline_progress") or {}
        outline_json = json.dumps(sections)
        if not outline:
            outline = CourseOutline(
                user_id=persona["user_id"],
                folder_name=folder,
                outline_json=outline_json,
                total_sections=len(sections),
                current_section=int(progress.get("current_section", 0)),
            )
            db.add(outline)
        else:
            outline.outline_json = outline_json
            outline.total_sections = len(sections)
            outline.current_section = int(progress.get("current_section", 0))

        db.commit()
    finally:
        db.close()


def seed_persona(persona_id: str) -> None:
    course = load_course()
    persona = load_persona(persona_id)
    db_path = _oma_db_path()

    import oma_provider
    oma_provider._student_orch = None
    oma_provider._student_recorder = None
    oma_provider._content_orch = None

    print(f"\n=== Seeding persona: {persona_id} (user {persona['user_id']}) ===")
    seed_content_course(db_path, persona["user_id"], course)
    seed_student_oma(db_path, persona, course)
    ensure_coast_user(persona, course)
    print(f"=== Done: {persona_id} ===")


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed Pedro audit personas")
    ap.add_argument("--persona", choices=PERSONA_IDS, help="Seed one persona only")
    ap.add_argument("--wipe-only", action="store_true", help="Wipe audit namespaces only")
    args = ap.parse_args()

    if not os.environ.get("RAG_PROVIDER"):
        os.environ["RAG_PROVIDER"] = "oma"
    os.environ.setdefault("STUDENT_OMA_ENABLED", "true")

    db_path = _oma_db_path()
    course = load_course()

    if args.wipe_only:
        for pid in PERSONA_IDS:
            p = load_persona(pid)
            _wipe_namespace(db_path, course_namespace(p["user_id"], course["folder"]))
            _wipe_identity(db_path, p["user_id"])
        print("Wiped all audit personas.")
        return 0

    targets = [args.persona] if args.persona else list(PERSONA_IDS)
    for pid in targets:
        seed_persona(pid)
    print("\nAll personas ready. Run audit:")
    print("  python3 scripts/persona_audit/run_audit.py --block-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
