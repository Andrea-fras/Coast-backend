"""Standalone CLI for testing Content OMA without Coast integration.

Commands:
  ingest <user_id> <folder_name> <pdf_path> [<pdf_path> ...]
  query  <user_id> <folder_name> <query>
  stats  <user_id> <folder_name>
  wipe   <user_id> <folder_name>

The database lives at ./content_oma.db and images at ./content_oma_images/
by default — override with OMA_DB_PATH and OMA_IMAGE_DIR.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .ingestion import IngestionPipeline
from .orchestrator import build_orchestrator
from .stores import make_namespace


def _paths() -> tuple[Path, Path]:
    db = Path(os.environ.get("OMA_DB_PATH", "./content_oma.db")).resolve()
    images = Path(os.environ.get("OMA_IMAGE_DIR", "./content_oma_images")).resolve()
    return db, images


def cmd_ingest(args: list[str]) -> int:
    if len(args) < 3:
        print("Usage: ingest [--with-vision] [--skip-images] <user_id> <folder_name> <pdf> [<pdf> ...]")
        print("  By default, images are saved without vision descriptions (fast).")
        print("  --skip-images: don't extract images at all (fastest, text only).")
        print("  --with-vision: run vision LLM inline during ingestion (slowest).")
        return 2
    with_vision = "--with-vision" in args
    skip_images = "--skip-images" in args
    args = [a for a in args if a not in ("--with-vision", "--skip-images")]
    if len(args) < 3:
        print("error: need <user_id> <folder> <pdf>")
        return 2
    user_id, folder, *pdf_args = args
    ns = make_namespace(user_id, folder)
    db, image_dir = _paths()
    orch = build_orchestrator(db, image_dir)
    pipeline = IngestionPipeline(
        orch.concept, orch.content, orch.images, image_dir,
        describe_images=with_vision,
        skip_images=skip_images,
    )
    pdfs = [Path(p).expanduser().resolve() for p in pdf_args]
    missing = [p for p in pdfs if not p.exists()]
    if missing:
        print(f"error: missing PDFs: {missing}")
        return 2
    import time as _t
    t0 = _t.time()
    stats = pipeline.ingest_folder(ns, pdfs, progress=lambda m: print(f"  → {m}", flush=True))
    elapsed = _t.time() - t0
    print(f"\nDone in {elapsed:.1f}s:")
    print(json.dumps(stats.to_dict(), indent=2))
    if not with_vision and stats.image_items > 0:
        print(f"\nNote: {stats.image_items} images saved without vision descriptions.")
        print(f"Run: python3 -m coast_content_oma.cli describe-images {user_id} \"{folder}\"")
    return 0


def cmd_describe_images(args: list[str]) -> int:
    if len(args) < 2:
        print("Usage: describe-images <user_id> <folder_name> [--limit N]")
        return 2
    user_id, folder = args[0], args[1]
    limit = None
    for i, a in enumerate(args):
        if a == "--limit" and i + 1 < len(args):
            try:
                limit = int(args[i + 1])
            except ValueError:
                pass
    ns = make_namespace(user_id, folder)
    db, image_dir = _paths()
    orch = build_orchestrator(db, image_dir)
    pipeline = IngestionPipeline(orch.concept, orch.content, orch.images, image_dir)
    result = pipeline.describe_pending_images(ns, progress=lambda m: print(f"  → {m}"), limit=limit)
    print(f"\nDone: {result}")
    return 0


def cmd_query(args: list[str]) -> int:
    if len(args) < 3:
        print("Usage: query <user_id> <folder_name> <query>")
        return 2
    user_id, folder, *q = args
    query = " ".join(q)
    ns = make_namespace(user_id, folder)
    db, image_dir = _paths()
    orch = build_orchestrator(db, image_dir)
    result = orch.retrieve(ns, query)
    print(f"Query: {query}")
    print(f"Class: {result.query_class}")
    print(f"Concepts: {[(c.store_specific or {}).get('name') for c in result.concept_candidates]}")
    print(f"Chunks: {len(result.chunks)}")
    print(f"Images: {len(result.images)}")
    print("\n--- Prompt block ---\n")
    print(result.to_prompt_block(max_chars=6000))
    return 0


def cmd_stats(args: list[str]) -> int:
    if len(args) < 2:
        print("Usage: stats <user_id> <folder_name>")
        return 2
    user_id, folder = args[0], args[1]
    ns = make_namespace(user_id, folder)
    db, image_dir = _paths()
    orch = build_orchestrator(db, image_dir)
    print(json.dumps({
        "namespace": ns,
        "concept": orch.concept.stats(ns),
        "content": orch.content.stats(ns),
        "image": orch.images.stats(ns),
    }, indent=2))
    return 0


def cmd_recanonicalize(args: list[str]) -> int:
    if len(args) < 2:
        print("Usage: recanonicalize <user_id> <folder_name>")
        return 2
    user_id, folder = args[0], args[1]
    ns = make_namespace(user_id, folder)
    db, image_dir = _paths()
    orch = build_orchestrator(db, image_dir)
    pipeline = IngestionPipeline(orch.concept, orch.content, orch.images, image_dir)
    result = pipeline.recanonicalize(ns, progress=lambda m: print(f"  → {m}"))
    id_map = result.get("id_map") or {}
    if id_map:
        from ..student.concept_remap import remap_student_concept_ids
        from ..student.orchestrator import build_student_orchestrator
        from ..student.stores import course_namespace
        student_ns = course_namespace(user_id, folder)
        student_orch = build_student_orchestrator(db)
        remap_stats = remap_student_concept_ids(student_orch, student_ns, id_map)
        result["student_remap"] = remap_stats
        print(f"  → remapped student evidence: {remap_stats}")
    print(f"\nDone: {result}")
    return 0


def cmd_wipe(args: list[str]) -> int:
    if len(args) < 2:
        print("Usage: wipe <user_id> <folder_name>")
        return 2
    user_id, folder = args[0], args[1]
    ns = make_namespace(user_id, folder)
    db, image_dir = _paths()
    orch = build_orchestrator(db, image_dir)
    n1 = orch.concept.delete_namespace(ns)
    n2 = orch.content.delete_namespace(ns)
    n3 = orch.images.delete_namespace(ns)
    print(f"Deleted: concepts={n1}, content={n2}, images={n3}")
    return 0


def main(argv: list[str]) -> int:
    load_dotenv()
    if not argv:
        print(__doc__)
        return 0
    cmd, *rest = argv
    if cmd == "ingest":
        return cmd_ingest(rest)
    if cmd == "query":
        return cmd_query(rest)
    if cmd == "stats":
        return cmd_stats(rest)
    if cmd == "recanonicalize" or cmd == "recanon":
        return cmd_recanonicalize(rest)
    if cmd == "describe-images" or cmd == "describe":
        return cmd_describe_images(rest)
    if cmd == "wipe":
        return cmd_wipe(rest)
    print(f"unknown command: {cmd}")
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
