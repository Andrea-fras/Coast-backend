"""Curated lesson folder configuration — shared across server, tutor, lesson."""

from __future__ import annotations

import json
import os
import uuid
import traceback
from pathlib import Path

CURATED_FOLDER_NAMES = {
    "Quantitative Methods 1",
    "Data Structures & Algorithms",
    "Prismatic System",
    "Science of Cooking",
}
CURATED_CONTENT_DIR = Path(__file__).parent / "curated_content"
CURATED_USER_ID = 0

FOLDER_TO_COURSE: dict[str, str] = {
    "Quantitative Methods 1": "QM1",
    "Data Structures & Algorithms": "DSA",
    "Prismatic System": "PRISMATIC",
    "Science of Cooking": "SOC",
}

_COURSE_KEYWORDS: dict[str, list[str]] = {
    "QM1": ["quantitative methods", "qm1", "statistics"],
    "DSA": ["data structures", "algorithms", "dsa"],
}

def get_course_for_folder(folder_name: str) -> str | None:
    """Return the past-paper course code for a folder, or None.

    Checks exact mapping first, then fuzzy-matches folder name
    against course keywords so user-created folders also work.
    """
    if folder_name in FOLDER_TO_COURSE:
        return FOLDER_TO_COURSE[folder_name]
    name_lower = folder_name.lower()
    for course, keywords in _COURSE_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return course
    return None

# Premade lessons ship with a fixed outline — no generate step for the student.
CURATED_STATIC_OUTLINES: dict[str, list[dict]] = {
    "Prismatic System": [
        {
            "title": "The Elements",
            "learning_objectives": [
                "Identify the three element types in the Prismatic System",
                "Apply the correct value rule for each element type",
                "Recognize how element types constrain later operations",
            ],
            "key_topics": [
                "prismatic elements",
                "element types",
                "value rules",
                "symbolic foundations",
            ],
            "source_notebooks": ["prismatic L1 elements"],
            "estimated_minutes": 25,
        },
        {
            "title": "The Binding Operation (⊕)",
            "learning_objectives": [
                "Apply the binding operation to pairs of elements",
                "Use the six binding rules that depend on type and order",
                "Predict binding results before computing them",
            ],
            "key_topics": [
                "binding operation",
                "type rules",
                "order rules",
                "binary composition",
            ],
            "source_notebooks": ["prismatic L2 binding"],
            "estimated_minutes": 30,
        },
        {
            "title": "The Folding Operation (∆)",
            "learning_objectives": [
                "Transform binding results using the folding operation",
                "Apply the three conditional folding rules correctly",
                "Produce a positive numeric result from any binding outcome",
            ],
            "key_topics": [
                "folding operation",
                "conditional rules",
                "positive transformation",
                "binding results",
            ],
            "source_notebooks": ["prismatic L3 folding"],
            "estimated_minutes": 25,
        },
        {
            "title": "Chains",
            "learning_objectives": [
                "Execute multi-step Prismatic sequences from start to finish",
                "Combine element, binding, and folding rules in long problems",
                "Debug chain errors by locating the first violated rule",
            ],
            "key_topics": [
                "multi-step chains",
                "rule integration",
                "sequential reasoning",
                "full system mastery",
            ],
            "source_notebooks": ["prismatic L4 chains"],
            "estimated_minutes": 35,
        },
    ],
    "Science of Cooking": [
        {
            "title": "The Maillard Reaction",
            "learning_objectives": [
                "Explain non-enzymatic browning and flavor synthesis at high heat",
                "Describe the amino acid–reducing sugar pathway to melanoidins",
                "Diagnose common searing and caramelization failures using chemistry",
            ],
            "key_topics": [
                "Maillard reaction",
                "thermal transformation",
                "melanoidins",
                "kinetic chemistry",
                "browning",
            ],
            "source_notebooks": ["science of cooking lecture 1 maillard"],
            "estimated_minutes": 30,
        },
        {
            "title": "Emulsification & Interfacial Science",
            "learning_objectives": [
                "Explain why oil and water phase-separate and how emulsifiers stabilize mixtures",
                "Distinguish oil-in-water vs water-in-oil emulsions in common kitchen products",
                "Troubleshoot broken emulsions using interfacial science concepts",
            ],
            "key_topics": [
                "emulsification",
                "surfactants",
                "interfacial tension",
                "mayonnaise",
                "hollandaise",
            ],
            "source_notebooks": ["science of cooking lecture 2 emulsification"],
            "estimated_minutes": 30,
        },
        {
            "title": "The Gluten Network",
            "learning_objectives": [
                "Describe how gliadin and glutenin form a viscoelastic protein network",
                "Relate kneading and hydration to polymer cross-linking in dough",
                "Predict texture outcomes when flour protein content is mismatched to a recipe",
            ],
            "key_topics": [
                "gluten network",
                "gliadin",
                "glutenin",
                "polymer physics",
                "dough elasticity",
            ],
            "source_notebooks": ["science of cooking lecture 3 gluten"],
            "estimated_minutes": 30,
        },
        {
            "title": "Thermodynamics of Leavening",
            "learning_objectives": [
                "Explain how yeast fermentation and gas laws drive bread volume expansion",
                "Describe nucleation, cavitation, and gas retention in rising dough",
                "Diagnose dense or collapsed loaves using leavening thermodynamics",
            ],
            "key_topics": [
                "leavening",
                "fermentation",
                "gas laws",
                "Charles's law",
                "bread rising",
            ],
            "source_notebooks": ["science of cooking lecture 4 leavening"],
            "estimated_minutes": 35,
        },
    ],
}

CURATED_LESSON_STRUCTURES = {
    "Quantitative Methods 1": {
        "description": "This course has THREE distinct parts that MUST be reflected as top-level groupings:",
        "parts": [
            {
                "name": "Statistics",
                "description": "Probability, distributions, hypothesis testing, confidence intervals",
                "source_patterns": ["Statistics", "week"],
            },
            {
                "name": "Mathematics",
                "description": "Functions, calculus, linear equations, algebra",
                "source_patterns": ["QMI_", "QM1.pdf"],
            },
            {
                "name": "Computer Skills",
                "description": "Excel, data tools, practical computing for data science",
                "source_patterns": ["Computer skills"],
            },
        ],
    },
    "Data Structures & Algorithms": {
        "description": (
            "This course covers fundamental data structures and algorithms. "
            "The teaching approach is CODE-FIRST: break every large code exercise or algorithm "
            "into small, individually understandable building blocks. Teach each block with a short "
            "explanation and a minimal code snippet. Only after all blocks are understood, combine "
            "them into the full solution. The student should write/complete the final assembled code."
        ),
        "pedagogy": (
            "CRITICAL TEACHING METHODOLOGY — Building-Blocks Approach:\n"
            "1. When presenting a code exercise or algorithm, NEVER show the full solution first.\n"
            "2. Decompose it into 3-6 small logical steps (building blocks).\n"
            "3. For each block: explain the concept in 2-3 sentences, then show a SHORT code snippet "
            "(5-15 lines max) that implements just that block.\n"
            "4. After each block, ask the student a quick check question to confirm understanding.\n"
            "5. Only at the END, present the full assembled solution and ask the student to trace through it.\n"
            "6. For complex algorithms (sorting, graph traversal, dynamic programming), use concrete "
            "small examples (arrays of 5-6 elements) to walk through each step before showing code."
        ),
        "parts": [
            {
                "name": "Foundations",
                "description": "Arrays, linked lists, stacks, queues, complexity analysis (Big-O)",
                "source_patterns": ["array", "linked", "stack", "queue", "complex", "big-o", "foundation", "intro"],
            },
            {
                "name": "Trees & Graphs",
                "description": "Binary trees, BSTs, heaps, graph representations, BFS, DFS",
                "source_patterns": ["tree", "graph", "bst", "heap", "bfs", "dfs", "traversal"],
            },
            {
                "name": "Sorting & Searching",
                "description": "Sorting algorithms, binary search, hashing",
                "source_patterns": ["sort", "search", "hash", "merge", "quick"],
            },
            {
                "name": "Advanced Algorithms",
                "description": "Dynamic programming, greedy algorithms, recursion, divide and conquer",
                "source_patterns": ["dynamic", "greedy", "recursion", "divide", "advanced", "dp"],
            },
        ],
    },
    "Prismatic System": {
        "description": (
            "Four-layer formal system: Elements → Binding (⊕) → Folding (∆) → Chains. "
            "Each lecture introduces rules that depend on all prior layers."
        ),
        "pedagogy": (
            "Teach in strict layer order. Use concrete worked examples before abstract rules. "
            "After each rule, give a short practice item. Chains should only appear after "
            "Layers 1–3 are solid — they are the capstone integration test."
        ),
        "parts": [
            {
                "name": "Layer 1 — Elements",
                "description": "Three element types and their value rules",
                "source_patterns": ["L1", "elements"],
            },
            {
                "name": "Layer 2 — Binding",
                "description": "The binding operation and six type/order rules",
                "source_patterns": ["L2", "binding"],
            },
            {
                "name": "Layer 3 — Folding",
                "description": "The folding operation and conditional transformation rules",
                "source_patterns": ["L3", "folding"],
            },
            {
                "name": "Layer 4 — Chains",
                "description": "Multi-step sequences combining every prior rule",
                "source_patterns": ["L4", "chains"],
            },
        ],
    },
    "Science of Cooking": {
        "description": (
            "Four lectures on culinary chemistry and soft-matter physics: "
            "Maillard browning → emulsification → gluten networks → bread leavening."
        ),
        "pedagogy": (
            "Connect every abstract concept to a concrete kitchen scenario. "
            "Use the interactive diagnostics at the end of each lecture as practice prompts. "
            "When explaining mechanisms, name the molecules and forces involved, then "
            "ask the student to diagnose a real cooking failure."
        ),
        "parts": [
            {
                "name": "Lecture 1 — Maillard Reaction",
                "description": "Non-enzymatic browning, flavor synthesis, thermal kinetics",
                "source_patterns": ["lecture 1", "maillard", "browning", "caramel"],
            },
            {
                "name": "Lecture 2 — Emulsification",
                "description": "Interfacial science, surfactants, stable vs unstable emulsions",
                "source_patterns": ["lecture 2", "emulsif", "mayonnaise", "hollandaise"],
            },
            {
                "name": "Lecture 3 — Gluten Network",
                "description": "Gliadin, glutenin, viscoelasticity, flour protein content",
                "source_patterns": ["lecture 3", "gluten", "gliadin", "glutenin", "dough"],
            },
            {
                "name": "Lecture 4 — Leavening",
                "description": "Fermentation, gas laws, nucleation, bread rising thermodynamics",
                "source_patterns": ["lecture 4", "leaven", "ferment", "bread rises", "yeast"],
            },
        ],
    },
}


def curated_source_uid(folder_name: str) -> int | None:
    """Return shared user_id (0) if folder is curated, else None."""
    return CURATED_USER_ID if folder_name in CURATED_FOLDER_NAMES else None


def get_lesson_structure(folder_name: str) -> dict | None:
    """Return the custom structure hints for a curated lesson, or None."""
    return CURATED_LESSON_STRUCTURES.get(folder_name)


def get_static_outline(folder_name: str) -> list[dict] | None:
    """Return a premade outline that skips generation, or None."""
    outline = CURATED_STATIC_OUTLINES.get(folder_name)
    return list(outline) if outline else None


def ensure_curated_outline(user_id: int, folder_name: str) -> bool:
    """Seed a static outline for premade lessons. Returns True if outline exists."""
    sections = get_static_outline(folder_name)
    if not sections:
        return False

    from datetime import datetime, timezone
    from database import SessionLocal, CourseOutline

    db = SessionLocal()
    try:
        existing = (
            db.query(CourseOutline)
            .filter(CourseOutline.user_id == user_id, CourseOutline.folder_name == folder_name)
            .first()
        )
        if existing:
            return True

        total_minutes = sum(int(s.get("estimated_minutes") or 20) for s in sections)
        db.add(CourseOutline(
            user_id=user_id,
            folder_name=folder_name,
            outline_json=json.dumps(sections),
            total_sections=len(sections),
            current_section=0,
            estimated_minutes=total_minutes,
        ))
        db.commit()
        return True
    finally:
        db.close()


def _curated_pdf_sources(folder_name: str) -> list[dict]:
    from database import SessionLocal, FolderSource

    db = SessionLocal()
    try:
        rows = db.query(FolderSource).filter(
            FolderSource.user_id == CURATED_USER_ID,
            FolderSource.folder_name == folder_name,
        ).all()
        out: list[dict] = []
        for s in rows:
            path = s.file_path
            if not path or not str(path).lower().endswith(".pdf"):
                continue
            if not Path(path).is_file():
                continue
            out.append({
                "path": path,
                "filename": s.filename or Path(path).name,
                "page_count": int(s.page_count or 0),
                "source_id": s.source_id,
            })
        return out
    finally:
        db.close()


def ensure_curated_oma(folder_name: str) -> bool:
    """Index curated PDFs into Content OMA (shared namespace user_id=0)."""
    import oma_provider

    if not oma_provider.is_oma_enabled():
        return True

    pdf_sources = _curated_pdf_sources(folder_name)
    if not pdf_sources:
        return False

    return oma_provider.ensure_oma_ready_for_outline(
        CURATED_USER_ID,
        folder_name,
        pdf_sources=pdf_sources,
    )


def bootstrap_curated_content(folder_name: str, *, force: bool = False) -> dict:
    """Build shared RAG + Content OMA once for a premade course (all users reuse it)."""
    import oma_provider

    if folder_name not in CURATED_FOLDER_NAMES:
        return {"error": "Not a premade lesson folder."}

    if not force and is_curated_content_ready(folder_name):
        pdf_sources = _curated_pdf_sources(folder_name)
        return {
            "ok": True,
            "folder": folder_name,
            "skipped": True,
            "content_ready": True,
            "sources": len(pdf_sources),
        }

    pdf_sources = _curated_pdf_sources(folder_name)
    if not pdf_sources:
        return {"error": f"No PDF sources found for {folder_name}."}

    rag_result: dict = {}
    try:
        import rag
        rag_result = rag.embed_all_in_folder(CURATED_USER_ID, folder_name) or {}
    except Exception:
        traceback.print_exc()

    oma_ready = ensure_curated_oma(folder_name)
    content_ready = oma_ready if oma_provider.is_oma_enabled() else True

    return {
        "ok": content_ready,
        "folder": folder_name,
        "skipped": False,
        "content_ready": content_ready,
        "oma_ready": oma_ready,
        "notebooks_embedded": rag_result.get("notebooks_embedded", 0),
        "total_chunks": rag_result.get("total_chunks", 0),
        "sources": len(pdf_sources),
    }


def bootstrap_all_curated_content(*, force: bool = False) -> dict[str, dict]:
    """Build shared content for every premade course. Run once on server startup."""
    ingest_curated_sources()
    results: dict[str, dict] = {}
    for folder_name in sorted(CURATED_FOLDER_NAMES):
        try:
            results[folder_name] = bootstrap_curated_content(folder_name, force=force)
            status = "ready" if results[folder_name].get("content_ready") else "pending"
            print(f"  [curated] bootstrap {folder_name}: {status}")
        except Exception:
            traceback.print_exc()
            results[folder_name] = {"error": "bootstrap failed"}
    return results


def prepare_curated_lesson(user_id: int, folder_name: str) -> dict:
    """Enroll a student in a premade lesson — instant when shared content is pre-built."""
    if folder_name not in CURATED_FOLDER_NAMES:
        return {"error": "Not a premade lesson folder."}

    shared_ready = is_curated_content_ready(folder_name)
    if not shared_ready:
        return {
            "error": "Premade course material is still being prepared. Try again in a minute.",
            "content_ready": False,
            "shared_content_ready": False,
            "has_outline": False,
        }

    outline_ready = ensure_curated_outline(user_id, folder_name)
    if not outline_ready:
        return {"error": "Could not create lesson outline."}

    return {
        "ok": True,
        "has_outline": True,
        "content_ready": True,
        "shared_content_ready": True,
        "oma_ready": True,
        "instant": True,
    }


def is_curated_content_ready(folder_name: str) -> bool:
    """Read-only check — true when RAG sources exist and Content OMA is indexed."""
    pdf_sources = _curated_pdf_sources(folder_name)
    if not pdf_sources:
        return False
    try:
        import oma_provider
        if not oma_provider.is_oma_enabled():
            return True
        from coast_content_oma.stores import make_namespace
        ns = make_namespace(CURATED_USER_ID, folder_name)
        orch = oma_provider._content_orchestrator()
        target = sum(max(1, int(s.get("page_count") or 0)) for s in pdf_sources)
        return orch.content.count(ns) >= target
    except Exception:
        return False


def ingest_curated_sources():
    """Scan curated_content/ directories and register any new PDFs/PPTX files.
    
    Runs on startup. Skips files already registered (matched by filename).
    Sources are stored under user_id=0 so they're shared with everyone.
    """
    if not CURATED_CONTENT_DIR.exists():
        return

    from database import SessionLocal, FolderSource

    db = SessionLocal()
    try:
        for folder_name in CURATED_FOLDER_NAMES:
            folder_path = CURATED_CONTENT_DIR / folder_name
            if not folder_path.is_dir():
                continue

            existing = {
                fs.filename
                for fs in db.query(FolderSource).filter(
                    FolderSource.user_id == CURATED_USER_ID,
                    FolderSource.folder_name == folder_name,
                ).all()
            }

            for file_path in sorted(folder_path.iterdir()):
                ext = file_path.suffix.lower()
                if ext not in (".pdf", ".pptx"):
                    continue
                if file_path.name in existing:
                    continue

                print(f"  [curated] Ingesting: {file_path.name} → {folder_name}")
                try:
                    raw_text, page_count = _extract_text(file_path, ext)
                    if not raw_text.strip():
                        print(f"  [curated] Skipped {file_path.name} — no text extracted")
                        continue

                    source_id = f"cur_{uuid.uuid4().hex[:10]}"
                    title = file_path.stem.replace("_", " ").replace("-", " ")

                    fs = FolderSource(
                        user_id=CURATED_USER_ID,
                        folder_name=folder_name,
                        source_id=source_id,
                        title=title,
                        filename=file_path.name,
                        source_type=ext.lstrip("."),
                        page_count=page_count,
                        raw_text=raw_text,
                        file_path=str(file_path),
                    )
                    db.add(fs)
                    db.commit()

                    try:
                        import rag
                        rag.embed_raw_source(CURATED_USER_ID, folder_name, source_id, title, raw_text)
                        print(f"  [curated] Embedded: {file_path.name}")
                    except Exception:
                        traceback.print_exc()

                    if ext == ".pdf":
                        try:
                            import oma_provider
                            if oma_provider.is_oma_enabled():
                                oma_provider.ingest_pdf_async(
                                    CURATED_USER_ID, folder_name, file_path,
                                )
                                print(f"  [curated] OMA ingest queued: {file_path.name}")
                        except Exception:
                            traceback.print_exc()

                    try:
                        from image_extractor import extract_and_store_images
                        extract_and_store_images(
                            file_path, ext, source_id, CURATED_USER_ID, folder_name
                        )
                    except Exception:
                        traceback.print_exc()

                except Exception:
                    print(f"  [curated] Failed to ingest {file_path.name}")
                    traceback.print_exc()
    finally:
        db.close()

    _extract_images_for_existing_sources()
    _ensure_oma_for_existing_sources()


def _ensure_oma_for_existing_sources():
    """Queue Content OMA ingest for curated PDFs not yet indexed."""
    import oma_provider

    if not oma_provider.is_oma_enabled():
        return

    for folder_name in CURATED_FOLDER_NAMES:
        pdf_sources = _curated_pdf_sources(folder_name)
        if not pdf_sources:
            continue
        try:
            ready = oma_provider.ensure_oma_ready_for_outline(
                CURATED_USER_ID,
                folder_name,
                pdf_sources=pdf_sources,
                wait_sec=0,
            )
            if not ready:
                for src in pdf_sources:
                    path = src.get("path")
                    if path and Path(path).is_file():
                        oma_provider.ingest_pdf_async(
                            CURATED_USER_ID, folder_name, path,
                        )
                print(f"  [curated] OMA background ingest queued for {folder_name}")
        except Exception:
            traceback.print_exc()


def _extract_images_for_existing_sources():
    """Extract images from curated sources that need (re-)extraction.

    Handles two cases:
    1. Sources with no SourceImage records at all
    2. Sources whose SourceImage records point to files that no longer exist
       (e.g. after a deploy moved storage to the persistent disk)
    """
    from database import SessionLocal, FolderSource, SourceImage

    db = SessionLocal()
    try:
        for folder_name in CURATED_FOLDER_NAMES:
            sources = db.query(FolderSource).filter(
                FolderSource.user_id == CURATED_USER_ID,
                FolderSource.folder_name == folder_name,
            ).all()

            for src in sources:
                existing_images = db.query(SourceImage).filter(
                    SourceImage.source_id == src.source_id,
                    SourceImage.user_id == CURATED_USER_ID,
                ).all()

                needs_extraction = False
                if not existing_images:
                    needs_extraction = True
                else:
                    any_exists = any(Path(si.image_path).exists() for si in existing_images)
                    if not any_exists:
                        print(f"  [curated] Deleting {len(existing_images)} orphaned image records for {src.title}")
                        for si in existing_images:
                            db.delete(si)
                        db.commit()
                        needs_extraction = True

                if not needs_extraction:
                    continue

                fpath = Path(src.file_path) if src.file_path else None
                if not fpath or not fpath.exists():
                    continue
                ext = fpath.suffix.lower()
                if ext not in (".pdf", ".pptx"):
                    continue
                print(f"  [curated] Extracting images: {src.title}")
                try:
                    from image_extractor import extract_and_store_images
                    extract_and_store_images(
                        fpath, ext, src.source_id, CURATED_USER_ID, folder_name
                    )
                except Exception:
                    traceback.print_exc()
    finally:
        db.close()


def _extract_text(file_path: Path, ext: str) -> tuple[str, int]:
    """Extract raw text and page count from a PDF or PPTX file."""
    if ext == ".pptx":
        from extractor import extract_content_from_pptx
        pages = extract_content_from_pptx(str(file_path))
        page_count = len(pages)
        raw_text = "\n\n".join(p.get("text", "") for p in pages if p.get("text"))
        return raw_text, page_count
    elif ext == ".pdf":
        from extractor import extract_content_from_pdf
        pages = extract_content_from_pdf(str(file_path))
        page_count = len(pages) if pages else 0
        raw_text = "\n\n".join(p.get("text", "") for p in (pages or []) if p.get("text"))
        return raw_text, page_count
    return "", 0
