"""Curated lesson folder configuration — shared across server, tutor, lesson."""

from __future__ import annotations

import os
import uuid
import traceback
from pathlib import Path

CURATED_FOLDER_NAMES = {"Quantitative Methods 1", "Data Structures & Algorithms"}
CURATED_CONTENT_DIR = Path(__file__).parent / "curated_content"
CURATED_USER_ID = 0

FOLDER_TO_COURSE: dict[str, str] = {
    "Quantitative Methods 1": "QM1",
    "Data Structures & Algorithms": "DSA",
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
}


def curated_source_uid(folder_name: str) -> int | None:
    """Return shared user_id (0) if folder is curated, else None."""
    return CURATED_USER_ID if folder_name in CURATED_FOLDER_NAMES else None


def get_lesson_structure(folder_name: str) -> dict | None:
    """Return the custom structure hints for a curated lesson, or None."""
    return CURATED_LESSON_STRUCTURES.get(folder_name)


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
