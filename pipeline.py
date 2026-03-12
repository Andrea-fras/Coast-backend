"""Main OCR pipeline – ties processors + extractor + validation together.

Supports two modes:
  1. run_pipeline()           – Extract questions from past papers
  2. run_notebook_pipeline()  – Generate study guides from lecture notes
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

from extractor import (
    extract_exam_paper,
    extract_lecture_notes,
    extract_lecture_notes_hybrid,
    extract_content_from_pdf,
    extract_content_from_pptx,
    match_questions_to_notebook,
)
from processors import extract_diagram_regions, load_images_from_path
from schema import ExamPaper, MultipleChoiceQuestion, OpenEndedQuestion
from notebook_schema import Notebook


def run_pipeline(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    provider: str = "openai",
    api_key: str | None = None,
    model: str | None = None,
    extra_instructions: str = "",
    save_page_images: bool = False,
) -> dict[str, Any]:
    """
    End-to-end OCR pipeline.

    1. Load the input file (PDF or image) into page images.
    2. Send pages to the vision LLM for structured extraction.
    3. Validate the result against the Pydantic schema.
    4. Optionally save the JSON output and page images.

    Args:
        input_path: Path to the PDF or image file.
        output_path: Where to write the JSON result. If None, auto-generated.
        provider: LLM provider – 'openai' or 'anthropic'.
        api_key: API key (or set env var based on provider).
        model: Model name (default depends on provider).
        extra_instructions: Extra context for the LLM.
        save_page_images: If True, save each page as a PNG alongside the output.

    Returns:
        The validated exam paper as a dict.
    """
    input_path = Path(input_path)

    # ── Step 1: Load images ──────────────────────────────────────────────
    images = load_images_from_path(input_path)

    # ── Step 2: Determine output directory ───────────────────────────────
    if output_path is None:
        output_dir = input_path.parent / f"{input_path.stem}_output"
    else:
        output_path = Path(output_path)
        output_dir = output_path.parent if output_path.suffix == ".json" else output_path

    output_dir.mkdir(parents=True, exist_ok=True)
    json_file = (
        Path(output_path) if output_path and str(output_path).endswith(".json")
        else output_dir / f"{input_path.stem}.json"
    )

    # ── Step 3: Optionally save page images ──────────────────────────────
    if save_page_images:
        images_dir = output_dir / "pages"
        images_dir.mkdir(exist_ok=True)
        for i, img in enumerate(images):
            img.save(images_dir / f"page_{i + 1}.png", "PNG")

    # ── Step 4: Extract via vision LLM ───────────────────────────────────
    raw_result = extract_exam_paper(
        images,
        provider=provider,
        api_key=api_key,
        model=model,
        extra_instructions=extra_instructions,
    )

    # ── Step 5: Validate with Pydantic ───────────────────────────────────
    validated = _validate_and_normalize(raw_result)

    # ── Step 6: Save JSON ────────────────────────────────────────────────
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(validated, f, indent=2, ensure_ascii=False)

    return validated


def _validate_and_normalize(data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate raw LLM output against the Pydantic schema, normalizing where
    needed so minor LLM deviations don't cause hard failures.
    """
    # Normalize questions list – the LLM sometimes omits optional fields
    questions = data.get("questions", [])
    normalized_questions = []

    for q in questions:
        qtype = q.get("type", "open-ended")

        # Ensure required fields have defaults
        q.setdefault("equation", None)
        q.setdefault("images", None)

        if qtype == "multiple-choice":
            q.setdefault("correctAnswerId", None)
            q.setdefault("options", [])
            # Normalize option ids to lowercase
            for opt in q.get("options", []):
                if isinstance(opt.get("id"), str):
                    opt["id"] = opt["id"].lower().strip()
            if q.get("correctAnswerId"):
                q["correctAnswerId"] = q["correctAnswerId"].lower().strip()
        else:
            q["type"] = "open-ended"
            q.setdefault("modelAnswer", None)
            q.setdefault("keyTerms", None)

        normalized_questions.append(q)

    data["questions"] = normalized_questions

    # Validate via Pydantic (will raise on truly invalid data)
    try:
        paper = ExamPaper.model_validate(data)
        return paper.model_dump(mode="json")
    except Exception:
        # If strict validation fails, return the normalized dict anyway
        # so the user still gets usable output
        return data


def run_pipeline_multi(
    input_paths: list[str | Path],
    output_dir: str | Path,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Run the pipeline on multiple files, saving all results into one directory.

    Args:
        input_paths: List of file paths (PDF or image).
        output_dir: Directory to save all JSON outputs.
        **kwargs: Additional arguments forwarded to run_pipeline.

    Returns:
        List of validated exam paper dicts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for path in input_paths:
        path = Path(path)
        out_json = output_dir / f"{path.stem}.json"
        result = run_pipeline(path, output_path=out_json, **kwargs)
        results.append(result)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# NOTEBOOK / STUDY GUIDE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_notebook_pipeline(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    paper_paths: list[str | Path] | None = None,
    provider: str = "openai",
    api_key: str | None = None,
    model: str | None = None,
    extra_instructions: str = "",
    detail: str = "low",
    save_page_images: bool = False,
    on_progress: Any = None,
) -> dict[str, Any]:
    """
    End-to-end notebook generation pipeline with chunked parallel processing.

    1. Load the input file (PDF or image of lecture slides) into page images.
    2. Send pages to the vision LLM (chunked + parallel for large decks).
    3. Validate the result against the Notebook Pydantic schema.
    4. If past paper JSONs are provided, match questions to notebook topics.
    5. Save the final notebook JSON.

    Args:
        input_path: Path to the PDF or image file (lecture slides/notes).
        output_path: Where to write the JSON result. If None, auto-generated.
        paper_paths: Optional list of paths to past paper JSON files for question matching.
        provider: LLM provider – 'openai', 'anthropic', or 'kimi'.
        api_key: API key.
        model: Model name.
        extra_instructions: Extra context for the LLM.
        detail: 'low' (default, good for slides) or 'high' (for dense handwriting).
        save_page_images: If True, save each page as a PNG alongside the output.
        on_progress: Optional callback(stage, current, total) for progress updates.

    Returns:
        The validated notebook as a dict.
    """
    import time as _time

    input_path = Path(input_path)
    ext = input_path.suffix.lower()
    t_start = _time.time()

    def _log(msg: str):
        elapsed = _time.time() - t_start
        print(f"  [{elapsed:6.1f}s] {msg}")

    _log(f"Pipeline start — {input_path.name} ({ext})")

    # ── Step 1: Determine output directory ───────────────────────────────
    if output_path is None:
        output_dir = input_path.parent / f"{input_path.stem}_notebook"
    else:
        output_path = Path(output_path)
        output_dir = output_path.parent if output_path.suffix == ".json" else output_path

    output_dir.mkdir(parents=True, exist_ok=True)
    json_file = (
        Path(output_path) if output_path and str(output_path).endswith(".json")
        else output_dir / f"{input_path.stem}_notebook.json"
    )

    # ── Step 2: Try hybrid extraction (text + embedded images) ───────────
    slides_data = None

    if ext == ".pptx":
        if on_progress:
            on_progress("extracting", 0, 1)
        slides_data = extract_content_from_pptx(str(input_path))
        total_imgs = sum(len(s["images"]) for s in slides_data)
        _log(f"PPTX extracted — {len(slides_data)} slides, {total_imgs} images → HYBRID mode")
        if on_progress:
            on_progress("loaded", len(slides_data), len(slides_data))

    elif ext == ".pdf":
        if on_progress:
            on_progress("extracting", 0, 1)
        t_extract = _time.time()
        slides_data = extract_content_from_pdf(str(input_path))
        _log(f"PDF text extraction took {_time.time() - t_extract:.1f}s")
        if slides_data:
            total_imgs = sum(len(s["images"]) for s in slides_data)
            _log(f"PDF hybrid — {len(slides_data)} pages, {total_imgs} embedded images → HYBRID mode")
            if on_progress:
                on_progress("loaded", len(slides_data), len(slides_data))
        else:
            _log("PDF has no text layer → VISION fallback mode")

    else:
        _log(f"Image file → VISION mode")

    # ── Step 3: Generate notebook ────────────────────────────────────────
    t_gen = _time.time()
    if slides_data is not None:
        raw_notebook = extract_lecture_notes_hybrid(
            slides_data,
            provider=provider,
            api_key=api_key,
            model=model,
            extra_instructions=extra_instructions,
            on_progress=on_progress,
        )
        _log(f"Hybrid LLM generation took {_time.time() - t_gen:.1f}s")
    else:
        t_img = _time.time()
        images = load_images_from_path(input_path)
        _log(f"Image loading took {_time.time() - t_img:.1f}s — {len(images)} pages")
        if on_progress:
            on_progress("loaded", len(images), len(images))

        if save_page_images:
            images_dir = output_dir / "pages"
            images_dir.mkdir(exist_ok=True)
            for i, img in enumerate(images):
                img.save(images_dir / f"page_{i + 1}.png", "PNG")

        raw_notebook = extract_lecture_notes(
            images,
            provider=provider,
            api_key=api_key,
            model=model,
            extra_instructions=extra_instructions,
            detail=detail,
            on_progress=on_progress,
        )
        _log(f"Vision LLM generation took {_time.time() - t_gen:.1f}s")

    # ── Step 4: Validate with Pydantic ───────────────────────────────────
    validated = _validate_notebook(raw_notebook)
    _log("Validation complete")

    # ── Step 5: Match past paper questions (if papers provided) ──────────
    matched_questions = []
    if paper_paths:
        if on_progress:
            on_progress("matching", 0, 1)
        t_match = _time.time()
        papers = _load_papers(paper_paths)
        matched_ids = match_questions_to_notebook(
            validated,
            papers,
            provider=provider,
            api_key=api_key,
            model=model,
        )
        matched_questions = _collect_matched_questions(papers, matched_ids)
        _log(f"Question matching took {_time.time() - t_match:.1f}s — {len(matched_questions)} matched")
        if on_progress:
            on_progress("matching", 1, 1)
        validated["matchedQuestions"] = matched_questions
        validated["questionCount"] = len(matched_questions)

        # Set paperId to the first paper that has matched questions
        if matched_questions and not validated.get("paperId"):
            # Find which paper contributed the most matches
            paper_counts: dict[str, int] = {}
            for mq in matched_questions:
                pid = mq.get("_paperId", "")
                paper_counts[pid] = paper_counts.get(pid, 0) + 1
            if paper_counts:
                validated["paperId"] = max(paper_counts, key=paper_counts.get)
    else:
        validated["matchedQuestions"] = []
        validated["questionCount"] = 0

    # Clean internal fields
    for mq in validated.get("matchedQuestions", []):
        mq.pop("_paperId", None)

    # Convert chatResponses from list to dict format for frontend compatibility
    if isinstance(validated.get("chatResponses"), list):
        chat_dict = {}
        for cr in validated["chatResponses"]:
            if isinstance(cr, dict):
                chat_dict[cr.get("keywords", "")] = cr.get("response", "")
        validated["chatResponses"] = chat_dict

    # ── Step 6: Save JSON ────────────────────────────────────────────────
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(validated, f, indent=2, ensure_ascii=False)

    sections = validated.get("sections", [])
    _log(f"DONE — {len(sections)} sections, saved to {json_file.name}")
    _log(f"Total pipeline time: {_time.time() - t_start:.1f}s")
    print()

    return validated


def _validate_notebook(data: dict[str, Any]) -> dict[str, Any]:
    """Validate raw LLM output against the Notebook Pydantic schema."""
    # Ensure chatResponses is a list (the schema expects it)
    cr = data.get("chatResponses", [])
    if isinstance(cr, dict):
        # Convert dict format to list format for validation
        data["chatResponses"] = [
            {"keywords": k, "response": v} for k, v in cr.items()
        ]

    try:
        notebook = Notebook.model_validate(data)
        return notebook.model_dump(mode="json")
    except Exception:
        # If strict validation fails, return normalized dict
        return data


def _load_papers(paper_paths: list[str | Path]) -> list[dict[str, Any]]:
    """Load past paper JSON files."""
    papers = []
    for p in paper_paths:
        p = Path(p)
        if p.exists() and p.suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                papers.append(json.load(f))
    return papers


def _collect_matched_questions(
    papers: list[dict[str, Any]],
    matched_ids: list[str],
) -> list[dict[str, Any]]:
    """Collect full question objects for the matched IDs."""
    # Build a lookup: compound_id → question dict
    question_lookup: dict[str, dict[str, Any]] = {}
    for paper in papers:
        paper_id = paper.get("id", "")
        for q in paper.get("questions", []):
            compound_id = f"{paper_id}_{q['id']}"
            question_lookup[compound_id] = {**q, "_paperId": paper_id}
            # Also index by plain ID for fuzzy matching
            question_lookup[q["id"]] = {**q, "_paperId": paper_id}

    matched = []
    seen = set()
    for mid in matched_ids:
        q = question_lookup.get(mid)
        if q and q["id"] not in seen:
            matched.append(q)
            seen.add(q["id"])

    return matched
