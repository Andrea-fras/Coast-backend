"""Ingest course material into the Content OMA stores.

World-class pipeline (Phase 1 — roadmap unlock):
  1. Extract pages + save images to disk instantly.
  2. IN PARALLEL: batched page classification + priority-page vision only.
  3. Bulk-write content_items + image_items (deferred figures as pending stubs).
  4. Folder-level concept pass (Tier A: embedding clusters only → unlock roadmap).
  5. Background Tier B: LLM definitions + prerequisite graph (upgrades same nodes).

Only figures on the first ~OMA_PRIORITY_PAGES pages per PDF block roadmap
generation. Remaining figures are stored with ``_pending_vision`` and
described in a background pass (section-ordered after outline exists).
"""

from __future__ import annotations

import time

import hashlib
import json
import logging
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from . import llm

try:  # pool threads keep the request's AI-cost attribution (student, feature)
    from ai_usage import carry as _carry
except ImportError:  # outside the server
    def _carry(fn):
        return fn
from .concept_identity import effective_merge_threshold
from .extraction import extract_pages
from .stores import (
    ConceptStore,
    ContentStore,
    ImageStore,
    MemoryItem,
    new_item_id,
)

logger = logging.getLogger(__name__)


CONTENT_CLASSIFY_SYSTEM = (
    "You are analysing pages of lecture material. For each page you "
    "identify what kinds of content it contains and which academic "
    "concepts it discusses. List concepts that are clearly the subject "
    "of the page (not every word mentioned)."
)


# Batched prompt — classifies many pages in a single LLM call.
BATCH_CLASSIFY_PROMPT_TEMPLATE = """Analyse these pages from a single lecture.

PAGES (JSON array, each with page_number and text):
{pages_json}

Respond ONLY with a JSON array — one object per page, in the same order:
[
  {{
    "page_number": <int>,
    "content_types": ["<one or more of: definition, theorem, proof, example, worked_example, exercise, narrative, figure_caption, summary, remark>"],
    "concepts": ["<concept name>", ...],
    "section_title": "<best guess at the section/topic this page covers>",
    "summary": "<one sentence summary of the page>"
  }},
  ...
]

Rules:
- Output ONLY the JSON array, no commentary.
- One entry per input page, in the same order, even for blank/structural pages.
- For empty / purely structural pages (TOC, references, blanks): content_types=["narrative"], concepts=[].
- Use canonical names for concepts where possible.
- Keep concepts focused — at most 5 per page.
"""


CONCEPT_CANONICALIZE_SYSTEM = (
    "You are organising the concept inventory for a course. Given a list "
    "of raw concept mentions extracted from lectures, your job is to "
    "deduplicate, canonicalize, and infer the prerequisite structure. "
    "You must preserve EVERY input concept — never silently drop one. "
    "If two inputs are aliases for the same concept, merge them as aliases "
    "under one canonical name. If they're distinct concepts, keep both."
)


CONCEPT_CLUSTER_CANONICALIZE_SYSTEM = (
    "You are organising the concept inventory for a course. Each cluster "
    "groups surface forms that refer to the same (or closely related) idea. "
    "Produce one canonical concept per cluster. Preserve every input term as "
    "either the canonical name or an alias. Prefer distinct concepts when in doubt."
)

CONCEPT_CLUSTER_CANONICALIZE_PROMPT_TEMPLATE = """Here are clusters of related concept mentions from lecture material:

{clusters_json}

For EACH cluster, output one canonical concept. Every input term must appear as the canonical name or in aliases.

Respond ONLY with a JSON array:
[
  {{
    "cluster_id": <int matching input>,
    "name": "<canonical name, lowercase>",
    "aliases": ["<other terms in this cluster>"],
    "definition": "<one short paragraph>",
    "prerequisites": ["<canonical name of prerequisite from ANY cluster>", ...],
    "related": ["<canonical name of related concept>", ...]
  }},
  ...
]

Rules:
- Output ONLY the JSON array.
- One object per input cluster.
- Use names from the clusters when referencing prerequisites/related.
- Keep definitions short (1-2 sentences).
- Do not invent concepts not present in the clusters.
"""


CROSS_CLUSTER_PREREQ_SYSTEM = (
    "You identify prerequisite relationships between academic concepts "
    "in a course. Only list strict prerequisites (must understand A before B)."
)

CROSS_CLUSTER_PREREQ_PROMPT_TEMPLATE = """These canonical concepts were extracted from a course:

{concepts_json}

Identify prerequisite pairs where understanding one concept is required before another.
Respond ONLY with a JSON array of objects:
[{{"concept": "<name>", "prerequisites": ["<prereq name>", ...]}}, ...]

Only include pairs you are confident about. Every name must be from the list above.
"""


CONCEPT_CANONICALIZE_PROMPT_TEMPLATE = """Here are concept mentions extracted from the lectures of a course:

{mentions_json}

Produce a clean concept inventory. Every single input mention must end up either as its own canonical concept OR as an alias of another canonical concept. Do NOT drop any input. For each canonical concept, identify which OTHER concepts in this list are prerequisites (must be understood first) and which are related (co-occur but not strict prereqs).

Respond ONLY with a JSON array of objects:
[
  {{
    "name": "<canonical concept name, lowercase>",
    "aliases": ["<other names found in the mentions that mean the same thing>"],
    "definition": "<one short paragraph defining the concept>",
    "prerequisites": ["<canonical name of prerequisite>", ...],
    "related": ["<canonical name of related concept>", ...]
  }},
  ...
]

Rules:
- Output ONLY the JSON array, no commentary.
- EVERY input concept must appear either as a canonical name or in someone's aliases.
- Use canonical names from the input list when referencing prerequisites and related.
- A concept can have zero prerequisites if it's foundational.
- Keep definitions short (1-2 sentences).
- Don't invent concepts not in the input list.
- Prefer keeping concepts distinct when in doubt — only merge if they truly refer to the same thing.
"""


@dataclass
class IngestStats:
    docs: int = 0
    pages: int = 0
    content_items: int = 0
    image_items: int = 0
    concepts: int = 0
    figures_pending: int = 0  # left for the background vision sweep; never fails a source
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "figures_pending": self.figures_pending,
            "docs": self.docs,
            "pages": self.pages,
            "content_items": self.content_items,
            "image_items": self.image_items,
            "concepts": self.concepts,
            "errors": self.errors,
        }


class IngestionPipeline:
    BATCH_SIZE = 8
    VISION_ATTEMPTS = 3
    CLUSTER_THRESHOLD = float(os.environ.get("OMA_CLUSTER_THRESHOLD", "0.85"))
    CONCEPT_MERGE_THRESHOLD = float(os.environ.get("OMA_CONCEPT_MERGE_THRESHOLD", "0.90"))
    VISION_BATCH_SIZE = int(os.environ.get("OMA_VISION_BATCH_SIZE", "8"))
    # Figures a file describes before it counts as read; the rest go to the background sweep, so a
    # photo-heavy deck (150 figures) doesn't hold up the next file, or the course, for minutes.
    INLINE_FIGURES = int(os.environ.get("OMA_INLINE_FIGURES", "24"))
    PRIORITY_PAGES = int(os.environ.get("OMA_PRIORITY_PAGES", "12"))

    def __init__(
        self,
        concept_store: ConceptStore,
        content_store: ContentStore,
        image_store: ImageStore,
        image_save_dir: Path,
        max_workers: int = 4,
        describe_images: bool = True,
        skip_images: bool = False,
        classify_workers: int | None = None,
        vision_workers: int | None = None,
    ):
        self.concept = concept_store
        self.content = content_store
        self.images = image_store
        self.image_save_dir = Path(image_save_dir)
        self.image_save_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.describe_images = describe_images
        self.skip_images = skip_images
        self.classify_workers = classify_workers or int(
            os.environ.get("OMA_CLASSIFY_WORKERS", str(min(6, max_workers)))
        )
        self.vision_workers = vision_workers or int(
            os.environ.get("OMA_VISION_WORKERS", str(min(3, max_workers)))
        )

    # ── Top-level ─────────────────────────────────────────────────

    def ingest_folder(
        self,
        namespace: str,
        pdf_paths: list[Path],
        progress: Optional[Callable[[str], None]] = None,
        on_timing: Optional[Callable[[str, dict], None]] = None,
        source_ids: Optional[dict[str, str]] = None,
        defer_concepts: bool = False,
    ) -> IngestStats:
        """Ingest PDFs into namespace. source_ids maps str(pdf_path) → FolderSource.source_id.

        When ``defer_concepts`` is True (per-PDF upload ingest), raw mentions are stored on
        items but canonicalization waits for a folder-level pass (see ``run_folder_concept_pass``).
        """
        log = progress or (lambda msg: logger.info(msg))
        tmark = on_timing or (lambda phase, meta: None)
        stats = IngestStats()
        all_concept_mentions: dict[str, set[str]] = {}
        source_ids = source_ids or {}

        for pdf in pdf_paths:
            try:
                sid = source_ids.get(str(pdf)) or source_ids.get(pdf.name)
                log(f"Extracting {pdf.name}")
                pages = extract_pages(pdf, extract_images=not self.skip_images)
                stats.docs += 1
                stats.pages += len(pages)
                source_doc_id = self._doc_id(pdf, source_id=sid)
                log(f"  → {len(pages)} pages extracted")

                saved_images = (
                    self._save_images_to_disk(pages, source_doc_id)
                    if not self.skip_images else []
                )

                from . import progressive
                progressive.register_pages(self.content.db_path, namespace, source_doc_id, pages, saved_images)
                from .stores.db import connect_db
                with connect_db(self.content.db_path) as conn:
                    completed_pages = {r[0] for r in conn.execute(
                        'SELECT page FROM source_page_progress WHERE namespace=? AND source_id=? AND text_ready=1', (namespace,source_doc_id))}
                pages_to_classify = [p for p in pages if p['page_number'] not in completed_pages]
                stats.content_items += sum(bool(p.get('text','').strip()) for p in pages if p['page_number'] in completed_pages)
                content_items_by_page = {}
                def _on_classify_batch(batch_pairs):
                    batch_pages = [p for p, _ in batch_pairs]
                    partial, n = self._write_content_items_for_pdf(
                        namespace, pdf, batch_pages, [r for _, r in batch_pairs],
                        source_doc_id, all_concept_mentions, log)
                    content_items_by_page.update(partial)
                    stats.content_items += n
                    progressive.mark_text(self.content.db_path, namespace, source_doc_id, batch_pages)

                def _on_vision_batch(results):
                    self._write_image_results_for_pdf(namespace, pdf, results, source_doc_id,
                        content_items_by_page, all_concept_mentions, log, stats)

                images_before = stats.image_items
                placeholders = [{**s, "description": "", "image_type": "figure", "concepts": [],
                    "_pending_vision": True, "vision_tier": "background"} for s in saved_images]
                _on_vision_batch(placeholders)
                # Empty/very short pages require no classification call, but still get a durable receipt.
                short_pages = [(p, None) for p in pages_to_classify if len((p.get('text') or '').strip()) < 30]
                if short_pages: _on_classify_batch(short_pages)
                tmark("extract", {"pdf": pdf.name, "pages": len(pages), "figures": len(saved_images)})
                priority = lambda page: progressive.page_rank(namespace, source_doc_id, page)
                pending_paths = {(it.store_specific or {}).get('file_path') for it in self.images.all(namespace) if it.source_doc_id == source_doc_id and self._image_describe_pending(it)}
                pending_images = [image for image in saved_images if image['file_path'] in pending_paths]
                pending_images.sort(key=lambda image: priority(image['page_number']))  # the student's section first
                stats.figures_pending += max(0, len(pending_images) - self.INLINE_FIGURES)
                pending_images = pending_images[:self.INLINE_FIGURES]
                describe = bool(self.describe_images and pending_images and not self.skip_images)
                tmark("classify_start", {"pdf": pdf.name, "pages": len(pages)})
                # Text is what teaching needs; undescribed figures are retried in the background
                # and can never block the source (or its sections) for good.
                def _on_figures(rows):
                    stats.figures_pending += sum(1 for row in rows if row.get("_pending_vision"))
                    _on_vision_batch(rows)
                todo_pages = [p for p in pages_to_classify if len((p.get('text') or '').strip()) >= 30]
                todo_images = pending_images if describe else []
                from . import remote
                if remote.enabled() and (todo_pages or todo_images):
                    # The AI work in a container; each batch is applied here as it arrives.
                    applied = {"pages": set(), "figures": set(), "left": set()}
                    try:
                        applied = remote.index(todo_pages, todo_images, priority, _on_classify_batch, _on_figures,
                                               self._remote_settings())
                    except Exception:
                        logger.exception("container indexing failed for %s; finishing it here", pdf.name)
                    todo_pages = [p for p in todo_pages if p["page_number"] not in applied["pages"]]
                    stats.figures_pending += len(applied["left"])
                    todo_images = [i for i in todo_images if i["file_path"] not in applied["figures"] | applied["left"]]
                if todo_pages or todo_images:  # here: without containers, or what a container left
                    # Publish each finished batch; the first section no longer waits for the whole PDF.
                    with ThreadPoolExecutor(max_workers=2) as phase_ex:
                        classify = phase_ex.submit(_carry(self._classify_pages_parallel), todo_pages, _on_classify_batch, priority)
                        vision = (phase_ex.submit(_carry(self._describe_saved_images_batched), todo_images, _on_vision_batch, priority)
                                  if todo_images else None)
                        classify.result()
                        if vision:  # published batch by batch above; count what is left for the background
                            stats.figures_pending += sum(1 for row in vision.result() if row.get("_pending_vision"))
                tmark("classify_done", {"pdf": pdf.name, "pages": len(pages)})
                if describe:
                    tmark("vision_done", {"pdf": pdf.name, "figures": len(saved_images)})
                stats.image_items = images_before + len(saved_images)
                tmark("pdf_done", {
                    "pdf": pdf.name,
                    "content_items": stats.content_items,
                    "image_items": stats.image_items,
                })

            except Exception as e:
                logger.exception(f"failed to ingest {pdf}")
                stats.errors.append(f"{pdf.name}: {e}")
                continue

        if defer_concepts:
            tmark("concepts_deferred", {"mentions": len(all_concept_mentions)})
            log(
                f"Deferred folder concept pass ({len(all_concept_mentions)} mentions "
                f"stored on items)"
            )
        else:
            result = self.run_folder_concept_pass(
                namespace,
                tier="B",
                mentions=all_concept_mentions,
                progress=log,
                on_timing=tmark,
            )
            stats.concepts = result.get("concepts_new", 0)

        return stats

    @staticmethod
    def collect_concept_mentions(namespace: str, *, content_store, image_store) -> dict[str, set[str]]:
        """Gather raw concept mentions from indexed content/image items."""
        mentions: dict[str, set[str]] = {}
        for it in content_store.all(namespace):
            for c in (it.store_specific or {}).get("concept_mentions_raw") or []:
                norm = llm.normalize_concept_name(c)
                if norm:
                    mentions.setdefault(norm, set()).add(c)
        for it in image_store.all(namespace):
            for c in (it.store_specific or {}).get("concept_mentions_raw") or []:
                norm = llm.normalize_concept_name(c)
                if norm:
                    mentions.setdefault(norm, set()).add(c)
        return mentions

    def run_folder_concept_pass(
        self,
        namespace: str,
        *,
        tier: str = "A",
        mentions: dict[str, set[str]] | None = None,
        progress: Optional[Callable[[str], None]] = None,
        on_timing: Optional[Callable[[str, dict], None]] = None,
    ) -> dict:
        """Folder-level concept canonicalization (Tier A fast or Tier B full LLM)."""
        log = progress or (lambda msg: logger.info(msg))
        tmark = on_timing or (lambda phase, meta: None)
        if mentions is None:
            mentions = self.collect_concept_mentions(
                namespace, content_store=self.content, image_store=self.images,
            )

        tmark("concepts_start", {"mentions": len(mentions), "tier": tier})
        if tier.upper() == "A":
            log(f"Tier A: canonicalizing {len(mentions)} mentions (embedding clusters only)")
            canonical = self._canonicalize_concepts_fast(mentions)
        else:
            log(f"Tier B: canonicalizing {len(mentions)} mentions (LLM definitions + prereqs)")
            canonical = self._canonicalize_concepts_clustered(mentions)
        tmark("concepts_clustered", {"tier": tier, "concepts": len(canonical)})

        name_to_id, n_new, n_merged = self._write_concepts_with_dedup(
            namespace, canonical, log,
        )
        tmark("concepts_written", {"tier": tier, "concepts_new": n_new, "concepts_merged": n_merged})
        log(f"Concepts: {n_new} new, {n_merged} merged into existing")

        log("Backfilling canonical concept ids on content/image items")
        self._backfill_canonical_entities(namespace, name_to_id)
        tmark("concepts_done", {
            "tier": tier,
            "concepts_new": n_new,
            "concepts_merged": n_merged,
        })
        return {
            "mentions": len(mentions),
            "concepts": len(canonical),
            "concepts_new": n_new,
            "concepts_merged": n_merged,
        }

    # ── Re-canonicalize without re-extracting ─────────────────────

    def recanonicalize(
        self,
        namespace: str,
        progress: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Re-run the concept canonicalization pass using `concept_mentions_raw`
        stored on each existing content/image item. Useful when the first
        canonicalization failed (e.g. rate limits) or to refresh with a
        better LLM."""
        log = progress or (lambda msg: logger.info(msg))
        mentions = self.collect_concept_mentions(
            namespace, content_store=self.content, image_store=self.images,
        )

        log(f"Re-canonicalizing {len(mentions)} concept mentions")

        old_concepts = self.concept.all(namespace)
        from ..student.concept_remap import build_concept_id_map

        # Wipe existing concept store entries for this namespace.
        wiped = self.concept.delete_namespace(namespace)
        log(f"Cleared {wiped} previous concept entries")

        canonical = self._canonicalize_concepts_clustered(mentions)
        log(f"LLM returned {len(canonical)} canonical concepts")

        name_to_id, n_new, n_merged = self._write_concepts_with_dedup(
            namespace, canonical, log,
        )
        id_map = build_concept_id_map(old_concepts, name_to_id)
        if id_map:
            log(f"Built concept id remap for {len(id_map)} student references")

        log("Backfilling canonical concept ids on content/image items")
        self._backfill_canonical_entities(namespace, name_to_id)
        return {
            "mentions": len(mentions),
            "concepts": len(canonical),
            "concepts_new": n_new,
            "id_map": id_map,
        }

    # ── Content item write (exposed for incremental progress) ───────

    @staticmethod
    def _split_pages_by_priority(pages: list[dict], limit: int | None = None) -> tuple[list[dict], list[dict]]:
        cap = limit if limit is not None else IngestionPipeline.PRIORITY_PAGES
        priority = [p for p in pages if int(p["page_number"]) <= cap]
        rest = [p for p in pages if int(p["page_number"]) > cap]
        return priority, rest

    def _write_content_items_for_pdf(
        self,
        namespace: str,
        pdf: Path,
        pages: list[dict],
        page_results: list,
        source_doc_id: str,
        all_concept_mentions: dict[str, set[str]],
        log: Callable[[str], None],
    ) -> tuple[dict[int, str], int]:
        """Bulk-write classified pages; returns page→item_id map and count."""
        content_items_by_page: dict[int, str] = {}
        content_items_to_write: list[MemoryItem] = []
        image_ids_by_page = {}
        for image in self.images.all(namespace):
            if image.source_doc_id == source_doc_id:
                image_ids_by_page.setdefault(image.store_specific.get('page_number'), []).append(image.id)
        for page, result in zip(pages, page_results):
            page_text = page["text"]
            if not page_text.strip():
                continue
            if not result:
                result = {
                    "content_types": ["unclassified"],
                    "concepts": [],
                    "section_title": "",
                    "summary": "",
                }
            ct = result.get("content_types") or ["narrative"]
            concepts = result.get("concepts") or []
            section_title = result.get("section_title") or ""
            summary = result.get("summary") or ""
            for c in concepts:
                norm = llm.normalize_concept_name(c)
                if norm:
                    all_concept_mentions.setdefault(norm, set()).add(c)

            item = MemoryItem(
                id="con_" + hashlib.sha256(f"{namespace}:{source_doc_id}:page:{page['page_number']}".encode()).hexdigest(),
                namespace=namespace,
                store="content",
                content=page_text,
                source_doc_id=source_doc_id,
                tags=list(ct),
                entities=[llm.normalize_concept_name(c) for c in concepts if c],
                importance=0.6,
                store_specific={
                    "content_types": list(ct),
                    "concept_mentions_raw": concepts,
                    "section_title": section_title,
                    "summary": summary,
                    "page_number": page["page_number"],
                    "source_filename": pdf.name,
                    "image_ids": image_ids_by_page.get(page["page_number"], []),
                },
            )
            content_items_to_write.append(item)
            content_items_by_page[page["page_number"]] = item.id

        if content_items_to_write:
            log(f"  → bulk-embedding {len(content_items_to_write)} content items")
            self.content.write_items_bulk(content_items_to_write)
        return content_items_by_page, len(content_items_to_write)

    def _write_image_results_for_pdf(
        self,
        namespace: str,
        pdf: Path,
        image_results: list[dict],
        source_doc_id: str,
        content_items_by_page: dict[int, str],
        all_concept_mentions: dict[str, set[str]],
        log: Callable[[str], None],
        stats: IngestStats,
    ) -> int:
        """Bulk-write image items and cross-link to content pages when available."""
        image_items_to_write: list[MemoryItem] = []
        image_to_page: dict[str, int] = {}
        for img_result in image_results:
            if not img_result:
                continue
            page_num = img_result["page_number"]
            desc = img_result.get("description", "")
            img_type = img_result.get("image_type", "figure")
            img_concepts = img_result.get("concepts", []) or []
            file_path = img_result["file_path"]
            width = img_result["width"]
            height = img_result["height"]

            for c in img_concepts:
                norm = llm.normalize_concept_name(c)
                if norm:
                    all_concept_mentions.setdefault(norm, set()).add(c)

            store_specific: dict[str, Any] = {
                "image_type": img_type,
                "concept_mentions_raw": img_concepts,
                "page_number": page_num,
                "source_filename": pdf.name,
                "file_path": file_path,
                "width": width,
                "height": height,
            }
            if img_result.get("also_on_pages"):
                store_specific["also_on_pages"] = img_result["also_on_pages"]  # the same figure on later slides
            if img_result.get("bbox"):
                store_specific["bbox"] = img_result["bbox"]  # position on the page, as fractions
            if img_result.get("_pending_vision"):
                store_specific["_pending_vision"] = True
            if img_result.get("vision_tier"):
                store_specific["vision_tier"] = img_result["vision_tier"]

            item = MemoryItem(
                id="ima_" + hashlib.sha256(f"{namespace}:{source_doc_id}:{Path(file_path).name}".encode()).hexdigest(),
                namespace=namespace,
                store="image",
                content=desc or "(no description)",
                source_doc_id=source_doc_id,
                tags=[img_type],
                entities=[llm.normalize_concept_name(c) for c in img_concepts if c],
                importance=0.4 if img_type == "decorative" else 0.6,
                store_specific=store_specific,
            )
            existing = self.images.get(item.id) if store_specific.get('_pending_vision') else None
            if existing and not self._image_describe_pending(existing):
                continue
            image_items_to_write.append(item)
            image_to_page[item.id] = page_num
            stats.image_items += 1

        if not image_items_to_write:
            return 0

        described = [it for it in image_items_to_write if not it.store_specific.get("_pending_vision")]
        pending = [it for it in image_items_to_write if it.store_specific.get("_pending_vision")]
        self.images.write_items_bulk(described)
        self.images.write_items_bulk(pending, embed=False)
        by_page: dict[int, list[str]] = {}
        for img in image_items_to_write:
            by_page.setdefault(image_to_page[img.id], []).append(img.id)
        for page_num, img_ids in by_page.items():
            content_id = content_items_by_page.get(page_num)
            if content_id:
                self.content.update_store_specific(content_id, {"image_ids": img_ids})
        return len(image_items_to_write)

    # ── Page classification ───────────────────────────────────────

    def _classify_pages_parallel(
        self,
        pages: list[dict],
        on_batch: Optional[Callable[[list[tuple[dict, Optional[dict]]]], None]] = None,
        priority=None,
    ) -> list[Optional[dict]]:
        """Classify all pages of a lecture in batches. Each batch is ONE
        LLM call that returns an array of per-page results — far fewer
        calls than per-page classification, runs in well under a minute."""
        results: list[Optional[dict]] = [None] * len(pages)

        # Skip the LLM for clearly empty pages.
        batches: list[tuple[list[int], list[dict]]] = []
        current_indices: list[int] = []
        current_pages: list[dict] = []
        for i, page in enumerate(pages):
            text = (page.get("text") or "").strip()
            if len(text) < 30:
                results[i] = {
                    "content_types": ["narrative"],
                    "concepts": [],
                    "section_title": "",
                    "summary": "",
                }
                continue
            current_indices.append(i)
            current_pages.append({"page_number": page["page_number"], "text": text[:3000]})
            if len(current_pages) >= self.BATCH_SIZE:
                batches.append((current_indices, current_pages))
                current_indices, current_pages = [], []
        if current_pages:
            batches.append((current_indices, current_pages))

        if not batches:
            return results

        workers = max(1, min(self.classify_workers, len(batches)))
        jobs = list(enumerate(batches))
        rank = lambda job: min((priority(p['page_number']) if priority else (0,p['page_number'])) for p in job[1][1])
        for job, parsed in _prioritized_batches(jobs, lambda job: self._classify_batch(job[1][1]), workers, rank):
                batch_n, (batch_indices, batch_pages) = job
                if isinstance(parsed, Exception):
                    logger.warning("batch classification failed: %s", parsed)
                    parsed = None
                # Map results back by page_number (LLM may reorder).
                if parsed is None:
                    parsed = []
                by_page = {p.get("page_number"): p for p in parsed if isinstance(p, dict)}
                batch_pairs: list[tuple[dict, Optional[dict]]] = []
                for orig_idx, src_page in zip(batch_indices, batch_pages):
                    pnum = src_page["page_number"]
                    res = by_page.get(pnum) or {
                        "content_types": ["unclassified"],
                        "concepts": [],
                        "section_title": "",
                        "summary": "",
                    }
                    results[orig_idx] = res
                    batch_pairs.append((pages[orig_idx], res))
                if on_batch and batch_pairs:
                    on_batch(batch_pairs)
        return results

    def _classify_batch(self, pages: list[dict]) -> Optional[list]:
        prompt = BATCH_CLASSIFY_PROMPT_TEMPLATE.format(
            pages_json=json.dumps(pages, ensure_ascii=False, indent=2),
        )
        out = llm.call_llm_json(prompt, system=CONTENT_CLASSIFY_SYSTEM, max_tokens=4000)
        if isinstance(out, list):
            return out
        return None

    # ── Image extraction + batched vision ─────────────────────────

    def _save_images_to_disk(
        self,
        pages: list[dict],
        source_doc_id: str,
    ) -> list[dict]:
        """Save extracted PIL images to disk; return metadata for vision batching."""
        out_dir = self.image_save_dir / source_doc_id
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: list[dict] = []
        for page in pages:
            page_num = page["page_number"]
            context = (page.get("text") or "")[:500]
            for img in page.get("images") or []:
                from .normalized_source import FIGURE_EXT, LazyImage, save_figure, store_once
                pil = img["pil_image"]
                try:
                    if isinstance(pil, LazyImage):  # one home for each figure: the page copy's file
                        file_path = pil.path
                    else:
                        file_path = out_dir / f"p{page_num}_i{img['idx']}.{FIGURE_EXT}"
                        save_figure(pil, file_path)
                except Exception as e:
                    logger.warning(f"failed saving image for page {page_num}: {e}")
                    continue
                saved.append({
                    "page_number": page_num,
                    "file_path": str(file_path),
                    "width": img["width"],
                    "height": img["height"],
                    "context_hint": context,
                    "img_idx": img["idx"],
                    "also_on_pages": img.get("also_on_pages") or [],
                    "bbox": img.get("bbox"),
                })
        return saved

    def _remote_settings(self) -> dict:
        """What a container needs to classify and describe exactly as this pipeline would."""
        return {"classify_workers": self.classify_workers, "vision_workers": self.vision_workers,
                "BATCH_SIZE": self.BATCH_SIZE, "VISION_BATCH_SIZE": self.VISION_BATCH_SIZE}

    def _describe_saved_images_batched(self, saved: list[dict], on_batch=None, priority=None) -> list[dict]:
        """Multi-image vision batches run in parallel."""
        if not saved:
            return []

        batches: list[list[dict]] = []
        for i in range(0, len(saved), self.VISION_BATCH_SIZE):
            batches.append(saved[i : i + self.VISION_BATCH_SIZE])

        results: list[dict] = [None] * len(saved)  # type: ignore
        workers = max(1, min(self.vision_workers, len(batches)))

        def _run_batch(batch: list[dict], global_offset: int) -> list[tuple[int, dict]]:
            batch_items = []
            for j, s in enumerate(batch):
                try:
                    import file_store
                    with open(file_store.local(s["file_path"]), "rb") as fh:
                        png_bytes = fh.read()
                except OSError as e:
                    logger.warning(f"vision read failed {s['file_path']}: {e}")
                    continue
                batch_items.append({
                    "image_index": j + 1,
                    "png_bytes": png_bytes,
                    "context_hint": s.get("context_hint", ""),
                    "_global_idx": global_offset + j,
                    "_meta": s,
                })

            if not batch_items:
                return []

            visions = llm.describe_images_batch(
                [{"image_index": b["image_index"], "png_bytes": b["png_bytes"],
                  "context_hint": b["context_hint"]} for b in batch_items],
            )

            out: list[tuple[int, dict]] = []
            for b, vis in zip(batch_items, visions):
                s = b["_meta"]
                if not isinstance(vis, dict):
                    vis = {"description": "", "image_type": "figure", "concepts": []}
                out.append((b["_global_idx"], {
                    "page_number": s["page_number"],
                    "file_path": s["file_path"],
                    "width": s["width"],
                    "height": s["height"],
                    "description": vis.get("description") or "",
                    "image_type": vis.get("image_type") or "figure",
                    "concepts": vis.get("concepts") or [],
                    "also_on_pages": s.get("also_on_pages") or [],
                    "bbox": s.get("bbox"),
                }))
            return out

        jobs = []
        offset = 0
        for batch in batches:
            jobs.append((batch, offset))
            offset += len(batch)
        rank = lambda job: min((priority(s['page_number']) if priority else (0,s['page_number'])) for s in job[0])
        for job, rows in _prioritized_batches(jobs, lambda job: _run_batch(*job), workers, rank):
            if isinstance(rows, Exception):
                logger.warning("vision batch failed: %s", rows)
                continue
            published=[]
            for idx,row in rows:
                if not (row.get('description') or '').strip(): row['_pending_vision']=True
                results[idx]=row
                published.append(row)
            if on_batch and published: on_batch(published)

        # Fill any gaps with empty descriptions.
        final: list[dict] = []
        for i, s in enumerate(saved):
            if results[i]:
                final.append(results[i])
            else:
                final.append({
                    "page_number": s["page_number"],
                    "file_path": s["file_path"],
                    "width": s["width"],
                    "height": s["height"],
                    "description": "",
                    "image_type": "figure",
                    "concepts": [],
                    "_pending_vision": True,
                })
        return final

    def _describe_images_parallel(self, pages: list[dict], source_doc_id: str) -> list[Optional[dict]]:
        """Legacy path — save then batch-describe."""
        saved = self._save_images_to_disk(pages, source_doc_id)
        if not saved:
            return []
        if not self.describe_images:
            return [{
                "page_number": s["page_number"],
                "file_path": s["file_path"],
                "width": s["width"],
                "height": s["height"],
                "description": "",
                "image_type": "figure",
                "concepts": [],
                "_pending_vision": True,
            } for s in saved]
        return self._describe_saved_images_batched(saved)

    # ── Optional backfill: describe images that were ingested without vision ──

    @staticmethod
    def _image_describe_pending(it) -> bool:
        ss = it.store_specific or {}
        if ss.get("_pending_vision"):
            return True
        body = (it.content or "").strip()
        if ss.get("file_path"):
            return not body or body in ("(no description)", "(no description yet)")
        return not body

    def _save_descriptions(self, results: list[dict], by_path: dict) -> int:
        """Write vision results onto their image items; returns how many were written."""
        n = 0
        for res in results:
            it = by_path.get(res.get("file_path", ""))
            if not it:
                continue
            desc = (res.get("description") or "").strip() or "(figure — description unavailable)"
            img_type = res.get("image_type") or "figure"
            concepts = res.get("concepts") or []
            ss = it.store_specific or {}
            ss.pop("_pending_vision", None)
            ss.pop("vision_tier", None)
            ss["image_type"] = img_type
            ss["concept_mentions_raw"] = concepts
            it.content = desc
            it.tags = [img_type]
            it.entities = [llm.normalize_concept_name(c) for c in concepts if c]
            self.images.update_store_specific(it.id, ss)
            self.images._insert(it)
            n += 1
        return n

    def describe_pending_images(
        self,
        namespace: str,
        progress: Optional[Callable[[str], None]] = None,
        max_workers: int | None = None,
        limit: Optional[int] = None,
        page_order: Optional[list[int]] = None,
        skip_doc_ids: Optional[set] = None,
    ) -> dict:
        """Run batched vision over pending images. Optional ``page_order`` prioritizes
        pages (e.g. section 0 first after outline generation)."""
        log = progress or (lambda m: logger.info(m))

        all_images = self.images.all(namespace)
        # Sources still being ingested describe their own figures; everything else
        # (including figures a source job could not describe) is swept here.
        skip = set(skip_doc_ids or ())
        pending = [it for it in all_images
                   if it.source_doc_id not in skip and self._image_describe_pending(it)]

        from . import progressive

        def _page_sort_key(it: MemoryItem) -> tuple:
            ss = it.store_specific or {}
            pn = int(ss.get("page_number") or 9999)
            if page_order:
                try:
                    return (0, page_order.index(pn))
                except ValueError:
                    return (1, pn)
            # The roadmap's order: the section the student is on first (progressive.set_priority).
            return (0, *progressive.page_rank(namespace, it.source_doc_id, pn))

        pending.sort(key=_page_sort_key)
        if limit:
            pending = pending[:limit]
        log(f"Found {len(pending)} pending images (of {len(all_images)} total)")

        saved: list[dict] = []
        by_path: dict[str, MemoryItem] = {}
        for it in pending:
            ss = it.store_specific or {}
            file_path = ss.get("file_path")
            import file_store  # in R2 when the server's disk does not hold it (read in a container)
            if not file_path or not file_store.available(file_path):
                continue
            saved.append({
                "page_number": ss.get("page_number"),
                "file_path": file_path,
                "width": ss.get("width", 0),
                "height": ss.get("height", 0),
                "context_hint": "",
            })
            by_path[file_path] = it

        if not saved:
            return {"described": 0, "pending_total": len(pending)}

        # In chunks, highest-priority pages first, each saved as soon as it is done: a section
        # waits for its own figures, not for the whole folder's sweep to finish.
        done = 0
        chunk = max(1, self.VISION_BATCH_SIZE * max(1, self.vision_workers))
        for start in range(0, len(saved), chunk):
            results: list[dict] = []
            remaining = saved[start:start + chunk]
            for attempt in range(self.VISION_ATTEMPTS):
                batch = self._describe_saved_images_batched(remaining)
                results += [r for r in batch if not r.get("_pending_vision")]
                remaining = [s for s, r in zip(remaining, batch) if r.get("_pending_vision")]
                if not remaining:
                    break
                log(f"{len(remaining)} figures not described (attempt {attempt + 1}); retrying")
                time.sleep(5 * (attempt + 1))
            # Give up on figures that keep failing: the section teaches from text instead
            # of waiting forever. They keep their file, so a later sweep can be forced.
            results += [{**s, "description": "", "_gave_up": True} for s in remaining]
            done += self._save_descriptions(results, by_path)
            log(f"Described {done}/{len(saved)} images")

        return {"described": done, "pending_total": len(pending)}

    # ── Canonicalization ──────────────────────────────────────────

    def _cluster_mentions(self, mentions: dict[str, set[str]]) -> list[dict]:
        """Greedy embedding clusters for canonicalization batching."""
        keys = list(mentions.keys())
        if not keys:
            return []
        if len(keys) == 1:
            return [{"cluster_id": 0, "terms": sorted(mentions[keys[0]])}]

        texts = [k for k in keys]
        embs = llm.embed_texts(texts)
        if not any(embs):
            return [
                {"cluster_id": i, "terms": sorted(surfaces)}
                for i, (_, surfaces) in enumerate(mentions.items())
            ]

        clusters: list[list[int]] = []
        assigned = [False] * len(keys)

        for i in range(len(keys)):
            if assigned[i] or not embs[i]:
                if not assigned[i]:
                    clusters.append([i])
                    assigned[i] = True
                continue
            group = [i]
            assigned[i] = True
            for j in range(i + 1, len(keys)):
                if assigned[j] or not embs[j]:
                    continue
                if llm.cosine_similarity(embs[i], embs[j]) >= self.CLUSTER_THRESHOLD:
                    group.append(j)
                    assigned[j] = True
            clusters.append(group)

        out = []
        for cid, idxs in enumerate(clusters):
            terms: set[str] = set()
            for ix in idxs:
                terms.update(mentions[keys[ix]])
            out.append({"cluster_id": cid, "terms": sorted(terms)})
        return out

    def _canonicalize_concepts_fast(self, mentions: dict[str, set[str]]) -> list[dict]:
        """Tier A: embedding clusters only — no LLM. Names from surface forms."""
        if not mentions:
            return []

        clusters = self._cluster_mentions(mentions)
        logger.info(
            "Tier A concept clustering: %d mentions → %d clusters (threshold %.2f)",
            len(mentions), len(clusters), self.CLUSTER_THRESHOLD,
        )
        canonical: list[dict] = []
        for cluster in clusters:
            terms = cluster.get("terms") or []
            if not terms:
                continue
            name = min(terms, key=lambda t: (len(t), t.lower())).lower().strip()
            aliases = sorted({
                t for t in terms if t.lower().strip() != name
            })
            canonical.append({
                "name": name,
                "aliases": aliases,
                "definition": "",
                "prerequisites": [],
                "related": [],
                "provisional": True,
            })
        return self._merge_canonical_by_name(canonical)

    def _canonicalize_concepts_clustered(self, mentions: dict[str, set[str]]) -> list[dict]:
        if not mentions:
            return []

        clusters = self._cluster_mentions(mentions)
        logger.info(
            "concept clustering: %d mentions → %d clusters (threshold %.2f)",
            len(mentions), len(clusters), self.CLUSTER_THRESHOLD,
        )

        if len(clusters) <= 3 and sum(len(c["terms"]) for c in clusters) <= 30:
            return self._canonicalize_concepts(mentions)

        prompt = CONCEPT_CLUSTER_CANONICALIZE_PROMPT_TEMPLATE.format(
            clusters_json=json.dumps(clusters, ensure_ascii=False, indent=2),
        )
        out = llm.call_llm_json(
            prompt,
            system=CONCEPT_CLUSTER_CANONICALIZE_SYSTEM,
            max_tokens=6000,
        )
        canonical: list[dict] = []
        if isinstance(out, list):
            for row in out:
                if not isinstance(row, dict):
                    continue
                name = (row.get("name") or "").lower().strip()
                if not name:
                    continue
                canonical.append({
                    "name": name,
                    "aliases": list(set(row.get("aliases") or [])),
                    "definition": row.get("definition") or "",
                    "prerequisites": list(set(row.get("prerequisites") or [])),
                    "related": list(set(row.get("related") or [])),
                })

        if not canonical:
            logger.warning("cluster canonicalization failed; falling back to mention-list pass")
            return self._canonicalize_concepts(mentions)

        if len(canonical) >= 2:
            canonical = self._merge_cross_cluster_prerequisites(canonical)

        return self._merge_canonical_by_name(canonical)

    def _merge_cross_cluster_prerequisites(self, canonical: list[dict]) -> list[dict]:
        """Second pass: infer prereqs between dissimilar clusters."""
        brief = [
            {"name": c["name"], "definition": (c.get("definition") or "")[:200]}
            for c in canonical
        ]
        prompt = CROSS_CLUSTER_PREREQ_PROMPT_TEMPLATE.format(
            concepts_json=json.dumps(brief, ensure_ascii=False, indent=2),
        )
        out = llm.call_llm_json(prompt, system=CROSS_CLUSTER_PREREQ_SYSTEM, max_tokens=2000)
        if not isinstance(out, list):
            return canonical

        by_name = {(c.get("name") or "").lower(): c for c in canonical}
        for row in out:
            if not isinstance(row, dict):
                continue
            name = (row.get("concept") or "").lower().strip()
            target = by_name.get(name)
            if not target:
                continue
            pre = set(target.get("prerequisites") or [])
            for p in row.get("prerequisites") or []:
                pn = (p or "").lower().strip()
                if pn in by_name and pn != name:
                    pre.add(pn)
            target["prerequisites"] = list(pre)
        return list(by_name.values())

    def _merge_canonical_by_name(self, canonical: list[dict]) -> list[dict]:
        merged: dict[str, dict] = {}
        for c in canonical:
            name = (c.get("name") or "").lower().strip()
            if not name:
                continue
            if name not in merged:
                merged[name] = {
                    "name": name,
                    "aliases": list(set(c.get("aliases") or [])),
                    "definition": c.get("definition") or "",
                    "prerequisites": list(set(c.get("prerequisites") or [])),
                    "related": list(set(c.get("related") or [])),
                }
            else:
                ex = merged[name]
                ex["aliases"] = list(set(ex["aliases"]) | set(c.get("aliases") or []))
                ex["prerequisites"] = list(set(ex["prerequisites"]) | set(c.get("prerequisites") or []))
                ex["related"] = list(set(ex["related"]) | set(c.get("related") or []))
                if not ex["definition"] and c.get("definition"):
                    ex["definition"] = c["definition"]
        return list(merged.values())

    def _canonicalize_concepts(self, mentions: dict[str, set[str]]) -> list[dict]:
        if not mentions:
            return []

        # Build the input list the LLM sees.
        mention_list = []
        for norm, surfaces in mentions.items():
            mention_list.append({
                "normalized": norm,
                "surface_forms": sorted(surfaces),
            })

        # If the mention list is huge, chunk it. The LLM call is one-shot;
        # we put up to ~300 mentions per call.
        canonical: list[dict] = []
        CHUNK = 250
        for i in range(0, len(mention_list), CHUNK):
            chunk = mention_list[i : i + CHUNK]
            prompt = CONCEPT_CANONICALIZE_PROMPT_TEMPLATE.format(
                mentions_json=json.dumps(chunk, ensure_ascii=False, indent=2),
            )
            out = llm.call_llm_json(prompt, system=CONCEPT_CANONICALIZE_SYSTEM, max_tokens=4000)
            if isinstance(out, list):
                canonical.extend([c for c in out if isinstance(c, dict)])

        # Merge any duplicates across chunks by canonical name.
        merged: dict[str, dict] = {}
        for c in canonical:
            name = (c.get("name") or "").lower().strip()
            if not name:
                continue
            if name not in merged:
                merged[name] = {
                    "name": name,
                    "aliases": list(set(c.get("aliases") or [])),
                    "definition": c.get("definition") or "",
                    "prerequisites": list(set(c.get("prerequisites") or [])),
                    "related": list(set(c.get("related") or [])),
                }
            else:
                existing = merged[name]
                existing["aliases"] = list(set(existing["aliases"]) | set(c.get("aliases") or []))
                existing["prerequisites"] = list(set(existing["prerequisites"]) | set(c.get("prerequisites") or []))
                existing["related"] = list(set(existing["related"]) | set(c.get("related") or []))
                if not existing["definition"] and c.get("definition"):
                    existing["definition"] = c["definition"]

        return list(merged.values())

    def _write_concepts_with_dedup(
        self,
        namespace: str,
        canonical: list[dict],
        log: Callable[[str], None],
    ) -> tuple[dict[str, str], int, int]:
        """Write canonical concepts, reusing existing namespace concepts.

        Match order per concept:
          1. exact name/alias match against existing concepts
          2. embedding cosine >= CONCEPT_MERGE_THRESHOLD
          3. otherwise create a new node

        Returns (name_to_id incl. existing ids, n_new, n_merged)."""
        db_by_name: dict[str, MemoryItem] = {}
        for it in self.concept.all(namespace):
            ss = it.store_specific or {}
            for n in [ss.get("name")] + (ss.get("aliases") or []):
                if n:
                    db_by_name[n.lower().strip()] = it

        # In-batch name lookup includes not-yet-written items; similarity
        # dedup only searches concepts already persisted (never pending batch).
        existing_by_name: dict[str, MemoryItem] = dict(db_by_name)

        name_to_id: dict[str, str] = {}
        new_items: list[MemoryItem] = []
        new_concepts: list[dict] = []
        n_merged = 0

        for concept in canonical:
            name = (concept.get("name") or "").lower().strip()
            if not name:
                continue
            aliases = [a for a in (concept.get("aliases") or []) if a]
            definition = concept.get("definition") or ""
            provisional = bool(concept.get("provisional")) and not definition

            hit = existing_by_name.get(name)
            if hit is None:
                for a in aliases:
                    hit = existing_by_name.get(a.lower().strip())
                    if hit is not None:
                        break

            if hit is None and not provisional and db_by_name:
                merge_threshold = effective_merge_threshold(self.CONCEPT_MERGE_THRESHOLD)
                sims = []
                if merge_threshold > 0:
                    sims = self.concept.find_similar(
                        namespace,
                        f"{name}: {definition}" if definition else name,
                        threshold=merge_threshold,
                    )
                if sims:
                    hit = sims[0][0]
                    logger.info(
                        "concept dedup: '%s' merged into '%s' (cos %.3f)",
                        name,
                        (hit.store_specific or {}).get("name"),
                        sims[0][1],
                    )

            if hit is not None:
                hss = dict(hit.store_specific or {})
                hit_name = (hss.get("name") or "").lower().strip()
                merged_aliases = list(dict.fromkeys(
                    (hss.get("aliases") or []) + [name] + aliases
                ))
                hss["aliases"] = [
                    a for a in merged_aliases if a and a.lower().strip() != hit_name
                ]
                if definition and not hss.get("definition"):
                    hss["definition"] = definition
                    hit.content = definition
                if definition or not concept.get("provisional"):
                    hss["provisional"] = False
                elif "provisional" not in hss:
                    hss["provisional"] = provisional
                hit.store_specific = hss
                hit.entities = [hss.get("name") or hit.id] + hss["aliases"]
                self.concept.write_item(hit)
                for n in [name] + aliases:
                    key = n.lower().strip()
                    name_to_id[key] = hit.id
                    existing_by_name[key] = hit
                n_merged += 1
                continue

            item = MemoryItem(
                id=new_item_id("concept"),
                namespace=namespace,
                store="concept",
                content=definition or name,
                entities=[name] + aliases,
                importance=0.7,
                store_specific={
                    "name": name,
                    "aliases": aliases,
                    "definition": definition,
                    "provisional": provisional,
                    "prerequisite_concept_ids": [],
                    "related_concept_ids": [],
                    "prerequisite_names_raw": concept.get("prerequisites") or [],
                    "related_names_raw": concept.get("related") or [],
                },
            )
            new_items.append(item)
            new_concepts.append(concept)
            for n in [name] + aliases:
                key = n.lower().strip()
                name_to_id[key] = item.id
                existing_by_name[key] = item

        if new_items:
            log(f"  → bulk-embedding {len(new_items)} new concepts")
            self.concept.write_items_bulk(new_items)

        # Resolve prerequisite/related names → ids (may point at existing
        # concepts too). Edges are unioned so a merge never loses links.
        def _resolve(names: list[str]) -> list[str]:
            out = []
            for n in names or []:
                key = (n or "").lower().strip()
                it = existing_by_name.get(key)
                if it is not None:
                    out.append(it.id)
            return out

        for concept in canonical:
            name = (concept.get("name") or "").lower().strip()
            cid = name_to_id.get(name)
            if not cid:
                continue
            pre = concept.get("prerequisites") or []
            rel = concept.get("related") or []
            if not pre and not rel:
                continue
            current = self.concept.get(cid)
            css = (current.store_specific or {}) if current else {}
            pre_ids = list(dict.fromkeys(
                (css.get("prerequisite_concept_ids") or [])
                + _resolve(concept.get("prerequisites"))
            ))
            rel_ids = list(dict.fromkeys(
                (css.get("related_concept_ids") or [])
                + _resolve(concept.get("related"))
            ))
            self.concept.update_store_specific(cid, {
                "prerequisite_concept_ids": [p for p in pre_ids if p != cid],
                "related_concept_ids": [r for r in rel_ids if r != cid],
            })

        return name_to_id, len(new_items), n_merged

    def _backfill_canonical_entities(self, namespace: str, name_to_id: dict[str, str]) -> None:
        """Rewrite each content/image item's `entities` from normalized
        concept names to canonical concept ids. Updates entities only —
        no embedding recompute (content text didn't change)."""
        import sqlite3
        for store in (self.content, self.images):
            items = store.all(namespace)
            updates: list[tuple[str, str, str]] = []  # (entities_json, fts_entities, id)
            for it in items:
                if not it.entities:
                    continue
                new_entities = []
                for e in it.entities:
                    cid = name_to_id.get(e.lower().strip())
                    new_entities.append(cid if cid else e)
                if new_entities != it.entities:
                    updates.append((
                        json.dumps(new_entities),
                        " ".join(new_entities),
                        it.id,
                    ))
            if not updates:
                continue
            from .stores.db import connect_db, fts_rowid
            with connect_db(store.db_path) as conn:
                conn.executemany(
                    f"UPDATE {store.table} SET entities = ? WHERE id = ?",
                    [(u[0], u[2]) for u in updates],
                )
                conn.executemany(
                    f"UPDATE {store.fts_table} SET entities = ? WHERE rowid = ?",
                    [(u[1], fts_rowid(u[2])) for u in updates],
                )

    # ── Misc ──────────────────────────────────────────────────────

    def _doc_id(self, pdf_path: Path, source_id: Optional[str] = None) -> str:
        if source_id:
            return f"doc_{source_id}"
        try:
            size = pdf_path.stat().st_size
        except OSError:
            size = 0
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", pdf_path.stem).strip("_").lower()
        suffix = uuid.uuid4().hex[:6]
        return f"doc_{slug}_{size}_{suffix}"


def _prioritized_batches(jobs, operation, workers, rank):
    pending=list(jobs)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        active={}
        while pending or active:
            while pending and len(active)<workers:
                pending.sort(key=rank)
                job=pending.pop(0)
                active[executor.submit(_carry(operation),job)]=job
            done,_=wait(active,return_when=FIRST_COMPLETED)
            for future in done:
                job=active.pop(future)
                try: result=future.result()
                except Exception as error: result=error
                yield job,result
