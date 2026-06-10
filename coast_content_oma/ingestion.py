"""Ingest course material into the Content OMA stores.

Pipeline:
  1. Extract pages from each PDF (text + PIL images).
  2. For each page: LLM call classifies content_type(s) + extracts
     mentioned concepts. ContentStore item created.
  3. For each image: vision LLM call writes ImageStore item.
  4. After all docs processed: canonicalization pass — dedupe concept
     mentions, infer prerequisites, write to ConceptStore.
  5. Backfill: update ContentStore/ImageStore items to use canonical
     concept_ids instead of raw mentions.

Parallelism: per-page LLM calls run via ThreadPoolExecutor (default 4).
Vision calls run in the same pool.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from . import llm
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
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "docs": self.docs,
            "pages": self.pages,
            "content_items": self.content_items,
            "image_items": self.image_items,
            "concepts": self.concepts,
            "errors": self.errors,
        }


class IngestionPipeline:
    def __init__(
        self,
        concept_store: ConceptStore,
        content_store: ContentStore,
        image_store: ImageStore,
        image_save_dir: Path,
        max_workers: int = 4,
        describe_images: bool = False,
        skip_images: bool = False,
    ):
        """describe_images: if False (default), images are saved and recorded
        with empty descriptions — much faster ingestion. Call
        describe_pending_images() later to fill in vision descriptions in
        batches. If True, vision runs inline during ingestion.

        skip_images: if True, images are not even extracted from the PDF.
        Use for the fastest possible text-only ingestion."""
        self.concept = concept_store
        self.content = content_store
        self.images = image_store
        self.image_save_dir = Path(image_save_dir)
        self.image_save_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.describe_images = describe_images
        self.skip_images = skip_images

    # ── Top-level ─────────────────────────────────────────────────

    def ingest_folder(
        self,
        namespace: str,
        pdf_paths: list[Path],
        progress: Optional[Callable[[str], None]] = None,
    ) -> IngestStats:
        log = progress or (lambda msg: logger.info(msg))
        stats = IngestStats()
        all_concept_mentions: dict[str, set[str]] = {}  # normalized -> {original surface forms}

        for pdf in pdf_paths:
            try:
                log(f"Extracting text from {pdf.name}")
                pages = extract_pages(pdf, extract_images=not self.skip_images)
                stats.docs += 1
                stats.pages += len(pages)
                source_doc_id = self._doc_id(pdf)
                log(f"  → {len(pages)} pages extracted")

                # 1. Classify pages in BATCHES (1 LLM call per ~8 pages).
                num_batches = max(1, (len(pages) + self.BATCH_SIZE - 1) // self.BATCH_SIZE)
                log(f"Classifying {len(pages)} pages from {pdf.name} ({num_batches} batched LLM calls)")
                page_results = self._classify_pages_parallel(pages)
                log(f"  → classification done")

                # 2. Process images in parallel.
                if self.skip_images:
                    image_results = []
                    log("  → skipping images (--skip-images)")
                else:
                    log(f"Processing images from {pdf.name}")
                    image_results = self._describe_images_parallel(pages, source_doc_id)
                    log(f"  → {len(image_results)} images")

                # 3. Build content items in memory, then bulk-write with
                # batched embeddings (much faster than per-item embed calls).
                # ALWAYS write the page even if classification failed.
                content_items_by_page: dict[int, str] = {}
                content_items_to_write: list[MemoryItem] = []
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
                        id=new_item_id("content"),
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
                            "image_ids": [],
                        },
                    )
                    content_items_to_write.append(item)
                    content_items_by_page[page["page_number"]] = item.id
                    stats.content_items += 1

                if content_items_to_write:
                    log(f"  → bulk-embedding {len(content_items_to_write)} content items")
                    self.content.write_items_bulk(content_items_to_write)

                # 4. Build image items in memory, then bulk-write.
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

                    item = MemoryItem(
                        id=new_item_id("image"),
                        namespace=namespace,
                        store="image",
                        content=desc or "(no description)",
                        source_doc_id=source_doc_id,
                        tags=[img_type],
                        entities=[llm.normalize_concept_name(c) for c in img_concepts if c],
                        importance=0.4 if img_type == "decorative" else 0.6,
                        store_specific={
                            "image_type": img_type,
                            "concept_mentions_raw": img_concepts,
                            "page_number": page_num,
                            "source_filename": pdf.name,
                            "file_path": file_path,
                            "width": width,
                            "height": height,
                        },
                    )
                    image_items_to_write.append(item)
                    image_to_page[item.id] = page_num
                    stats.image_items += 1

                if image_items_to_write:
                    log(f"  → bulk-writing {len(image_items_to_write)} image items")
                    self.images.write_items_bulk(image_items_to_write)

                    # Cross-link images back into their content items in one pass.
                    by_page: dict[int, list[str]] = {}
                    for img in image_items_to_write:
                        by_page.setdefault(image_to_page[img.id], []).append(img.id)
                    for page_num, img_ids in by_page.items():
                        content_id = content_items_by_page.get(page_num)
                        if content_id:
                            self.content.update_store_specific(content_id, {"image_ids": img_ids})

            except Exception as e:
                logger.exception(f"failed to ingest {pdf}")
                stats.errors.append(f"{pdf.name}: {e}")
                continue

        # 5. Concept canonicalization pass — dedup-aware against any
        # concepts already in the namespace (incremental per-PDF ingests
        # must NOT create a second "eigenvalue" node).
        log(f"Canonicalizing {len(all_concept_mentions)} concept mentions")
        canonical = self._canonicalize_concepts(all_concept_mentions)
        name_to_id, n_new, n_merged = self._write_concepts_with_dedup(
            namespace, canonical, log,
        )
        log(f"Concepts: {n_new} new, {n_merged} merged into existing")
        stats.concepts = n_new

        # 6. Backfill content/image items: replace raw concept mentions with
        # canonical ids in entities.
        log("Backfilling canonical concept ids on content/image items")
        self._backfill_canonical_entities(namespace, name_to_id)

        return stats

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
        mentions: dict[str, set[str]] = {}

        for it in self.content.all(namespace):
            for c in (it.store_specific or {}).get("concept_mentions_raw") or []:
                norm = llm.normalize_concept_name(c)
                if norm:
                    mentions.setdefault(norm, set()).add(c)
        for it in self.images.all(namespace):
            for c in (it.store_specific or {}).get("concept_mentions_raw") or []:
                norm = llm.normalize_concept_name(c)
                if norm:
                    mentions.setdefault(norm, set()).add(c)

        log(f"Re-canonicalizing {len(mentions)} concept mentions")

        old_concepts = self.concept.all(namespace)
        from ..student.concept_remap import build_concept_id_map

        # Wipe existing concept store entries for this namespace.
        wiped = self.concept.delete_namespace(namespace)
        log(f"Cleared {wiped} previous concept entries")

        canonical = self._canonicalize_concepts(mentions)
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

    # ── Page classification ───────────────────────────────────────

    BATCH_SIZE = 8  # pages per LLM classification call

    def _classify_pages_parallel(self, pages: list[dict]) -> list[Optional[dict]]:
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

        # Run batches in parallel (kept small — 2 — to avoid rate-limit thrash).
        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = {}
            for batch_n, (_indices, batch_pages) in enumerate(batches):
                futs[ex.submit(self._classify_batch, batch_pages)] = batch_n
            for fut in as_completed(futs):
                batch_n = futs[fut]
                batch_indices, batch_pages = batches[batch_n]
                try:
                    parsed = fut.result()
                except Exception as e:
                    logger.warning(f"batch classification failed: {e}")
                    parsed = None
                # Map results back by page_number (LLM may reorder).
                if parsed is None:
                    parsed = []
                by_page = {p.get("page_number"): p for p in parsed if isinstance(p, dict)}
                for orig_idx, src_page in zip(batch_indices, batch_pages):
                    pnum = src_page["page_number"]
                    results[orig_idx] = by_page.get(pnum) or {
                        "content_types": ["unclassified"],
                        "concepts": [],
                        "section_title": "",
                        "summary": "",
                    }
        return results

    def _classify_batch(self, pages: list[dict]) -> Optional[list]:
        prompt = BATCH_CLASSIFY_PROMPT_TEMPLATE.format(
            pages_json=json.dumps(pages, ensure_ascii=False, indent=2),
        )
        out = llm.call_llm_json(prompt, system=CONTENT_CLASSIFY_SYSTEM, max_tokens=4000)
        if isinstance(out, list):
            return out
        return None

    # ── Image description ────────────────────────────────────────

    def _describe_images_parallel(self, pages: list[dict], source_doc_id: str) -> list[Optional[dict]]:
        tasks: list[tuple[int, int, dict]] = []
        for p_idx, page in enumerate(pages):
            for img in page.get("images") or []:
                tasks.append((p_idx, img["idx"], img))

        if not tasks:
            return []

        # Vision calls are token-heavy (~50K each). Cap parallelism lower
        # than text-only classification to stay under per-minute TPM limits.
        vision_workers = max(1, min(2, self.max_workers))

        results: list[Optional[dict]] = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=vision_workers) as ex:
            futs = {}
            for ti, (p_idx, img_idx, img) in enumerate(tasks):
                page = pages[p_idx]
                fut = ex.submit(
                    self._describe_one_image,
                    img["pil_image"],
                    img["width"],
                    img["height"],
                    page["text"],
                    page["page_number"],
                    source_doc_id,
                    img_idx,
                )
                futs[fut] = ti
            for fut in as_completed(futs):
                ti = futs[fut]
                try:
                    results[ti] = fut.result()
                except Exception as e:
                    logger.warning(f"image description failed: {e}")
                    results[ti] = None
        return results

    def _describe_one_image(
        self,
        pil_image,
        width: int,
        height: int,
        context_text: str,
        page_number: int,
        source_doc_id: str,
        img_idx: int,
    ) -> Optional[dict]:
        out_dir = self.image_save_dir / source_doc_id
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir / f"p{page_number}_i{img_idx}.png"
        try:
            pil_image.save(file_path, "PNG", optimize=True)
        except Exception as e:
            logger.warning(f"failed saving image {file_path}: {e}")
            return None

        # Skip the vision call if requested — image is still saved and
        # recorded so describe_pending_images() can backfill later.
        if not self.describe_images:
            return {
                "page_number": page_number,
                "file_path": str(file_path),
                "width": width,
                "height": height,
                "description": "",
                "image_type": "figure",
                "concepts": [],
                "_pending_vision": True,
            }

        vision = llm.describe_image(pil_image, context_hint=context_text)
        if not isinstance(vision, dict):
            vision = {"description": "", "image_type": "figure", "concepts": []}

        return {
            "page_number": page_number,
            "file_path": str(file_path),
            "width": width,
            "height": height,
            "description": vision.get("description", ""),
            "image_type": vision.get("image_type", "figure"),
            "concepts": vision.get("concepts", []) or [],
        }

    # ── Optional backfill: describe images that were ingested without vision ──

    def describe_pending_images(
        self,
        namespace: str,
        progress: Optional[Callable[[str], None]] = None,
        max_workers: int = 2,
        limit: Optional[int] = None,
    ) -> dict:
        """Run vision over all images in `namespace` that were stored
        with empty descriptions. Returns counts."""
        from PIL import Image  # local import to keep top deps light
        log = progress or (lambda m: logger.info(m))

        all_images = self.images.all(namespace)
        pending = [
            it for it in all_images
            if not (it.content or "").strip() or (it.store_specific or {}).get("_pending_vision")
        ]
        if limit:
            pending = pending[:limit]
        log(f"Found {len(pending)} pending images (of {len(all_images)} total)")

        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {}
            for it in pending:
                ss = it.store_specific or {}
                file_path = ss.get("file_path")
                if not file_path or not Path(file_path).exists():
                    continue
                try:
                    pil = Image.open(file_path)
                except Exception as e:
                    logger.warning(f"could not open {file_path}: {e}")
                    continue
                futs[ex.submit(llm.describe_image, pil, context_hint="")] = it

            for fut in as_completed(futs):
                it = futs[fut]
                try:
                    vision = fut.result()
                except Exception as e:
                    logger.warning(f"vision failed for image {it.id}: {e}")
                    continue
                if not isinstance(vision, dict):
                    continue
                desc = vision.get("description") or ""
                img_type = vision.get("image_type") or "figure"
                concepts = vision.get("concepts") or []
                # Update the existing item: rewrite content + tags + entities + store_specific.
                ss = it.store_specific or {}
                ss.pop("_pending_vision", None)
                ss["image_type"] = img_type
                ss["concept_mentions_raw"] = concepts
                it.content = desc
                it.tags = [img_type]
                it.entities = [llm.normalize_concept_name(c) for c in concepts if c]
                self.images.update_store_specific(it.id, ss)
                self.images._insert(it)  # re-insert to refresh embeddings/FTS
                done += 1
                if done % 10 == 0:
                    log(f"Described {done}/{len(pending)} images")

        return {"described": done, "pending_total": len(pending)}

    # ── Canonicalization ──────────────────────────────────────────

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

    # Cosine similarity above which a new concept is merged into an
    # existing one instead of creating a duplicate node. 0 disables.
    CONCEPT_MERGE_THRESHOLD = float(os.environ.get("OMA_CONCEPT_MERGE_THRESHOLD", "0.90"))

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
        existing_by_name: dict[str, MemoryItem] = {}
        for it in self.concept.all(namespace):
            ss = it.store_specific or {}
            for n in [ss.get("name")] + (ss.get("aliases") or []):
                if n:
                    existing_by_name[n.lower().strip()] = it

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

            hit = existing_by_name.get(name)
            if hit is None:
                for a in aliases:
                    hit = existing_by_name.get(a.lower().strip())
                    if hit is not None:
                        break

            if hit is None and self.CONCEPT_MERGE_THRESHOLD > 0 and existing_by_name:
                sims = self.concept.find_similar(
                    namespace,
                    f"{name}: {definition}" if definition else name,
                    threshold=self.CONCEPT_MERGE_THRESHOLD,
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
            from .stores.db import connect_db
            with connect_db(store.db_path) as conn:
                conn.executemany(
                    f"UPDATE {store.table} SET entities = ? WHERE id = ?",
                    [(u[0], u[2]) for u in updates],
                )
                conn.executemany(
                    f"UPDATE {store.fts_table} SET entities = ? WHERE id = ?",
                    [(u[1], u[2]) for u in updates],
                )

    # ── Misc ──────────────────────────────────────────────────────

    def _doc_id(self, pdf_path: Path) -> str:
        # Stable id from filename + size.
        try:
            size = pdf_path.stat().st_size
        except OSError:
            size = 0
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", pdf_path.stem).strip("_").lower()
        suffix = uuid.uuid4().hex[:6]
        return f"doc_{slug}_{size}_{suffix}"
