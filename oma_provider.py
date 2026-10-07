"""Coast ↔ OMA bridge.

One module that wires Content OMA + Student OMA into Coast's existing
upload and chat endpoints. Designed to be additive: when RAG_PROVIDER=flat
(default) Coast behaves exactly as before. When RAG_PROVIDER=oma the
chat retrieves via Content OMA and records episodes into Student OMA.

Env vars:
  RAG_PROVIDER         flat | oma | shadow   (default: flat)
  STUDENT_OMA_ENABLED  true | false           (default: true when oma)
  OMA_DB_PATH          path to SQLite db      (default: ./oma_data/oma.db)
  OMA_IMAGE_DIR        path to image store    (default: ./oma_data/images)

The Coast server.py calls:
  - ingest_pdf_into_oma(user_id, folder, pdf_path)   from upload background thread
  - get_folder_context(user_id, folder, query)       in chat instead of rag.build_folder_context
  - get_student_profile_block(user_id, folder, q)    optional context for Pedro's prompt
  - record_chat_episode(...)                         after each Pedro response
"""

from __future__ import annotations

import logging
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("oma_provider")
logger.setLevel(logging.INFO)
# Ensure messages reach stdout (Coast doesn't configure root logging).
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[oma] %(message)s"))
    logger.addHandler(_h)
    logger.propagate = False


# ── Configuration ────────────────────────────────────────────────────

def _env_truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


_RENDER_DATA = Path("/data")
_ON_RENDER_DISK = bool(os.getenv("RENDER") and _RENDER_DATA.is_dir())


def _resolve_rag_provider() -> str:
    # OMA is how Coast reads courses and remembers students; "flat" (Chroma) remains only as an
    # explicit RAG_PROVIDER=flat, never as a silent fallback when the setting is missing.
    return (os.environ.get("RAG_PROVIDER") or "oma").strip().strip("'\"").lower()


_runtime_rag_provider: str | None = None


def get_rag_provider() -> str:
    if _runtime_rag_provider is not None:
        return _runtime_rag_provider
    return _resolve_rag_provider()


def set_rag_provider(mode: str) -> dict:
    global _runtime_rag_provider
    m = (mode or "flat").strip().lower()
    if m not in ("flat", "oma", "shadow"):
        m = "flat"
    _runtime_rag_provider = m
    logger.info("content provider → %s", m)
    return content_provider_status()


def content_provider_status() -> dict:
    p = get_rag_provider()
    student = _env_truthy(os.environ.get("STUDENT_OMA_ENABLED")) or p in ("oma", "shadow")
    return {
        "provider": p,
        "oma_enabled": p in ("oma", "shadow"),
        "student_oma_enabled": student,
        "label": "Content OMA" if p in ("oma", "shadow") else "RAG",
    }


def _resolve_data_path(env_key: str, local_default: str, render_default: str) -> Path:
    raw = os.environ.get(env_key)
    path = Path(raw) if raw else Path(render_default if _ON_RENDER_DISK else local_default)
    # CLI scripts and the server must resolve relative paths to the same data.
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path.resolve()


RAG_PROVIDER = get_rag_provider()
STUDENT_OMA_ENABLED = _env_truthy(os.environ.get("STUDENT_OMA_ENABLED")) or RAG_PROVIDER in ("oma", "shadow")

OMA_DB_PATH = _resolve_data_path("OMA_DB_PATH", "./oma_data/oma.db", "/data/oma_data/oma.db")
OMA_IMAGE_DIR = _resolve_data_path("OMA_IMAGE_DIR", "./oma_data/images", "/data/oma_data/images")

# Per-turn retrieval capture. Each chat turn calls reset_content_retrieval_log()
# which creates a fresh capture dict and points this thread at it. Deep retrieval
# calls append via the thread-local pointer; the caller keeps a direct reference
# so the summary is correct even if the response is finalized on another thread.
# (Module-level lists here used to race between concurrent users.)
_capture_tls = threading.local()


def _current_capture() -> dict | None:
    return getattr(_capture_tls, "ctx", None)

# Track background OMA ingests so outline generation can wait for them.
_ingest_lock = threading.Lock()
_ingest_active: dict[str, int] = {}
_ingest_source_active: set[str] = set()
_ingest_timings: dict[str, list[dict]] = {}
_ingest_timing_t0: dict[str, float] = {}
_ingest_slots = max(1, min(4, int(os.getenv("OMA_DOCUMENT_CONCURRENCY", "2"))))
_ingest_semaphore = threading.Semaphore(_ingest_slots)  # limit parallel PDF ingests (LLM-heavy)


def _ingest_key(user_id: int | str, folder: str) -> str:
    return f"{user_id}:{folder}"


def record_ingest_timing(user_id: int | str, folder: str, phase: str, **meta) -> None:
    """Append a timestamped ingest phase entry (inspect via /oma-ingest or server logs)."""
    key = _ingest_key(user_id, folder)
    now = time.time()
    with _ingest_lock:
        if key not in _ingest_timing_t0:
            _ingest_timing_t0[key] = now
        t0 = _ingest_timing_t0[key]
        entry = {
            "phase": phase,
            "elapsed_ms": int((now - t0) * 1000),
            **meta,
        }
        _ingest_timings.setdefault(key, []).append(entry)
    logger.info("[oma-timing] folder=%s phase=%s elapsed_ms=%d %s", folder, phase, entry["elapsed_ms"], meta)


def get_ingest_timings(user_id: int | str, folder: str) -> list[dict]:
    key = _ingest_key(user_id, folder)
    with _ingest_lock:
        return list(_ingest_timings.get(key, []))


def clear_ingest_timings(user_id: int | str, folder: str) -> None:
    key = _ingest_key(user_id, folder)
    with _ingest_lock:
        _ingest_timings.pop(key, None)
        _ingest_timing_t0.pop(key, None)


def make_ingest_timing_callback(user_id: int | str, folder: str):
    def on_timing(phase: str, meta: dict) -> None:
        record_ingest_timing(user_id, folder, phase, **(meta or {}))
    return on_timing


def reset_content_retrieval_log() -> dict:
    """Start a fresh retrieval capture for this chat turn.

    Returns the capture dict; pass it back to summarize_content_retrieval()
    so concurrent users never see each other's retrieval metadata.
    """
    ctx = {"entries": [], "images": [], "oma_meta": {}}
    _capture_tls.ctx = ctx
    return ctx


def _image_has_description(content: str, store_specific: dict | None) -> bool:
    ss = store_specific or {}
    if ss.get("_pending_vision"):
        return False
    desc = (content or "").strip()
    return bool(desc) and desc not in ("(no description)", "(no description yet)")


def _priority_pages_limit() -> int:
    return int(os.environ.get("OMA_PRIORITY_PAGES", "12"))


def _image_needs_vision(it) -> bool:
    ss = it.store_specific or {}
    if ss.get("_pending_vision"):
        return True
    body = (it.content or "").strip()
    return bool(ss.get("file_path")) and (not body or body == "(no description)")


def _count_pending_vision(orch, ns, *, priority_only: bool = False) -> tuple[int, int]:
    """Return (priority_pending, background_pending) image counts."""
    limit = _priority_pages_limit()
    priority = 0
    background = 0
    for it in orch.images.all(ns):
        if not _image_needs_vision(it):
            continue
        pn = int((it.store_specific or {}).get("page_number") or 0)
        if pn <= limit:
            priority += 1
        else:
            background += 1
    if priority_only:
        return priority, background
    return priority, background


def _priority_vision_complete(orch, ns) -> bool:
    if _env_truthy(os.environ.get("OMA_SKIP_IMAGES", "false")):
        return True
    limit = _priority_pages_limit()
    for it in orch.images.all(ns):
        ss = it.store_specific or {}
        pn = int(ss.get("page_number") or 0)
        if pn > limit:
            continue
        if _image_needs_vision(it):
            return False
    return True


def pages_for_outline_section(orch, ns: str, section: dict) -> set[int]:
    """Best-effort page numbers for an outline section from OMA content metadata."""
    key_topics = [str(t).lower() for t in (section.get("key_topics") or [])]
    source_nbs = [str(s).lower() for s in (section.get("source_notebooks") or [])]
    pages: set[int] = set()
    for it in orch.content.all(ns):
        ss = it.store_specific or {}
        pn = ss.get("page_number")
        if pn is None:
            continue
        pn = int(pn)
        fname = (ss.get("source_filename") or "").lower()
        sec_title = (ss.get("section_title") or "").lower()
        summary = (it.content or "")[:800].lower()
        if source_nbs and any(sn in fname for sn in source_nbs):
            pages.add(pn)
        if key_topics and any(kt in sec_title or kt in summary for kt in key_topics):
            pages.add(pn)
    if not pages:
        limit = _priority_pages_limit()
        by_source: dict[str, list[int]] = {}
        for it in orch.content.all(ns):
            ss = it.store_specific or {}
            pn = ss.get("page_number")
            if pn is None:
                continue
            fname = ss.get("source_filename") or "?"
            by_source.setdefault(fname, []).append(int(pn))
        for pnums in by_source.values():
            for pn in sorted(set(pnums))[:limit]:
                pages.add(pn)
    return pages


def section_page_priority(outline_sections: list[dict], orch, ns: str) -> list[int]:
    """Flat page order: section 0 pages first, then section 1, etc."""
    seen: set[int] = set()
    order: list[int] = []
    for sec in outline_sections:
        for pn in sorted(pages_for_outline_section(orch, ns, sec)):
            if pn not in seen:
                seen.add(pn)
                order.append(pn)
    return order


def is_section_content_ready(
    user_id: int | str,
    folder: str,
    section_index: int,
    outline_sections: list[dict] | None = None,
) -> bool:
    """True when the student can start/advance — priority figures only.

    Background-deferred figures never block; Pedro teaches from text until they
    are described (orchestrator skips ``_pending_vision`` images).
    """
    if outline_sections and 0 <= section_index < len(outline_sections) and outline_sections[section_index].get('preparation_version') == 1:
        from coast_content_oma import progressive
        return progressive.status_for_section(user_id,folder,outline_sections[section_index])['ready']
    if not is_oma_enabled():
        return True
    if _env_truthy(os.environ.get("OMA_SKIP_IMAGES", "false")):
        return True
    from coast_content_oma.stores import make_namespace

    ns = make_namespace(user_id, folder)
    orch = _content_orchestrator()
    return _priority_vision_complete(orch, ns)


_background_vision_started: set[str] = set()
_consolidation_active: set[str] = set()
_consolidation_pending: set[str] = set()


def kickoff_background_vision_async(
    user_id: int | str,
    folder: str,
    page_order: list[int] | None = None,
) -> bool:
    """Describe deferred figures in a background thread (section-ordered when possible)."""
    if not is_oma_enabled():
        return False
    if _env_truthy(os.environ.get("OMA_SKIP_IMAGES", "false")):
        return False
    if not _env_truthy(os.environ.get("OMA_DESCRIBE_IMAGES", "true")):
        return False

    key = _ingest_key(user_id, folder)
    with _ingest_lock:
        if key in _background_vision_started:
            return False
        _background_vision_started.add(key)

    def _run() -> None:
        try:
            from coast_content_oma.stores import make_namespace

            ns = make_namespace(user_id, folder)
            orch = _content_orchestrator()
            priority_pending, background_pending = _count_pending_vision(orch, ns)
            if priority_pending + background_pending == 0:
                return
            logger.info(
                "background vision starting folder=%s (priority=%d background=%d)",
                folder, priority_pending, background_pending,
            )
            pipeline = _content_ingest_pipeline()
            with _ingest_lock:
                ingesting = {"doc_" + sid for sid in _ingest_source_active}
            result = pipeline.describe_pending_images(
                ns,
                progress=lambda m: logger.info(f"OMA background vision: {m}"),
                page_order=page_order,
                skip_doc_ids=ingesting,
            )
            logger.info(
                "background vision done folder=%s described=%d",
                folder, result.get("described", 0),
            )
        except Exception:
            logger.exception("background vision failed folder=%s", folder)
        finally:
            with _ingest_lock:
                _background_vision_started.discard(key)

    threading.Thread(target=_run, daemon=True).start()
    return True


def _record_retrieved_images(image_chunks) -> None:
    """Track vision-described diagrams offered to Pedro for this chat turn."""
    ctx = _current_capture()
    if ctx is None:
        return
    images_log = ctx["images"]
    seen = {e["id"] for e in images_log}
    for ch in image_chunks or []:
        it = ch.item
        if it.id in seen:
            continue
        ss = it.store_specific or {}
        desc = (it.content or "").strip()
        if not _image_has_description(desc, ss):
            continue
        if any(t.lower() == "decorative" for t in (it.tags or [])):
            continue
        entry = {
            "id": it.id,
            "description": desc[:500],
            "image_type": ss.get("image_type") or (it.tags[0] if it.tags else "figure"),
            "page": ss.get("page_number"),
            "source": ss.get("source_filename"),
            "url": f"{_oma_image_base_url()}/{it.id}",
            "why": getattr(ch, "why", ""),
        }
        images_log.append(entry)
        seen.add(it.id)


# Markdown/HTML image embeds only — not bare ids mentioned in prose.
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)", re.I)
_HTML_IMAGE = re.compile(r"""<img[^>]+src=["']([^"']+)["']""", re.I)


def _embedded_image_refs(text: str) -> set[str]:
    """URLs from ![alt](url) or <img src="..."> in Pedro's reply."""
    refs: set[str] = set()
    for pat in (_MD_IMAGE, _HTML_IMAGE):
        for m in pat.finditer(text or ""):
            ref = (m.group(1) or "").strip()
            if ref:
                refs.add(ref)
    return refs


def _ref_matches_image(ref: str, img: dict) -> bool:
    iid = img.get("id") or ""
    url = img.get("url") or ""
    if not iid:
        return False
    if ref == url or ref == iid:
        return True
    # Relative or absolute path ending with the item id.
    if ref.rstrip("/").endswith(f"/{iid}") or ref.endswith(iid):
        return True
    return False


def _images_used_in_reply(assistant_reply: str, candidates: list[dict]) -> list[dict]:
    """Keep only diagrams Pedro embedded as markdown/HTML images."""
    if not assistant_reply or not candidates:
        return []
    embedded = _embedded_image_refs(assistant_reply)
    if not embedded:
        return []
    used: list[dict] = []
    seen: set[str] = set()
    for img in candidates:
        iid = img.get("id") or ""
        if not iid or iid in seen:
            continue
        if any(_ref_matches_image(ref, img) for ref in embedded):
            used.append(img)
            seen.add(iid)
    return used


def summarize_content_retrieval(assistant_reply: str | None = None, capture: dict | None = None) -> dict:
    """Summary sent to the frontend for DevTools console logging."""
    ctx = capture if capture is not None else _current_capture()
    if ctx is None:
        ctx = {"entries": [], "images": []}
    retrieval_log = ctx["entries"]
    candidates = list(ctx["images"])
    used = _images_used_in_reply(assistant_reply or "", candidates)

    if not retrieval_log:
        return {"primary": None, "entries": [], "images": used, "images_offered": len(candidates)}

    sources = [e["source"] for e in retrieval_log]
    if any(s == "OMA" for s in sources):
        primary = "OMA"
    elif any(s.startswith("RAG") for s in sources):
        primary = "RAG"
    elif any(s == "FALLBACK" for s in sources):
        primary = "FALLBACK"
    else:
        primary = sources[0]

    if candidates:
        logger.info(
            "diagrams: %d used in Pedro's reply (%d offered as candidates)",
            len(used), len(candidates),
        )

    return {
        "primary": primary,
        "entries": list(retrieval_log),
        "images": used,
        "images_offered": len(candidates),
        **ctx.get("oma_meta", {}),
    }


def is_oma_enabled() -> bool:
    return get_rag_provider() in ("oma", "shadow")


def is_student_enabled() -> bool:
    return _env_truthy(os.environ.get("STUDENT_OMA_ENABLED")) or get_rag_provider() in ("oma", "shadow")


# ── Lazy singletons ──────────────────────────────────────────────────

_content_orch = None
_student_orch = None
_student_recorder = None
_content_pipeline = None


def _content_orchestrator():
    global _content_orch
    if _content_orch is None:
        from coast_content_oma.orchestrator import build_orchestrator
        OMA_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        OMA_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        _content_orch = build_orchestrator(OMA_DB_PATH, OMA_IMAGE_DIR)
    return _content_orch


def _student_orchestrator():
    global _student_orch
    if _student_orch is None:
        from coast_content_oma.student.orchestrator import build_student_orchestrator
        OMA_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _student_orch = build_student_orchestrator(OMA_DB_PATH)
    return _student_orch


def _student_recorder_singleton():
    global _student_recorder
    if _student_recorder is None:
        from coast_content_oma.student.recorder import StudentRecorder
        orch = _student_orchestrator()
        _student_recorder = StudentRecorder(
            active_context=orch.active,
            concept_mastery=orch.mastery,
            episodes=orch.episodes,
        )
    return _student_recorder


def _content_ingest_pipeline():
    global _content_pipeline
    if _content_pipeline is None:
        from coast_content_oma.ingestion import IngestionPipeline
        orch = _content_orchestrator()
        # Vision on critical path — batched multi-image keeps latency manageable.
        describe = _env_truthy(os.environ.get("OMA_DESCRIBE_IMAGES", "true"))
        workers = int(os.environ.get("OMA_INGEST_WORKERS", "2" if _ON_RENDER_DISK else "4"))
        skip_images = _env_truthy(os.environ.get("OMA_SKIP_IMAGES", "false"))
        classify_workers = int(os.environ.get(
            "OMA_CLASSIFY_WORKERS", "3" if _ON_RENDER_DISK else "6",
        ))
        vision_workers = int(os.environ.get(
            "OMA_VISION_WORKERS", "2" if _ON_RENDER_DISK else "3",
        ))
        _content_pipeline = IngestionPipeline(
            orch.concept, orch.content, orch.images, OMA_IMAGE_DIR,
            max_workers=max(1, workers),
            describe_images=describe,
            skip_images=skip_images,
            classify_workers=classify_workers,
            vision_workers=vision_workers,
        )
        logger.info(
            "OMA ingest pipeline: workers=%d classify=%d vision=%d describe=%s skip_images=%s",
            workers, classify_workers, vision_workers, describe, skip_images,
        )
    return _content_pipeline


_folder_concept_locks: dict[str, threading.Lock] = {}
_folder_concept_pass_active: set[str] = set()
_background_concept_refine_started: set[str] = set()
_tier_b_pending: set[str] = set()


def _folder_concept_lock(key: str) -> threading.Lock:
    with _ingest_lock:
        lock = _folder_concept_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _folder_concept_locks[key] = lock
        return lock


def _names_for_pdf_source(src: dict) -> set[str]:
    names: set[str] = set()
    if src.get("filename"):
        names.add(src["filename"])
    path = src.get("path")
    if path:
        names.add(Path(path).name)
    sid = src.get("source_id")
    if sid:
        names.add(f"{sid}.pdf")
    return names


def _pages_for_pdf_source(src: dict, content_items: list) -> int:
    names = _names_for_pdf_source(src)
    if not names:
        return 0
    pages: set = set()
    for it in content_items:
        ss = it.store_specific or {}
        if ss.get("source_filename") in names:
            pn = ss.get("page_number")
            pages.add(pn if pn is not None else it.id)
    return len(pages)


def _source_page_gap_ok(have: int, need: int) -> bool:
    if have >= need:
        return True
    if need <= 1:
        return have >= 1
    return have >= need - max(2, int(need * 0.05))


def _folder_pages_and_vision_ready(
    pdf_sources: list[dict],
    orch,
    ns: str,
    *,
    trust_content_indexed: bool = False,
) -> bool:
    if trust_content_indexed and orch.content.count(ns) > 0:
        return True
    if not _priority_vision_complete(orch, ns):
        return False

    items = orch.content.all(ns)
    if not items:
        return False
    gaps = 0
    for src in pdf_sources:
        need = max(1, int(src.get("page_count") or 0))
        if not _source_page_gap_ok(_pages_for_pdf_source(src, items), need):
            gaps += 1
    if gaps:
        total_have = sum(_pages_for_pdf_source(s, items) for s in pdf_sources)
        total_need = sum(max(1, int(s.get("page_count") or 0)) for s in pdf_sources)
        if total_have < max(1, int(total_need * 0.97)):
            return False
    return True


def _all_sources_content_indexed(pdf_sources: list[dict]) -> bool:
    from coast_content_oma import ingest_status as ist

    for src in pdf_sources:
        sid = src.get("source_id") or ""
        st = ist.get_status(sid) if sid else ist.STATUS_PENDING
        if st not in ist.CONTENT_DONE:
            return False
    return True


def _promote_content_indexed_sources_to_ready(pdf_sources: list[dict]) -> None:
    from coast_content_oma import ingest_status as ist

    for src in pdf_sources:
        sid = src.get("source_id")
        if not sid:
            continue
        if ist.get_status(sid) == ist.STATUS_CONTENT_INDEXED:
            ist.set_status(sid, ist.STATUS_READY)


def maybe_finalize_folder_concepts(
    user_id: int | str,
    folder: str,
    pdf_sources: list[dict] | None = None,
) -> bool:
    """Tier A folder concept pass once every PDF is content-indexed. Unlocks roadmap."""
    if not is_oma_enabled():
        return False
    from coast_content_oma import ingest_status as ist
    from coast_content_oma.stores import make_namespace

    sources = pdf_sources or load_folder_pdf_sources(user_id, folder)
    if not sources:
        return False

    key = _ingest_key(user_id, folder)
    with _ingest_lock:
        if _ingest_active.get(key, 0) > 0:
            return False
        if key in _folder_concept_pass_active:
            return False
        _folder_concept_pass_active.add(key)

    try:
        for src in sources:
            sid = src.get("source_id") or ""
            st = ist.get_status(sid) if sid else ist.STATUS_PENDING
            if st in (ist.STATUS_INGESTING, ist.STATUS_PENDING, ist.STATUS_FAILED):
                return False

        if not _all_sources_content_indexed(sources):
            return False

        ns = make_namespace(user_id, folder)
        orch = _content_orchestrator()
        if not _folder_pages_and_vision_ready(
            sources, orch, ns, trust_content_indexed=True,
        ):
            return False

        with _folder_concept_lock(key):
            if orch.concept.count(ns) == 0:
                logger.info(
                    "folder concept Tier A starting folder=%s (%d PDFs)",
                    folder, len(sources),
                )
                pipeline = _content_ingest_pipeline()
                pipeline.run_folder_concept_pass(
                    ns,
                    tier="A",
                    progress=lambda m: logger.info(f"OMA folder concepts: {m}"),
                    on_timing=make_ingest_timing_callback(user_id, folder),
                )
            _promote_content_indexed_sources_to_ready(sources)

        ready = orch.concept.count(ns) > 0 or orch.content.count(ns) <= 2
        if ready:
            kickoff_background_concept_refinement_async(user_id, folder)
        return ready
    finally:
        with _ingest_lock:
            _folder_concept_pass_active.discard(key)


# Refinement writes many concept rows; serialize it across folders so startup
# recovery cannot flood the provider queue and SQLite writer simultaneously.
_concept_refine_semaphore = threading.Semaphore(1)


def kickoff_background_concept_refinement_async(
    user_id: int | str,
    folder: str,
) -> bool:
    """Tier B: LLM definitions + prerequisite graph (upgrades provisional concepts)."""
    if not is_oma_enabled():
        return False
    if _env_truthy(os.environ.get("OMA_SKIP_CONCEPT_REFINE", "false")):
        return False

    key = _ingest_key(user_id, folder)
    with _ingest_lock:
        if key in _background_concept_refine_started:
            _tier_b_pending.add(key)
            return False
        _background_concept_refine_started.add(key)

    def _run() -> None:
        rerun = False
        try:
            with _concept_refine_semaphore:
                from coast_content_oma.stores import make_namespace

                ns = make_namespace(user_id, folder)
                logger.info("folder concept Tier B starting folder=%s", folder)
                record_ingest_timing(user_id, folder, "concepts_tier_b_start")
                pipeline = _content_ingest_pipeline()
                import hashlib
                from coast_content_oma.stores.db import connect_db
                mentions=pipeline.collect_concept_mentions(ns,content_store=pipeline.content,image_store=pipeline.images)
                signature=hashlib.sha256(json.dumps(sorted((name,sorted(ids)) for name,ids in mentions.items())).encode()).hexdigest()
                with connect_db(OMA_DB_PATH) as conn:
                    conn.execute('CREATE TABLE IF NOT EXISTS concept_refinement_receipts (namespace TEXT PRIMARY KEY, fingerprint TEXT NOT NULL)')
                    previous=conn.execute('SELECT fingerprint FROM concept_refinement_receipts WHERE namespace=?',(ns,)).fetchone()
                if previous and previous[0]==signature:
                    return
                result = pipeline.run_folder_concept_pass(
                    ns, tier="B", mentions=mentions,
                    progress=lambda m: logger.info(f"OMA concept refine: {m}"),
                    on_timing=make_ingest_timing_callback(user_id, folder),
                )
                if not mentions or result.get('concepts',0)>0:
                    with connect_db(OMA_DB_PATH) as conn:
                        conn.execute('INSERT OR REPLACE INTO concept_refinement_receipts VALUES (?,?)',(ns,signature))
                record_ingest_timing(
                    user_id, folder, "concepts_tier_b_done",
                    concepts_new=result.get("concepts_new", 0),
                )
                logger.info(
                    "folder concept Tier B done folder=%s (new=%d merged=%d)",
                    folder,
                    result.get("concepts_new", 0),
                    result.get("concepts_merged", 0),
                )
        except Exception:
            logger.exception("folder concept Tier B failed folder=%s", folder)
        finally:
            with _ingest_lock:
                _background_concept_refine_started.discard(key)
                rerun = key in _tier_b_pending
                if rerun:
                    _tier_b_pending.discard(key)
            if rerun:
                kickoff_background_concept_refinement_async(user_id, folder)

    threading.Thread(target=_run, daemon=True).start()
    return True


# ── Upload-time: ingest a single PDF into Content OMA ────────────────

def ingest_pdf_into_oma(
    user_id: int | str,
    folder: str,
    pdf_path: str | Path,
    source_id: Optional[str] = None,
) -> None:
    """Run synchronously (callers should put this in a background thread)."""
    if not is_oma_enabled():
        return
    from coast_content_oma import ingest_status as ist

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.warning(f"OMA ingest skipped — file missing: {pdf_path}")
        return

    source_reserved = False
    if source_id:
        with _ingest_lock:
            if source_id in _ingest_source_active:
                logger.info("OMA ingest skipped — in-process for %s", source_id)
                return
            _ingest_source_active.add(source_id)
            source_reserved = True
        st = ist.get_status(source_id)
        if st in ist.CONTENT_DONE:
            logger.info("OMA ingest skipped — already %s for %s", st, source_id)
            with _ingest_lock:
                _ingest_source_active.discard(source_id)
            return
        if st != ist.STATUS_INGESTING and not ist.try_claim(source_id):
            logger.info("OMA ingest claim failed for %s (status=%s)", source_id, st)
            with _ingest_lock:
                _ingest_source_active.discard(source_id)
            return

    from coast_content_oma.stores import make_namespace
    ns = make_namespace(user_id, folder)
    key = _ingest_key(user_id, folder)
    with _ingest_lock:
        _ingest_active[key] = _ingest_active.get(key, 0) + 1
        if key not in _ingest_timing_t0:
            _ingest_timing_t0[key] = time.time()
    record_ingest_timing(user_id, folder, "ingest_thread_start", pdf=pdf_path.name)
    pipeline = _content_ingest_pipeline()
    try:
        with _ingest_semaphore:
            sid_map: dict[str, str] = {}
            if source_id:
                sid_map[str(pdf_path)] = source_id
                sid_map[pdf_path.name] = source_id
            stats = pipeline.ingest_folder(
                ns, [pdf_path],
                progress=lambda m: logger.info(f"OMA ingest: {m}"),
                on_timing=make_ingest_timing_callback(user_id, folder),
                source_ids=sid_map,
                defer_concepts=True,
            )
        if stats.errors or not stats.content_items:
            raise RuntimeError("; ".join(stats.errors) or "No content was indexed")
        logger.info(f"OMA ingest complete for {pdf_path.name}: {stats.to_dict()}")
        record_ingest_timing(user_id, folder, "ingest_thread_done", pdf=pdf_path.name)
        if source_id:
            from database import FolderSource, SessionLocal
            from coast_content_oma.stores.db import connect_db
            with SessionLocal() as db:
                source = db.query(FolderSource).filter_by(source_id=source_id, user_id=int(user_id)).first()
                label = source.filename if source else None
            if label:
                with connect_db(OMA_DB_PATH) as conn:
                    for table in ('content_items', 'image_items'):
                        conn.execute(f"UPDATE {table} SET store_specific=json_set(COALESCE(store_specific,'{{}}'), '$.source_filename', ?) WHERE namespace=? AND source_doc_id=?", (label, ns, 'doc_' + source_id))
            ist.set_status(source_id, ist.STATUS_CONTENT_INDEXED)
        kickoff_background_vision_async(user_id, folder)
    except Exception as exc:
        logger.exception(f"OMA ingest failed for {pdf_path}")
        if source_id:
            ist.set_status(source_id, ist.STATUS_FAILED, error=str(exc))
    finally:
        should_finalize = False
        with _ingest_lock:
            _ingest_active[key] = max(0, _ingest_active.get(key, 1) - 1)
            should_finalize = _ingest_active.get(key, 0) == 0
            if source_id and source_reserved:
                _ingest_source_active.discard(source_id)
        if should_finalize:
            maybe_finalize_folder_concepts(user_id, folder)


def ingest_pdf_async(
    user_id: int | str,
    folder: str,
    pdf_path: str | Path,
    source_id: Optional[str] = None,
) -> None:
    """Fire-and-forget wrapper for convenience from the upload endpoint."""
    if not is_oma_enabled():
        return
    threading.Thread(
        target=ingest_pdf_into_oma,
        args=(user_id, folder, pdf_path),
        kwargs={"source_id": source_id},
        daemon=True,
    ).start()


def _folder_uploads_dir() -> Path:
    return Path(os.environ.get("FOLDER_UPLOADS_DIR", "./folder_uploads"))


def resolve_source_pdf_path(
    source_id: str,
    filename: str,
    file_path: str | None = None,
) -> Path | None:
    """Resolve on-disk PDF path for a FolderSource row."""
    if file_path:
        p = Path(file_path)
        if p.is_file() and p.suffix.lower() in (".pdf", ".pptx"):
            return p
    p = _folder_uploads_dir() / f"{source_id}{Path(filename).suffix.lower()}"
    if p.is_file() and p.suffix.lower() in (".pdf", ".pptx"):
        return p
    return None


def load_folder_pdf_sources(user_id: int | str, folder: str) -> list[dict]:
    """Load PDF sources from folder_sources for OMA backfill."""
    from database import FolderSource, SessionLocal

    db = SessionLocal()
    try:
        rows = (
            db.query(FolderSource)
            .filter(
                FolderSource.user_id == user_id,
                FolderSource.folder_name == folder,
            )
            .all()
        )
        out: list[dict] = []
        for s in rows:
            path = resolve_source_pdf_path(s.source_id, s.filename, s.file_path)
            if not path:
                continue
            out.append({
                "path": str(path),
                "filename": s.filename or path.name,
                "page_count": int(s.page_count or 0),
                "source_id": s.source_id,
                "oma_ingest_status": getattr(s, "oma_ingest_status", None) or "PENDING",
            })
        return out
    finally:
        db.close()


def get_folder_ingest_progress(user_id: int | str, folder: str) -> dict:
    """Progress snapshot for UI — per-source status + folder readiness."""
    from coast_content_oma import ingest_status as ist
    from coast_content_oma.stores import make_namespace

    from coast_content_oma import progressive
    quick_status = progressive.folder_progress(user_id,folder)
    if quick_status is not None:
        return quick_status

    pdf_sources = load_folder_pdf_sources(user_id, folder)
    ns = make_namespace(user_id, folder)
    orch = _content_orchestrator()

    pending_priority = 0
    pending_background = 0
    for it in orch.images.all(ns):
        if not _image_needs_vision(it):
            continue
        pn = int((it.store_specific or {}).get("page_number") or 0)
        if pn <= _priority_pages_limit():
            pending_priority += 1
        else:
            pending_background += 1
    pending_vision = pending_priority + pending_background

    sources_out = []
    for src in pdf_sources:
        st = (src.get("oma_ingest_status") or ist.STATUS_PENDING).upper()
        sources_out.append({
            "source_id": src.get("source_id"),
            "filename": src.get("filename"),
            "page_count": src.get("page_count"),
            "status": st,
            "ready": st in ist.TERMINAL_OK,
        })

    total_pages = sum(max(1, int(s.get("page_count") or 0)) for s in pdf_sources)

    if is_oma_enabled() and pdf_sources:
        maybe_finalize_folder_concepts(user_id, folder, pdf_sources=pdf_sources)

    ready = False
    if is_oma_enabled() and pdf_sources:
        ready = ensure_oma_ready_for_outline(
            user_id, folder,
            pdf_sources=pdf_sources,
            wait_sec=0,
            allow_sync_ingest=False,
        )

    key = _ingest_key(user_id, folder)
    with _ingest_lock:
        active_threads = _ingest_active.get(key, 0)
        timings = list(_ingest_timings.get(key, []))

    pages_indexed = orch.content.count(ns)
    images_indexed = orch.images.count(ns)
    concepts_count = orch.concept.count(ns)
    ingesting_n = sum(
        1 for s in sources_out if s["status"] == ist.STATUS_INGESTING
    )

    if ready:
        phase = "ready"
    elif pending_priority > 0:
        phase = "vision"
    elif pages_indexed < max(1, int(total_pages * 0.88)) and (active_threads or ingesting_n):
        phase = "classifying"
    elif concepts_count == 0 and pages_indexed >= max(1, int(total_pages * 0.85)):
        phase = "concepts"
    elif active_threads or ingesting_n:
        phase = "classifying" if pages_indexed < max(1, int(total_pages * 0.88)) else "concepts"
    else:
        phase = "classifying"

    return {
        "folder": folder,
        "oma_enabled": is_oma_enabled(),
        "phase": phase,
        "ready_for_roadmap": ready,
        "ingest_threads_active": active_threads,
        "pages_indexed": pages_indexed,
        "pages_expected": total_pages,
        "concepts": concepts_count,
        "images_indexed": images_indexed,
        "pending_vision": pending_vision,
        "pending_priority_vision": pending_priority,
        "background_vision_pending": pending_background,
        "priority_pages": _priority_pages_limit(),
        "timings": timings,
        "sources": sources_out,
    }


_backfill_started: set[str] = set()


def oma_content_page_count(user_id: int | str, folder: str) -> int:
    if not is_oma_enabled():
        return 0
    from coast_content_oma.stores import make_namespace

    ns = make_namespace(user_id, folder)
    return _content_orchestrator().content.count(ns)


def queue_folder_oma_backfill(
    user_id: int | str,
    folder: str,
    *,
    reason: str = "",
) -> bool:
    """Background Content OMA ingest when a folder has PDFs but OMA is empty."""
    if not is_oma_enabled():
        return False
    key = _ingest_key(user_id, folder)
    with _ingest_lock:
        if key in _backfill_started or _ingest_active.get(key, 0) > 0:
            return False
        if oma_content_page_count(user_id, folder) > 0:
            return False
        _backfill_started.add(key)

    def _run() -> None:
        try:
            sources = load_folder_pdf_sources(user_id, folder)
            if not sources:
                logger.info(
                    "OMA backfill skipped — no PDFs folder=%s user=%s",
                    folder, user_id,
                )
                return
            logger.info(
                "OMA backfill starting folder=%s user=%s pdfs=%d %s",
                folder, user_id, len(sources), reason,
            )
            ensure_oma_ready_for_outline(
                user_id,
                folder,
                pdf_sources=sources,
                wait_sec=2400,
            )
        except Exception:
            logger.exception("OMA backfill failed folder=%s user=%s", folder, user_id)
        finally:
            with _ingest_lock:
                _backfill_started.discard(key)

    threading.Thread(target=_run, daemon=True).start()
    return True


def backfill_all_missing_folders() -> dict:
    """Queue OMA ingest for every (user, folder) that has PDFs but empty OMA."""
    if not is_oma_enabled():
        return {"queued": 0, "folders": []}

    from database import FolderSource, SessionLocal

    db = SessionLocal()
    try:
        rows = db.query(FolderSource).all()
    finally:
        db.close()

    seen: set[tuple[int, str]] = set()
    queued: list[dict] = []
    for s in rows:
        if not resolve_source_pdf_path(s.source_id, s.filename, s.file_path):
            continue
        pair = (int(s.user_id), s.folder_name)
        if pair in seen:
            continue
        seen.add(pair)
        if oma_content_page_count(pair[0], pair[1]) > 0:
            continue
        if queue_folder_oma_backfill(pair[0], pair[1], reason="bulk backfill"):
            queued.append({"user_id": pair[0], "folder": pair[1]})
    return {"queued": len(queued), "folders": queued}


def ensure_oma_ready_for_outline(
    user_id: int | str,
    folder: str,
    *,
    expected_pages: int = 0,
    pdf_sources: list[dict] | None = None,
    wait_sec: float = 600,
    poll_sec: float = 2,
    allow_sync_ingest: bool | None = None,
) -> bool:
    """Block until Content OMA has indexed uploaded PDFs (for OMA-driven outlines).

    Waits for background upload ingests to finish. On Render, never runs heavy
    sync ingest inside an HTTP request — that OOMs the web worker.
    """
    if not is_oma_enabled():
        return False
    from coast_content_oma.stores import make_namespace

    if allow_sync_ingest is None:
        allow_sync_ingest = not _ON_RENDER_DISK
    # Poll-only callers (progress UI) must never block on sync ingest.
    if wait_sec <= 0:
        allow_sync_ingest = False
    if _ON_RENDER_DISK:
        wait_sec = min(wait_sec, float(os.environ.get("OMA_OUTLINE_WAIT_SEC", "180")))

    ns = make_namespace(user_id, folder)
    orch = _content_orchestrator()
    key = _ingest_key(user_id, folder)
    if expected_pages:
        target = expected_pages
    elif pdf_sources:
        target = sum(max(1, int(s.get("page_count") or 0)) for s in pdf_sources)
    else:
        target = 1
    target = max(1, target)

    content_cache: list | None = None
    queued_sids: set[str] = set()

    def _refresh_content_items() -> list:
        nonlocal content_cache
        content_cache = orch.content.all(ns)
        return content_cache

    def _content_items():
        if content_cache is None:
            return _refresh_content_items()
        return content_cache

    def _source_gap_ok(have: int, need: int) -> bool:
        """PDF page_count includes blank pages; OMA skips blanks during ingest."""
        if have >= need:
            return True
        if need <= 1:
            return have >= 1
        return have >= need - max(2, int(need * 0.05))

    def _count() -> int:
        return orch.content.count(ns)

    def _active() -> int:
        with _ingest_lock:
            return _ingest_active.get(key, 0)

    def _names_for_source(src: dict) -> set[str]:
        names: set[str] = set()
        if src.get("filename"):
            names.add(src["filename"])
        path = src.get("path")
        if path:
            names.add(Path(path).name)
        sid = src.get("source_id")
        if sid:
            names.add(f"{sid}.pdf")
        return names

    def _pages_for_source(src: dict) -> int:
        """Distinct indexed pages for a source (ignores duplicate ingest rows)."""
        names = _names_for_source(src)
        if not names:
            return 0
        pages: set = set()
        for it in _content_items():
            ss = it.store_specific or {}
            if ss.get("source_filename") in names:
                pn = ss.get("page_number")
                pages.add(pn if pn is not None else it.id)
        return len(pages)

    def _vision_complete() -> bool:
        return _priority_vision_complete(orch, ns)

    def _concepts_ready() -> bool:
        n_content = _count()
        if n_content == 0:
            return False
        if orch.concept.count(ns) > 0:
            return True
        return n_content <= 2

    def _pages_all_ok() -> bool:
        if not pdf_sources:
            return _count() >= target
        from coast_content_oma import ingest_status as ist

        if _active() == 0:
            statuses = [
                ist.get_status(s["source_id"]) if s.get("source_id") else ist.STATUS_PENDING
                for s in pdf_sources
            ]
            if all(st in ist.CONTENT_DONE for st in statuses) and _count() > 0:
                return True

        gaps = 0
        for src in pdf_sources:
            need = max(1, int(src.get("page_count") or 0))
            if not _source_gap_ok(_pages_for_source(src), need):
                gaps += 1
        if gaps == 0:
            return True
        total_have = sum(_pages_for_source(s) for s in pdf_sources)
        total_need = sum(max(1, int(s.get("page_count") or 0)) for s in pdf_sources)
        return total_have >= max(1, int(total_need * 0.97))

    def _kickoff_background_ingests() -> int:
        """Queue async ingest for PDFs still missing — never blocks the web worker."""
        from coast_content_oma import ingest_status as ist

        if not pdf_sources:
            return 0
        with _ingest_lock:
            if _ingest_active.get(key, 0) > 0:
                return 0
        queued = 0
        for src in pdf_sources:
            path = src.get("path")
            need = max(1, int(src.get("page_count") or 0))
            sid = src.get("source_id") or ""
            if sid in queued_sids:
                continue
            if not path:
                continue
            with _ingest_lock:
                if sid and sid in _ingest_source_active:
                    continue
            st = ist.get_status(sid) if sid else ist.STATUS_PENDING
            if st in ist.TERMINAL_OK and _source_gap_ok(_pages_for_source(src), need):
                continue
            if st == ist.STATUS_INGESTING:
                continue
            if st == ist.STATUS_CONTENT_INDEXED:
                continue
            p = Path(path)
            if not p.is_file():
                continue
            if _source_gap_ok(_pages_for_source(src), need) and _vision_complete() and _concepts_ready():
                continue
            queued_sids.add(sid or path)
            ingest_pdf_async(user_id, folder, p, source_id=sid or None)
            queued += 1
        if queued:
            logger.info(
                "outline: queued background OMA ingest for %d PDF(s) folder=%s",
                queued, folder,
            )
        return queued

    def _is_ready(*, refresh: bool = False) -> bool:
        from coast_content_oma import ingest_status as ist

        if refresh:
            _refresh_content_items()

        if pdf_sources and _active() == 0:
            maybe_finalize_folder_concepts(user_id, folder, pdf_sources=pdf_sources)
            if refresh:
                _refresh_content_items()

        if pdf_sources:
            statuses = [
                ist.get_status(s["source_id"]) if s.get("source_id") else ist.STATUS_PENDING
                for s in pdf_sources
            ]
            if all(st in ist.TERMINAL_OK for st in statuses):
                if _vision_complete() and _concepts_ready():
                    return True

        if not _pages_all_ok():
            return False
        if not _vision_complete():
            return False
        if not _concepts_ready():
            return False

        if pdf_sources:
            statuses = [
                (s.get("oma_ingest_status") or ist.STATUS_PENDING).upper()
                for s in pdf_sources
            ]
            if all(st in ist.TERMINAL_OK for st in statuses):
                return True
            if _active() > 0:
                return False
            return True

        return _active() == 0

    if wait_sec == 600 and target > 80 and allow_sync_ingest:
        wait_sec = min(2400, 120 + target * 3)

    if _is_ready(refresh=True):
        logger.info("outline: Content OMA already ready folder=%s (%d pages)", folder, _count())
        kickoff_background_vision_async(user_id, folder)
        return True

    _kickoff_background_ingests()

    # Log why we're waiting when total page count already looks complete.
    if pdf_sources and _count() >= target and _active() == 0:
        _refresh_content_items()
        for src in pdf_sources:
            need = max(1, int(src.get("page_count") or 0))
            have = _pages_for_source(src)
            if have < need:
                logger.info(
                    "outline: per-source gap folder=%s names=%s have=%d need=%d",
                    folder, sorted(_names_for_source(src)), have, need,
                )

    logger.info(
        "outline: waiting for Content OMA folder=%s (pages %d/%d, %d ingest threads)...",
        folder, _count(), target, _active(),
    )
    deadline = time.time() + wait_sec
    while time.time() < deadline:
        if _is_ready(refresh=_active() == 0):
            logger.info("outline: Content OMA ready folder=%s (%d pages)", folder, _count())
            kickoff_background_vision_async(user_id, folder)
            return True
        if _active() == 0:
            _kickoff_background_ingests()
        time.sleep(poll_sec)

    if not allow_sync_ingest:
        logger.info(
            "outline: Content OMA not ready after %.0fs (background ingest continues) folder=%s",
            wait_sec, folder,
        )
        return _is_ready(refresh=True)

    # Local dev only — sync ingest (too heavy for Render web workers).
    _refresh_content_items()
    to_ingest: list[Path] = []
    sid_map: dict[str, str] = {}
    for src in pdf_sources or []:
        path = src.get("path")
        need = max(1, int(src.get("page_count") or 0))
        if not path:
            continue
        p = Path(path)
        if not p.is_file():
            continue
        have = _pages_for_source(src)
        if not _source_gap_ok(have, need):
            to_ingest.append(p)
            sid = src.get("source_id")
            if sid:
                sid_map[str(p)] = sid
                sid_map[p.name] = sid

    if to_ingest:
        with _ingest_lock:
            if _ingest_active.get(key, 0) > 0:
                logger.info(
                    "outline: skip sync ingest — background ingest active folder=%s",
                    folder,
                )
                return _is_ready(refresh=True)
        logger.info(
            "outline: sync ingesting %d PDF(s) for folder=%s (have %d/%d pages)",
            len(to_ingest), folder, _count(), target,
        )
        try:
            from coast_content_oma import ingest_status as ist

            for src in pdf_sources or []:
                path = src.get("path")
                if not path:
                    continue
                p = Path(path)
                if p not in to_ingest and str(p) not in {str(x) for x in to_ingest}:
                    continue
                sid = src.get("source_id")
                st = ist.get_status(sid) if sid else ist.STATUS_PENDING
                if sid and st in ist.TERMINAL_OK:
                    continue
                ingest_pdf_into_oma(user_id, folder, p, source_id=sid)
        except Exception:
            logger.exception("OMA sync ingest failed during outline for folder=%s", folder)
        _refresh_content_items()

    ready = _is_ready(refresh=True)
    logger.info(
        "outline: Content OMA %s folder=%s (%d/%d pages)",
        "ready" if ready else "not ready",
        folder, _count(), target,
    )
    if ready:
        kickoff_background_vision_async(user_id, folder)
    return ready


def kickoff_folder_oma_ingest(user_id: int | str, folder: str) -> int:
    """Queue background OMA ingest for PDFs in this folder that are not indexed yet."""
    if not is_oma_enabled():
        return 0
    pdf_sources = load_folder_pdf_sources(user_id, folder)
    if not pdf_sources:
        return 0
    from coast_content_oma.stores import make_namespace

    ns = make_namespace(user_id, folder)
    orch = _content_orchestrator()
    items = orch.content.all(ns)
    from coast_content_oma import ingest_status as ist

    kicked: set[str] = set()
    queued = 0
    folder_key = _ingest_key(user_id, folder)
    with _ingest_lock:
        if _ingest_active.get(folder_key, 0) > 0:
            return 0
    for src in pdf_sources:
        path = src.get("path")
        sid = src.get("source_id") or ""
        kick_key = sid or path or ""
        if kick_key in kicked or not path:
            continue
        with _ingest_lock:
            if sid and sid in _ingest_source_active:
                continue
        st = ist.get_status(sid) if sid else ist.STATUS_PENDING
        if st in ist.TERMINAL_OK:
            continue
        if st == ist.STATUS_CONTENT_INDEXED:
            maybe_finalize_folder_concepts(user_id, folder, pdf_sources=pdf_sources)
            continue
        if st == ist.STATUS_INGESTING:
            continue
        p = Path(path)
        if not p.is_file():
            continue
        names = {src.get("filename") or p.name, p.name}
        if sid:
            names.add(f"{sid}.pdf")
        have = sum(
            1 for it in items
            if (it.store_specific or {}).get("source_filename") in names
        )
        need = max(1, int(src.get("page_count") or 0))
        if have >= need or (need > 1 and have >= need - max(2, int(need * 0.05))):
            continue
        kicked.add(kick_key)
        ingest_pdf_async(user_id, folder, p, source_id=sid or None)
        queued += 1
    if queued:
        logger.info("kickoff: queued OMA ingest for %d PDF(s) folder=%s", queued, folder)
    return queued


# ── Chat-time: folder context block ──────────────────────────────────

def _oma_image_base_url() -> str:
    api_base = (os.environ.get("API_BASE_URL") or "http://localhost:8000").rstrip("/")
    return f"{api_base}/api/oma/images"


def get_oma_image_path(item_id: str) -> Path | None:
    """Resolve on-disk PNG path for a Content OMA image item."""
    try:
        item = _content_orchestrator().images.get(item_id)
        if not item:
            return None
        fp = (item.store_specific or {}).get("file_path")
        if not fp:
            return None
        path = Path(fp)
        return path if path.is_file() else None
    except Exception:
        logger.exception("OMA image lookup failed for %s", item_id)
        return None


def log_content_source(
    source: str,
    *,
    context_type: str,
    folder: str,
    user_id: int | str | None = None,
    chars: int = 0,
    detail: str = "",
) -> None:
    """Print which retrieval backend supplied Pedro's course material."""
    entry = {
        "source": source,
        "context_type": context_type,
        "folder": folder,
        "chars": chars,
    }
    if user_id is not None:
        entry["user_id"] = user_id
    if detail:
        entry["detail"] = detail
    ctx = _current_capture()
    if ctx is not None:
        ctx["entries"].append(entry)

    parts = [
        f"CONTENT SOURCE: {source}",
        f"context={context_type}",
        f"folder={folder}",
    ]
    if user_id is not None:
        parts.append(f"user={user_id}")
    parts.append(f"{chars} chars")
    if detail:
        parts.append(detail)
    logger.info(" | ".join(parts))


def _set_oma_meta(**fields) -> None:
    ctx = _current_capture()
    if ctx is not None:
        ctx.setdefault("oma_meta", {}).update(fields)


def _wait_for_oma_if_indexing(
    user_id: int | str,
    folder: str,
    *,
    max_wait: float = 45.0,
    poll_sec: float = 2.0,
) -> bool:
    """If OMA ingest is running for this folder, wait briefly for pages to appear."""
    if oma_content_page_count(user_id, folder) > 0:
        return True
    key = _ingest_key(user_id, folder)
    with _ingest_lock:
        active = _ingest_active.get(key, 0) > 0 or key in _backfill_started
    if not active:
        return False
    logger.info("Waiting for Content OMA ingest folder=%s (up to %.0fs)", folder, max_wait)
    deadline = time.time() + max_wait
    while time.time() < deadline:
        if oma_content_page_count(user_id, folder) > 0:
            return True
        with _ingest_lock:
            if _ingest_active.get(key, 0) <= 0 and key not in _backfill_started:
                break
        time.sleep(poll_sec)
    return oma_content_page_count(user_id, folder) > 0


def resolve_folder_content(
    user_id: int | str,
    folder: str,
    query: str,
    context_type: str = "folder",
    max_chars: int = 14000,
    *,
    max_content: int = 8,
    max_images: int = 4,
) -> tuple[str, str, list[str]]:
    """Try Content OMA, then flat RAG. Returns (block, source_label, concept_ids).

    source_label is one of: OMA, RAG, none
    """
    import rag

    concept_ids: list[str] = []
    oma_pages = oma_content_page_count(user_id, folder) if is_oma_enabled() else 0
    _set_oma_meta(
        oma_enabled=is_oma_enabled(),
        rag_provider=get_rag_provider(),
        student_oma_enabled=is_student_enabled(),
        oma_pages=oma_pages,
    )

    if is_oma_enabled():
        block, concept_ids = get_folder_context(
            user_id, folder, query,
            max_chars=max_chars,
            max_content=max_content,
            max_images=max_images,
        )
        if block:
            log_content_source(
                "OMA",
                context_type=context_type,
                folder=folder,
                user_id=user_id,
                chars=len(block),
                detail=f"{len(concept_ids)} concepts",
            )
            _set_oma_meta(oma_pages=oma_content_page_count(user_id, folder))
            return block, "OMA", concept_ids

        if _wait_for_oma_if_indexing(user_id, folder):
            block, concept_ids = get_folder_context(
                user_id, folder, query,
                max_chars=max_chars,
                max_content=max_content,
                max_images=max_images,
            )
            if block:
                log_content_source(
                    "OMA",
                    context_type=context_type,
                    folder=folder,
                    user_id=user_id,
                    chars=len(block),
                    detail=f"{len(concept_ids)} concepts (after ingest wait)",
                )
                _set_oma_meta(oma_pages=oma_content_page_count(user_id, folder))
                return block, "OMA", concept_ids

    block = rag.build_folder_context(user_id, folder, query, max_chars=max_chars)
    if block:
        label = "RAG (OMA empty)" if is_oma_enabled() else "RAG"
        log_content_source(
            label,
            context_type=context_type,
            folder=folder,
            user_id=user_id,
            chars=len(block),
            detail=f"oma_pages={oma_content_page_count(user_id, folder)}" if is_oma_enabled() else "",
        )
        return block, "RAG", []

    log_content_source(
        "none",
        context_type=context_type,
        folder=folder,
        user_id=user_id,
        detail="no material retrieved",
    )
    return "", "none", []


def get_folder_context(
    user_id: int | str,
    folder: str,
    query: str,
    max_chars: int = 14000,
    *,
    max_content: int = 8,
    max_images: int = 4,
) -> tuple[str, list[str]]:
    """Returns (context_block, concept_ids_surfaced).

    concept_ids is the list of concept ids OMA pulled — useful so the
    caller can pass them to record_chat_episode() for accurate mastery
    updates."""
    if not is_oma_enabled():
        return "", []
    try:
        from coast_content_oma.stores import make_namespace
        ns = make_namespace(user_id, folder)
        orch = _content_orchestrator()
        result = orch.retrieve(
            ns, query, max_content=max_content, max_images=max_images,
        )
        _record_retrieved_images(result.images)
        block = result.to_prompt_block(
            max_chars=max_chars,
            image_base_url=_oma_image_base_url(),
        )
        concept_ids = [c.id for c in result.concept_candidates]
        return block, concept_ids
    except Exception:
        logger.exception("OMA folder context retrieval failed")
        return "", []


# ── Outline generation: structured course index ───────────────────────

def build_outline_context(
    user_id: int | str,
    folder: str,
    *,
    source_meta: list[dict] | None = None,
    max_chars: int = 80_000,
) -> str:
    """Build a structured course index from Content OMA for outline generation.

    source_meta: optional list of {source_id, title, page_count, filename}
    from FolderSource rows — used for human-readable titles and ordering.
    Returns empty string if OMA is disabled or the folder has no ingested content.
    """
    if not is_oma_enabled():
        return ""
    try:
        from coast_content_oma.stores import make_namespace
        ns = make_namespace(user_id, folder)
        orch = _content_orchestrator()
        content_items = orch.content.all(ns)
        if not content_items:
            return ""

        # filename stem → display title
        title_by_file: dict[str, str] = {}
        order_by_file: dict[str, int] = {}
        if source_meta:
            for i, sm in enumerate(source_meta):
                sid = sm.get("source_id") or ""
                fname = sm.get("filename") or f"{sid}.pdf"
                title = sm.get("title") or sid or fname
                title_by_file[fname] = title
                title_by_file[f"{sid}.pdf"] = title
                order_by_file[fname] = i
                order_by_file[f"{sid}.pdf"] = i

        def _file_title(fname: str) -> str:
            return title_by_file.get(fname, fname.replace(".pdf", "").replace("src_", ""))

        def _concept_label(cid: str) -> str:
            c = orch.concept.get(cid)
            if not c:
                return cid
            ss = c.store_specific or {}
            return ss.get("name") or (c.content or cid)[:60]

        # Group pages by source PDF.
        by_source: dict[str, list] = {}
        for it in content_items:
            ss = it.store_specific or {}
            fname = ss.get("source_filename") or it.source_doc_id or "unknown"
            by_source.setdefault(fname, []).append(it)

        for pages in by_source.values():
            pages.sort(key=lambda x: (x.store_specific or {}).get("page_number") or 0)

        sorted_sources = sorted(
            by_source.keys(),
            key=lambda f: (order_by_file.get(f, 999), f),
        )

        # Concept map with prerequisite names.
        concepts = orch.concept.all_concepts(ns)
        concept_by_id = {c.id: c for c in concepts}

        def _prereq_names(c) -> list[str]:
            pre_ids = (c.store_specific or {}).get("prerequisite_concept_ids") or []
            return [_concept_label(pid) for pid in pre_ids if pid in concept_by_id]

        foundational: list[str] = []
        dependent: list[str] = []
        for c in sorted(concepts, key=lambda x: ((x.store_specific or {}).get("name") or x.content or "").lower()):
            ss = c.store_specific or {}
            name = ss.get("name") or (c.content or "?")[:80]
            definition = (ss.get("definition") or c.content or "").strip().replace("\n", " ")
            if len(definition) > 220:
                definition = definition[:220] + "…"
            pre = _prereq_names(c)
            srcs = ss.get("lecture_sources") or []
            src_hint = ""
            if srcs:
                src_hint = f" [from {_file_title(srcs[0] + '.pdf' if not srcs[0].endswith('.pdf') else srcs[0])}]"
            line = f"- {name}{src_hint}: {definition}"
            if pre:
                line += f"  (prerequisites: {', '.join(pre)})"
            if pre:
                dependent.append(line)
            else:
                foundational.append(line)

        parts: list[str] = [
            "--- COURSE INDEX (Content OMA — structured from uploaded lectures) ---",
            f"Folder: {folder} | {len(content_items)} pages indexed | {len(concepts)} canonical concepts",
            "",
            "Use this index to design the outline. Each source below lists every page with its "
            "section title, content types, and concepts. Split large sources (10+ pages) into "
            "2–3 lesson sections. Order sections by concept prerequisites (foundational first).",
            "In source_notebooks, use the human-readable source titles shown in quotes below.",
            "",
        ]

        total_pages = len(content_items)
        budget = max_chars - 4000  # reserve for concept map + header
        per_source_budget = max(800, budget // max(len(sorted_sources), 1))

        for fname in sorted_sources:
            pages = by_source[fname]
            title = _file_title(fname)
            header = f'## Source: "{title}" ({fname}, {len(pages)} pages)\n'
            if len(header) > per_source_budget:
                parts.append(header + "  (page list truncated)\n")
                continue

            lines = [header]
            page_budget = per_source_budget - len(header)
            per_page = max(120, page_budget // max(len(pages), 1))

            for it in pages:
                ss = it.store_specific or {}
                page = ss.get("page_number", "?")
                sec = ss.get("section_title") or "Untitled section"
                types = ss.get("content_types") or it.tags or []
                type_str = ", ".join(types[:4]) if types else "narrative"
                raw_concepts = ss.get("concept_mentions_raw") or []
                canon = [_concept_label(e) for e in (it.entities or [])[:5]]
                concept_str = ", ".join(raw_concepts[:4] or canon[:4]) or "—"
                summary = (ss.get("summary") or "").strip().replace("\n", " ")
                line = f"  p.{page} — {sec} [{type_str}] | concepts: {concept_str}"
                if summary and len(line) < per_page - 20:
                    room = per_page - len(line) - 12
                    if room > 40:
                        line += f"\n      {summary[:room]}"
                if len("\n".join(lines)) + len(line) > page_budget:
                    lines.append(f"  … ({len(pages) - len(lines) + 1} more pages in this source)")
                    break
                lines.append(line)

            parts.append("\n".join(lines) + "\n")

        parts.append("## Concept map (canonical — respect prerequisite order)\n")
        if foundational:
            parts.append("Foundational:\n" + "\n".join(foundational[:40]))
        if dependent:
            parts.append("\nBuilds on prior concepts:\n" + "\n".join(dependent[:60]))
        parts.append("\n--- END COURSE INDEX ---")

        body = "\n".join(parts)
        if len(body) > max_chars:
            body = body[:max_chars] + "\n[... truncated ...]\n--- END COURSE INDEX ---"

        log_content_source(
            "OMA",
            context_type="outline",
            folder=folder,
            user_id=user_id,
            chars=len(body),
            detail=f"{len(sorted_sources)} sources, {total_pages} pages, {len(concepts)} concepts",
        )
        return body
    except Exception:
        logger.exception("OMA outline context build failed")
        return ""


# ── Chat-time: student profile block ─────────────────────────────────

def build_student_analysis(user_id: int | str, folder: str) -> dict:
    """Structured analysis payload for the student Analysis screen."""
    if not is_student_enabled():
        return {"error": "Student OMA not enabled"}
    from coast_content_oma.stores import make_namespace
    from coast_content_oma.student.stores import course_namespace

    student_orch = _student_orchestrator()
    content_orch = _content_orchestrator()
    profile = student_orch.build_profile(user_id, folder)
    profile["progress_ledger"] = _load_progress_ledger(user_id, folder)
    profile["section_mistakes"] = _load_section_mistakes(user_id, folder)
    profile["struggling_topics"] = _load_struggling_topics(user_id, folder)

    content_ns = make_namespace(user_id, folder)
    course_ns = course_namespace(user_id, folder)
    concepts = {c.id: c for c in content_orch.concept.all_concepts(content_ns)}

    nodes: list[dict] = []
    touched: dict[str, dict] = {}
    struggling_ids = {t["concept_id"] for t in profile["struggling_topics"]}
    for it in student_orch.mastery.all(course_ns):
        ss = it.store_specific or {}
        cid = ss.get("concept_id")
        if not cid:
            continue
        score = float(ss.get("mastery_score", 0.5))
        if cid in struggling_ids:
            status = "struggling"
        elif score >= 0.75:
            status = "mastered"
        else:
            status = "developing"
        touched[cid] = {
            "id": cid,
            "name": ss.get("concept_name") or cid,
            "score": round(score, 2),
            "status": status,
            "successes": ss.get("successes", 0),
            "struggles": ss.get("struggles", 0),
        }

    for cid, node in touched.items():
        c = concepts.get(cid)
        pre_ids = (c.store_specific or {}).get("prerequisite_concept_ids") or [] if c else []
        node["prerequisite_ids"] = [p for p in pre_ids if p in touched]
        nodes.append(node)

    nodes.sort(key=lambda n: (-n["score"], (n.get("name") or "").lower()))

    edges: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for n in nodes:
        for pid in n.get("prerequisite_ids") or []:
            key = (pid, n["id"])
            if key not in seen:
                edges.append({"source": pid, "target": n["id"], "type": "prerequisite"})
                seen.add(key)

    ac = profile.get("active_context") or {}
    timeline = [
        {"text": t.get("text"), "when": t.get("as_of")}
        for t in (ac.get("recent_topics") or [])[:10]
        if t.get("text")
    ]

    return {
        "folder": folder,
        "profile": profile,
        "graph": {"nodes": nodes, "edges": edges},
        "timeline": timeline,
    }


def build_student_global_summary(user_id: int | str) -> dict:
    """Cross-course summary for the dashboard Analysis card."""
    if not is_student_enabled():
        return {"error": "Student OMA not enabled", "courses": []}
    orch = _student_orchestrator()
    profile = orch.build_global_profile(user_id)
    courses = []
    for c in profile.get("courses") or []:
        mo = c.get("mastery_overview") or {}
        acc = c.get("accomplishments") or {}
        courses.append({
            "folder": c.get("folder"),
            "n_concepts": mo.get("n_concepts", 0),
            "avg_mastery": round(float(mo.get("avg_mastery") or 0), 2),
            "n_mastered": mo.get("n_mastered", 0),
            "n_struggling": mo.get("n_struggling", 0),
            "current_focus": c.get("current_focus") or "",
            "n_episodes": (c.get("recent_window") or {}).get("n_episodes", 0),
            "highlights": acc.get("narrative_lines") or [],
            "sections_completed": acc.get("sections_completed") or [],
        })
    return {
        "courses": courses,
        "identity_traits": profile.get("identity_traits") or [],
    }


def _mastery_status(score: float) -> str:
    if score >= 0.75:
        return "mastered"
    if score <= 0.35:
        return "struggling"
    return "developing"


def build_student_global_mindmap(user_id: int | str) -> dict:
    """Aggregate knowledge graph across every course the student has touched."""
    if not is_student_enabled():
        return {"error": "Student OMA not enabled", "graph": {"nodes": [], "edges": []}, "courses": []}

    from collections import defaultdict
    from coast_content_oma.stores import make_namespace
    from coast_content_oma.student.stores import list_course_namespaces, parse_course_namespace

    student_orch = _student_orchestrator()
    content_orch = _content_orchestrator()
    db_path = student_orch.mastery.db_path

    nodes: list[dict] = []
    edges: list[dict] = []
    seen_edges: set[tuple[str, str, str]] = set()
    courses_meta: list[dict] = []
    name_index: dict[str, list[str]] = defaultdict(list)

    for course_ns in list_course_namespaces(db_path, user_id):
        _, folder_slug = parse_course_namespace(course_ns)
        if not folder_slug:
            continue
        folder = folder_slug.replace("_", " ")
        content_ns = make_namespace(user_id, folder_slug)
        concepts = {c.id: c for c in content_orch.concept.all_concepts(content_ns)}

        touched: dict[str, str] = {}
        for it in student_orch.mastery.all(course_ns):
            ss = it.store_specific or {}
            cid = ss.get("concept_id")
            if not cid:
                continue
            node_id = f"{folder_slug}::{cid}"
            score = float(ss.get("mastery_score", 0.5))
            name = ss.get("concept_name") or cid
            touched[cid] = node_id
            nodes.append({
                "id": node_id,
                "concept_id": cid,
                "folder": folder,
                "folder_slug": folder_slug,
                "name": name,
                "score": round(score, 2),
                "status": _mastery_status(score),
                "successes": ss.get("successes", 0),
                "struggles": ss.get("struggles", 0),
            })
            name_index[name.lower().strip()].append(node_id)

        if not touched:
            continue

        courses_meta.append({
            "folder": folder,
            "folder_slug": folder_slug,
            "n_concepts": len(touched),
        })

        for cid, node_id in touched.items():
            c = concepts.get(cid)
            if not c:
                continue
            ss = c.store_specific or {}
            for pid in ss.get("prerequisite_concept_ids") or []:
                if pid not in touched:
                    continue
                key = (touched[pid], node_id, "prerequisite")
                if key not in seen_edges:
                    edges.append({"source": touched[pid], "target": node_id, "type": "prerequisite"})
                    seen_edges.add(key)
            for rid in ss.get("related_concept_ids") or []:
                if rid not in touched:
                    continue
                src, tgt = sorted([touched[rid], node_id])
                key = (src, tgt, "related")
                if key not in seen_edges:
                    edges.append({"source": src, "target": tgt, "type": "related"})
                    seen_edges.add(key)

    # Cross-course links when the same concept name appears in multiple folders.
    for _name, ids in name_index.items():
        if len(ids) < 2:
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                src, tgt = sorted([ids[i], ids[j]])
                if src.split("::")[0] == tgt.split("::")[0]:
                    continue
                key = (src, tgt, "cross_course")
                if key not in seen_edges:
                    edges.append({"source": src, "target": tgt, "type": "cross_course"})
                    seen_edges.add(key)

    summary = build_student_global_summary(user_id)
    return {
        "graph": {"nodes": nodes, "edges": edges},
        "courses": courses_meta,
        "stats": {
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "n_courses": len(courses_meta),
        },
        "summary": summary,
    }


def get_global_student_profile_block(
    user_id: int | str,
    max_chars: int = 1600,
    query: str = "",
) -> str:
    """Cross-course profile for global Pedro chat — aggregates every
    folder/course the student has history in."""
    if not is_student_enabled():
        return ""
    try:
        orch = _student_orchestrator()
        profile = orch.build_global_profile(user_id)
        from coast_content_oma.student.recall import recall_memories
        profile["cross_course_memories"] = recall_memories(orch.episodes.db_path, user_id, query)
        if not profile.get("courses") and not profile["cross_course_memories"]:
            return ""
        return orch.to_global_prompt_block(profile, max_chars=max_chars)
    except Exception:
        logger.exception("Global student profile build failed")
        return ""


def _prior_learning(user_id, folder, concept_refs) -> list[dict]:
    try:
        from coast_content_oma.student.bridges import related_prior_learning
        return related_prior_learning(_student_orchestrator(), user_id, folder, concept_refs)
    except Exception:
        logger.exception("prior-learning links failed")
        return []


def _refs_for_ids(user_id, folder, concept_ids) -> list[dict]:
    if not concept_ids:
        return []
    items = _content_orchestrator().concept.get_many(list(concept_ids))
    return [{"concept_id": it.id, "concept_name": (it.store_specific or {}).get("name") or it.content[:60]}
            for it in items]


def get_student_profile_block(
    user_id: int | str,
    folder: str,
    current_concept_ids: Optional[list[str]] = None,
    max_chars: int = 1200,
    query: str = "",
) -> str:
    """Returns the personalized profile block for Pedro. Empty string
    if the student has no recorded history yet or the system is disabled."""
    if not is_student_enabled():
        return ""
    try:
        orch = _student_orchestrator()
        profile = orch.build_profile(user_id, folder, current_concept_ids=current_concept_ids)
        profile['current_query'] = query
        from coast_content_oma.student.recall import recall_memories
        from coast_content_oma.student.stores import course_namespace
        concept_names = " ".join(m.get("name", "") for m in profile.get("focused_mastery") or [])
        profile["cross_course_memories"] = recall_memories(orch.episodes.db_path, user_id,
            f"{query} {concept_names}", exclude_namespace=course_namespace(user_id, folder))
        profile["progress_ledger"] = _load_progress_ledger(user_id, folder)
        profile["section_mistakes"] = _load_section_mistakes(user_id, folder)
        profile["struggling_topics"] = _load_struggling_topics(user_id, folder)
        profile["prior_learning"] = _prior_learning(user_id, folder, _refs_for_ids(user_id, folder, current_concept_ids))
        has_mastery = bool((profile.get("mastery_overview") or {}).get("n_concepts"))
        has_progress = bool(profile.get("progress_ledger", {}).get("completed_sections"))
        has_mistakes = bool(profile.get("section_mistakes"))
        has_identity = bool(profile.get("identity_traits"))
        has_golden = bool(profile.get("golden_moments"))
        if not (has_mastery or has_progress or has_mistakes or has_identity or has_golden
                or profile["cross_course_memories"] or profile["prior_learning"]):
            return ""
        return orch.to_prompt_block(profile, max_chars=max_chars)
    except Exception:
        logger.exception("Student profile build failed")
        return ""


def get_course_intro_student_block(
    user_id: int | str,
    folder: str,
    sections: list,
    max_chars: int = 1100,
) -> str:
    """Cross-course Student OMA + onboarding traits for a lesson's first-section intro."""
    if not is_student_enabled():
        return ""
    try:
        lines: list[str] = []
        seen: set[str] = set()

        try:
            import onboarding as onboarding_mod
            for t in onboarding_mod.get_saved_onboarding_traits(user_id):
                desc = (t.get("description") or "").strip()
                if desc and desc not in seen:
                    seen.add(desc)
                    lines.append(f"How they learn (from onboarding): {desc}")
        except Exception:
            pass

        from coast_content_oma.student.stores import identity_namespace

        ns = identity_namespace(user_id)
        orch = _student_orchestrator()
        for it in orch.identity.all_traits(ns, min_confidence=0.0)[:6]:
            desc = (it.content or "").strip()
            if not desc or desc in seen:
                continue
            seen.add(desc)
            ttype = (it.store_specific or {}).get("trait_type") or "trait"
            label = str(ttype).replace("_", " ").title()
            quote = (it.store_specific or {}).get("evidence_quote")
            lines.append(f"{label}: {desc}" + (f' (their words: "{quote[:160]}")' if quote else ""))

        for link in _prior_learning(user_id, folder, _section_refs(user_id, folder, 0)):
            lines.append(f"Builds on {link['course']} ({link['when']}): they worked on {link['concept']} "
                         f"({link['state']}) — relates to {link['relates_to']} here.")

        course_topics: list[str] = []
        for sec in sections or []:
            title = sec.get("title")
            if title:
                course_topics.append(str(title))
            for t in sec.get("key_topics") or []:
                if t:
                    course_topics.append(str(t))
        if course_topics:
            unique = list(dict.fromkeys(course_topics))[:12]
            lines.append(f"Course vocabulary (connect traits if relevant): {', '.join(unique)}")

        if not lines:
            return ""

        block = [
            "--- STUDENT OMA (course intro — personalize if relevant) ---",
            *lines,
            "Teach in their stated learning style from this very first message (do it, don't just announce it). "
            "If a 'Builds on' link is listed, connect ONE specific idea from that earlier course to this one. "
            "Do not invent traits or links that are not listed.",
            "--- END STUDENT OMA ---",
        ]
        out = "\n".join(block)
        if len(out) > max_chars:
            out = out[:max_chars] + "\n[... truncated ...]"
        return out
    except Exception:
        logger.exception("Course intro student block failed")
        return ""


def get_section_intro_student_block(
    user_id: int | str,
    folder: str,
    section_index: int,
    key_topics: list | None = None,
    max_chars: int = 900,
) -> str:
    """Student OMA facts for this section — Pedro weaves into the intro when relevant."""
    if not is_student_enabled():
        return ""
    try:
        import lesson as lesson_mod
        from coast_content_oma.student.mastery_tier import compute_mastery_tier
        from coast_content_oma.student.stores import course_namespace

        refs = lesson_mod.get_section_concept_refs(int(user_id), folder, int(section_index))
        concept_ids = {r["concept_id"] for r in refs if r.get("concept_id")}
        name_by_id = {
            r["concept_id"]: r.get("concept_name") or r["concept_id"]
            for r in refs if r.get("concept_id")
        }
        topic_tokens = {str(t).lower() for t in (key_topics or []) if t}

        course_ns = course_namespace(user_id, folder)
        orch = _student_orchestrator()
        lines: list[str] = []

        strong: list[str] = []
        partial: list[str] = []
        weak: list[str] = []
        for cid in concept_ids:
            name = name_by_id.get(cid, cid)
            it = orch.mastery.for_concept(course_ns, cid)
            if not it:
                continue
            tier = compute_mastery_tier(it.store_specific or {})
            if tier == "green":
                strong.append(name)
            elif tier == "yellow":
                partial.append(name)
            else:
                weak.append(name)

        if strong:
            lines.append(f"Already confident here: {', '.join(strong[:4])}")
        if partial:
            lines.append(f"Partially grasped (reinforce): {', '.join(partial[:4])}")
        if weak:
            lines.append(f"Previously struggled with: {', '.join(weak[:4])}")

        mistakes = [
            m for m in _load_section_mistakes(user_id, folder)
            if m.get("section_index") == section_index
        ]
        if mistakes:
            snippets = []
            for m in mistakes[:3]:
                msg = (m.get("user_message") or "").strip()
                if msg:
                    snippets.append(f'"{msg[:70]}"')
            if snippets:
                lines.append(f"Prior wrong answers in this section: {'; '.join(snippets)}")

        struggling = [
            t for t in _load_struggling_topics(user_id, folder)
            if t.get("concept_id") in concept_ids
        ]
        if struggling:
            names = ", ".join(t.get("name") or "" for t in struggling[:3] if t.get("name"))
            if names:
                lines.append(f"Persistently struggling with: {names}")

        due: list[str] = []
        for it in orch.mastery.due_for_review(course_ns, k=8):
            ss = it.store_specific or {}
            cid = ss.get("concept_id")
            if cid in concept_ids:
                due.append(ss.get("concept_name") or name_by_id.get(cid, cid))
        if due:
            lines.append(f"Due for review: {', '.join(due[:3])}")

        ac = orch.active.snapshot(course_ns)
        related: list[str] = []
        for q in (ac.get("open_questions") or [])[:3]:
            text = (q.get("text") or "").strip()
            if not text:
                continue
            lower = text.lower()
            if any(t in lower for t in topic_tokens) or any(
                n.lower() in lower for n in name_by_id.values() if n
            ):
                related.append(text[:120])
        lr = ac.get("last_unresolved") or {}
        lr_text = (lr.get("text") or "").strip()
        if lr_text:
            lower = lr_text.lower()
            if any(t in lower for t in topic_tokens) or any(
                n.lower() in lower for n in name_by_id.values() if n
            ):
                related.append(lr_text[:120])
        if related:
            lines.append("Open from last session: " + " | ".join(related[:2]))

        for link in _prior_learning(user_id, folder, refs):
            lines.append(f"Builds on {link['course']} ({link['when']}): {link['concept']} ({link['state']}) "
                         f"→ {link['relates_to']}")

        if not lines:
            return ""

        block = [
            "--- STUDENT OMA (this section — mention in intro if relevant) ---",
            *lines,
            "If anything above applies, weave ONE brief personalized line into your opening "
            "(e.g. build on partial mastery, watch for a prior mistake). Do not invent struggles "
            "or list raw scores. If nothing applies, skip personalization.",
            "--- END STUDENT OMA ---",
        ]
        out = "\n".join(block)
        if len(out) > max_chars:
            out = out[:max_chars] + "\n[... truncated ...]"
        return out
    except Exception:
        logger.exception("Section intro student block failed")
        return ""


def _load_progress_ledger(user_id: int | str, folder: str) -> dict:
    try:
        import lesson as lesson_mod
        return lesson_mod.get_authoritative_progress(int(user_id), folder) or {}
    except Exception:
        return {}


def _load_section_mistakes(user_id: int | str, folder: str) -> list[dict]:
    """One-off wrong answers — Pedro re-teaches; not the same as struggling."""
    try:
        from coast_content_oma.student.stores import course_namespace
        ns = course_namespace(user_id, folder)
        orch = _student_orchestrator()
        out: list[dict] = []
        for ep in orch.episodes.by_types(ns, ("exercise_attempt",)):
            ss = ep.store_specific or {}
            cids = set(ss.get("concept_ids") or [])
            if ss.get("outcome") == "success":
                # A later correct answer on the same concept resolves the mistake.
                out = [m for m in out if not (cids and cids & set(m["concept_ids"]))]
                continue
            if ss.get("outcome") not in ("mistake", "struggle"):
                continue
            if (ss.get("signals") or {}).get("resolved_by_evaluation"):
                continue
            out.append({
                "section_index": ss.get("section_index"),
                "user_message": (ss.get("user_message") or "")[:200],
                "concept_ids": ss.get("concept_ids") or [],
                "when": (ep.created_at or "")[:10],
            })
        out.sort(key=lambda m: (m.get("section_index") if m.get("section_index") is not None else -1, m.get("user_message") or ""))
        return out
    except Exception:
        return []


def _load_struggling_topics(user_id: int | str, folder: str) -> list[dict]:
    """Concepts the student is struggling with NOW (recent graded answers)."""
    try:
        from coast_content_oma.student.stores import course_namespace
        from coast_content_oma.student.struggles import struggling_concepts
        orch = _student_orchestrator()
        return struggling_concepts(orch.episodes, orch.mastery, course_namespace(user_id, folder),
                                   name_for=lambda cid: _lookup_concept_name(user_id, folder, cid))
    except Exception:
        logger.exception("struggling topics failed")
        return []


# ── Pedro grading tags (machine-readable, stripped in the UI) ─────────

TAG_SECTION_COMPLETE = "[SECTION_COMPLETE]"
TAG_TEST_OUT_PASSED = "[TEST_OUT_PASSED]"

# Capture tags — Pedro emits these to grow long-term memory mid-conversation.
# Format: [REMEMBER: <trait_type>: <short description>]  and  [CLICKED: <what clicked>]
# They are parsed (written to identity / pattern stores) then stripped from
# the visible reply. trait_type is constrained by the prompt to:
#   learning_style, session_pattern, motivation_pattern,
#   general_strength, general_weakness.
_REMEMBER_TAG_RE = re.compile(
    r"\[REMEMBER:\s*([A-Za-z_]+)\s*:\s*([^\]\n]+?)\s*\]",
    re.IGNORECASE,
)
_CLICKED_TAG_RE = re.compile(
    r"\[CLICKED:\s*([^\]\n]+?)\s*\]",
    re.IGNORECASE,
)
# Combined pattern for stripping without parsing (used by strip_pedro_tags).
_CAPTURE_TAG_RE = re.compile(
    r"\[(?:REMEMBER:[^\]\n]*|CLICKED:[^\]\n]*)\]",
    re.IGNORECASE,
)


def strip_pedro_tags(text: str) -> str:
    from coast_content_oma.student.grading import strip_ui_tags
    return _CAPTURE_TAG_RE.sub("", strip_ui_tags(text)).strip()


def extract_capture_tags(text: str) -> tuple[list[dict], list[str], str]:
    """Pull [REMEMBER ...] and [CLICKED ...] tags out of a Pedro reply.

    Returns (remembers, clickeds, cleaned_text) where:
      remembers  — list of {"trait_type": str, "description": str}
      clickeds   — list of description strings (what made it click)
      cleaned    — the reply with the applied capture tags removed (whitespace tidied)

    Only tags in Pedro's own voice count: one shown inside code or a quote is left in
    the text and never becomes memory. trait_type must be one Coast knows.
    """
    from coast_content_oma.student.grading import own_voice, tag_body
    from coast_content_oma.student.stores.academic_identity import CANONICAL_TRAIT_TYPES
    out = text or ""
    voice = own_voice(out)
    remembers: list[dict] = []
    spans: list[tuple[int, int]] = []
    for m in _REMEMBER_TAG_RE.finditer(voice):
        trait_type = (m.group(1) or "").strip().lower()
        desc = tag_body(out[m.start(2):m.end(2)]).strip().strip("\"'")
        spans.append(m.span())
        if trait_type in CANONICAL_TRAIT_TYPES and desc:
            remembers.append({"trait_type": trait_type, "description": desc})
    clickeds: list[str] = []
    for m in _CLICKED_TAG_RE.finditer(voice):
        desc = tag_body(out[m.start(1):m.end(1)]).strip().strip("\"'")
        spans.append(m.span())
        if desc:
            clickeds.append(desc)
    cleaned = out
    for start, end in sorted(spans, reverse=True):
        cleaned = cleaned[:start] + cleaned[end:]
    # Collapse the blank lines / stray spaces left behind by removed tags.
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return remembers, clickeds, cleaned


def apply_capture_tags(
    user_id: int | str,
    folder: Optional[str],
    remembers: list[dict],
    clickeds: list[str],
    focus_concept_id: Optional[str] = None,
    section_index: Optional[int] = None,
    user_message: Optional[str] = None,
) -> None:
    """Write parsed capture tags into Student OMA stores.

    [REMEMBER ...] → AcademicIdentityStore (cross-course traits).
    [CLICKED ...]  → PatternStore as a golden_moment for the current course.
    Safe no-op when Student OMA is disabled or folder is missing for clickeds.
    """
    if not is_student_enabled():
        return
    if not remembers and not clickeds:
        return
    try:
        from coast_content_oma.student.stores import (
            course_namespace,
            identity_namespace,
        )
        orch = _student_orchestrator()
        id_ns = identity_namespace(user_id)

        for rem in remembers:
            trait_type = rem["trait_type"]
            desc = rem["description"]
            dedupe = f"pedro_tag:{trait_type}:{desc[:60].lower()}"
            try:
                orch.identity.upsert_trait(
                    id_ns,
                    trait_type,
                    desc,
                    confidence=0.7,
                    evidence_courses=[folder] if folder else [],
                    derivation="pedro_remember_tag",
                    dedupe_key=dedupe,
                    evidence_quote=(user_message or "").strip() or None,
                )
            except Exception:
                logger.exception(
                    "apply_capture_tags: identity upsert failed trait=%s", trait_type
                )

        if clickeds and folder:
            course_ns = course_namespace(user_id, folder)
            section_refs = _section_refs(user_id, folder, section_index)
            for desc in clickeds:
                # Link the analogy to the concept(s) it explains so it returns with them.
                if focus_concept_id:
                    related = [focus_concept_id]
                else:
                    low = desc.lower()
                    related = [c["concept_id"] for c in section_refs
                               if (c.get("concept_name") or "").lower() in low]
                    related = related or [c["concept_id"] for c in section_refs]
                dedupe = f"pedro_clicked:{desc[:60].lower()}"
                try:
                    orch.patterns.upsert(
                        course_ns,
                        "golden_moment",
                        desc,
                        confidence=0.85,
                        evidence_count=1,
                        related_concept_ids=related,
                        derivation="pedro_clicked_tag",
                        dedupe_key=dedupe,
                    )
                except Exception:
                    logger.exception(
                        "apply_capture_tags: pattern upsert failed desc=%s", desc[:80]
                    )
    except Exception:
        logger.exception("apply_capture_tags failed user=%s folder=%s", user_id, folder)


def general_namespace(user_id: int | str) -> str:
    """Student OMA namespace for conversations not tied to any course."""
    return f"u{user_id}__general"


_CONSOLIDATE_EVERY_N_TURNS = int(os.environ.get("OMA_CONSOLIDATE_EVERY_N_TURNS", "8"))
_turns_since_consolidation: dict[str, int] = {}


def _section_refs(user_id, folder, section_index) -> list[dict]:
    if section_index is None:
        return []
    try:
        import lesson as lesson_mod
        return lesson_mod.get_section_concept_refs(int(user_id), folder, int(section_index))
    except Exception:
        return []


def _resolve_graded_concept(user_id, folder, grade, section_refs, focus_concept_id) -> list[dict]:
    """Which concept did this graded answer test? One concept, or none — never the whole section."""
    from coast_content_oma.student.grading import match_concept
    if grade.concept:
        hit = match_concept(grade.concept, section_refs)
        if hit:
            return [hit]
        try:
            from coast_content_oma.course_identity import content_namespace_for_student
            item = _content_orchestrator().concept.find_by_name(content_namespace_for_student(user_id, folder), grade.concept)
            if item:
                return [{"concept_id": item.id, "concept_name": (item.store_specific or {}).get("name") or grade.concept}]
        except Exception:
            pass
        return []
    if focus_concept_id:
        return [{"concept_id": focus_concept_id, "concept_name": _lookup_concept_name(user_id, folder, focus_concept_id)}]
    return section_refs[:1] if len(section_refs) == 1 else []


# One ordered writer per process: memory writes never delay Pedro's reply, and
# turns are applied in the order they happened. A write lost to a crash can be
# rebuilt from chat_messages (scripts/backfill_student_turns.py is idempotent).
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor

_student_writer = _ThreadPoolExecutor(max_workers=1, thread_name_prefix="student-oma-writer")


def flush_student_writes(timeout: float = 30.0) -> None:
    """Block until every queued Student OMA write has been applied."""
    _student_writer.submit(lambda: None).result(timeout=timeout)


def record_conversation_turn(*args, **kwargs) -> None:
    """Queue one Pedro turn for recording (see _record_conversation_turn)."""
    if not is_student_enabled():
        return
    _student_writer.submit(_record_conversation_turn, *args, **kwargs)


def _record_conversation_turn(
    user_id: int | str,
    context_type: str,
    folder: Optional[str],
    user_message: str,
    assistant_response: str,
    *,
    user_message_id: Optional[int] = None,
    pedro_message_id: Optional[int] = None,
    section_index: Optional[int] = None,
    focus_concept_id: Optional[str] = None,
) -> None:
    """Record one Pedro turn in Student OMA — every surface (lesson, workshop,
    folder, general chat).

    Graded answers ([ANSWER_CORRECT: concept] / [ANSWER_WRONG: concept]) become
    exercise_attempt episodes that update mastery for exactly that concept.
    Everything else becomes a qa episode carrying behaviour signals. Episodes
    reference the chat_messages rows (the canonical transcript) instead of
    copying Pedro's reply.
    """
    if not is_student_enabled() or context_type == "onboarding":
        return
    if _is_lesson_intro(user_message):
        return  # automatic section opener — not the student's own words
    try:
        from coast_content_oma.student.grading import parse_grades, parse_tutor_corrections
        orch = _student_orchestrator()
        message_ids = [int(i) for i in (user_message_id, pedro_message_id) if i]
        _, signals = _classify_outcome_and_signals(user_message, assistant_response)
        signals.pop("tutor_affirmed", None)
        signals.pop("tutor_corrected", None)
        source = context_type or "chat"
        student_text = (user_message or "").strip()

        if not folder:
            ns = general_namespace(user_id)
            if pedro_message_id and orch.episodes.has_message(ns, pedro_message_id):
                return
            orch.episodes.record(ns, "qa", summary=student_text[:500], signals=signals, source=source,
                                 chat_message_ids=message_ids)
            return

        from coast_content_oma.student.stores import course_namespace
        ns = course_namespace(user_id, folder)
        if pedro_message_id and orch.episodes.has_message(ns, pedro_message_id):
            return
        rec = _student_recorder_singleton()
        section_refs = _section_refs(user_id, folder, section_index)
        grades = parse_grades(assistant_response)
        corrections = parse_tutor_corrections(assistant_response)
        if corrections:
            # Pedro corrected an error of his own: the student's answer on that concept that
            # followed it was not their mistake, and this reply should not mark them wrong either.
            grades = [g for g in grades if g.correct]
        from coast_content_oma.student.grading import Grade
        # Concepts are resolved first (Content OMA reads), so the transaction below stays short.
        corrected = [(label, [r["concept_id"] for r in _resolve_graded_concept(
                        user_id, folder, Grade(False, label), section_refs, None)] if label else None)
                     for label in corrections] if section_index is not None else []
        graded = [(grade, _resolve_graded_concept(user_id, folder, grade, section_refs, focus_concept_id))
                  for grade in grades]
        touched = []
        if not grades:
            text = f"{student_text} {assistant_response or ''}".lower()
            touched = [c for c in section_refs if (c.get("concept_name") or "").lower() in text]
            if focus_concept_id and not touched:
                touched = [{"concept_id": focus_concept_id,
                            "concept_name": _lookup_concept_name(user_id, folder, focus_concept_id)}]

        from coast_content_oma.stores.db import atomic
        # One transaction per turn: a failure part-way through leaves nothing behind, so the
        # message-level duplicate check above never skips a half-recorded turn on retry.
        with atomic(orch.episodes.db_path):
            for label, ids in corrected:
                for cid in orch.episodes.mark_tutor_error(ns, int(section_index), label, ids) or []:
                    # Recompute from the remaining evidence, as if the withdrawn answer never happened.
                    orch.mastery.rebuild(ns, cid, orch.episodes.valid_attempts(ns, cid))
            for grade, refs in graded:
                rec.record_episode(
                    user_id, folder, "exercise_attempt",
                    summary=student_text[:500],
                    outcome="success" if grade.correct else "mistake",
                    concept_refs=refs,
                    user_message=student_text[:300],
                    # "with help", "on their own" and "remembered later" stay distinct evidence.
                    signals={**signals, "delayed_recall": True} if grade.recall else signals,
                    section_index=section_index,
                    source=source,
                    chat_message_ids=message_ids,
                    hinted=grade.hinted,
                    concept_label=grade.concept,
                )
            if not grades:
                rec.record_episode(
                    user_id, folder, "qa",
                    summary=student_text[:500],
                    outcome="neutral",
                    concept_refs=touched,
                    signals=signals,
                    section_index=section_index,
                    source=source,
                    chat_message_ids=message_ids,
                )

        key = _ingest_key(user_id, folder)
        with _ingest_lock:
            n = _turns_since_consolidation.get(key, 0) + 1
            _turns_since_consolidation[key] = 0 if n >= _CONSOLIDATE_EVERY_N_TURNS else n
        if n >= _CONSOLIDATE_EVERY_N_TURNS:
            kickoff_course_consolidation_async(user_id, folder)
    except Exception:
        logger.exception("Recording conversation turn failed user=%s folder=%s", user_id, folder)


def record_section_completed_authoritative(user_id, folder, section_index, section_title):
    from learning_jobs import enqueue_committed
    return enqueue_committed(int(user_id), folder, int(section_index), section_title)


def kickoff_post_section_pipeline_async(user_id, folder, section_index, section_title):
    from learning_jobs import enqueue_committed
    enqueue_committed(int(user_id), folder, int(section_index), section_title)
    return True


def kickoff_course_consolidation_async(user_id: int | str, folder: str) -> bool:
    """Derive Student OMA patterns off the request thread so section advance stays fast."""
    if not is_student_enabled():
        return False
    key = _ingest_key(user_id, folder)
    with _ingest_lock:
        if key in _consolidation_active:
            _consolidation_pending.add(key)
            return False
        _consolidation_active.add(key)

    def _run() -> None:
        try:
            while True:
                _run_course_consolidation(user_id, folder)
                with _ingest_lock:
                    if key not in _consolidation_pending:
                        break
                    _consolidation_pending.discard(key)
        except Exception:
            logger.exception("Background course consolidation failed folder=%s", folder)
        finally:
            with _ingest_lock:
                _consolidation_active.discard(key)
                _consolidation_pending.discard(key)

    threading.Thread(target=_run, daemon=True).start()
    return True


def _run_course_consolidation(user_id: int | str, folder: str) -> None:
    """Derive patterns from episodes — slow layer, not per chat turn."""
    try:
        from coast_content_oma.student.consolidator import CourseConsolidator
        from coast_content_oma.student.stores import course_namespace
        ns = course_namespace(user_id, folder)
        orch = _student_orchestrator()
        CourseConsolidator(orch.episodes, orch.mastery, orch.patterns).run(ns)
        # Prune stale neutral chat episodes so profile scans stay fast.
        pruned = orch.episodes.compact(ns, before_days=180.0)
        if pruned:
            logger.info("Compacted %d old episodes in %s", pruned, ns)
    except Exception:
        logger.exception("Course consolidation failed")

    # Roll course-level patterns into the cross-course identity namespace
    # so the next course starts with what we know about how they learn.
    try:
        from coast_content_oma.student.consolidator import IdentityConsolidator
        from coast_content_oma.student.stores import list_course_namespaces
        orch = _student_orchestrator()
        namespaces = list_course_namespaces(orch.episodes.db_path, user_id)
        if namespaces:
            IdentityConsolidator(orch.patterns, orch.identity, namespaces).run(user_id)
    except Exception:
        logger.exception("Identity consolidation failed")


def record_section_completed(
    user_id: int | str,
    folder: str,
    section_title: str,
    section_index: Optional[int] = None,
) -> None:
    if not is_student_enabled():
        return
    try:
        rec = _student_recorder_singleton()
        rec.record_section_completed(
            user_id, folder,
            section_title=section_title,
            section_index=section_index,
        )
    except Exception:
        logger.exception("Recording section completion failed")


# ── Maintenance: concept dedup for existing namespaces ───────────────

def dedupe_folder_concepts(
    user_id: int | str,
    folder: str,
    threshold: float = 0.90,
    dry_run: bool = False,
) -> dict:
    """Merge near-duplicate concept nodes in a folder's Content OMA
    namespace.

    When the alias ledger is enabled (default), merges append ledger entries
    and skip destructive Student OMA remaps. When disabled, live merge is
    refused unless dry_run=True."""
    from coast_content_oma.concept_identity import alias_ledger_enabled
    from coast_content_oma.stores import make_namespace
    from coast_content_oma.stores.concept_alias import ConceptAliasStore
    from coast_content_oma.student.stores import course_namespace

    use_ledger = alias_ledger_enabled()
    if not dry_run and not use_ledger:
        return {
            "error": "Live concept merge frozen. Set OMA_ALIAS_LEDGER_ENABLED=true or use dry_run=True.",
            "concepts": 0,
            "merged": 0,
            "groups": [],
        }

    ns = make_namespace(user_id, folder)
    orch = _content_orchestrator()
    items = {it.id: it for it in orch.concept.all(ns)}
    if len(items) < 2:
        return {"concepts": len(items), "merged": 0, "groups": []}

    pairs: list[tuple[str, str]] = []

    # 1. Exact name/alias collisions.
    by_name: dict[str, str] = {}
    for it in items.values():
        ss = it.store_specific or {}
        for n in [ss.get("name")] + (ss.get("aliases") or []):
            key = (n or "").lower().strip()
            if not key:
                continue
            if key in by_name and by_name[key] != it.id:
                pairs.append((by_name[key], it.id))
            else:
                by_name.setdefault(key, it.id)

    # 2. Embedding cosine pairs.
    embs = dict(orch.concept.embeddings_for_namespace(ns))
    ids_with_emb = [i for i in items if i in embs]
    if threshold > 0 and len(ids_with_emb) >= 2:
        try:
            import numpy as np
            mat = np.array([embs[i] for i in ids_with_emb], dtype=np.float32)
            mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
            sims = mat @ mat.T
            for a in range(len(ids_with_emb)):
                for b in range(a + 1, len(ids_with_emb)):
                    if float(sims[a, b]) >= threshold:
                        pairs.append((ids_with_emb[a], ids_with_emb[b]))
        except ImportError:
            logger.warning("numpy unavailable — skipping embedding dedup pass")

    if not pairs:
        return {"concepts": len(items), "merged": 0, "groups": []}

    # Union-find; canonical root = earliest created_at.
    parent = {i: i for i in items}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        keep, drop = sorted(
            (ra, rb), key=lambda i: items[i].created_at or "",
        )
        parent[drop] = keep

    groups: dict[str, list[str]] = {}
    for i in items:
        root = find(i)
        if root != i:
            groups.setdefault(root, []).append(i)

    group_report = [
        {
            "canonical": (items[root].store_specific or {}).get("name"),
            "merged": [(items[s].store_specific or {}).get("name") for s in srcs],
        }
        for root, srcs in groups.items()
    ]
    if dry_run:
        return {"concepts": len(items), "merged": sum(len(s) for s in groups.values()),
                "groups": group_report, "dry_run": True}

    course_ns = course_namespace(user_id, folder)
    student_orch = _student_orchestrator() if is_student_enabled() else None
    alias_store = ConceptAliasStore(orch.concept.db_path)
    id_map: dict[str, str] = {}
    n_merged = 0
    for root, srcs in groups.items():
        for src in srcs:
            if orch.concept.merge_into(
                ns, src, root,
                alias_store=alias_store,
                use_ledger=use_ledger,
                merge_confidence=threshold,
                merge_reason="dedupe_folder_concepts",
            ):
                id_map[src] = root
                n_merged += 1

    if id_map and not use_ledger:
        _remap_entity_ids(orch, ns, id_map)
        if student_orch is not None:
            from coast_content_oma.student.concept_remap import remap_student_concept_ids
            remap_student_concept_ids(student_orch, course_ns, id_map)

    logger.info("concept dedup %s: %d merged across %d groups", ns, n_merged, len(groups))
    return {"concepts": len(items), "merged": n_merged, "groups": group_report}


def _remap_entity_ids(orch, namespace: str, id_map: dict[str, str]) -> None:
    """Rewrite content/image entities that point at merged concept ids."""
    import json as _json
    from coast_content_oma.stores.db import connect_db
    for store in (orch.content, orch.images):
        updates = []
        for it in store.all(namespace):
            if not it.entities or not any(e in id_map for e in it.entities):
                continue
            new_entities = list(dict.fromkeys(id_map.get(e, e) for e in it.entities))
            updates.append((_json.dumps(new_entities), " ".join(new_entities), it.id))
        if not updates:
            continue
        with connect_db(store.db_path) as conn:
            conn.executemany(
                f"UPDATE {store.table} SET entities = ? WHERE id = ?",
                [(u[0], u[2]) for u in updates],
            )
            conn.executemany(
                f"UPDATE {store.fts_table} SET entities = ? WHERE id = ?",
                [(u[1], u[2]) for u in updates],
            )


def _lookup_concept_name(user_id: int | str, folder: str, concept_id: str) -> str:
    try:
        from coast_content_oma.stores import make_namespace
        ns = make_namespace(user_id, folder)
        orch = _content_orchestrator()
        item = orch.concept.get(concept_id)
        if item:
            return (item.store_specific or {}).get("name") or concept_id
    except Exception:
        pass
    return concept_id


# ── Heuristic outcome / signal detection ─────────────────────────────

_SIGNAL_REGEX = {
    "asked_for_example": re.compile(r"\b(an? )?example\b|\bworked example\b|\bshow me\b", re.I),
    "asked_for_diagram": re.compile(r"\b(diagram|figure|picture|graph|chart|visual|sketch)\b", re.I),
    "asked_for_definition_first": re.compile(r"\bwhat (is|are|does)\b|\bdefine\b|\bdefinition of\b|\bmeaning of\b", re.I),
    "asked_step_by_step": re.compile(r"\bstep[- ]by[- ]step\b|\bwalk me through\b|\bone step at a time\b", re.I),
    "asked_for_shorter": re.compile(r"\bshorter\b|\bbriefly\b|\btoo long\b|\btl;?dr\b", re.I),
    "asked_followup": re.compile(r"\bbut\b|\bhowever\b|\bwait,?\b|\bso\b.*\?", re.I),
}


# Student self-signals.
_SUCCESS_HINTS = re.compile(
    r"\b(thanks|got it|makes sense|understood|now i get|that helps|perfect|nice|cool|"
    r"ok i see|i see now|i already know|i know this|i've got this|got this|"
    r"easy|simple|clear now)\b",
    re.I,
)
_STRUGGLE_HINTS = re.compile(
    r"\b(i don'?t (understand|get)|still confused|that doesn'?t make sense|wait,? what|huh|"
    r"no that'?s wrong|this isn'?t right|i'?m lost|no idea|can'?t follow|"
    r"not sure|what do you mean|im confused|i'?m confused)\b",
    re.I,
)
_GIVEUP_HINTS = re.compile(r"\b(give up|forget it|skip this|too hard|easier)\b", re.I)
_MASTERY_CLAIM = re.compile(
    r"\b(100\s*%|100\s*percent|full mastery|prove (i|that i) (have|know)|"
    r"already (know|master)|test me|quiz me)\b",
    re.I,
)

# Tutor-side judgements that strongly indicate the student's last answer
# was right or wrong. Pedro is consistent enough in tone that these are
# reliable signals.
_TUTOR_STRONG_SUCCESS = re.compile(
    r"\b(spot on|nailed|nailed it|perfect|excellent|exactly right|"
    r"that'?s right|well done|great work|good job|nice work|"
    r"you'?ve got it|you got it|absolutely right|you'?re right|"
    r"perfect score|perfect scores|brilliant|you'?ve mastered|"
    r"100%\s*accurate|completely mastered|perfect execution)\b",
    re.I,
)
_TUTOR_WEAK_SUCCESS = re.compile(
    r"\b(on the right track|great start|good start|nice try|"
    r"good effort|keep going)\b",
    re.I,
)
# Tutor signals the student's last answer was wrong or incomplete.
# Avoid bare "mistake"/"correct" — they fire on pedagogical asides
# ("one tiny mistake…", "Correct Rule").
_TUTOR_CORRECTION = re.compile(
    r"\b(not quite|that'?s not (quite |exactly )?right|"
    r"very close|re-examine|let'?s try again|think again|"
    r"reconsider|the issue is|incorrect|"
    r"to be precise|actually,? it'?s not|actually,? the|"
    r"the correct answer is|"
    r"(?:your|made a|that|a) mistake\b|mistake in your)\b",
    re.I,
)

# Lesson section openers the frontend sends automatically — not real student work.
_LESSON_INTRO = re.compile(
    r"^(i'?m ready to learn about|i'?d like to reach 100% mastery)\b",
    re.I,
)


def _is_lesson_intro(user_msg: str) -> bool:
    return bool(_LESSON_INTRO.match((user_msg or "").strip()))


def _extract_section_title(user_msg: str) -> str | None:
    m = re.search(
        r"(?:ready to learn about|reach 100% mastery on) [\"'](.+?)[\"']",
        user_msg or "",
        re.I,
    )
    return m.group(1).strip() if m else None


def _infer_concept_refs_from_text(
    user_id: int | str,
    folder: str,
    user_msg: str,
    assistant_msg: str,
    outcome: str,
) -> list[dict]:
    """Attach concept refs when the topic is mentioned in the turn."""
    if outcome not in ("success", "struggle"):
        return []
    text = f"{user_msg} {assistant_msg}".lower()
    refs: list[dict] = []
    seen: set[str] = set()

    def _add(cid: str, name: str) -> None:
        nl = name.lower()
        if not cid or nl in seen:
            return
        if nl in text or (len(nl) >= 5 and nl in text):
            refs.append({"concept_id": cid, "concept_name": name})
            seen.add(nl)

    # 1) Content OMA canonical concepts for this course.
    try:
        from coast_content_oma.stores import make_namespace
        content_ns = make_namespace(user_id, folder)
        content_orch = _content_orchestrator()
        for it in content_orch.concept.all(content_ns):
            ss = it.store_specific or {}
            name = ss.get("name") or it.content or ""
            cid = it.id
            _add(cid, name)
    except Exception:
        pass

    # 2) Existing mastery rows (may already exist mid-session).
    try:
        from coast_content_oma.student.stores import course_namespace
        orch = _student_orchestrator()
        ns = course_namespace(user_id, folder)
        for it in orch.mastery.all(ns):
            ss = it.store_specific or {}
            _add(ss.get("concept_id", ""), ss.get("concept_name") or "")
    except Exception:
        pass

    return refs[:4]


def _classify_outcome_and_signals(user_msg: str, assistant_reply: str) -> tuple[str, dict]:
    """Return (outcome, signals). Strong tutor praise wins over soft
    partial-correction phrasing ('almost there' + 'nailed the first')."""
    signals: dict = {}
    for k, rx in _SIGNAL_REGEX.items():
        if rx.search(user_msg or ""):
            signals[k] = True
    if _GIVEUP_HINTS.search(user_msg or ""):
        signals["gave_up"] = True
        signals["requested_easier"] = True

    student_struggle = bool(_STRUGGLE_HINTS.search(user_msg or "") or signals.get("gave_up"))
    student_success = bool(
        _SUCCESS_HINTS.search(user_msg or "") or _MASTERY_CLAIM.search(user_msg or "")
    )
    tutor_correction = bool(_TUTOR_CORRECTION.search(assistant_reply or ""))
    tutor_strong_success = bool(_TUTOR_STRONG_SUCCESS.search(assistant_reply or ""))
    tutor_weak_success = bool(_TUTOR_WEAK_SUCCESS.search(assistant_reply or ""))

    # Correction beats weak praise ("great start… to be precise").
    # Strong praise beats correction ("nailed the first step, but…").
    # Weak praise + correction = partial credit teaching → neutral.
    if tutor_correction and not tutor_strong_success and not tutor_weak_success:
        outcome = "struggle"
    elif tutor_strong_success:
        outcome = "success"
    elif tutor_correction or tutor_weak_success:
        outcome = "neutral"
    elif student_struggle:
        outcome = "struggle"
    elif student_success:
        outcome = "success"
    else:
        outcome = "neutral"

    if tutor_strong_success or tutor_weak_success:
        signals["tutor_affirmed"] = True
    if tutor_correction:
        signals["tutor_corrected"] = True
    return outcome, signals
