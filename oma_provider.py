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
    """Use Content OMA on Render production (persistent /data disk).

    render.yaml defines RAG_PROVIDER=oma but many Render services were created
    before the blueprint — the env var is missing and defaults to flat.
    """
    explicit = (os.environ.get("RAG_PROVIDER") or "").strip().lower()
    if _ON_RENDER_DISK and explicit in ("", "flat"):
        return "oma"
    return explicit or "flat"


def _resolve_data_path(env_key: str, local_default: str, render_default: str) -> Path:
    raw = os.environ.get(env_key)
    if raw:
        return Path(raw).resolve()
    if _ON_RENDER_DISK:
        return Path(render_default).resolve()
    return Path(local_default).resolve()


RAG_PROVIDER = _resolve_rag_provider()
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
_ingest_slots = 1 if _ON_RENDER_DISK else 2
_ingest_semaphore = threading.Semaphore(_ingest_slots)  # limit parallel PDF ingests (LLM-heavy)


def _ingest_key(user_id: int | str, folder: str) -> str:
    return f"{user_id}:{folder}"


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
    return RAG_PROVIDER in ("oma", "shadow")


def is_student_enabled() -> bool:
    return STUDENT_OMA_ENABLED


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
        describe = _env_truthy(os.environ.get("OMA_DESCRIBE_IMAGES", "true"))
        if _ON_RENDER_DISK:
            # Vision + parallel page work can OOM a small Render instance.
            describe = _env_truthy(os.environ.get("OMA_DESCRIBE_IMAGES", "false"))
        workers = int(os.environ.get("OMA_INGEST_WORKERS", "1" if _ON_RENDER_DISK else "4"))
        skip_images = _env_truthy(os.environ.get("OMA_SKIP_IMAGES", "false"))
        _content_pipeline = IngestionPipeline(
            orch.concept, orch.content, orch.images, OMA_IMAGE_DIR,
            max_workers=max(1, workers),
            describe_images=describe,
            skip_images=skip_images,
        )
        logger.info(
            "OMA ingest pipeline: workers=%d describe_images=%s skip_images=%s",
            workers, describe, skip_images,
        )
    return _content_pipeline


# ── Upload-time: ingest a single PDF into Content OMA ────────────────

def ingest_pdf_into_oma(user_id: int | str, folder: str, pdf_path: str | Path) -> None:
    """Run synchronously (callers should put this in a background thread).
    Ingestion of one PDF: text extraction + page classification + concept
    canonicalization for the new lecture, blended into any existing course
    namespace."""
    if not is_oma_enabled():
        return
    from coast_content_oma.stores import make_namespace
    ns = make_namespace(user_id, folder)
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.warning(f"OMA ingest skipped — file missing: {pdf_path}")
        return
    key = _ingest_key(user_id, folder)
    with _ingest_lock:
        _ingest_active[key] = _ingest_active.get(key, 0) + 1
    pipeline = _content_ingest_pipeline()
    try:
        with _ingest_semaphore:
            stats = pipeline.ingest_folder(ns, [pdf_path], progress=lambda m: logger.info(f"OMA ingest: {m}"))
        logger.info(f"OMA ingest complete for {pdf_path.name}: {stats.to_dict()}")
    except Exception:
        logger.exception(f"OMA ingest failed for {pdf_path}")
    finally:
        with _ingest_lock:
            _ingest_active[key] = max(0, _ingest_active.get(key, 1) - 1)


def ingest_pdf_async(user_id: int | str, folder: str, pdf_path: str | Path) -> None:
    """Fire-and-forget wrapper for convenience from the upload endpoint."""
    if not is_oma_enabled():
        return
    threading.Thread(
        target=ingest_pdf_into_oma,
        args=(user_id, folder, pdf_path),
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
        if p.is_file() and p.suffix.lower() == ".pdf":
            return p
    p = _folder_uploads_dir() / f"{source_id}{Path(filename).suffix.lower()}"
    if p.is_file() and p.suffix.lower() == ".pdf":
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
            })
        return out
    finally:
        db.close()


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

    def _content_items():
        nonlocal content_cache
        if content_cache is None:
            content_cache = orch.content.all(ns)
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
        names = _names_for_source(src)
        if not names:
            return 0
        n = 0
        for it in _content_items():
            ss = it.store_specific or {}
            if ss.get("source_filename") in names:
                n += 1
        return n

    def _kickoff_background_ingests() -> int:
        """Queue async ingest for PDFs still missing — never blocks the web worker."""
        if not pdf_sources:
            return 0
        kicked: set[str] = set()
        queued = 0
        for src in pdf_sources:
            path = src.get("path")
            need = max(1, int(src.get("page_count") or 0))
            sid = src.get("source_id") or path or ""
            if sid in kicked:
                continue
            if not path:
                continue
            p = Path(path)
            if not p.is_file():
                continue
            if _source_gap_ok(_pages_for_source(src), need):
                continue
            kicked.add(sid)
            ingest_pdf_async(user_id, folder, p)
            queued += 1
        if queued:
            logger.info(
                "outline: queued background OMA ingest for %d PDF(s) folder=%s",
                queued, folder,
            )
        return queued

    def _is_ready() -> bool:
        if _active() > 0:
            return False
        if not pdf_sources:
            return _count() >= target
        total_have = 0
        total_need = 0
        gaps = 0
        for src in pdf_sources:
            need = max(1, int(src.get("page_count") or 0))
            have = _pages_for_source(src)
            total_have += have
            total_need += need
            if not _source_gap_ok(have, need):
                gaps += 1
        if gaps == 0:
            return True
        return total_have >= max(1, int(total_need * 0.97))

    if wait_sec == 600 and target > 80 and allow_sync_ingest:
        wait_sec = min(2400, 120 + target * 3)

    if _is_ready():
        return True

    _kickoff_background_ingests()

    # Log why we're waiting when total page count already looks complete.
    if pdf_sources and _count() >= target and _active() == 0:
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
        if _is_ready():
            logger.info("outline: Content OMA ready folder=%s (%d pages)", folder, _count())
            return True
        if _active() == 0:
            _kickoff_background_ingests()
        time.sleep(poll_sec)

    if not allow_sync_ingest:
        logger.info(
            "outline: Content OMA not ready after %.0fs (background ingest continues) folder=%s",
            wait_sec, folder,
        )
        return _is_ready()

    # Local dev only — sync ingest (too heavy for Render web workers).
    to_ingest: list[Path] = []
    for src in pdf_sources or []:
        path = src.get("path")
        need = max(1, int(src.get("page_count") or 0))
        if not path:
            continue
        p = Path(path)
        if not p.is_file():
            continue
        if _pages_for_source(src) < need and not _source_gap_ok(_pages_for_source(src), need):
            to_ingest.append(p)

    if to_ingest:
        logger.info(
            "outline: sync ingesting %d PDF(s) for folder=%s (have %d/%d pages)",
            len(to_ingest), folder, _count(), target,
        )
        try:
            pipeline = _content_ingest_pipeline()
            pipeline.ingest_folder(
                ns, to_ingest,
                progress=lambda m: logger.info(f"OMA sync ingest: {m}"),
            )
        except Exception:
            logger.exception("OMA sync ingest failed during outline for folder=%s", folder)

    ready = _is_ready()
    logger.info(
        "outline: Content OMA %s folder=%s (%d/%d pages)",
        "ready" if ready else "not ready",
        folder, _count(), target,
    )
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
    kicked: set[str] = set()
    queued = 0
    for src in pdf_sources:
        path = src.get("path")
        sid = src.get("source_id") or path or ""
        if sid in kicked or not path:
            continue
        p = Path(path)
        if not p.is_file():
            continue
        names = {src.get("filename") or p.name, p.name}
        if src.get("source_id"):
            names.add(f"{src['source_id']}.pdf")
        have = sum(
            1 for it in items
            if (it.store_specific or {}).get("source_filename") in names
        )
        need = max(1, int(src.get("page_count") or 0))
        if have >= need or (need > 1 and have >= need - max(2, int(need * 0.05))):
            continue
        kicked.add(sid)
        ingest_pdf_async(user_id, folder, p)
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
        rag_provider=RAG_PROVIDER,
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
    for it in student_orch.mastery.all(course_ns):
        ss = it.store_specific or {}
        cid = ss.get("concept_id")
        if not cid:
            continue
        score = float(ss.get("mastery_score", 0.5))
        successes = int(ss.get("successes", 0) or 0)
        from coast_content_oma.student.mastery_tier import is_topic_struggling
        mistake_n = 0
        for ep in student_orch.episodes.all(course_ns):
            ess = ep.store_specific or {}
            if ess.get("episode_type") != "exercise_attempt":
                continue
            if ess.get("outcome") not in ("mistake", "struggle"):
                continue
            if cid in (ess.get("concept_ids") or []):
                mistake_n += 1
        if is_topic_struggling(successes, mistake_n):
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
) -> str:
    """Cross-course profile for global Pedro chat — aggregates every
    folder/course the student has history in."""
    if not is_student_enabled():
        return ""
    try:
        orch = _student_orchestrator()
        profile = orch.build_global_profile(user_id)
        if not profile.get("courses"):
            return ""
        return orch.to_global_prompt_block(profile, max_chars=max_chars)
    except Exception:
        logger.exception("Global student profile build failed")
        return ""


def get_student_profile_block(
    user_id: int | str,
    folder: str,
    current_concept_ids: Optional[list[str]] = None,
    max_chars: int = 1200,
) -> str:
    """Returns the personalized profile block for Pedro. Empty string
    if the student has no recorded history yet or the system is disabled."""
    if not is_student_enabled():
        return ""
    try:
        orch = _student_orchestrator()
        profile = orch.build_profile(user_id, folder, current_concept_ids=current_concept_ids)
        profile["progress_ledger"] = _load_progress_ledger(user_id, folder)
        profile["section_mistakes"] = _load_section_mistakes(user_id, folder)
        profile["struggling_topics"] = _load_struggling_topics(user_id, folder)
        has_mastery = bool((profile.get("mastery_overview") or {}).get("n_concepts"))
        has_progress = bool(profile.get("progress_ledger", {}).get("completed_sections"))
        has_mistakes = bool(profile.get("section_mistakes"))
        if not has_mastery and not has_progress and not has_mistakes:
            return ""
        return orch.to_prompt_block(profile, max_chars=max_chars)
    except Exception:
        logger.exception("Student profile build failed")
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
        for ep in orch.episodes.all(ns):
            ss = ep.store_specific or {}
            if ss.get("episode_type") != "exercise_attempt":
                continue
            if ss.get("outcome") not in ("mistake", "struggle"):
                continue
            idx = ss.get("section_index")
            out.append({
                "section_index": idx,
                "user_message": (ss.get("user_message") or "")[:200],
                "concept_ids": ss.get("concept_ids") or [],
            })
        out.sort(key=lambda m: (m.get("section_index") if m.get("section_index") is not None else -1, m.get("user_message") or ""))
        return out
    except Exception:
        return []


def _load_struggling_topics(user_id: int | str, folder: str) -> list[dict]:
    """Topics where wrong/(wrong+right) > 50% or wrong >= 2."""
    try:
        from coast_content_oma.student.mastery_tier import is_topic_struggling
        from coast_content_oma.student.stores import course_namespace
        ns = course_namespace(user_id, folder)
        orch = _student_orchestrator()

        mistakes_by_concept: dict[str, int] = {}
        for ep in orch.episodes.all(ns):
            ss = ep.store_specific or {}
            if ss.get("episode_type") != "exercise_attempt":
                continue
            if ss.get("outcome") not in ("mistake", "struggle"):
                continue
            for cid in ss.get("concept_ids") or []:
                if cid:
                    mistakes_by_concept[cid] = mistakes_by_concept.get(cid, 0) + 1

        out: list[dict] = []
        seen: set[str] = set()
        for it in orch.mastery.all(ns):
            ss = it.store_specific or {}
            cid = ss.get("concept_id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            mistakes = mistakes_by_concept.get(cid, 0)
            successes = int(ss.get("successes", 0) or 0)
            if not is_topic_struggling(successes, mistakes):
                continue
            out.append({
                "concept_id": cid,
                "name": ss.get("concept_name") or cid,
                "mistakes": mistakes,
                "successes": successes,
            })
        for cid, mistakes in mistakes_by_concept.items():
            if cid in seen:
                continue
            if not is_topic_struggling(0, mistakes):
                continue
            out.append({
                "concept_id": cid,
                "name": _lookup_concept_name(user_id, folder, cid),
                "mistakes": mistakes,
                "successes": 0,
            })
        out.sort(key=lambda t: (-t["mistakes"], t.get("name") or ""))
        return out
    except Exception:
        return []


# ── Pedro grading tags (machine-readable, stripped in the UI) ─────────

TAG_SECTION_COMPLETE = "[SECTION_COMPLETE]"
TAG_TEST_OUT_PASSED = "[TEST_OUT_PASSED]"
TAG_ANSWER_WRONG = "[ANSWER_WRONG]"
TAG_ANSWER_CORRECT = "[ANSWER_CORRECT]"
_PEDRO_TAGS = (TAG_SECTION_COMPLETE, TAG_TEST_OUT_PASSED, TAG_ANSWER_WRONG, TAG_ANSWER_CORRECT)


def strip_pedro_tags(text: str) -> str:
    out = text or ""
    for tag in _PEDRO_TAGS:
        out = out.replace(tag, "")
    return out.strip()


def _answer_outcome_from_tags(assistant_response: str) -> str | None:
    """Return struggle/success when Pedro emitted a grading tag, else None."""
    ar = assistant_response or ""
    if TAG_ANSWER_WRONG in ar:
        return "struggle"
    if TAG_ANSWER_CORRECT in ar:
        return "success"
    return None

def _merge_concept_refs(*groups: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for group in groups:
        for c in group or []:
            cid = c.get("concept_id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            out.append({
                "concept_id": cid,
                "concept_name": c.get("concept_name") or cid,
            })
    return out


def _enrich_concept_refs_for_section(
    user_id: int | str,
    folder: str,
    section_index: int,
    concept_refs: list[dict],
    user_msg: str,
    assistant_msg: str,
) -> list[dict]:
    """Attach section concepts when text inference missed them."""
    try:
        import lesson as lesson_mod

        section_refs = lesson_mod.get_section_concept_refs(user_id, folder, section_index)
        if not section_refs:
            return concept_refs

        text = f"{user_msg} {assistant_msg}".lower()
        matched: list[dict] = []
        for c in section_refs:
            name = (c.get("concept_name") or "").lower()
            if name and (name in text or (len(name) >= 5 and name in text)):
                matched.append(c)

        if matched:
            merged = _merge_concept_refs(concept_refs, matched)
        else:
            merged = concept_refs

        struggled = bool(
            _TUTOR_CORRECTION.search(assistant_msg or "")
            or _STRUGGLE_HINTS.search(user_msg or "")
        )
        if struggled and section_refs:
            return _merge_concept_refs(merged, section_refs[:4])
        return merged
    except Exception:
        return concept_refs


def _record_graded_mistake(
    user_id: int | str,
    folder: str,
    user_message: str,
    assistant_response: str,
    *,
    section_index: Optional[int] = None,
    focus_concept_id: Optional[str] = None,
    duration_sec: Optional[int] = None,
) -> None:
    """Log a one-off wrong answer. Does not update mastery — Pedro re-teaches."""
    rec = _student_recorder_singleton()
    clean_response = strip_pedro_tags(assistant_response)
    concept_refs = _concept_refs_for_graded_turn(
        user_id, folder, section_index, focus_concept_id,
    )
    rec.record_episode(
        user_id,
        folder,
        "exercise_attempt",
        summary=user_message[:200],
        outcome="mistake",
        concept_refs=concept_refs,
        user_message=user_message[:1500],
        assistant_response=clean_response[:1500],
        duration_sec=duration_sec,
        signals={"tutor_corrected": True},
        section_index=section_index,
        source="chat",
    )


def _record_practice_success(
    user_id: int | str,
    folder: str,
    *,
    section_index: Optional[int] = None,
    focus_concept_id: Optional[str] = None,
    user_message: str = "",
) -> None:
    """Correct practice answer — updates mastery only, no episode (keeps log lean)."""
    from coast_content_oma.student.stores import course_namespace
    rec = _student_recorder_singleton()
    ns = course_namespace(user_id, folder)
    concept_refs = _concept_refs_for_graded_turn(
        user_id, folder, section_index, focus_concept_id,
    )
    for c in concept_refs:
        rec.mastery.record_evidence(
            ns,
            concept_id=c["concept_id"],
            concept_name=c["concept_name"],
            outcome="success",
        )


def _concept_refs_for_graded_turn(
    user_id: int | str,
    folder: str,
    section_index: Optional[int],
    focus_concept_id: Optional[str],
) -> list[dict]:
    concept_refs: list[dict] = []
    if section_index is not None:
        try:
            import lesson as lesson_mod
            concept_refs = lesson_mod.get_section_concept_refs(
                user_id, folder, section_index,
            )[:4]
        except Exception:
            pass
    if focus_concept_id:
        name = _lookup_concept_name(user_id, folder, focus_concept_id)
        concept_refs = _merge_concept_refs(
            concept_refs,
            [{"concept_id": focus_concept_id, "concept_name": name}],
        )
    return concept_refs


def _record_graded_attempt(
    user_id: int | str,
    folder: str,
    user_message: str,
    assistant_response: str,
    outcome: str,
    *,
    section_index: Optional[int] = None,
    focus_concept_id: Optional[str] = None,
    duration_sec: Optional[int] = None,
    signals: Optional[dict] = None,
) -> None:
    rec = _student_recorder_singleton()
    clean_response = strip_pedro_tags(assistant_response)

    concept_refs: list[dict] = []
    if section_index is not None:
        try:
            import lesson as lesson_mod
            concept_refs = lesson_mod.get_section_concept_refs(
                user_id, folder, section_index,
            )[:4]
        except Exception:
            pass
    if focus_concept_id:
        name = _lookup_concept_name(user_id, folder, focus_concept_id)
        concept_refs = _merge_concept_refs(
            concept_refs,
            [{"concept_id": focus_concept_id, "concept_name": name}],
        )

    sig = dict(signals or {})
    if outcome == "struggle":
        sig.setdefault("tutor_corrected", True)
    elif outcome == "success":
        sig.setdefault("tutor_affirmed", True)

    rec.record_episode(
        user_id,
        folder,
        "exercise_attempt",
        summary=user_message[:200],
        outcome=outcome,
        concept_refs=concept_refs,
        user_message=user_message[:1500],
        assistant_response=clean_response[:1500],
        duration_sec=duration_sec,
        signals=sig,
        section_index=section_index,
        source="chat",
    )


def record_chat_episode(
    user_id: int | str,
    folder: Optional[str],
    user_message: str,
    assistant_response: str,
    concept_ids_touched: Optional[list[str]] = None,
    duration_sec: Optional[int] = None,
    section_index: Optional[int] = None,
    focus_concept_id: Optional[str] = None,
) -> None:
    """Record graded practice: wrong → mistake log; correct → mastery only."""
    if not is_student_enabled():
        return
    if not folder:
        return
    if _is_lesson_intro(user_message):
        return

    tagged = _answer_outcome_from_tags(assistant_response)
    try:
        if tagged == "struggle":
            _record_graded_mistake(
                user_id, folder, user_message, assistant_response,
                section_index=section_index,
                focus_concept_id=focus_concept_id,
                duration_sec=duration_sec,
            )
        elif tagged == "success":
            _record_practice_success(
                user_id, folder,
                section_index=section_index,
                focus_concept_id=focus_concept_id,
                user_message=user_message,
            )
    except Exception:
        logger.exception("Recording chat episode failed")


def record_section_completed_authoritative(
    user_id: int | str,
    folder: str,
    section_index: int,
    section_title: str,
) -> None:
    """Log section completion from CourseOutline advance — ground truth."""
    if not is_student_enabled():
        return
    try:
        rec = _student_recorder_singleton()

        rec.record_section_completed(
            user_id, folder,
            section_title=section_title,
            section_index=section_index,
        )

        # Mastery comes only from [ANSWER_CORRECT] during verification — not bulk on advance.
        _run_course_consolidation(user_id, folder)
    except Exception:
        logger.exception("Recording authoritative section completion failed")


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
    namespace, then remap Student OMA mastery rows, episode concept_ids,
    and content/image entity references so evidence doesn't fragment.

    Duplicate signal: exact name/alias collision OR embedding cosine
    >= threshold. Canonical node per group = oldest (first created)."""
    from coast_content_oma.stores import make_namespace
    from coast_content_oma.student.stores import course_namespace

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
    id_map: dict[str, str] = {}
    n_merged = 0
    for root, srcs in groups.items():
        for src in srcs:
            if orch.concept.merge_into(ns, src, root):
                id_map[src] = root
                n_merged += 1

    if id_map:
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
