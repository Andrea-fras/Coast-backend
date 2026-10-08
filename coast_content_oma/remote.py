"""Reading and indexing course files in containers (Modal), away from the web server.

The container does the heavy part of a course file: reading the PDF (text, figures, page copy)
and every AI call that classifies pages and describes figures. It reaches files through the file
store, at the same paths as on the server (fetched from R2), and streams each finished batch
back; the server applies the batches to its databases exactly as when it does the work itself,
so a section still opens before its whole file is read. If no container can be reached, the
server does the work itself: callers fall back.

On with INDEX_IN_CONTAINERS=on, MODAL_TOKEN_ID/MODAL_TOKEN_SECRET set and the file store on.
coast_modal.py defines the container; Render's build deploys it with every push.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Iterator

log = logging.getLogger(__name__)

APP = "coast-indexing"
DOWN_FOR = 300  # after a failed call, the server works alone this long (s)
_down_until = 0.0


def enabled() -> bool:
    if os.getenv("INDEX_IN_CONTAINERS", "off").lower() not in ("on", "1", "true", "yes"):
        return False
    if time.monotonic() < _down_until:
        return False  # containers just failed: memory admission and workers plan for local work
    if not (os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET")):
        return False
    import file_store
    return file_store.enabled()


def _context() -> dict:
    """Where the server keeps its files, so a container uses the same paths and R2 keys."""
    import backups
    return {"data_root": str(backups.data_root()), "r2_prefix": os.getenv("R2_PREFIX", "")}


# ── what runs inside the container ────────────────────────────────────────────
def _adopt(context: dict) -> None:
    os.environ["BACKUP_DATA_ROOT"] = context["data_root"]
    os.environ["R2_PREFIX"] = context["r2_prefix"]
    Path(context["data_root"]).mkdir(parents=True, exist_ok=True)


def read_upload_work(path: str, context: dict) -> dict:
    """Read an uploaded file (its page copy goes to R2); returns the pages' text and what R2 now
    holds, so the server can record it."""
    import json
    import file_store
    from .extraction import extract_and_cache
    from .normalized_source import cache_dir
    import time
    _adopt(context)
    marks = [time.monotonic()]
    file_store.local(path)
    marks.append(time.monotonic())
    pages = json.loads(json.dumps(extract_and_cache(path)))  # as the server's reader hands them over
    marks.append(time.monotonic())
    directory = cache_dir(Path(path))
    file_store.publish_tree(directory, wait=True)
    marks.append(time.monotonic())
    held = [(file_store.key_for(p), p.stat().st_size) for p in directory.rglob("*") if p.is_file()]
    took = {name: round(b - a, 1) for name, a, b in zip(("fetch", "read", "publish"), marks, marks[1:])}
    return {"pages": pages, "published": held, "took": took}


def index_work(job: dict) -> Iterator[tuple]:
    """Classify a file's pages and describe its figures, yielding each finished batch:
    ("pages", [(page_number, result)]), ("figures", [row]), ("usage", [record]), then
    ("figures_left", [file_path]) for figures no description came back for (the server's
    background sweep retries them, as when it reads a file itself) and ("done", {})."""
    import ai_usage
    _adopt(job["context"])
    usage: list = []
    usage_lock = threading.Lock()

    def keep(provider, record, *, model="", latency_ms=0, ok=True):  # the server records it
        with usage_lock:
            usage.append({"provider": provider, "usage": record, "model": model, "latency_ms": latency_ms, "ok": ok})
    original, ai_usage.record = ai_usage.record, keep
    try:
        yield from _index_batches(job, usage, usage_lock)
    finally:
        ai_usage.record = original


def _index_batches(job: dict, usage: list, usage_lock) -> Iterator[tuple]:
    from .ingestion import IngestionPipeline

    pipeline = IngestionPipeline.__new__(IngestionPipeline)  # its compute only: no databases here
    for name, value in job["settings"].items():
        setattr(pipeline, name, value)
    ranks = {int(page): tuple(rank) for page, rank in job.get("ranks", {}).items()}
    priority = lambda page: ranks.get(page, (100000, page))
    out: queue.Queue = queue.Queue()

    def classify():
        pipeline._classify_pages_parallel(
            job["pages"], lambda pairs: out.put(("pages", [(p["page_number"], r) for p, r in pairs])), priority)

    def vision():
        if not job.get("images"):
            return
        sent: set = set()

        def send(rows):
            sent.update(row["file_path"] for row in rows)
            out.put(("figures", rows))
        final = pipeline._describe_saved_images_batched(job["images"], send, priority)
        out.put(("figures_left", [row["file_path"] for row in final if row["file_path"] not in sent]))

    def run(fn):
        try:
            fn()
        except Exception as exc:
            out.put(("error", f"{type(exc).__name__}: {exc}"))
    workers = [threading.Thread(target=run, args=(fn,), daemon=True) for fn in (classify, vision)]
    for w in workers:
        w.start()
    while any(w.is_alive() for w in workers) or not out.empty():
        try:
            item = out.get(timeout=0.5)
        except queue.Empty:
            continue
        with usage_lock:
            records, usage[:] = list(usage), []
        if records:
            yield ("usage", records)
        yield item
    with usage_lock:
        if usage:
            yield ("usage", list(usage))
    yield ("done", {})


# ── what the server calls ─────────────────────────────────────────────────────
def _note_down(exc: BaseException) -> None:
    global _down_until
    _down_until = time.monotonic() + DOWN_FOR
    log.warning("containers unavailable (%s: %s); the server reads and indexes alone for %ds",
                type(exc).__name__, exc, DOWN_FOR)

_indexer: list = []  # the deployed container class, looked up once


def _function(name: str):
    """A method of the deployed Indexer (coast_modal.py): read_upload or index_pages."""
    import modal
    if not _indexer:
        _indexer.append(modal.Cls.from_name(APP, "Indexer")())
    return getattr(_indexer[0], name)


async def read_upload(path: str) -> list[dict]:
    """The pages' text of an upload read in a container (its page copy lands in R2). Sending and
    recording happen off the event loop: every other request is served meanwhile."""
    import asyncio
    import file_store
    path = str(Path(path).resolve())  # as the R2 key is made
    started = time.monotonic()
    await asyncio.to_thread(file_store.publish, path, True)  # the container fetches it from R2
    sent = time.monotonic()
    try:
        result = await _function("read_upload").remote.aio(path, _context())
    except Exception as exc:
        _note_down(exc)
        raise
    await asyncio.to_thread(file_store.mark_many, [(k, size) for k, size in result["published"] if k])
    print(f"[upload] {Path(path).name}: to R2 {sent - started:.1f}s, read in a container {time.monotonic() - sent:.1f}s "
          f"{result.get('took')}")
    return result["pages"]


def index(pages: list[dict], images: list[dict], priority: Callable, on_pages: Callable, on_figures: Callable,
          settings: dict) -> dict:
    """Run a file's AI work in a container, applying each batch here as it arrives. Returns what
    was applied ("left": figures left for the background sweep), so a caller that falls back
    after an interruption only redoes the rest."""
    import ai_usage
    import file_store
    file_store.ensure_published([i["file_path"] for i in images])
    by_number = {p["page_number"]: p for p in pages}
    job = {
        "pages": [{"page_number": p["page_number"], "text": (p.get("text") or "").strip()[:3000]} for p in pages],
        "images": [{k: i.get(k) for k in ("page_number", "file_path", "width", "height", "context_hint",
                                          "also_on_pages", "bbox", "img_idx")} for i in images],
        "ranks": {str(p["page_number"]): list(priority(p["page_number"])) for p in pages}
                 | {str(i["page_number"]): list(priority(i["page_number"])) for i in images},
        "settings": settings,  # the server pipeline's batch sizes and workers
        "context": _context(),
    }
    applied = {"pages": set(), "figures": set(), "left": set()}
    try:
        for kind, payload in _function("index_pages").remote_gen(job):
            if kind == "pages":
                on_pages([(by_number[n], r) for n, r in payload if n in by_number])
                applied["pages"] |= {n for n, _ in payload}
            elif kind == "figures":
                on_figures(payload)
                applied["figures"] |= {row["file_path"] for row in payload}
            elif kind == "figures_left":
                applied["left"] |= set(payload)
            elif kind == "usage":
                for r in payload:
                    ai_usage.record(r["provider"], r["usage"], model=r.get("model") or "", latency_ms=r["latency_ms"], ok=r["ok"])
            elif kind == "error":
                log.warning("container indexing reported: %s", payload)
    except Exception as exc:  # what arrived stays applied; the caller does the rest
        log.exception("container indexing stopped after %d pages and %d figures",
                      len(applied["pages"]), len(applied["figures"]))
        _note_down(exc)
    return applied
