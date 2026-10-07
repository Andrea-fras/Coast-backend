"""Cloudflare R2 holds every uploaded file; the server's disk is a cache in front of it.

A file's key mirrors its path under the data root, under "store/": "store/folder_uploads/src_1.pdf",
"store/folder_uploads/src_1.pdf.pages/p3_i0.webp". Code that writes a file publishes it (an upload
in the background); code that reads one asks for local(path), which fetches it from R2 if the disk
no longer has it. When the disk fills, files R2 already holds that nobody used recently are
cleared from it. A deleted file moves to "trash/" in R2 for 30 days, so a backup restored from
before the deletion still finds it.

On when R2 is configured and FILE_STORE is not "off" (on Render by default; locally FILE_STORE=on
turns it on). Off, every function leaves the local disk exactly as it was.
"""
from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger(__name__)

STORE, TRASH = "store/", "trash/"
TRASH_DAYS = 30
EVICT_ABOVE, EVICT_TO = 0.75, 0.60  # share of the disk in use
_TOUCH_AFTER = 3600  # a read refreshes a file's place in the cache at most hourly
_pool = ThreadPoolExecutor(8, thread_name_prefix="coast-store")
_db_lock = threading.Lock()
_fetching: dict[str, threading.Lock] = {}
_fetch_lock = threading.Lock()


_r2_cache: list = []  # (client, bucket), made once: a boto3 client is thread-safe and slow to create


def _client():
    if not _r2_cache:
        import backups
        _r2_cache.append(backups._r2())
    return _r2_cache[0]


def enabled() -> bool:
    setting = os.getenv("FILE_STORE", "on" if os.getenv("RENDER") else "off").lower()
    return setting not in ("off", "0", "false", "no") and _client()[0] is not None


def _obj(key: str) -> str:
    """The object's name in the bucket (R2_PREFIX keeps tests apart, as for backups)."""
    import backups
    return backups.key(key)


def key_for(path) -> Optional[str]:
    """The R2 key of a file under the data root (None for files that live elsewhere, like the
    premade workshops shipped with the code)."""
    import backups
    try:
        return STORE + Path(path).resolve().relative_to(backups.data_root().resolve()).as_posix()
    except (ValueError, OSError):
        return None


# ── which files R2 holds ──────────────────────────────────────────────────────
def _db() -> sqlite3.Connection:
    import backups
    conn = sqlite3.connect(backups.data_root() / "file_store.db", timeout=30)
    conn.execute("pragma journal_mode=wal")
    conn.execute("create table if not exists stored (key text primary key, size integer, published real)")
    return conn


def _mark(key: str, size: int) -> None:
    with _db_lock, _db() as c:
        c.execute("insert or replace into stored values (?, ?, ?)", (key, size, time.time()))


def _forget(keys: Iterable[str]) -> None:
    keys = list(keys)
    with _db_lock, _db() as c:
        c.executemany("delete from stored where key = ?", [(k,) for k in keys])


def published(key: str) -> bool:
    with _db() as c:
        return c.execute("select 1 from stored where key = ?", (key,)).fetchone() is not None


# ── writing ───────────────────────────────────────────────────────────────────
def _upload(path: Path, key: str) -> None:
    client, bucket = _client()
    for attempt in range(5):
        try:
            client.upload_file(str(path), bucket, _obj(key))
            _mark(key, path.stat().st_size)
            return
        except FileNotFoundError:
            return  # deleted before it was sent
        except Exception:
            if attempt == 4:
                log.exception("file store: could not upload %s (the nightly sweep will retry)", key)
            time.sleep(2 ** attempt)


def publish(path, wait: bool = False) -> None:
    """Send a newly written file to R2 (in the background unless wait)."""
    if not enabled():
        return
    path = Path(path)
    key = key_for(path)
    if key and path.is_file():
        future = _pool.submit(_upload, path, key)
        if wait:
            future.result()


def publish_tree(directory, wait: bool = False) -> None:
    """Every file in a folder (a page copy: its manifest and figures)."""
    directory = Path(directory)
    if enabled() and directory.is_dir():
        futures = [_pool.submit(_upload, p, key_for(p)) for p in directory.rglob("*")
                   if p.is_file() and key_for(p) and not _regenerable(p)]
        if wait:
            for f in futures:
                f.result()


def _regenerable(path: Path) -> bool:
    """Page images rendered on demand are re-made from the PDF; they never go to R2."""
    return any(part.startswith("render-") for part in path.parts)


# ── reading ───────────────────────────────────────────────────────────────────
def local(path) -> Path:
    """The file on the local disk, fetched from R2 first if the cache no longer has it."""
    path = Path(path)
    if path.is_file():
        try:
            if time.time() - path.stat().st_mtime > _TOUCH_AFTER:
                os.utime(path)  # recently used: kept in the cache longer
        except OSError:
            pass
        return path
    key = key_for(path) if enabled() else None
    if not key:
        return path
    with _fetch_lock:
        lock = _fetching.setdefault(key, threading.Lock())
    with lock:  # one download per file, however many readers ask at once
        if not path.is_file():
            client, bucket = _client()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".part")
            try:
                client.download_file(bucket, _obj(key), str(tmp))
                os.replace(tmp, path)
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                log.warning("file store: %s is neither on disk nor in R2 (%s)", key, type(exc).__name__)
    return path


def local_tree(directory, names: Iterable[str]) -> None:
    """Make sure the named files of a folder are on the disk, fetching missing ones together."""
    directory = Path(directory)
    missing = [directory / n for n in names if not (directory / n).is_file()]
    if missing and enabled():
        list(_pool.map(local, missing))


# ── deleting ──────────────────────────────────────────────────────────────────
def remove(paths: Iterable) -> None:
    """Files or folders deleted on purpose: gone from the disk now, kept in R2's trash for 30 days
    (prune_trash removes them for good). Goes by what R2 holds, so a page copy the cache had
    already cleared is removed too."""
    keys: set[str] = set()
    for path in paths:
        path = Path(path)
        key = key_for(path)
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
        if key and enabled():
            with _db() as c:  # the file itself, or everything under the folder
                keys |= {k for (k,) in c.execute(
                    "select key from stored where key = ? or substr(key, 1, ?) = ?", (key, len(key) + 1, key + "/"))}
    if keys:
        _pool.submit(_trash, sorted(keys))


def _trash(keys: list[str]) -> None:
    client, bucket = _client()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")

    def keep(key):
        try:
            client.copy_object(Bucket=bucket, Key=_obj(f"{TRASH}{stamp}/{key[len(STORE):]}"),
                               CopySource={"Bucket": bucket, "Key": _obj(key)})
        except Exception:
            pass  # never sent (an upload that failed): nothing to keep
    with ThreadPoolExecutor(16) as copies:  # a course's 100 files in about a second, not 25
        list(copies.map(keep, keys))
    for i in range(0, len(keys), 1000):
        client.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": _obj(k)} for k in keys[i:i + 1000]], "Quiet": True})
    _forget(keys)


# ── housekeeping ──────────────────────────────────────────────────────────────
def evict(disk=None) -> int:
    """Clear files R2 holds from the disk, least recently used first, once the disk is fuller than
    EVICT_ABOVE; page renders older than a day go first. Returns how many files were cleared."""
    import backups
    disk = disk or backups.data_root()
    usage = shutil.disk_usage(disk)
    if usage.used / usage.total < EVICT_ABOVE:
        return 0
    target = usage.total * EVICT_TO
    freed, cleared = usage.used, 0
    root = backups.data_root()
    renders = sorted((p for p in root.rglob("render-*/*") if p.is_file()), key=lambda p: p.stat().st_mtime)
    fresh = {p for p in renders if time.time() - p.stat().st_mtime < 86400}  # in use today: kept
    with _db() as c:
        held = [root / k[len(STORE):] for (k,) in c.execute("select key from stored")]
    candidates = [p for p in renders if p not in fresh] + sorted((p for p in held if p.is_file()), key=lambda p: p.stat().st_mtime)
    for p in candidates:
        if freed <= target:
            break
        try:
            size = p.stat().st_size
            p.unlink()
            freed -= size
            cleared += 1
        except OSError:
            pass
    log.info("file store: cleared %d files from the disk cache", cleared)
    return cleared


def sweep(dirs: Iterable[Path]) -> int:
    """Send any file under these folders that R2 does not hold yet (files from before the store,
    or uploads that failed). Returns how many were sent."""
    if not enabled():
        return 0
    with _db() as c:
        held = {k for (k,) in c.execute("select key from stored")}
    todo = [(p, key_for(p)) for d in dirs if Path(d).is_dir() for p in Path(d).rglob("*")
            if p.is_file() and not _regenerable(p) and not p.name.endswith(".part")]
    todo = [(p, k) for p, k in todo if k and k not in held]
    list(_pool.map(lambda item: _upload(*item), todo))
    return len(todo)


def prune_trash() -> None:
    """Remove trashed files older than TRASH_DAYS for good."""
    if not enabled():
        return
    import backups
    client, bucket = _client()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=TRASH_DAYS)).strftime("%Y%m%d")
    old = [k for k in backups._listing(client, bucket, TRASH) if k[len(TRASH):len(TRASH) + 8] < cutoff]
    for i in range(0, len(old), 1000):
        client.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": backups.key(k)} for k in old[i:i + 1000]], "Quiet": True})


def start(dirs: Iterable[Path]) -> None:
    """On server start: send what R2 lacks, then keep the disk cache in bounds every 10 minutes."""
    if not enabled():
        return
    dirs = list(dirs)

    def loop():
        time.sleep(60)
        try:
            log.info("file store: sent %d files R2 did not hold", sweep(dirs))
        except Exception:
            log.exception("file store sweep failed")
        while True:
            try:
                evict()
            except Exception:
                log.exception("file store eviction failed")
            time.sleep(600)
    threading.Thread(target=loop, name="coast-file-store", daemon=True).start()
