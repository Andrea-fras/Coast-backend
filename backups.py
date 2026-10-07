"""Nightly backups of everything a student would lose if the disk went: both databases, copied
consistently while the app runs, and the files on the disk (uploaded PDFs, extracted images,
generated content).

On the disk, /data/backups/db-<stamp>/ keeps the last three nights, so a bad migration or a mistaken
delete can be undone without the network. Off-site, in Cloudflare R2 (any S3-compatible store):

  db/<stamp>/coast.db.gz, db/<stamp>/oma.db.gz, db/<stamp>/manifest.json   every night, kept 30 days
  files/<path on the disk>                                                 each file once (they never change)
  removed/<path on the disk>                                               when a file left the disk; the copy
                                                                           goes 30 days later, with the last
                                                                           database that refers to it

Configured by R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT and R2_BUCKET; without them only
the copies on the disk are made. scripts/restore_backup.py brings a backup back and checks it.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

EVERY_SECONDS = 24 * 3600
KEEP_ON_DISK = int(os.getenv("BACKUP_KEEP_ON_DISK", "3"))  # R2 holds the longer history
KEEP_DAYS_OFFSITE = int(os.getenv("BACKUP_KEEP_DAYS", "30"))
# Page images rendered on demand ("<pdf>.pages/render-…/") are re-made from the PDF. The page copy
# itself (manifest and figures) is kept: lessons check it, and its figures are OMA's files anyway.
_SKIP_PREFIXES = ("render-",)
_lock = threading.Lock()


def data_root() -> Path:
    """The disk everything lives on: /data on Render, the backend folder locally."""
    if os.getenv("BACKUP_DATA_ROOT"):
        return Path(os.environ["BACKUP_DATA_ROOT"]).resolve()
    return Path("/data") if Path("/data").is_dir() else Path(__file__).resolve().parent


def key(name: str) -> str:
    """An object's key in the bucket; R2_PREFIX keeps tests apart from the real backups."""
    return os.getenv("R2_PREFIX", "") + name


def databases() -> dict[str, Path]:
    import database
    import oma_provider
    return {"coast.db": Path(database.DB_PATH), "oma.db": Path(oma_provider.OMA_DB_PATH)}


def file_dirs() -> list[Path]:
    """Directories whose files must survive: uploads (with their extracted images), OMA's images
    and generated content. Code, caches and the databases themselves are not among them."""
    import oma_provider
    root = data_root()  # the same defaults server.py uses
    dirs = [Path(os.getenv("FOLDER_UPLOADS_DIR", root / "folder_uploads")), Path(oma_provider.OMA_IMAGE_DIR),
            Path(os.getenv("GENERATED_DIR", root / "generated")), Path(os.getenv("MEDIA_DIR", root / "media"))]
    return [d.resolve() for d in dirs if d.is_dir()]


def _r2():
    keys = ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT", "R2_BUCKET")
    if not all(os.getenv(k) for k in keys):
        return None, None
    import boto3
    from botocore.config import Config
    client = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT"], region_name="auto",
                          aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                          aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                          config=Config(retries={"max_attempts": 5, "mode": "standard"}))
    return client, os.environ["R2_BUCKET"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot(src: Path, dst: Path) -> dict:
    """A consistent copy of a live SQLite database (the backup API waits out writers), gzipped."""
    raw = dst.with_suffix("")
    a = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=60)
    b = sqlite3.connect(raw)
    try:
        a.backup(b)
        tables = [r[0] for r in b.execute("select name from sqlite_master where type='table' "
                                          "and name not like 'sqlite_%' and name not like '%_fts%'")]
        rows = {t: b.execute(f'select count(*) from "{t}"').fetchone()[0] for t in tables}
        ok = b.execute("pragma integrity_check").fetchone()[0]
    finally:
        a.close()
        b.close()
    if ok != "ok":
        raise RuntimeError(f"{src.name}: integrity check failed on the copy: {ok}")
    sha = _sha256(raw)
    with open(raw, "rb") as f, gzip.open(dst, "wb", compresslevel=6) as g:
        shutil.copyfileobj(f, g, 1 << 20)
    raw.unlink()
    return {"sha256": sha, "rows": rows, "gz_bytes": dst.stat().st_size}


def _files() -> tuple[dict[str, Path], dict[str, str]]:
    """The files to keep, each stored once, and the second names of hard-linked ones: a figure
    lives in the page copy and in OMA's image folder as one file under two names."""
    root = data_root()
    out, links, first = {}, {}, {}
    for d in file_dirs():
        for p in sorted(d.rglob("*")):
            if p.is_file() and not any(part.startswith(_SKIP_PREFIXES) for part in p.parts):
                try:
                    rel = str(p.relative_to(root))
                except ValueError:  # a directory configured outside the data disk
                    rel = str(p)
                st = p.stat()
                inode = (st.st_dev, st.st_ino)
                if inode in first:
                    links[rel] = first[inode]
                else:
                    first[inode] = rel
                    out[rel] = p
    return out, links


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return os.getenv("RENDER_GIT_COMMIT", "")[:7]


def run_backup() -> dict:
    """One backup now. Returns its manifest. Safe to call while students use the app."""
    with _lock:
        started = time.time()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        local = data_root() / "backups" / f"db-{stamp}"
        local.mkdir(parents=True, exist_ok=True)
        import file_store
        in_store = file_store.enabled()  # files already live in R2 ("store/"): only databases to copy
        files, links = ({}, {}) if in_store else _files()
        manifest = {"stamp": stamp, "created_at": datetime.now(timezone.utc).isoformat(), "commit": _commit(),
                    "data_root": str(data_root()), "databases": {}, "files": {"count": len(files),
                    "bytes": sum(p.stat().st_size for p in files.values()), "in_store": in_store},
                    "links": links, "file_keys": sorted(files)}
        for name, path in databases().items():
            manifest["databases"][name] = {**_snapshot(path, local / f"{name}.gz"), "path": str(path)}
        (local / "manifest.json").write_text(json.dumps(manifest, indent=1))
        _prune_local()

        client, bucket = _r2()
        if client:
            for f in sorted(local.iterdir()):
                client.upload_file(str(f), bucket, key(f"db/{stamp}/{f.name}"))
            if in_store:
                file_store.prune_trash()
                # Files kept the old way ("files/", before the store) expire with the backups that use them.
                cutoff = datetime.now(timezone.utc) - timedelta(days=KEEP_DAYS_OFFSITE)
                legacy = [n for prefix in ("files/", "removed/")
                          for n, o in _listing(client, bucket, prefix).items() if o["LastModified"] < cutoff]
                _delete(client, bucket, legacy)
                manifest["offsite"] = {"files_in_store": True, "legacy_expired": len(legacy)}
            else:
                manifest["offsite"] = _sync_files(client, bucket, files)
            _prune_offsite(client, bucket)
            client.put_object(Bucket=bucket, Key=key(f"db/{stamp}/manifest.json"),
                              Body=json.dumps(manifest, indent=1).encode())
        else:
            manifest["offsite"] = None
        manifest["seconds"] = round(time.time() - started, 1)
        (local / "manifest.json").write_text(json.dumps(manifest, indent=1))
        (data_root() / "backups" / "last.json").write_text(json.dumps(manifest, indent=1))
        log.info("backup %s done in %ss (offsite=%s)", stamp, manifest["seconds"], bool(client))
        return manifest


def _listing(client, bucket: str, prefix: str) -> dict[str, dict]:
    """Objects under `prefix`, keyed without R2_PREFIX."""
    out, cut = {}, len(key(""))
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=key(prefix)):
        for obj in page.get("Contents", []):
            out[obj["Key"][cut:]] = obj
    return out


_PARALLEL = 16  # requests at once: one at a time, 3,700 files took ~16 min across the Atlantic


def _parallel(fn, items) -> None:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(_PARALLEL) as pool:  # boto3 clients are thread-safe
        list(pool.map(fn, items))


def _delete(client, bucket: str, names) -> None:
    """Remove objects in batches of 1,000 (one request each)."""
    names = list(names)
    for i in range(0, len(names), 1000):
        client.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": key(n)} for n in names[i:i + 1000]], "Quiet": True})


def _sync_files(client, bucket: str, files: dict[str, Path]) -> dict:
    remote = _listing(client, bucket, "files/")
    removed = _listing(client, bucket, "removed/")
    send = [(rel, path) for rel, path in files.items()
            if f"files/{rel}" not in remote or remote[f"files/{rel}"]["Size"] != path.stat().st_size]
    _parallel(lambda item: client.upload_file(str(item[1]), bucket, key(f"files/{item[0]}")), send)
    _delete(client, bucket, [f"removed/{rel}" for rel in files if f"removed/{rel}" in removed])  # back on the disk
    stamp = datetime.now(timezone.utc).isoformat().encode()
    gone = [name[len("files/"):] for name in remote if name[len("files/"):] not in files and f"removed/{name[len('files/'):]}" not in removed]
    _parallel(lambda rel: client.put_object(Bucket=bucket, Key=key(f"removed/{rel}"), Body=stamp), gone)
    return {"files_sent": len(send), "files_offsite": len(set(remote) | {f"files/{r}" for r in files}), "marked_removed": len(gone)}


def _prune_local() -> None:
    dirs = sorted((data_root() / "backups").glob("db-*"))
    for old in dirs[:-KEEP_ON_DISK]:
        shutil.rmtree(old, ignore_errors=True)


def _prune_offsite(client, bucket: str) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=KEEP_DAYS_OFFSITE)
    stamps = sorted({k.split("/")[1] for k in _listing(client, bucket, "db/")})
    for stamp in stamps[:-KEEP_ON_DISK]:  # always keep the newest few, however old
        if datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc) < cutoff:
            _delete(client, bucket, _listing(client, bucket, f"db/{stamp}/"))
    expired = [name for name, obj in _listing(client, bucket, "removed/").items() if obj["LastModified"] < cutoff]
    # gone from the disk for longer than any kept database: the file and its marker go
    _delete(client, bucket, [f"files/{name[len('removed/'):]}" for name in expired] + expired)


def last_backup() -> Optional[dict]:
    try:
        return json.loads((data_root() / "backups" / "last.json").read_text())
    except (OSError, ValueError):
        return None


def _loop() -> None:
    time.sleep(300)  # let startup and recovery jobs settle first
    while True:
        last = last_backup()
        age = time.time() - datetime.fromisoformat(last["created_at"]).timestamp() if last else None
        if age is None or age >= EVERY_SECONDS:
            try:
                run_backup()
            except Exception:
                log.exception("backup failed; retrying within the hour")
        time.sleep(3600)


def start() -> None:
    """Back up once a day from the running server (one process holds the disk on Render).
    On by default on Render; BACKUPS=on or off overrides."""
    setting = os.getenv("BACKUPS", "on" if os.getenv("RENDER") else "off").lower()
    if setting in ("on", "1", "true", "yes"):
        threading.Thread(target=_loop, name="coast-backups", daemon=True).start()
