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
_SKIP_SUFFIXES = (".pages",)  # page renders, re-made from the PDF on demand
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


def _files() -> dict[str, Path]:
    root = data_root()
    out = {}
    for d in file_dirs():
        for p in d.rglob("*"):
            if p.is_file() and not any(part.endswith(_SKIP_SUFFIXES) for part in p.parts):
                try:
                    out[str(p.relative_to(root))] = p
                except ValueError:  # a directory configured outside the data disk
                    out[str(p)] = p
    return out


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
        files = _files()
        manifest = {"stamp": stamp, "created_at": datetime.now(timezone.utc).isoformat(), "commit": _commit(),
                    "data_root": str(data_root()), "databases": {}, "files": {"count": len(files),
                    "bytes": sum(p.stat().st_size for p in files.values())}}
        for name, path in databases().items():
            manifest["databases"][name] = {**_snapshot(path, local / f"{name}.gz"), "path": str(path)}
        (local / "manifest.json").write_text(json.dumps(manifest, indent=1))
        _prune_local()

        client, bucket = _r2()
        if client:
            for f in sorted(local.iterdir()):
                client.upload_file(str(f), bucket, key(f"db/{stamp}/{f.name}"))
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


def _sync_files(client, bucket: str, files: dict[str, Path]) -> dict:
    remote = _listing(client, bucket, "files/")
    removed = _listing(client, bucket, "removed/")
    sent = 0
    for rel, path in files.items():
        name = f"files/{rel}"
        if name not in remote or remote[name]["Size"] != path.stat().st_size:
            client.upload_file(str(path), bucket, key(name))
            sent += 1
        if f"removed/{rel}" in removed:  # back on the disk (restored): no longer due to go
            client.delete_object(Bucket=bucket, Key=key(f"removed/{rel}"))
    now = datetime.now(timezone.utc)
    for name in remote:
        rel = name[len("files/"):]
        if rel not in files and f"removed/{rel}" not in removed:
            client.put_object(Bucket=bucket, Key=key(f"removed/{rel}"), Body=now.isoformat().encode())
    return {"files_sent": sent, "files_offsite": len(set(remote) | {f"files/{r}" for r in files})}


def _prune_local() -> None:
    dirs = sorted((data_root() / "backups").glob("db-*"))
    for old in dirs[:-KEEP_ON_DISK]:
        shutil.rmtree(old, ignore_errors=True)


def _prune_offsite(client, bucket: str) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=KEEP_DAYS_OFFSITE)
    stamps = sorted({k.split("/")[1] for k in _listing(client, bucket, "db/")})
    for stamp in stamps[:-KEEP_ON_DISK]:  # always keep the newest few, however old
        if datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc) < cutoff:
            for name in _listing(client, bucket, f"db/{stamp}/"):
                client.delete_object(Bucket=bucket, Key=key(name))
    for name, obj in _listing(client, bucket, "removed/").items():
        if obj["LastModified"] < cutoff:  # gone from the disk for longer than any kept database
            client.delete_object(Bucket=bucket, Key=key("files/" + name[len("removed/"):]))
            client.delete_object(Bucket=bucket, Key=key(name))


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
