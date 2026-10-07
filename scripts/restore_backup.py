"""Bring a backup back and prove it works.

    python3 scripts/restore_backup.py --to /tmp/restore                # the latest night, from R2
    python3 scripts/restore_backup.py --to /tmp/restore --stamp 20261008T030000Z
    python3 scripts/restore_backup.py --to /tmp/restore --local backups/db-<stamp>   # databases only, from the disk

The restored folder has the disk's layout (coast.db, oma_data/oma.db, folder_uploads/, …), so in a
real disaster its contents go onto a fresh /data disk as they are. Before saying it worked, it checks:
the databases match the backup's checksums and row counts and pass SQLite's integrity check, every
uploaded file the database names is there, and the app itself, started on the restored data, serves
students their own courses, sources, conversations and notes.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

OCR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OCR))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(args, to: Path) -> dict:
    """The databases (gzipped) and manifest into `to`, and from R2 every file too."""
    to.mkdir(parents=True, exist_ok=True)
    if args.local:
        src = Path(args.local)
        for f in src.iterdir():
            shutil.copy2(f, to / f.name)
        return json.loads((to / "manifest.json").read_text())
    from dotenv import load_dotenv
    load_dotenv(OCR / ".env")
    import backups
    client, bucket = backups._r2()
    if not client:
        sys.exit("R2 is not configured (R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT, R2_BUCKET)")
    stamps = sorted({k.split("/")[1] for k in backups._listing(client, bucket, "db/")})
    stamp = stamps[-1] if args.stamp in (None, "latest") else args.stamp
    if stamp not in stamps:
        sys.exit(f"no backup {stamp}; available: {', '.join(stamps[-10:])}")
    for name in backups._listing(client, bucket, f"db/{stamp}/"):
        client.download_file(bucket, backups.key(name), str(to / name.split("/")[-1]))
    files = backups._listing(client, bucket, "files/")
    got = 0
    for name, obj in files.items():
        dest = to / name[len("files/"):]
        if dest.exists() and dest.stat().st_size == obj["Size"]:
            continue  # resumed run
        dest.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, backups.key(name), str(dest))
        got += 1
    print(f"fetched backup {stamp}: databases and {len(files)} files ({got} downloaded now)")
    return json.loads((to / "manifest.json").read_text())


def unpack(manifest: dict, to: Path) -> dict[str, Path]:
    """coast.db and oma_data/oma.db in place, each checked against the manifest."""
    places = {"coast.db": to / "coast.db", "oma.db": to / "oma_data" / "oma.db"}
    for name, dest in places.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(to / f"{name}.gz", "rb") as g, open(dest, "wb") as f:
            shutil.copyfileobj(g, f, 1 << 20)
        want = manifest["databases"][name]
        if sha256(dest) != want["sha256"]:
            sys.exit(f"FAIL {name}: checksum differs from the backup")
        c = sqlite3.connect(dest)
        ok = c.execute("pragma integrity_check").fetchone()[0]
        rows = {t: c.execute(f'select count(*) from "{t}"').fetchone()[0] for t in want["rows"]}
        c.close()
        if ok != "ok" or rows != want["rows"]:
            sys.exit(f"FAIL {name}: integrity {ok}; rows differ: "
                     f"{ {t: (rows[t], n) for t, n in want['rows'].items() if rows[t] != n} }")
        print(f"ok   {name}: checksum, integrity and {sum(rows.values())} rows in {len(rows)} tables match")
    return places


def check_files(manifest: dict, to: Path, with_files: bool) -> None:
    c = sqlite3.connect(to / "coast.db")
    paths = [p for (p,) in c.execute("select file_path from folder_sources where file_path is not null and file_path != ''")]
    c.close()
    root = manifest["data_root"].rstrip("/") + "/"
    on_disk = [p for p in paths if p.startswith(root)]
    # A path outside the data disk (the code folder) was never the disk's to keep: files shipped with
    # the code, or uploads saved there before the disk existed and lost at the next deploy.
    elsewhere = len(paths) - len(on_disk)
    missing = [p for p in on_disk if not (to / p[len(root):]).exists()]
    note = f"; {elsewhere} named outside the disk, which no backup holds" if elsewhere else ""
    if not with_files:
        print(f"skip files: databases-only restore ({len(paths)} uploads are named in the database)")
    elif missing:
        sys.exit(f"FAIL files: {len(missing)} of {len(on_disk)} uploads on the disk missing, e.g. {missing[:3]}{note}")
    else:
        print(f"ok   files: all {len(on_disk)} uploads on the disk are present{note}")


APP_CHECK = r'''
import json, os, sqlite3, sys
sys.path.insert(0, os.environ["OCR"])
os.chdir(os.environ["OCR"])
from fastapi.testclient import TestClient
import auth, server
client = TestClient(server.app)  # no startup hooks: nothing runs in the background, nothing is written
assert client.get("/api/health").json() == {"status": "ok"}
db = sqlite3.connect(os.environ["DATABASE_PATH"])
users = db.execute("select u.id, u.email from users u join chat_messages m on m.user_id = u.id "
                   "group by u.id order by count(*) desc limit 5").fetchall()
checked = []
for uid, email in users:
    h = {"Authorization": "Bearer " + auth.create_access_token(uid, email)}
    me = client.get("/api/auth/me", headers=h).json()
    assert me.get("id") == uid, me
    folders = client.get("/api/notebooks/folders", headers=h).json()
    want = [r[0] for r in db.execute("select name from study_folders where user_id=? order by created_at", (uid,))]
    assert folders == want, (folders, want)
    convs = client.get("/api/chat/conversations", headers=h)
    assert convs.status_code == 200, convs.text
    sources = 0
    for name in folders[:3]:
        r = client.get(f"/api/folders/{name}/sources", headers=h)
        assert r.status_code == 200, r.text
        sources += len(r.json() if isinstance(r.json(), list) else r.json().get("sources", []))
    notes = client.get("/api/lesson-notes/all", headers=h)
    assert notes.status_code == 200, notes.text
    checked.append({"user": uid, "folders": len(folders), "sources_in_first_3": sources})
print(json.dumps(checked))
'''


def check_app(to: Path, places: dict[str, Path]) -> None:
    env = {**os.environ, "OCR": str(OCR), "DATABASE_PATH": str(places["coast.db"]),
           "OMA_DB_PATH": str(places["oma.db"]), "OMA_IMAGE_DIR": str(to / "oma_data" / "images"),
           "FOLDER_UPLOADS_DIR": str(to / "folder_uploads"), "GENERATED_DIR": str(to / "generated"),
           "CHROMA_PATH": str(to / "chroma"), "BACKUPS": "off", "JWT_SECRET": os.urandom(16).hex()}
    env.pop("RENDER", None)
    r = subprocess.run([sys.executable, "-c", APP_CHECK], env=env, capture_output=True, text=True, timeout=600)
    if r.returncode:
        sys.exit("FAIL app on restored data:\n" + r.stderr[-3000:])
    users = json.loads(r.stdout.strip().splitlines()[-1])
    print(f"ok   app: started on the restored data and served {len(users)} students their own account, "
          f"courses, sources, conversations and notes: {users}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--to", required=True, help="an empty folder to restore into")
    ap.add_argument("--stamp", help="which night (default: the latest)")
    ap.add_argument("--local", help="a backups/db-<stamp> folder on the disk instead of R2 (no files)")
    args = ap.parse_args()
    to = Path(args.to).resolve()
    if to.exists() and any(p for p in to.iterdir() if p.name not in ("folder_uploads", "oma_data", "generated")):
        sys.exit(f"{to} is not empty")
    manifest = fetch(args, to)
    places = unpack(manifest, to)
    check_files(manifest, to, with_files=not args.local)
    check_app(to, places)
    print(f"RESTORE OK: backup {manifest['stamp']} (commit {manifest.get('commit') or '?'}) is complete and usable at {to}")


if __name__ == "__main__":
    main()
