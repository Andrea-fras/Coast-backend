"""Put a restored backup in place on this server's disk: a move to a new region, or after losing one.

    python scripts/restore_backup.py --to /data/incoming --stamp <stamp>   # fetch and check it
    python scripts/adopt_restore.py /data/incoming                          # then restart the service

Copies the databases and files into the paths this server uses. It refuses to overwrite a server
that already has accounts (that would be a live production), unless --force is given.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from pathlib import Path

OCR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OCR))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("restored", help="a folder restore_backup.py filled and checked")
    ap.add_argument("--force", action="store_true", help="overwrite a server that already has accounts")
    args = ap.parse_args()
    src = Path(args.restored).resolve()
    import backups
    import database
    import oma_provider
    root = backups.data_root()

    live = Path(database.DB_PATH)
    if live.exists() and not args.force:
        with sqlite3.connect(f"file:{live}?mode=ro", uri=True) as c:
            try:
                users = c.execute("select count(*) from users").fetchone()[0]
            except sqlite3.DatabaseError:
                users = 0
        if users:
            sys.exit(f"Refusing: {live} already has {users} accounts. This looks like a live server; use --force only if you mean it.")

    for name, target in (("coast.db", live), ("oma_data/oma.db", Path(oma_provider.OMA_DB_PATH))):
        for side in ("-wal", "-shm"):
            Path(str(target) + side).unlink(missing_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / name, target)
        print(f"database {name} -> {target}")

    dirs = {"folder_uploads": Path(os.getenv("FOLDER_UPLOADS_DIR", root / "folder_uploads")),
            "oma_data/images": Path(oma_provider.OMA_IMAGE_DIR),
            "generated": Path(os.getenv("GENERATED_DIR", root / "generated")),
            "media": Path(os.getenv("MEDIA_DIR", root / "media"))}
    for name, target in dirs.items():
        if (src / name).is_dir():
            shutil.copytree(src / name, target, dirs_exist_ok=True, copy_function=shutil.copy2)
            print(f"files {name}/ -> {target}")
    print("Done. Restart the service now so it opens the restored databases.")


if __name__ == "__main__":
    main()
