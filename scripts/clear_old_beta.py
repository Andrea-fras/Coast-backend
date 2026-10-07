"""Start the new beta on a clean disk, keeping only the named logins.

    python3 scripts/clear_old_beta.py --keep andreaf.fraschetti@gmail.com             # dry run
    python3 scripts/clear_old_beta.py --keep andreaf.fraschetti@gmail.com --confirm   # do it

Kept: the named accounts as logins (email, password, verified status, so admin access survives),
empty of any courses or history, plus beta codes and the past papers Coast ships. Removed: every
other account; every course, source, file, conversation, note, notebook, progress, memory and
analytics row of anyone (the kept accounts included); the premade courses' indexed copy, which the
app rebuilds from the current course files on its next start; all Content and Student OMA data; the
old Chroma index; generated files. --confirm refuses unless R2 holds a complete backup from the last
24 hours, which keeps everything removed here for 30 days. Restart the service afterwards.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

OCR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OCR))

SHARED_TABLES = {"papers", "beta_codes"}  # no owner: past papers (reloaded at startup) and invite codes


def plan(db: sqlite3.Connection, keep: set[int]) -> dict[str, tuple[str, int]]:
    """Per table: the WHERE clause of the rows to remove, and how many there are."""
    ids = ",".join(str(i) for i in sorted(keep))
    out = {}
    for (t,) in db.execute("select name from sqlite_master where type='table' and name not like 'sqlite_%'").fetchall():
        if t in SHARED_TABLES:
            continue
        cols = {r[1] for r in db.execute(f'pragma table_info("{t}")')}
        if t == "users":
            where = f"id not in ({ids})"
        elif t == "email_verifications" and "email" in cols:
            where = f"lower(email) not in (select lower(email) from users where id in ({ids}))"
        else:
            where = "1"  # every other table holds learning data, which goes for everyone
        n = db.execute(f'select count(*) from "{t}" where {where}').fetchone()[0]
        if n:
            out[t] = (where, n)
    return out


def oma_tables(oma: sqlite3.Connection) -> dict[str, int]:
    out = {}
    for (t,) in oma.execute("select name from sqlite_master where type='table' and name not like 'sqlite_%' "
                            "and name not like '%_fts%'").fetchall():
        n = oma.execute(f'select count(*) from "{t}"').fetchone()[0]
        if n:
            out[t] = n
    return out


def complete_offsite_backup():
    """The newest backup in R2 if it finished (files included) less than 24 hours ago."""
    import backups
    client, bucket = backups._r2()
    if not client:
        return None
    manifests = sorted(k for k in backups._listing(client, bucket, "db/") if k.endswith("/manifest.json"))
    if not manifests:
        return None
    m = json.loads(client.get_object(Bucket=bucket, Key=backups.key(manifests[-1]))["Body"].read())
    fresh = datetime.now(timezone.utc) - datetime.fromisoformat(m["created_at"]) < timedelta(hours=24)
    return m if fresh and m.get("offsite") else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", action="append", required=True, help="email of a login to keep (repeatable)")
    ap.add_argument("--confirm", action="store_true", help="remove (default: only report)")
    args = ap.parse_args()
    import backups
    import database
    import oma_provider
    from rag import CHROMA_PATH
    root = backups.data_root()
    dirs = [Path(os.getenv("FOLDER_UPLOADS_DIR", root / "folder_uploads")), Path(oma_provider.OMA_IMAGE_DIR),
            Path(os.getenv("GENERATED_DIR", root / "generated")), Path(os.getenv("MEDIA_DIR", root / "media")),
            Path(CHROMA_PATH)]
    dirs = [d for d in dirs if d.is_dir()]

    db = sqlite3.connect(database.DB_PATH, timeout=60)
    oma = sqlite3.connect(oma_provider.OMA_DB_PATH, timeout=60)
    wanted = [e.strip().lower() for e in args.keep]
    found = dict(db.execute(f"select lower(email), id from users where lower(email) in ({','.join('?' * len(wanted))})",
                            wanted).fetchall())
    missing = [e for e in wanted if e not in found]
    if missing:
        sys.exit(f"Refusing: no account for {missing}; nothing changed.")
    admins = [e for e, i in found.items() if db.execute("select email_verified from users where id=?", (i,)).fetchone()[0]]
    keep = set(found.values())

    rows, omarows = plan(db, keep), oma_tables(oma)
    files = [p for d in dirs for p in d.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    total = db.execute("select count(*) from users").fetchone()[0]

    print(f"Keeping the logins of {', '.join(f'{e} (id {i})' for e, i in found.items())}"
          f" (verified: {', '.join(admins) or 'none'}), beta codes and past papers.")
    print(f"Removing {rows.get('users', (None, 0))[1]} of {total} accounts and all learning data:")
    for t, (_, n) in sorted(rows.items(), key=lambda x: -x[1][1]):
        print(f"  {t:28} {n:7} rows")
    for t, n in omarows.items():
        print(f"  oma.{t:24} {n:7} rows")
    print(f"  files: {len(files)} in {', '.join(d.name for d in dirs)} ({size / 1e9:.2f} GB)")
    if not args.confirm:
        print("Dry run: nothing changed. Run again with --confirm to remove the above.")
        return

    backup = complete_offsite_backup()
    if not backup:
        sys.exit("Refusing: no complete backup in R2 from the last 24 hours (files included).")
    print(f"Backup {backup['stamp']} is complete in R2; it keeps everything removed here for 30 days.")

    with db:
        for t, (where, _) in rows.items():
            db.execute(f'delete from "{t}" where {where}')
    with oma:
        for t in omarows:
            oma.execute(f'delete from "{t}"')
    for (fts,) in oma.execute("select name from sqlite_master where type='table' and name like '%_fts'").fetchall():
        try:
            oma.execute(f"insert into {fts}({fts}) values('rebuild')")  # empty the search postings too
            oma.commit()
        except sqlite3.DatabaseError:
            pass
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
    db.execute("vacuum")
    oma.execute("vacuum")
    print(json.dumps({"removed_rows": {t: n for t, (_, n) in rows.items()}, "removed_oma_rows": omarows,
                      "removed_files": len(files), "freed_gb": round(size / 1e9, 2)}))
    print("Done. Restart the service: it rebuilds the premade courses and starts clean.")


if __name__ == "__main__":
    main()
