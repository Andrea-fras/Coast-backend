"""Create, list and revoke single-use beta invite codes.

    python3 scripts/manage_beta_codes.py create 5 --note "ML society"
    python3 scripts/manage_beta_codes.py list [--unused]
    python3 scripts/manage_beta_codes.py revoke COAST-7KQ4-M9XP

Works on whichever database DATABASE_PATH points at (coast.db locally,
/data/coast.db on the server). The admin Control Center does the same in the app.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import beta_codes  # noqa: E402
from database import SessionLocal, init_db  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="make new codes")
    create.add_argument("count", nargs="?", type=int, default=1)
    create.add_argument("--note", default="", help="who the codes are for")
    listing = sub.add_parser("list", help="show codes and who used them")
    listing.add_argument("--unused", action="store_true")
    revoke = sub.add_parser("revoke", help="disable an unused code")
    revoke.add_argument("code")
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    try:
        if args.command == "create":
            count = max(1, min(500, args.count))
            for row in beta_codes.create(db, count=count, note=args.note, created_by="cli"):
                print(beta_codes.display(row.code))
        elif args.command == "list":
            rows = beta_codes.list_all(db)
            if args.unused:
                rows = [r for r in rows if beta_codes.status(r) == "unused"]
            for row in rows:
                info = beta_codes.as_dict(row)
                used = f"{info['used_email']} on {info['used_at'][:10]}" if info["used_at"] else ""
                print(f"{info['code']:<17} {info['status']:<8} {info['note'][:30]:<30} {used}")
            print(f"{len(rows)} code(s)")
        elif args.command == "revoke":
            row = beta_codes.revoke(db, args.code)
            print(f"Revoked {beta_codes.display(row.code)}")
    except beta_codes.BetaCodeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
