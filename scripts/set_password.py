"""Set a new password for an existing email/password account.

Passwords are stored as one-way bcrypt hashes, so they can't be read back;
this replaces the hash instead. The new password is typed at a hidden prompt.

    python3 scripts/set_password.py you@example.com

Uses whichever database DATABASE_PATH points at (coast.db locally,
/data/coast.db on the Render server, so run it in the Render shell to change
your production password).
"""
from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from auth import hash_password  # noqa: E402
from auth_email import normalize_email  # noqa: E402
from database import DB_PATH, SessionLocal, User, init_db  # noqa: E402

MIN_LENGTH = 8


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    email = normalize_email(sys.argv[1])
    init_db()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if user is None:
            print(f"No account for {email} in {DB_PATH}.", file=sys.stderr)
            return 1
        if user.google_id:
            print(f"{email} signs in with Google, so it has no password to change.", file=sys.stderr)
            return 1
        password = getpass.getpass(f"New password for {email}: ")
        if len(password) < MIN_LENGTH:
            print(f"Use at least {MIN_LENGTH} characters.", file=sys.stderr)
            return 1
        if getpass.getpass("Type it again: ") != password:
            print("The two passwords didn't match. Nothing changed.", file=sys.stderr)
            return 1
        user.password_hash = hash_password(password)
        db.commit()
        print(f"Password updated for {email} ({DB_PATH}).")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
