#!/usr/bin/env python3
"""Set a user's password in the local coast.db (dev only)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auth import hash_password
from auth_email import normalize_email
from database import SessionLocal, User


def main() -> None:
    if len(sys.argv) not in (3, 4):
        print("Usage: python3 scripts/dev_set_password.py <email> <password> [display-name]")
        sys.exit(1)

    email = normalize_email(sys.argv[1])
    password = sys.argv[2]
    display_name = sys.argv[3].strip() if len(sys.argv) == 4 else email.split("@", 1)[0].title()
    if len(password) < 6:
        print("Password must be at least 6 characters.")
        sys.exit(1)

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            user = User(
                email=email,
                name=display_name,
                password_hash=hash_password(password),
                email_verified=True,
            )
            db.add(user)
            db.commit()
            print(f"Created local account for {email} (name: {display_name})")
            return
        if user.google_id:
            print(f"{email} is a Google account — password login is disabled.")
            sys.exit(1)

        user.password_hash = hash_password(password)
        if display_name:
            user.name = display_name
        db.commit()
        print(f"Updated local password for {email}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
