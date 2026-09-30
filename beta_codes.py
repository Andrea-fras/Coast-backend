"""Single-use beta invite codes.

Creating an account (email or Google) requires an unused code while
COAST_REQUIRE_BETA_CODE is on (the default). The code is consumed in the same
transaction that creates the user, so one code can never open two accounts.

Codes look like COAST-7KQ4-M9XP. Students may type them in any case, with or
without dashes or the COAST prefix; they are stored in compact form.
"""
from __future__ import annotations

import os
import re
import secrets
from datetime import datetime, timezone

from sqlalchemy import update

from database import BetaCode

PREFIX = "COAST"
_ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"  # no 0/O, 1/I/L
_BODY_LEN = 8


class BetaCodeError(ValueError):
    """A code that can't be used; the message is safe to show the student."""


def required() -> bool:
    return os.getenv("COAST_REQUIRE_BETA_CODE", "1").strip().lower() not in ("0", "false", "no", "off")


def exempt(email: str) -> bool:
    """Synthetic load-test / smoke-test accounts, only on a local server."""
    return not os.getenv("RENDER") and email.endswith(("@loadtest.local", "@test.local"))


def normalize(raw: str) -> str:
    compact = re.sub(r"[^A-Z0-9]", "", (raw or "").upper())
    if len(compact) == _BODY_LEN and not compact.startswith(PREFIX):
        compact = PREFIX + compact
    return compact


def display(code: str) -> str:
    if code.startswith(PREFIX) and len(code) == len(PREFIX) + _BODY_LEN:
        body = code[len(PREFIX):]
        return f"{PREFIX}-{body[:4]}-{body[4:]}"
    return code


def _new_code() -> str:
    return PREFIX + "".join(secrets.choice(_ALPHABET) for _ in range(_BODY_LEN))


def create(db, count: int = 1, note: str = "", created_by: str = "") -> list[BetaCode]:
    rows = []
    for _ in range(count):
        code = _new_code()
        while db.get(BetaCode, code) is not None:
            code = _new_code()
        row = BetaCode(code=code, note=(note or "").strip()[:255], created_by=created_by)
        db.add(row)
        rows.append(row)
    db.commit()
    return rows


def check(db, raw: str) -> str:
    """Return the compact code if it can be used right now, else raise BetaCodeError."""
    code = normalize(raw)
    if not code:
        raise BetaCodeError("Enter the beta code you were given.")
    row = db.get(BetaCode, code)
    if row is None:
        raise BetaCodeError("That beta code isn't valid. Check it and try again.")
    if row.revoked:
        raise BetaCodeError("That beta code is no longer active.")
    if row.used_at is not None:
        raise BetaCodeError("That beta code has already been used.")
    return code


def consume(db, code: str, user_id: int, email: str) -> None:
    """Mark the code used by this user inside the caller's transaction.

    The conditional UPDATE is the race guard: if another signup used the code
    first, no row matches and the caller must roll back.
    """
    result = db.execute(
        update(BetaCode)
        .where(BetaCode.code == code, BetaCode.used_at.is_(None), BetaCode.revoked.is_(False))
        .values(used_by_user_id=user_id, used_email=email, used_at=datetime.now(timezone.utc))
    )
    if result.rowcount != 1:
        raise BetaCodeError("That beta code has already been used.")


def revoke(db, raw: str) -> BetaCode:
    row = db.get(BetaCode, normalize(raw))
    if row is None:
        raise BetaCodeError("No such code.")
    if row.used_at is not None:
        raise BetaCodeError("That code has already been used, so it can't be revoked.")
    row.revoked = True
    db.commit()
    return row


def status(row: BetaCode) -> str:
    if row.used_at is not None:
        return "used"
    return "revoked" if row.revoked else "unused"


def as_dict(row: BetaCode) -> dict:
    return {
        "code": display(row.code),
        "note": row.note or "",
        "status": status(row),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "created_by": row.created_by or "",
        "used_email": row.used_email,
        "used_at": row.used_at.isoformat() if row.used_at else None,
    }


def list_all(db) -> list[BetaCode]:
    return db.query(BetaCode).order_by(BetaCode.created_at.desc()).all()
