"""What each student can do in a month, and how much of it they've used.

Beta students make up to 5 lessons or workshops (together), upload 20 files and send Pedro 600
messages a month. Founding students (the one-time Founding Student pass) get double. Admins have
no limits. Months are calendar months in UTC: everything resets on the 1st.

Use is counted in usage_events, a log that only grows, so deleting a lesson, a file or a chat
doesn't hand back what it used. The one exception is a lesson deleted before its roadmap was
made: it cost nothing, so it gives its slot back. Premade lessons and workshops never count.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import func

from database import CourseOutline, SourceUpload, UsageEvent, User

LIMITS = {"lessons": 5, "uploads": 20, "messages": 600}
FOUNDER_MULTIPLIER = 2
FOUNDER_PRICE_EUR = 14.99
PLANS = ("beta", "founder")
# The website reads this header on a 429 to tell a used-up allowance from "too many requests".
LIMIT_HEADER = "X-Coast-Limit"

_SPENT = {
    "lessons": "You've made {limit} lessons and workshops this month.",
    "uploads": "You've uploaded {limit} files this month.",
    "messages": "You've sent Pedro {limit} messages this month.",
}
_NEXT = {
    "lessons": "You can make more from {reset}.",
    "uploads": "You can upload more from {reset}.",
    "messages": "You can message him again from {reset}.",
}


def month_start(now: datetime | None = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    return now.astimezone(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def next_reset(now: datetime | None = None) -> datetime:
    start = month_start(now)
    return start.replace(year=start.year + 1, month=1) if start.month == 12 else start.replace(month=start.month + 1)


def _reset_text(now: datetime | None = None) -> str:
    reset = next_reset(now)
    return f"{reset.day} {reset.strftime('%B')}"


def unlimited(user: User) -> bool:
    from security import is_admin
    return is_admin(user)


def plan_of(user: User) -> str:
    return "founder" if (getattr(user, "plan", None) or "beta") == "founder" else "beta"


def has_founder_badge(user: User) -> bool:
    """Founding students wear the founder badge, and so does Coast's own admin account."""
    return plan_of(user) == "founder" or unlimited(user)


def limits_for(user: User) -> dict | None:
    """The month's allowances, or None for no limits."""
    if unlimited(user):
        return None
    times = FOUNDER_MULTIPLIER if plan_of(user) == "founder" else 1
    return {kind: n * times for kind, n in LIMITS.items()}


def used(db, user_id: int, now: datetime | None = None) -> dict:
    """This month's lessons, uploads and messages so far."""
    rows = (db.query(UsageEvent.kind, func.count(UsageEvent.id))
            .filter(UsageEvent.user_id == user_id, UsageEvent.created_at >= month_start(now).replace(tzinfo=None))
            .group_by(UsageEvent.kind).all())
    counts = dict.fromkeys(LIMITS, 0)
    counts.update({kind: n for kind, n in rows if kind in counts})
    return counts


def refuse(kind: str, limit: int, room: int = 0, now: datetime | None = None):
    if room > 0:  # a batch of uploads larger than what's left
        message = f"You can upload {room} more {'file' if room == 1 else 'files'} this month. Choose fewer files."
    else:
        message = f"{_SPENT[kind].format(limit=limit)} {_NEXT[kind].format(reset=_reset_text(now))}"
    raise HTTPException(429, message, headers={LIMIT_HEADER: kind})


def check(db, user: User, kind: str, incoming: int = 1, pending: int = 0) -> None:
    """Refuse when `incoming` more (on top of `pending` already on their way) would pass the limit."""
    limits = limits_for(user)
    if limits is None or incoming <= 0:
        return
    have = used(db, user.id)[kind] + pending
    if have + incoming > limits[kind]:
        refuse(kind, limits[kind], room=max(0, limits[kind] - have) if incoming > 1 else 0)


def record(db, user_id: int, kind: str, ref: str = "") -> None:
    """Count one use. The caller commits, so it lands together with what it counts."""
    db.add(UsageEvent(user_id=user_id, kind=kind, ref=str(ref or "")[:200],
                      created_at=datetime.now(timezone.utc)))


def uploads_on_their_way(db, user_id: int, exclude: set[str]) -> int:
    """Files registered but not yet finished, in any lesson: they'll count once they arrive."""
    import time
    return (db.query(SourceUpload)
            .filter(SourceUpload.user_id == user_id, SourceUpload.status.in_(("queued", "processing")),
                    SourceUpload.expires_at > time.time(), SourceUpload.upload_id.notin_(exclude or {""}))
            .count())


def rename_lesson(db, user_id: int, old: str, new: str) -> None:
    db.query(UsageEvent).filter_by(user_id=user_id, kind="lessons", ref=old).update({"ref": new})


def give_back_unused_lesson(db, user_id: int, name: str) -> None:
    """A lesson deleted before its roadmap was made cost nothing: this month, it frees its slot."""
    if db.query(CourseOutline).filter_by(user_id=user_id, folder_name=name).first():
        return
    event = (db.query(UsageEvent)
             .filter(UsageEvent.user_id == user_id, UsageEvent.kind == "lessons", UsageEvent.ref == name,
                     UsageEvent.created_at >= month_start().replace(tzinfo=None))
             .order_by(UsageEvent.id.desc()).first())
    if event:
        db.delete(event)


def summary(db, user: User) -> dict:
    """Everything the "Your plan" panel shows."""
    limits = limits_for(user)
    counts = used(db, user.id)
    plan = "unlimited" if limits is None else plan_of(user)
    return {
        "plan": plan,
        "founder": has_founder_badge(user),
        "period": {"start": month_start().isoformat(), "resets_at": next_reset().isoformat()},
        "usage": {kind: {"used": counts[kind], "limit": None if limits is None else limits[kind]} for kind in LIMITS},
        "founder_offer": {
            "price_eur": FOUNDER_PRICE_EUR,
            "limits": {kind: n * FOUNDER_MULTIPLIER for kind, n in LIMITS.items()},
            "interested": bool(getattr(user, "founder_interest_at", None)),
        },
    }


def legacy_usage(db, user: User) -> dict:
    """The older message counter fields some screens still read, now monthly."""
    limits = limits_for(user)
    counts = used(db, user.id)
    limit = 999999 if limits is None else limits["messages"]
    sent = 0 if limits is None else counts["messages"]
    return {"chat_messages_used": sent, "chat_messages_limit": limit,
            "chat_messages_remaining": max(0, limit - sent)}
