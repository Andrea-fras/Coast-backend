"""The numbers that say whether Coast is growing, for the Control Center (and investors).

Everything counts students only: test bots (@loadtest.local) and the admin accounts are left out,
so the founders' own use never flatters a number. Days and weeks are UTC; weeks start on Monday.

A student is *active* on a day when they did something on purpose: sent Pedro a message (not the
automatic section opener), finished a section, made a lesson or uploaded a file, or spent time in
the app (the time log only counts visible, recently used tabs).
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import func

_OPENERS = ("I'm ready to learn about%", "Im ready to learn about%",
            "I'd like to reach 100% mastery%", "Id like to reach 100% mastery%")
_CACHE: dict = {}
_LOCK = threading.Lock()
CACHE_SECONDS = 60
MAX_DAY_MINUTES = 6 * 60


def _day(value) -> str | None:
    if value is None:
        return None
    return value.strftime("%Y-%m-%d") if hasattr(value, "strftime") else str(value)[:10]


def _monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _pct(part, whole) -> float:
    return round(part / whole * 100, 1) if whole else 0.0


def _growth(now_n, before_n) -> float | None:
    return round((now_n - before_n) / before_n * 100, 1) if before_n else None


def students(db) -> dict[int, dict]:
    """Every real student: id → {signup day, onboarded, plan, wants the pass}."""
    from database import User
    from security import ADMIN_EMAILS
    out = {}
    for u in db.query(User.id, User.email, User.created_at, User.onboarding_completed, User.plan,
                      User.founder_interest_at):
        email = (u.email or "").lower()
        if email.endswith("@loadtest.local") or email in ADMIN_EMAILS:
            continue
        out[u.id] = {"signup": _day(u.created_at), "onboarded": bool(u.onboarding_completed),
                     "founder": u.plan == "founder", "wants_pass": u.founder_interest_at is not None}
    return out


def _typed_messages(db, ChatMessage):
    q = db.query(ChatMessage).filter(ChatMessage.role == "user", ChatMessage.context_type != "onboarding")
    for pattern in _OPENERS:
        q = q.filter(~ChatMessage.content.like(pattern))
    return q


def compute(db, now: datetime | None = None) -> dict:
    from database import ActivityEvent, BetaCode, ChatMessage, CourseOutline, SectionRewardClaim, UsageEvent
    import plans

    now = now or datetime.now(timezone.utc)
    today = now.date()
    t = today.isoformat()
    people = students(db)
    ids = set(people)

    # Every day each student was active, from everything they did on purpose.
    active: dict[int, set[str]] = defaultdict(set)
    minutes_by_day: dict[tuple[int, str], float] = defaultdict(float)
    for uid, d, ms in (db.query(ActivityEvent.user_id, ActivityEvent.event_date, func.sum(ActivityEvent.duration_ms))
                       .filter(ActivityEvent.duration_ms > 0).group_by(ActivityEvent.user_id, ActivityEvent.event_date)):
        if uid in ids:
            active[uid].add(d)
            # Before 9 October 2026 an open tab counted even when nobody was there: no day counts past 6 hours.
            minutes_by_day[(uid, d)] = min(MAX_DAY_MINUTES, minutes_by_day[(uid, d)] + (ms or 0) / 60000)
    messages_by_day: dict[str, int] = defaultdict(int)
    message_day = func.substr(ChatMessage.created_at, 1, 10)
    lesson_talkers = set()
    for uid, d, ctx, n in (_typed_messages(db, ChatMessage)
                           .with_entities(ChatMessage.user_id, message_day, ChatMessage.context_type, func.count())
                           .group_by(ChatMessage.user_id, message_day, ChatMessage.context_type)):
        if uid in ids and d:
            active[uid].add(d)
            messages_by_day[d] += n
            if ctx in ("lesson", "test_out", "folder"):
                lesson_talkers.add(uid)
    sections_by_day: dict[str, int] = defaultdict(int)
    finished_section = set()
    claim_day = func.substr(SectionRewardClaim.created_at, 1, 10)
    for uid, d, n in (db.query(SectionRewardClaim.user_id, claim_day, func.count())
                      .group_by(SectionRewardClaim.user_id, claim_day)):
        if uid in ids and d:
            active[uid].add(d)
            sections_by_day[d] += n
            finished_section.add(uid)
    made: dict[str, dict[str, int]] = {"lessons": defaultdict(int), "uploads": defaultdict(int)}
    event_day = func.substr(UsageEvent.created_at, 1, 10)
    for uid, kind, d, n in (db.query(UsageEvent.user_id, UsageEvent.kind, event_day, func.count())
                            .filter(UsageEvent.kind.in_(("lessons", "uploads")))
                            .group_by(UsageEvent.user_id, UsageEvent.kind, event_day)):
        if uid in ids and d:
            active[uid].add(d)
            made[kind][d] += n

    def active_between(start: date, end: date) -> set[int]:
        a, b = start.isoformat(), end.isoformat()
        return {uid for uid, days in active.items() if any(a <= d <= b for d in days)}

    dau = active_between(today, today)
    wau = active_between(today - timedelta(days=6), today)
    mau = active_between(today - timedelta(days=29), today)
    avg_dau_7d = sum(len(active_between(today - timedelta(days=i), today - timedelta(days=i))) for i in range(7)) / 7

    def signups_between(start: date, end: date) -> int:
        a, b = start.isoformat(), end.isoformat()
        return sum(1 for p in people.values() if p["signup"] and a <= p["signup"] <= b)

    new_7d = signups_between(today - timedelta(days=6), today)
    new_prev_7d = signups_between(today - timedelta(days=13), today - timedelta(days=7))
    new_30d = signups_between(today - timedelta(days=29), today)
    new_prev_30d = signups_between(today - timedelta(days=59), today - timedelta(days=30))

    # Retention: of the students who signed up at least N days ago, how many were active on day N
    # (classic Dn), and how many came back at any point in days 1-7 (first-week retention).
    def returned(offset: int) -> dict:
        eligible = back = 0
        for uid, p in people.items():
            if not p["signup"]:
                continue
            target = date.fromisoformat(p["signup"]) + timedelta(days=offset)
            if target > today:
                continue
            eligible += 1
            back += target.isoformat() in active.get(uid, ())
        return {"pct": _pct(back, eligible), "of": eligible}

    eligible = back = 0
    for uid, p in people.items():
        if not p["signup"]:
            continue
        start = date.fromisoformat(p["signup"])
        if start + timedelta(days=7) > today:
            continue
        eligible += 1
        window = {(start + timedelta(days=i)).isoformat() for i in range(1, 8)}
        back += bool(window & active.get(uid, set()))
    first_week = {"pct": _pct(back, eligible), "of": eligible}

    # Weekly cohorts: of the students who signed up in a week, the share active in each later week.
    this_monday = _monday(today)
    cohorts = []
    for w in range(7, -1, -1):
        start = this_monday - timedelta(weeks=w)
        members = [uid for uid, p in people.items()
                   if p["signup"] and start <= date.fromisoformat(p["signup"]) < start + timedelta(days=7)]
        weeks = []
        for k in range(5):
            ws = start + timedelta(weeks=k)
            if ws > today:
                weeks.append(None)
                continue
            seen = active_between(ws, min(ws + timedelta(days=6), today))
            weeks.append(_pct(sum(1 for uid in members if uid in seen), len(members)) if members else None)
        cohorts.append({"week": start.isoformat(), "size": len(members), "active_pct": weeks})

    # Week by week, last 12 weeks.
    weekly = []
    for w in range(11, -1, -1):
        start = this_monday - timedelta(weeks=w)
        end = min(start + timedelta(days=6), today)
        a, b = start.isoformat(), end.isoformat()
        weekly.append({
            "week": a,
            "signups": signups_between(start, end),
            "active": len(active_between(start, end)),
            "messages": sum(n for d, n in messages_by_day.items() if a <= d <= b),
            "sections": sum(n for d, n in sections_by_day.items() if a <= d <= b),
        })
    daily = []
    for i in range(29, -1, -1):
        d = (today - timedelta(days=i)).isoformat()
        daily.append({"date": d, "signups": sum(1 for p in people.values() if p["signup"] == d),
                      "active": sum(1 for days in active.values() if d in days)})

    # The path from signing up to coming back.
    started = {uid for (uid,) in db.query(CourseOutline.user_id).distinct() if uid in ids}
    started |= {uid for (uid,) in db.query(UsageEvent.user_id).filter(UsageEvent.kind == "lessons").distinct() if uid in ids}
    came_back = {uid for uid, days in active.items() if len(days) >= 2}
    n = len(people)
    funnel = [
        {"step": "Signed up", "count": n},
        {"step": "Finished the welcome", "count": sum(1 for p in people.values() if p["onboarded"])},
        {"step": "Started a lesson or workshop", "count": len(started)},
        {"step": "Studied with Pedro", "count": len(lesson_talkers)},
        {"step": "Finished a section", "count": len(finished_section)},
        {"step": "Came back another day", "count": len(came_back)},
    ]
    for step in funnel:
        step["pct"] = _pct(step["count"], n)

    # Engagement over the last 7 days.
    week_start = (today - timedelta(days=6)).isoformat()
    student_days = [(uid, d) for uid, days in active.items() for d in days if week_start <= d <= t]
    timed = [m for (uid, d), m in minutes_by_day.items() if week_start <= d <= t]
    msgs_7d = sum(c for d, c in messages_by_day.items() if d >= week_start)
    engagement = {
        "active_days_per_student": round(len(student_days) / len(wau), 1) if wau else 0,
        "minutes_per_study_day": round(sum(timed) / len(timed), 1) if timed else 0,
        "messages_per_active_student": round(msgs_7d / len(wau), 1) if wau else 0,
        "messages_7d": msgs_7d,
        "sections_7d": sum(c for d, c in sections_by_day.items() if d >= week_start),
        "lessons_7d": sum(c for d, c in made["lessons"].items() if d >= week_start),
        "uploads_7d": sum(c for d, c in made["uploads"].items() if d >= week_start),
        "minutes_7d": round(sum(timed)),
    }

    # Willingness to pay and what each student costs.
    month = plans.month_start(now).replace(tzinfo=None)
    use = defaultdict(lambda: dict.fromkeys(plans.LIMITS, 0))
    for uid, kind, c in (db.query(UsageEvent.user_id, UsageEvent.kind, func.count())
                         .filter(UsageEvent.created_at >= month).group_by(UsageEvent.user_id, UsageEvent.kind)):
        if uid in ids and kind in plans.LIMITS:
            use[uid][kind] = c
    at_cap = near_cap = 0
    for uid, counts in use.items():
        times = plans.FOUNDER_MULTIPLIER if people[uid]["founder"] else 1
        shares = [counts[k] / (plans.LIMITS[k] * times) for k in plans.LIMITS]
        at_cap += max(shares) >= 1
        near_cap += max(shares) >= 0.8
    import ai_usage
    ai = ai_usage.summary(30)["total"]
    founders = sum(1 for p in people.values() if p["founder"])
    wants = sum(1 for p in people.values() if p["wants_pass"] and not p["founder"])
    used_codes = db.query(BetaCode).filter(BetaCode.used_at.isnot(None)).count()
    open_codes = db.query(BetaCode).filter(BetaCode.used_at.is_(None), BetaCode.revoked.is_(False)).count()
    monetization = {
        "founders": founders,
        "wants_pass": wants,
        "pass_intent_pct": _pct(founders + wants, n),
        "founder_revenue_eur": round(founders * plans.FOUNDER_PRICE_EUR, 2),
        "at_cap_this_month": at_cap,
        "near_cap_this_month": near_cap,
        "ai_cost_30d_usd": round(ai["cost_usd"], 2),
        "ai_cost_per_active_student_usd": round(ai["cost_usd"] / len(mau), 2) if mau else 0,
        "invites_used": used_codes,
        "invites_open": open_codes,
        "invite_use_pct": _pct(used_codes, used_codes + open_codes),
    }

    totals = {
        "students": n,
        "messages": sum(messages_by_day.values()),
        "sections": sum(sections_by_day.values()),
        "lessons": sum(made["lessons"].values()),  # made, counting ones since deleted (premade ones never count)
        "files": sum(made["uploads"].values()),
        "study_hours": round(sum(minutes_by_day.values()) / 60, 1),
    }
    return {
        "generated_at": now.isoformat(),
        "headline": {
            "students": n,
            "new_today": sum(1 for p in people.values() if p["signup"] == t),
            "new_7d": new_7d, "new_prev_7d": new_prev_7d, "growth_wow_pct": _growth(new_7d, new_prev_7d),
            "new_30d": new_30d, "new_prev_30d": new_prev_30d, "growth_mom_pct": _growth(new_30d, new_prev_30d),
            "dau": len(dau), "wau": len(wau), "mau": len(mau),
            "stickiness_pct": _pct(avg_dau_7d, len(mau)),
            "activation_pct": _pct(len(finished_section), n),
            "messages_today": messages_by_day.get(t, 0),
        },
        "retention": {"d1": returned(1), "d7": returned(7), "d30": returned(30), "first_week": first_week},
        "cohorts": cohorts,
        "weekly": weekly,
        "daily": daily,
        "funnel": funnel,
        "engagement": engagement,
        "monetization": monetization,
        "totals": totals,
    }


def cached(db_factory) -> dict:
    """The Control Center refreshes often; the numbers are worked out at most once a minute."""
    with _LOCK:
        hit = _CACHE.get("growth")
        if hit and time.monotonic() - hit[0] < CACHE_SECONDS:
            return hit[1]
    with db_factory() as db:
        result = compute(db)
    with _LOCK:
        _CACHE["growth"] = (time.monotonic(), result)
    return result


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()
