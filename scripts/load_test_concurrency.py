#!/usr/bin/env python3
"""Ramp concurrent virtual users against production to find breaking point.

Usage:
  python3 scripts/load_test_concurrency.py
  python3 scripts/load_test_concurrency.py --base https://coast-backend-dlg6.onrender.com
  python3 scripts/load_test_concurrency.py --max-users 30 --skip-chat

Each virtual user: register → load map + premade lesson → optional Pedro chat.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import requests

DEFAULT_BASE = "http://localhost:8000"
RAMP = [2, 4, 6, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40]
FAIL_RATE_STOP = 0.20
P95_STOP_SEC = 90.0
CHAT_TIMEOUT = 120


@dataclass
class StepResult:
    ok: bool
    latency: float
    status: int = 0
    error: str = ""


@dataclass
class UserSession:
    tag: str
    token: str = ""
    register: StepResult | None = None
    map_load: StepResult | None = None
    lesson: StepResult | None = None
    chat: StepResult | None = None
    errors: list[str] = field(default_factory=list)


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _timed(fn) -> StepResult:
    t0 = time.perf_counter()
    try:
        ok, status, err = fn()
        return StepResult(ok=ok, latency=time.perf_counter() - t0, status=status, error=err)
    except requests.Timeout:
        return StepResult(ok=False, latency=time.perf_counter() - t0, error="timeout")
    except Exception as e:
        return StepResult(ok=False, latency=time.perf_counter() - t0, error=str(e)[:120])


def register_user(base: str, tag: str) -> UserSession:
    sess = UserSession(tag=tag)
    email = f"load_{tag}_{uuid.uuid4().hex[:8]}@loadtest.local"

    def do_reg():
        r = requests.post(
            f"{base}/api/auth/register",
            json={"email": email, "password": "loadtest1234", "name": f"Load {tag}", "course": "Test"},
            timeout=60,
        )
        if r.status_code != 200:
            return False, r.status_code, r.text[:200]
        data = r.json()
        sess.token = data.get("token", "")
        return bool(sess.token), r.status_code, ""

    sess.register = _timed(do_reg)
    if not sess.register.ok:
        sess.errors.append(f"register: {sess.register.error or sess.register.status}")
    return sess


def run_session(base: str, sess: UserSession, *, do_chat: bool) -> UserSession:
    if not sess.token:
        return sess

    def map_load():
        r = requests.get(f"{base}/api/map", headers=_headers(sess.token), timeout=60)
        return r.status_code == 200, r.status_code, r.text[:120]

    def lesson_load():
        r = requests.get(
            f"{base}/api/folders/Prismatic%20System/lesson",
            headers=_headers(sess.token),
            timeout=60,
        )
        return r.status_code == 200, r.status_code, r.text[:120]

    def chat_send():
        r = requests.post(
            f"{base}/api/chat/send",
            headers=_headers(sess.token),
            json={"message": "Reply with exactly one word: hello", "context_type": "global"},
            timeout=CHAT_TIMEOUT,
        )
        if r.status_code != 200:
            return False, r.status_code, r.text[:200]
        data = r.json()
        reply = (data.get("reply") or "").strip()
        return bool(reply), r.status_code, "" if reply else "empty reply"

    sess.map_load = _timed(map_load)
    sess.lesson = _timed(lesson_load)
    if do_chat:
        sess.chat = _timed(chat_send)

    for step in (sess.map_load, sess.lesson, sess.chat):
        if step and not step.ok:
            sess.errors.append(step.error or str(step.status))
    return sess


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    i = min(len(s) - 1, int(len(s) * p))
    return s[i]


def summarize_phase(sessions: list[UserSession], do_chat: bool) -> dict:
    n = len(sessions)
    reg_ok = sum(1 for s in sessions if s.register and s.register.ok)
    map_ok = sum(1 for s in sessions if s.map_load and s.map_load.ok)
    lesson_ok = sum(1 for s in sessions if s.lesson and s.lesson.ok)
    chat_ok = sum(1 for s in sessions if s.chat and s.chat.ok) if do_chat else n

    map_lat = [s.map_load.latency for s in sessions if s.map_load and s.map_load.ok]
    lesson_lat = [s.lesson.latency for s in sessions if s.lesson and s.lesson.ok]
    chat_lat = [s.chat.latency for s in sessions if s.chat and s.chat.ok]

    if do_chat:
        full_ok = sum(1 for s in sessions if s.register and s.register.ok and s.map_load and s.map_load.ok
                      and s.lesson and s.lesson.ok and s.chat and s.chat.ok)
    else:
        full_ok = sum(1 for s in sessions if s.register and s.register.ok and s.map_load and s.map_load.ok
                      and s.lesson and s.lesson.ok)

    fail_rate = 1 - (full_ok / n if n else 0)
    return {
        "n": n,
        "register_ok": reg_ok,
        "map_ok": map_ok,
        "lesson_ok": lesson_ok,
        "chat_ok": chat_ok,
        "full_ok": full_ok,
        "fail_rate": fail_rate,
        "map_p50": statistics.median(map_lat) if map_lat else 0,
        "map_p95": _pct(map_lat, 0.95),
        "lesson_p50": statistics.median(lesson_lat) if lesson_lat else 0,
        "lesson_p95": _pct(lesson_lat, 0.95),
        "chat_p50": statistics.median(chat_lat) if chat_lat else 0,
        "chat_p95": _pct(chat_lat, 0.95),
        "sample_errors": [e for s in sessions for e in s.errors[:2]][:5],
    }


def print_row(concurrency: int, stats: dict, do_chat: bool):
    chat_col = ""
    if do_chat:
        chat_col = f" chat={stats['chat_ok']}/{stats['n']} p95={stats['chat_p95']:.1f}s"
    print(
        f"  {concurrency:3d} users | full OK {stats['full_ok']}/{stats['n']} "
        f"({100*(1-stats['fail_rate']):.0f}%) | "
        f"map p95={stats['map_p95']:.1f}s lesson p95={stats['lesson_p95']:.1f}s"
        f"{chat_col}"
    )
    if stats["sample_errors"]:
        print(f"         errors: {stats['sample_errors'][0][:100]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--max-users", type=int, default=40)
    ap.add_argument("--skip-chat", action="store_true", help="Read-only load (map+lesson, no LLM)")
    ap.add_argument("--levels", default="", help="Comma-separated concurrency levels")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    do_chat = not args.skip_chat
    levels = [int(x) for x in args.levels.split(",") if x.strip()] if args.levels else RAMP
    levels = [l for l in levels if l <= args.max_users]

    print(f"Target: {base}")
    print(f"Mode: {'map+lesson+chat' if do_chat else 'map+lesson only'}")
    print(f"Ramp: {levels}")
    print()

    # Warm up (Render cold start)
    try:
        t0 = time.perf_counter()
        r = requests.get(f"{base}/docs", timeout=90)
        print(f"Warmup: /docs {r.status_code} in {time.perf_counter()-t0:.1f}s")
    except Exception as e:
        print(f"Warmup failed: {e}")
        sys.exit(1)

    last_good = 0
    first_bad = None

    for n in levels:
        print(f"== {n} concurrent users ==")
        tag_base = f"n{n}_{int(time.time())}"

        # Register all users in parallel
        users: list[UserSession] = []
        with ThreadPoolExecutor(max_workers=min(n, 20)) as ex:
            futs = [ex.submit(register_user, base, f"{tag_base}_{i}") for i in range(n)]
            for f in as_completed(futs):
                users.append(f.result())

        reg_fail = sum(1 for u in users if not u.register or not u.register.ok)
        if reg_fail:
            print(f"  WARNING: {reg_fail}/{n} registrations failed")

        # Run sessions in parallel
        t_phase = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as ex:
            futs = [ex.submit(run_session, base, u, do_chat=do_chat) for u in users]
            sessions = [f.result() for f in as_completed(futs)]
        phase_sec = time.perf_counter() - t_phase

        stats = summarize_phase(sessions, do_chat)
        print_row(n, stats, do_chat)
        print(f"         phase wall time: {phase_sec:.1f}s")

        if stats["fail_rate"] <= FAIL_RATE_STOP:
            last_good = n
        elif first_bad is None:
            first_bad = n

        chat_p95 = stats["chat_p95"] if do_chat else 0
        if stats["fail_rate"] > FAIL_RATE_STOP or (do_chat and chat_p95 > P95_STOP_SEC):
            print()
            print("Stopping ramp — failure threshold exceeded.")
            break

        print()
        time.sleep(3)  # brief pause between levels

    print("=" * 60)
    print(f"Estimated safe concurrent users: ~{last_good}")
    if first_bad:
        print(f"First level with significant failures: {first_bad}")
    else:
        print(f"All tested levels up to {levels[-1] if levels else 0} passed.")
    print()
    print("Notes:")
    print("- Render runs 1 gunicorn worker; Pedro chat blocks a thread until Gemini responds.")
    print("- Beyond ~15-20 simultaneous chats, requests queue rather than fail (higher latency).")
    print("- SQLite WAL handles many readers; writes (chat saves) serialize briefly.")


if __name__ == "__main__":
    main()
