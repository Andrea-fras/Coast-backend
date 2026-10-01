#!/usr/bin/env python3
"""Recover onboarding answers the old code never read.

Until 30 Sep 2026 the traits were extracted before the final onboarding turn was saved, so the
student's last answer (often how they want to practise, or what they want when stuck) was never
remembered, and every later call reused that first result. For each student with an onboarding
conversation this re-reads the whole conversation and adds only traits based on an answer that
no saved trait already covers. It never removes or rewrites a trait. It also takes Pedro's hidden
tags out of the stored onboarding replies and refreshes the legacy learning_preferences.

Dry run by default; --apply backs up oma.db and coast.db first.

  python3 scripts/backfill_onboarding_traits.py
  python3 scripts/backfill_onboarding_traits.py --user-id 9107 --apply
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import oma_provider  # noqa: E402
import onboarding  # noqa: E402
from database import ChatMessage, SessionLocal, User  # noqa: E402


def _covers(quote: str, answer: str) -> bool:
    words = onboarding._words(quote)
    return bool(words) and len(words & onboarding._words(answer)) >= 0.8 * len(words)


def _saved(user_id: int) -> list:
    from coast_content_oma.student.stores import identity_namespace
    store = oma_provider._student_orchestrator().identity
    return [it for it in store.all_traits(identity_namespace(user_id), min_confidence=0.0)
            if "onboarding" in ((it.store_specific or {}).get("evidence_courses") or [])]


def _backup() -> None:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for path in (oma_provider.OMA_DB_PATH, ROOT / "coast.db"):
        dest = Path(f"{path}.bak-onboarding-{stamp}")
        with sqlite3.connect(path) as src, sqlite3.connect(dest) as out:
            src.backup(out)  # consistent even while the server is writing
        print(f"backup: {dest}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", type=int)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if not oma_provider.is_student_enabled():
        print("Student OMA is disabled; nothing to do.")
        return 1
    if args.apply:
        _backup()

    with SessionLocal() as db:
        q = db.query(ChatMessage).filter(ChatMessage.context_type == "onboarding")
        if args.user_id:
            q = q.filter(ChatMessage.user_id == args.user_id)
        conversations: dict[int, dict[str, list]] = {}
        for m in q.order_by(ChatMessage.id).all():
            conversations.setdefault(m.user_id, {}).setdefault(m.conversation_id, []).append(m)

        for uid, convs in conversations.items():
            user = db.get(User, uid)
            label = f"user {uid} ({user.name if user else '?'})"
            rows = max(convs.values(), key=len)  # the conversation they actually had
            messages = [{"role": m.role, "content": m.content} for m in rows]

            # Hidden tags in stored replies.
            dirty = [m for m in rows if m.role == "pedro" and onboarding.strip_onboarding_tags(m.content) != (m.content or "").strip()]
            for m in dirty:
                print(f"{label}: tags in stored reply {m.id}")
                if args.apply:
                    m.content = onboarding.strip_onboarding_tags(m.content)

            answers = onboarding._student_lines_from_messages(messages)
            saved = _saved(uid)
            quotes = [(it.store_specific or {}).get("evidence_quote") or "" for it in saved]
            open_answers = [a for a in answers if not any(_covers(q, a) for q in quotes)]
            if not open_answers:
                print(f"{label}: every answer already remembered ({len(saved)} traits)")
            else:
                for a in open_answers:
                    print(f"{label}: not remembered yet: {a[:120]!r}")
                new = [t for t in onboarding.extract_traits_from_conversation(messages)
                       if any(_covers(t.get("evidence") or "", a) for a in open_answers)]
                for t in new:
                    print(f"{label}:   + {t['trait_type']}: {t['description']}"
                          + ("" if t.get("stated") else "  (inferred)"))
                if not new:
                    print(f"{label}:   nothing durable in those answers")
                if args.apply and new:
                    onboarding.save_traits_to_student_oma(uid, new)

            if args.apply and user is not None:
                traits = [{"trait_type": (it.store_specific or {}).get("trait_type"), "description": it.content}
                          for it in _saved(uid)]
                user.learning_preferences = json.dumps(onboarding.traits_to_preferences(traits))
        if args.apply:
            db.commit()
    print("applied" if args.apply else "dry run: nothing written (use --apply)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
