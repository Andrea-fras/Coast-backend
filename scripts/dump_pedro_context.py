#!/usr/bin/env python3
"""Show exactly what Pedro receives for one chat turn, without calling a model.

Runs the real prompt-building path (tutor.send_message_stream) against throwaway
copies of coast.db and oma.db, intercepts the Claude request just before it is
sent, and writes the system prompt, conversation and token counts to a Markdown
file. Nothing is written to the real databases.

    python3 scripts/dump_pedro_context.py --user 11 --folder "Network Science" \
        --section 9 --message 'I'm ready to learn about "Paths and Distances in Networks". Please teach me this section.' \
        --out /tmp/section10-opener.md
    python3 scripts/dump_pedro_context.py --user 11 --global \
        --message "does my learner profile suggest i am already strong in basic graph theory" --out /tmp/global.md

Course retrieval embeds the student's message once (OpenAI embeddings, a fraction
of a cent). Token counts use Anthropic's free count_tokens endpoint when available.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, required=True)
    ap.add_argument("--folder")
    ap.add_argument("--section", type=int, help="0-based section index (lesson turns)")
    ap.add_argument("--global", dest="is_global", action="store_true", help="a global chat turn")
    ap.add_argument("--conversation", help="replay inside an existing conversation (its history is included)")
    ap.add_argument("--message", help="the student's message (default: the text of --until)")
    ap.add_argument("--until", type=int, help="replay the turn of this stored student message: history stops just before it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", help="also save the request (system blocks + messages) as JSON for replay")
    ap.add_argument("--history", help="JSON list of [role, text] turns to replay as a fresh conversation (roles: user, pedro)")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    tmp = Path(tempfile.mkdtemp(prefix="pedro-context-"))
    for name, env in (("coast.db", "DATABASE_PATH"), ("oma.db", "OMA_DB_PATH")):
        src = Path(os.environ.get(env) or (ROOT / ("oma_data/oma.db" if name == "oma.db" else name)))
        shutil.copy2(src, tmp / name)
        for suffix in ("-wal", "-shm"):
            if Path(str(src) + suffix).exists():
                shutil.copy2(str(src) + suffix, str(tmp / name) + suffix)
        os.environ[env] = str(tmp / name)
    os.environ["CHROMA_PATH"] = str(tmp / "chroma")
    if args.until:
        import sqlite3
        from datetime import datetime, timezone
        with sqlite3.connect(tmp / "coast.db") as db:
            row = db.execute("SELECT conversation_id, content, section_index, created_at FROM chat_messages WHERE id = ?",
                             (args.until,)).fetchone()
            if not row:
                raise SystemExit(f"message {args.until} not found")
            args.conversation = args.conversation or row[0]
            args.message = args.message or row[1]
            if args.section is None and row[2] is not None:
                args.section = row[2]
            # Rewind the student's whole record to that moment: later turns of every
            # conversation, and any digest that may summarise them.
            db.execute("DELETE FROM chat_messages WHERE user_id = ? AND id >= ?", (args.user, args.until))
            db.execute("DELETE FROM conversation_digests WHERE conversation_id = ?", (row[0],))
        # Student OMA stamps local time; chat_messages stores UTC.
        cutoff = (datetime.fromisoformat(row[3]).replace(tzinfo=timezone.utc).astimezone()
                  .replace(tzinfo=None).isoformat(timespec="seconds"))
        with sqlite3.connect(tmp / "oma.db") as db:
            for table in ("pattern_items", "academic_identity_items"):
                db.execute(f"DELETE FROM {table} WHERE namespace LIKE ? AND created_at >= ?", (f"u{args.user}__%", cutoff))
    if args.history:
        import json
        import sqlite3
        from datetime import datetime, timedelta
        turns = json.loads(Path(args.history).read_text())
        args.conversation = "conv_eval_" + Path(args.history).stem.replace(".", "_")
        start = datetime.utcnow() - timedelta(minutes=len(turns) + 1)
        with sqlite3.connect(tmp / "coast.db") as db:
            for i, (role, text) in enumerate(turns):
                db.execute("INSERT INTO chat_messages (user_id, conversation_id, role, content, context_type, context_id, "
                           "created_at, section_index) VALUES (?, ?, ?, ?, 'lesson', ?, ?, ?)",
                           (args.user, args.conversation, role, text, args.folder,
                            (start + timedelta(minutes=i)).isoformat(sep=" "), args.section))
    if not args.message:
        raise SystemExit("--message or --until is required")
    os.environ["COAST_AI_USAGE"] = "off"

    import claude_chat
    import tutor

    captured: dict = {}

    def capture(system, convo, max_tokens=16000):
        captured.update(system=system, convo=convo, max_tokens=max_tokens)
        yield "(captured)"

    claude_chat.stream_request = capture  # both context versions end here
    # Keep the dump free of background model calls and memory writes.
    tutor._trigger_memo_update_bg = lambda *a, **k: None
    import oma_provider
    oma_provider.record_conversation_turn = lambda *a, **k: None

    kwargs = dict(user_id=args.user, message=args.message, conversation_id=args.conversation)
    if args.is_global:
        kwargs.update(context_type="global")
    else:
        kwargs.update(context_type="lesson", context_id=args.folder, section_index=args.section)
    for _ in tutor.send_message_stream(**kwargs):
        pass
    if not captured:
        raise SystemExit("Pedro was not called (check the provider settings).")

    system_blocks, convo = captured["system"], captured["convo"]
    counts = token_counts(system_blocks, convo)
    lines = [f"# Pedro context · {'global chat' if args.is_global else f'{args.folder} · section {args.section + 1}'}",
             "", f"- Context: `{os.getenv('PEDRO_CONTEXT', 'v1')}` · model `{claude_chat.PEDRO_MODEL}` · effort "
                 f"`{claude_chat.PEDRO_EFFORT}` · max_tokens {captured['max_tokens']}",
             f"- Student message: {args.message!r}"]
    if counts:
        lines += [f"- Tokens: {counts}"]
    for i, block in enumerate(system_blocks):
        cache = " (cached)" if block.get("cache_control") else ""
        lines += ["", f"## System block {i + 1}{cache} · {len(block['text']):,} chars", "", "```text", block["text"], "```"]
    for m in convo:
        blocks = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        images = sum(b["type"] == "image" for b in blocks)
        size = sum(len(b.get("text", "")) for b in blocks)
        lines += ["", f"## {m['role']} · {size:,} chars" + (f" · {images} page images" if images else ""), ""]
        for b in blocks:
            cache = " (cache breakpoint)" if b.get("cache_control") else ""
            if b["type"] == "image":
                lines += [f"[page image, {b['source']['media_type']}, {len(b['source']['data']) * 3 // 4 // 1024} KB]{cache}"]
            else:
                lines += ["```text", b["text"], "```" + cache]
    Path(args.out).write_text("\n".join(lines))
    if args.json:
        import json
        Path(args.json).write_text(json.dumps({"system": system_blocks, "messages": convo,
                                              "model": claude_chat.PEDRO_MODEL, "effort": claude_chat.PEDRO_EFFORT}, indent=1))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {args.out} · {len(system_blocks)} system blocks, {len(convo)} messages · {counts or 'token count unavailable'}")


def token_counts(system_blocks, convo):
    try:
        import anthropic
        client = anthropic.Anthropic()
        import claude_chat
        total = client.messages.count_tokens(model=claude_chat.PEDRO_MODEL, system=system_blocks, messages=convo).input_tokens
        parts = []
        for i, block in enumerate(system_blocks):
            n = client.messages.count_tokens(model=claude_chat.PEDRO_MODEL, system=[{"type": "text", "text": block["text"]}],
                                             messages=[{"role": "user", "content": "."}]).input_tokens
            parts.append(f"system {i + 1} ≈ {n:,}")
        first = convo[0]["content"]
        if isinstance(first, list) and any(b.get("cache_control") for b in first):
            upto = next(i for i, b in enumerate(first) if b.get("cache_control")) + 1
            n = client.messages.count_tokens(model=claude_chat.PEDRO_MODEL, system=system_blocks,
                                             messages=[{"role": "user", "content": first[:upto]}]).input_tokens
            parts.append(f"system + slides ≈ {n:,} (cached for the section)")
        return f"{total:,} input in total ({', '.join(parts)})"
    except Exception as exc:  # counting is informational only
        return f"(count_tokens failed: {type(exc).__name__})"


if __name__ == "__main__":
    main()
