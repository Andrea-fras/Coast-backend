"""Pedro on GPT-5.6 luna: lessons, workshops and global chat.

In side-by-side lessons on the same slides luna taught as well as Claude Sonnet 5 at about a
tenth of the cost per turn, so it writes these turns and Claude answers any turn luna can't.

pedro_context builds each turn for Claude: system blocks, and messages that carry the
section's slide images and cache breakpoints. This sends the same request to OpenAI's Chat
Completions API: images go as data URLs at full detail (Pedro reads numbers off slides), and
the cache breakpoints are dropped, since OpenAI caches a repeated prefix on its own;
prompt_cache_key keeps the turns of one conversation on the same cache.

Env:
  OPENAI_API_KEY
  PEDRO_LUNA         on (default); off puts these turns back on Claude
  PEDRO_LUNA_MODEL   default gpt-5.6-luna
  PEDRO_LUNA_EFFORT  default medium (Sonnet ran at medium too)
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Iterator, Optional

import provider_capacity

log = logging.getLogger(__name__)

ENABLED = os.getenv("PEDRO_LUNA", "on").strip().lower() not in ("off", "0", "false", "no")
LUNA_MODEL = os.getenv("PEDRO_LUNA_MODEL", "gpt-5.6-luna")
LUNA_EFFORT = os.getenv("PEDRO_LUNA_EFFORT", "medium")

_client = None
_client_lock = threading.Lock()


class OpenAIUnavailable(RuntimeError):
    """OpenAI gave no reply; the caller can hand the turn to Claude."""


def _get_client():
    global _client
    with _client_lock:
        if _client is None:
            from openai import OpenAI
            _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""), timeout=300, max_retries=1)
    return _client


def _part(block: dict) -> Optional[dict]:
    kind = block.get("type")
    if kind == "text":
        return {"type": "text", "text": block.get("text", "")}
    if kind == "image":
        source = block.get("source") or {}
        if source.get("type") == "base64":
            url = f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
        elif source.get("type") == "url":
            url = source.get("url", "")
        else:
            return None
        return {"type": "image_url", "image_url": {"url": url, "detail": "high"}}
    return None


def to_openai(system, messages: list[dict]) -> list[dict]:
    """A Claude-shaped request (system blocks, messages with text and image blocks) as Chat
    Completions messages."""
    if isinstance(system, list):
        system = "\n\n".join(b.get("text", "") for b in system if b.get("type") == "text")
    out = [{"role": "developer", "content": system or ""}]
    for message in messages:
        role, content = message["role"], message["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
        elif role == "assistant":  # Pedro's own turns are text
            out.append({"role": "assistant",
                        "content": "".join(b.get("text", "") for b in content if b.get("type") == "text")})
        else:
            out.append({"role": role, "content": [p for p in map(_part, content) if p]})
    return out


def stream_request(system, messages: list[dict], *, cache_key: Optional[str] = None,
                   max_tokens: int = 16000) -> Iterator[str]:
    """Stream Pedro's reply text. Raises OpenAIUnavailable when nothing could be produced, so the
    caller can fail over; a reply that already started is never restarted."""
    if not os.getenv("OPENAI_API_KEY"):
        raise OpenAIUnavailable("OPENAI_API_KEY not set")
    kwargs = dict(model=LUNA_MODEL, messages=to_openai(system, messages), stream=True,
                  stream_options={"include_usage": True}, reasoning_effort=LUNA_EFFORT,
                  max_completion_tokens=max_tokens)
    if cache_key:
        kwargs["prompt_cache_key"] = cache_key[:64]
    started = False
    finish = None
    try:
        for chunk in provider_capacity.stream("openai", lambda: _get_client().chat.completions.create(**kwargs),
                                              priority="interactive"):
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish = choice.finish_reason or finish
            text = getattr(choice.delta, "content", None)
            if text:
                started = True
                yield text
    except Exception as exc:
        if started:
            log.warning("luna reply cut off after it started: %s", exc)
            return
        raise OpenAIUnavailable(f"{type(exc).__name__}: {exc}") from exc
    if finish == "length":
        log.warning("luna reply hit max_completion_tokens=%s and was cut off", max_tokens)
    if not started:
        raise OpenAIUnavailable(f"empty reply (finish_reason={finish})")
