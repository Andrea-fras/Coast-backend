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
  ANTHROPIC_FIRST    off (default): grading, roadmaps, section reviews, onboarding traits and Ask
                     sources run on luna first too, with Claude only as their fallback; on puts Claude
                     back in front of them
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Iterator, Optional

import provider_capacity

log = logging.getLogger(__name__)

ENABLED = os.getenv("PEDRO_LUNA", "on").strip().lower() not in ("off", "0", "false", "no")
LUNA_MODEL = os.getenv("PEDRO_LUNA_MODEL", "gpt-5.6-luna")
LUNA_EFFORT = os.getenv("PEDRO_LUNA_EFFORT", "medium")
ANTHROPIC_FIRST = os.getenv("ANTHROPIC_FIRST", "off").strip().lower() in ("on", "1", "true", "yes")

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


def structured(prompt: str, schema: dict, *, system: Optional[str] = None, effort: str = "medium",
               max_tokens: int = 16000, priority: str = "background") -> dict:
    """One luna call whose answer matches `schema` (strict structured output). Raises
    OpenAIUnavailable when there is no usable answer, so the caller can fail over."""
    if not os.getenv("OPENAI_API_KEY"):
        raise OpenAIUnavailable("OPENAI_API_KEY not set")
    messages = ([{"role": "developer", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    try:
        resp = provider_capacity.call("openai", lambda: _get_client().chat.completions.create(
            model=LUNA_MODEL, messages=messages, reasoning_effort=effort, max_completion_tokens=max_tokens,
            response_format={"type": "json_schema", "json_schema": {"name": "result", "schema": schema, "strict": True}},
        ), priority=priority)
    except Exception as exc:
        raise OpenAIUnavailable(f"{type(exc).__name__}: {exc}") from exc
    choice = resp.choices[0]
    if getattr(choice.message, "refusal", None) or choice.finish_reason == "length" or not choice.message.content:
        raise OpenAIUnavailable(f"no structured answer (finish_reason={choice.finish_reason})")
    try:
        return json.loads(choice.message.content)
    except ValueError as exc:
        raise OpenAIUnavailable(f"unreadable structured answer: {exc}") from exc


def stream_messages(messages: list[dict], *, effort: str = LUNA_EFFORT, max_tokens: int = 8000,
                    priority: str = "interactive") -> Iterator[str]:
    """Stream luna's reply to plain chat messages ({role, content} strings; a system message becomes the
    developer message). Raises OpenAIUnavailable if nothing came back; a started reply is never restarted."""
    if not os.getenv("OPENAI_API_KEY"):
        raise OpenAIUnavailable("OPENAI_API_KEY not set")
    converted = [{"role": "developer" if m["role"] == "system" else m["role"], "content": m["content"]} for m in messages]
    kwargs = dict(model=LUNA_MODEL, messages=converted, stream=True, stream_options={"include_usage": True},
                  reasoning_effort=effort, max_completion_tokens=max_tokens)
    started, finish = False, None
    try:
        for chunk in provider_capacity.stream("openai", lambda: _get_client().chat.completions.create(**kwargs),
                                              priority=priority):
            if not chunk.choices:
                continue
            finish = chunk.choices[0].finish_reason or finish
            text = getattr(chunk.choices[0].delta, "content", None)
            if text:
                started = True
                yield text
    except Exception as exc:
        if started:
            log.warning("luna reply cut off after it started: %s", exc)
            return
        raise OpenAIUnavailable(f"{type(exc).__name__}: {exc}") from exc
    if not started:
        raise OpenAIUnavailable(f"empty reply (finish_reason={finish})")

