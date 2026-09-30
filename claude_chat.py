"""Claude for Pedro (streaming chat) and for structured background judgements.

Pedro's system prompt starts with a large, fixed block (identity, teaching rules,
grading protocol) followed by per-turn context (student profile, course material).
The fixed block is sent as its own cached system block, so every turn after the
first reads it from the prompt cache instead of paying for it again.

Models are pinned through env vars:
  ANTHROPIC_PEDRO_MODEL      default claude-sonnet-5 (claude-opus-5-5 scored higher on the Pedro
                             eval but costs about twice as much per reply)
  ANTHROPIC_PEDRO_EFFORT     default medium   (low | medium | high)
  ANTHROPIC_EVAL_MODEL       default claude-opus-5-5
  ANTHROPIC_EVAL_EFFORT      default high
"""
from __future__ import annotations

import json
import logging
import os
from typing import Iterator, Optional

import provider_capacity

log = logging.getLogger(__name__)

PEDRO_MODEL = os.getenv("ANTHROPIC_PEDRO_MODEL", "claude-sonnet-5")
PEDRO_EFFORT = os.getenv("ANTHROPIC_PEDRO_EFFORT", "medium")
EVAL_MODEL = os.getenv("ANTHROPIC_EVAL_MODEL", "claude-opus-5-5")
EVAL_EFFORT = os.getenv("ANTHROPIC_EVAL_EFFORT", "high")

_client = None


class ClaudeUnavailable(RuntimeError):
    """Claude could not produce a usable answer (no key, refusal, truncation, API error)."""


def available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(max_retries=2, timeout=180.0)
    return _client


def _system_blocks(system_text: str, cached_prefix: Optional[str]) -> list[dict]:
    """Split the system prompt so the fixed prefix is cached and the per-turn rest is not."""
    if cached_prefix and system_text.startswith(cached_prefix) and len(system_text) > len(cached_prefix):
        return [
            {"type": "text", "text": cached_prefix, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": system_text[len(cached_prefix):]},
        ]
    return [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]


def _to_claude(messages: list[dict], cached_prefix: Optional[str]) -> tuple[list[dict], list[dict]]:
    """OpenAI-style [{role, content}] -> (system blocks, messages). The first system
    message is Pedro's system prompt; later ones (conversation summaries) are appended
    as uncached system text."""
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    system = _system_blocks(system_parts[0], cached_prefix) if system_parts else []
    system += [{"type": "text", "text": extra} for extra in system_parts[1:]]
    convo = [{"role": "assistant" if m["role"] == "assistant" else "user", "content": m["content"]}
             for m in messages if m["role"] != "system" and (m.get("content") or "").strip()]
    if not convo or convo[0]["role"] != "user":
        convo.insert(0, {"role": "user", "content": "(conversation continues)"})
    return system, convo


def stream_pedro(messages: list[dict], cached_prefix: Optional[str] = None,
                 max_tokens: int = 16000) -> Iterator[str]:
    """Yield Pedro's reply text as it streams. Raises ClaudeUnavailable on refusal or
    failure so the caller can fail over to another provider."""
    system, convo = _to_claude(messages, cached_prefix)
    yield from stream_request(system, convo, max_tokens)


def stream_request(system: list[dict], convo: list[dict], max_tokens: int = 16000) -> Iterator[str]:
    """Stream Pedro's reply for a request that is already in Claude's shape (system
    blocks, and messages that may carry images and cache breakpoints)."""
    if not available():
        raise ClaudeUnavailable("ANTHROPIC_API_KEY not set")
    import anthropic
    try:
        events = provider_capacity.stream("anthropic", lambda: _get_client().messages.create(
            model=PEDRO_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=convo,
            thinking={"type": "adaptive"},
            output_config={"effort": PEDRO_EFFORT},
            stream=True,
        ), priority="interactive")
        stop_reason = None
        for event in events:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                yield event.delta.text
            elif event.type == "message_delta":
                stop_reason = event.delta.stop_reason
        if stop_reason == "refusal":
            raise ClaudeUnavailable("refusal")
        if stop_reason == "max_tokens":
            log.warning("Pedro's reply hit max_tokens=%s and was cut off", max_tokens)
    except ClaudeUnavailable:
        raise
    except anthropic.APIStatusError as exc:
        raise ClaudeUnavailable(f"API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise ClaudeUnavailable(f"connection error: {exc}") from exc
    except (provider_capacity.ProviderUnavailable, TimeoutError) as exc:
        raise ClaudeUnavailable(str(exc)) from exc
    except Exception as exc:  # a dropped stream (httpx read error, malformed event): same handling as an outage
        log.exception("Pedro's Claude stream failed")
        raise ClaudeUnavailable(f"stream failed: {exc}") from exc


def complete_pedro(messages: list[dict], cached_prefix: Optional[str] = None) -> str:
    return "".join(stream_pedro(messages, cached_prefix))


def structured(prompt: str, schema: dict, *, model: Optional[str] = None, effort: Optional[str] = None,
               max_tokens: int = 16000, priority: str = "background") -> dict:
    """One-shot call whose answer is guaranteed to match `schema` (JSON Schema)."""
    if not available():
        raise ClaudeUnavailable("ANTHROPIC_API_KEY not set")
    import anthropic
    try:
        response = provider_capacity.call("anthropic", lambda: _get_client().messages.create(
            model=model or EVAL_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": effort or EVAL_EFFORT,
                           "format": {"type": "json_schema", "schema": schema}},
        ), priority=priority)
    except anthropic.APIStatusError as exc:
        raise ClaudeUnavailable(f"API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise ClaudeUnavailable(f"connection error: {exc}") from exc
    except (provider_capacity.ProviderUnavailable, TimeoutError) as exc:
        raise ClaudeUnavailable(str(exc)) from exc
    if response.stop_reason in ("refusal", "max_tokens"):
        raise ClaudeUnavailable(f"stop_reason={response.stop_reason}")
    text = next((b.text for b in response.content if b.type == "text"), "")
    usage = response.usage
    log.info("claude structured model=%s in=%s out=%s", response.model, usage.input_tokens, usage.output_tokens)
    return json.loads(text)
