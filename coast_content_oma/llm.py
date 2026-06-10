"""LLM helpers for ingestion and orchestration.

Defaults to Gemini Flash for speed + cost on ingestion. Falls back to
OpenAI gpt-4o-mini if Gemini isn't configured. All functions are
synchronous; the ingestion pipeline parallelizes them itself.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import random
import re
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _retry_on_rate_limit(fn, *, max_attempts: int = 5, base_delay: float = 2.0):
    """Call fn() with exponential backoff on rate-limit / 429 errors."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            result = fn()
            if result is None and attempt < max_attempts - 1:
                # Backoff once on None too — could be a transient parse failure.
                time.sleep(base_delay * (1.5 ** attempt))
                continue
            return result
        except Exception as e:
            msg = str(e).lower()
            last_exc = e
            if "rate" in msg or "429" in msg or "resource_exhausted" in msg or "quota" in msg:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"rate-limited (attempt {attempt+1}/{max_attempts}); sleeping {delay:.1f}s")
                time.sleep(delay)
                continue
            raise
    if last_exc:
        logger.warning(f"giving up after {max_attempts} attempts: {last_exc}")
    return None

GEMINI_TEXT_MODEL = os.environ.get("OMA_GEMINI_MODEL", "gemini-flash-latest")
GEMINI_VISION_MODEL = os.environ.get("OMA_GEMINI_VISION_MODEL", "gemini-flash-latest")
OPENAI_TEXT_MODEL = os.environ.get("OMA_OPENAI_MODEL", "gpt-4o-mini")
OPENAI_VISION_MODEL = os.environ.get("OMA_OPENAI_VISION_MODEL", "gpt-4o-mini")


def _gemini_client():
    if not os.environ.get("GEMINI_API_KEY"):
        return None
    try:
        from google import genai  # google-genai >= 1.x
        return genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    except ImportError:
        try:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=os.environ["GEMINI_API_KEY"])
            return ("legacy", genai_legacy)
        except ImportError:
            return None


def _openai_client():
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
        return OpenAI()
    except ImportError:
        return None


def call_llm_json(
    prompt: str,
    *,
    system: Optional[str] = None,
    max_tokens: int = 2000,
    temperature: float = 0.1,
) -> dict | list | None:
    """Call the LLM and parse a JSON response. Tries Gemini first, then
    OpenAI. Returns None if both fail or the response isn't valid JSON."""
    text = _call_text(prompt, system=system, max_tokens=max_tokens, temperature=temperature)
    if not text:
        return None
    return _parse_json_loose(text)


def call_llm_text(
    prompt: str,
    *,
    system: Optional[str] = None,
    max_tokens: int = 2000,
    temperature: float = 0.2,
) -> Optional[str]:
    return _call_text(prompt, system=system, max_tokens=max_tokens, temperature=temperature)


def _call_text(
    prompt: str,
    *,
    system: Optional[str],
    max_tokens: int,
    temperature: float,
) -> Optional[str]:
    # Gemini first, with backoff.
    client = _gemini_client()
    if client is not None:
        text = _retry_on_rate_limit(
            lambda: _gemini_text(client, prompt, system, max_tokens, temperature),
            max_attempts=3,
        )
        if text:
            return text

    oai = _openai_client()
    if oai is not None:
        return _retry_on_rate_limit(
            lambda: _openai_text(oai, prompt, system, max_tokens, temperature),
            max_attempts=5,
        )
    return None


def _gemini_text(client, prompt: str, system: Optional[str], max_tokens: int, temperature: float) -> Optional[str]:
    try:
        if isinstance(client, tuple) and client[0] == "legacy":
            _, genai = client
            model = genai.GenerativeModel(GEMINI_TEXT_MODEL)
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            resp = model.generate_content(
                full_prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
            return getattr(resp, "text", None)

        # New SDK path.
        from google.genai import types as _types
        contents = []
        if system:
            contents.append(_types.Content(role="user", parts=[_types.Part.from_text(text=system)]))
        contents.append(_types.Content(role="user", parts=[_types.Part.from_text(text=prompt)]))
        # Disable thinking so the full token budget goes to actual output.
        # The new flash models burn most of max_output_tokens on hidden
        # reasoning otherwise, truncating structured-output responses.
        config = _types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=_types.ThinkingConfig(thinking_budget=0),
        )
        resp = client.models.generate_content(model=GEMINI_TEXT_MODEL, contents=contents, config=config)
        return getattr(resp, "text", None)
    except Exception as e:
        logger.warning(f"Gemini call failed: {e}")
        return None


def _openai_text(client, prompt: str, system: Optional[str], max_tokens: int, temperature: float) -> Optional[str]:
    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=OPENAI_TEXT_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.warning(f"OpenAI call failed: {e}")
        return None


def describe_image(pil_image, *, context_hint: str = "", max_tokens: int = 400) -> Optional[dict]:
    """Run vision analysis on a PIL image. Returns a dict like:
        {"description": "...", "image_type": "diagram", "concepts": ["..."]}
    or None on failure."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    prompt = (
        "Describe this figure from a lecture slide. Identify what it shows, "
        "what concepts it illustrates, and what type of image it is.\n\n"
        f"Surrounding text from the slide (for context, may be unrelated): {context_hint[:500]}\n\n"
        "Respond ONLY with a JSON object:\n"
        '{"description": "<one short paragraph of what this image shows>",\n'
        ' "image_type": "<one of: diagram, equation, graph, table, photo, figure, screenshot, decorative>",\n'
        ' "concepts": ["<concept name>", ...]\n'
        "}\n"
        "Use 'decorative' if it's a logo, header image, or has no educational content."
    )

    # Gemini vision first, with backoff.
    client = _gemini_client()
    if client is not None:
        result = _retry_on_rate_limit(
            lambda: _gemini_vision(client, prompt, img_bytes, max_tokens),
            max_attempts=3,
        )
        if result:
            return result

    oai = _openai_client()
    if oai is not None:
        return _retry_on_rate_limit(
            lambda: _openai_vision(oai, prompt, img_bytes, max_tokens),
            max_attempts=5,
        )
    return None


def _gemini_vision(client, prompt: str, img_bytes: bytes, max_tokens: int) -> Optional[dict]:
    try:
        if isinstance(client, tuple) and client[0] == "legacy":
            from PIL import Image
            _, genai = client
            model = genai.GenerativeModel(GEMINI_VISION_MODEL)
            pil = Image.open(io.BytesIO(img_bytes))
            resp = model.generate_content([prompt, pil], generation_config={"max_output_tokens": max_tokens, "temperature": 0.2})
            text = getattr(resp, "text", None)
            return _parse_json_loose(text) if text else None

        from google.genai import types as _types
        parts = [
            _types.Part.from_text(text=prompt),
            _types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
        ]
        contents = [_types.Content(role="user", parts=parts)]
        config = _types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=max_tokens,
            thinking_config=_types.ThinkingConfig(thinking_budget=0),
        )
        resp = client.models.generate_content(model=GEMINI_VISION_MODEL, contents=contents, config=config)
        text = getattr(resp, "text", None)
        return _parse_json_loose(text) if text else None
    except Exception as e:
        logger.warning(f"Gemini vision failed: {e}")
        return None


def _openai_vision(client, prompt: str, img_bytes: bytes, max_tokens: int) -> Optional[dict]:
    try:
        b64 = base64.b64encode(img_bytes).decode("ascii")
        resp = client.chat.completions.create(
            model=OPENAI_VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        text = resp.choices[0].message.content
        return _parse_json_loose(text) if text else None
    except Exception as e:
        logger.warning(f"OpenAI vision failed: {e}")
        return None


# ── JSON parsing ─────────────────────────────────────────────────────

def _parse_json_loose(text: str) -> dict | list | None:
    """Parse JSON from an LLM response that may be wrapped in markdown fences."""
    if not text:
        return None
    text = text.strip()
    # Strip ```json ... ``` fences.
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    # Sometimes LLMs prefix with "Here is the JSON:" etc.
    first_brace = text.find("{")
    first_bracket = text.find("[")
    if first_brace < 0 and first_bracket < 0:
        return None
    start = min(p for p in (first_brace, first_bracket) if p >= 0)
    text = text[start:]
    # Trim anything after the last closing brace/bracket.
    last = max(text.rfind("}"), text.rfind("]"))
    if last >= 0:
        text = text[: last + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to recover a truncated array by finding complete top-level objects.
    if text.startswith("["):
        recovered = _recover_truncated_array(text)
        if recovered:
            return recovered

    # Last-ditch: try to find a JSON object anywhere.
    for m in re.finditer(r"(\{.*\}|\[.*\])", text, re.DOTALL):
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
    return None


def _recover_truncated_array(text: str) -> Optional[list]:
    """Given a JSON array that may be truncated, extract whichever
    top-level objects parsed completely.

    Walks the string tracking brace depth; when we close a top-level
    object (depth 0 inside the outer array), try parsing the slice
    that contains it.
    """
    if not text.startswith("["):
        return None
    items: list = []
    depth = 0
    in_str = False
    escape = False
    obj_start = -1
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start >= 0:
                fragment = text[obj_start : i + 1]
                try:
                    items.append(json.loads(fragment))
                except json.JSONDecodeError:
                    pass
                obj_start = -1
    return items or None


def normalize_concept_name(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace; for hashing/aliasing."""
    s = re.sub(r"[^a-z0-9\s]+", " ", (name or "").lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s
