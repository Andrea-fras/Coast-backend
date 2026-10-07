"""LLM helpers for ingestion and orchestration.

Defaults to Gemini Flash for speed + cost on ingestion. Falls back to
OpenAI gpt-4o-mini if Gemini isn't configured. All functions are
synchronous; the ingestion pipeline parallelizes them itself.
"""

from __future__ import annotations

import provider_capacity

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


def _is_rate_limit(exc):
    status = getattr(exc, 'status_code', None) or getattr(exc, 'code', None)
    return status == 429 or any(word in str(exc).lower() for word in ('429', 'resource_exhausted', 'rate limit', 'quota'))


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
            last_exc = e
            if provider_capacity.is_credit_error(e):
                logger.warning('Provider credits unavailable; skipping rate-limit retries')
                return None
            if _is_rate_limit(e):
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"rate-limited (attempt {attempt+1}/{max_attempts}); sleeping {delay:.1f}s")
                if attempt < max_attempts - 1:
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


def _openai_limits(model: str, max_tokens: int, temperature: float) -> dict:
    """Reasoning models (gpt-5 family) take max_completion_tokens, which also covers their
    reasoning, and only their default temperature."""
    if model.startswith(("gpt-5", "o")):
        return {"max_completion_tokens": max_tokens + 2000, "reasoning_effort": "low"}
    return {"max_tokens": max_tokens, "temperature": temperature}


def _gemini_client():
    # OMA_READER=openai reads uploads with OpenAI alone (e.g. while Gemini is overloaded).
    if not os.environ.get("GEMINI_API_KEY") or os.environ.get("OMA_READER", "gemini").lower() == "openai":
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
        return OpenAI(max_retries=0, timeout=45)
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
            resp = provider_capacity.call('gemini', lambda: model.generate_content(
                full_prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            ), priority='background')
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
        resp = provider_capacity.call('gemini', lambda: client.models.generate_content(model=GEMINI_TEXT_MODEL, contents=contents, config=config), priority='background')
        return getattr(resp, "text", None)
    except Exception as e:
        if _is_rate_limit(e):
            raise
        logger.warning(f"Gemini call failed: {e}")
        return None


def _openai_text(client, prompt: str, system: Optional[str], max_tokens: int, temperature: float) -> Optional[str]:
    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = provider_capacity.call('openai', lambda: client.chat.completions.create(
            model=OPENAI_TEXT_MODEL,
            messages=messages,
            **_openai_limits(OPENAI_TEXT_MODEL, max_tokens, temperature),
        ), priority='background')
        return resp.choices[0].message.content
    except Exception as e:
        if _is_rate_limit(e):
            raise
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
            resp = provider_capacity.call('gemini', lambda: model.generate_content([prompt, pil], generation_config={"max_output_tokens": max_tokens, "temperature": 0.2}), priority='background')
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
        resp = provider_capacity.call('gemini', lambda: client.models.generate_content(model=GEMINI_VISION_MODEL, contents=contents, config=config), priority='background')
        text = getattr(resp, "text", None)
        return _parse_json_loose(text) if text else None
    except Exception as e:
        if _is_rate_limit(e):
            raise
        logger.warning(f"Gemini vision failed: {e}")
        return None


def _openai_vision(client, prompt: str, img_bytes: bytes, max_tokens: int) -> Optional[dict]:
    try:
        b64 = base64.b64encode(img_bytes).decode("ascii")
        resp = provider_capacity.call('openai', lambda: client.chat.completions.create(
            model=OPENAI_VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            **_openai_limits(OPENAI_VISION_MODEL, max_tokens, 0.2),
        ), priority='background')
        text = resp.choices[0].message.content
        return _parse_json_loose(text) if text else None
    except Exception as e:
        if _is_rate_limit(e):
            raise
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


# ── Bulk embeddings (canonicalization clustering) ───────────────────

EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
EMBED_BATCH = 128


def embed_texts(texts: list[str]) -> list[Optional[list[float]]]:
    """Embed many strings in bulk. Returns None per item when API unavailable."""
    if not texts:
        return []
    oai = _openai_client()
    if oai is None:
        return [None] * len(texts)
    out: list[Optional[list[float]]] = [None] * len(texts)
    for start in range(0, len(texts), EMBED_BATCH):
        chunk = [(t or "")[:8000] for t in texts[start : start + EMBED_BATCH]]
        try:
            resp = provider_capacity.call('openai', lambda: oai.embeddings.create(model=EMBED_MODEL, input=chunk), priority='background')
            for i, data in enumerate(resp.data):
                out[start + i] = list(data.embedding)
        except Exception as e:
            logger.warning(f"bulk embed failed: {e}")
    return out


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


# ── Multi-image vision batch ─────────────────────────────────────────

VISION_BATCH_SYSTEM = (
    "You analyse figures extracted from lecture PDFs. For each image, describe "
    "what it shows, classify its type, and list concepts it illustrates."
)


def describe_images_batch(
    items: list[dict],
    *,
    max_tokens: int = 4000,
) -> list[Optional[dict]]:
    """Describe multiple figures in one vision call.

    Each item: {image_index, png_bytes, context_hint}
    Returns list aligned with input order (None on failure for that slot).
    """
    if not items:
        return []
    if len(items) == 1:
        from PIL import Image
        pil = Image.open(io.BytesIO(items[0]["png_bytes"]))
        one = describe_image(pil, context_hint=items[0].get("context_hint", ""))
        return [one]

    n = len(items)
    index_lines = []
    for it in items:
        idx = it.get("image_index", 0)
        hint = (it.get("context_hint") or "")[:300]
        index_lines.append(f"Image {idx}: slide context — {hint or '(none)'}")

    prompt = (
        f"You are given {n} lecture figures, numbered {items[0].get('image_index', 1)}"
        f" through {items[-1].get('image_index', n)}.\n\n"
        + "\n".join(index_lines)
        + "\n\nRespond ONLY with a JSON array — one object per image, in order:\n"
        '[{"image_index": <int>, "description": "...", '
        '"image_type": "<diagram|equation|graph|table|photo|figure|screenshot|decorative>", '
        '"concepts": ["...", ...]}, ...]\n'
        "Use decorative for logos/headers with no educational content."
    )

    png_list = [it["png_bytes"] for it in items]

    client = _gemini_client()
    if client is not None:
        result = _retry_on_rate_limit(
            lambda: _gemini_vision_multi(client, prompt, png_list, max_tokens),
            max_attempts=3,
        )
        if result is not None:
            return result

    oai = _openai_client()
    if oai is not None:
        result = _retry_on_rate_limit(
            lambda: _openai_vision_multi(oai, prompt, png_list, max_tokens),
            max_attempts=5,
        )
        if result is not None:
            return result

    return [None] * n


def _align_vision_batch_results(
    items: list[dict],
    parsed: list | None,
) -> list[Optional[dict]]:
    """Map LLM array output back to input order by image_index."""
    out: list[Optional[dict]] = [None] * len(items)
    if not isinstance(parsed, list):
        return out
    by_idx = {}
    for row in parsed:
        if isinstance(row, dict) and row.get("image_index") is not None:
            by_idx[int(row["image_index"])] = row
    for i, it in enumerate(items):
        idx = int(it.get("image_index", i + 1))
        row = by_idx.get(idx)
        if isinstance(row, dict):
            out[i] = {
                "description": row.get("description") or "",
                "image_type": row.get("image_type") or "figure",
                "concepts": row.get("concepts") or [],
            }
    return out


def _gemini_vision_multi(client, prompt: str, png_list: list[bytes], max_tokens: int) -> list[Optional[dict]]:
    try:
        from google.genai import types as _types

        parts = [_types.Part.from_text(text=prompt)]
        for png in png_list:
            parts.append(_types.Part.from_bytes(data=png, mime_type="image/png"))
        contents = [_types.Content(role="user", parts=parts)]
        config = _types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=max_tokens,
            thinking_config=_types.ThinkingConfig(thinking_budget=0),
        )
        resp = provider_capacity.call('gemini', lambda: client.models.generate_content(
            model=GEMINI_VISION_MODEL, contents=contents, config=config,
        ), priority='background')
        text = getattr(resp, "text", None)
        parsed = _parse_json_loose(text) if text else None
        items = [{"image_index": i + 1} for i in range(len(png_list))]
        return _align_vision_batch_results(items, parsed if isinstance(parsed, list) else None)
    except Exception as e:
        if _is_rate_limit(e):
            raise
        logger.warning(f"Gemini multi-vision failed: {e}")
        return None


def _openai_vision_multi(client, prompt: str, png_list: list[bytes], max_tokens: int) -> list[Optional[dict]]:
    try:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for png in png_list:
            b64 = base64.b64encode(png).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        resp = provider_capacity.call('openai', lambda: client.chat.completions.create(
            model=OPENAI_VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            **_openai_limits(OPENAI_VISION_MODEL, max_tokens, 0.2),
        ), priority='background')
        text = resp.choices[0].message.content
        parsed = _parse_json_loose(text) if text else None
        items = [{"image_index": i + 1} for i in range(len(png_list))]
        return _align_vision_batch_results(items, parsed if isinstance(parsed, list) else None)
    except Exception as e:
        if _is_rate_limit(e):
            raise
        logger.warning(f"OpenAI multi-vision failed: {e}")
        return None
