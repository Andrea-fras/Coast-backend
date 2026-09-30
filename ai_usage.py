"""Token and cost accounting for every AI provider call.

provider_capacity.call/stream hand each response (or stream chunk) to this
module, which reads the provider's own usage numbers, tags them with the
student and feature from the current request, and writes them in batches off
the request path. Costs are computed when read, from AI_PRICES, so fixing a
price re-prices history.
"""
from __future__ import annotations

import contextvars
import logging
import os
import queue
import threading
import time
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)

# USD per million tokens: (input, cached input, output). Anthropic cache writes
# are billed at 1.25x input (5-minute cache). Checked against the providers'
# pricing pages on 2026-09-26; longest matching prefix wins.
AI_PRICES = {
    "claude-opus-5-5": (4.00, 0.20, 20.00),
    "claude-opus-5": (5.00, 0.50, 25.00),
    "claude-sonnet-5": (2.00, 0.20, 10.00),
    "claude-sonnet-4-6": (3.00, 0.30, 15.00),
    "claude-sonnet-4-5": (3.00, 0.30, 15.00),
    "claude-sonnet-4": (3.00, 0.30, 15.00),
    "claude-haiku-4-5": (1.00, 0.10, 5.00),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4o": (2.50, 1.25, 10.00),
    "text-embedding-3-small": (0.02, 0.02, 0.0),
    "gemini-3.1-pro": (2.00, 0.20, 12.00),
    "gemini-3-flash": (0.50, 0.05, 3.00),
    "gemini-2.5-flash": (0.30, 0.03, 2.50),
    "gemini-flash-latest": (0.30, 0.03, 2.50),
}
_CACHE_WRITE_MULTIPLIER = 1.25


def price_for(model: str):
    model = (model or "").lower().removeprefix("models/")
    best = max((k for k in AI_PRICES if model.startswith(k)), key=len, default=None)
    return AI_PRICES.get(best)


def cost_usd(model, input_tokens, cached_tokens, cache_write_tokens, output_tokens):
    """None when the model has no known price."""
    price = price_for(model)
    if not price:
        return None
    base_in, cached_in, out = price
    fresh = max(0, input_tokens - cached_tokens - cache_write_tokens)
    return (fresh * base_in + cached_tokens * cached_in
            + cache_write_tokens * base_in * _CACHE_WRITE_MULTIPLIER + output_tokens * out) / 1e6


# ── Attribution ──────────────────────────────────────────────────────────────
# One mutable dict per request, so code deeper in the request (e.g. Pedro
# knowing it is a lesson turn) can refine the feature for every later call.
_scope: contextvars.ContextVar[dict | None] = contextvars.ContextVar("ai_usage_scope", default=None)


def begin(user_id=None, feature=None):
    """Start attribution for this request or job. Returns a token for end()."""
    return _scope.set({"user_id": user_id, "feature": feature})


def end(token):
    _scope.reset(token)


def tag(feature=None, user_id=None):
    """Refine the current scope (or start one on a thread that has none)."""
    scope = _scope.get()
    if scope is None:
        scope = {}
        _scope.set(scope)
    if feature:
        scope["feature"] = feature
    if user_id is not None:
        scope["user_id"] = user_id


def carry(fn):
    """Run fn on another thread with this request's attribution."""
    ctx = contextvars.copy_context()
    return lambda *args, **kwargs: ctx.run(fn, *args, **kwargs)


def feature_from_path(path: str) -> str:
    """'/api/folders/Cell Bio/outline' → 'folders/outline' (ids and names dropped)."""
    parts = [p for p in path.split("/") if p and p != "api"]
    if parts and parts[0] in ("folders", "notebooks", "workshops") and len(parts) > 1:
        parts = [parts[0]] + parts[2:]
    return "/".join(p for p in parts[:3] if not p.isdigit())[:60] or "api"


# ── Reading usage from provider responses ────────────────────────────────────
def _num(obj, *names):
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _sub(obj, name):
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def read_usage(provider: str, response) -> dict | None:
    """Normalise one response's usage: input (incl. cached), cached, cache_write, output, model."""
    if provider == "anthropic":
        if getattr(response, "type", None) == "message_start":
            response = response.message
        usage = _sub(response, "usage")
        if usage is None:
            return None
        cached = _num(usage, "cache_read_input_tokens")
        written = _num(usage, "cache_creation_input_tokens")
        return {"input": _num(usage, "input_tokens") + cached + written, "cached": cached,
                "cache_write": written, "output": _num(usage, "output_tokens"),
                "model": _sub(response, "model") or ""}
    if provider == "gemini":
        usage = _sub(response, "usage_metadata")
        if usage is None:
            return None
        return {"input": _num(usage, "prompt_token_count"), "cached": _num(usage, "cached_content_token_count"),
                "cache_write": 0,
                "output": _num(usage, "candidates_token_count") + _num(usage, "thoughts_token_count"),
                "model": _sub(response, "model_version") or ""}
    usage = _sub(response, "usage")  # OpenAI chat, embeddings, and compatible APIs
    if usage is None:
        return None
    details = _sub(usage, "prompt_tokens_details")
    return {"input": _num(usage, "prompt_tokens", "input_tokens"),
            "cached": _num(details, "cached_tokens") if details is not None else 0, "cache_write": 0,
            "output": _num(usage, "completion_tokens", "output_tokens"),
            "model": _sub(response, "model") or ""}


class StreamMeter:
    """Accumulates usage across a streamed response."""

    def __init__(self, provider):
        self.provider = provider
        self.usage = None

    def see(self, chunk):
        if self.provider == "anthropic":
            kind = getattr(chunk, "type", None)
            if kind == "message_start":
                self.usage = read_usage("anthropic", chunk)
            elif kind == "message_delta" and self.usage is not None:
                self.usage["output"] = _num(chunk.usage, "output_tokens") or self.usage["output"]
            return
        usage = read_usage(self.provider, chunk)
        if usage and (usage["input"] or usage["output"]):
            # Gemini repeats running totals on every chunk; OpenAI sends one final usage chunk.
            self.usage = {**usage, "model": usage["model"] or (self.usage or {}).get("model", "")}


# ── Recording ────────────────────────────────────────────────────────────────
_ENABLED = os.getenv("COAST_AI_USAGE", "on") != "off"
_RETENTION_DAYS = int(os.getenv("COAST_AI_USAGE_RETENTION_DAYS", "400"))
_rows: queue.SimpleQueue = queue.SimpleQueue()
_writer = None
_writer_lock = threading.Lock()


def record(provider, usage, *, model="", latency_ms=0, ok=True):
    if not _ENABLED:
        return
    scope = _scope.get() or {}
    usage = usage or {}
    feature = scope.get("feature") or threading.current_thread().name.removeprefix("coast-") or "background"
    if feature.startswith(("Thread-", "AnyIO worker")):
        feature = "background"
    _rows.put({
        "created_at": datetime.now(timezone.utc),
        "user_id": scope.get("user_id"),
        "feature": feature[:60],
        "provider": provider,
        "model": (usage.get("model") or model or "")[:80],
        "input_tokens": usage.get("input", 0),
        "cached_tokens": usage.get("cached", 0),
        "cache_write_tokens": usage.get("cache_write", 0),
        "output_tokens": usage.get("output", 0),
        "latency_ms": int(latency_ms),
        "ok": ok,
    })
    _ensure_writer()


def _ensure_writer():
    global _writer
    if _writer is not None:
        return
    with _writer_lock:
        if _writer is None:
            _writer = threading.Thread(target=_write_loop, name="coast-ai-usage-writer", daemon=True)
            _writer.start()


def flush():
    """Write everything queued so far (tests and shutdown)."""
    batch = []
    while True:
        try:
            batch.append(_rows.get_nowait())
        except queue.Empty:
            break
    if not batch:
        return 0
    from database import AiUsage, SessionLocal
    try:
        with SessionLocal() as db:
            db.bulk_insert_mappings(AiUsage, batch)
            db.commit()
    except Exception:
        log.exception("ai_usage: could not write %d rows", len(batch))
    return len(batch)


def _prune():
    from database import AiUsage, SessionLocal
    cutoff = datetime.now(timezone.utc) - timedelta(days=_RETENTION_DAYS)
    with SessionLocal() as db:
        db.query(AiUsage).filter(AiUsage.created_at < cutoff).delete()
        db.commit()


def _write_loop():
    last_prune = 0.0
    while True:
        time.sleep(3)  # one small transaction every few seconds, never one per call
        flush()
        if time.monotonic() - last_prune > 24 * 3600:
            last_prune = time.monotonic()
            try:
                _prune()
            except Exception:
                log.exception("ai_usage: prune failed")


# ── Reporting ────────────────────────────────────────────────────────────────
def summary(days: int = 7) -> dict:
    """Totals and breakdowns for the admin Control Center."""
    from sqlalchemy import case, func
    from database import AiUsage, SessionLocal, User
    flush()
    since = datetime.now(timezone.utc) - timedelta(days=days)
    day = func.date(AiUsage.created_at)
    with SessionLocal() as db:
        rows = (db.query(day, AiUsage.feature, AiUsage.provider, AiUsage.model, AiUsage.user_id,
                         func.count(), func.sum(AiUsage.input_tokens), func.sum(AiUsage.cached_tokens),
                         func.sum(AiUsage.cache_write_tokens), func.sum(AiUsage.output_tokens),
                         func.sum(AiUsage.latency_ms), func.sum(case((AiUsage.ok.is_(False), 1), else_=0)))
                .filter(AiUsage.created_at >= since)
                .group_by(day, AiUsage.feature, AiUsage.provider, AiUsage.model, AiUsage.user_id).all())
        user_ids = {r[4] for r in rows if r[4] is not None}
        names = {u.id: u.email for u in db.query(User.id, User.email).filter(User.id.in_(user_ids))} if user_ids else {}

    def bucket():
        return {"calls": 0, "failed": 0, "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0,
                "cost_usd": 0.0, "unpriced_calls": 0, "latency_ms": 0}

    total, by_feature, by_model, by_day, by_user = bucket(), {}, {}, {}, {}
    unpriced = set()
    for d, feature, provider, model, uid, calls, inp, cached, written, out, latency, failed in rows:
        inp, cached, written, out = inp or 0, cached or 0, written or 0, out or 0
        cost = cost_usd(model, inp, cached, written, out)
        targets = [total, by_feature.setdefault(feature, bucket()),
                   by_model.setdefault((provider, model or "unknown"), bucket()), by_day.setdefault(str(d), bucket())]
        if uid is not None:
            targets.append(by_user.setdefault(uid, bucket()))
        for b in targets:
            b["calls"] += calls
            b["failed"] += failed or 0
            b["input_tokens"] += inp
            b["cached_tokens"] += cached
            b["output_tokens"] += out
            b["latency_ms"] += latency or 0
            if cost is None:
                b["unpriced_calls"] += calls
            else:
                b["cost_usd"] += cost
        if cost is None and (inp or out):
            unpriced.add(model or f"{provider} (model not reported)")

    def finish(b, **extra):
        calls = b["calls"] or 1
        return {**extra, **b, "cost_usd": round(b["cost_usd"], 4),
                "avg_input_tokens": round(b["input_tokens"] / calls),
                "avg_output_tokens": round(b["output_tokens"] / calls),
                "avg_latency_ms": round(b["latency_ms"] / calls),
                "cache_share": round(b["cached_tokens"] / b["input_tokens"], 3) if b["input_tokens"] else 0}

    students = len(by_user)
    return {
        "days": days,
        "prices_checked": "2026-09-26",
        "total": finish(total, students=students,
                        cost_per_student_usd=round(total["cost_usd"] / students, 4) if students else 0),
        "by_feature": sorted((finish(b, feature=k) for k, b in by_feature.items()), key=lambda r: -r["cost_usd"]),
        "by_model": sorted((finish(b, provider=p, model=m) for (p, m), b in by_model.items()), key=lambda r: -r["cost_usd"]),
        "by_day": [finish(by_day[d], day=d) for d in sorted(by_day)],
        "top_students": sorted((finish(b, user_id=u, email=names.get(u, "")) for u, b in by_user.items()),
                               key=lambda r: -r["cost_usd"])[:15],
        "unpriced_models": sorted(unpriced),
    }
