"""Concept identity flags — alias ledger rollout and dedup freeze."""

from __future__ import annotations

import os


def _env_truthy(val: str | None) -> bool:
    return (val or "").strip().lower() in ("1", "true", "yes", "on")


def alias_ledger_enabled() -> bool:
    """When False, live concept merges are frozen (embedding dedup disabled)."""
    return _env_truthy(os.environ.get("OMA_ALIAS_LEDGER_ENABLED", "true"))


def effective_merge_threshold(default: float = 0.90) -> float:
    """Embedding-similarity merge threshold; 0 when ledger is off."""
    if not alias_ledger_enabled():
        return 0.0
    raw = os.environ.get("OMA_CONCEPT_MERGE_THRESHOLD")
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default
