"""Shared building blocks for Content OMA stores.

Adapted from the original Orchestrated Memory Architecture (Andrea, 2026)
for educational content. The key changes vs the personal-memory OMA:

- Items are namespaced by (user_id, folder_slug) so one storage backend
  can host many folders without cross-contamination.
- store_specific carries content-type tags and concept_id references
  used by every store, not just one.
- Supersession is preserved (course material can be updated, e.g. a
  professor uploads Lecture 5 v2), but contradiction handling is far
  less common than in personal memory.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def new_item_id(store: str) -> str:
    """Generate a short stable ID like `con_20260518_a1b2c3`."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"{store[:3]}_{stamp}_{suffix}"


def make_namespace(user_id: int | str, folder_name: str) -> str:
    """Build a deterministic, filesystem-safe namespace key.

    Example: (1, "Linear Algebra") -> "u1__linear_algebra"
    """
    slug = re.sub(r"[^a-z0-9]+", "_", folder_name.lower()).strip("_")
    return f"u{user_id}__{slug}"


@dataclass
class MemoryItem:
    """Unified item schema across all Content OMA stores.

    - id: stable unique identifier
    - namespace: (user_id, folder) scope key — every read filters on this
    - store: which store owns this item (concept / content / image)
    - content: the human-readable text core of the item
    - entities: linked concept names (canonicalized after ingestion)
    - tags: free-form tags (e.g. 'definition', 'worked_example')
    - importance: 0..1, used in ranking
    - access_count: incremented on each retrieval (popularity signal)
    - superseded_by: set when a newer item replaces this one
    - store_specific: per-store payload (content_type, concept_ids,
      prerequisite_ids, source_doc_id, page_number, image_path, ...)
    """

    id: str
    namespace: str
    store: str
    content: str
    created_at: str = field(default_factory=now_iso)
    last_accessed: str = field(default_factory=now_iso)
    access_count: int = 0
    importance: float = 0.5
    source_doc_id: Optional[str] = None
    entities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    superseded_by: Optional[str] = None
    store_specific: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryItem":
        allowed = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)


def age_days(item: MemoryItem, ref: Optional[datetime] = None) -> float:
    ref = ref or datetime.now()
    try:
        created = datetime.fromisoformat(item.created_at)
    except ValueError:
        return 0.0
    return max(0.0, (ref - created).total_seconds() / 86400.0)
