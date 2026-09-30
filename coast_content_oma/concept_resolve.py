"""Universal concept ID resolution — Content OMA + Student OMA boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .stores.concept_alias import ConceptAliasStore


class ConceptResolver:
    """Resolve concept IDs through the alias ledger."""

    def __init__(self, alias_store: ConceptAliasStore):
        self.aliases = alias_store

    def resolve(self, namespace: str, concept_id: str) -> str:
        if not concept_id:
            return concept_id
        return self.aliases.resolve(namespace, concept_id)

    def resolve_many(self, namespace: str, concept_ids: list[str]) -> list[str]:
        return self.aliases.resolve_many(namespace, concept_ids)

    def ids_for_canonical(self, namespace: str, canonical_id: str) -> list[str]:
        return self.aliases.ids_for_canonical(namespace, canonical_id)

    def resolve_edge_ids(self, namespace: str, ids: list[str]) -> list[str]:
        """Resolve prerequisite/related ID lists, dedupe, drop self-loops."""
        return self.resolve_many(namespace, ids or [])


def stamp_concept_ids(
    content_namespace: str,
    concept_ids: list[str],
    resolver: ConceptResolver,
) -> tuple[list[str], list[str]]:
    """Return (matched, resolved_canonical) for episode attribution."""
    matched = [c for c in (concept_ids or []) if c]
    resolved = resolver.resolve_many(content_namespace, matched) if matched else []
    return matched, resolved


def resolver_for_db(db_path: Path | str) -> ConceptResolver:
    return ConceptResolver(ConceptAliasStore(db_path))
