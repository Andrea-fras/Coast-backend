"""ContentStore — text chunks from lectures with rich metadata.

Each item is a logical chunk of lecture content (often one page or a
small section) tagged with:
  - tags: content types this chunk contains (e.g. ['definition', 'example'])
  - entities: concept_ids this chunk discusses
  - store_specific:
      content_types: list of content_type strings present in this chunk
      concept_ids: list of canonical concept ids
      source_doc_id: the lecture this chunk came from
      page_number: page in the original PDF
      section_title: best-guess section heading
      image_ids: list of ImageStore ids whose figures appear in this chunk
      original_text: the verbatim extracted text
"""

from __future__ import annotations

from typing import Optional

from ._semantic_base import SemanticStoreBase
from .base import MemoryItem


CONTENT_TYPES = (
    "definition",
    "theorem",
    "proof",
    "example",
    "worked_example",
    "exercise",
    "narrative",
    "figure_caption",
    "summary",
    "remark",
)


class ContentStore(SemanticStoreBase):
    STORE_NAME = "content"

    def find_by_concept(
        self,
        namespace: str,
        concept_id: str,
        content_type: Optional[str] = None,
        max_results: int = 8,
    ) -> list[MemoryItem]:
        """Return chunks tagged with this concept. Optionally filter by type."""
        items = self.find_by_entity(namespace, concept_id, max_results=max_results * 3)
        if content_type:
            ct = content_type.lower()
            items = [it for it in items if any(t.lower() == ct for t in it.tags)]
        items.sort(key=lambda it: it.importance, reverse=True)
        return items[:max_results]

    def find_definitions_of(self, namespace: str, concept_id: str, max_results: int = 3) -> list[MemoryItem]:
        return self.find_by_concept(namespace, concept_id, content_type="definition", max_results=max_results)

    def find_examples_of(self, namespace: str, concept_id: str, max_results: int = 3) -> list[MemoryItem]:
        # Combine 'example' and 'worked_example'
        ex = self.find_by_concept(namespace, concept_id, content_type="example", max_results=max_results)
        we = self.find_by_concept(namespace, concept_id, content_type="worked_example", max_results=max_results)
        seen = set()
        out: list[MemoryItem] = []
        for it in we + ex:
            if it.id in seen:
                continue
            seen.add(it.id)
            out.append(it)
        return out[:max_results]

    def find_exercises_of(self, namespace: str, concept_id: str, max_results: int = 3) -> list[MemoryItem]:
        return self.find_by_concept(namespace, concept_id, content_type="exercise", max_results=max_results)

    def hybrid_query(
        self,
        namespace: str,
        query: str,
        concept_ids: Optional[list[str]] = None,
        content_types: Optional[list[str]] = None,
        max_results: int = 8,
    ) -> list[MemoryItem]:
        """Vector + FTS search with optional concept/content_type filters."""
        return self.search(
            namespace,
            query,
            max_results=max_results,
            tag_filter=content_types,
            entity_filter=concept_ids,
        )
