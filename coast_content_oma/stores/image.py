"""ImageStore — extracted lecture images with vision-analyzed descriptions.

Each item represents one extracted image:
  - content: the vision model's description of what the image shows
  - tags: image_type (diagram / equation / graph / table / photo / figure)
  - entities: concept_ids the image relates to
  - store_specific:
      image_type: same as the primary tag (for fast filtering)
      concept_ids: canonical concept ids depicted
      source_doc_id: lecture this image came from
      page_number: page in the source PDF
      file_path: absolute path to PNG on disk
      width, height
      caption: nearby caption text from the lecture (if found)
"""

from __future__ import annotations

from typing import Optional

from ._semantic_base import SemanticStoreBase
from .base import MemoryItem


IMAGE_TYPES = (
    "diagram",
    "equation",
    "graph",
    "table",
    "photo",
    "figure",
    "screenshot",
    "decorative",  # filtered out at retrieval time
)


class ImageStore(SemanticStoreBase):
    STORE_NAME = "image"

    def find_for_concept(
        self,
        namespace: str,
        concept_id: str,
        image_type: Optional[str] = None,
        exclude_decorative: bool = True,
        max_results: int = 4,
    ) -> list[MemoryItem]:
        items = self.find_by_entity(namespace, concept_id, max_results=max_results * 3)
        if exclude_decorative:
            items = [it for it in items if not any(t.lower() == "decorative" for t in it.tags)]
        if image_type:
            it_l = image_type.lower()
            items = [it for it in items if any(t.lower() == it_l for t in it.tags)]
        items.sort(key=lambda it: it.importance, reverse=True)
        return items[:max_results]

    def find_diagrams_for(self, namespace: str, concept_id: str, max_results: int = 3) -> list[MemoryItem]:
        return self.find_for_concept(namespace, concept_id, image_type="diagram", max_results=max_results)

    def find_by_query(
        self,
        namespace: str,
        query: str,
        concept_ids: Optional[list[str]] = None,
        max_results: int = 4,
    ) -> list[MemoryItem]:
        return self.search(
            namespace,
            query,
            max_results=max_results,
            entity_filter=concept_ids,
        )
