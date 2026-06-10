"""Content OMA orchestrator.

Classifies an education query, routes to the right stores, merges and
ranks hits, and produces a structured context block.

Query classes for education:
  - definitional   : "what is X", "define X"
  - example_request: "show me a worked example", "give me an example of X"
  - exercise       : "give me a practice problem", "test me on X"
  - how_to         : "how do I solve...", "what's the method for..."
  - prerequisite   : "what do I need to know before X", "before learning X"
  - figure         : "show me a diagram", "is there a figure for X"
  - lesson_outline : "build me a lesson on X" (used by lesson generator)
  - explanation    : "explain X", "I don't understand X"
  - general        : fallback — hits all stores with concept-aware ranking
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .stores import (
    ConceptStore,
    ContentStore,
    ImageStore,
    MemoryItem,
    make_namespace,
)

logger = logging.getLogger(__name__)


QueryClass = str  # see module docstring for the set


_PATTERNS: list[tuple[QueryClass, list[str]]] = [
    ("definitional", [
        r"\bwhat (is|are|does)\b",
        r"\bdefine\b",
        r"\bdefinition of\b",
        r"\bmeaning of\b",
    ]),
    ("example_request", [
        r"\b(show|give|provide) (me )?(an? )?(worked )?example\b",
        r"\bexamples? (of|for)\b",
        r"\bdemonstrate\b",
        r"\bwalk me through\b",
    ]),
    ("exercise", [
        r"\b(practice|exercise|problem|drill|test me)\b",
        r"\bquiz me\b",
        r"\b(let me|i want to) (try|practice)\b",
    ]),
    ("how_to", [
        r"\bhow (do|can|would|should) (i|we|you)\b",
        r"\bsteps to\b",
        r"\bmethod for\b",
        r"\bprocedure (for|to)\b",
    ]),
    ("prerequisite", [
        r"\b(prerequisites?|pre-?requisites?)\b",
        r"\b(need to know|need to learn|should know|should learn) (before|first|prior)\b",
        r"\bwhat (do|should) i (know|learn|study) (first|before|prior)\b",
        r"\b(learn|study|know|understand|cover) .{0,30}\bbefore\b",
        r"\bbefore (i|we|you) (can )?(learn|study|tackle|understand|cover|approach|do)\b",
        r"\bwhat comes before\b",
        r"\b(foundation|background) (for|to|needed for)\b",
    ]),
    ("figure", [
        r"\b(diagram|figure|picture|image|graph|chart|visual)\b",
        r"\bshow me\b",
    ]),
    ("explanation", [
        r"\bexplain\b",
        r"\bi don'?t (understand|get|see)\b",
        r"\bconfused (about|by)\b",
        r"\bwhy (is|does|do)\b",
        r"\bhelp me understand\b",
    ]),
]


def classify_query(query: str) -> QueryClass:
    q = (query or "").lower().strip()
    if not q:
        return "general"
    for label, patterns in _PATTERNS:
        if any(re.search(p, q) for p in patterns):
            return label
    return "general"


@dataclass
class ContextChunk:
    """A unit of retrieved context with provenance."""
    item: MemoryItem
    store: str
    score: float
    why: str  # human-readable reason for inclusion


@dataclass
class RetrievalResult:
    """What gets returned to the caller and turned into a prompt block."""
    query: str
    query_class: QueryClass
    concept_candidates: list[MemoryItem]
    chunks: list[ContextChunk]
    images: list[ContextChunk]
    fallback_used: bool = False

    def to_prompt_block(self, max_chars: int = 8000, image_base_url: Optional[str] = None) -> str:
        """Render as a structured text block suitable for injecting into
        Pedro's system prompt. Keeps under max_chars by truncating
        content items, not by dropping them.
        """
        out: list[str] = []
        out.append(f"--- COURSE MATERIAL (retrieved via Content OMA, query class: {self.query_class}) ---")

        if self.concept_candidates:
            out.append("\n[Relevant concepts]")
            for c in self.concept_candidates[:4]:
                ss = c.store_specific or {}
                name = ss.get("name") or "?"
                definition = ss.get("definition") or c.content
                pre_ids = ss.get("prerequisite_concept_ids") or []
                pre_note = f"  (prerequisites: {len(pre_ids)})" if pre_ids else ""
                out.append(f"- {name}{pre_note}")
                if definition:
                    out.append(f"    {definition[:300]}")

        if self.chunks:
            out.append("\n[Course material excerpts]")
            for ch in self.chunks[:8]:
                it = ch.item
                ss = it.store_specific or {}
                page = ss.get("page_number")
                src = ss.get("source_filename") or it.source_doc_id or "?"
                types = ss.get("content_types") or it.tags
                tag_str = ", ".join(types[:3]) if types else ""
                header = f"- [{src} p.{page}] ({tag_str})" if page else f"- [{src}] ({tag_str})"
                out.append(header)
                body = (it.content or "").strip().replace("\n", " ")
                # Each chunk capped at 700 chars in the prompt.
                if len(body) > 700:
                    body = body[:700] + "…"
                out.append(f"    {body}")

        diagram_lines: list[str] = []
        for ch in self.images[:4]:
            it = ch.item
            ss = it.store_specific or {}
            desc = (it.content or "").strip()
            if not desc or desc in ("(no description)", "(no description yet)"):
                continue
            if ss.get("_pending_vision"):
                continue
            page = ss.get("page_number")
            src = ss.get("source_filename") or "?"
            img_type = ss.get("image_type") or "figure"
            if image_base_url:
                url = f"{image_base_url.rstrip('/')}/{it.id}"
                diagram_lines.append(f"- ![{img_type} on p.{page}]({url})")
            else:
                diagram_lines.append(f"- [{img_type} from {src} p.{page}, id={it.id}]")
            diagram_lines.append(f"    {desc[:300]}")
        if diagram_lines:
            out.append("\n[Available diagrams]")
            out.extend(diagram_lines)
            out.append(
                "\nUse a diagram with markdown ![desc](URL) ONLY when its description "
                "clearly matches what you're explaining."
            )

        if self.fallback_used:
            out.append("\n(Note: structured retrieval found limited matches; broader semantic search used as fallback.)")

        out.append("--- END COURSE MATERIAL ---")

        body = "\n".join(out)
        if len(body) > max_chars:
            body = body[:max_chars] + "\n[... truncated ...]"
        return body


class ContentOrchestrator:
    """Routes queries across the three Content OMA stores."""

    def __init__(
        self,
        concept_store: ConceptStore,
        content_store: ContentStore,
        image_store: ImageStore,
    ):
        self.concept = concept_store
        self.content = content_store
        self.images = image_store

    def retrieve(
        self,
        namespace: str,
        query: str,
        *,
        max_content: int = 8,
        max_images: int = 4,
    ) -> RetrievalResult:
        qclass = classify_query(query)

        # Step 1: find concept candidates for this query (always — gives
        # us concept_ids to filter content/image queries).
        concept_candidates = self.concept.find_candidates(namespace, query, max_results=4)
        concept_ids = [c.id for c in concept_candidates]

        chunks: list[ContextChunk] = []
        images: list[ContextChunk] = []
        fallback_used = False

        # Step 2: route by class.
        if qclass == "definitional":
            chunks = self._collect_definitions(namespace, concept_ids, query, max_content)
            images = self._collect_images(
                namespace, concept_ids, query, max_images,
                prefer_types=["diagram", "figure", "graph"],
            )

        elif qclass == "example_request":
            chunks = self._collect_examples(namespace, concept_ids, query, max_content)
            images = self._collect_images(namespace, concept_ids, query, max_images, prefer_types=["figure", "diagram"])

        elif qclass == "exercise":
            chunks = self._collect_exercises(namespace, concept_ids, query, max_content)

        elif qclass == "how_to":
            chunks = self._collect_examples(namespace, concept_ids, query, max_content)
            # Add narrative/proof content for explanatory how-tos.
            chunks += self._collect_by_types(namespace, concept_ids, query, ["narrative", "proof"], max_content // 2)
            images = self._collect_images(
                namespace, concept_ids, query, max_images,
                prefer_types=["diagram", "figure"],
            )

        elif qclass == "prerequisite":
            if concept_candidates:
                top_concept = concept_candidates[0]
                pre_chain = self.concept.traverse_prereq_chain(namespace, top_concept.id, max_depth=4)
                # Return the prerequisite concepts themselves as the primary result.
                concept_candidates = [top_concept] + pre_chain
                # Also pull definitions from ContentStore for each prereq.
                chunks = []
                for p in pre_chain[:4]:
                    chunks += self._collect_definitions(namespace, [p.id], query, 2)

        elif qclass == "figure":
            images = self._collect_images(namespace, concept_ids, query, max_images * 2, prefer_types=["diagram", "graph", "figure"])
            chunks = self._collect_general(namespace, concept_ids, query, max_content // 2)

        elif qclass == "explanation":
            # Definitions + best examples + diagrams.
            chunks = self._collect_definitions(namespace, concept_ids, query, max_content // 2)
            chunks += self._collect_examples(namespace, concept_ids, query, max_content // 2)
            images = self._collect_images(namespace, concept_ids, query, max_images, prefer_types=["diagram", "figure"])

        else:  # general
            chunks = self._collect_general(namespace, concept_ids, query, max_content)
            images = self._collect_images(namespace, concept_ids, query, max_images)

        # Step 3: fallback — if we got nothing, do a broad semantic search.
        if not chunks and not images and not concept_candidates:
            fallback_used = True
            results = self.content.search(namespace, query, max_results=max_content)
            chunks = [
                ContextChunk(item=r, store="content", score=1.0, why="fallback-semantic")
                for r in results
            ]
            img_results = self.images.search(namespace, query, max_results=max_images * 2)
            images = [
                ContextChunk(item=r, store="image", score=1.0, why="fallback-semantic")
                for r in img_results
                if ContentOrchestrator._image_is_usable(r)
            ][:max_images]

        # Dedupe and rank.
        chunks = self._dedupe(chunks)[:max_content]
        images = self._dedupe(images)[:max_images]

        return RetrievalResult(
            query=query,
            query_class=qclass,
            concept_candidates=concept_candidates,
            chunks=chunks,
            images=images,
            fallback_used=fallback_used,
        )

    # ── Collectors ────────────────────────────────────────────────

    def _collect_definitions(self, namespace: str, concept_ids: list[str], query: str, k: int) -> list[ContextChunk]:
        """Reciprocal rank fusion across three retrieval strategies.
        Earlier-ranked items get higher scores so a literal answer chunk
        retrieved by plain semantic search can still outrank a weakly
        related concept-filtered chunk."""
        out: list[ContextChunk] = []
        # 1. Concept-linked definitions (highest weight — strongest signal).
        rank = 0
        for cid in concept_ids:
            for it in self.content.find_definitions_of(namespace, cid, max_results=2):
                out.append(ContextChunk(item=it, store="content",
                                        score=1.5 / (rank + 1), why=f"definition of {cid}"))
                rank += 1
        # 2. Definition-tagged semantic search.
        for r, it in enumerate(self.content.hybrid_query(namespace, query, content_types=["definition"], max_results=k * 2)):
            out.append(ContextChunk(item=it, store="content",
                                    score=1.0 / (r + 1), why="definition-search"))
        # 3. Plain semantic search — catches answers whose chunks aren't
        # tagged 'definition' but literally contain the answer.
        for r, it in enumerate(self.content.search(namespace, query, max_results=k * 2)):
            out.append(ContextChunk(item=it, store="content",
                                    score=0.9 / (r + 1), why="semantic"))
        return out[:k * 3]  # dedupe in retrieve() picks the best.

    def _collect_examples(self, namespace: str, concept_ids: list[str], query: str, k: int) -> list[ContextChunk]:
        out: list[ContextChunk] = []
        rank = 0
        for cid in concept_ids:
            for it in self.content.find_examples_of(namespace, cid, max_results=2):
                out.append(ContextChunk(item=it, store="content",
                                        score=1.5 / (rank + 1), why=f"example of {cid}"))
                rank += 1
        for r, it in enumerate(self.content.hybrid_query(namespace, query, content_types=["example", "worked_example"], max_results=k * 2)):
            out.append(ContextChunk(item=it, store="content",
                                    score=1.0 / (r + 1), why="example-search"))
        for r, it in enumerate(self.content.search(namespace, query, max_results=k * 2)):
            out.append(ContextChunk(item=it, store="content",
                                    score=0.9 / (r + 1), why="semantic"))
        return out[:k * 3]

    def _collect_exercises(self, namespace: str, concept_ids: list[str], query: str, k: int) -> list[ContextChunk]:
        out: list[ContextChunk] = []
        for cid in concept_ids:
            for it in self.content.find_exercises_of(namespace, cid, max_results=2):
                out.append(ContextChunk(item=it, store="content", score=it.importance, why=f"exercise on {cid}"))
        if len(out) < k:
            extra = self.content.hybrid_query(namespace, query, content_types=["exercise"], max_results=k)
            for it in extra:
                out.append(ContextChunk(item=it, store="content", score=it.importance, why="exercise-search"))
        return out[:k]

    def _collect_by_types(self, namespace: str, concept_ids: list[str], query: str, types: list[str], k: int) -> list[ContextChunk]:
        results = self.content.hybrid_query(namespace, query, concept_ids=concept_ids, content_types=types, max_results=k)
        return [ContextChunk(item=it, store="content", score=it.importance, why=f"type-match {types}") for it in results]

    def _collect_general(self, namespace: str, concept_ids: list[str], query: str, k: int) -> list[ContextChunk]:
        """Rank-based fusion of concept-filtered + plain semantic search.
        Concept hits get a modest boost so they outrank ties, but a strong
        plain-semantic match can still win when the concept lookup misses."""
        chunks: list[ContextChunk] = []
        if concept_ids:
            for r, it in enumerate(self.content.hybrid_query(namespace, query, concept_ids=concept_ids, max_results=k * 2)):
                chunks.append(ContextChunk(item=it, store="content",
                                           score=1.2 / (r + 1), why="concept-hybrid"))
        for r, it in enumerate(self.content.search(namespace, query, max_results=k * 2)):
            chunks.append(ContextChunk(item=it, store="content",
                                       score=1.0 / (r + 1), why="semantic"))
        return chunks

    def _collect_images(self, namespace: str, concept_ids: list[str], query: str, k: int, prefer_types: Optional[list[str]] = None) -> list[ContextChunk]:
        out: list[ContextChunk] = []
        for cid in concept_ids:
            for it in self.images.find_for_concept(namespace, cid, max_results=2):
                if self._image_is_usable(it):
                    out.append(ContextChunk(item=it, store="image", score=it.importance, why=f"image for {cid}"))
        if len(out) < k:
            extra = self.images.find_by_query(namespace, query, concept_ids=concept_ids, max_results=k * 2)
            for it in extra:
                if not self._image_is_usable(it):
                    continue
                if it.id not in {c.item.id for c in out}:
                    out.append(ContextChunk(item=it, store="image", score=it.importance, why="image-search"))
        # Prefer certain image types if requested.
        if prefer_types:
            pref_set = {t.lower() for t in prefer_types}
            out.sort(key=lambda ch: (
                0 if any(t.lower() in pref_set for t in ch.item.tags) else 1,
                -ch.score,
            ))
        return out[:k]

    @staticmethod
    def _image_is_usable(item: MemoryItem) -> bool:
        ss = item.store_specific or {}
        if ss.get("_pending_vision"):
            return False
        desc = (item.content or "").strip()
        if not desc or desc in ("(no description)", "(no description yet)"):
            return False
        if any(t.lower() == "decorative" for t in (item.tags or [])):
            return False
        return True

    def _dedupe(self, chunks: list[ContextChunk]) -> list[ContextChunk]:
        seen: dict[str, ContextChunk] = {}
        for ch in chunks:
            existing = seen.get(ch.item.id)
            if existing is None or ch.score > existing.score:
                seen[ch.item.id] = ch
        return sorted(seen.values(), key=lambda c: c.score, reverse=True)


def build_orchestrator(db_path: Path, image_dir: Path) -> ContentOrchestrator:
    """One-line factory: returns an orchestrator with all three stores
    wired to a single SQLite database."""
    concept = ConceptStore(db_path)
    content = ContentStore(db_path)
    images = ImageStore(db_path)
    image_dir.mkdir(parents=True, exist_ok=True)
    return ContentOrchestrator(concept, content, images)
