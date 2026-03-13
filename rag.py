"""RAG engine — chunking, embedding, ChromaDB storage, and semantic search."""

from __future__ import annotations

import json
import os
import re
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from database import SavedNotebook, SessionLocal

CHROMA_PATH = Path(os.environ.get("CHROMA_PATH", str(Path(__file__).parent / "chroma_data")))
CHROMA_PATH.mkdir(parents=True, exist_ok=True)

_chroma_client = None

def _get_chroma():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return _chroma_client

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_TARGET_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150


def _collection_name(user_id: int, folder: str) -> str:
    slug = re.sub(r"[^a-z0-9]", "_", folder.lower().strip())[:40]
    name = f"u{user_id}_{slug}"
    if len(name) < 3:
        name = name + "___"
    return name[:63]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_notebook(notebook_json: dict) -> list[dict]:
    """Split notebook into chunks following section boundaries."""
    title = notebook_json.get("title", "Untitled")
    sections = notebook_json.get("sections") or []
    chunks = []

    for sec in sections:
        sec_title = sec.get("title", "")
        parts = [sec.get("content", "") or ""]

        for sub in (sec.get("subsections") or []):
            sub_text = f"{sub.get('title', '')}: {sub.get('content', '') or ''}"
            for b in (sub.get("bullets") or []):
                sub_text += f"\n  - {b}"
            parts.append(sub_text)

        section_text = "\n".join(parts).strip()
        if not section_text:
            continue

        if len(section_text) <= CHUNK_TARGET_CHARS:
            chunks.append({
                "text": section_text,
                "notebook_title": title,
                "section_title": sec_title,
            })
        else:
            sentences = re.split(r"(?<=[.!?\n])\s+", section_text)
            current = ""
            for sent in sentences:
                if len(current) + len(sent) > CHUNK_TARGET_CHARS and current:
                    chunks.append({
                        "text": current.strip(),
                        "notebook_title": title,
                        "section_title": sec_title,
                    })
                    overlap_start = max(0, len(current) - CHUNK_OVERLAP_CHARS)
                    current = current[overlap_start:] + " " + sent
                else:
                    current = current + " " + sent if current else sent
            if current.strip():
                chunks.append({
                    "text": current.strip(),
                    "notebook_title": title,
                    "section_title": sec_title,
                })

    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _get_embeddings(texts: list[str]) -> list[list[float]]:
    """Compute embeddings using OpenAI text-embedding-3-small."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def embed_notebook(user_id: int, folder: str, notebook_id: str, notebook_json: dict) -> int:
    """Chunk a notebook, compute embeddings, store in ChromaDB. Returns chunk count."""
    col_name = _collection_name(user_id, folder)

    delete_notebook_embeddings(user_id, folder, notebook_id)

    chunks = chunk_notebook(notebook_json)
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    try:
        embeddings = _get_embeddings(texts)
    except Exception:
        traceback.print_exc()
        return 0

    collection = _get_chroma().get_or_create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"{notebook_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "notebook_id": notebook_id,
            "notebook_title": c["notebook_title"],
            "section_title": c["section_title"],
        }
        for c in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    return len(chunks)


def delete_notebook_embeddings(user_id: int, folder: str, notebook_id: str):
    """Remove all embeddings for a specific notebook from the folder collection."""
    col_name = _collection_name(user_id, folder)
    try:
        collection = _get_chroma().get_collection(name=col_name)
        existing = collection.get(where={"notebook_id": notebook_id})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_folder(user_id: int, folder: str, query: str, top_k: int = 8) -> list[dict]:
    """Semantic search across all notebooks in a folder."""
    col_name = _collection_name(user_id, folder)
    try:
        collection = _get_chroma().get_collection(name=col_name)
    except Exception:
        return []

    try:
        query_embedding = _get_embeddings([query])[0]
    except Exception:
        traceback.print_exc()
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count() or 1),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "text": results["documents"][0][i],
            "notebook_id": results["metadatas"][0][i].get("notebook_id", ""),
            "notebook_title": results["metadatas"][0][i].get("notebook_title", ""),
            "section_title": results["metadatas"][0][i].get("section_title", ""),
            "distance": results["distances"][0][i],
        })

    return hits


def build_folder_context(user_id: int, folder: str, query: str, max_chars: int = 10000) -> str:
    """Search folder and format results as context for Pedro's system prompt."""
    hits = search_folder(user_id, folder, query, top_k=10)
    if not hits:
        return ""

    parts = []
    total = 0
    for h in hits:
        block = (
            f'From "{h["notebook_title"]}" > "{h["section_title"]}":\n'
            f'{h["text"]}\n'
        )
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)

    return "\n---\n".join(parts)


# ---------------------------------------------------------------------------
# Folder management helpers
# ---------------------------------------------------------------------------

def embed_all_in_folder(user_id: int, folder: str) -> dict:
    """Embed all notebooks in a folder. Returns summary."""
    db = SessionLocal()
    try:
        notebooks = (
            db.query(SavedNotebook)
            .filter(
                SavedNotebook.user_id == user_id,
                SavedNotebook.folder == folder,
                SavedNotebook.deleted_at == None,
            )
            .all()
        )
        total_chunks = 0
        embedded_count = 0
        for nb in notebooks:
            try:
                data = json.loads(nb.notebook_json)
                count = embed_notebook(user_id, folder, nb.notebook_id, data)
                total_chunks += count
                if count > 0:
                    embedded_count += 1
            except Exception:
                traceback.print_exc()

        return {
            "notebooks_embedded": embedded_count,
            "total_chunks": total_chunks,
            "notebooks_in_folder": len(notebooks),
        }
    finally:
        db.close()


def get_folder_sources(user_id: int, folder: str) -> list[dict]:
    """Get metadata about embedded notebooks in a folder."""
    col_name = _collection_name(user_id, folder)
    try:
        collection = _get_chroma().get_collection(name=col_name)
        all_data = collection.get(include=["metadatas"])
    except Exception:
        return []

    nb_map: dict[str, dict] = {}
    for meta in all_data["metadatas"]:
        nb_id = meta.get("notebook_id", "")
        if nb_id not in nb_map:
            nb_map[nb_id] = {
                "notebook_id": nb_id,
                "notebook_title": meta.get("notebook_title", ""),
                "chunk_count": 0,
                "sections": set(),
            }
        nb_map[nb_id]["chunk_count"] += 1
        nb_map[nb_id]["sections"].add(meta.get("section_title", ""))

    return [
        {
            "notebook_id": v["notebook_id"],
            "notebook_title": v["notebook_title"],
            "chunk_count": v["chunk_count"],
            "section_count": len(v["sections"]),
        }
        for v in nb_map.values()
    ]


def generate_study_plan(user_id: int, folder: str, user_name: str) -> str:
    """Generate a study plan from all sources in a folder using LLM."""
    db = SessionLocal()
    try:
        notebooks = (
            db.query(SavedNotebook)
            .filter(
                SavedNotebook.user_id == user_id,
                SavedNotebook.folder == folder,
                SavedNotebook.deleted_at == None,
            )
            .all()
        )
        if not notebooks:
            return "No sources in this folder yet. Add some notebooks to generate a study plan!"

        overview_parts = []
        for nb in notebooks:
            data = json.loads(nb.notebook_json)
            title = data.get("title", "Untitled")
            sections = data.get("sections") or []
            sec_titles = [s.get("title", "") for s in sections]
            overview_parts.append(f'"{title}": covers {", ".join(sec_titles[:8])}')

        overview = "\n".join(overview_parts)
        if len(overview) > 6000:
            overview = overview[:6000] + "\n...[truncated]"

        system = (
            "You are Pedro, a study planning expert. Create a clear, actionable study plan "
            "based on the student's available sources. The plan should:\n"
            "1. Suggest a logical sequence for studying the topics\n"
            "2. Group related concepts across different sources\n"
            "3. Highlight dependencies (what to learn before what)\n"
            "4. Suggest time estimates for each study block\n"
            "5. Reference specific notebooks/sources by name\n\n"
            "Keep it concise and practical. Use markdown formatting."
        )

        context = (
            f"Student: {user_name}\n"
            f"Folder: {folder}\n"
            f"Number of sources: {len(notebooks)}\n\n"
            f"Sources and their topics:\n{overview}"
        )

        from google import genai
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if gemini_key:
            client = genai.Client(api_key=gemini_key)
            try:
                response = client.models.generate_content(
                    model="gemini-3.1-pro-preview",
                    contents=context,
                    config={
                        "system_instruction": system,
                        "max_output_tokens": 1500,
                        "temperature": 0.7,
                        "thinking_config": {"thinking_budget": 1024},
                    },
                )
                text = ""
                if response.candidates and response.candidates[0].content:
                    for part in (response.candidates[0].content.parts or []):
                        if hasattr(part, "text") and part.text:
                            text += part.text
                if text.strip():
                    return text.strip()
            except Exception:
                traceback.print_exc()

        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key:
            client = OpenAI(api_key=openai_key)
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": context},
            ]
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                messages=messages,
                max_tokens=1500,
                temperature=0.7,
            )
            text = resp.choices[0].message.content or ""
            if text.strip():
                return text.strip()

        return f"Study plan couldn't be generated right now. You have {len(notebooks)} sources in this folder."

    except Exception:
        traceback.print_exc()
        return "Study plan generation failed. Please try again."
    finally:
        db.close()
