"""Lesson engine — course outline generation, progress tracking, lesson prompts."""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from database import CourseOutline, FolderSource, SavedNotebook, SessionLocal
import rag


def generate_outline(user_id: int, folder_name: str) -> dict:
    """Generate a structured course outline from all sources in a folder."""
    db = SessionLocal()
    try:
        notebooks = (
            db.query(SavedNotebook)
            .filter(
                SavedNotebook.user_id == user_id,
                SavedNotebook.folder == folder_name,
                SavedNotebook.deleted_at == None,
            )
            .all()
        )
        raw_sources = (
            db.query(FolderSource)
            .filter(
                FolderSource.user_id == user_id,
                FolderSource.folder_name == folder_name,
            )
            .all()
        )

        if not notebooks and not raw_sources:
            return {"error": "No sources in this folder yet."}

        source_summaries = []
        for nb in notebooks:
            data = json.loads(nb.notebook_json)
            title = data.get("title", "Untitled")
            sections = data.get("sections") or []
            sec_info = []
            for s in sections[:12]:
                sec_title = s.get("title", "")
                content_preview = (s.get("content", "") or "")[:200]
                sec_info.append(f"  - {sec_title}: {content_preview}")
            source_summaries.append(f'Source: "{title}"\nSections:\n' + "\n".join(sec_info))

        for src in raw_sources:
            preview = (src.raw_text or "")[:1500]
            source_summaries.append(
                f'Source: "{src.title}" (raw document, {src.page_count} pages)\n'
                f'Content preview:\n{preview}'
            )

        sources_text = "\n\n".join(source_summaries)
        if len(sources_text) > 8000:
            sources_text = sources_text[:8000] + "\n...[truncated]"

        system = (
            "You are a course designer. Given the student's source materials, create a structured "
            "course outline that covers ALL the key topics across all sources in a logical learning sequence.\n\n"
            "Rules:\n"
            "- Create 4-10 sections depending on the amount of material\n"
            "- Order sections so prerequisites come first\n"
            "- Each section should be a coherent learning unit (15-30 minutes)\n"
            "- Reference which source notebooks each section draws from\n"
            "- Include 2-3 specific learning objectives per section\n"
            "- Estimate minutes per section based on content density\n\n"
            "Return ONLY valid JSON — an array of section objects. No markdown fences, no explanation.\n"
            'Format: [{"title": "...", "learning_objectives": ["...", "..."], '
            '"key_topics": ["...", "..."], "source_notebooks": ["..."], "estimated_minutes": 20}]'
        )

        total_sources = len(notebooks) + len(raw_sources)
        context = f"Folder: {folder_name}\nNumber of sources: {total_sources}\n\nSource materials:\n{sources_text}"

        outline_sections = _call_llm_for_outline(system, context)
        if not outline_sections:
            return {"error": "Failed to generate outline. Please try again."}

        total_minutes = sum(s.get("estimated_minutes", 20) for s in outline_sections)

        existing = (
            db.query(CourseOutline)
            .filter(CourseOutline.user_id == user_id, CourseOutline.folder_name == folder_name)
            .first()
        )

        if existing:
            existing.outline_json = json.dumps(outline_sections)
            existing.total_sections = len(outline_sections)
            existing.current_section = 0
            existing.estimated_minutes = total_minutes
            existing.updated_at = datetime.now(timezone.utc)
        else:
            existing = CourseOutline(
                user_id=user_id,
                folder_name=folder_name,
                outline_json=json.dumps(outline_sections),
                total_sections=len(outline_sections),
                current_section=0,
                estimated_minutes=total_minutes,
            )
            db.add(existing)

        db.commit()

        return {
            "sections": outline_sections,
            "total_sections": len(outline_sections),
            "current_section": 0,
            "estimated_minutes": total_minutes,
        }

    except Exception:
        traceback.print_exc()
        return {"error": "Outline generation failed."}
    finally:
        db.close()


def _call_llm_for_outline(system: str, context: str) -> list[dict] | None:
    """Call LLM to generate the outline JSON."""
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model="gemini-3.1-pro-preview",
                contents=context,
                config={
                    "system_instruction": system,
                    "max_output_tokens": 4096,
                    "temperature": 0.4,
                    "thinking_config": {"thinking_budget": 2048},
                },
            )
            text = ""
            if response.candidates and response.candidates[0].content:
                for part in (response.candidates[0].content.parts or []):
                    if hasattr(part, "text") and part.text:
                        text += part.text
            if text.strip():
                return _parse_json_array(text.strip())
        except Exception:
            traceback.print_exc()

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": context},
                ],
                max_tokens=4096,
                temperature=0.4,
            )
            text = resp.choices[0].message.content or ""
            if text.strip():
                return _parse_json_array(text.strip())
        except Exception:
            traceback.print_exc()

    return None


def _parse_json_array(text: str) -> list[dict] | None:
    """Parse JSON array from LLM output, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None


def get_lesson_state(user_id: int, folder_name: str) -> dict:
    """Get current lesson state for a folder."""
    db = SessionLocal()
    try:
        outline = (
            db.query(CourseOutline)
            .filter(CourseOutline.user_id == user_id, CourseOutline.folder_name == folder_name)
            .first()
        )
        if not outline:
            return {"has_outline": False}

        sections = json.loads(outline.outline_json)
        return {
            "has_outline": True,
            "sections": sections,
            "total_sections": outline.total_sections,
            "current_section": outline.current_section,
            "estimated_minutes": outline.estimated_minutes,
            "progress_percent": round((outline.current_section / max(outline.total_sections, 1)) * 100),
            "is_complete": outline.current_section >= outline.total_sections,
        }
    finally:
        db.close()


def advance_section(user_id: int, folder_name: str) -> dict:
    """Advance to the next section."""
    db = SessionLocal()
    try:
        outline = (
            db.query(CourseOutline)
            .filter(CourseOutline.user_id == user_id, CourseOutline.folder_name == folder_name)
            .first()
        )
        if not outline:
            return {"error": "No outline found"}

        if outline.current_section < outline.total_sections:
            outline.current_section += 1
            outline.updated_at = datetime.now(timezone.utc)
            db.commit()

        sections = json.loads(outline.outline_json)
        return {
            "current_section": outline.current_section,
            "total_sections": outline.total_sections,
            "is_complete": outline.current_section >= outline.total_sections,
            "next_section": sections[outline.current_section] if outline.current_section < len(sections) else None,
        }
    finally:
        db.close()


def reset_lesson(user_id: int, folder_name: str) -> dict:
    """Reset lesson progress to start over."""
    db = SessionLocal()
    try:
        outline = (
            db.query(CourseOutline)
            .filter(CourseOutline.user_id == user_id, CourseOutline.folder_name == folder_name)
            .first()
        )
        if not outline:
            return {"error": "No outline found"}

        outline.current_section = 0
        outline.updated_at = datetime.now(timezone.utc)
        db.commit()
        return {"status": "reset", "current_section": 0}
    finally:
        db.close()


def build_lesson_prompt(user_id: int, folder_name: str) -> str | None:
    """Build a specialized system prompt for the current lesson section."""
    db = SessionLocal()
    try:
        outline = (
            db.query(CourseOutline)
            .filter(CourseOutline.user_id == user_id, CourseOutline.folder_name == folder_name)
            .first()
        )
        if not outline:
            return None

        sections = json.loads(outline.outline_json)
        current_idx = outline.current_section
        if current_idx >= len(sections):
            return None

        current = sections[current_idx]
        section_title = current.get("title", f"Section {current_idx + 1}")
        objectives = current.get("learning_objectives", [])
        key_topics = current.get("key_topics", [])
        source_nbs = current.get("source_notebooks", [])

        query_parts = [section_title] + key_topics + objectives
        query = " ".join(query_parts)
        rag_context = rag.build_folder_context(user_id, folder_name, query, max_chars=12000)

        if not rag_context:
            db2 = SessionLocal()
            try:
                from database import FolderSource as FS, SavedNotebook as SN
                raw_sources = db2.query(FS).filter(
                    FS.user_id == user_id, FS.folder_name == folder_name
                ).all()
                notebooks = db2.query(SN).filter(
                    SN.user_id == user_id, SN.folder == folder_name, SN.deleted_at == None
                ).all()

                fallback_parts = []
                for src in raw_sources:
                    text = (src.raw_text or "")[:4000]
                    if text.strip():
                        fallback_parts.append(f'--- From "{src.title}" ---\n{text}')
                for nb in notebooks:
                    try:
                        data = json.loads(nb.notebook_json)
                        secs = data.get("sections") or []
                        nb_text = "\n".join(
                            f"{s.get('title','')}: {(s.get('content','') or '')[:500]}"
                            for s in secs[:10]
                        )
                        if nb_text.strip():
                            fallback_parts.append(f'--- From "{data.get("title", nb.title)}" ---\n{nb_text}')
                    except Exception:
                        pass

                if fallback_parts:
                    rag_context = "\n\n".join(fallback_parts)
                    if len(rag_context) > 12000:
                        rag_context = rag_context[:12000] + "\n...[truncated]"
            finally:
                db2.close()

        completed = [s.get("title", "") for s in sections[:current_idx]]
        upcoming = [s.get("title", "") for s in sections[current_idx + 1:]]

        outline_overview = "\n".join(
            f"  {'[DONE]' if i < current_idx else '[CURRENT]' if i == current_idx else '[   ]'} "
            f"{i+1}. {s.get('title', '')}"
            for i, s in enumerate(sections)
        )

        prompt = (
            f"\n--- LESSON MODE ---\n"
            f"You are teaching a structured course. The student is on section {current_idx + 1} of {len(sections)}.\n\n"
            f"COURSE OUTLINE:\n{outline_overview}\n\n"
            f"CURRENT SECTION: {section_title}\n"
            f"Learning objectives: {', '.join(objectives)}\n"
            f"Key topics to cover: {', '.join(key_topics)}\n"
            f"Source materials: {', '.join(source_nbs)}\n\n"
        )

        if completed:
            prompt += f"Already covered: {', '.join(completed)}\n"
        if upcoming:
            prompt += f"Coming next: {', '.join(upcoming[:3])}\n"

        prompt += (
            "\nINSTRUCTIONS:\n"
            "1. You MUST teach ONLY from the source material provided below. Do NOT say you don't have the material — it is included below.\n"
            "2. Use specific facts, definitions, formulas, and examples from the source material. Quote or paraphrase directly.\n"
            "3. Reference which source document the information comes from by name.\n"
            "4. After explaining the key concepts, ask 1-2 comprehension questions to check understanding.\n"
            "5. When the student answers correctly, congratulate them and ask if they're ready for the next section.\n"
            "6. If the student struggles, provide hints and re-explain using different approaches from the source material.\n"
            "7. Keep your teaching concise but thorough — aim for the depth of a great tutor, not a textbook.\n"
            "8. NEVER claim you don't have access to the student's uploaded documents. The relevant content is provided below.\n"
            "9. When you believe the student has understood the section, include the exact phrase "
            '"[SECTION_COMPLETE]" at the end of your message (the frontend uses this to show a Next Section button).\n\n'
        )

        if rag_context:
            prompt += (
                "--- SOURCE MATERIAL FOR THIS SECTION (from the student's uploaded documents) ---\n"
                + rag_context
                + "\n--- END SOURCE MATERIAL ---\n"
            )
        else:
            prompt += (
                "NOTE: No source material was retrieved for this section. Teach using your general knowledge "
                "of the topic, but let the student know you're drawing from general knowledge rather than "
                "their specific uploaded materials.\n"
            )

        return prompt
    except Exception:
        traceback.print_exc()
        return None
    finally:
        db.close()
