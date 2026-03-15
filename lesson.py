"""Lesson engine — course outline generation, progress tracking, lesson prompts."""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from database import CourseOutline, FolderSource, SavedNotebook, SessionLocal, SourceImage
import rag


def generate_outline(user_id: int, folder_name: str, source_user_id: int | None = None, structure: dict | None = None) -> dict:
    """Generate a structured course outline from all sources in a folder.
    
    source_user_id: if set, read sources from this user (for curated/shared folders).
    structure: optional dict with custom structure hints for the outline.
    Outline is always saved under user_id (per-user progress).
    """
    src_uid = source_user_id if source_user_id is not None else user_id
    db = SessionLocal()
    try:
        notebooks = (
            db.query(SavedNotebook)
            .filter(
                SavedNotebook.user_id == src_uid,
                SavedNotebook.folder == folder_name,
                SavedNotebook.deleted_at == None,
            )
            .all()
        )
        raw_sources = (
            db.query(FolderSource)
            .filter(
                FolderSource.user_id == src_uid,
                FolderSource.folder_name == folder_name,
            )
            .all()
        )

        if not notebooks and not raw_sources:
            return {"error": "No sources in this folder yet."}

        total_sources = len(notebooks) + len(raw_sources)
        total_budget = 80_000
        per_source_budget = max(400, total_budget // max(total_sources, 1))

        source_summaries = []
        for nb in notebooks:
            data = json.loads(nb.notebook_json)
            title = data.get("title", "Untitled")
            sections = data.get("sections") or []
            sec_info = []
            for s in sections[:16]:
                sec_title = s.get("title", "")
                content_preview = (s.get("content", "") or "")[:min(300, per_source_budget // 8)]
                sec_info.append(f"  - {sec_title}: {content_preview}")
            source_summaries.append(f'Source: "{title}"\nSections:\n' + "\n".join(sec_info))

        for src in raw_sources:
            text = src.raw_text or ""
            if len(text) <= per_source_budget:
                preview = text
            else:
                chunk = per_source_budget // 3
                mid = len(text) // 2
                preview = (
                    text[:chunk]
                    + "\n[...]\n"
                    + text[mid - chunk // 2 : mid + chunk // 2]
                    + "\n[...]\n"
                    + text[-chunk:]
                )
            source_summaries.append(
                f'Source: "{src.title}" (raw document, {src.page_count} pages)\n'
                f'Content:\n{preview}'
            )

        sources_text = "\n\n".join(source_summaries)
        if len(sources_text) > total_budget:
            sources_text = sources_text[:total_budget] + "\n...[truncated]"

        max_sections = min(25, max(4, total_sources // 2 + 3))

        structure_block = ""
        if structure:
            parts_desc = []
            for i, part in enumerate(structure.get("parts", []), 1):
                patterns = ", ".join(part.get("source_patterns", []))
                parts_desc.append(
                    f"  Part {i}: \"{part['name']}\" — {part['description']}"
                    + (f" (sources matching: {patterns})" if patterns else "")
                )
            pedagogy = structure.get("pedagogy", "")
            pedagogy_block = f"\n\nTEACHING METHODOLOGY:\n{pedagogy}\n" if pedagogy else ""
            structure_block = (
                f"\n\nCOURSE STRUCTURE (you MUST follow this):\n"
                f"{structure.get('description', '')}\n"
                + "\n".join(parts_desc)
                + "\n\nOrganize sections within each part in logical teaching order. "
                "Use the part name as a prefix in section titles, e.g. "
                "\"Statistics: Probability Theory\", \"Mathematics: Linear Equations\", "
                "\"Computer Skills: Excel Basics\".\n"
                + pedagogy_block
            )

        system = (
            "You are a course designer. Given the student's source materials, create a structured "
            "course outline that covers ALL the key topics across ALL sources in a logical learning sequence.\n\n"
            "CRITICAL: You MUST include content from EVERY source listed. Do NOT skip any sources — "
            "even those listed last. The student uploaded all of them and expects the course to cover "
            "all their material.\n"
            + structure_block +
            "\n\nRules:\n"
            f"- Create 4-{max_sections} sections depending on the amount of material\n"
            "- Order sections so prerequisites come first within each part/group\n"
            "- Each section should be a coherent learning unit (15-30 minutes)\n"
            "- Reference which source notebooks/documents each section draws from\n"
            "- Include 2-3 specific learning objectives per section\n"
            "- Estimate minutes per section based on content density\n"
            "- If sources cover similar topics, merge them into one section\n"
            "- If a source covers multiple distinct topics, split across sections\n\n"
            "Return ONLY valid JSON — an array of section objects. No markdown fences, no explanation.\n"
            'Format: [{"title": "...", "learning_objectives": ["...", "..."], '
            '"key_topics": ["...", "..."], "source_notebooks": ["..."], "estimated_minutes": 20}]'
        )

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
                    "max_output_tokens": 8192,
                    "temperature": 0.4,
                    "thinking_config": {"thinking_budget": 4096},
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


def _find_relevant_images(
    user_id: int,
    folder_name: str,
    section_title: str,
    key_topics: list[str],
    source_notebooks: list[str],
    max_images: int = 6,
) -> str:
    """Find images from folder sources that are relevant to the current section."""
    db = SessionLocal()
    try:
        all_images = db.query(SourceImage).filter(
            SourceImage.user_id == user_id,
            SourceImage.folder_name == folder_name,
        ).all()

        if not all_images:
            return ""

        search_terms = [t.lower() for t in [section_title] + key_topics if t]

        def _score(si: SourceImage) -> int:
            ctx = (si.context_text or "").lower()
            score = 0
            for term in search_terms:
                words = term.split()
                for word in words:
                    if len(word) > 3 and word in ctx:
                        score += 5
            for nb_name in source_notebooks:
                if nb_name.lower() in (si.source_id or "").lower():
                    score += 20

            from database import FolderSource as FS
            src = db.query(FS).filter(FS.source_id == si.source_id).first()
            if src:
                title_lower = (src.title or "").lower()
                for nb_name in source_notebooks:
                    if nb_name.lower() in title_lower or title_lower in nb_name.lower():
                        score += 30
                for term in search_terms:
                    if term in title_lower:
                        score += 10
            return score

        scored = [(si, _score(si)) for si in all_images]
        scored = [(si, sc) for si, sc in scored if sc > 0]
        scored.sort(key=lambda x: -x[1])

        top = scored[:max_images]
        if not top:
            return ""

        api_base = os.getenv("API_BASE_URL", "https://coast-backend-dlg6.onrender.com")
        lines = []
        for si, _ in top:
            src = db.query(FolderSource).filter(FolderSource.source_id == si.source_id).first()
            src_title = src.title if src else si.source_id
            url = f"{api_base}/api/source-images/{si.id}"
            ctx_preview = (si.context_text or "")[:120].replace("\n", " ")
            lines.append(
                f"- Image {si.id}: from \"{src_title}\" page {si.page_number} | "
                f"context: \"{ctx_preview}\" | URL: {url}"
            )

        return (
            "\n--- DIAGRAMS FROM SOURCE MATERIALS ---\n"
            "These diagrams were extracted from the student's uploaded documents. You SHOULD embed them "
            "in your explanation whenever they illustrate a concept you're currently teaching — for example, "
            "a distribution curve when explaining distributions, a scatter plot when discussing correlation, "
            "a function graph when teaching derivatives. Use markdown: ![brief description](URL)\n"
            "Embed them naturally within your explanation at the point where they're most helpful. "
            "Don't dump them all at once or list them separately — weave them into the teaching.\n\n"
            + "\n".join(lines)
            + "\n--- END DIAGRAMS ---\n"
        )
    except Exception:
        traceback.print_exc()
        return ""
    finally:
        db.close()


def _fallback_source_context(
    user_id: int,
    folder_name: str,
    section_title: str,
    key_topics: list[str],
    source_notebooks: list[str],
    max_chars: int = 24000,
) -> str:
    """When RAG is unavailable, build section context by matching sources to section topics."""
    from database import FolderSource as FS

    db = SessionLocal()
    try:
        all_sources = db.query(FS).filter(
            FS.user_id == user_id, FS.folder_name == folder_name
        ).all()
        if not all_sources:
            return ""

        search_terms = [t.lower() for t in [section_title] + key_topics + source_notebooks if t]

        def _relevance_score(src) -> int:
            title_lower = (src.title or "").lower()
            score = 0
            for term in search_terms:
                if term.lower() in title_lower:
                    score += 10
            for nb_name in source_notebooks:
                if nb_name.lower() in title_lower or title_lower in nb_name.lower():
                    score += 50
            return score

        scored = [(s, _relevance_score(s)) for s in all_sources]
        scored.sort(key=lambda x: -x[1])

        matched = [s for s, sc in scored if sc > 0]
        unmatched = [s for s, sc in scored if sc == 0]

        if not matched:
            matched = all_sources
            unmatched = []

        budget_matched = int(max_chars * 0.85)
        budget_other = max_chars - budget_matched

        parts = []
        used = 0

        if matched:
            per_source = max(2000, budget_matched // len(matched))
            for src in matched:
                raw = (src.raw_text or "").strip()
                if not raw:
                    continue
                if len(raw) <= per_source:
                    snippet = raw
                else:
                    third = per_source // 3
                    snippet = raw[:third] + "\n\n[...middle of document...]\n\n" + raw[len(raw)//2 - third//2 : len(raw)//2 + third//2] + "\n\n[...end of document...]\n\n" + raw[-third:]
                parts.append(f'--- From "{src.title}" ---\n{snippet}')
                used += len(parts[-1])
                if used >= budget_matched:
                    break

        if unmatched and used < max_chars:
            remaining = max_chars - used
            per_source = max(500, min(remaining // len(unmatched), 1500))
            for src in unmatched:
                raw = (src.raw_text or "").strip()
                if not raw:
                    continue
                parts.append(f'--- From "{src.title}" ---\n{raw[:per_source]}')
                used += len(parts[-1])
                if used >= max_chars:
                    break

        return "\n\n".join(parts) if parts else ""
    finally:
        db.close()


def build_lesson_prompt(user_id: int, folder_name: str, source_user_id: int | None = None, structure: dict | None = None) -> str | None:
    """Build a specialized system prompt for the current lesson section.
    
    source_user_id: if set, read sources/RAG from this user (for curated/shared folders).
    structure: optional dict with pedagogy hints for curated lessons.
    """
    src_uid = source_user_id if source_user_id is not None else user_id
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
        rag_context = rag.build_folder_context(src_uid, folder_name, query, max_chars=24000)

        if not rag_context:
            rag_context = _fallback_source_context(
                src_uid, folder_name, section_title, key_topics, source_nbs, max_chars=24000
            )

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
            "\nTEACHING APPROACH:\n"
            "Assume the student is a COMPLETE BEGINNER with ZERO prior knowledge of this topic. "
            "They have NOT read the lecture notes or slides beforehand. Your job is to teach them "
            "everything from scratch so they fully understand every concept and can solve any exercise "
            "from the source material by the end of this section.\n\n"

            "INSTRUCTIONS:\n"
            "1. You MUST teach from the source material provided below. Do NOT say you don't have it — it is included below.\n"
            "2. START with the basics: define every term, explain every concept from the ground up. "
            "Never assume the student already knows something — if a concept is needed, explain it first.\n"
            "3. Build knowledge step by step: foundations first, then build up to more complex ideas. "
            "Use clear, direct explanations with concrete examples. Only use analogies when a concept is truly abstract "
            "and hard to grasp without one — most of the time, a detailed explanation with a worked example is more helpful.\n"
            "4. When the source material contains formulas or equations, explain WHAT each variable means, "
            "WHY the formula works, and walk through a concrete numerical example.\n"
            "4b. Use markdown TABLES when comparing concepts, listing properties, showing data side-by-side, "
            "or organizing formulas. Tables render beautifully in the chat and help students see structure.\n"
            "4c. DIAGRAMS: If relevant diagrams from the source materials are listed in the DIAGRAMS section below, "
            "you MUST proactively embed them in your explanation when they help illustrate a concept — "
            "for example, showing a distribution curve when teaching about distributions, a graph when explaining "
            "functions, or a chart when discussing data patterns. Use the markdown syntax ![description](URL). "
            "Don't wait for the student to ask — if a diagram makes the concept clearer, include it naturally "
            "in your explanation. Only skip a diagram if it truly doesn't add value to what you're currently teaching.\n"
            "5. When the source material contains exercises or problems, work through them WITH the student: "
            "explain the approach, show each step, and make sure they could solve a similar problem on their own.\n"
            "6. After teaching the key concepts, ask 1-2 quick comprehension questions to check understanding. "
            "These should be answerable PURELY from what you just taught.\n"
            "7. If the student struggles, don't just repeat yourself. Re-explain using a different angle, "
            "simpler language, a concrete example, or break it into even smaller steps.\n"
            "8. BEFORE completing the section, you MUST do a PRACTICE ROUND:\n"
            "   a) FIRST: Carefully search through ALL the source material below for exercises, problems, worked examples, "
            "practice questions, exam questions, or tutorial sheets that relate to the topics covered in this section. "
            "These are extremely valuable because they match the student's actual course.\n"
            "   b) If exercises exist in the source material, present them to the student one at a time — use the EXACT "
            "wording from the source so the student practices with real course material. Guide them if they get stuck, "
            "but let them try first.\n"
            "   c) If the source material has a textbook or exercise book (look for sources with 'exercise', 'tutorial', "
            "'problem set', or 'worksheet' in the title), prioritize questions from there.\n"
            "   d) If NO exercises exist in the source material for these topics, create 2-3 practice problems yourself that "
            "test the key concepts. Make them progressively harder.\n"
            "   e) If the student gets a problem wrong, explain what went wrong, identify the concept they're weak on, "
            "and tell them clearly: 'You should review [specific topic] more — this is an area to focus on.'\n"
            "   f) Only after the student has attempted the practice problems should you consider the section complete.\n"
            "9. When the student has completed the practice round successfully, congratulate them, summarize what they learned, "
            "note any weak areas they should revisit, and ask if they're ready for the next section.\n"
            "10. NEVER claim you don't have access to the student's uploaded documents. The relevant content is provided below.\n"
            "11. When you believe the student has understood the section AND completed the practice round, include the exact phrase "
            '"[SECTION_COMPLETE]" at the end of your message (the frontend uses this to show a Next Section button).\n\n'
        )

        if structure and structure.get("pedagogy"):
            prompt += (
                "--- SPECIAL TEACHING METHODOLOGY FOR THIS COURSE ---\n"
                + structure["pedagogy"]
                + "\n--- END METHODOLOGY ---\n\n"
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

        image_block = _find_relevant_images(src_uid, folder_name, section_title, key_topics, source_nbs)
        if image_block:
            prompt += image_block

        return prompt
    except Exception:
        traceback.print_exc()
        return None
    finally:
        db.close()
