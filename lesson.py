"""Lesson engine — course outline generation, progress tracking, lesson prompts."""

from __future__ import annotations

import json
import os
import re
import traceback
from collections import defaultdict
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from database import (
    ChatMessage,
    CourseOutline,
    FolderSource,
    SavedNotebook,
    SectionRewardClaim,
    SectionVerification,
    SessionLocal,
    SourceImage,
)


_RECAP_REQUEST = re.compile(
    r"\b(summary|summarize|summarise|recap|overview|catch me up|"
    r"what we (?:have |'ve )?(?:done|covered|learned|studied)|"
    r"what did we (?:do|cover|learn)|review (?:what|our)|"
    r"everything we(?:'ve| have) (?:done|covered))\b",
    re.I,
)


def is_recap_request(message: str | None) -> bool:
    return bool(message and _RECAP_REQUEST.search(message))


def _fetch_lesson_conversation_recap(
    user_id: int,
    folder_name: str,
    max_chars: int = 9000,
) -> str:
    """Recent Pedro lesson chat — what was actually taught in sessions."""
    db = SessionLocal()
    try:
        rows = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.user_id == user_id,
                ChatMessage.context_type.in_(("lesson", "folder")),
                ChatMessage.context_id == folder_name,
            )
            .order_by(ChatMessage.created_at.desc())
            .limit(60)
            .all()
        )
        if not rows:
            return ""
        rows = list(reversed(rows))
        lines: list[str] = []
        used = 0
        for msg in rows:
            role = "Student" if msg.role == "user" else "Pedro"
            sec = msg.section_index
            prefix = f"[Section {sec + 1}] " if sec is not None else ""
            text = (msg.content or "").strip()
            if not text or _SECTION_OPENER.match(text):
                continue
            line = f"{prefix}{role}: {text[:600]}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)
        if not lines:
            return ""
        return (
            "--- RECENT LESSON CONVERSATIONS (what Pedro already taught) ---\n"
            + "\n".join(lines)
            + "\n--- END RECENT LESSON CONVERSATIONS ---\n"
        )
    finally:
        db.close()


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
        total_pages = sum(getattr(s, "page_count", 0) or 0 for s in raw_sources)
        total_budget = 80_000

        # Prefer Content OMA structured index when available.
        sources_text = ""
        outline_via_oma = False
        oma_required = False
        try:
            import oma_provider
            source_meta = [
                {
                    "source_id": s.source_id,
                    "title": s.title,
                    "page_count": s.page_count,
                    "filename": s.filename or f"{s.source_id}.pdf",
                }
                for s in raw_sources
            ]
            pdf_sources = oma_provider.load_folder_pdf_sources(src_uid, folder_name)
            oma_required = oma_provider.is_oma_enabled() and bool(pdf_sources)
            if oma_required:
                ready = oma_provider.ensure_oma_ready_for_outline(
                    src_uid, folder_name,
                    expected_pages=total_pages,
                    pdf_sources=pdf_sources,
                )
                if not ready:
                    return {
                        "error": (
                            "Content OMA is still indexing your uploads. "
                            "Please wait and try Generate Lesson again."
                        ),
                    }
            oma_index = oma_provider.build_outline_context(
                src_uid, folder_name,
                source_meta=source_meta,
                max_chars=total_budget,
            )
            if oma_index:
                sources_text = oma_index
                outline_via_oma = True
            elif oma_required:
                return {
                    "error": (
                        "Content OMA could not build a course index from your uploads. "
                        "Try re-uploading or wait a bit longer."
                    ),
                }
        except Exception:
            traceback.print_exc()
            if oma_required:
                return {"error": "Content OMA failed while preparing the lesson outline."}

        if not sources_text:
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

        if outline_via_oma and total_pages:
            max_sections = min(40, max(6, total_pages // 8))
        else:
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

        oma_rules = ""
        if outline_via_oma:
            oma_rules = (
                "\n\nThe source materials below are a Content OMA structured index: every page "
                "with section titles, content types (definition, example, exercise, etc.), "
                "and a concept map with prerequisites. Use this to:\n"
                "- Create one section per major topic within each source; split any source "
                "with 10+ pages into 2–3 sections aligned to its page groupings\n"
                "- Order sections so prerequisites in the concept map come first\n"
                "- Set key_topics from the concepts listed for each page group\n"
                "- Put exact source titles (in quotes in the index) in source_notebooks\n"
            )

        system = (
            "You are a course designer. Given the student's source materials, create a structured "
            "course outline that covers ALL the key topics across ALL sources in a logical learning sequence.\n\n"
            "CRITICAL: You MUST include content from EVERY source listed. Do NOT skip any sources — "
            "even those listed last. The student uploaded all of them and expects the course to cover "
            "all their material.\n"
            + oma_rules
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

        material_label = "Content OMA course index" if outline_via_oma else "Source materials"
        context = (
            f"Folder: {folder_name}\nNumber of sources: {total_sources}\n"
            f"Total pages: {total_pages or 'unknown'}\n\n{material_label}:\n{sources_text}"
        )

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
            # ever_mastered + lesson notes are intentionally preserved on regenerate.
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

        oma_pages = 0
        try:
            import oma_provider
            if oma_provider.is_oma_enabled():
                oma_pages = oma_provider.oma_content_page_count(src_uid, folder_name)
        except Exception:
            pass

        return {
            "sections": outline_sections,
            "total_sections": len(outline_sections),
            "current_section": 0,
            "estimated_minutes": total_minutes,
            "outline_source": "oma" if outline_via_oma else "raw",
            "oma_pages_indexed": oma_pages,
            "outline_note": (
                None if outline_via_oma
                else "OMA index not ready — used raw text. Try regenerating in a minute."
            ),
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


# Auto-generated lesson openers — not evidence the student started working.
_SECTION_OPENER = re.compile(
    r"^(i'?m ready to learn about|i'?d like to reach 100% mastery)\b",
    re.I,
)

# Student saying "next" does not pass verification — only real answers do.
_SKIP_OR_ADVANCE = re.compile(
    r"^(next|skip|continue|move on|go on|done|finished|let'?s move|proceed)[\s!.?]*$",
    re.I,
)
_READY_ACK = re.compile(
    r"^(yes|yeah|yep|yup|ready|ok|okay|sure|let'?s go|i'?m ready|next section)[\s!.?]*$",
    re.I,
)


def _verification_row(db, user_id: int, folder_name: str, section_index: int) -> SectionVerification:
    row = (
        db.query(SectionVerification)
        .filter(
            SectionVerification.user_id == user_id,
            SectionVerification.folder_name == folder_name,
            SectionVerification.section_index == section_index,
        )
        .first()
    )
    if not row:
        row = SectionVerification(
            user_id=user_id,
            folder_name=folder_name,
            section_index=section_index,
            is_active=False,
        )
        db.add(row)
    return row


def is_section_verified(user_id: int, folder_name: str, section_index: int) -> bool:
    db = SessionLocal()
    try:
        row = (
            db.query(SectionVerification)
            .filter(
                SectionVerification.user_id == user_id,
                SectionVerification.folder_name == folder_name,
                SectionVerification.section_index == section_index,
                SectionVerification.is_active.is_(True),
            )
            .first()
        )
        return row is not None
    finally:
        db.close()


def _section_reward_claimed(user_id: int, folder_name: str, section_index: int) -> bool:
    db = SessionLocal()
    try:
        row = (
            db.query(SectionRewardClaim)
            .filter(
                SectionRewardClaim.user_id == user_id,
                SectionRewardClaim.folder_name == folder_name,
                SectionRewardClaim.section_index == section_index,
            )
            .first()
        )
        return row is not None
    finally:
        db.close()


def _pedro_marked_section_complete(user_id: int, folder_name: str, section_index: int) -> bool:
    db = SessionLocal()
    try:
        row = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.user_id == user_id,
                ChatMessage.context_type == "lesson",
                ChatMessage.context_id == folder_name,
                ChatMessage.section_index == section_index,
                ChatMessage.role == "pedro",
                ChatMessage.content.contains("[SECTION_COMPLETE]"),
            )
            .first()
        )
        return row is not None
    finally:
        db.close()


def can_advance_from_section(user_id: int, folder_name: str, section_index: int) -> bool:
    """Section is advanceable once Pedro verified, reward was claimed, or [SECTION_COMPLETE] was emitted."""
    if is_section_verified(user_id, folder_name, section_index):
        return True
    if _section_reward_claimed(user_id, folder_name, section_index):
        return True
    return _pedro_marked_section_complete(user_id, folder_name, section_index)


def mark_section_verified(user_id: int, folder_name: str, section_index: int) -> None:
    db = SessionLocal()
    try:
        row = _verification_row(db, user_id, folder_name, section_index)
        row.is_active = True
        row.verified_at = datetime.now(timezone.utc)
        row.updated_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()


def invalidate_section_verification(user_id: int, folder_name: str, section_index: int) -> None:
    db = SessionLocal()
    try:
        row = (
            db.query(SectionVerification)
            .filter(
                SectionVerification.user_id == user_id,
                SectionVerification.folder_name == folder_name,
                SectionVerification.section_index == section_index,
            )
            .first()
        )
        if row and row.is_active:
            row.is_active = False
            row.updated_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()


def clear_section_verification(user_id: int, folder_name: str, section_index: int) -> None:
    db = SessionLocal()
    try:
        row = (
            db.query(SectionVerification)
            .filter(
                SectionVerification.user_id == user_id,
                SectionVerification.folder_name == folder_name,
                SectionVerification.section_index == section_index,
            )
            .first()
        )
        if row:
            db.delete(row)
            db.commit()
    finally:
        db.close()


def should_invalidate_verification(message: str) -> bool:
    """No longer revoke verification on follow-up questions — only advance clears it."""
    return False


def handle_pre_lesson_message(
    user_id: int,
    folder_name: str,
    section_index: int | None,
    message: str,
) -> None:
    """Previously revoked verification on follow-ups; verification now persists until advance."""
    return


def get_verification_prompt_block(
    user_id: int,
    folder_name: str,
    section_index: int,
    message: str | None = None,
) -> str:
    verified = is_section_verified(user_id, folder_name, section_index)
    block = (
        "\n--- SECTION VERIFICATION GATE (mandatory) ---\n"
        "The student CANNOT advance to the next section until YOU verify readiness.\n"
        "- You MUST run a practice/verification round (step 8) before considering the section done.\n"
        "- Present verification questions ONE AT A TIME. Grade each answer with [ANSWER_WRONG] or "
        "[ANSWER_CORRECT] at the end of your message.\n"
        "- Do NOT emit [SECTION_COMPLETE] until EVERY required verification question has been "
        "answered correctly. Partial progress is not enough.\n"
        "- If the student says 'next', 'skip', 'continue', or similar WITHOUT answering, explain "
        "they must complete the remaining verification questions first. Do NOT emit [SECTION_COMPLETE].\n"
        "- If the student asks a side question during verification: answer it fully, then return "
        "to any verification questions they have NOT yet answered correctly. Never drop pending "
        "verification questions — track them mentally and re-ask until passed.\n"
        "- [SECTION_COMPLETE] is the ONLY signal that unlocks the Next Section button. The map "
        "reflects concepts you verified with [ANSWER_CORRECT] — never skip verification.\n"
    )
    if verified:
        block += (
            "STATUS: Section is currently VERIFIED — student may use Next Section. "
            "Answer any follow-up questions fully; verification stays active until they advance.\n"
        )
    else:
        block += "STATUS: Section is NOT verified — continue teaching and verification until all pass.\n"
    if message and _SKIP_OR_ADVANCE.match(message.strip()):
        block += (
            "The student just tried to skip without answering. Remind them verification is "
            "required and continue with the next unanswered verification question.\n"
        )
    block += "--- END VERIFICATION GATE ---\n"
    return block


def _sections_started(user_id: int, folder_name: str, current_section: int) -> set[int]:
    """Sections the student actually engaged with (not just auto-openers)."""
    started: set[int] = set()
    db = SessionLocal()
    try:
        rows = (
            db.query(ChatMessage.section_index, ChatMessage.content)
            .filter(
                ChatMessage.user_id == user_id,
                ChatMessage.context_type.in_(("lesson", "folder")),
                ChatMessage.context_id == folder_name,
                ChatMessage.role == "user",
                ChatMessage.section_index.isnot(None),
            )
            .all()
        )
        for idx, content in rows:
            if idx is None:
                continue
            text = (content or "").strip()
            if text and not _SECTION_OPENER.match(text):
                started.add(int(idx))
    finally:
        db.close()
    # Sections the student finished and advanced past.
    for i in range(max(0, current_section)):
        started.add(i)
    return started


def _group_episodes_by_section(student_orch, course_ns: str, sections: list) -> dict[int, list]:
    """Episodes grouped by lesson section (ground truth for section-scoped mastery)."""
    by_sec: dict[int, list] = defaultdict(list)
    title_to_idx = {
        (s.get("title") or "").strip().lower(): i
        for i, s in enumerate(sections)
    }
    for it in student_orch.episodes.all(course_ns):
        ss = it.store_specific or {}
        idx = ss.get("section_index")
        if idx is None:
            st = (ss.get("section_title") or "").strip().lower()
            if st in title_to_idx:
                idx = title_to_idx[st]
        if idx is not None:
            by_sec[int(idx)].append(it)
    return by_sec


def _build_chat_section_index(user_id: int, folder_name: str) -> dict[str, int]:
    """Map student message text → section index (for legacy episode backfill)."""
    index: dict[str, int] = {}
    db = SessionLocal()
    try:
        rows = (
            db.query(ChatMessage.section_index, ChatMessage.content)
            .filter(
                ChatMessage.user_id == user_id,
                ChatMessage.context_type.in_(("lesson", "folder")),
                ChatMessage.context_id == folder_name,
                ChatMessage.role == "user",
                ChatMessage.section_index.isnot(None),
            )
            .all()
        )
        for idx, content in rows:
            if idx is None:
                continue
            text = (content or "").strip()
            if not text or _SECTION_OPENER.match(text):
                continue
            index[text[:200]] = int(idx)
    finally:
        db.close()
    return index


def _episodes_for_section(
    section_index: int,
    episodes_by_section: dict[int, list],
    all_episodes: list,
    chat_index: dict[str, int],
) -> list:
    """Episodes for one section — tagged directly or matched via chat text."""
    eps = list(episodes_by_section.get(section_index, []))
    seen = {id(ep) for ep in eps}
    for ep in all_episodes:
        if id(ep) in seen:
            continue
        ss = ep.store_specific or {}
        if ss.get("section_index") is not None:
            continue
        um = (ss.get("user_message") or "").strip()[:200]
        if chat_index.get(um) == section_index:
            eps.append(ep)
            seen.add(id(ep))
    return eps


def _section_mastery_pct(episodes: list, mastery_by_id: dict[str, float]) -> int:
    """Mastery % from concepts touched in THIS section only."""
    concept_ids: set[str] = set()
    for ep in episodes:
        for cid in (ep.store_specific or {}).get("concept_ids") or []:
            if cid:
                concept_ids.add(cid)

    scores = [mastery_by_id[cid] for cid in concept_ids if cid in mastery_by_id]
    if scores:
        return round(sum(scores) / len(scores) * 100)

    outcomes: list[float] = []
    for ep in episodes:
        o = (ep.store_specific or {}).get("outcome")
        if o == "success":
            outcomes.append(1.0)
        elif o == "struggle":
            outcomes.append(0.0)
    if outcomes:
        return round(sum(outcomes) / len(outcomes) * 100)
    return 0


def _section_is_finished(section_index: int, current_section: int, episodes: list) -> bool:
    """A section counts as finished once the student advances past it or completes it."""
    if section_index < current_section:
        return True
    for ep in episodes:
        ss = ep.store_specific or {}
        if ss.get("episode_type") == "section_completed" and ss.get("section_index") == section_index:
            return True
    return False


def get_section_concept_refs(
    user_id: int,
    folder_name: str,
    section_index: int,
    source_user_id: int | None = None,
) -> list[dict]:
    """Concept refs for a lesson section (for OMA mastery updates)."""
    src_uid = source_user_id if source_user_id is not None else user_id
    db = SessionLocal()
    try:
        outline = (
            db.query(CourseOutline)
            .filter(CourseOutline.user_id == user_id, CourseOutline.folder_name == folder_name)
            .first()
        )
        if not outline:
            return []
        sections = json.loads(outline.outline_json)
        if section_index < 0 or section_index >= len(sections):
            return []

        import oma_provider
        from coast_content_oma.stores import make_namespace

        sec = sections[section_index]
        content_ns = make_namespace(src_uid, folder_name)
        content_orch = oma_provider._content_orchestrator()
        matched = _resolve_section_concepts(
            content_orch,
            content_ns,
            sec.get("key_topics") or [],
            sec.get("title") or "",
        )
        refs: list[dict] = []
        seen: set[str] = set()
        for cid, c in matched.items():
            if cid in seen:
                continue
            seen.add(cid)
            name = (c.store_specific or {}).get("name") or cid
            refs.append({"concept_id": cid, "concept_name": name})
        return refs
    finally:
        db.close()


def get_section_mastery_list(
    user_id: int,
    folder_name: str,
    sections: list,
    current_section: int,
    source_user_id: int | None = None,
) -> list[dict]:
    """Per-section mastery — only sections the student started; scores from
    concepts/episodes tagged to that section (no cross-section bleed)."""
    try:
        import oma_provider
        from coast_content_oma.student.stores import course_namespace

        if not oma_provider.is_student_enabled():
            return [_empty_section_progress(i, current_section) for i in range(len(sections))]

        course_ns = course_namespace(user_id, folder_name)
        student_orch = oma_provider._student_orchestrator()
        started = _sections_started(user_id, folder_name, current_section)
        episodes_by_section = _group_episodes_by_section(student_orch, course_ns, sections)
        all_episodes = student_orch.episodes.all(course_ns)
        chat_index = _build_chat_section_index(user_id, folder_name)

        mastery_by_id: dict[str, float] = {}
        for it in student_orch.mastery.all(course_ns):
            ss = it.store_specific or {}
            cid = ss.get("concept_id")
            if cid:
                mastery_by_id[cid] = float(ss.get("mastery_score", 0))

        out = []
        for i in range(len(sections)):
            if i not in started:
                out.append({
                    "index": i,
                    "mastery_pct": None,
                    "attempted": False,
                    "mastered": False,
                })
                continue

            sec_eps = _episodes_for_section(i, episodes_by_section, all_episodes, chat_index)
            pct = _section_mastery_pct(sec_eps, mastery_by_id)
            out.append({
                "index": i,
                "mastery_pct": pct,
                "attempted": True,
                "mastered": pct >= 100,
            })
        return out
    except Exception:
        traceback.print_exc()
        return [_empty_section_progress(i, current_section) for i in range(len(sections))]


def _empty_section_progress(index: int, current_section: int) -> dict:
    return {
        "index": index,
        "mastery_pct": None,
        "attempted": False,
        "mastered": False,
    }


def get_lesson_state(user_id: int, folder_name: str, source_user_id: int | None = None) -> dict:
    """Get current lesson state for a folder."""
    db = SessionLocal()
    try:
        content_ready = True
        shared_content_ready = True
        is_curated = False
        try:
            from curated_config import (
                CURATED_FOLDER_NAMES,
                ensure_curated_outline,
                is_curated_content_ready,
            )
            is_curated = folder_name in CURATED_FOLDER_NAMES
            if is_curated:
                shared_content_ready = is_curated_content_ready(folder_name)
                content_ready = shared_content_ready
        except Exception:
            pass

        outline = (
            db.query(CourseOutline)
            .filter(CourseOutline.user_id == user_id, CourseOutline.folder_name == folder_name)
            .first()
        )

        # Auto-enroll premade lessons when shared content is already built.
        if is_curated and shared_content_ready and not outline:
            ensure_curated_outline(user_id, folder_name)
            outline = (
                db.query(CourseOutline)
                .filter(CourseOutline.user_id == user_id, CourseOutline.folder_name == folder_name)
                .first()
            )

        if not outline:
            return {
                "has_outline": False,
                "content_ready": content_ready,
                "shared_content_ready": shared_content_ready,
            }

        sections = json.loads(outline.outline_json)
        src_uid = source_user_id if source_user_id is not None else user_id
        section_progress = get_section_mastery_list(
            user_id, folder_name, sections, outline.current_section, source_user_id=src_uid,
        )
        is_complete = outline.current_section >= outline.total_sections
        current_section = int(outline.current_section)
        for i, sp in enumerate(section_progress):
            finished = i < current_section or (is_complete and i < len(sections))
            if finished:
                sp["mastered"] = True
                sp["attempted"] = True
                if sp.get("mastery_pct") is None or sp["mastery_pct"] < 100:
                    sp["mastery_pct"] = 100

        if _sync_ever_mastered(outline, section_progress):
            outline.updated_at = datetime.now(timezone.utc)
            db.commit()
        ever_mastered = bool(getattr(outline, "ever_mastered", False))

        return {
            "has_outline": True,
            "content_ready": content_ready,
            "shared_content_ready": shared_content_ready,
            "sections": sections,
            "total_sections": outline.total_sections,
            "current_section": outline.current_section,
            "estimated_minutes": outline.estimated_minutes,
            "progress_percent": round((outline.current_section / max(outline.total_sections, 1)) * 100),
            "is_complete": is_complete,
            "ever_mastered": ever_mastered,
            "section_progress": section_progress,
            "section_verified": can_advance_from_section(
                user_id, folder_name, int(outline.current_section),
            ) if outline.current_section < len(sections) else False,
        }
    finally:
        db.close()


def get_authoritative_progress(user_id: int, folder_name: str) -> dict | None:
    """Ground-truth lesson progress from CourseOutline — not inferred from chat."""
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
        cs = int(outline.current_section)
        completed = []
        for i in range(min(cs, len(sections))):
            sec = sections[i]
            completed.append({
                "index": i,
                "title": sec.get("title") or f"Section {i + 1}",
                "key_topics": sec.get("key_topics") or [],
            })
        current = sections[cs] if cs < len(sections) else None
        all_topics: list[str] = []
        for c in completed:
            all_topics.extend(c.get("key_topics") or [])
        return {
            "current_section": cs,
            "total_sections": outline.total_sections,
            "is_complete": cs >= outline.total_sections,
            "completed_sections": completed,
            "topics_covered": list(dict.fromkeys(all_topics)),
            "current_section_title": (current or {}).get("title") if current else None,
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

        idx = int(outline.current_section)
        if idx < outline.total_sections and not can_advance_from_section(user_id, folder_name, idx):
            return {
                "error": "Pedro must verify this section before you can continue. "
                "Complete all practice questions first.",
            }

        sections = json.loads(outline.outline_json)
        finished_title = ""
        if outline.current_section < outline.total_sections:
            finished_idx = int(outline.current_section)
            if finished_idx < len(sections):
                sec = sections[finished_idx]
                finished_title = sec.get("title") or f"Section {finished_idx + 1}"
                try:
                    import oma_provider
                    if oma_provider.is_student_enabled():
                        oma_provider.record_section_completed_authoritative(
                            user_id,
                            folder_name,
                            section_index=finished_idx,
                            section_title=finished_title,
                        )
                except Exception:
                    traceback.print_exc()

            outline.current_section += 1
            outline.updated_at = datetime.now(timezone.utc)
            clear_section_verification(user_id, folder_name, finished_idx)
            db.commit()

        is_complete = outline.current_section >= outline.total_sections
        if is_complete:
            _sync_ever_mastered(outline)
            outline.updated_at = datetime.now(timezone.utc)
            db.commit()

        return {
            "current_section": outline.current_section,
            "total_sections": outline.total_sections,
            "is_complete": is_complete,
            "ever_mastered": bool(getattr(outline, "ever_mastered", False)),
            "next_section": sections[outline.current_section] if outline.current_section < len(sections) else None,
        }
    finally:
        db.close()


def claim_section_reward(
    user_id: int,
    folder_name: str,
    section_index: int | None = None,
) -> dict:
    """XP + map reward when Pedro emits [SECTION_COMPLETE] (before Next Section)."""
    db = SessionLocal()
    try:
        outline = (
            db.query(CourseOutline)
            .filter(CourseOutline.user_id == user_id, CourseOutline.folder_name == folder_name)
            .first()
        )
        if not outline:
            return {"error": "No outline found"}

        sections = json.loads(outline.outline_json or "[]")
        idx = int(outline.current_section if section_index is None else section_index)
        if idx < 0 or idx >= len(sections):
            return {"error": "Invalid section index"}

        if not can_advance_from_section(user_id, folder_name, idx):
            return {"error": "Section not verified by Pedro yet"}

        sec = sections[idx]
        title = sec.get("title") or f"Section {idx + 1}"
        mins = max(int(sec.get("estimated_minutes") or 20), 25)
        lesson_complete = (idx + 1) >= outline.total_sections

        import map_world
        return map_world.claim_section_reward(
            user_id,
            folder_name,
            idx,
            section_title=title,
            lesson_complete=lesson_complete,
            section_minutes=mins,
        )
    finally:
        db.close()


def _sync_ever_mastered(outline, section_progress: list | None = None) -> bool:
    """Persist course-level mastery badge — survives lesson reset/replay."""
    if bool(getattr(outline, "ever_mastered", False)):
        return True
    if outline.total_sections > 0 and outline.current_section >= outline.total_sections:
        outline.ever_mastered = True
        return True
    if section_progress and outline.total_sections > 0:
        all_done = all(
            (section_progress[i].get("mastery_pct") or 0) >= 100
            for i in range(min(len(section_progress), outline.total_sections))
            if section_progress[i].get("attempted") or section_progress[i].get("mastered")
        )
        attempted_count = sum(
            1 for i in range(min(len(section_progress), outline.total_sections))
            if section_progress[i].get("attempted") or section_progress[i].get("mastered")
        )
        if attempted_count >= outline.total_sections and all_done:
            outline.ever_mastered = True
            return True
    return False


def reset_lesson(user_id: int, folder_name: str) -> dict:
    """Reset lesson progress pointer for replay — keeps ever_mastered trophy and notes."""
    db = SessionLocal()
    try:
        outline = (
            db.query(CourseOutline)
            .filter(CourseOutline.user_id == user_id, CourseOutline.folder_name == folder_name)
            .first()
        )
        if not outline:
            return {"error": "No outline found"}

        ever_mastered = bool(getattr(outline, "ever_mastered", False))

        outline.current_section = 0
        outline.updated_at = datetime.now(timezone.utc)

        sections = json.loads(outline.outline_json)
        for i in range(len(sections)):
            clear_section_verification(user_id, folder_name, i)

        db.commit()
        return {
            "status": "reset",
            "current_section": 0,
            "ever_mastered": ever_mastered,
        }
    finally:
        db.close()


def build_test_out_prompt(
    user_id: int,
    folder_name: str,
    target_section_index: int,
    source_user_id: int | None = None,
    student_message: str | None = None,
) -> str | None:
    """Socratic placement-test prompt to unlock a future section."""
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
        current_idx = int(outline.current_section)
        target_idx = int(target_section_index)

        if target_idx <= current_idx or target_idx >= len(sections):
            return None

        target = sections[target_idx]
        target_title = target.get("title") or f"Section {target_idx + 1}"
        skipped = sections[current_idx:target_idx]

        prereq_topics: list[str] = []
        prereq_objectives: list[str] = []
        for sec in skipped:
            prereq_topics.extend(sec.get("key_topics") or [])
            prereq_objectives.extend(sec.get("learning_objectives") or [])

        query_parts = [target_title] + prereq_topics + prereq_objectives
        query = " ".join(dict.fromkeys(query_parts))
        material_context, _ = _fetch_section_material(
            src_uid, folder_name, query, max_chars=16000,
        )

        skipped_lines = []
        for i, sec in enumerate(skipped, start=current_idx + 1):
            title = sec.get("title") or f"Section {i}"
            topics = ", ".join(sec.get("key_topics") or []) or "(general concepts)"
            skipped_lines.append(f"  - Section {i}: {title} — topics: {topics}")

        prompt = (
            "You are Pedro, running a focused PLACEMENT TEST (test-out) session.\n\n"
            f"The student wants to skip ahead to Section {target_idx + 1}: \"{target_title}\" "
            f"without completing Sections {current_idx + 1} through {target_idx} in order.\n\n"
            "SECTIONS THEY ARE SKIPPING:\n"
            + "\n".join(skipped_lines)
            + "\n\n"
            "YOUR JOB:\n"
            "- Run a tight, Socratic assessment — ONE question at a time.\n"
            "- Test conceptual mastery of the SKIPPED sections' key topics. Do NOT teach full lectures.\n"
            "- Ask 4–6 questions spanning different skipped topics. Increase difficulty gradually.\n"
            "- Keep replies concise (2–4 sentences plus the question).\n"
            "- Grade each student answer with [ANSWER_CORRECT] or [ANSWER_WRONG] at the end of your message.\n"
            "- If they struggle, give a brief hint and ask a simpler follow-up on the same topic.\n"
            "- Emit [TEST_OUT_PASSED] ONLY when the student has answered at least 4 questions correctly "
            "across at least 3 different topic areas from the skipped sections, demonstrating they already "
            "know the prerequisite material.\n"
            "- Do NOT emit [TEST_OUT_PASSED] if they have fewer than 4 [ANSWER_CORRECT] gradings in this session.\n"
            "- Do NOT emit [SECTION_COMPLETE] — this is a placement test, not a lesson section.\n"
            "- Start by briefly explaining you'll run a quick placement test, then ask your first question.\n\n"
        )

        if material_context:
            prompt += (
                "--- REFERENCE MATERIAL (for question design only — do not lecture from it) ---\n"
                + material_context
                + "\n--- END REFERENCE MATERIAL ---\n"
            )

        return prompt
    except Exception:
        traceback.print_exc()
        return None
    finally:
        db.close()


def apply_test_out(user_id: int, folder_name: str, target_section_index: int) -> dict:
    """Jump the student to target_section after passing a placement test."""
    db = SessionLocal()
    try:
        outline = (
            db.query(CourseOutline)
            .filter(CourseOutline.user_id == user_id, CourseOutline.folder_name == folder_name)
            .first()
        )
        if not outline:
            return {"error": "No outline found"}

        sections = json.loads(outline.outline_json)
        current_idx = int(outline.current_section)
        target_idx = int(target_section_index)

        if target_idx <= current_idx:
            return {"error": "Section already unlocked"}
        if target_idx >= len(sections):
            return {"error": "Invalid section index"}

        for i in range(current_idx, target_idx):
            mark_section_verified(user_id, folder_name, i)
            sec = sections[i]
            title = sec.get("title") or f"Section {i + 1}"
            try:
                import oma_provider
                if oma_provider.is_student_enabled():
                    oma_provider.record_section_completed_authoritative(
                        user_id,
                        folder_name,
                        section_index=i,
                        section_title=title,
                    )
            except Exception:
                traceback.print_exc()

        outline.current_section = target_idx
        outline.updated_at = datetime.now(timezone.utc)
        if target_idx >= outline.total_sections:
            _sync_ever_mastered(outline)
        db.commit()

        return {
            "current_section": target_idx,
            "total_sections": outline.total_sections,
            "is_complete": target_idx >= outline.total_sections,
            "ever_mastered": bool(getattr(outline, "ever_mastered", False)),
            "skipped_sections": list(range(current_idx, target_idx)),
        }
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
    """Find images from folder sources that are relevant to the current section.

    Strategy: first restrict to images from the section's source documents,
    then rank by topic keyword overlap. Only falls back to all-folder search
    if no source-matched images exist.
    """
    db = SessionLocal()
    try:
        all_images = db.query(SourceImage).filter(
            SourceImage.user_id == user_id,
            SourceImage.folder_name == folder_name,
        ).all()

        if not all_images:
            return ""

        all_sources = db.query(FolderSource).filter(
            FolderSource.user_id == user_id,
            FolderSource.folder_name == folder_name,
        ).all()
        src_by_id = {s.source_id: s for s in all_sources}

        nb_lower = [n.lower().strip() for n in source_notebooks if n and n.strip()]

        matched_source_ids: set[str] = set()
        for src in all_sources:
            title = (src.title or "").lower()
            for nb in nb_lower:
                if nb in title or title in nb:
                    matched_source_ids.add(src.source_id)
                    break

        candidate_images = [si for si in all_images if si.source_id in matched_source_ids]
        if not candidate_images:
            candidate_images = all_images

        search_terms = [t.lower().strip() for t in [section_title] + key_topics if t and t.strip()]

        def _score(si: SourceImage) -> float:
            ctx = (si.context_text or "").lower()
            score = 0.0
            for term in search_terms:
                if len(term) > 4 and term in ctx:
                    score += 15
                else:
                    words = [w for w in term.split() if len(w) > 3]
                    matched = sum(1 for w in words if w in ctx)
                    if words and matched > 0:
                        score += 5 * (matched / len(words))
            if si.source_id in matched_source_ids:
                score += 20
            return score

        scored = [(si, _score(si)) for si in candidate_images]
        scored.sort(key=lambda x: -x[1])
        top = scored[:max_images]

        if not top:
            return ""

        api_base = os.getenv("API_BASE_URL", "https://coast-backend-dlg6.onrender.com")
        lines = []
        for si, sc in top:
            src = src_by_id.get(si.source_id)
            src_title = src.title if src else si.source_id
            url = f"{api_base}/api/source-images/{si.id}"
            ctx_preview = (si.context_text or "")[:150].replace("\n", " ")
            lines.append(
                f"- Image {si.id}: from \"{src_title}\" page {si.page_number} "
                f"(relevance={sc:.0f}) | context: \"{ctx_preview}\" | URL: {url}"
            )

        return (
            "\n--- DIAGRAMS FROM SOURCE MATERIALS ---\n"
            f"Current section: \"{section_title}\".\n"
            f"Topics: {', '.join(key_topics[:5])}.\n"
            "ONLY use a diagram if its 'context' field clearly relates to what you are "
            "currently explaining. If the context mentions a DIFFERENT topic, skip it. "
            "It is better to show NO diagram than a wrong one.\n"
            "Use markdown: ![brief description](URL)\n\n"
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
    search_terms_extra: list[str] | None = None,
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
        if search_terms_extra:
            for term in search_terms_extra:
                t = term.lower().strip()
                if len(t) > 2 and t not in search_terms:
                    search_terms.append(t)

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


def _fetch_section_material(
    user_id: int,
    folder_name: str,
    query: str,
    max_chars: int = 22000,
) -> tuple[str, bool]:
    """Retrieve course material for a lesson section.

    Returns (context_text, used_oma). Tries Content OMA first, then flat RAG.
    """
    try:
        import oma_provider
        block, source, _ = oma_provider.resolve_folder_content(
            user_id, folder_name, query,
            context_type="lesson",
            max_chars=max_chars,
            max_content=12,
            max_images=3,
        )
        return block, source == "OMA"
    except Exception:
        traceback.print_exc()
        return "", False


def build_lesson_prompt(
    user_id: int,
    folder_name: str,
    source_user_id: int | None = None,
    structure: dict | None = None,
    student_message: str | None = None,
    section_index: int | None = None,
) -> str | None:
    """Build a specialized system prompt for the current lesson section.
    
    source_user_id: if set, read sources/RAG from this user (for curated/shared folders).
    structure: optional dict with pedagogy hints for curated lessons.
    student_message: when set, also retrieve Content OMA material targeted at the
        student's current question (skipped for automatic section-openers).
    section_index: active section in chat (defaults to outline progress pointer).
    """
    src_uid = source_user_id if source_user_id is not None else user_id
    recap = is_recap_request(student_message)
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
        current_idx = int(section_index if section_index is not None else outline.current_section)
        if current_idx >= len(sections):
            current_idx = max(0, len(sections) - 1)

        current = sections[current_idx]
        section_title = current.get("title", f"Section {current_idx + 1}")
        objectives = current.get("learning_objectives", [])
        key_topics = current.get("key_topics", [])
        source_nbs = current.get("source_notebooks", [])

        query_parts = [section_title] + key_topics + objectives
        if recap and student_message:
            for s in sections:
                query_parts.append(s.get("title") or "")
                query_parts.extend(s.get("key_topics") or [])
            query_parts.insert(0, student_message)
        elif student_message:
            query_parts.insert(0, student_message)
        query = " ".join(dict.fromkeys(p for p in query_parts if p))
        max_material = 32000 if recap else 22000
        material_context, used_oma = _fetch_section_material(
            src_uid, folder_name, query, max_chars=max_material,
        )

        # For follow-up questions in lesson chat, pull query-specific excerpts too.
        if student_message:
            try:
                import oma_provider
                if oma_provider.is_oma_enabled() and not oma_provider._is_lesson_intro(student_message):
                    extra, q_concept_ids = oma_provider.get_folder_context(
                        src_uid, folder_name, student_message,
                        max_chars=12000 if recap else 8000,
                        max_content=12 if recap else 8,
                        max_images=2,
                    )
                    if extra and extra not in material_context:
                        material_context += (
                            "\n\n--- ADDITIONAL MATERIAL FOR THIS QUESTION "
                            "(retrieved via Content OMA) ---\n"
                            + extra
                        )
                        used_oma = True
                        oma_provider.log_content_source(
                            "OMA",
                            context_type="lesson-question",
                            folder=folder_name,
                            user_id=src_uid,
                            chars=len(extra),
                            detail=f"{len(q_concept_ids)} concepts",
                        )
            except Exception:
                traceback.print_exc()

        if not material_context:
            fallback_query = query
            if recap:
                fallback_query = " ".join(
                    s.get("title", "") for s in sections
                ) + " " + (student_message or "")
            material_context = _fallback_source_context(
                src_uid, folder_name, section_title, key_topics, source_nbs,
                max_chars=32000 if recap else 24000,
                search_terms_extra=fallback_query.split() if recap else None,
            )
            if material_context:
                try:
                    import oma_provider
                    oma_provider.log_content_source(
                        "FALLBACK",
                        context_type="lesson",
                        folder=folder_name,
                        user_id=src_uid,
                        chars=len(material_context),
                        detail="title-matched raw text",
                    )
                except Exception:
                    pass

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
        )
        prompt += get_verification_prompt_block(
            user_id, folder_name, current_idx, student_message,
        )
        prompt += (
            f"Source materials: {', '.join(source_nbs)}\n\n"
        )

        if completed:
            prompt += f"Already covered: {', '.join(completed)}\n"
        if upcoming:
            prompt += f"Coming next: {', '.join(upcoming[:3])}\n"

        if recap:
            prompt += (
                "\nRECAP / SUMMARY MODE:\n"
                "The student wants a summary of what they've covered in this course. "
                "Use ALL of the following — do NOT say you lack their lecture slides:\n"
                "1) SOURCE MATERIAL below (from their uploaded PDFs via Content OMA)\n"
                "2) RECENT LESSON CONVERSATIONS below (what you already taught them)\n"
                "3) STUDENT PROFILE (sections finished, mastery, accomplishments)\n"
                "Organize by section or topic. Be specific to THEIR course materials.\n"
            )
            recap_block = _fetch_lesson_conversation_recap(user_id, folder_name)
            if recap_block:
                prompt += "\n" + recap_block

        prompt += (
            "\nTEACHING APPROACH:\n"
            "Adapt your starting point to the STUDENT PROFILE block (which appears earlier in this "
            "system prompt). The profile records this student's prior interactions with this course's "
            "concepts.\n"
            "- If the profile is empty or shows no prior coverage of the concepts in this section, "
            "assume the student is a COMPLETE BEGINNER and teach this section from scratch.\n"
            "- If the profile shows the student has already mastered the concepts in this section "
            "(mastery scores at or near 1.00, listed under 'Confident in'), treat this as a REVIEW: "
            "briefly recap each key idea in 1–2 sentences, then move directly to harder practice "
            "problems and exam-style questions. Do NOT re-teach material they've already mastered.\n"
            "- If the profile shows the student is BORDERLINE on some concepts (mid mastery) and "
            "WEAK on others ('Struggling with'), focus your teaching on the weak and borderline "
            "concepts. Skim the mastered ones with a quick 'as you already know, ...' framing.\n"
            "- The STUDENT PROFILE is your source of truth for what the student already knows in "
            "this course. Trust it over any other signal.\n"
            "- NEVER tell the student they said something was 'tricky', 'hard', or 'confusing' "
            "unless the profile's Open questions or Unresolved fields explicitly say so. "
            "Mastery scores alone do not mean the student complained — 'Has struggled with' "
            "means you corrected them on a quiz, not that they self-reported difficulty.\n\n"

            "INSTRUCTIONS:\n"
            "1. You MUST teach from the source material provided below. Do NOT say you don't have it — it is included below.\n"
            "2. For concepts NOT yet mastered (per STUDENT PROFILE): define every term, explain "
            "from the ground up, never assume prior knowledge. For concepts the profile shows as "
            "mastered: reference them briefly and confirm with a quick check-question.\n"
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
            "and tell them clearly: 'You should review [specific topic] more — this is an area to focus on.' "
            "Then include the exact tag [ANSWER_WRONG] at the very end of your message (after your explanation). "
            "This logs a practice mistake for review — it does NOT mean the student is struggling; "
            "you will re-teach and verify understanding.\n"
            "   f) When the student answers a practice problem correctly, include [ANSWER_CORRECT] at the end "
            "of that message.\n"
            "   g) Only use [ANSWER_WRONG] / [ANSWER_CORRECT] when grading a student answer attempt — never during "
            "teaching, explanations, or when the student has not submitted an answer.\n"
            "   h) Only after the student has attempted the practice problems should you consider the section complete.\n"
            "9. When the student has completed the practice round successfully, congratulate them, summarize what they learned, "
            "note any weak areas they should revisit, and ask if they're ready for the next section.\n"
            "10. NEVER claim you don't have access to the student's uploaded documents. The relevant content is provided below.\n"
            "11. Emit [SECTION_COMPLETE] ONLY when ALL verification questions in this section were answered correctly "
            "(each graded with [ANSWER_CORRECT]). Never because the student said 'next', 'skip', or 'continue'. "
            "If they ask a follow-up question after passing some checks, teach them, then re-ask any they missed.\n"
            "12. When you believe the student has understood the section AND completed the practice round, include the exact phrase "
            '"[SECTION_COMPLETE]" at the end of your message (the frontend uses this to show a Next Section button).\n\n'
        )

        if structure and structure.get("pedagogy"):
            prompt += (
                "--- SPECIAL TEACHING METHODOLOGY FOR THIS COURSE ---\n"
                + structure["pedagogy"]
                + "\n--- END METHODOLOGY ---\n\n"
            )

        if material_context:
            source_label = (
                "Content OMA — structured retrieval from the student's uploaded documents"
                if used_oma
                else "from the student's uploaded documents"
            )
            prompt += (
                f"--- SOURCE MATERIAL FOR THIS SECTION ({source_label}) ---\n"
                + material_context
                + "\n--- END SOURCE MATERIAL ---\n"
            )
        else:
            prompt += (
                "NOTE: No source material was retrieved for this section. Teach using your general knowledge "
                "of the topic, but let the student know you're drawing from general knowledge rather than "
                "their specific uploaded materials.\n"
            )

        # OMA blocks already include relevant diagrams with URLs; only fall back
        # to legacy SourceImage lookup when flat RAG was used.
        if not used_oma:
            image_block = _find_relevant_images(src_uid, folder_name, section_title, key_topics, source_nbs)
            if image_block:
                prompt += image_block

        return prompt
    except Exception:
        traceback.print_exc()
        return None
    finally:
        db.close()


def get_section_constellation(
    user_id: int,
    folder_name: str,
    section_index: int,
    source_user_id: int | None = None,
) -> dict:
    """Build per-section mastery constellation from outline key_topics + Content OMA."""
    src_uid = source_user_id if source_user_id is not None else user_id
    db = SessionLocal()
    try:
        outline = (
            db.query(CourseOutline)
            .filter(CourseOutline.user_id == user_id, CourseOutline.folder_name == folder_name)
            .first()
        )
        if not outline:
            return {"error": "No lesson outline for this folder."}

        sections = json.loads(outline.outline_json)
        if section_index < 0 or section_index >= len(sections):
            return {"error": "Invalid section index."}

        section = sections[section_index]
        key_topics = section.get("key_topics") or []
        section_title = section.get("title") or f"Section {section_index + 1}"

        import oma_provider
        from coast_content_oma.stores import make_namespace
        from coast_content_oma.student.mastery_tier import compute_mastery_tier, edge_link_state
        from coast_content_oma.student.stores import course_namespace

        content_ns = make_namespace(src_uid, folder_name)
        course_ns = course_namespace(user_id, folder_name)
        content_orch = oma_provider._content_orchestrator()
        student_orch = oma_provider._student_orchestrator()

        concept_map = _resolve_section_concepts(
            content_orch, content_ns, key_topics, section_title,
        )

        mastery_by_id = {}
        for it in student_orch.mastery.all(course_ns):
            ss = it.store_specific or {}
            cid = ss.get("concept_id")
            if cid:
                mastery_by_id[cid] = ss

        nodes = []
        tier_counts = {"red": 0, "orange": 0, "yellow": 0, "green": 0}

        for cid, concept in concept_map.items():
            ss = mastery_by_id.get(cid)
            tier = compute_mastery_tier(ss)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            css = concept.store_specific or {}
            nodes.append({
                "id": cid,
                "name": css.get("name") or concept.content[:80] or cid,
                "tier": tier,
                "score": round(float(ss.get("mastery_score", 0)), 2) if ss else 0,
                "attempted": ss is not None,
                "successes": ss.get("successes", 0) if ss else 0,
                "struggles": ss.get("struggles", 0) if ss else 0,
                "definition": (concept.content or "")[:400],
                "focus_opener": _concept_focus_opener(
                    css.get("name") or cid, tier, ss,
                ),
            })

        node_ids = {n["id"] for n in nodes}
        edges = []
        seen = set()
        for cid in node_ids:
            concept = concept_map.get(cid)
            if not concept:
                continue
            pre_ids = (concept.store_specific or {}).get("prerequisite_concept_ids") or []
            for pid in pre_ids:
                if pid not in node_ids:
                    continue
                key = (pid, cid)
                if key in seen:
                    continue
                seen.add(key)
                prereq_tier = next((n["tier"] for n in nodes if n["id"] == pid), "red")
                edges.append({
                    "source": pid,
                    "target": cid,
                    "type": "prerequisite",
                    "link_state": edge_link_state(prereq_tier),
                })

        all_green = bool(nodes) and all(n["tier"] == "green" for n in nodes)

        return {
            "folder": folder_name,
            "section_index": section_index,
            "section_title": section_title,
            "key_topics": key_topics,
            "nodes": nodes,
            "edges": edges,
            "stats": {
                **tier_counts,
                "total": len(nodes),
                "all_green": all_green,
            },
        }
    except Exception:
        traceback.print_exc()
        return {"error": "Failed to build constellation."}
    finally:
        db.close()


def _resolve_section_concepts(content_orch, namespace: str, key_topics: list, section_title: str) -> dict:
    """Map outline key_topics to Content OMA concept items (+ direct prereqs)."""
    matched: dict = {}

    def add(c):
        if c and c.namespace == namespace and c.id not in matched:
            matched[c.id] = c

    search_terms = [t.strip() for t in (key_topics or []) if t and str(t).strip()]
    if section_title:
        search_terms.append(section_title.strip())

    for topic in search_terms:
        found = content_orch.concept.find_by_name(namespace, topic)
        if found:
            add(found)
            continue
        for cand in content_orch.concept.find_candidates(namespace, topic, max_results=2):
            add(cand)

    if len(matched) < 3 and search_terms:
        query = " ".join(search_terms[:6])
        try:
            result = content_orch.retrieve(namespace, query, max_content=4, max_images=0)
            for cand in result.concept_candidates:
                add(cand)
        except Exception:
            pass

    # Include direct prerequisites so dependency links render.
    extra: dict = {}
    for cid, concept in list(matched.items()):
        pre_ids = (concept.store_specific or {}).get("prerequisite_concept_ids") or []
        for pid in pre_ids:
            if pid not in matched:
                p = content_orch.concept.get(pid)
                if p and p.namespace == namespace:
                    extra[pid] = p
    matched.update(extra)

    if not matched:
        for c in content_orch.concept.all_concepts(namespace)[:8]:
            add(c)

    return matched


def _concept_focus_opener(name: str, tier: str, ss: dict | None) -> str:
    if tier == "red" and not ss:
        return f"Let's work on {name} — we'll build this up step by step."
    if tier == "red":
        return f"Last time {name} was tricky — let's pin down exactly where the confusion is."
    if tier == "orange":
        return f"Your understanding of {name} is getting there — let's sharpen it with a few targeted questions."
    if tier == "yellow":
        return f"You know {name} — let's test it on a fresh problem to make it stick."
    return f"Quick review of {name} — you're in good shape here."


def build_concept_focus_prompt(
    user_id: int,
    folder_name: str,
    concept_id: str,
    section_index: int,
    source_user_id: int | None = None,
    student_message: str | None = None,
) -> str | None:
    """Focused Pedro session for a single concept from the mastery map."""
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
        if section_index < 0 or section_index >= len(sections):
            return None

        section = sections[section_index]
        section_title = section.get("title", "")

        import oma_provider
        from coast_content_oma.stores import make_namespace
        from coast_content_oma.student.mastery_tier import compute_mastery_tier
        from coast_content_oma.student.stores import course_namespace

        content_ns = make_namespace(src_uid, folder_name)
        course_ns = course_namespace(user_id, folder_name)
        content_orch = oma_provider._content_orchestrator()
        student_orch = oma_provider._student_orchestrator()

        concept = content_orch.concept.get(concept_id)
        if not concept or concept.namespace != content_ns:
            return None

        css = concept.store_specific or {}
        name = css.get("name") or concept_id
        definition = (concept.content or "")[:2000]

        mastery = student_orch.mastery.for_concept(course_ns, concept_id)
        ss = mastery.store_specific if mastery else None
        tier = compute_mastery_tier(ss)
        opener = _concept_focus_opener(name, tier, ss)

        material_context, _ = _fetch_section_material(
            src_uid, folder_name, name, max_chars=8000,
        )
        if student_message and oma_provider.is_oma_enabled():
            extra, _ = oma_provider.get_folder_context(
                src_uid, folder_name, student_message,
                max_chars=4000, max_content=4, max_images=1,
            )
            if extra:
                material_context = (material_context or "") + "\n\n" + extra

        profile_block = oma_provider.get_student_profile_block(
            user_id, folder_name, current_concept_ids=[concept_id],
        )

        pre_names = []
        for pid in (css.get("prerequisite_concept_ids") or [])[:4]:
            p = content_orch.concept.get(pid)
            if p:
                pre_names.append((p.store_specific or {}).get("name") or pid)

        prompt = (
            f"\n--- CONCEPT FOCUS MODE ---\n"
            f"The student tapped the \"{name}\" node on their section mastery map.\n"
            f"Section: {section_title} (section {section_index + 1})\n"
            f"Mastery tier: {tier.upper()}\n\n"
            f"OPENING TONE: Start with something like: \"{opener}\"\n\n"
            f"RULES:\n"
            f"- Teach ONLY \"{name}\" — do not drift to other section topics.\n"
            f"- Use Socratic dialogue; ask before telling when tier is orange/red.\n"
            f"- For yellow tier: give a novel application problem without hints first.\n"
            f"- Keep responses concise (2–4 short paragraphs max).\n"
            f"- Do NOT emit [SECTION_COMPLETE] — this is concept practice, not a full section.\n"
            f"- When grading a practice answer: [ANSWER_WRONG] if wrong, [ANSWER_CORRECT] if right "
            f"(at the end of the message, after your explanation).\n"
        )

        if pre_names:
            prompt += f"- Prerequisites they should have: {', '.join(pre_names)}\n"

        if profile_block:
            prompt += f"\n{profile_block}\n"

        prompt += (
            f"\nCONCEPT DEFINITION:\n{definition}\n"
        )

        if material_context:
            prompt += (
                f"\n--- SOURCE MATERIAL ---\n{material_context}\n--- END ---\n"
            )

        return prompt
    except Exception:
        traceback.print_exc()
        return None
    finally:
        db.close()
