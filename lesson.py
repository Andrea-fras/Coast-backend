"""Lesson engine — course outline generation, progress tracking, lesson prompts."""

from __future__ import annotations

from collections import OrderedDict
import provider_capacity

import json
import os
import re
import traceback
import logging
import time
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
from lesson_sources import source_catalog, citation_instructions
from workshops import decorate_sections, is_workshop, workshop_instructions, workshop_gate, prior_work


class OutlineProviderError(RuntimeError):
    """A provider failure safe to show without exposing raw API diagnostics."""


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


WORKSHOP_OUTLINE_RULES = (
    "\n\nFORMAT: WORKSHOP — the student learns by BUILDING something real from this material, "
    "one milestone per section. Each section must produce a concrete piece of the student's own work "
    "(code, a model, a design, a worked analysis, a plan) that the next milestone builds on. For every "
    "section add a \"workshop\" object: {\"title\": same as the section title, \"outcome\": what the "
    "student will have made, \"criteria\": 2-4 observable things the student must show in their OWN work "
    "(never 'understands X'), \"coaching\": how Pedro should coach it — what to explain, what the student "
    "builds, likely pitfalls, \"course_outcome\": the same one-sentence final deliverable in every section}.\n"
)

# A workshop a student makes from their own files is an assignment they have been set: the roadmap
# follows it exercise by exercise and Pedro guides them through doing it (pedro_context ASSIGNMENT_CORE).
ASSIGNMENT_PLAN = (
    "You are planning a guided assignment workshop. The student uploaded an assignment they have been set (a "
    "problem sheet, lab, exercise notebook or coursework brief) and will do it themselves, guided by their tutor. "
    "The roadmap IS the assignment: follow it, never redesign it.\n\n"
    "Rules:\n"
    "- One milestone per exercise, task or question in the brief, in the brief's own order and numbering. Its parts "
    "(1a, 1b, ...) stay together in that milestone; split an exercise only where the brief itself splits it into "
    "separately titled tasks.\n"
    "- Title each milestone with the brief's own label and name, e.g. \"Exercise 1 · What does the policy need to know?\".\n"
    "- Add nothing the brief doesn't ask for: no warm-up, notation, background, review or summary milestones, and no "
    "extra tasks inside a milestone. What the brief gives as reference (notation, setup, starter code) belongs to the "
    "exercises that use it.\n"
    "- Optional exercises and extensions come last with \"(optional)\" in the title; short optional extensions may "
    "share one milestone.\n"
    "- Pages that only cover submission or logistics are skipped, but any constraint they set (allowed tools, length, "
    "format, deadline) goes into the coaching of the exercises it applies to.\n"
    "- If the material sets no tasks at all (lecture notes rather than an assignment), make one milestone per major "
    "topic, each a concrete piece of work that applies it.\n"
    "- learning_objectives: the exercise's parts, briefly. estimated_minutes: the brief's own estimate if it gives one.\n"
    "For every milestone add a \"workshop\" object: {\"title\": same as the section title, \"outcome\": what the "
    "exercise asks the student to produce, in the brief's terms, \"criteria\": the exercise's parts and requirements "
    "as the brief states them, one per part, 1-5 (never 'understands X', never something the brief doesn't require), "
    "\"coaching\": what the student needs to know or watch for to do it themselves (concepts it relies on, likely "
    "pitfalls, constraints from the brief), never the answer, \"course_outcome\": \"The assignment, completed in "
    "the student's own work\" or a closer sentence naming it, the same in every milestone}.\n"
)
ASSIGNMENT_PAGES = (
    "\nThese are page-addressed excerpts from the uploaded files, grouped into UNITs of up to 8 pages. In "
    "source_units give the exact pages each milestone uses, as ranges inside one source such as \"src_ab12:3-4\" "
    "(a UNIT identifier also works and means all its pages): the pages where its exercise is set, plus any reference "
    "pages it needs (notation, setup, starter code). Several milestones may use the same page. Pages with no exercise "
    "content (title, submission logistics) go in skipped_pages. Never invent a source identifier or page. The full "
    "pages will be available while the student works.\n"
)


def folder_kind(user_id: int, folder_name: str) -> str:
    """'workshop' or 'lesson' — fixed when the course was created (Workshops vs Lessons)."""
    from database import StudyFolder
    with SessionLocal() as db:
        row = db.query(StudyFolder).filter_by(user_id=user_id, name=folder_name).first()
        return (row.kind if row and row.kind else "lesson")


def _apply_course_format(sections: list[dict], course_format: str, assignment: bool = False) -> list[dict]:
    """Workshop roadmaps always carry a valid contract per milestone; lessons carry none."""
    from workshops import contract_from_section, validate_contract
    if course_format != "workshop":
        for section in sections:
            section.pop("workshop", None)
        return sections
    course_outcome = next((str(s["workshop"].get("course_outcome")) for s in sections
                           if isinstance(s.get("workshop"), dict) and s["workshop"].get("course_outcome")), None)
    for section in sections:
        raw = section.get("workshop") if isinstance(section.get("workshop"), dict) else {}
        section["workshop"] = (validate_contract({**raw, "title": section.get("title")}, course_outcome)
                               or contract_from_section(section, course_outcome))
        if assignment:
            section["workshop"]["kind"] = "assignment"
    return sections


DEPTHS = ("essentials", "complete")
ESSENTIALS_SECTION_PAGES = 14  # a normal section; essentials cuts topics, never makes sections bigger

ESSENTIALS_RULES = (
    "\n\nDEPTH: ESSENTIALS. The student wants the important part of this course, not every page of it. "
    "Keep the topics a student must genuinely understand: the central definitions, results and methods, the "
    "foundations later topics build on, and what is typically examined in a course like this. Judge importance "
    "from the material itself (what the lecturer defines, states as results, recaps, sets exercises on and "
    "keeps coming back to) and from your own knowledge of how this subject is taught and examined at university. "
    "Together the kept topics should cover about 95% of what matters. Leave out the rest: history and "
    "motivation, anecdotes, side topics and extensions, optional or advanced material, long proofs and "
    "derivations whose result is what students use, and repeated recaps. Expect roughly half the sections a "
    "complete roadmap of this material would need.\n"
    "Sections stay their normal size: each section covers at most " + str(ESSENTIALS_SECTION_PAGES) + " pages "
    "(15–30 minutes). Never make a section bigger to keep more pages: with fewer sections of normal size, you "
    "decide which topics are worth keeping and leave the others out. Give each kept topic all the pages it needs (its definition, worked examples and "
    "exercises together). Pages of topics you leave out belong to no section: list each left-out topic in "
    "left_out with its exact page ranges.\n"
)


def generate_outline(user_id: int, folder_name: str, source_user_id: int | None = None, structure: dict | None = None,
                     expected_source_ids: list[str] | None = None, course_format: str = "lesson",
                     depth: str | None = None) -> dict:
    """Generate a structured course outline from all sources in a folder.
    
    source_user_id: if set, read sources from this user (for curated/shared folders).
    structure: optional dict with custom structure hints for the outline.
    Outline is always saved under user_id (per-user progress).
    """
    src_uid = source_user_id if source_user_id is not None else user_id
    # The student's own workshop (not a curated one): an assignment to follow, not material to teach.
    assignment = course_format == "workshop" and not structure and src_uid == user_id
    db = SessionLocal()
    try:
        # Depth is a lesson choice: workshops and premade courses follow their own plan. A regenerate
        # keeps the roadmap's depth unless the student picks another; a new lesson starts at essentials.
        if course_format == "workshop" or structure:
            depth = None
        else:
            if depth not in DEPTHS:
                previous = db.query(CourseOutline.depth).filter_by(user_id=user_id, folder_name=folder_name).scalar()
                depth = previous if previous in DEPTHS else "essentials"
        essentials = depth == "essentials"
        import upload_lifecycle
        from fastapi import HTTPException
        # Curated lesson behaviour stays on its existing path.
        if not structure:
            upload_lifecycle.assert_ready(db, src_uid, folder_name)
        source_snapshot = upload_lifecycle.source_signature(db, src_uid, folder_name)
        if expected_source_ids is not None and tuple(sorted(set(expected_source_ids))) != source_snapshot[0]:
            return {"error": "Your source list changed. Review the files and generate the roadmap again.", "status_code": 409}
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
        sources_text = ""
        outline_source = "raw"
        outline_via_rag = False
        outline_via_oma = False
        planning_units = None

        try:
            import oma_provider
            if oma_provider.is_oma_enabled():
                from coast_content_oma import progressive
                try:
                    quick = progressive.overview(raw_sources, page_chars=8000 if assignment else 1000) if raw_sources and not notebooks else None
                except ValueError as exc:
                    return {"error": str(exc)}
                if quick:
                    sources_text, planning_units = quick
                    outline_via_oma = True
                    outline_source = "oma_progressive"
                else:
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
                    if pdf_sources:
                        ready = oma_provider.ensure_oma_ready_for_outline(
                            src_uid, folder_name,
                            expected_pages=total_pages,
                            pdf_sources=pdf_sources,
                            allow_sync_ingest=not os.getenv("RENDER"),
                        )
                        if not ready:
                            return {
                                "error": (
                                    "Content OMA is still indexing this course's uploads. "
                                    "Wait 2–3 minutes, then tap Generate Lesson again."
                                ),
                                "oma_indexing": True,
                            }
                    oma_index = oma_provider.build_outline_context(
                        src_uid, folder_name,
                        source_meta=source_meta,
                        max_chars=total_budget,
                    )
                    if oma_index:
                        sources_text = oma_index
                        outline_via_oma = True
                        outline_source = "oma"
                    elif pdf_sources:
                        return {
                            "error": (
                                "Content OMA could not build a course index from your uploads. "
                                "Try re-uploading or wait a bit longer."
                            ),
                        }
        except Exception:
            traceback.print_exc()

        if not sources_text:
            try:
                import rag
                rag_query = (
                    f"course outline syllabus topics learning objectives {folder_name} "
                    + " ".join(s.title for s in raw_sources[:8])
                )
                rag_block = rag.build_folder_context(
                    src_uid, folder_name, rag_query, max_chars=total_budget,
                )
                if rag_block:
                    sources_text = rag_block
                    outline_via_rag = True
                    outline_source = "rag"
            except Exception:
                traceback.print_exc()

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

        if (outline_via_oma or outline_via_rag) and total_pages:
            max_sections = min(40, max(6, total_pages // 8))
        else:
            max_sections = min(25, max(4, total_sources // 2 + 3))

        min_sections = min(max_sections,max(1,(len(planning_units)+1)//2)) if planning_units else 4
        if essentials:  # fewer sections of the same size, not the same pages in bigger sections
            max_sections = max(3, round(max_sections * 0.55))
            min_sections = max(2, min(max_sections, round(min_sections * 0.5)))
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

        if planning_units:
            oma_rules = (
                "\nThese are page-addressed planning excerpts from EVERY uploaded source, grouped into UNITs of up to "
                "8 pages for readability. In source_units give the exact pages each section teaches, as ranges inside "
                "one source such as \"src_ab12:17-36\" (a UNIT identifier also works and means all its pages). "
                "Start and end each section where the topic changes, not where a UNIT ends: a definition, its worked "
                "example and its exercises belong together. "
                + ("A page belongs to at most one section; pages of topics you leave out belong to none (list them in "
                   "left_out). Purely administrative pages (course logistics, grading rules, schedules, reading lists, "
                   "title, agenda and thank-you slides) go in skipped_pages. " if essentials else
                   "Every page belongs to exactly one section, except purely "
                   "administrative pages (course logistics, grading rules, schedules, reading lists, title, agenda and "
                   "thank-you slides), which you list in skipped_pages instead. ")
                + "A section takes 15–30 minutes to teach, "
                "usually 6–20 slides. Don't add review or summary sections without pages of their own. Put exercise "
                "and exam-question pages with the lecture pages they practise, and image pages with their neighbouring "
                "explanations. Order sections by prerequisites. Never invent a source identifier or page. "
                "The full source pages and diagrams will be available when teaching; these excerpts are only for planning.\n"
            )

        format_line = (
            "Return ONLY valid JSON — an array of section objects. No markdown fences, no explanation.\n"
            + ("Pages you leave out simply belong to no section.\n" if essentials else "")
            +
            'Format: [{"title": "...", "learning_objectives": ["...", "..."], '
            '"key_topics": ["...", "..."], "source_notebooks": ["..."], "estimated_minutes": 20'
            + (', "source_units": ["exact UNIT identifier"]' if planning_units else '')
            + (', "workshop": {"title": "...", "outcome": "...", "criteria": ["..."], "coaching": "...", '
               '"course_outcome": "..."}' if course_format == "workshop" else '') + '}]'
        )
        coverage_rule = (
            "create a structured course outline of the important topics across ALL sources, in a logical learning "
            "sequence.\n\nEvery source's important topics must be represented, even sources listed last.\n"
            if essentials else
            "create a structured course outline that covers ALL the key topics across ALL sources in a logical "
            "learning sequence.\n\n"
            "CRITICAL: You MUST include content from EVERY source listed. Do NOT skip any sources — "
            "even those listed last. The student uploaded all of them and expects the course to cover "
            "all their material.\n")
        system = (
            "You are a course designer. Given the student's source materials, " + coverage_rule
            + oma_rules
            + (ESSENTIALS_RULES if essentials else "")
            + structure_block +
            "\n\nRules:\n"
            f"- Create {min_sections}-{max_sections} sections depending on the amount of material\n"
            "- Order sections so prerequisites come first within each part/group\n"
            "- Each section should be a coherent learning unit (15-30 minutes)\n"
            "- Reference which source notebooks/documents each section draws from\n"
            "- Include 2-3 specific, concise learning objectives per section (at most 12 words each)\n"
            "- Estimate minutes per section based on content density\n"
            "- If sources cover similar topics, merge them into one section\n"
            "- If a source covers multiple distinct topics, split across sections\n\n"
            + (WORKSHOP_OUTLINE_RULES if course_format == "workshop" else "")
        )
        if assignment:
            system = ASSIGNMENT_PLAN + (ASSIGNMENT_PAGES if planning_units else "")

        material_label = (
            "Content OMA course index" if outline_via_oma
            else "RAG-retrieved source material" if outline_via_rag
            else "Source materials"
        )
        context = (
            f"Folder: {folder_name}\nNumber of sources: {total_sources}\n"
            f"Total pages: {total_pages or 'unknown'}\n\n{material_label}:\n{sources_text}"
        )

        db.rollback()  # Do not hold a read transaction over the model request.
        outline_sections, skipped_pages, left_out = _plan_outline(system, format_line, context, course_format, essentials)
        if not outline_sections:
            return {"error": "Failed to generate outline. Please try again."}

        if planning_units:
            try:
                progressive.bind_sections(outline_sections, planning_units, skipped=skipped_pages,
                                          merge_same=not assignment, fill_gaps=not essentials)
            except ValueError as error:
                repaired, skipped_again, left_again = _plan_outline(
                    system, format_line,
                    context + "\nRepair the prior response: " + str(error) + "\nPrior response: " + json.dumps(outline_sections),
                    course_format, essentials)
                left_out = left_again or left_out
                try:
                    # The sections are still good; if the references are unusable again,
                    # teach the pages in source order rather than fail the student.
                    outline_sections = progressive.bind_sections(repaired or outline_sections, planning_units, spread=True,
                                                                 skipped=skipped_again or skipped_pages,
                                                                 merge_same=not assignment, fill_gaps=not essentials)
                except ValueError as final_error:
                    return {"error": str(final_error)}

        if essentials and planning_units:
            outline_sections = _cap_section_size(outline_sections, planning_units, system, format_line, context,
                                                 course_format, skipped_pages)
        outline_sections = _apply_course_format(outline_sections, course_format, assignment)
        total_minutes = sum(s.get("estimated_minutes", 20) for s in outline_sections)
        left_out_topics = _left_out_topics(left_out, outline_sections, planning_units, skipped_pages) if essentials and planning_units else []

        if not structure:
            from sqlalchemy import text
            db.execute(text("BEGIN IMMEDIATE"))
            upload_lifecycle.assert_ready(db, src_uid, folder_name)
            if upload_lifecycle.source_signature(db, src_uid, folder_name) != source_snapshot:
                return {"error": "Your sources changed during preparation. Your existing roadmap was kept. Review the files and try again.", "status_code": 409}

        existing = (
            db.query(CourseOutline)
            .filter(CourseOutline.user_id == user_id, CourseOutline.folder_name == folder_name)
            .first()
        )

        if existing:
            from database import CourseChatEpoch
            from sqlalchemy import func
            last_message = db.query(func.max(ChatMessage.id)).filter_by(user_id=user_id,
                context_type='lesson', context_id=folder_name).scalar() or 0
            db.merge(CourseChatEpoch(user_id=user_id, folder_name=folder_name, through_message_id=last_message))
            existing.outline_json = json.dumps(outline_sections)
            existing.total_sections = len(outline_sections)
            existing.current_section = 0
            existing.estimated_minutes = total_minutes
            existing.depth = depth
            existing.left_out_json = json.dumps(left_out_topics)
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
                depth=depth,
                left_out_json=json.dumps(left_out_topics),
            )
            db.add(existing)

        db.commit()

        if planning_units:
            from coast_content_oma.stores import make_namespace
            progressive.set_priority(make_namespace(src_uid, folder_name), outline_sections)

        if outline_via_oma and not planning_units:
            try:
                import oma_provider
                from coast_content_oma.stores import make_namespace

                ns = make_namespace(src_uid, folder_name)
                orch = oma_provider._content_orchestrator()
                page_order = oma_provider.section_page_priority(
                    outline_sections, orch, ns,
                )
                oma_provider.kickoff_background_vision_async(
                    src_uid, folder_name, page_order=page_order or None,
                )
            except Exception:
                traceback.print_exc()

        return {
            "sections": outline_sections,
            "total_sections": len(outline_sections),
            "current_section": 0,
            "estimated_minutes": total_minutes,
            "outline_source": outline_source,
            "depth": depth,
            "left_out": left_out_topics,
        }

    except HTTPException as error:
        return {"error": error.detail, "status_code": error.status_code}
    except OutlineProviderError as error:
        return {"error": str(error)}
    except Exception:
        traceback.print_exc()
        return {"error": "Outline generation failed."}
    finally:
        db.close()


def _section_pages(section: dict) -> int:
    return sum(len(r.get("pages") or []) for r in section.get("source_refs") or [])


def _cap_section_size(sections, units, system, format_line, context, course_format, skipped):
    """An essentials roadmap keeps sections their normal size: oversized sections are sent back once for the
    planner to trim (dropping less important pages) or split; any still too big are split in page order."""
    from coast_content_oma import progressive
    limit = ESSENTIALS_SECTION_PAGES + 2  # a little slack for a worked example that runs over
    big = [(i, s) for i, s in enumerate(sections) if _section_pages(s) > limit]
    if big:
        note = "; ".join(f'"{s["title"]}" has {_section_pages(s)} pages' for _, s in big)
        repaired, _, _ = _plan_outline(
            system, format_line,
            context + f"\nYour roadmap broke the size rule ({note}; at most {ESSENTIALS_SECTION_PAGES} pages each). "
            "Return the whole roadmap again: leave out the less important pages of those topics, or split a topic "
            "into two sections if all of it matters.\nPrior response: " + json.dumps(
                [{k: s.get(k) for k in ("title", "learning_objectives", "key_topics", "source_notebooks",
                                         "estimated_minutes", "source_units")} for s in sections]),
            course_format, True)
        if repaired:
            try:
                sections = progressive.bind_sections(repaired, units, skipped=skipped, fill_gaps=False)
            except ValueError:
                pass
    out = []
    for section in sections:
        pages = [(r, p) for r in section.get("source_refs") or [] for p in r.get("pages") or []]
        if len(pages) <= limit:
            out.append(section)
            continue
        parts = -(-len(pages) // ESSENTIALS_SECTION_PAGES)
        size = -(-len(pages) // parts)
        for n in range(parts):
            chunk = pages[n * size:(n + 1) * size]
            refs = {}
            for ref, page in chunk:
                r = refs.setdefault(ref["source_id"], {**{k: v for k, v in ref.items() if k not in ("pages", "text_chars")},
                                                       "pages": [], "text_chars": 0})
                r["pages"].append(page)
            part = {**section, "title": f'{section["title"]}' + (f" ({n + 1}/{parts})" if parts > 1 else ""),
                    "source_refs": list(refs.values()),
                    "estimated_minutes": max(15, round((section.get("estimated_minutes") or 20) / parts))}
            part["source_units"] = [f"{r['source_id']}:{span}" for r in part["source_refs"]
                                    for span in progressive._spans(r["pages"])]
            out.append(part)
    return out


def _left_out_topics(left_out: list[dict], sections: list[dict], units: dict, skipped: list[str]) -> list[dict]:
    """What an essentials roadmap leaves out, by topic: the planner's own list, checked against the pages
    no section teaches (and not administrative), plus any pages it left out without naming a topic."""
    def keys(refs):
        found = set()
        for ref in refs or []:
            if not isinstance(ref, str):
                continue
            if ref in units:
                found |= {(units[ref]["source_id"], p) for p in units[ref]["pages"]}
                continue
            source, _, span = ref.rpartition(":")
            first, _, last = span.partition("-")
            try:
                found |= {(source, p) for p in range(int(first), int(last or first) + 1)}
            except ValueError:
                pass
        return found
    every = {(u["source_id"], p) for u in units.values() for p in u["pages"]}
    taught = {(r["source_id"], p) for s in sections for r in s.get("source_refs") or [] for p in r.get("pages") or []}
    out = every - taught - keys(skipped)
    topics, named = [], set()
    for item in left_out or []:
        pages = (keys(item.get("source_units")) & out) - named
        if item.get("topic") and pages:
            named |= pages
            topics.append({"topic": item["topic"], "pages": len(pages),
                           "source_units": sorted({f"{s}:{p}" for s, p in pages}, key=lambda x: (x.split(":")[0], int(x.split(":")[1])))})
    rest = out - named
    if rest:
        topics.append({"topic": "Other pages", "pages": len(rest),
                       "source_units": sorted({f"{s}:{p}" for s, p in rest}, key=lambda x: (x.split(":")[0], int(x.split(":")[1])))})
    return topics


def _outline_schema(course_format: str, essentials: bool = False) -> dict:
    text, texts = {"type": "string"}, {"type": "array", "items": {"type": "string"}}
    section = {"title": text, "learning_objectives": texts, "key_topics": texts, "source_notebooks": texts,
               "estimated_minutes": {"type": "integer"}, "source_units": texts}
    if course_format == "workshop":
        section["workshop"] = {"type": "object", "additionalProperties": False,
                               "required": ["title", "outcome", "criteria", "coaching", "course_outcome"],
                               "properties": {"title": text, "outcome": text, "criteria": texts,
                                              "coaching": text, "course_outcome": text}}
    schema = {"type": "object", "additionalProperties": False, "required": ["sections", "skipped_pages"],
              "properties": {"sections": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                                     "required": list(section), "properties": section}},
                             "skipped_pages": texts}}
    if essentials:
        schema["required"].append("left_out")
        schema["properties"]["left_out"] = {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["topic", "source_units"],
            "properties": {"topic": text, "source_units": texts}}}
    return schema


def _plan_outline(system: str, format_line: str, context: str, course_format: str,
                  essentials: bool = False) -> tuple[list[dict] | None, list[str], list[dict]]:
    """(sections, skipped administrative page ranges, left-out topics). Claude plans with a guaranteed
    JSON shape; Gemini and OpenAI remain as fallbacks (they cannot skip pages or name left-out topics;
    at essentials the pages they don't use are simply left out)."""
    prompt = (system + "\nReturn the sections in `sections` and the administrative page ranges in `skipped_pages`"
              + (", and the topics you leave out in `left_out`" if essentials else "") + ".\n\n" + context)
    import openai_chat
    if not openai_chat.ANTHROPIC_FIRST:  # luna plans; Claude is the fallback
        try:
            started = time.monotonic()
            plan = openai_chat.structured(prompt, _outline_schema(course_format, essentials), effort="medium",
                                          max_tokens=32000, priority="interactive")
            logging.getLogger(__name__).info("outline provider=luna seconds=%.1f sections=%d",
                                             time.monotonic() - started, len(plan.get("sections") or []))
            if plan.get("sections"):
                return (plan["sections"], [str(r) for r in plan.get("skipped_pages") or []],
                        [t for t in plan.get("left_out") or [] if isinstance(t, dict)])
        except openai_chat.OpenAIUnavailable as exc:
            logging.getLogger(__name__).warning("luna roadmap planning failed (%s); trying Claude", exc)
    try:
        import claude_chat
        if claude_chat.available():
            started = time.monotonic()
            plan = claude_chat.structured(
                prompt, _outline_schema(course_format, essentials),
                model=os.getenv("ANTHROPIC_OUTLINE_MODEL", claude_chat.EVAL_MODEL),
                effort=os.getenv("ANTHROPIC_OUTLINE_EFFORT", "medium"), max_tokens=32000, priority="interactive")
            logging.getLogger(__name__).info("outline provider=anthropic seconds=%.1f sections=%d skipped=%s",
                                             time.monotonic() - started, len(plan.get("sections") or []),
                                             plan.get("skipped_pages"))
            if plan.get("sections"):
                return (plan["sections"], [str(r) for r in plan.get("skipped_pages") or []],
                        [t for t in plan.get("left_out") or [] if isinstance(t, dict)])
    except Exception:
        logging.getLogger(__name__).exception("Claude roadmap planning failed; trying the other providers")
    return _call_llm_for_outline(system + format_line, context), [], []


def _request_gemini_outline(system, context, max_output):
    # Use the documented REST shape here because older installed google-genai
    # versions cannot express Gemini 3's thinkingLevel setting.
    import httpx
    model=os.getenv('GEMINI_OUTLINE_MODEL','gemini-3.1-pro-preview')
    thinking={'thinkingLevel':'low'} if model.startswith('gemini-3') else {'thinkingBudget':1024}
    response=httpx.post(
        f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
        headers={'x-goog-api-key':os.environ['GEMINI_API_KEY']},
        json={'systemInstruction':{'parts':[{'text':system}]},
              'contents':[{'role':'user','parts':[{'text':context}]}],
              'generationConfig':{'responseMimeType':'application/json','maxOutputTokens':max_output,
                                  'thinkingConfig':thinking}},
        timeout=120,
    )
    if response.is_error:
        try: detail=response.json().get('error',{}).get('message','Provider request failed')
        except ValueError: detail='Provider request failed'
        raise RuntimeError(f'Gemini HTTP {response.status_code}: {detail}')
    return response.json()


def _call_llm_for_outline(system: str, context: str) -> list[dict] | None:
    """Structured roadmap generation with live-request priority and usable failures."""
    errors=[]
    max_output=min(16384,max(8192,4096+context.count('\nUNIT ')*180))
    log=logging.getLogger(__name__)
    def failure(provider,error):
        if provider_capacity.is_credit_error(error):
            detail='reports exhausted API credits or quota; check provider billing'
        elif '429' in str(error) or 'rate limit' in str(error).lower():
            detail='is rate limited; retry shortly or check the project’s model limits'
        elif isinstance(error,TimeoutError):
            detail='request queue is busy; retry shortly'
        else:
            detail='could not complete the request; retry shortly'
        errors.append(provider+' '+detail)
        log.warning('outline provider=%s failed error_type=%s',provider,type(error).__name__)

    gemini_key=os.getenv('GEMINI_API_KEY','')
    if gemini_key:
        started=time.monotonic()
        try:
            response=provider_capacity.call('gemini',lambda:_request_gemini_outline(system,context,max_output),priority='interactive')
            text=''.join(part.get('text','') for candidate in response.get('candidates',[])
                         for part in candidate.get('content',{}).get('parts',[]) if not part.get('thought'))
            parsed=_parse_json_array(text) if text.strip() else None
            log.info('outline provider=gemini seconds=%.2f chars=%d finish=%s parsed=%s',
                     time.monotonic()-started,len(text),[c.get('finishReason') for c in response.get('candidates',[])],bool(parsed))
            if parsed:
                return parsed
            errors.append('Gemini returned an incomplete or invalid roadmap')
        except Exception as error:
            failure('Gemini',error)

    if os.getenv('OPENAI_API_KEY',''):
        started=time.monotonic()
        try:
            from openai import OpenAI
            client=OpenAI(api_key=os.environ['OPENAI_API_KEY'],max_retries=0,timeout=120)
            response=provider_capacity.call('openai',lambda:client.chat.completions.create(
                model=os.getenv('OPENAI_OUTLINE_MODEL','gpt-4o-mini'),
                messages=[{'role':'system','content':system},{'role':'user','content':context}],
                max_tokens=max_output,temperature=0.3,
            ),priority='interactive')
            text=response.choices[0].message.content or ''
            parsed=_parse_json_array(text) if text.strip() else None
            log.info('outline provider=openai seconds=%.2f chars=%d parsed=%s',time.monotonic()-started,len(text),bool(parsed))
            if parsed:
                return parsed
            errors.append('OpenAI returned an incomplete or invalid roadmap')
        except Exception as error:
            failure('OpenAI',error)
    if errors:
        raise OutlineProviderError('Roadmap generation failed: '+ '; '.join(errors)+'. Your uploaded sources are saved.')
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


def _parse_json_object(text: str) -> dict | None:
    """Parse JSON object from LLM output, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                result = json.loads(text[start:end + 1])
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass
    return None


from coast_content_oma.student.grading import UI_TAG_RE as _PEDRO_UI_TAG_RE

_section_feedback_cache: "OrderedDict[tuple[int, str, int], dict]" = OrderedDict()
_FEEDBACK_CACHE_LIMIT = 2000


def _remember_feedback(key: tuple[int, str, int], feedback: dict) -> None:
    """Keep recent section feedback without letting the cache grow for the life of the process."""
    _section_feedback_cache[key] = feedback
    _section_feedback_cache.move_to_end(key)
    while len(_section_feedback_cache) > _FEEDBACK_CACHE_LIMIT:
        _section_feedback_cache.popitem(last=False)


def _strip_pedro_ui_tags(text: str) -> str:
    return _PEDRO_UI_TAG_RE.sub("", text or "").strip()


def _normalize_feedback_payload(raw: dict | None) -> dict:
    raw = raw or {}
    return {
        "strengths": [str(s).strip() for s in (raw.get("strengths") or []) if str(s).strip()],
        "weaknesses": [str(s).strip() for s in (raw.get("weaknesses") or []) if str(s).strip()],
        "tips": [str(s).strip() for s in (raw.get("tips") or []) if str(s).strip()],
    }


def _fallback_section_feedback(rows: list[ChatMessage], section_title: str) -> dict:
    user_msgs = [r for r in rows if r.role == "user" and (r.content or "").strip()]
    strengths = ["Stayed engaged through the section"]
    if len(user_msgs) >= 3:
        strengths.append("Answered multiple practice questions")
    tips = [f"Review key ideas from \"{section_title or 'this section'}\" before moving on"]
    return {"strengths": strengths, "weaknesses": [], "tips": tips}


_SECTION_REVIEW_PROMPT = (
    "You write a short end-of-section review for a student, from their tutoring conversation with Pedro, "
    "Coast's AI tutor. Address the student as \"you\" and base every point on specific moments in the conversation.\n"
    "Pedro's grading tags: [ANSWER_CORRECT: concept] means the student's previous answer was correct on their own "
    "(\"| hinted\" means after a hint); [ANSWER_WRONG: concept] an incorrect attempt; [TUTOR_CORRECTION: concept] "
    "Pedro corrected a mistake of his own, so the student's answer before it doesn't count against them; "
    "[SECTION_COMPLETE] Pedro confirmed the section.\n"
    "- strengths: 1-3 things the student showed they can do, naming the concept and the evidence.\n"
    "- weaknesses: 0-3 points that are still shaky, each with its evidence (a wrong or hinted answer). "
    "Leave it empty when there were none; never invent one.\n"
    "- tips: 1-3 concrete next steps (what to practise or revisit, with the lecture page when the conversation names one).\n"
    "One sentence per item, under 30 words. Review the student's learning only, never the tutor or the session format."
)
_SECTION_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "array", "items": {"type": "string"}} for k in ("strengths", "weaknesses", "tips")},
    "required": ["strengths", "weaknesses", "tips"],
    "additionalProperties": False,
}
_REVIEW_MAX_CHARS = 60000


def _review_transcript(rows: list[ChatMessage]) -> str:
    """The whole section conversation, grading tags kept. A very long one keeps its
    opening exchange and as much of the ending as fits, since the end holds the practice."""
    lines = []
    for row in rows:
        content = (row.content or "").strip()  # [REMEMBER]/[CLICKED] are stripped before a reply is stored
        if content:
            lines.append(f"{'Pedro' if row.role == 'pedro' else 'Student'}: {content}")
    if sum(len(line) + 1 for line in lines) <= _REVIEW_MAX_CHARS:
        return "\n".join(lines)
    head, tail, used = lines[:2], [], sum(len(line) + 1 for line in lines[:2]) + 40
    for line in reversed(lines[2:]):
        if used + len(line) + 1 > _REVIEW_MAX_CHARS:
            break
        tail.insert(0, line)
        used += len(line) + 1
    return "\n".join(head + ["(… middle of the conversation omitted …)"] + tail)


def generate_section_feedback(
    user_id: int,
    folder_name: str,
    section_index: int,
    section_title: str = "",
) -> dict:
    """Structured strengths/weaknesses/tips for a completed lesson section."""
    cache_key = (int(user_id), folder_name, int(section_index))
    if cache_key in _section_feedback_cache:
        return {"feedback": _section_feedback_cache[cache_key]}

    db = SessionLocal()
    try:
        rows = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.user_id == user_id,
                ChatMessage.context_type == "lesson",
                ChatMessage.context_id == folder_name,
                ChatMessage.section_index == int(section_index),
            )
            .order_by(ChatMessage.created_at.asc())
            .all()
        )
    finally:
        db.close()

    if not rows:
        return {"feedback": {"strengths": [], "weaknesses": [], "tips": []}}

    transcript = _review_transcript(rows)
    context = (
        f"Section {section_index + 1}: {section_title or 'Untitled'}\n\n"
        f"Conversation:\n{transcript}"
    )

    parsed = None
    if transcript:
        import openai_chat
        if not openai_chat.ANTHROPIC_FIRST:  # luna reviews; Claude is the fallback
            try:
                parsed = openai_chat.structured(_SECTION_REVIEW_PROMPT + "\n\n" + context, _SECTION_REVIEW_SCHEMA,
                                                effort="low", max_tokens=6000)
            except openai_chat.OpenAIUnavailable as exc:
                print(f"[review] luna failed ({exc}); trying Claude")
        if not parsed:
            try:
                import claude_chat
                parsed = claude_chat.structured(
                    _SECTION_REVIEW_PROMPT + "\n\n" + context, _SECTION_REVIEW_SCHEMA,
                    model=claude_chat.PEDRO_MODEL, effort="low", max_tokens=4000)
            except Exception:
                traceback.print_exc()

    if not parsed and transcript:
        system = _SECTION_REVIEW_PROMPT + (
            '\nReturn ONLY valid JSON, no markdown fences: {"strengths": [...], "weaknesses": [...], "tips": [...]}')
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_key)
                resp = provider_capacity.call('openai', lambda: client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": context},
                    ],
                    max_tokens=800,
                    temperature=0.3,
                ))
                parsed = _parse_json_object(resp.choices[0].message.content or "")
            except Exception:
                traceback.print_exc()

    feedback = _normalize_feedback_payload(parsed) if parsed else _fallback_section_feedback(rows, section_title)
    _remember_feedback(cache_key, feedback)
    return {"feedback": feedback}


def get_all_section_feedback(user_id: int, folder_name: str) -> dict:
    """Feedback cards for every section the student has chat history for."""
    db = SessionLocal()
    try:
        rows = (
            db.query(ChatMessage.section_index)
            .filter(
                ChatMessage.user_id == user_id,
                ChatMessage.context_type == "lesson",
                ChatMessage.context_id == folder_name,
            )
            .distinct()
            .all()
        )
        outline = (
            db.query(CourseOutline)
            .filter(CourseOutline.user_id == user_id, CourseOutline.folder_name == folder_name)
            .first()
        )
        sections = json.loads(outline.outline_json) if outline else []
    finally:
        db.close()

    out = []
    for (idx,) in sorted(rows, key=lambda r: r[0]):
        if idx is None:
            continue
        title = ""
        if 0 <= int(idx) < len(sections):
            title = sections[int(idx)].get("title") or ""
        result = generate_section_feedback(user_id, folder_name, int(idx), title)
        out.append({"section_index": int(idx), "feedback": result.get("feedback") or {}})
    return {"sections": out}


# Auto-generated lesson openers — not evidence the student started working.
_SECTION_OPENER = re.compile(
    r"^(i'?m ready to learn about|i'?d like to reach 100% mastery)\b",
    re.I,
)


def _is_section_opener_message(message: str | None) -> bool:
    if not message:
        return False
    try:
        import oma_provider
        return oma_provider._is_lesson_intro(message)
    except Exception:
        return bool(_SECTION_OPENER.match(message.strip()))


def _fetch_prior_section_teaching(
    user_id: int,
    folder_name: str,
    section_index: int,
    max_chars: int = 1400,
) -> str:
    """Substantive Pedro teaching from a prior section — used for section transitions."""
    db = SessionLocal()
    try:
        rows = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.user_id == user_id,
                ChatMessage.context_type == "lesson",
                ChatMessage.context_id == folder_name,
                ChatMessage.section_index == section_index,
                ChatMessage.role == "pedro",
            )
            .order_by(ChatMessage.created_at.asc())
            .all()
        )
        snippets: list[str] = []
        used = 0
        for msg in rows:
            text = _strip_pedro_ui_tags(msg.content or "")
            if len(text) < 80:
                continue
            chunk = " ".join(text.split())[:420].strip()
            if not chunk:
                continue
            line = f"- {chunk}"
            if used + len(line) > max_chars:
                break
            snippets.append(line)
            used += len(line)
        if not snippets:
            return ""
        return "\n".join(snippets[-3:])
    finally:
        db.close()


def _build_section_opening_block(
    user_id: int,
    folder_name: str,
    current_idx: int,
    sections: list,
    current: dict,
) -> str:
    """Prompt block for automatic section-start messages — dynamic transitions, no reunion filler."""
    title = current.get("title") or f"Section {current_idx + 1}"
    key_topics = current.get("key_topics") or []
    objectives = current.get("learning_objectives") or []

    block = (
        "\n--- SECTION OPENING (mandatory for your FIRST reply this section) ---\n"
        "The student's message is an automatic section-start prompt — not a real question.\n"
        "Do NOT treat it as small talk. Start teaching this section immediately after a brief transition.\n\n"
        "BANNED OPENINGS:\n"
        "- Reunion phrases: 'nice to see you again', 'great to have you back', 'welcome back'\n"
        "- Course/interest restatements: 'since you're interested in...', 'as someone studying...'\n"
        "- Re-introducing yourself or repeating the course name\n"
        "- Empty hype ('Let's dive in!' / 'Exciting section ahead!') without substance\n\n"
    )

    if current_idx <= 0:
        course_titles = [
            s.get("title") or f"Section {i + 1}"
            for i, s in enumerate(sections)
        ]
        course_topics: list[str] = []
        for sec in sections:
            course_topics.extend(str(t) for t in (sec.get("key_topics") or []) if t)

        block += (
            "THIS IS THE FIRST SECTION OF THIS COURSE:\n"
            f"- Full course ({len(sections)} sections): "
            + " → ".join(course_titles[:8])
            + ("…" if len(course_titles) > 8 else "")
            + "\n"
        )
        if course_topics:
            unique_topics = list(dict.fromkeys(course_topics))[:14]
            block += f"- Big-picture topics: {', '.join(unique_topics)}\n"
        block += (
            f"- Starting section: \"{title}\"\n"
            f"- Section key topics: {', '.join(key_topics) or '(from outline)'}\n\n"
            "REQUIRED FIRST-SECTION OPENING:\n"
            "1. In 2–3 sentences, explain what this ENTIRE course covers — the through-line "
            "from first to last section — so the student sees the big picture.\n"
            "2. If STUDENT OMA below lists learning preferences or traits, connect ONE "
            "naturally to how you'll teach this course (pace, examples, visuals, etc.). "
            "Do not invent preferences.\n"
            "3. Introduce Section 1 and start teaching the first key topic in the same reply.\n"
            "Keep steps 1–2 to at most 4 sentences total before teaching.\n"
        )
    else:
        prev = sections[current_idx - 1]
        prev_title = prev.get("title") or f"Section {current_idx}"
        prev_topics = prev.get("key_topics") or []
        prior_teaching = _fetch_prior_section_teaching(user_id, folder_name, current_idx - 1)

        block += (
            f"TRANSITION FROM SECTION {current_idx} → SECTION {current_idx + 1}:\n"
            f"- Previous section: \"{prev_title}\"\n"
            f"- Previous key topics: {', '.join(prev_topics) or '(see teaching notes below)'}\n"
            f"- Current section: \"{title}\"\n"
            f"- Current key topics: {', '.join(key_topics) or '(from outline)'}\n"
            f"- Current objectives: {', '.join(objectives) or '(from outline)'}\n"
        )
        if prior_teaching:
            block += f"\nWhat you recently taught in the previous section:\n{prior_teaching}\n"

        block += (
            "\nREQUIRED TRANSITION (adapt naturally — do not copy verbatim):\n"
            "1. Bridge: connect ONE idea from the previous section's curriculum. "
            "Only say you taught or practised it together when actual prior teaching is provided; "
            "a roadmap position or tested-out section is not evidence of a shared conversation. "
            "Otherwise say 'This builds on X' without inventing a past session.\n"
            "2. Connect: explain how this section builds on or applies that idea "
            "(e.g. 'Now we'll use X to understand Y because ...').\n"
            "3. Teach: start the first key topic of THIS section in the same reply.\n"
            "Keep steps 1–2 to at most 2–3 sentences total, then move straight into teaching.\n"
        )

    try:
        import oma_provider
        if oma_provider.is_student_enabled():
            if current_idx <= 0:
                course_oma = oma_provider.get_course_intro_student_block(
                    user_id, folder_name, sections,
                )
                if course_oma:
                    block += f"\n{course_oma}\n"
            if current_idx > 0:
                section_oma = oma_provider.get_section_intro_student_block(
                    user_id, folder_name, current_idx, key_topics,
                )
                if section_oma:
                    block += f"\n{section_oma}\n"
    except Exception:
        pass

    block += "--- END SECTION OPENING ---\n"
    return block


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
    """A stored [SECTION_COMPLETE] counts under the same rule as the live reply: never in
    a reply that also marks an answer wrong."""
    from coast_content_oma.student.grading import completes_section
    db = SessionLocal()
    try:
        rows = (
            db.query(ChatMessage.content)
            .filter(
                ChatMessage.user_id == user_id,
                ChatMessage.context_type == "lesson",
                ChatMessage.context_id == folder_name,
                ChatMessage.section_index == section_index,
                ChatMessage.role == "pedro",
                ChatMessage.content.contains("[SECTION_COMPLETE]"),
            )
            .all()
        )
        return any(completes_section(content) for (content,) in rows)
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
        from learning_jobs import enqueue
        enqueue(db, user_id, folder_name, section_index)
        db.commit()
        from learning_jobs import wake
        wake()
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
        "- Present verification questions ONE AT A TIME. Grade each answer with [ANSWER_WRONG: <concept>] or "
        "[ANSWER_CORRECT: <concept>] at the end of your message.\n"
        "- Do NOT emit [SECTION_COMPLETE] until EVERY required verification question has been "
        "answered correctly. Partial progress is not enough.\n"
        "- If the student says 'next', 'skip', 'continue', or similar WITHOUT answering, explain "
        "they must complete the remaining verification questions first. Do NOT emit [SECTION_COMPLETE].\n"
        "- If the student asks a side question during verification: answer it fully, then return "
        "to any verification questions they have NOT yet answered correctly. Never drop pending "
        "verification questions — track them mentally and re-ask until passed.\n"
        "- [SECTION_COMPLETE] is the ONLY signal that unlocks the Next Section button. The map "
        "reflects concepts you verified with [ANSWER_CORRECT: <concept>] — never skip verification.\n"
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
    """How much of this section the student has shown, from the graded answers in it: each
    concept counts by its latest answer here, 1 when right on their own (or remembered),
    ½ when right only with help, 0 when wrong. 100% means every concept tested in the
    section was last answered correctly without help. (The long-run mastery score can
    never reach 1: it starts from a neutral prior, so every finished section read as
    "worth a review".) Marks Pedro withdrew as his own error don't count."""
    latest: dict[str, float] = {}
    graded = [ep for ep in episodes if (ep.store_specific or {}).get("episode_type") == "exercise_attempt"
              and not ((ep.store_specific or {}).get("signals") or {}).get("tutor_error")]
    graded.sort(key=lambda ep: (ep.created_at or "", (ep.store_specific or {}).get("seq") or 0))
    for ep in graded:
        ss = ep.store_specific or {}
        outcome = ss.get("outcome")
        if outcome not in ("success", "mistake", "struggle"):
            continue
        credit = 0.0 if outcome != "success" else 0.5 if ss.get("hinted") else 1.0
        for key in (ss.get("concept_ids") or ([ss["concept_label"].lower()] if ss.get("concept_label") else [])):
            latest[key] = credit
    if latest:
        return round(100 * sum(latest.values()) / len(latest))

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
        if sec.get('preparation_version') == 1:
            from coast_content_oma import progressive
            return progressive.section_concept_refs(content_orch,content_ns,sec)
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
            if pct >= 100 and not (_section_is_finished(i, current_section, all_episodes)
                                   or is_section_verified(user_id, folder_name, i)):
                # 100% reads as "done" on the roadmap and unlocks map rewards: only a finished section
                # may show it, not the current one after its first right answers.
                pct = 99
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

        src_uid = source_user_id if source_user_id is not None else user_id

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
                "format": folder_kind(user_id, folder_name),
                "content_ready": content_ready,
                "shared_content_ready": shared_content_ready,
            }

        sections = decorate_sections(folder_name, json.loads(outline.outline_json))
        current_preparation = sections[min(int(outline.current_section),len(sections)-1)] if sections else {}
        if not is_curated and current_preparation.get('preparation_version') != 1:
            try:
                import oma_provider
                if oma_provider.is_oma_enabled():
                    content_ready = oma_provider.is_section_content_ready(
                        src_uid,
                        folder_name,
                        int(outline.current_section),
                        outline_sections=sections,
                    )
            except Exception:
                pass

        preparation = None
        if sections and sections[min(int(outline.current_section),len(sections)-1)].get('preparation_version') == 1:
            import oma_provider
            from coast_content_oma import progressive
            from coast_content_oma.stores import make_namespace
            ns = make_namespace(src_uid,folder_name)
            progressive.set_priority(ns,sections,int(outline.current_section))
            preparation = progressive.status_for_section(src_uid,folder_name,sections[min(int(outline.current_section),len(sections)-1)])
            content_ready = preparation['ready']
            if content_ready and all(source.oma_ingest_status in ('CONTENT_INDEXED','READY_FOR_ROADMAP','COMPLETE')
                                     for source in progressive._sources(src_uid,folder_name)):
                oma_provider.kickoff_background_concept_refinement_async(src_uid,folder_name)

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
            "source_references": source_catalog(db, user_id, folder_name) if not is_curated else [],
            "section_preparation": preparation,
            "content_ready": content_ready,
            "shared_content_ready": shared_content_ready,
            "sections": sections,
            "format": "workshop" if is_workshop(sections) or folder_kind(user_id, folder_name) == "workshop" else "lesson",
            "total_sections": outline.total_sections,
            "current_section": outline.current_section,
            "estimated_minutes": outline.estimated_minutes,
            "depth": getattr(outline, "depth", None),
            "left_out": json.loads(getattr(outline, "left_out_json", None) or "[]"),
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

        sections = decorate_sections(folder_name, json.loads(outline.outline_json))
        finished_title = ""
        if outline.current_section < outline.total_sections:
            finished_idx = int(outline.current_section)
            if finished_idx < len(sections):
                sec = sections[finished_idx]
                finished_title = sec.get("title") or f"Section {finished_idx + 1}"
                from learning_jobs import enqueue
                enqueue(db, user_id, folder_name, finished_idx, finished_title)

            outline.current_section += 1
            outline.updated_at = datetime.now(timezone.utc)
            db.query(SectionVerification).filter_by(user_id=user_id, folder_name=folder_name, section_index=finished_idx).delete()
            db.commit()

            try:
                import oma_provider
                if oma_provider.is_oma_enabled():
                    from coast_content_oma.stores import make_namespace

                    ns = make_namespace(user_id, folder_name)
                    orch = oma_provider._content_orchestrator()
                    remaining = sections[int(outline.current_section):]
                    page_order = oma_provider.section_page_priority(remaining, orch, ns)
                    oma_provider.kickoff_background_vision_async(
                        user_id, folder_name, page_order=page_order or None,
                    )
            except Exception:
                traceback.print_exc()

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
            "sections": sections,
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
    conversation_id: str | None = None,
) -> str | None:
    """Adaptive placement: check the skipped sections one at a time, in order."""
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

        sections = decorate_sections(folder_name, json.loads(outline.outline_json))
        current_idx = int(outline.current_section)
        target_idx = int(target_section_index)
        if target_idx <= current_idx or target_idx > len(sections):
            return None

        import placement
        progress = (placement.state(user_id, conversation_id) if conversation_id else None) or {}
        checking = progress.get("checking_section", current_idx)
        passed = progress.get("passed_count", 0)
        goal = "the end of the course" if target_idx >= len(sections) else \
            f"Section {target_idx + 1}: \"{sections[target_idx].get('title') or ''}\""

        lines = []
        for i in range(current_idx, target_idx):
            sec = sections[i]
            mark = "passed" if i < current_idx + passed else ("CHECKING NOW" if i == checking else "later")
            lines.append(f"  - Section {i + 1} [{mark}]: {sec.get('title') or ''} — "
                         f"topics: {', '.join(sec.get('key_topics') or []) or '(general)'}")

        if checking is None:
            place_at = progress.get('place_at', current_idx) + 1
            return (
                "You are Pedro. The placement check is finished. Congratulate the student briefly on what they "
                f"demonstrated and tell them they now start at Section {place_at}: the button below takes them "
                "there. Answer any question they have, but do not ask further test questions and do not start "
                f"teaching Section {place_at} here; teaching happens in the lesson.\n"
            )

        sec = sections[checking]
        focus = ("What to check: the section's core idea — can they explain or apply it without help?\n"
                 f"Objectives: {'; '.join(sec.get('learning_objectives') or sec.get('objectives') or []) or '(see topics)'}")

        prompt = (
            "You are Pedro, running an ADAPTIVE PLACEMENT CHECK. The student says they already know some of this "
            f"course and wants to skip ahead towards {goal}.\n\n"
            "SECTIONS IN THIS CHECK:\n" + "\n".join(lines) + "\n\n"
            f"CHECKING NOW: Section {checking + 1}: \"{sec.get('title') or ''}\"\n{focus}\n\n"
            "HOW TO RUN IT:\n"
            "- On the very first message, say in one sentence that you'll check one section at a time and they "
            "can stop whenever they like; then ask your first question.\n"
            "- Ask ONE targeted question about the section being checked — the idea someone who really knows it "
            "could answer quickly, not trivia. Keep replies to 1–3 sentences plus the question.\n"
            "- Grade every answer: [ANSWER_CORRECT: <concept>] or [ANSWER_WRONG: <concept>]; add \" | hinted\" "
            "to a correct answer only if you helped.\n"
            f"- Correct and confident → confirm in one sentence and end the reply with both tags, "
            f"[ANSWER_CORRECT: <concept>] and then [PLACEMENT_PASSED: {checking + 1}]. "
            + (f"Then ask a question about Section {checking + 2}.\n" if checking + 1 < target_idx else
               f"This is the last section to check: ask nothing more. Tell them they have placed into {goal} and "
               "that the button below takes them there. Never start teaching a new section in this check.\n")
            + "- Partly right or unsure → ask exactly ONE follow-up on the same section.\n"
            f"- Still not demonstrated (or they say they don't know it) → be encouraging, emit [PLACEMENT_STOP] and "
            f"tell them they'll start at Section {checking + 1}. Do not teach the section now.\n"
            "- Never emit [PLACEMENT_PASSED] for a section you have not checked, or without a correct answer.\n"
            "- Never emit [SECTION_COMPLETE] in a placement check.\n\n"
        )
        query = " ".join(dict.fromkeys([sec.get("title") or ""] + list(sec.get("key_topics") or [])
                                       + list(sec.get("learning_objectives") or [])))
        material_context, _ = _fetch_section_material(src_uid, folder_name, query, max_chars=12000)
        if material_context:
            prompt += (
                "--- REFERENCE MATERIAL for the section being checked (question design only — do not lecture) ---\n"
                + material_context
                + "\n--- END REFERENCE MATERIAL ---\n"
            )
        return prompt
    except Exception:
        traceback.print_exc()
        return None
    finally:
        db.close()


def apply_test_out(user_id: int, folder_name: str, target_section_index: int, conversation_id: str | None = None) -> dict:
    """Jump the student to target_section after passing a placement test."""
    db = SessionLocal()
    try:
        from sqlalchemy import text
        db.execute(text("BEGIN IMMEDIATE"))
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

        import placement
        try:
            permit = placement.authorize(db, user_id, folder_name, target_idx, conversation_id, outline)
        except ValueError as exc:
            return {"error": str(exc)}
        if permit.consumed:
            return {"current_section": current_idx, "total_sections": outline.total_sections,
                    "is_complete": current_idx >= outline.total_sections,
                    "ever_mastered": bool(outline.ever_mastered), "sections": sections, "skipped_sections": []}
        if target_idx <= current_idx:
            return {"error": "Section already unlocked"}
        if target_idx > len(sections):
            return {"error": "Invalid section index"}
        # Place the student at the first section they did not demonstrate.
        place_idx = min(target_idx, current_idx + int(permit.passed_count or 0))

        for i in range(current_idx, place_idx):
            from database import SectionVerification
            verification = db.get(SectionVerification, (user_id, folder_name, i))
            if verification is None:
                verification = SectionVerification(user_id=user_id, folder_name=folder_name, section_index=i)
                db.add(verification)
            verification.is_active = True
            verification.verified_at = datetime.now(timezone.utc)
            sec = sections[i]
            title = sec.get("title") or f"Section {i + 1}"
            from learning_jobs import enqueue
            enqueue(db, user_id, folder_name, i, title, completion_type="tested_out")

        permit.consumed = True
        permit.done = True
        outline.current_section = place_idx
        outline.updated_at = datetime.now(timezone.utc)
        if place_idx >= outline.total_sections:
            _sync_ever_mastered(outline)
        db.commit()

        return {
            "current_section": place_idx,
            "total_sections": outline.total_sections,
            "is_complete": place_idx >= outline.total_sections,
            "ever_mastered": bool(getattr(outline, "ever_mastered", False)),
            "sections": sections,
            "skipped_sections": list(range(current_idx, place_idx)),
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
    """Content OMA when enabled, otherwise Chroma RAG."""
    try:
        import oma_provider
        if oma_provider.is_oma_enabled():
            block, source, _ = oma_provider.resolve_folder_content(
                user_id, folder_name, query,
                context_type="lesson",
                max_chars=max_chars,
                max_content=12,
                max_images=3,
            )
            return block, source == "OMA"
        import rag
        block = rag.build_folder_context(user_id, folder_name, query, max_chars=max_chars)
        if block:
            oma_provider.log_content_source(
                "RAG",
                context_type="lesson",
                folder=folder_name,
                user_id=user_id,
                chars=len(block),
            )
        return block, False
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
    student_message: when set, also retrieve RAG material targeted at the
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

        sections = decorate_sections(folder_name, json.loads(outline.outline_json))
        current_idx = int(section_index if section_index is not None else outline.current_section)
        if current_idx >= len(sections):
            current_idx = max(0, len(sections) - 1)

        current = sections[current_idx]
        workshop = current.get("workshop")
        section_title = current.get("title", f"Section {current_idx + 1}")
        objectives = workshop["criteria"] if workshop else current.get("learning_objectives", [])
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
        if current.get('preparation_version') == 1 and not recap:
            import oma_provider
            from coast_content_oma import progressive
            from coast_content_oma.stores import make_namespace
            material_context = progressive.section_context(oma_provider._content_orchestrator(), make_namespace(src_uid,folder_name), current)
            used_oma = True
        else:
            material_context, used_oma = _fetch_section_material(
                src_uid, folder_name, query, max_chars=max_material,
            )

        if student_message:
            try:
                import oma_provider
                if not oma_provider._is_lesson_intro(student_message):
                    if oma_provider.is_oma_enabled():
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
                    else:
                        import rag
                        extra = rag.build_folder_context(
                            src_uid, folder_name, student_message,
                            max_chars=12000 if recap else 8000,
                        )
                        if extra and extra not in material_context:
                            material_context += (
                                "\n\n--- ADDITIONAL MATERIAL FOR THIS QUESTION ---\n"
                                + extra
                            )
                            oma_provider.log_content_source(
                                "RAG",
                                context_type="lesson-question",
                                folder=folder_name,
                                user_id=src_uid,
                                chars=len(extra),
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
            f"CURRENT SECTION: {workshop['title'] if workshop else section_title}\n"
            f"Learning objectives: {', '.join(objectives)}\n"
            f"Key topics to cover: {', '.join(objectives if workshop else key_topics)}\n"
        )
        if workshop:
            prompt += workshop_instructions(workshop)
            prompt += workshop_gate(is_section_verified(user_id, folder_name, current_idx))
            prompt += prior_work(db, user_id, folder_name, current_idx)
        else:
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

        if not workshop and _is_section_opener_message(student_message):
            prompt += _build_section_opening_block(
                user_id, folder_name, current_idx, sections, current,
            )

        if recap:
            prompt += (
                "\nRECAP / SUMMARY MODE:\n"
                "The student wants a summary of what they've covered in this course. "
                "Use ALL of the following — do NOT say you lack their lecture slides:\n"
                "1) SOURCE MATERIAL below (from their uploaded PDFs)\n"
                "2) RECENT LESSON CONVERSATIONS below (what you already taught them)\n"
                "3) STUDENT PROFILE (sections finished, mastery, accomplishments)\n"
                "Organize by section or topic. Be specific to THEIR course materials.\n"
            )
            recap_block = _fetch_lesson_conversation_recap(user_id, folder_name)
            if recap_block:
                prompt += "\n" + recap_block

        prompt += (
            "\nTEACHING APPROACH: Adapt the amount of explanation to recorded OMA evidence and the current attempt. "
            "For a beginner, model one small example before their own activity. For experienced students, "
            "ask them to demonstrate the milestone directly; do not add harder exam questions. "
            "Repair a specific gap with a hint or different explanation. "
            "Prior completion is not proof of current ability; neither infer struggles nor preferences without evidence.\n"
        ) if workshop else (
            "\nTEACHING APPROACH:\n"
            "Adapt your starting point to the STUDENT PROFILE block (which appears earlier in this "
            "system prompt). The profile records this student's prior interactions with this course's "
            "concepts.\n"
            "- If the profile is empty or shows no prior coverage of the concepts in this section, "
            "assume the student is a COMPLETE BEGINNER and teach this section from scratch.\n"
            "- If the profile shows the student has already mastered the concepts in this section "
            "(the relevant evidence summary shows strong/resolved coverage), treat this as a REVIEW: "
            "give a 2–3 sentence recap TOTAL, then move directly to harder practice "
            "problems and exam-style questions. Do NOT re-teach material they've already mastered.\n"
            "- If the profile shows the student is BORDERLINE on some concepts (mid mastery) and "
            "WEAK on others (especially ACTIVE_MISCONCEPTION or UNDER_OBSERVATION), focus on the weak and borderline "
            "concepts. Skim the mastered ones with a quick 'as you already know, ...' framing.\n"
            "- The STUDENT PROFILE describes prior evidence, not guaranteed current understanding. "
            "Use the student's current answer to update your teaching; unassessed concepts remain unknown.\n"
            "- NEVER tell the student they said something was 'tricky', 'hard', or 'confusing' "
            "unless the profile's Open questions or Unresolved fields explicitly say so. "
            "Mastery scores alone do not mean the student complained — 'Has struggled with' "
            "means you corrected them on a quiz, not that they self-reported difficulty.\n\n"

        )

        if not workshop:
            prompt += (
                "INSTRUCTIONS:\n"
                "1. You MUST teach from the source material provided below. Do NOT say you don't have it — it is included below.\n"
                "2. For concepts NOT yet mastered (per STUDENT PROFILE): define every term, explain "
                "from the ground up, never assume prior knowledge. For concepts the profile shows as "
                "mastered: reference them briefly and confirm with a quick check-question.\n"
                "3. Build knowledge step by step: foundations first, then build up to more complex ideas. "
                "Use clear, direct explanations with concrete examples. Only use analogies when a concept is truly abstract "
                "and hard to grasp without one — most of the time, a detailed explanation with a worked example is more helpful.\n"
                "4. When the CURRENT teaching step introduces a formula or equation, explain WHAT each variable means, "
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
                "6. After each coherent teaching step, ask ONE diagnostic comprehension question, then wait. "
                "It must be answerable from what you just taught. Cover the rest of the section across subsequent turns.\n"
                "7. If the student struggles, don't just repeat yourself. Re-explain using a different angle, "
                "simpler language, a concrete example, or break it into even smaller steps. "
                "Stay on that concept: ask a fresh probe and wait for the student's demonstration before teaching the next dependency.\n"
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
                "Then include the tag [ANSWER_WRONG: <concept>] at the very end of your message (after your explanation). "
                "This logs a practice mistake for review — it does NOT mean the student is struggling; "
                "you will re-teach and verify understanding.\n"
                "   f) When the student answers a practice problem correctly, include [ANSWER_CORRECT: <concept>] at the end "
                "of that message.\n"
                "   g) Only use [ANSWER_WRONG: <concept>] / [ANSWER_CORRECT: <concept>] when grading a student answer attempt — never during "
                "teaching, explanations, or when the student has not submitted an answer.\n"
                "   h) Attempts alone do not verify understanding. Require successful independent practice after any hints or worked answers.\n"
                "9. When the student has completed the practice round successfully, congratulate them, summarize what they learned, "
                "note any weak areas they should revisit, and ask if they're ready for the next section.\n"
                "10. NEVER claim you don't have access to the student's uploaded documents. The relevant content is provided below.\n"
                "11. Emit [SECTION_COMPLETE] ONLY when ALL verification questions in this section were answered correctly "
                "(each graded with [ANSWER_CORRECT: <concept>]). Never because the student said 'next', 'skip', or 'continue'. "
                "If they ask a follow-up question after passing some checks, teach them, then re-ask any they missed.\n"
                "12. When you believe the student has understood the section AND completed the practice round, include the exact phrase "
                '"[SECTION_COMPLETE]" at the end of your message (the frontend uses this to show a Next Section button).\n\n'
            )

        if not workshop and structure and structure.get("pedagogy"):
            prompt += (
                "--- SPECIAL TEACHING METHODOLOGY FOR THIS COURSE ---\n"
                + structure["pedagogy"]
                + "\n--- END METHODOLOGY ---\n\n"
            )

        prompt += (
            "\nPACING AND SOURCE FIDELITY:\n"
            "For a workshop, follow the action/evidence contract above rather than an exam-question sequence. "
            "Use one brief explanation and one student action per turn; expand when they need depth. "
            "Cite source pages for factual explanations when provided. "
        ) if workshop else (
            "\nPACING AND SOURCE FIDELITY:\n"
            "Preserve the full depth of every learning objective across the conversation, not in one opening lecture. "
            "For an ordinary teaching turn, aim for roughly 150–300 words on ONE coherent step, one relevant "
            "source diagram or worked example, and ONE question. Use more space when a complete derivation "
            "or the student's request needs it; never truncate a necessary explanation. "
            "A section opener needs only a short orientation followed by the first step. "
            "Follow the profile's Teaching stance when present. REVIEW means 2–3 recap sentences total "
            "and ONE challenging transfer question, without its solution. REPAIR means a minimal contrast "
            "targeting that misconception and ONE short probe; defer the full algorithm, lengthy worked example, "
            "and next topic until the distinction is demonstrated.\n"
            "When a student answers incorrectly, correct and re-probe that point before introducing new material. "
            "A hint request is not an answer attempt: reveal only the next useful step, not the final answer. "
            "An assisted correct answer is progress, not proof of independent mastery. "
            "Do not complete the section while an objective or required practice remains unchecked.\n"
        )
        prompt += (
            "Cite the provided source title and page beside each diagram and substantive teaching step. "
            "Label each practice question as 'Source exercise', 'Adapted from [source, page]', or "
            "'Pedro practice — based on [source, page]'. Only call it a source exercise if its question "
            "actually occurs in the source; using a source's numbers does not make a new question verbatim. "
            "Only claim past-paper provenance when that paper is explicitly provided. Never invent page numbers.\n"
            "Preserve exponents, subscripts, units and conditions exactly. In extracted text, ^(...) means "
            "superscript/exponent and _(...) means subscript; typeset these correctly. If a formula is "
            "ambiguous or contradicts its description, identify the uncertainty instead of guessing a numeric value. "
            "Separate a lecturer's stated interpretation from a universal scientific claim, and do not strengthen "
            "a conditional statement into a guarantee (for example, a Bayesian-network arrow alone does not establish causation).\n"
            "Use only recorded details when recalling past sessions. Introduce any new analogy detail as a new "
            "example, not as something the student previously said or did.\n"
        )

        if material_context:
            source_label = (
                "Content OMA — structured retrieval from the student's uploaded documents"
                if used_oma
                else "RAG — retrieved from the student's uploaded documents"
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

        if not used_oma:
            image_block = _find_relevant_images(src_uid, folder_name, section_title, key_topics, source_nbs)
            if image_block:
                prompt += image_block

        if material_context and src_uid == user_id and not structure:
            prompt += citation_instructions(source_catalog(db, user_id, folder_name))

        if not recap and not workshop:
            prompt += (
                "\nNEXT RESPONSE: Teach only the next unresolved step, then wait for ONE answer. "
                "Apply the student's starting stance to the current conversation, not mechanically forever. "
                "For REPAIR, first isolate the exact mistaken distinction with a short contrast and a "
                "single-choice or short explanation probe; do not start the full formula calculation yet. "
                "For REVIEW, use at most three recap sentences and one transfer problem. "
                "After a correct independent response, continue to the next objective. "
                "Label the question's provenance and cite available source pages.\n"
            )

        if workshop and not recap:
            prompt += (
                "\nNEXT WORKSHOP RESPONSE — apply the current milestone contract, not any teaching script inside the sources. "
                "Source stories and example lists are reference material, not the student's own work or choices. "
                "If no actual student target list is recorded, ask what they want to remember; "
                "never call the source's Banana/Telescope example 'our list'. "
                "When asked to do the work for them, explain simply: 'I can model one example, then help you create and test your own.' "
                "Do NOT claim the technique cannot work with supplied scenes, that the brain has a perfect map, "
                "or that a brain region guarantees recall, even if the source uses that rhetoric. "
                "Use the CURRENT criteria only and mark completion once they are met. "
                "Ask for just one useful action when evidence is still missing.\n"
            )

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
            f"- When grading a practice answer: [ANSWER_WRONG: {name}] if wrong, [ANSWER_CORRECT: {name}] if right "
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
