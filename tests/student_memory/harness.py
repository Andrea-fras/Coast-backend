"""Isolated local harness for Student OMA / Pedro memory tests.

Importing this module BEFORE any Coast module:
  * points coast.db and oma.db at a fresh temp directory (never your real data),
  * blanks every provider API key so nothing can reach the network,
  * replaces Pedro's LLM with a scripted fake that also captures the prompt.

Tests drive the real code paths (tutor.send_message_stream, learning_jobs,
oma_provider) and only stub the model output.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
TMP = Path(tempfile.mkdtemp(prefix="coast_memtest_")).resolve()

os.environ.update({
    "DATABASE_PATH": str(TMP / "coast.db"),
    "OMA_DB_PATH": str(TMP / "oma.db"),
    "OMA_IMAGE_DIR": str(TMP / "images"),
    "FOLDER_UPLOADS_DIR": str(TMP / "uploads"),
    "GENERATED_DIR": str(TMP / "generated"),
    "CHROMA_PATH": str(TMP / "chroma"),
    "RAG_PROVIDER": "oma",
    "STUDENT_OMA_ENABLED": "true",
    "PEDRO_PROVIDER": "openai",
    "OMA_SKIP_IMAGES": "true",
    "OMA_DESCRIBE_IMAGES": "false",
    "OMA_ALIAS_LEDGER_ENABLED": "true",
})
# Empty (not unset) so load_dotenv() elsewhere cannot re-populate real keys.
for _key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "KIMI_API_KEY", "RESEND_API_KEY"):
    os.environ[_key] = ""
os.environ.pop("RENDER", None)

sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import database  # noqa: E402

assert str(Path(database.DB_PATH).resolve()).startswith(str(TMP)), f"refusing to run against {database.DB_PATH}"
database.init_db()

import tutor  # noqa: E402
import oma_provider  # noqa: E402
import lesson  # noqa: E402,F401  (re-exported for tests)

assert str(oma_provider.OMA_DB_PATH).startswith(str(TMP)), f"refusing to run against {oma_provider.OMA_DB_PATH}"

from database import SessionLocal, User, CourseOutline, ChatMessage  # noqa: E402
from coast_content_oma import course_identity  # noqa: E402
from coast_content_oma.stores import make_namespace  # noqa: E402
from coast_content_oma.student.stores import course_namespace, identity_namespace  # noqa: E402

CONCEPT_ID_RE = re.compile(r"\bcon_\d{8}_\d{6}_[0-9a-f]{6}\b")
_ids = itertools.count(1000)


# ── Fake Pedro ───────────────────────────────────────────────────────

class FakePedro:
    """Scripted replacement for the chat model. Records every prompt it sees."""

    def __init__(self):
        self.prompts: list[list[dict]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._next_reply = "OK."

    def _create(self, *, messages, stream=False, **_):
        if not stream:  # conversation summaries etc.
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="(summary)"))])
        self.prompts.append(messages)
        reply = self._next_reply
        chunks = [reply[i:i + 40] for i in range(0, len(reply), 40)] or [""]
        return iter(SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=c))]) for c in chunks)

    @property
    def last_system_prompt(self) -> str:
        """The instructions Pedro saw for the last turn: the system messages, plus the final
        student message, which carries Coast's per-turn note in the v2 context."""
        if not self.prompts:
            return ""
        messages = self.prompts[-1]
        parts = [m["content"] for m in messages if m["role"] == "system"]
        if messages and messages[-1]["role"] == "user":
            parts.append(messages[-1]["content"])
        return "\n".join(parts)


import evaluator  # noqa: E402

ORIGINAL_EVALUATE = evaluator.evaluate_section_transcript  # stub_evaluator() replaces it per test

PEDRO = FakePedro()
tutor.CHAT_PROVIDER = "openai"
tutor._get_client = lambda provider="openai": (PEDRO, "fake-pedro")


# ── Fixtures ─────────────────────────────────────────────────────────

def make_student(name: str = "Sam") -> int:
    with SessionLocal() as db:
        n = next(_ids)
        user = User(email=f"student{n}@test.local", name=name, onboarding_completed=True)
        db.add(user)
        db.commit()
        return user.id


def make_course(user_id: int, folder: str, sections: list[dict]) -> dict[str, str]:
    """Create an outline and a Content OMA concept graph for it.

    sections: [{"title": ..., "key_topics": [concept names]}]
    Returns {concept name (lower): concept_id}.
    """
    outline = [{
        "title": s["title"],
        "key_topics": s["key_topics"],
        "objectives": s.get("objectives", [f"Understand {t}" for t in s["key_topics"]]),
        "estimated_minutes": 20,
        **({"workshop": s["workshop"]} if "workshop" in s else {}),
    } for s in sections]
    with SessionLocal() as db:
        course_identity.register(db, user_id, folder)
        db.add(CourseOutline(user_id=user_id, folder_name=folder, outline_json=json.dumps(outline),
                             total_sections=len(outline), current_section=0))
        db.commit()
    orch = oma_provider._content_orchestrator()
    ns = make_namespace(user_id, folder)
    ids = {}
    for s in sections:
        for name in s["key_topics"]:
            if name.lower() in ids:
                continue
            item = orch.concept.write(ns, f"{name}: definition of {name}.", entities=[name],
                                      store_specific={"name": name.lower(), "aliases": [], "definition": f"Definition of {name}."})
            ids[name.lower()] = item.id
    return ids


def set_current_section(user_id: int, folder: str, index: int) -> None:
    with SessionLocal() as db:
        row = db.query(CourseOutline).filter_by(user_id=user_id, folder_name=folder).first()
        row.current_section = index
        db.commit()


def chat(user_id: int, message: str, reply: str, *, context_type: str = "lesson",
         context_id: str | None = None, section_index: int | None = None,
         conversation_id: str | None = None) -> dict:
    """Run one real Pedro turn with a scripted reply. Returns the final result
    dict plus the system prompt Pedro saw (result['system_prompt'])."""
    PEDRO._next_reply = reply
    result = None
    for _token, final in tutor.send_message_stream(
        user_id, message, conversation_id, context_type,
        context_id=context_id, section_index=section_index,
    ):
        if final is not None:
            result = final
    oma_provider.flush_student_writes()
    result = dict(result or {})
    result["system_prompt"] = PEDRO.last_system_prompt
    return result


def stub_evaluator(verdicts: dict[str, str] | None = None, open_questions=None, golden=None, work_summary=""):
    """Make the section evaluator deterministic (it normally calls an LLM).

    verdicts: {concept name (lower): final_state}
    """
    import evaluator

    def fake(transcript, section_index, section_title, concept_refs, *, require_model=False, workshop=None):
        raw = {
            "work_summary": work_summary if workshop else "",
            "section_summary": f"Stubbed evaluation of {section_title}",
            "concepts": [{
                "concept_id": c["concept_id"], "concept_name": c["concept_name"],
                "final_state": (verdicts or {}).get((c["concept_name"] or "").lower(), "not_touched"),
                "note": "stub",
            } for c in concept_refs],
            "golden_moments": golden or [],
            "open_questions": open_questions or [],
        }
        return evaluator._normalize_evaluation(raw, concept_refs)

    evaluator.evaluate_section_transcript = fake


def run_learning_jobs(max_jobs: int = 20) -> int:
    import learning_jobs
    n = 0
    while n < max_jobs and learning_jobs.run_one(kind="completion"):
        n += 1
    return n


def job_rows() -> list:
    from database import LearningJob
    with SessionLocal() as db:
        return [(j.status, j.attempts, j.last_error) for j in db.query(LearningJob).all()]


# ── Student OMA inspection ───────────────────────────────────────────

def student():
    return oma_provider._student_orchestrator()


def episodes(user_id: int, folder: str | None = None) -> list:
    """All episodes for a student (optionally one course), oldest first."""
    if folder is not None:
        return student().episodes.all(course_namespace(user_id, folder))
    from coast_content_oma.stores.db import connect_db
    from coast_content_oma.stores._semantic_base import _row_to_item
    with connect_db(student().episodes.db_path) as conn:
        rows = conn.execute("SELECT * FROM episode_items ORDER BY created_at").fetchall()
    return [_row_to_item(r, "episode") for r in rows if r[1].startswith(f"u{user_id}__")]


def seed_answers(user_id: int, folder: str, concept_name: str, concept_id: str, outcomes: list[bool],
                 section_index: int = 0) -> None:
    """Write graded answers straight into the episode log (the source of truth),
    independent of Pedro's tag format — for tests of the rules built on top of it."""
    rec = oma_provider._student_recorder_singleton()
    for ok in outcomes:
        rec.record_episode(user_id, folder, "exercise_attempt", summary=f"answer on {concept_name}",
                           outcome="success" if ok else "mistake", section_index=section_index,
                           concept_refs=[{"concept_id": concept_id, "concept_name": concept_name}])


def mastery(user_id: int, folder: str, concept_id: str) -> dict | None:
    item = student().mastery.for_concept(course_namespace(user_id, folder), concept_id)
    return dict(item.store_specific) if item else None


def mastery_rows(user_id: int, folder: str) -> dict[str, dict]:
    return {(it.store_specific or {}).get("concept_name", "").lower(): dict(it.store_specific or {})
            for it in student().mastery.all(course_namespace(user_id, folder))}


def identity_traits(user_id: int) -> list[str]:
    return [it.content for it in student().identity.all(identity_namespace(user_id))]


def consolidate(user_id: int, folder: str) -> None:
    oma_provider._run_course_consolidation(user_id, folder)


def course_block(user_id: int, folder: str, query: str = "", concept_ids=None) -> str:
    return oma_provider.get_student_profile_block(user_id, folder, current_concept_ids=concept_ids, query=query)


def global_block(user_id: int, query: str = "") -> str:
    return oma_provider.get_global_student_profile_block(user_id, query=query)


def intro_block(user_id: int, folder: str) -> str:
    return oma_provider.get_course_intro_student_block(user_id, folder, [])


def pedro_messages(user_id: int, folder: str, section_index: int | None = None) -> list:
    with SessionLocal() as db:
        q = db.query(ChatMessage).filter_by(user_id=user_id, context_id=folder)
        if section_index is not None:
            q = q.filter_by(section_index=section_index)
        return q.order_by(ChatMessage.id).all()
