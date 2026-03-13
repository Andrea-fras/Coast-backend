"""FastAPI server for Coast — auth, notebooks, sessions, and OCR pipeline."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()

from auth import create_access_token, decode_access_token, hash_password, verify_password
from database import (
    ChatMessage,
    FolderSource,
    Paper,
    QuizSession,
    SavedNotebook,
    SessionAnswer,
    SessionLocal,
    SkillProfile,
    StudyFolder,
    TutorMemo,
    User,
    init_db,
    load_papers_from_json,
)
import tutor
import spaced_rep
import rag
import lesson
from viz_router import viz_router

app = FastAPI(title="Coast API", version="2.0.0")
app.include_router(viz_router)

_cors_origins = ["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"]
_extra_origin = os.getenv("FRONTEND_URL", "")
if _extra_origin:
    _cors_origins.append(_extra_origin.rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from starlette.requests import Request as StarletteRequest

_cors_headers = {
    "Access-Control-Allow-Origin": _extra_origin.rstrip("/") if _extra_origin else "*",
    "Access-Control-Allow-Credentials": "true",
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
    "Access-Control-Allow-Headers": "Authorization, Content-Type",
}

@app.exception_handler(HTTPException)
async def http_exception_handler(request: StarletteRequest, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=_cors_headers,
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: StarletteRequest, exc: Exception):
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
        headers=_cors_headers,
    )

GENERATED_DIR = Path(__file__).parent / "generated"
GENERATED_DIR.mkdir(exist_ok=True)

FOLDER_UPLOADS_DIR = Path(os.environ.get("FOLDER_UPLOADS_DIR", str(Path(__file__).parent / "folder_uploads")))
FOLDER_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

PAPERS_DIR = Path(os.getenv("PAPERS_DIR", str(Path(__file__).parent.parent / "Coast" / "testing" / "src" / "data")))

QUESTIONS_PER_BATCH = 10

RATE_LIMIT_CHAT_MESSAGES = 500      # max Pedro messages per user per week
RATE_LIMIT_NOTEBOOKS = 20           # max notebook uploads per user
RATE_LIMIT_WINDOW_DAYS = 7          # rolling window for chat limit


# ═══════════════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
def on_startup():
    init_db()
    if PAPERS_DIR.exists():
        load_papers_from_json(PAPERS_DIR)
        print(f"Loaded papers from {PAPERS_DIR}")


# ═══════════════════════════════════════════════════════════════════════════
# AUTH DEPENDENCY
# ═══════════════════════════════════════════════════════════════════════════

def get_current_user(authorization: Optional[str] = Header(None)) -> User:
    """Extract and verify the JWT from the Authorization header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Not authenticated")

    token = authorization.split(" ", 1)[1]
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(401, "Invalid or expired token")

    db = SessionLocal()
    user = db.query(User).filter(User.id == int(payload["sub"])).first()
    db.close()

    if not user:
        raise HTTPException(401, "User not found")
    return user


ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "andreaf.fraschetti@gmail.com")


def _get_user_usage(user_id: int):
    """Return current usage counts for rate limiting. Admin gets unlimited."""
    from datetime import timedelta

    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.id == user_id).first()
        if admin and admin.email == ADMIN_EMAIL:
            return {
                "chat_messages_used": 0,
                "chat_messages_limit": 999999,
                "chat_messages_remaining": 999999,
                "notebooks_used": 0,
                "notebooks_limit": 999999,
                "notebooks_remaining": 999999,
            }

        cutoff = datetime.now(timezone.utc) - timedelta(days=RATE_LIMIT_WINDOW_DAYS)
        chat_count = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.user_id == user_id,
                ChatMessage.role == "user",
                ChatMessage.created_at >= cutoff,
            )
            .count()
        )
        notebook_count = (
            db.query(SavedNotebook)
            .filter(
                SavedNotebook.user_id == user_id,
                SavedNotebook.is_premade == False,
                SavedNotebook.deleted_at == None,
            )
            .count()
        )
        return {
            "chat_messages_used": chat_count,
            "chat_messages_limit": RATE_LIMIT_CHAT_MESSAGES,
            "chat_messages_remaining": max(0, RATE_LIMIT_CHAT_MESSAGES - chat_count),
            "notebooks_used": notebook_count,
            "notebooks_limit": RATE_LIMIT_NOTEBOOKS,
            "notebooks_remaining": max(0, RATE_LIMIT_NOTEBOOKS - notebook_count),
        }
    finally:
        db.close()


@app.get("/api/usage")
def get_usage(user: User = Depends(get_current_user)):
    """Return the user's current usage against rate limits."""
    return _get_user_usage(user.id)


# ═══════════════════════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

class RegisterRequest(BaseModel):
    email: str
    name: str
    password: str
    course: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/api/auth/register")
def register(req: RegisterRequest):
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == req.email.lower().strip()).first()
        if existing:
            raise HTTPException(400, "Email already registered")

        user = User(
            email=req.email.lower().strip(),
            name=req.name.strip(),
            password_hash=hash_password(req.password),
            course=req.course.strip(),
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        token = create_access_token(user.id, user.email)
        return {
            "token": token,
            "user": {"id": user.id, "email": user.email, "name": user.name, "course": user.course},
        }
    finally:
        db.close()


@app.post("/api/auth/login")
def login(req: LoginRequest):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == req.email.lower().strip()).first()
        if not user or not verify_password(req.password, user.password_hash):
            raise HTTPException(401, "Invalid email or password")

        token = create_access_token(user.id, user.email)
        return {
            "token": token,
            "user": {"id": user.id, "email": user.email, "name": user.name, "course": user.course},
        }
    finally:
        db.close()


@app.get("/api/auth/me")
def get_me(user: User = Depends(get_current_user)):
    return {"id": user.id, "email": user.email, "name": user.name, "course": user.course}


# ═══════════════════════════════════════════════════════════════════════════
# PAPERS / QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/papers")
def list_papers():
    """List all available papers."""
    db = SessionLocal()
    try:
        papers = db.query(Paper).all()
        return [
            {
                "id": p.paper_id,
                "title": p.title,
                "description": p.description,
                "course": p.course,
                "questionCount": p.question_count,
            }
            for p in papers
        ]
    finally:
        db.close()


@app.get("/api/papers/{paper_id}")
def get_paper(paper_id: str):
    """Get a full paper with all questions."""
    db = SessionLocal()
    try:
        paper = db.query(Paper).filter(Paper.paper_id == paper_id).first()
        if not paper:
            raise HTTPException(404, "Paper not found")
        return {
            "id": paper.paper_id,
            "title": paper.title,
            "description": paper.description,
            "course": paper.course,
            "questions": json.loads(paper.questions_json),
        }
    finally:
        db.close()


@app.get("/api/papers/{paper_id}/questions")
def get_questions_paginated(paper_id: str, batch: int = 1, user: User = Depends(get_current_user)):
    """Get a batch of questions (10 at a time). Tracks what the user has already seen."""
    db = SessionLocal()
    try:
        paper = db.query(Paper).filter(Paper.paper_id == paper_id).first()
        if not paper:
            raise HTTPException(404, "Paper not found")

        all_questions = json.loads(paper.questions_json)

        # Get questions already answered by this user for this paper
        answered_ids = set()
        prev_sessions = (
            db.query(QuizSession)
            .filter(QuizSession.user_id == user.id, QuizSession.paper_id == paper_id, QuizSession.completed == True)
            .all()
        )
        for sess in prev_sessions:
            for ans in sess.answers:
                answered_ids.add(ans.question_id)

        # Filter to unanswered questions first, then pad with answered if needed
        unanswered = [q for q in all_questions if q["id"] not in answered_ids]
        remaining = unanswered if unanswered else all_questions

        # Paginate
        start = (batch - 1) * QUESTIONS_PER_BATCH
        end = start + QUESTIONS_PER_BATCH
        batch_questions = remaining[start:end]
        total_batches = math.ceil(len(remaining) / QUESTIONS_PER_BATCH)

        return {
            "paper_id": paper_id,
            "paper_title": paper.title,
            "batch": batch,
            "total_batches": total_batches,
            "total_questions": len(remaining),
            "questions": batch_questions,
            "has_more": end < len(remaining),
        }
    finally:
        db.close()


@app.get("/api/questions/by-tags")
def get_questions_by_tags(tags: str, batch: int = 1):
    """Get questions matching any of the given tags (comma-separated). Searches across all papers."""
    tag_list = [t.strip().lower() for t in tags.split(",") if t.strip()]
    if not tag_list:
        raise HTTPException(400, "No tags provided")

    db = SessionLocal()
    try:
        papers = db.query(Paper).all()
        matched = []

        for paper in papers:
            questions = json.loads(paper.questions_json)
            for q in questions:
                q_text = q.get("text", "").lower()
                q_tags = [t.lower() for t in (q.get("tags", []) or [])]
                q_key_terms = [t.lower() for t in (q.get("keyTerms", []) or [])]
                searchable = q_text + " " + " ".join(q_tags) + " ".join(q_key_terms)

                if any(tag in searchable for tag in tag_list):
                    matched.append({**q, "_paper_id": paper.paper_id, "_paper_title": paper.title})

        # Paginate
        start = (batch - 1) * QUESTIONS_PER_BATCH
        end = start + QUESTIONS_PER_BATCH
        total_batches = math.ceil(len(matched) / QUESTIONS_PER_BATCH) if matched else 1

        return {
            "tags": tag_list,
            "batch": batch,
            "total_batches": total_batches,
            "total_matched": len(matched),
            "questions": matched[start:end],
            "has_more": end < len(matched),
        }
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# QUIZ SESSIONS
# ═══════════════════════════════════════════════════════════════════════════

class StartSessionRequest(BaseModel):
    paper_id: str
    paper_title: str = ""
    batch_number: int = 1


class SubmitAnswerRequest(BaseModel):
    question_id: str
    question_text: str = ""
    user_answer: str
    correct_answer: str = ""
    is_correct: bool
    time_spent_ms: int = 0


class CompleteSessionRequest(BaseModel):
    score: int
    total: int


@app.post("/api/sessions")
def start_session(req: StartSessionRequest, user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        session = QuizSession(
            user_id=user.id,
            paper_id=req.paper_id,
            paper_title=req.paper_title,
            batch_number=req.batch_number,
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        return {"session_id": session.id}
    finally:
        db.close()


@app.post("/api/sessions/{session_id}/answer")
def submit_answer(session_id: int, req: SubmitAnswerRequest, user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        session = db.query(QuizSession).filter(QuizSession.id == session_id, QuizSession.user_id == user.id).first()
        if not session:
            raise HTTPException(404, "Session not found")

        answer = SessionAnswer(
            session_id=session_id,
            question_id=req.question_id,
            question_text=req.question_text,
            user_answer=req.user_answer,
            correct_answer=req.correct_answer,
            is_correct=req.is_correct,
            time_spent_ms=req.time_spent_ms,
        )
        db.add(answer)
        db.commit()
        return {"status": "ok"}
    finally:
        db.close()


@app.post("/api/sessions/{session_id}/complete")
def complete_session(session_id: int, req: CompleteSessionRequest, user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        session = db.query(QuizSession).filter(QuizSession.id == session_id, QuizSession.user_id == user.id).first()
        if not session:
            raise HTTPException(404, "Session not found")

        session.score = req.score
        session.total = req.total
        session.completed = True
        session.completed_at = datetime.now(timezone.utc)
        db.commit()

        # Update skill profile in background (non-blocking)
        try:
            tutor.update_skill_profile(user.id)
        except Exception as e:
            print(f"[Skill Update] Warning: {e}")

        return {"status": "completed", "score": req.score, "total": req.total}
    finally:
        db.close()


@app.get("/api/sessions/history")
def get_session_history(user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        sessions = (
            db.query(QuizSession)
            .filter(QuizSession.user_id == user.id)
            .order_by(QuizSession.started_at.desc())
            .limit(50)
            .all()
        )
        return [
            {
                "id": s.id,
                "paper_id": s.paper_id,
                "paper_title": s.paper_title,
                "score": s.score,
                "total": s.total,
                "batch": s.batch_number,
                "completed": s.completed,
                "started_at": s.started_at.isoformat() if s.started_at else None,
                "completed_at": s.completed_at.isoformat() if s.completed_at else None,
            }
            for s in sessions
        ]
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# ANSWER EVALUATION (LLM-powered grading for open-ended questions)
# ═══════════════════════════════════════════════════════════════════════════

class MarkPointInput(BaseModel):
    point: str
    marks: int = 1

class EvaluateAnswerRequest(BaseModel):
    question_text: str
    student_answer: str
    model_answer: Optional[str] = None
    key_terms: Optional[list[str]] = None
    mark_scheme: Optional[list[MarkPointInput]] = None
    total_marks: Optional[int] = None


GRADING_PROMPT_WITH_SCHEME = """You are a university exam marker. Grade the student's answer against the mark scheme.

QUESTION: {question_text}

MARK SCHEME:
{mark_scheme_text}

TOTAL MARKS: {total_marks}

MODEL ANSWER (for reference):
{model_answer}

STUDENT ANSWER:
{student_answer}

INSTRUCTIONS:
- Award marks for each marking point the student demonstrates, even if worded differently.
- A concise correct answer deserves full marks — do not penalise brevity.
- Focus on MEANING and conceptual accuracy, not exact wording.
- Be fair but rigorous — only award marks for points that are clearly addressed.

Respond with ONLY valid JSON (no markdown fences):
{{
  "marks_awarded": <int>,
  "total_marks": {total_marks},
  "points_hit": ["<marking point text that was addressed>", ...],
  "points_missed": ["<marking point text that was NOT addressed>", ...],
  "feedback": "<1-2 sentence constructive feedback>"
}}"""

GRADING_PROMPT_HOLISTIC = """You are a university exam marker. Grade the student's answer against the model answer.

QUESTION: {question_text}

MODEL ANSWER:
{model_answer}

KEY TERMS EXPECTED: {key_terms}

STUDENT ANSWER:
{student_answer}

INSTRUCTIONS:
- Award a score from 0 to 100 based on how well the student's answer captures the key concepts.
- A concise correct answer deserves a high score — do not penalise brevity.
- Focus on MEANING and conceptual accuracy, not exact wording or length.
- If the student uses different words to express the same concept, give credit.
- If key terms or their equivalents are missing, note them.

Respond with ONLY valid JSON (no markdown fences):
{{
  "score": <0-100>,
  "matched_terms": ["<terms the student addressed>", ...],
  "missing_terms": ["<important terms/concepts NOT addressed>", ...],
  "feedback": "<1-2 sentence constructive feedback>"
}}"""


@app.post("/api/evaluate-answer")
def evaluate_answer(req: EvaluateAnswerRequest):
    """Use GPT-4o-mini to evaluate an open-ended answer."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(500, "OpenAI API key not configured")

    client = OpenAI(api_key=api_key)

    has_scheme = req.mark_scheme and len(req.mark_scheme) > 0
    total_marks = req.total_marks or (sum(p.marks for p in req.mark_scheme) if has_scheme else 5)

    if has_scheme:
        scheme_lines = []
        for p in req.mark_scheme:
            scheme_lines.append(f"- [{p.marks} mark{'s' if p.marks != 1 else ''}] {p.point}")
        prompt = GRADING_PROMPT_WITH_SCHEME.format(
            question_text=req.question_text,
            mark_scheme_text="\n".join(scheme_lines),
            total_marks=total_marks,
            model_answer=req.model_answer or "(not provided)",
            student_answer=req.student_answer,
        )
    else:
        prompt = GRADING_PROMPT_HOLISTIC.format(
            question_text=req.question_text,
            model_answer=req.model_answer or "(not provided)",
            key_terms=", ".join(req.key_terms) if req.key_terms else "(none specified)",
            student_answer=req.student_answer,
        )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        result = json.loads(raw)
    except Exception as e:
        raise HTTPException(500, f"Evaluation failed: {str(e)}")

    if has_scheme:
        marks = min(result.get("marks_awarded", 0), total_marks)
        return {
            "mode": "mark_scheme",
            "marks_awarded": marks,
            "total_marks": total_marks,
            "score": marks / total_marks if total_marks > 0 else 0,
            "is_correct": marks >= total_marks * 0.5,
            "points_hit": result.get("points_hit", []),
            "points_missed": result.get("points_missed", []),
            "feedback": result.get("feedback", ""),
        }
    else:
        score_pct = min(max(result.get("score", 0), 0), 100)
        return {
            "mode": "holistic",
            "score": score_pct / 100,
            "is_correct": score_pct >= 50,
            "matched_terms": result.get("matched_terms", []),
            "missing_terms": result.get("missing_terms", []),
            "feedback": result.get("feedback", ""),
        }


@app.get("/api/stats")
def get_user_stats(user: User = Depends(get_current_user)):
    """Aggregate stats for the user's dashboard, including streak."""
    from datetime import timedelta

    db = SessionLocal()
    try:
        sessions = db.query(QuizSession).filter(QuizSession.user_id == user.id, QuizSession.completed == True).all()
        total_sessions = len(sessions)
        total_questions = sum(s.total for s in sessions)
        total_correct = sum(s.score for s in sessions)
        avg_score = (total_correct / total_questions * 100) if total_questions > 0 else 0

        # Compute streak — consecutive days with at least one completed session
        today = datetime.now(timezone.utc).date()
        active_dates = set()
        for s in sessions:
            if s.completed_at:
                active_dates.add(s.completed_at.date())

        # Count streak backwards from today
        streak = 0
        check_date = today
        while check_date in active_dates:
            streak += 1
            check_date -= timedelta(days=1)

        # Build week activity (last 7 days, Mon-Sun aligned to current week)
        # Find the Monday of current week
        monday = today - timedelta(days=today.weekday())
        week_days = []
        day_labels = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']
        for i in range(7):
            d = monday + timedelta(days=i)
            if d > today:
                status = 'future'
            elif d in active_dates:
                status = 'active'
            else:
                status = 'missed'
            week_days.append({'label': day_labels[i], 'status': status})

        return {
            "total_sessions": total_sessions,
            "total_questions": total_questions,
            "total_correct": total_correct,
            "average_score": round(avg_score, 1),
            "streak": streak,
            "week": week_days,
        }
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# NOTEBOOKS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/notebooks")
def list_notebooks(user: User = Depends(get_current_user)):
    """List user's saved notebooks + all premade notebooks."""
    db = SessionLocal()
    try:
        notebooks = (
            db.query(SavedNotebook)
            .filter(
                (SavedNotebook.user_id == user.id) | (SavedNotebook.is_premade == True),
                SavedNotebook.deleted_at == None,
            )
            .order_by(SavedNotebook.created_at.desc())
            .all()
        )
        results = []
        for nb in notebooks:
            data = json.loads(nb.notebook_json)
            data["_saved_id"] = nb.id
            data["_is_premade"] = nb.is_premade
            data["_folder"] = nb.folder or ""
            results.append(data)
        return results
    finally:
        db.close()


@app.get("/api/notebooks/folders")
def list_folders(user: User = Depends(get_current_user)):
    """Return persisted folder names for this user."""
    db = SessionLocal()
    try:
        rows = db.query(StudyFolder).filter(StudyFolder.user_id == user.id).order_by(StudyFolder.created_at).all()
        return [r.name for r in rows]
    finally:
        db.close()


class MoveNotebookRequest(BaseModel):
    folder: str = ""


@app.put("/api/notebooks/{saved_id}/move")
def move_notebook(saved_id: int, req: MoveNotebookRequest, user: User = Depends(get_current_user)):
    """Move a notebook into a folder (or root if folder is empty)."""
    db = SessionLocal()
    try:
        nb = db.query(SavedNotebook).filter(
            SavedNotebook.id == saved_id, SavedNotebook.user_id == user.id
        ).first()
        if not nb:
            raise HTTPException(404, "Notebook not found")
        old_folder = nb.folder
        new_folder = req.folder.strip()[:100]
        nb.folder = new_folder
        db.commit()

        if old_folder and old_folder != new_folder:
            try:
                rag.delete_notebook_embeddings(user.id, old_folder, nb.notebook_id)
            except Exception:
                pass
        if new_folder:
            try:
                data = json.loads(nb.notebook_json)
                rag.embed_notebook(user.id, new_folder, nb.notebook_id, data)
            except Exception:
                import traceback
                traceback.print_exc()

        return {"status": "moved", "folder": nb.folder}
    finally:
        db.close()


class CreateFolderRequest(BaseModel):
    name: str


@app.post("/api/notebooks/folders")
def create_folder(req: CreateFolderRequest, user: User = Depends(get_current_user)):
    """Create and persist a new folder."""
    name = req.name.strip()[:100]
    if not name:
        raise HTTPException(400, "Folder name cannot be empty")
    db = SessionLocal()
    try:
        existing = db.query(StudyFolder).filter(
            StudyFolder.user_id == user.id, StudyFolder.name == name
        ).first()
        if existing:
            return {"folder": name}
        folder = StudyFolder(user_id=user.id, name=name)
        db.add(folder)
        db.commit()
        return {"folder": name}
    finally:
        db.close()


@app.delete("/api/notebooks/folders/{folder_name}")
def delete_folder(folder_name: str, user: User = Depends(get_current_user)):
    """Delete a folder and move its notebooks back to root."""
    db = SessionLocal()
    try:
        notebooks = db.query(SavedNotebook).filter(
            SavedNotebook.user_id == user.id, SavedNotebook.folder == folder_name
        ).all()
        for nb in notebooks:
            nb.folder = ""
        folder = db.query(StudyFolder).filter(
            StudyFolder.user_id == user.id, StudyFolder.name == folder_name
        ).first()
        if folder:
            db.delete(folder)
        db.commit()
        return {"status": "deleted", "moved_count": len(notebooks)}
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# FOLDER RAG
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/folders/{folder_name}/embed")
def embed_folder(folder_name: str, user: User = Depends(get_current_user)):
    """Embed all notebooks in a folder into ChromaDB."""
    try:
        result = rag.embed_all_in_folder(user.id, folder_name)
        return result
    except Exception:
        import traceback
        traceback.print_exc()
        return {"notebooks_embedded": 0, "total_chunks": 0, "error": "Embedding failed"}

@app.post("/api/folders/{folder_name}/upload")
async def upload_folder_source(
    folder_name: str,
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
):
    """Upload a raw document to a folder — extract text, embed, no notebook generation."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Not authenticated")
    token = authorization.split(" ", 1)[1]
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(401, "Invalid token")
    db = SessionLocal()
    user = db.query(User).filter(User.id == int(payload["sub"])).first()
    db.close()
    if not user:
        raise HTTPException(401, "User not found")

    if not file.filename:
        raise HTTPException(400, "No filename provided")

    ext = Path(file.filename).suffix.lower()
    allowed = {".pdf", ".png", ".jpg", ".jpeg", ".pptx", ".tiff", ".bmp", ".webp"}
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        raw_text = ""
        page_count = 0
        source_type = ext.lstrip(".")

        if ext == ".pptx":
            from extractor import extract_content_from_pptx
            pages = extract_content_from_pptx(str(tmp_path))
            page_count = len(pages)
            raw_text = "\n\n".join(p.get("text", "") for p in pages if p.get("text"))
        elif ext == ".pdf":
            from extractor import extract_content_from_pdf
            pages = extract_content_from_pdf(str(tmp_path))
            if pages:
                page_count = len(pages)
                raw_text = "\n\n".join(p.get("text", "") for p in pages if p.get("text"))
            else:
                return JSONResponse(status_code=400, content={"detail": "Could not extract text from PDF"})
        else:
            return JSONResponse(status_code=400, content={"detail": "Image files must be uploaded via the full notebook pipeline"})

        if not raw_text.strip():
            return JSONResponse(status_code=400, content={"detail": "No text could be extracted from this file"})

        source_id = f"src_{uuid.uuid4().hex[:10]}"
        title = Path(file.filename).stem.replace("_", " ").replace("-", " ")

        stored_path = FOLDER_UPLOADS_DIR / f"{source_id}{ext}"
        shutil.copy2(str(tmp_path), str(stored_path))

        db = SessionLocal()
        try:
            fs = FolderSource(
                user_id=user.id,
                folder_name=folder_name,
                source_id=source_id,
                title=title,
                filename=file.filename,
                source_type=source_type,
                page_count=page_count,
                raw_text=raw_text,
                file_path=str(stored_path),
            )
            db.add(fs)
            db.commit()
        finally:
            db.close()

        chunk_count = 0
        try:
            chunk_count = rag.embed_raw_source(user.id, folder_name, source_id, title, raw_text)
        except Exception:
            import traceback
            traceback.print_exc()

        return {
            "source_id": source_id,
            "title": title,
            "page_count": page_count,
            "chunk_count": chunk_count,
            "filename": file.filename,
        }
    except HTTPException:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": "Upload failed — server error"})
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)


@app.delete("/api/folders/{folder_name}/sources/{source_id}")
def delete_folder_source(folder_name: str, source_id: str, user: User = Depends(get_current_user)):
    """Delete a raw source from a folder."""
    db = SessionLocal()
    file_path = None
    try:
        fs = db.query(FolderSource).filter(
            FolderSource.user_id == user.id,
            FolderSource.folder_name == folder_name,
            FolderSource.source_id == source_id,
        ).first()
        if not fs:
            raise HTTPException(404, "Source not found")
        file_path = fs.file_path
        db.delete(fs)
        db.commit()
    finally:
        db.close()

    if file_path:
        Path(file_path).unlink(missing_ok=True)

    try:
        rag.delete_notebook_embeddings(user.id, folder_name, source_id)
    except Exception:
        pass

    return {"status": "deleted", "source_id": source_id}


@app.get("/api/folders/{folder_name}/sources/{source_id}/file")
def get_source_file(folder_name: str, source_id: str, user: User = Depends(get_current_user)):
    """Serve the original uploaded file for a folder source."""
    db = SessionLocal()
    try:
        fs = db.query(FolderSource).filter(
            FolderSource.user_id == user.id,
            FolderSource.folder_name == folder_name,
            FolderSource.source_id == source_id,
        ).first()
        if not fs:
            raise HTTPException(404, "Source not found")
        if not fs.file_path or not Path(fs.file_path).exists():
            raise HTTPException(404, "File not available")
        file_path = Path(fs.file_path)
        filename = fs.filename
    finally:
        db.close()

    media_types = {
        ".pdf": "application/pdf",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }
    media_type = media_types.get(file_path.suffix.lower(), "application/octet-stream")

    from fastapi.responses import FileResponse
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@app.post("/api/folders/{folder_name}/sources/{source_id}/generate-notebook")
async def generate_notebook_from_source(
    folder_name: str,
    source_id: str,
    authorization: Optional[str] = Header(None),
):
    """Run the full notebook pipeline on a stored folder source, returning SSE stream."""
    import asyncio
    import queue
    import threading

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Not authenticated")
    token = authorization.split(" ", 1)[1]
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(401, "Invalid token")
    db = SessionLocal()
    user = db.query(User).filter(User.id == int(payload["sub"])).first()
    fs = db.query(FolderSource).filter(
        FolderSource.source_id == source_id,
        FolderSource.folder_name == folder_name,
    ).first() if user else None
    db.close()
    if not user:
        raise HTTPException(401, "User not found")
    if not fs:
        raise HTTPException(404, "Source not found")
    if not fs.file_path or not Path(fs.file_path).exists():
        raise HTTPException(404, "Original file not available")

    if user:
        usage = _get_user_usage(user.id)
        if usage["notebooks_remaining"] <= 0:
            raise HTTPException(429, "Notebook limit reached.")

    src_path = Path(fs.file_path)
    progress_q: queue.Queue = queue.Queue()

    def on_progress(stage, current, total):
        msg = _STAGE_MESSAGES.get(stage, stage)
        try:
            msg = msg.format(current=current, total=total)
        except (KeyError, IndexError):
            pass
        progress_q.put({"stage": stage, "message": msg, "current": current, "total": total})

    def run_pipeline():
        try:
            from pipeline import run_notebook_pipeline
            paper_paths = _get_paper_paths()
            result = run_notebook_pipeline(
                src_path,
                output_path=None,
                paper_paths=paper_paths if paper_paths else None,
                provider="openai",
                extra_instructions="",
                detail="low",
                on_progress=on_progress,
            )
            nb_id = result.get("id", f"nb_{uuid.uuid4().hex[:8]}")
            save_path = GENERATED_DIR / f"{nb_id}.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            db2 = SessionLocal()
            try:
                title = result.get("title", "Untitled")
                saved = SavedNotebook(
                    user_id=user.id,
                    notebook_id=nb_id,
                    title=title,
                    course=result.get("course", ""),
                    notebook_json=json.dumps(result),
                    is_premade=False,
                    folder=folder_name,
                )
                db2.add(saved)
                db2.commit()
                db2.refresh(saved)
                result["_saved_id"] = saved.id
                result["_folder"] = folder_name

                try:
                    rag.embed_notebook(user.id, folder_name, nb_id, result)
                except Exception:
                    import traceback
                    traceback.print_exc()
                try:
                    spaced_rep.create_cards_for_notebook(user.id, nb_id, result)
                except Exception:
                    import traceback
                    traceback.print_exc()
            finally:
                db2.close()

            progress_q.put({"stage": "done", "notebook": result})
        except Exception as exc:
            progress_q.put({"stage": "error", "message": str(exc)})

    threading.Thread(target=run_pipeline, daemon=True).start()

    async def event_stream():
        while True:
            try:
                item = progress_q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.3)
                continue
            yield f"data: {json.dumps(item)}\n\n"
            if item.get("stage") in ("done", "error"):
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/folders/{folder_name}/sources")
def folder_sources(folder_name: str, user: User = Depends(get_current_user)):
    """List all sources (notebooks + raw documents) with metadata."""
    try:
        chroma_sources = rag.get_folder_sources(user.id, folder_name)
    except Exception:
        import traceback
        traceback.print_exc()
        chroma_sources = []

    embedded_ids = {s["notebook_id"] for s in chroma_sources}

    db = SessionLocal()
    try:
        notebooks = (
            db.query(SavedNotebook)
            .filter(
                SavedNotebook.user_id == user.id,
                SavedNotebook.folder == folder_name,
                SavedNotebook.deleted_at == None,
            )
            .all()
        )
        raw_sources = (
            db.query(FolderSource)
            .filter(
                FolderSource.user_id == user.id,
                FolderSource.folder_name == folder_name,
            )
            .all()
        )

        all_sources = []
        for nb in notebooks:
            data = json.loads(nb.notebook_json)
            all_sources.append({
                "notebook_id": nb.notebook_id,
                "saved_id": nb.id,
                "title": data.get("title", nb.title),
                "course": data.get("course", nb.course),
                "section_count": len(data.get("sections") or []),
                "embedded": nb.notebook_id in embedded_ids,
                "type": "notebook",
            })
        for src in raw_sources:
            all_sources.append({
                "notebook_id": src.source_id,
                "source_id": src.source_id,
                "title": src.title,
                "filename": src.filename,
                "source_type": src.source_type,
                "page_count": src.page_count,
                "embedded": src.source_id in embedded_ids,
                "type": "document",
            })

        return {"sources": all_sources, "embedding_stats": chroma_sources}
    finally:
        db.close()

@app.post("/api/folders/{folder_name}/study-plan")
def folder_study_plan(folder_name: str, user: User = Depends(get_current_user)):
    """Generate a study plan from all sources in a folder."""
    plan = rag.generate_study_plan(user.id, folder_name, user.name)
    return {"plan": plan}

@app.get("/api/folders/{folder_name}/notebooks")
def folder_notebooks(folder_name: str, user: User = Depends(get_current_user)):
    """List all notebooks in a folder."""
    db = SessionLocal()
    try:
        notebooks = (
            db.query(SavedNotebook)
            .filter(
                SavedNotebook.user_id == user.id,
                SavedNotebook.folder == folder_name,
                SavedNotebook.deleted_at == None,
            )
            .all()
        )
        return [
            {
                "id": nb.notebook_id,
                "_saved_id": nb.id,
                "title": nb.title,
                "course": nb.course,
                **json.loads(nb.notebook_json),
            }
            for nb in notebooks
        ]
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# LESSON / COURSE OUTLINE
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/folders/{folder_name}/outline")
def generate_outline(folder_name: str, user: User = Depends(get_current_user)):
    """Generate or regenerate a course outline from folder sources."""
    result = lesson.generate_outline(user.id, folder_name)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.get("/api/folders/{folder_name}/lesson")
def get_lesson_state(folder_name: str, user: User = Depends(get_current_user)):
    """Get current lesson state — outline, progress, current section."""
    return lesson.get_lesson_state(user.id, folder_name)


@app.post("/api/folders/{folder_name}/lesson/advance")
def advance_lesson(folder_name: str, user: User = Depends(get_current_user)):
    """Advance to the next lesson section."""
    result = lesson.advance_section(user.id, folder_name)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.post("/api/folders/{folder_name}/lesson/reset")
def reset_lesson(folder_name: str, user: User = Depends(get_current_user)):
    """Reset lesson progress to start over."""
    result = lesson.reset_lesson(user.id, folder_name)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.post("/api/notebooks/save")
def save_notebook(notebook: dict, user: User = Depends(get_current_user)):
    """Save a generated notebook to the user's account."""
    db = SessionLocal()
    try:
        nb_id = notebook.get("id", f"nb_{uuid.uuid4().hex[:8]}")
        saved = SavedNotebook(
            user_id=user.id,
            notebook_id=nb_id,
            title=notebook.get("title", "Untitled"),
            course=notebook.get("course", ""),
            notebook_json=json.dumps(notebook),
            is_premade=False,
            folder=notebook.get("folder", ""),
        )
        db.add(saved)
        db.commit()
        db.refresh(saved)

        try:
            spaced_rep.create_cards_for_notebook(user.id, nb_id, notebook)
        except Exception:
            import traceback
            traceback.print_exc()

        folder = notebook.get("folder", "")
        if folder:
            try:
                rag.embed_notebook(user.id, folder, nb_id, notebook)
            except Exception:
                import traceback
                traceback.print_exc()

        return {"status": "saved", "id": saved.id}
    finally:
        db.close()


@app.delete("/api/notebooks/{notebook_id}")
def delete_notebook(notebook_id: int, user: User = Depends(get_current_user)):
    print(f"  [delete] user={user.id} notebook_id={notebook_id}")
    db = SessionLocal()
    try:
        nb = db.query(SavedNotebook).filter(SavedNotebook.id == notebook_id, SavedNotebook.user_id == user.id).first()
        if not nb:
            print(f"  [delete] NOT FOUND — id={notebook_id} user={user.id}")
            raise HTTPException(404, "Notebook not found")

        slug = nb.notebook_id
        title = nb.title[:40]
        db.delete(nb)
        db.commit()

        print(f"  [delete] HARD-DELETED — id={notebook_id} slug={slug} title={title}")
        return {"status": "deleted", "notebook_id": slug}
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# PEDRO CHAT (AI TUTOR)
# ═══════════════════════════════════════════════════════════════════════════

class ChatSendRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    context_type: str = "global"  # "notebook", "global", "session"
    context_id: Optional[str] = None
    notebook_ids: Optional[list[str]] = None


@app.post("/api/chat/send")
def chat_send(req: ChatSendRequest, user: User = Depends(get_current_user)):
    """Send a message to Pedro and get a Socratic response."""
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")
    if req.context_type not in ("notebook", "global", "session", "folder", "lesson"):
        raise HTTPException(400, "Invalid context_type")

    usage = _get_user_usage(user.id)
    if usage["chat_messages_remaining"] <= 0:
        raise HTTPException(
            429,
            "You've reached your weekly message limit. "
            "Your limit resets in a few days — thanks for testing Coast!",
        )

    try:
        result = tutor.send_message(
            user_id=user.id,
            message=req.message.strip(),
            conversation_id=req.conversation_id,
            context_type=req.context_type,
            context_id=req.context_id,
            notebook_ids=req.notebook_ids,
        )
        result["usage"] = _get_user_usage(user.id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Chat error: {str(e)}")


@app.post("/api/chat/stream")
def chat_stream(req: ChatSendRequest, user: User = Depends(get_current_user)):
    """Streaming version of chat/send — returns SSE with token chunks."""
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")
    if req.context_type not in ("notebook", "global", "session", "folder", "lesson"):
        raise HTTPException(400, "Invalid context_type")

    usage = _get_user_usage(user.id)
    if usage["chat_messages_remaining"] <= 0:
        raise HTTPException(429, "Weekly message limit reached.")

    def event_stream():
        try:
            for token, meta in tutor.send_message_stream(
                user_id=user.id,
                message=req.message.strip(),
                conversation_id=req.conversation_id,
                context_type=req.context_type,
                context_id=req.context_id,
                notebook_ids=req.notebook_ids,
            ):
                if token is not None:
                    yield f"data: {json.dumps({'token': token})}\n\n"
                if meta is not None:
                    meta["usage"] = _get_user_usage(user.id)
                    yield f"data: {json.dumps({'done': True, **meta})}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/chat/history")
def chat_history(conversation_id: str, user: User = Depends(get_current_user)):
    """Get all messages for a specific conversation."""
    messages = tutor.get_chat_history(conversation_id, user.id)
    return messages


@app.get("/api/chat/conversations")
def chat_conversations(user: User = Depends(get_current_user)):
    """List all of a user's conversations with Pedro."""
    return tutor.get_conversations(user.id)


class AddNoteRequest(BaseModel):
    pedro_message: str
    notebook_id: Optional[str] = None


class ExerciseRequest(BaseModel):
    section_title: str
    section_content: str
    action: str = "generate"  # "generate" or "evaluate"
    question: str = ""
    answer: str = ""


@app.post("/api/exercise")
def exercise(req: ExerciseRequest, user: User = Depends(get_current_user)):
    """Generate a practice question or evaluate a student's answer."""
    try:
        result = tutor.handle_exercise(
            user_id=user.id,
            section_title=req.section_title,
            section_content=req.section_content,
            action=req.action,
            question=req.question,
            student_answer=req.answer,
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Exercise error: {str(e)}")


@app.post("/api/chat/add-note")
def chat_add_note(req: AddNoteRequest, user: User = Depends(get_current_user)):
    """Condense a Pedro message into a study note for the notebook."""
    if not req.pedro_message.strip():
        raise HTTPException(400, "Message cannot be empty")

    try:
        note_html = tutor.generate_note_for_notebook(req.pedro_message.strip())
        return {"note_html": note_html}
    except Exception as e:
        raise HTTPException(500, f"Note generation error: {str(e)}")


@app.get("/api/skill-profile")
def skill_profile(user: User = Depends(get_current_user)):
    """Get the user's topic skill profile."""
    return tutor.get_skill_profile(user.id)


@app.get("/api/tutor-memo")
def tutor_memo_endpoint(user: User = Depends(get_current_user)):
    """Get Pedro's memo about the user (for transparency / debugging)."""
    return tutor.get_tutor_memo(user.id)


# ═══════════════════════════════════════════════════════════════════════════
# SPACED REPETITION
# ═══════════════════════════════════════════════════════════════════════════

class ConceptExtractRequest(BaseModel):
    notebook_id: str

class ReviewSubmitRequest(BaseModel):
    card_id: int
    quality: int  # 0-5

@app.post("/api/concepts/extract")
def extract_concepts_endpoint(req: ConceptExtractRequest, user: User = Depends(get_current_user)):
    """Extract review concepts from a saved notebook."""
    db = SessionLocal()
    try:
        nb = db.query(SavedNotebook).filter(
            SavedNotebook.notebook_id == req.notebook_id,
            SavedNotebook.user_id == user.id,
        ).first()
        if not nb:
            raise HTTPException(status_code=404, detail="Notebook not found")
        notebook_json = json.loads(nb.notebook_json)
        count = spaced_rep.create_cards_for_notebook(user.id, req.notebook_id, notebook_json)
        return {"status": "ok", "cards_created": count}
    finally:
        db.close()

@app.get("/api/review/due")
def review_due(user: User = Depends(get_current_user)):
    """Get review cards due now."""
    cards = spaced_rep.get_due_cards(user.id)
    return {"cards": cards, "count": len(cards)}

@app.post("/api/review/submit")
def review_submit(req: ReviewSubmitRequest, user: User = Depends(get_current_user)):
    """Submit a review quality grade and update SM-2 schedule."""
    result = spaced_rep.submit_review(req.card_id, user.id, req.quality)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.get("/api/review/stats")
def review_stats(user: User = Depends(get_current_user)):
    """Get spaced repetition stats for the dashboard."""
    return spaced_rep.get_review_stats(user.id)

@app.get("/api/dashboard/briefing")
def dashboard_briefing(user: User = Depends(get_current_user)):
    """Get Pedro's personalized daily briefing."""
    message = spaced_rep.generate_briefing(user.id, user.name)
    stats = spaced_rep.get_review_stats(user.id)
    return {"message": message, "review_stats": stats}


# ═══════════════════════════════════════════════════════════════════════════
# GENERATE NOTES (OCR PIPELINE)
# ═══════════════════════════════════════════════════════════════════════════

_STAGE_MESSAGES = {
    "extracting": "Extracting text and images...",
    "loaded":     "Found {total} pages",
    "chunking":   "Splitting into {total} chunks...",
    "analyzing":  "Analyzing chunk {current} of {total} with AI...",
    "merging":    "Merging sections into final guide...",
    "matching":   "Matching past paper questions...",
}


@app.post("/api/generate-notes")
async def generate_notes(
    file: UploadFile = File(...),
    instructions: str = Form(""),
    provider: str = Form("openai"),
    detail: str = Form("low"),
    folder: str = Form(""),
    authorization: Optional[str] = Header(None),
):
    """Upload a PDF/image and stream progress via SSE, final event has the notebook."""
    import asyncio
    import queue
    import threading

    user = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
        payload = decode_access_token(token)
        if payload:
            db = SessionLocal()
            user = db.query(User).filter(User.id == int(payload["sub"])).first()
            db.close()

    if user:
        usage = _get_user_usage(user.id)
        if usage["notebooks_remaining"] <= 0:
            raise HTTPException(
                429,
                f"You've reached your notebook limit ({RATE_LIMIT_NOTEBOOKS} notebooks). "
                "Delete an existing notebook to free up a slot.",
            )

    if not file.filename:
        raise HTTPException(400, "No filename provided")

    ext = Path(file.filename).suffix.lower()
    allowed = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp", ".gif", ".pptx"}
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    if detail not in ("low", "high"):
        detail = "low"
    if provider not in ("openai", "anthropic", "kimi"):
        provider = "openai"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    progress_q: queue.Queue = queue.Queue()

    def on_progress(stage: str, current: int, total: int):
        msg = _STAGE_MESSAGES.get(stage, stage)
        try:
            msg = msg.format(current=current, total=total)
        except (KeyError, IndexError):
            pass
        progress_q.put({"stage": stage, "message": msg, "current": current, "total": total})

    def run_pipeline_thread():
        try:
            from pipeline import run_notebook_pipeline

            paper_paths = _get_paper_paths()
            result = run_notebook_pipeline(
                tmp_path,
                output_path=None,
                paper_paths=paper_paths if paper_paths else None,
                provider=provider,
                extra_instructions=instructions,
                detail=detail,
                on_progress=on_progress,
            )

            nb_id = result.get("id", f"nb_{uuid.uuid4().hex[:8]}")
            save_path = GENERATED_DIR / f"{nb_id}.json"
            counter = 1
            while save_path.exists():
                save_path = GENERATED_DIR / f"{nb_id}_{counter}.json"
                counter += 1

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            if user:
                db = SessionLocal()
                try:
                    from datetime import timedelta
                    title = result.get("title", "Untitled")
                    recent_cutoff = datetime.now(timezone.utc) - timedelta(seconds=60)
                    duplicate = (
                        db.query(SavedNotebook)
                        .filter(
                            SavedNotebook.user_id == user.id,
                            SavedNotebook.title == title,
                            SavedNotebook.created_at >= recent_cutoff,
                        )
                        .first()
                    )
                    if duplicate:
                        result["_saved_id"] = duplicate.id
                        result["_folder"] = duplicate.folder or ""
                    else:
                        target_folder = folder.strip()[:100] if folder else ""
                        saved = SavedNotebook(
                            user_id=user.id,
                            notebook_id=nb_id,
                            title=title,
                            course=result.get("course", ""),
                            notebook_json=json.dumps(result),
                            is_premade=False,
                            folder=target_folder,
                        )
                        db.add(saved)
                        db.commit()
                        db.refresh(saved)
                        result["_saved_id"] = saved.id
                        result["_folder"] = target_folder

                        if target_folder:
                            try:
                                rag.embed_notebook(user.id, target_folder, nb_id, result)
                            except Exception:
                                import traceback
                                traceback.print_exc()

                        try:
                            spaced_rep.create_cards_for_notebook(user.id, nb_id, result)
                        except Exception:
                            import traceback
                            traceback.print_exc()
                finally:
                    db.close()

            progress_q.put({"stage": "done", "notebook": result})
        except Exception as exc:
            progress_q.put({"stage": "error", "message": str(exc)})
        finally:
            tmp_path.unlink(missing_ok=True)
            notebook_dir = tmp_path.parent / f"{tmp_path.stem}_notebook"
            if notebook_dir.exists():
                shutil.rmtree(notebook_dir, ignore_errors=True)

    async def event_stream():
        loop = asyncio.get_event_loop()
        threading.Thread(target=run_pipeline_thread, daemon=True).start()

        while True:
            try:
                msg = await loop.run_in_executor(None, lambda: progress_q.get(timeout=5))
            except queue.Empty:
                yield ": keepalive\n\n"
                continue

            yield f"data: {json.dumps(msg)}\n\n"

            if msg.get("stage") in ("done", "error"):
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# ADMIN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════


@app.get("/api/admin/overview")
def admin_overview(user: User = Depends(get_current_user)):
    """Return all users with stats, skill profiles, and tutor memos. Admin only."""
    if user.email != ADMIN_EMAIL:
        raise HTTPException(403, "Admin access only")

    from datetime import timedelta

    db = SessionLocal()
    try:
        users = db.query(User).order_by(User.created_at.desc()).all()
        result = []

        for u in users:
            # Sessions
            sessions = db.query(QuizSession).filter(
                QuizSession.user_id == u.id, QuizSession.completed == True
            ).all()
            total_q = sum(s.total for s in sessions)
            total_correct = sum(s.score for s in sessions)

            # Streak
            today = datetime.now(timezone.utc).date()
            active_dates = set()
            for s in sessions:
                if s.completed_at:
                    active_dates.add(s.completed_at.date())
            streak = 0
            check = today
            while check in active_dates:
                streak += 1
                check -= timedelta(days=1)

            # Skill profile
            sp = db.query(SkillProfile).filter(SkillProfile.user_id == u.id).first()
            skill = json.loads(sp.profile_json) if sp else {}

            # Tutor memo
            memo = db.query(TutorMemo).filter(TutorMemo.user_id == u.id).first()

            # Chat message count
            msg_count = db.query(ChatMessage).filter(ChatMessage.user_id == u.id).count()

            # Notebooks count
            nb_count = db.query(SavedNotebook).filter(
                SavedNotebook.user_id == u.id, SavedNotebook.is_premade == False
            ).count()

            result.append({
                "id": u.id,
                "name": u.name,
                "email": u.email,
                "course": u.course,
                "created_at": u.created_at.isoformat() if u.created_at else None,
                "sessions_completed": len(sessions),
                "total_questions": total_q,
                "total_correct": total_correct,
                "accuracy": round(total_correct / total_q * 100, 1) if total_q > 0 else 0,
                "streak": streak,
                "skill_profile": skill,
                "tutor_memo": memo.memo_text if memo else "",
                "memo_updated_at": memo.updated_at.isoformat() if memo and memo.updated_at else None,
                "chat_messages": msg_count,
                "notebooks_generated": nb_count,
            })

        return {"users": result, "total_users": len(result)}
    finally:
        db.close()


@app.get("/api/admin/export-cohort")
def export_cohort(user: User = Depends(get_current_user)):
    """Full data export for the current cohort — quiz answers, chat logs, skill profiles."""
    if user.email != ADMIN_EMAIL:
        raise HTTPException(403, "Admin access only")

    db = SessionLocal()
    try:
        users = db.query(User).all()
        export = []

        for u in users:
            if u.email == ADMIN_EMAIL:
                continue

            sessions = db.query(QuizSession).filter(QuizSession.user_id == u.id).all()
            session_data = []
            for s in sessions:
                answers = db.query(SessionAnswer).filter(SessionAnswer.session_id == s.id).all()
                session_data.append({
                    "paper_id": s.paper_id,
                    "paper_title": s.paper_title,
                    "score": s.score,
                    "total": s.total,
                    "completed": s.completed,
                    "started_at": s.started_at.isoformat() if s.started_at else None,
                    "completed_at": s.completed_at.isoformat() if s.completed_at else None,
                    "answers": [
                        {
                            "question_id": a.question_id,
                            "question_text": a.question_text,
                            "user_answer": a.user_answer,
                            "correct_answer": a.correct_answer,
                            "is_correct": a.is_correct,
                            "time_spent_ms": a.time_spent_ms,
                        }
                        for a in answers
                    ],
                })

            messages = (
                db.query(ChatMessage)
                .filter(ChatMessage.user_id == u.id)
                .order_by(ChatMessage.created_at)
                .all()
            )
            chat_data = [
                {
                    "conversation_id": m.conversation_id,
                    "role": m.role,
                    "content": m.content,
                    "context_type": m.context_type,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in messages
            ]

            sp = db.query(SkillProfile).filter(SkillProfile.user_id == u.id).first()
            memo = db.query(TutorMemo).filter(TutorMemo.user_id == u.id).first()

            nb_count = (
                db.query(SavedNotebook)
                .filter(SavedNotebook.user_id == u.id, SavedNotebook.is_premade == False)
                .count()
            )

            export.append({
                "user_id": u.id,
                "name": u.name,
                "email": u.email,
                "course": u.course,
                "created_at": u.created_at.isoformat() if u.created_at else None,
                "skill_profile": json.loads(sp.profile_json) if sp else {},
                "tutor_memo": memo.memo_text if memo else "",
                "notebooks_generated": nb_count,
                "quiz_sessions": session_data,
                "chat_messages": chat_data,
            })

        return {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "total_students": len(export),
            "students": export,
        }
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
def health():
    db = SessionLocal()
    try:
        paper_count = db.query(Paper).count()
        user_count = db.query(User).count()
        return {"status": "ok", "papers": paper_count, "users": user_count}
    finally:
        db.close()


def _get_paper_paths() -> list[Path]:
    paper_files = []
    if PAPERS_DIR.exists():
        for f in PAPERS_DIR.glob("*.json"):
            if f.name in ("notebooks.json", "notebookContent.json"):
                continue
            paper_files.append(f)
    return paper_files


if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("  Coast API Server")
    print("=" * 50)
    print(f"  Papers dir:  {PAPERS_DIR}")
    print(f"  Generated:   {GENERATED_DIR}")
    print(f"  Database:    {Path(__file__).parent / 'coast.db'}")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
