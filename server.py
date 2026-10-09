"""FastAPI server for Coast — auth, notebooks, sessions, and OCR pipeline."""

from __future__ import annotations
from notes_service import sanitize_notes, notes_revision

import asyncio
import json
import math
import os
import shutil
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

import ai_usage

load_dotenv()


def _local_dev_auth() -> bool:
    """Skip production-only signup gates when running the local server."""
    return not os.getenv("RENDER")


def _email_verification_required() -> bool:
    from auth_email import email_verification_required
    return email_verification_required()

from auth import create_access_token, decode_access_token, hash_password, verify_password
from database import (
    ActivityEvent,
    ChatMessage,
    FolderSource,
    LessonNotes,
    Paper,
    QuizSession,
    SavedNotebook,
    SessionAnswer,
    SessionLocal,
    SkillProfile,
    SourceImage,
    StudyFolder,
    TutorMemo,
    User,
    UserFeedback,
    init_db,
    load_papers_from_json,
)
import threading
import tutor
import spaced_rep
import rag
import lesson


def _bg_post_process(user_id: int, folder: str, nb_id: str, notebook_data: dict):
    """Run RAG embedding and card creation in background after notebook save."""
    if folder:
        try:
            rag.embed_notebook(user_id, folder, nb_id, notebook_data)
        except Exception:
            import traceback
            traceback.print_exc()
    try:
        spaced_rep.create_cards_for_notebook(user_id, nb_id, notebook_data)
    except Exception:
        import traceback
        traceback.print_exc()
    print(f"[BG] Post-processing done for notebook {nb_id}")

app = FastAPI(title="Coast API", version="2.0.0")

_ALLOWED_ORIGINS = [
    "http://localhost:5173", "http://localhost:5174", "http://localhost:3000",
    "https://dist-delta-eight-99.vercel.app",
    "https://app.coast.academy",
    "https://coast.academy",
    "https://www.coast.academy",
]
for _origin in os.getenv("FRONTEND_URL", "").split(","):
    _origin = _origin.strip().rstrip("/")
    if _origin and _origin not in _ALLOWED_ORIGINS:
        _ALLOWED_ORIGINS.append(_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

import security as _security  # noqa: E402
app.middleware("http")(_security.security_headers)

_PROD_ORIGIN = next(
    (o for o in _ALLOWED_ORIGINS if o.startswith("https://") and "localhost" not in o),
    "https://app.coast.academy",
)

@app.options("/{rest:path}")
async def preflight_catchall(request: StarletteRequest):
    """Catch-all for OPTIONS preflight — always return CORS headers."""
    origin = request.headers.get("origin", _PROD_ORIGIN)
    allowed = origin if origin in _ALLOWED_ORIGINS else _PROD_ORIGIN
    return StarletteResponse(
        status_code=204,
        headers={
            "Access-Control-Allow-Origin": allowed,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "Authorization, Content-Type, X-Requested-With, Accept, Origin",
            "Access-Control-Max-Age": "600",
        },
    )

@app.middleware("http")
async def cors_safety_net(request: StarletteRequest, call_next):
    """Last-resort: inject CORS headers if CORSMiddleware didn't."""
    response = await call_next(request)
    if "access-control-allow-origin" not in response.headers:
        origin = request.headers.get("origin", _PROD_ORIGIN)
        allowed = origin if origin in _ALLOWED_ORIGINS else _PROD_ORIGIN
        response.headers["Access-Control-Allow-Origin"] = allowed
        response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


@app.middleware("http")
async def track_request_traffic(request: StarletteRequest, call_next):
    """In-memory request counters for admin traffic charts."""
    global _request_total
    response = await call_next(request)
    if request.method != "OPTIONS":
        _request_total += 1
        minute_key = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
        _request_counts[minute_key] = _request_counts.get(minute_key, 0) + 1
        if len(_request_counts) > 180:
            oldest = sorted(_request_counts.keys())[:-120]
            for k in oldest:
                del _request_counts[k]
    return response

@app.middleware("http")
async def attribute_ai_usage(request: StarletteRequest, call_next):
    """Tag every AI call made while serving this request with the student and feature."""
    import ai_usage
    user_id = None
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        payload = decode_access_token(auth_header[7:])
        if payload and str(payload.get("sub", "")).isdigit():
            user_id = int(payload["sub"])
    token = ai_usage.begin(user_id, ai_usage.feature_from_path(request.url.path))
    try:
        return await call_next(request)
    finally:
        ai_usage.end(token)

_cors_headers = {
    "Access-Control-Allow-Origin": _PROD_ORIGIN,
    "Access-Control-Allow-Credentials": "true",
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
    "Access-Control-Allow-Headers": "Authorization, Content-Type",
}

@app.exception_handler(HTTPException)
async def http_exception_handler(request: StarletteRequest, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers={**_cors_headers, **(exc.headers or {})},  # e.g. Retry-After, X-Coast-Limit
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

_PERSISTENT_DISK = Path("/data")
_default_generated = str(_PERSISTENT_DISK / "generated") if _PERSISTENT_DISK.is_dir() else str(Path(__file__).parent / "generated")
GENERATED_DIR = Path(os.environ.get("GENERATED_DIR", _default_generated))
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
_default_uploads = str(_PERSISTENT_DISK / "folder_uploads") if _PERSISTENT_DISK.is_dir() else str(Path(__file__).parent / "folder_uploads")
FOLDER_UPLOADS_DIR = Path(os.environ.get("FOLDER_UPLOADS_DIR", _default_uploads))
FOLDER_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_IMAGES_DIR = FOLDER_UPLOADS_DIR / "images"
SOURCE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

PAPERS_DIR = Path(os.getenv("PAPERS_DIR", str(Path(__file__).parent.parent / "Coast" / "testing" / "src" / "data")))

from curated_config import curated_source_uid as _curated_uid, get_lesson_structure

QUESTIONS_PER_BATCH = 10

RATE_LIMIT_NOTEBOOKS = 20           # max notebook uploads per user (older notebook pipeline)

_live_users: dict[int, dict] = {}
HEARTBEAT_TIMEOUT = 60

_server_started_at = datetime.now(timezone.utc)
_request_total = 0
_request_counts: dict[str, int] = {}


def _path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except OSError:
        pass
    return total


def _traffic_series(minutes: int = 60) -> list[dict]:
    now = datetime.now(timezone.utc)
    return [
        {
            "minute": (now - timedelta(minutes=offset)).strftime("%Y-%m-%dT%H:%M"),
            "count": _request_counts.get(
                (now - timedelta(minutes=offset)).strftime("%Y-%m-%dT%H:%M"), 0
            ),
        }
        for offset in range(minutes - 1, -1, -1)
    ]


def _admin_server_metrics() -> dict:
    import oma_provider
    from rag import CHROMA_PATH

    storage_items = [
        ("Primary DB", Path(os.getenv("DATABASE_PATH", "coast.db"))),
        ("OMA DB", oma_provider.OMA_DB_PATH),
        ("Uploads", FOLDER_UPLOADS_DIR),
        ("Chroma", CHROMA_PATH),
        ("OMA Images", oma_provider.OMA_IMAGE_DIR),
    ]
    breakdown = []
    for label, p in storage_items:
        p = Path(p)
        breakdown.append({
            "label": label,
            "path": str(p),
            "bytes": _path_size_bytes(p),
            "exists": p.exists(),
        })

    data_mount_breakdown = []
    data_root = Path("/data")
    if data_root.is_dir():
        for child in sorted(data_root.iterdir()):
            data_mount_breakdown.append({
                "label": child.name + ("/" if child.is_dir() else ""),
                "path": str(child),
                "bytes": _path_size_bytes(child),
            })
        data_mount_breakdown.sort(key=lambda x: x["bytes"], reverse=True)

    disk = None
    for mount in (Path("/data"), Path("/"), Path(".")):
        try:
            usage = shutil.disk_usage(mount)
            disk = {
                "mount": str(mount.resolve()),
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
                "used_pct": round(usage.used / usage.total * 100, 1) if usage.total else 0,
            }
            break
        except OSError:
            continue

    from database import CourseOutline
    db = SessionLocal()
    try:
        row_counts = {
            "users": db.query(User).count(),
            "chat_messages": db.query(ChatMessage).count(),
            "notebooks": db.query(SavedNotebook).count(),
            "study_folders": db.query(StudyFolder).count(),
            "course_outlines": db.query(CourseOutline).count(),
            "activity_events": db.query(ActivityEvent).count(),
            "feedback": db.query(UserFeedback).count(),
        }
    finally:
        db.close()

    rpm_window = _traffic_series(5)
    requests_last_5m = sum(b["count"] for b in rpm_window)

    return {
        "uptime_seconds": int((datetime.now(timezone.utc) - _server_started_at).total_seconds()),
        "started_at": _server_started_at.isoformat(),
        "environment": "production" if os.getenv("RENDER") else "development",
        "storage": {"breakdown": breakdown, "disk": disk, "data_mount": data_mount_breakdown},
        "database": row_counts,
        "traffic": {
            "total_requests": _request_total,
            "requests_last_5m": requests_last_5m,
            "requests_per_minute": round(requests_last_5m / 5, 1),
            "series_60m": _traffic_series(60),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def _raise_threadpool_cap():
    # Sync endpoints (incl. streaming chat) each hold one threadpool slot
    # for their full duration. Default cap is 40 — raise it so ~100
    # concurrent users don't queue behind long LLM responses.
    import anyio.to_thread
    anyio.to_thread.current_default_thread_limiter().total_tokens = 120


@app.on_event("startup")
def on_startup():
    init_db()
    import oma_provider
    from coast_content_oma.course_identity import initialize
    initialize(oma_provider.OMA_DB_PATH)
    import learning_jobs
    learning_jobs.start()
    import backups
    backups.start()
    import file_store
    file_store.start(backups.file_dirs())
    threading.Thread(target=learning_jobs.recover_sources, name="coast-source-recovery", daemon=True).start()
    # Past papers are no longer part of Coast: nothing loads them at startup.

    def _bg_curated():
        try:
            from curated_config import bootstrap_all_curated_content
            print("  [curated] Building shared Content OMA for premade courses (one-time)…")
            bootstrap_all_curated_content()
            print("  [curated] Premade course bootstrap complete.")
        except Exception:
            import traceback; traceback.print_exc()
    threading.Thread(target=_bg_curated, daemon=True).start()

    try:
        import oma_provider
        print(
            f"  [oma] RAG_PROVIDER={oma_provider.get_rag_provider()} "
            f"student_oma={oma_provider.is_student_enabled()} "
            f"db={oma_provider.OMA_DB_PATH}"
        )
    except Exception:
        import traceback; traceback.print_exc()


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


from security import ADMIN_EMAILS, HOUR, MINUTE, client_ip, is_admin, limiter  # noqa: E402


def _require_curated_write_access(folder_name: str, user: User) -> None:
    """Premade (curated) folders are shared — only admins may modify them."""
    if _curated_uid(folder_name) is not None and not is_admin(user):
        raise HTTPException(403, "Premade course content is read-only.")



def _get_user_usage(user_id: int):
    """Return current usage counts for rate limiting: this month's messages (see plans.py) and
    notebooks of the older notebook pipeline. Admin gets unlimited."""
    import plans
    db = SessionLocal()
    try:
        account = db.query(User).filter(User.id == user_id).first()
        if account and is_admin(account):
            return {
                "chat_messages_used": 0,
                "chat_messages_limit": 999999,
                "chat_messages_remaining": 999999,
                "notebooks_used": 0,
                "notebooks_limit": 999999,
                "notebooks_remaining": 999999,
            }
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
            **(plans.legacy_usage(db, account) if account else {}),
            "notebooks_used": notebook_count,
            "notebooks_limit": RATE_LIMIT_NOTEBOOKS,
            "notebooks_remaining": max(0, RATE_LIMIT_NOTEBOOKS - notebook_count),
        }
    finally:
        db.close()


if os.getenv("COAST_ENABLE_MANIM", "false").lower() == "true":
    from viz_router import viz_router
    app.include_router(viz_router, dependencies=[Depends(get_current_user)])


@app.get("/api/usage")
def get_usage(user: User = Depends(get_current_user)):
    """Return the user's current usage against rate limits."""
    return _get_user_usage(user.id)


def _check_messages(user: User) -> None:
    """Refuse a new message to Pedro once this month's are used up."""
    import plans
    with SessionLocal() as db:
        plans.check(db, user, "messages")


def _counts_as_message(req) -> bool:
    """The welcome chat, and the opener Coast sends for the student when a section or a
    test-out starts, don't use up messages."""
    if req.context_type == "onboarding":
        return False
    if req.context_type in ("lesson", "test_out"):
        from oma_provider import _is_lesson_intro
        return not _is_lesson_intro(req.message)
    return True


def _count_message(user: User, ref: str = "") -> None:
    import plans
    with SessionLocal() as db:
        plans.record(db, user.id, "messages", ref)
        db.commit()


@app.get("/api/plan")
def get_plan(user: User = Depends(get_current_user)):
    """The student's plan and what they've used of it this month."""
    import plans
    with SessionLocal() as db:
        return plans.summary(db, db.get(User, user.id))


@app.post("/api/plan/founder-interest")
def founder_interest(user: User = Depends(get_current_user)):
    """Payments aren't open yet: note that the student wants the Founding Student pass."""
    import plans
    with SessionLocal() as db:
        account = db.get(User, user.id)
        if not account.founder_interest_at:
            account.founder_interest_at = datetime.now(timezone.utc)
            db.commit()
            print(f"[plan] user {account.id} wants the Founding Student pass")
        return plans.summary(db, account)


# ═══════════════════════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

class RegisterRequest(BaseModel):
    email: str
    name: str
    password: str
    course: str = ""
    beta_code: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class VerifyEmailSendRequest(BaseModel):
    email: str
    beta_code: str = ""  # checked before any email is sent, so a code is only sent to an invitee


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    email: str
    code: str
    password: str


class VerifyEmailCheckRequest(BaseModel):
    email: str
    code: str


class GoogleAuthRequest(BaseModel):
    credential: str
    beta_code: str = ""


def _user_payload(user: User) -> dict:
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "course": user.course,
        "onboarding_completed": bool(user.onboarding_completed),
        "email_verified": bool(getattr(user, "email_verified", True)),
        "auth_provider": "google" if getattr(user, "google_id", None) else "email",
        "is_admin": is_admin(user),
        "plan": "founder" if getattr(user, "plan", None) == "founder" else "beta",
        "founder": getattr(user, "plan", None) == "founder" or is_admin(user),  # the founder badge
    }


def _auth_token_response(user: User) -> dict:
    token = create_access_token(user.id, user.email)
    return {"token": token, "user": _user_payload(user)}


@app.get("/api/auth/config")
def auth_config():
    """Public auth configuration for the login screen."""
    import os
    import beta_codes
    return {
        "google_client_id": os.environ.get("GOOGLE_CLIENT_ID", "").strip(),
        "email_verification_enabled": _email_verification_required(),
        "beta_code_required": beta_codes.required(),
        # Codes can be emailed (Resend set up; or shown on screen locally): "forgot password" is offered.
        "email_enabled": bool(os.environ.get("RESEND_API_KEY", "").strip())
                         or (os.environ.get("AUTH_DEV_EXPOSE_CODES", "").lower() in ("1", "true", "yes") and not os.getenv("RENDER")),
    }


@app.post("/api/auth/verify-email/send")
def send_verify_email(req: VerifyEmailSendRequest, request: Request):
    from auth_email import generate_code, normalize_email, send_verification_email, validate_email_address
    from database import EmailVerification

    email = normalize_email(req.email)
    ok, err = validate_email_address(email)
    if not ok:
        raise HTTPException(400, err)
    limiter.check(f"verify-send:ip:{client_ip(request)}", 20, HOUR)
    limiter.check(f"verify-send:{email}", 5, HOUR, "Too many codes requested for this email. Try again in an hour.")

    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            raise HTTPException(400, "Email already registered. Sign in instead.")
        import beta_codes
        if beta_codes.required() and not beta_codes.exempt(email):
            try:
                beta_codes.check(db, req.beta_code)
            except beta_codes.BetaCodeError as exc:
                raise HTTPException(400, str(exc))

        code = generate_code()
        expires = datetime.now(timezone.utc) + timedelta(minutes=15)
        row = db.query(EmailVerification).filter(EmailVerification.email == email).first()
        if row:
            row.code = code
            row.expires_at = expires
            row.verified = False
        else:
            db.add(EmailVerification(email=email, code=code, expires_at=expires, verified=False))
        db.commit()
        limiter.reset(f"verify-check:{email}")  # a fresh code gets fresh attempts

        sent, detail = send_verification_email(email, code)
        if sent:
            return {"ok": True, "message": "Verification code sent. Check your inbox."}
        if detail.startswith("dev_code:") and not os.getenv("RENDER"):
            return {"ok": True, "message": "Dev mode: use the code shown below.", "dev_code": detail.split(":", 1)[1]}
        print(f"[auth] verification email not sent: {detail}")
        raise HTTPException(503, "We couldn't send the email right now. Please try again or sign in with Google.")
    finally:
        db.close()


@app.post("/api/auth/verify-email/check")
def check_verify_email(req: VerifyEmailCheckRequest, request: Request):
    import hmac
    from auth_email import normalize_email
    from database import EmailVerification

    email = normalize_email(req.email)
    code = req.code.strip()
    limiter.check(f"verify-check:ip:{client_ip(request)}", 30, 15 * MINUTE)
    # Five wrong guesses burn the code: a six-digit code can't be brute-forced.
    if limiter.blocked(f"verify-check:{email}", 5, HOUR):
        raise HTTPException(429, "Too many wrong codes. Request a new code.")
    db = SessionLocal()
    try:
        row = db.query(EmailVerification).filter(EmailVerification.email == email).first()
        if not row or not hmac.compare_digest(str(row.code or ""), code):
            limiter.record(f"verify-check:{email}")
            raise HTTPException(400, "Invalid verification code.")
        exp = row.expires_at
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if exp < datetime.now(timezone.utc):
            raise HTTPException(400, "Code expired. Request a new one.")
        row.verified = True
        db.commit()
        return {"ok": True, "verified": True}
    finally:
        db.close()


@app.post("/api/auth/google")
def google_auth(req: GoogleAuthRequest, request: Request):
    import os
    limiter.check(f"google:ip:{client_ip(request)}", 30, 15 * MINUTE)
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token

    client_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    if not client_id:
        raise HTTPException(503, "Google sign-in is not configured.")

    try:
        idinfo = id_token.verify_oauth2_token(
            req.credential, google_requests.Request(), client_id,
        )
    except Exception:
        raise HTTPException(401, "Invalid Google sign-in.")

    email = (idinfo.get("email") or "").lower().strip()
    google_sub = idinfo.get("sub")
    name = (idinfo.get("name") or email.split("@")[0] or "Student").strip()
    if not email or not google_sub:
        raise HTTPException(400, "Google account did not provide an email.")
    if not idinfo.get("email_verified"):
        raise HTTPException(400, "Your Google email is not verified.")

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.google_id == google_sub).first()
        if not user:
            user = db.query(User).filter(User.email == email).first()
        if user:
            if not user.google_id:
                user.google_id = google_sub
            user.email_verified = True
            if name and not user.name:
                user.name = name
            db.commit()
            db.refresh(user)
            return _auth_token_response(user)

        # A new account: during the beta it needs an unused invite code. Without one the app asks
        # for it and sends the same Google sign-in again (it stays valid for an hour).
        import beta_codes
        code = None
        if beta_codes.required():
            try:
                code = beta_codes.check(db, req.beta_code)
            except beta_codes.BetaCodeError as exc:
                message = ("Enter your beta code to finish creating your account." if not (req.beta_code or "").strip()
                           else str(exc))
                raise HTTPException(403, {"message": message, "needs_beta_code": True})

        user = User(
            email=email,
            name=name,
            password_hash=hash_password(os.urandom(32).hex()),
            google_id=google_sub,
            email_verified=True,
        )
        db.add(user)
        db.flush()
        if code:
            try:
                beta_codes.consume(db, code, user.id, email)
            except beta_codes.BetaCodeError as exc:
                db.rollback()
                raise HTTPException(403, str(exc))
        db.commit()
        db.refresh(user)
        return _auth_token_response(user)
    finally:
        db.close()


@app.post("/api/auth/register")
def register(req: RegisterRequest, request: Request):
    import beta_codes
    from auth_email import normalize_email, validate_email_address
    from database import EmailVerification

    email = normalize_email(req.email)
    ok, err = validate_email_address(email)
    if not ok:
        raise HTTPException(400, err)
    limiter.check(f"register:ip:{client_ip(request)}", 10, HOUR)
    if len(req.password or "") < 8:
        raise HTTPException(400, "Use a password of at least 8 characters.")

    if os.getenv("RENDER") and email.endswith("@loadtest.local"):
        raise HTTPException(403, "Load-test accounts are disabled on production.")

    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            raise HTTPException(400, "Email already registered")

        code = None
        if beta_codes.required() and not beta_codes.exempt(email):
            try:
                code = beta_codes.check(db, req.beta_code)
            except beta_codes.BetaCodeError as exc:
                raise HTTPException(400, str(exc))

        vrow = db.query(EmailVerification).filter(
            EmailVerification.email == email,
            EmailVerification.verified == True,
        ).first()
        # Team addresses always need proof of ownership, whatever the global setting.
        if not vrow and (_email_verification_required() or email in ADMIN_EMAILS):
            raise HTTPException(400, "Verify your email before creating an account.")

        user = User(
            email=email,
            name=req.name.strip(),
            password_hash=hash_password(req.password),
            course=req.course.strip(),
            email_verified=bool(vrow),  # only an emailed code proves the address
        )
        db.add(user)
        if vrow:
            db.delete(vrow)
        db.flush()
        if code:
            try:
                beta_codes.consume(db, code, user.id, email)
            except beta_codes.BetaCodeError as exc:
                db.rollback()
                raise HTTPException(400, str(exc))
        db.commit()
        db.refresh(user)

        return _auth_token_response(user)
    finally:
        db.close()


@app.post("/api/auth/login")
def login(req: LoginRequest, request: Request):
    from auth_email import normalize_email

    email = normalize_email(req.email)
    limiter.check(f"login:ip:{client_ip(request)}", 30, 15 * MINUTE)
    if limiter.blocked(f"login-fail:{email}", 8, 15 * MINUTE):
        raise HTTPException(429, "Too many failed sign-ins for this account. Wait 15 minutes and try again.")
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            limiter.record(f"login-fail:{email}")
            raise HTTPException(401, "Invalid email or password")
        # Google is an extra way in, not a replacement: a password the student set keeps working.
        if not verify_password(req.password, user.password_hash):
            limiter.record(f"login-fail:{email}")
            if user.google_id:
                raise HTTPException(401, "Wrong password. This account also signs in with Google: use Continue with Google, "
                                         "or reset your password.")
            raise HTTPException(401, "Invalid email or password")
        limiter.reset(f"login-fail:{email}")
        # New accounts prove their email at sign-up; accounts made before that keep signing in.
        return _auth_token_response(user)
    finally:
        db.close()


def _code_hash(email: str, code: str) -> str:
    import hashlib
    return hashlib.sha256(f"{email}:{code}".encode()).hexdigest()


@app.post("/api/auth/password/forgot")
def forgot_password(req: ForgotPasswordRequest, request: Request):
    """Email a reset code to an account that signs in with a password. The answer is the same
    whether or not the account exists, so this can't be used to find out who has one."""
    from auth_email import generate_code, normalize_email, send_code_email
    from database import PasswordReset

    email = normalize_email(req.email)
    limiter.check(f"reset-send:ip:{client_ip(request)}", 20, HOUR)
    limiter.check(f"reset-send:{email}", 5, HOUR, "Too many codes requested for this email. Try again in an hour.")
    answer = {"ok": True, "message": "If an account uses this email, we've sent it a code."}
    with SessionLocal() as db:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            return answer
        # Accounts that use Google can set a password this way too: the code proves the inbox is theirs.
        code = generate_code()
        row = db.get(PasswordReset, email) or PasswordReset(email=email)
        row.code_hash = _code_hash(email, code)
        row.expires_at = datetime.now(timezone.utc) + timedelta(minutes=15)
        db.merge(row)
        db.commit()
    limiter.reset(f"reset-check:{email}")
    sent, detail = send_code_email(email, code, "reset")
    if not sent:
        if detail.startswith("dev_code:") and not os.getenv("RENDER"):
            return {**answer, "dev_code": detail.split(":", 1)[1]}
        print(f"[auth] reset email not sent: {detail}")
        raise HTTPException(503, "We couldn't send the email right now. Please try again in a few minutes.")
    return answer


@app.post("/api/auth/password/reset")
def reset_password(req: ResetPasswordRequest, request: Request):
    import hmac
    from auth_email import normalize_email
    from database import PasswordReset

    email = normalize_email(req.email)
    limiter.check(f"reset-check:ip:{client_ip(request)}", 30, 15 * MINUTE)
    if limiter.blocked(f"reset-check:{email}", 5, HOUR):
        raise HTTPException(429, "Too many wrong codes. Request a new code.")
    if len(req.password or "") < 8:
        raise HTTPException(400, "Use a password of at least 8 characters.")
    with SessionLocal() as db:
        row = db.get(PasswordReset, email)
        if not row or not hmac.compare_digest(row.code_hash, _code_hash(email, req.code.strip())):
            limiter.record(f"reset-check:{email}")
            raise HTTPException(400, "That code isn't right. Check the email or request a new code.")
        expires = row.expires_at if row.expires_at.tzinfo else row.expires_at.replace(tzinfo=timezone.utc)
        if expires < datetime.now(timezone.utc):
            raise HTTPException(400, "That code has expired. Request a new one.")
        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise HTTPException(400, "That code isn't right. Check the email or request a new code.")
        user.password_hash = hash_password(req.password)
        user.email_verified = True  # the code reached this inbox
        db.delete(row)
        db.commit()
        db.refresh(user)
        limiter.reset(f"login-fail:{email}")
        return _auth_token_response(user)


class DeleteAccountRequest(BaseModel):
    confirm_email: str


@app.get("/api/account/export")
def export_my_data(user: User = Depends(get_current_user)):
    """Download everything Coast holds about you, as JSON."""
    limiter.check(f"account-export:{user.id}", 10, HOUR, "You've downloaded your data several times. Try again in an hour.")
    import account_deletion
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return JSONResponse(account_deletion.export_account(user.id),
                        headers={"Content-Disposition": f'attachment; filename="coast-data-{stamp}.json"'})


@app.post("/api/account/delete")
def delete_my_account(req: DeleteAccountRequest, user: User = Depends(get_current_user)):
    """Delete your account and everything with it. You confirm by typing your account's email."""
    if (req.confirm_email or "").strip().lower() != (user.email or "").strip().lower():
        raise HTTPException(400, "Type your account's email exactly to confirm.")
    limiter.check(f"account-delete:{user.id}", 5, HOUR)
    import account_deletion
    result = account_deletion.delete_account(user.id)
    try:
        import map_world
        map_world.invalidate_map_cache(user.id)
    except Exception:
        pass
    print(f"[account] user {user.id} deleted their account: {result.get('removed')}")
    return {"deleted": bool(result.get("deleted"))}


@app.get("/api/auth/me")
def get_me(user: User = Depends(get_current_user)):
    return _user_payload(user)


class OnboardingRequest(BaseModel):
    preferences: dict = {}
    conversation_id: Optional[str] = None


@app.post("/api/auth/onboarding")
def complete_onboarding(req: OnboardingRequest, user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        db_user = db.query(User).filter(User.id == user.id).first()
        if not db_user:
            raise HTTPException(404, "User not found")

        traits_saved: list = []
        prefs = dict(req.preferences or {})
        if req.conversation_id:
            try:
                import onboarding as onboarding_mod
                traits_saved = onboarding_mod.get_saved_onboarding_traits(user.id)
                if not traits_saved:
                    traits_saved = onboarding_mod.finalize_onboarding(user.id, req.conversation_id)
                extracted = onboarding_mod.traits_to_preferences(traits_saved)
                prefs = {**extracted, **prefs}
            except Exception:
                import traceback
                traceback.print_exc()

        db_user.learning_preferences = json.dumps(prefs)
        db_user.onboarding_completed = True
        db.commit()
        return {
            "id": db_user.id,
            "email": db_user.email,
            "name": db_user.name,
            "course": db_user.course,
            "onboarding_completed": True,
            "traits_saved": traits_saved,
        }
    finally:
        db.close()


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
def get_questions_by_tags(tags: str, batch: int = 1, course: str = ""):
    """Get questions matching any of the given tags (comma-separated). Optionally filter by course."""
    tag_list = [t.strip().lower() for t in tags.split(",") if t.strip()]
    if not tag_list:
        raise HTTPException(400, "No tags provided")

    db = SessionLocal()
    try:
        if course:
            papers = db.query(Paper).filter(Paper.course == course).all()
        else:
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


@app.get("/api/folders/{folder_name}/section-questions")
def get_section_questions(
    folder_name: str,
    user: User = Depends(get_current_user),
    max_questions: int = 5,
):
    """Find past paper questions matching the current lesson section's topics.

    Scores each question by overlap between question tags and section key_topics.
    Returns empty list if no matching questions — frontend hides the exam block.
    """
    from database import CourseOutline
    db = SessionLocal()
    try:
        outline = db.query(CourseOutline).filter(
            CourseOutline.user_id == user.id,
            CourseOutline.folder_name == folder_name,
        ).first()
        if not outline:
            return {"questions": [], "section_title": ""}

        sections = json.loads(outline.outline_json)
        idx = outline.current_section
        if idx >= len(sections):
            return {"questions": [], "section_title": ""}

        section = sections[idx]
        section_title = section.get("title", "")
        key_topics = [t.lower() for t in section.get("key_topics", [])]
        objectives = [o.lower() for o in section.get("learning_objectives", [])]

        search_terms = key_topics + objectives + [section_title.lower()]
        search_words = set()
        for term in search_terms:
            for word in term.split():
                if len(word) > 3:
                    search_words.add(word)

        from curated_config import get_course_for_folder
        course_code = get_course_for_folder(folder_name)
        if not course_code:
            return {"questions": [], "section_title": section_title}
        papers = db.query(Paper).filter(Paper.course == course_code).all()

        answered_ids = set()
        answered_rows = (
            db.query(SessionAnswer.question_id)
            .join(QuizSession)
            .filter(QuizSession.user_id == user.id)
            .all()
        )
        for row in answered_rows:
            answered_ids.add(row[0])

        scored = []
        for paper in papers:
            questions = json.loads(paper.questions_json)
            for q in questions:
                compound_id = f"{paper.paper_id}_{q.get('id', '')}"
                if compound_id in answered_ids or q.get("id", "") in answered_ids:
                    continue

                q_tags = [t.lower() for t in (q.get("tags", []) or [])]
                q_key_terms = [t.lower() for t in (q.get("keyTerms", []) or [])]
                q_text_lower = q.get("text", "").lower()

                score = 0
                for topic in key_topics:
                    for tag in q_tags:
                        if topic in tag or tag in topic:
                            score += 10
                    for kt in q_key_terms:
                        if topic in kt or kt in topic:
                            score += 5
                    if topic in q_text_lower:
                        score += 3
                for word in search_words:
                    if any(word in tag for tag in q_tags):
                        score += 2
                    if word in q_text_lower:
                        score += 1

                if score > 0:
                    scored.append((q, paper, score))

        scored.sort(key=lambda x: -x[2])
        top = scored[:max_questions]

        result_questions = []
        for q, paper, sc in top:
            out = {**q}
            out["_paper_id"] = paper.paper_id
            out["_paper_title"] = paper.title
            out["_match_score"] = sc
            result_questions.append(out)

        return {
            "questions": result_questions,
            "section_title": section_title,
            "section_topics": key_topics,
        }
    finally:
        db.close()


@app.post("/api/folders/{folder_name}/section-answers")
def submit_section_answers(
    folder_name: str,
    request: dict = {},
    user: User = Depends(get_current_user),
):
    """Submit answers for section past-paper questions and track per-topic results."""
    answers = request.get("answers", [])
    if not answers:
        return {"status": "no_answers"}

    db = SessionLocal()
    try:
        session = QuizSession(
            user_id=user.id,
            paper_id=f"lesson_{folder_name}",
            paper_title=f"Lesson: {folder_name}",
            score=sum(1 for a in answers if a.get("is_correct")),
            total=len(answers),
            completed=True,
            completed_at=datetime.now(timezone.utc),
        )
        db.add(session)
        db.flush()

        for a in answers:
            sa = SessionAnswer(
                session_id=session.id,
                question_id=a.get("question_id", ""),
                question_text=a.get("question_text", ""),
                user_answer=a.get("user_answer", ""),
                correct_answer=a.get("correct_answer", ""),
                is_correct=a.get("is_correct", False),
                tags_json=json.dumps(a.get("tags", [])),
            )
            db.add(sa)

        db.commit()

        def _bg_update_skills():
            try:
                tutor.update_skill_profile(user.id)
            except Exception:
                import traceback as tb
                tb.print_exc()
        threading.Thread(target=ai_usage.carry(_bg_update_skills), daemon=True).start()

        return {
            "status": "recorded",
            "score": session.score,
            "total": session.total,
            "session_id": session.id,
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
def evaluate_answer(req: EvaluateAnswerRequest, user: User = Depends(get_current_user)):
    """Use GPT-4o-mini to evaluate an open-ended answer."""
    from openai import OpenAI

    limiter.check(f"evaluate:{user.id}", 60, HOUR)

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
        import provider_capacity
        response = provider_capacity.call('openai', lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.2,
        ), priority='interactive')
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


def _local_day(ts, offset_min):
    """The calendar day of a stored UTC timestamp for someone offset_min minutes east of UTC."""
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return (ts + timedelta(minutes=offset_min)).date()


def study_days(db, user_id, offset_min=0, since_days=400):
    """Local days on which the user studied: wrote to Pedro (lessons, workshops, chat),
    finished a section, a quiz, a flashcard review or a treasure quiz, or a focus block."""
    from database import ReviewHistory, SectionRewardClaim, TreasureChestOpen
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=since_days)
    stamps = []
    for column, *filters in (
        (ChatMessage.created_at, ChatMessage.user_id == user_id, ChatMessage.role == "user"),
        (QuizSession.completed_at, QuizSession.user_id == user_id, QuizSession.completed == True),  # noqa: E712
        (SectionRewardClaim.created_at, SectionRewardClaim.user_id == user_id),
        (ReviewHistory.reviewed_at, ReviewHistory.user_id == user_id),
        (TreasureChestOpen.opened_at, TreasureChestOpen.user_id == user_id),
        (ActivityEvent.created_at, ActivityEvent.user_id == user_id,
         ActivityEvent.feature == "focus", ActivityEvent.action == "complete"),
    ):
        stamps += [t for (t,) in db.query(column).filter(*filters, column != None, column >= cutoff).distinct()]  # noqa: E711
    return {_local_day(t, offset_min) for t in stamps}


def study_streak(days, today):
    """Consecutive study days ending today, or ending yesterday while today is still open."""
    check = today if today in days else today - timedelta(days=1)
    streak = 0
    while check in days:
        streak += 1
        check -= timedelta(days=1)
    return streak


@app.get("/api/stats")
def get_user_stats(tz_offset: int = 0, user: User = Depends(get_current_user)):
    """Aggregate stats for the user's dashboard, including streak.

    tz_offset: the browser's minutes east of UTC, so days roll over at local midnight.
    """
    tz_offset = max(-840, min(840, tz_offset))
    db = SessionLocal()
    try:
        sessions = db.query(QuizSession).filter(QuizSession.user_id == user.id, QuizSession.completed == True).all()
        total_sessions = len(sessions)
        total_questions = sum(s.total for s in sessions)
        total_correct = sum(s.score for s in sessions)
        avg_score = (total_correct / total_questions * 100) if total_questions > 0 else 0

        today = _local_day(datetime.now(timezone.utc), tz_offset)
        active_dates = study_days(db, user.id, tz_offset)
        streak = study_streak(active_dates, today)

        # Build week activity (last 7 days, Mon-Sun aligned to current week)
        # Find the Monday of current week
        monday = today - timedelta(days=today.weekday())
        week_days = []
        day_labels = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']
        for i in range(7):
            d = monday + timedelta(days=i)
            if d in active_dates:
                status = 'active'
            elif d >= today:
                status = 'future'  # today is still open
            else:
                status = 'missed'
            week_days.append({'label': day_labels[i], 'status': status})

        return {
            "total_sessions": total_sessions,
            "total_questions": total_questions,
            "total_correct": total_correct,
            "average_score": round(avg_score, 1),
            "streak": streak,
            "studied_today": today in active_dates,
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
def list_folders(detail: bool = False, user: User = Depends(get_current_user)):
    """Folder names for this user; with ?detail=true, [{name, kind}] (lesson or workshop)."""
    db = SessionLocal()
    try:
        rows = db.query(StudyFolder).filter(StudyFolder.user_id == user.id).order_by(StudyFolder.created_at).all()
        if detail:
            return [{"name": r.name, "kind": r.kind or "lesson"} for r in rows]
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
    kind: Optional[str] = "lesson"  # "lesson" or "workshop"


@app.post("/api/notebooks/folders")
def create_folder(req: CreateFolderRequest, user: User = Depends(get_current_user)):
    """Create and persist a new folder."""
    name = req.name.strip()[:100]
    if not name:
        raise HTTPException(400, "Folder name cannot be empty")
    kind = (req.kind or "lesson").strip().lower()
    if kind not in ("lesson", "workshop"):
        raise HTTPException(400, "Choose a lesson or a workshop")
    db = SessionLocal()
    try:
        existing = db.query(StudyFolder).filter(
            StudyFolder.user_id == user.id, StudyFolder.name == name
        ).first()
        if existing:
            return {"folder": name, "kind": existing.kind or "lesson"}
        import plans
        premade = _curated_uid(name) is not None  # opening a premade lesson or workshop is free
        if not premade:
            plans.check(db, user, "lessons")
        from coast_content_oma.course_identity import register
        register(db, user.id, name)
        folder = StudyFolder(user_id=user.id, name=name, kind=kind)
        db.add(folder)
        if not premade:
            plans.record(db, user.id, "lessons", name)
        db.commit()
        return {"folder": name, "kind": kind}
    finally:
        db.close()


class RenameFolderRequest(BaseModel):
    new_name: str


@app.put("/api/notebooks/folders/{folder_name}/rename")
def rename_folder(folder_name: str, req: RenameFolderRequest, user: User = Depends(get_current_user)):
    """Rename a folder and update all notebooks that reference it."""
    new_name = req.new_name.strip()
    if not new_name:
        raise HTTPException(400, "Name cannot be empty")

    db = SessionLocal()
    try:
        folder = db.query(StudyFolder).filter(
            StudyFolder.user_id == user.id, StudyFolder.name == folder_name
        ).first()
        if not folder:
            raise HTTPException(404, "Folder not found")

        existing = db.query(StudyFolder).filter(
            StudyFolder.user_id == user.id, StudyFolder.name == new_name
        ).first()
        if existing:
            raise HTTPException(409, "A folder with that name already exists")

        from database import CourseChatEpoch, CourseIdentity, SectionVerification, SectionRewardClaim, MapTileProvenance, PlacementTestSession, LearningJob
        from coast_content_oma.course_identity import register
        running = db.query(LearningJob).filter_by(status='running').all()
        if any((p := json.loads(job.payload_json)).get('user_id') == user.id and p.get('folder') == folder_name for job in running):
            raise HTTPException(409, "Pedro is saving this lesson. Retry the rename in a moment.")
        register(db, user.id, folder_name)
        identity = db.get(CourseIdentity, (user.id, folder_name))
        if db.get(CourseIdentity, (user.id, new_name)):
            raise HTTPException(409, "That course name is already reserved by learning history")
        from database import SourceChatTurn
        import time as _time
        if db.query(SourceChatTurn).filter_by(user_id=user.id, folder_name=folder_name, status='running').filter(SourceChatTurn.started_at > _time.time() - 300).first():
            raise HTTPException(409, "Wait for the source answer to finish before renaming this lesson.")
        db.query(SourceChatTurn).filter_by(user_id=user.id, folder_name=folder_name).update({'folder_name': new_name})
        identity.folder_name = new_name
        import plans
        plans.rename_lesson(db, user.id, folder_name, new_name)
        db.query(ChatMessage).filter(ChatMessage.user_id == user.id, ChatMessage.context_id == folder_name,
            ChatMessage.context_type.in_(['lesson','folder','test_out','sources'])).update({ChatMessage.context_id: new_name}, synchronize_session=False)
        for model in (CourseChatEpoch, SectionVerification, SectionRewardClaim, MapTileProvenance, PlacementTestSession):
            db.query(model).filter(model.user_id == user.id, model.folder_name == folder_name).update({model.folder_name: new_name}, synchronize_session=False)
        for job in db.query(LearningJob).filter(LearningJob.status.in_(['queued','failed'])).all():
            payload = json.loads(job.payload_json)
            if payload.get('user_id') == user.id and payload.get('folder') == folder_name:
                payload['folder'] = new_name
                job.payload_json = json.dumps(payload)
        folder.name = new_name
        db.query(SavedNotebook).filter(
            SavedNotebook.user_id == user.id, SavedNotebook.folder == folder_name
        ).update({SavedNotebook.folder: new_name})
        db.query(FolderSource).filter(
            FolderSource.user_id == user.id, FolderSource.folder_name == folder_name
        ).update({FolderSource.folder_name: new_name})
        db.query(SourceImage).filter(
            SourceImage.user_id == user.id, SourceImage.folder_name == folder_name
        ).update({SourceImage.folder_name: new_name})
        db.query(LessonNotes).filter(
            LessonNotes.user_id == user.id, LessonNotes.folder_name == folder_name
        ).update({LessonNotes.folder_name: new_name})
        from database import CourseOutline
        db.query(CourseOutline).filter(
            CourseOutline.user_id == user.id, CourseOutline.folder_name == folder_name
        ).update({CourseOutline.folder_name: new_name})
        db.commit()
        import map_world
        map_world.invalidate_map_cache(user.id)
        return {"status": "renamed", "old_name": folder_name, "new_name": new_name}
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
            import plans
            plans.give_back_unused_lesson(db, user.id, folder_name)
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
        # Curated folders are pre-embedded at bootstrap; skip instead of letting
        # any user trigger an expensive shared re-embed.
        if _curated_uid(folder_name) is not None and not is_admin(user):
            return {"notebooks_embedded": 0, "total_chunks": 0, "skipped": "curated"}
        embed_uid = _curated_uid(folder_name) if _curated_uid(folder_name) is not None else user.id
        import oma_provider
        if oma_provider.is_oma_enabled():
            return {"notebooks_embedded": 0, "total_chunks": 0, "skipped": "oma"}
        return rag.embed_all_in_folder(embed_uid, folder_name)
    except Exception:
        import traceback
        traceback.print_exc()
        return {"notebooks_embedded": 0, "total_chunks": 0, "error": "Embedding failed"}

class UploadFileIntent(BaseModel):
    upload_id: str
    filename: str
    size_bytes: int


class UploadBatchIntent(BaseModel):
    files: list[UploadFileIntent]


@app.post("/api/folders/{folder_name}/uploads")
def reserve_folder_uploads(folder_name: str, body: UploadBatchIntent, user: User = Depends(get_current_user)):
    _require_curated_write_access(folder_name, user)
    import upload_lifecycle
    return {"uploads": upload_lifecycle.reserve(user.id, folder_name, [f.model_dump() for f in body.files])}


@app.get("/api/folders/{folder_name}/uploads")
def get_folder_uploads(folder_name: str, user: User = Depends(get_current_user)):
    import upload_lifecycle
    return {"uploads": upload_lifecycle.list_uploads(user.id, folder_name)}


@app.delete("/api/folders/{folder_name}/uploads/{upload_id}")
def cancel_folder_upload(folder_name: str, upload_id: str, user: User = Depends(get_current_user)):
    _require_curated_write_access(folder_name, user)
    import upload_lifecycle
    upload_lifecycle.cancel(user.id, folder_name, upload_id)
    return {"status": "cancelled"}


# Reading a PDF holds every page (and its images) in memory, so only a couple
# run at once; further uploads wait their turn instead of exhausting RAM.
_EXTRACT_SLOTS = threading.BoundedSemaphore(max(1, int(os.getenv("COAST_EXTRACT_CONCURRENCY", "2"))))
_READ_SLOTS = None


async def _read_upload(path: str) -> list[dict]:
    """Read an uploaded PDF or PowerPoint in a separate Python process (the PDF readers are
    pure Python and would otherwise freeze every other request for tens of seconds), saving
    the page copy that indexing loads. Returns the pages' text. With containers on, a container
    reads it instead, and the server only waits."""
    from coast_content_oma import remote
    if remote.enabled():
        try:
            return await remote.read_upload(path)
        except Exception as exc:
            print(f"[upload] reading {Path(path).name} in a container failed ({type(exc).__name__}: {exc}); reading it here")
    global _READ_SLOTS
    if _READ_SLOTS is None:
        _READ_SLOTS = asyncio.Semaphore(max(1, int(os.getenv("COAST_EXTRACT_CONCURRENCY", "2"))))
    async with _READ_SLOTS:
        import memory_budget
        waited = 0.0
        while not memory_budget.has_room(memory_budget.READ_UPLOAD_MB) and waited < 600:
            await asyncio.sleep(1)  # wait for room rather than push the machine over its limit
            waited += 1
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "coast_content_oma.read_upload", path,
            cwd=str(Path(__file__).parent), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=float(os.getenv("COAST_EXTRACT_TIMEOUT", "600")))
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError("reading the file took too long")
    if proc.returncode != 0:
        raise RuntimeError(err.decode(errors="replace")[-400:])
    return json.loads(out)


@app.post("/api/folders/{folder_name}/upload")
async def upload_folder_source(
    folder_name: str,
    file: UploadFile = File(...),
    upload_id: Optional[str] = Form(None),
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

    _require_curated_write_access(folder_name, user)

    if not file.filename:
        raise HTTPException(400, "No filename provided")

    ext = Path(file.filename).suffix.lower()
    allowed = {".pdf", ".png", ".jpg", ".jpeg", ".pptx", ".tiff", ".bmp", ".webp"}
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    tmp_path = stored_path = None
    claim = None
    committed = False
    try:
        import upload_lifecycle
        if (getattr(file, "size", None) or 0) > upload_lifecycle.MAX_UPLOAD_BYTES:
            raise HTTPException(413, upload_lifecycle.TOO_LARGE)
        # Spool to disk in chunks so a large upload never sits in memory whole.
        size = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = Path(tmp.name)
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > upload_lifecycle.MAX_UPLOAD_BYTES:
                    raise HTTPException(413, upload_lifecycle.TOO_LARGE)
                tmp.write(chunk)
        # Older API callers can still upload directly; the website registers the whole batch first.
        if not upload_id:
            upload_id = uuid.uuid4().hex
            upload_lifecycle.reserve(user.id, folder_name, [{"upload_id": upload_id,
                "filename": file.filename, "size_bytes": size}])
        claim, replay = upload_lifecycle.begin(user.id, folder_name, upload_id, file.filename, size)
        if replay:
            return replay
        # After begin(), so a refused file shows as failed in the upload list and can be removed.
        _security.check_upload(tmp_path, ext)

        raw_text = ""
        page_count = 0
        source_type = ext.lstrip(".")

        import asyncio

        if ext not in (".pdf", ".pptx"):
            raise HTTPException(400, "Image files must be uploaded via the full notebook pipeline")

        source_id = f"src_{uuid.uuid4().hex[:10]}"
        title = Path(file.filename).stem.replace("_", " ").replace("-", " ")
        stored_path = FOLDER_UPLOADS_DIR / f"{source_id}{ext}"
        shutil.copy2(str(tmp_path), str(stored_path))

        # Read the file once, in a separate process: text for this reply, and the saved page
        # copy (figures included) that indexing loads in about a second.
        try:
            pages = await _read_upload(str(stored_path))
        except Exception as exc:
            print(f"[upload] Could not read {ext}: {type(exc).__name__}")
            raise HTTPException(400, "This file could not be read. Try saving a new PDF or PowerPoint (.pptx) copy.") from exc

        if not pages:
            raise HTTPException(400, "Could not extract text from this file")
        page_count = len(pages)
        raw_text = "\n\n".join(p.get("text", "") for p in pages if p.get("text"))
        if not raw_text.strip():
            raise HTTPException(400, "No text could be extracted from this file")

        db = SessionLocal()
        try:
            upload_lifecycle.finish(db, user.id, folder_name, upload_id, claim, source_id)
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
            from coast_content_oma.course_identity import register
            register(db, user.id, folder_name)
            db.add(fs)
            db.flush()
            import oma_provider
            if oma_provider.is_oma_enabled():
                from learning_jobs import enqueue_source
                enqueue_source(db, fs)
            db.commit()
            committed = True
        finally:
            db.close()
        import file_store  # the file and its page copy to R2; the disk keeps a cached copy
        from coast_content_oma.normalized_source import cache_dir as _page_copy_dir
        file_store.publish(stored_path)
        file_store.publish_tree(_page_copy_dir(stored_path))

        if _curated_uid(folder_name) is None:
            import source_search
            source_search.schedule(source_id)

        def _bg_embed():
            try:
                rag.embed_raw_source(user.id, folder_name, source_id, title, raw_text)
                print(f"[BG] Embedded source {source_id} for folder {folder_name}")
            except Exception:
                import traceback
                traceback.print_exc()

        def _bg_images():
            from image_extractor import extract_and_store_images
            with _EXTRACT_SLOTS:
                extract_and_store_images(stored_path, ext, source_id, user.id, folder_name, SOURCE_IMAGES_DIR)


        import oma_provider
        if oma_provider.is_oma_enabled():
            from learning_jobs import wake
            wake()
        else:
            threading.Thread(target=ai_usage.carry(_bg_embed), daemon=True).start()
            threading.Thread(target=ai_usage.carry(_bg_images), daemon=True).start()

        return {
            "source_id": source_id,
            "title": title,
            "page_count": page_count,
            "filename": file.filename,
        }
    except HTTPException as exc:
        if claim:
            upload_lifecycle.fail(user.id, folder_name, upload_id, claim, exc.detail)
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        if claim:
            upload_lifecycle.fail(user.id, folder_name, upload_id, claim, "Upload failed. Retry this file.")
        return JSONResponse(status_code=500, content={"detail": "Upload failed: server error"},
                            headers=_cors_headers)
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)
        if stored_path and not committed:
            stored_path.unlink(missing_ok=True)
            shutil.rmtree(str(stored_path) + '.pages', ignore_errors=True)


@app.delete("/api/folders/{folder_name}/sources/{source_id}")
def delete_folder_source(folder_name: str, source_id: str, user: User = Depends(get_current_user)):
    """Delete a raw source from a folder."""
    _require_curated_write_access(folder_name, user)
    owner_id = _curated_uid(folder_name) if _curated_uid(folder_name) is not None else user.id
    db = SessionLocal()
    file_path = None
    try:
        from sqlalchemy import text
        from database import LearningJob
        db.execute(text('BEGIN IMMEDIATE'))
        fs = db.query(FolderSource).filter(
            FolderSource.user_id == owner_id,
            FolderSource.folder_name == folder_name,
            FolderSource.source_id == source_id,
        ).first()
        if not fs:
            raise HTTPException(404, "Source not found")
        jobs = db.query(LearningJob).filter(
            text("json_extract(payload_json, '$.source_id') = :sid")
        ).params(sid=source_id).all()
        if any(job.status == 'running' for job in jobs) or fs.oma_ingest_status == 'INGESTING':
            raise HTTPException(409, 'This source is being processed. Retry deletion when processing finishes.')
        import oma_provider
        from coast_content_oma.stores.base import make_namespace
        from coast_content_oma.source_lifecycle import remove_source_material
        remove_source_material(oma_provider.OMA_DB_PATH, make_namespace(owner_id, folder_name), source_id)
        for job in jobs:
            job.status = 'done'
            job.last_error = 'Source intentionally deleted'
        file_path = fs.file_path
        from database import SourceSearchIndex
        db.query(SourceSearchIndex).filter_by(source_id=source_id).delete()
        db.delete(fs)
        # Cascade: remove extracted image rows for this source.
        try:
            from database import SourceImage
            db.query(SourceImage).filter(
                SourceImage.user_id == owner_id,
                SourceImage.source_id == source_id,
            ).delete()
        except Exception:
            pass
        db.commit()
    finally:
        db.close()

    if file_path:
        import file_store
        from coast_content_oma.normalized_source import cache_dir
        file_store.remove([file_path, cache_dir(file_path)])  # disk now, R2's trash for 30 days

    try:
        rag.delete_notebook_embeddings(owner_id, folder_name, source_id)
    except Exception:
        pass

    return {"status": "deleted", "source_id": source_id}


def _inline_disposition(filename: str) -> str:
    """Content-Disposition for a stored file: an ASCII fallback name plus the real name,
    so quotes or line breaks in an uploaded file's name can't break the header."""
    from urllib.parse import quote
    name = str(filename or "file").replace("\r", " ").replace("\n", " ")
    ascii_name = "".join(c if 32 <= ord(c) < 127 and c not in '"\\' else "_" for c in name)
    return f"inline; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(name)}"


@app.get("/api/folders/{folder_name}/sources/{source_id}/file")
def get_source_file(folder_name: str, source_id: str, user: User = Depends(get_current_user)):
    """Serve the original uploaded file for a folder source."""
    owner_id = _curated_uid(folder_name) if _curated_uid(folder_name) is not None else user.id
    db = SessionLocal()
    try:
        fs = db.query(FolderSource).filter(
            FolderSource.user_id == owner_id,
            FolderSource.folder_name == folder_name,
            FolderSource.source_id == source_id,
        ).first()
        if not fs:
            raise HTTPException(404, "Source not found")
        import file_store
        file_path = file_store.local(fs.file_path) if fs.file_path else None
        if not file_path or not file_path.exists():
            raise HTTPException(404, "File not available")
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
        headers={"Content-Disposition": _inline_disposition(filename)},
    )


def _resolve_image_path(stored_path: str) -> Path | None:
    """Find the actual image file, trying persistent disk if stored path is stale."""
    fp = Path(stored_path)
    if fp.exists():
        return fp
    _OLD_PREFIX = "/opt/render/project/src/folder_uploads"
    _NEW_PREFIX = "/data/folder_uploads"
    if stored_path.startswith(_OLD_PREFIX):
        alt = Path(_NEW_PREFIX + stored_path[len(_OLD_PREFIX):])
        if alt.exists():
            return alt
    elif stored_path.startswith(_NEW_PREFIX):
        alt = Path(_OLD_PREFIX + stored_path[len(_NEW_PREFIX):])
        if alt.exists():
            return alt
    return None


@app.get("/api/image-access")
def issue_image_access(user: User = Depends(get_current_user)):
    from auth import create_image_token
    return JSONResponse({"access": create_image_token(user.id)}, headers={"Cache-Control": "no-store"})


def get_image_user(authorization: Optional[str] = Header(None), access: str | None = None):
    from auth import decode_image_token
    payload = None
    if authorization and authorization.startswith("Bearer "):
        payload = decode_access_token(authorization.split(" ", 1)[1])
    if not payload and access:
        payload = decode_image_token(access)
    if not payload:
        raise HTTPException(401, "Sign in to view this image")
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == int(payload["sub"])).first()
        if not user:
            raise HTTPException(401, "User no longer exists")
        return user
    finally:
        db.close()


@app.get("/api/source-images/{image_id}")
def serve_source_image(image_id: int, user: User = Depends(get_image_user)):
    """Serve an extracted source image by its DB id."""
    db = SessionLocal()
    try:
        si = db.query(SourceImage).filter(SourceImage.id == image_id).first()
        if not si or (si.user_id != user.id and _curated_uid(si.folder_name) != si.user_id):
            raise HTTPException(404, "Image not found")
        resolved = _resolve_image_path(si.image_path)
        if not resolved:
            print(f"[images] File missing: {si.image_path}  (id={image_id}, source={si.source_id})")
            raise HTTPException(404, "Image file missing")
        if str(resolved) != si.image_path:
            si.image_path = str(resolved)
            db.commit()
        from fastapi.responses import FileResponse
        return FileResponse(
            path=str(resolved),
            media_type=_image_media_type(resolved),
            headers={"Cache-Control": "private, no-store", "Referrer-Policy": "no-referrer"},
        )
    finally:
        db.close()


def _image_media_type(path) -> str:
    """Figures are saved as WebP now; older ones are PNG."""
    return {".webp": "image/webp", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif"}.get(
        Path(path).suffix.lower(), "image/png")


@app.get("/api/source-pages/{source_id}/{page_number}")
def serve_source_page(source_id: str, page_number: int, user: User = Depends(get_image_user)):
    """One lecture page as an image, for slides Pedro shows in the chat (image-token auth,
    so a plain <img> can load it). Curated courses serve their owner's uploads."""
    from curated_config import CURATED_FOLDER_NAMES
    with SessionLocal() as db:
        source = db.query(FolderSource).filter_by(source_id=source_id).first()
        allowed = source and (source.user_id == user.id or (
            source.folder_name in CURATED_FOLDER_NAMES and _curated_uid(source.folder_name) == source.user_id))
        if not allowed or not source.file_path or not 1 <= page_number <= (source.page_count or 0):
            raise HTTPException(404, "Page not found")
        path = source.file_path
    import pedro_context
    rendered = pedro_context.page_image(path, page_number)
    if not rendered:
        raise HTTPException(404, "Page preview unavailable")
    from fastapi.responses import Response
    data, media = rendered
    return Response(data, media_type=media,
                    headers={"Cache-Control": "private, max-age=3600", "Referrer-Policy": "no-referrer"})


@app.get("/api/debug/images")
def debug_images(user: User = Depends(get_current_user)):
    """Debug endpoint: show image status and curated source file_paths."""
    if not is_admin(user):
        raise HTTPException(403, "Admin only")
    db = SessionLocal()
    try:
        all_imgs = db.query(SourceImage).all()
        by_folder: dict[str, dict] = {}
        for si in all_imgs:
            f = si.folder_name
            if f not in by_folder:
                by_folder[f] = {"total": 0, "exists": 0, "missing": 0, "sample_ids": [], "sample_paths": []}
            by_folder[f]["total"] += 1
            if Path(si.image_path).exists():
                by_folder[f]["exists"] += 1
            else:
                by_folder[f]["missing"] += 1
            if len(by_folder[f]["sample_ids"]) < 3:
                by_folder[f]["sample_ids"].append(si.id)
                by_folder[f]["sample_paths"].append(si.image_path)

        curated_sources = []
        from curated_config import CURATED_FOLDER_NAMES, CURATED_USER_ID
        for fn in CURATED_FOLDER_NAMES:
            sources = db.query(FolderSource).filter(
                FolderSource.user_id == CURATED_USER_ID,
                FolderSource.folder_name == fn,
            ).all()
            for s in sources[:3]:
                fp = Path(s.file_path) if s.file_path else None
                curated_sources.append({
                    "folder": fn,
                    "source_id": s.source_id,
                    "title": s.title[:50],
                    "file_path": s.file_path,
                    "file_exists": fp.exists() if fp else False,
                })

        data_dir = Path("/data")
        return {
            "images_by_folder": by_folder,
            "curated_sources_sample": curated_sources,
            "persistent_disk_exists": data_dir.is_dir(),
            "data_contents": [str(p) for p in data_dir.iterdir()] if data_dir.is_dir() else [],
        }
    finally:
        db.close()


@app.get("/api/folders/{folder_name}/images")
def list_folder_images(folder_name: str, user: User = Depends(get_current_user)):
    """List all extracted images for a folder."""
    owner_id = _curated_uid(folder_name) if _curated_uid(folder_name) is not None else user.id
    db = SessionLocal()
    try:
        images = db.query(SourceImage).filter(
            SourceImage.user_id == owner_id,
            SourceImage.folder_name == folder_name,
        ).all()
        return [
            {
                "id": si.id,
                "source_id": si.source_id,
                "page_number": si.page_number,
                "context_text": si.context_text[:200],
                "width": si.width,
                "height": si.height,
            }
            for si in images
        ]
    finally:
        db.close()


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
    fs = None
    if user:
        owner_id = _curated_uid(folder_name) if _curated_uid(folder_name) is not None else user.id
        fs = db.query(FolderSource).filter(
            FolderSource.user_id == owner_id,
            FolderSource.source_id == source_id,
            FolderSource.folder_name == folder_name,
        ).first()
    db.close()
    if not user:
        raise HTTPException(401, "User not found")
    if not fs:
        raise HTTPException(404, "Source not found")
    import asyncio
    import file_store
    if not fs.file_path or not (await asyncio.to_thread(file_store.local, fs.file_path)).exists():  # from R2 if the cache cleared it
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
            from curated_config import get_course_for_folder
            course = get_course_for_folder(folder_name)
            paper_paths = _get_paper_paths(course)
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

                threading.Thread(
                    target=ai_usage.carry(_bg_post_process),
                    args=(user.id, folder_name, nb_id, result),
                    daemon=True,
                ).start()
            finally:
                db2.close()

            progress_q.put({"stage": "done", "notebook": result})
        except Exception as exc:
            progress_q.put({"stage": "error", "message": str(exc)})

    threading.Thread(target=ai_usage.carry(run_pipeline), daemon=True).start()

    async def event_stream():
        idle_ticks = 0
        while True:
            try:
                item = progress_q.get_nowait()
                idle_ticks = 0
            except queue.Empty:
                await asyncio.sleep(0.5)
                idle_ticks += 1
                if idle_ticks % 20 == 0:
                    yield ": keepalive\n\n"
                continue
            yield f"data: {json.dumps(item)}\n\n"
            if item.get("stage") in ("done", "error"):
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/folders/{folder_name}/sources")
def folder_sources(folder_name: str, user: User = Depends(get_current_user)):
    """List all sources (notebooks + raw documents) with metadata."""
    src_uid = _curated_uid(folder_name) if _curated_uid(folder_name) is not None else user.id
    try:
        chroma_sources = rag.get_folder_sources(src_uid, folder_name)
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
                "oma_ingest_status": getattr(src, "oma_ingest_status", None) or "PENDING",
            })

        return {"sources": all_sources, "embedding_stats": chroma_sources}
    finally:
        db.close()


@app.get("/api/folders/{folder_name}/oma-ingest")
def folder_oma_ingest_status(folder_name: str, user: User = Depends(get_current_user)):
    """Per-source OMA ingest progress (pages, vision, concepts, roadmap readiness)."""
    import oma_provider
    src_uid = _curated_uid(folder_name) if _curated_uid(folder_name) is not None else user.id
    return oma_provider.get_folder_ingest_progress(src_uid, folder_name)

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

class OutlineSourceSelection(BaseModel):
    source_ids: Optional[list[str]] = None


@app.post("/api/folders/{folder_name}/outline")
def generate_outline(folder_name: str, body: Optional[OutlineSourceSelection] = None, user: User = Depends(get_current_user)):
    """Generate or regenerate a course outline from folder sources."""
    src_uid = _curated_uid(folder_name) if _curated_uid(folder_name) is not None else user.id
    structure = get_lesson_structure(folder_name)
    result = lesson.generate_outline(user.id, folder_name, source_user_id=src_uid, structure=structure,
                                     expected_source_ids=body.source_ids if body else None,
                                     course_format=lesson.folder_kind(user.id, folder_name))
    if "error" in result:
        raise HTTPException(result.get("status_code", 400), result["error"])
    return result


@app.post("/api/folders/{folder_name}/prepare-curated")
def prepare_curated_lesson(folder_name: str, user: User = Depends(get_current_user)):
    """Index premade lesson sources (Content OMA + RAG) and seed the static outline."""
    from curated_config import prepare_curated_lesson as _prepare
    if _curated_uid(folder_name) is None:
        raise HTTPException(400, "Not a premade lesson folder.")
    result = _prepare(user.id, folder_name)
    if result.get("error"):
        raise HTTPException(400, result["error"])
    return result


@app.get("/api/folders/{folder_name}/lesson/sections/{section_index}/constellation")
def get_section_constellation(
    folder_name: str,
    section_index: int,
    user: User = Depends(get_current_user),
):
    """Per-section mastery constellation for the lesson map."""
    src_uid = _curated_uid(folder_name) if _curated_uid(folder_name) is not None else user.id
    result = lesson.get_section_constellation(
        user.id, folder_name, section_index, source_user_id=src_uid,
    )
    if result.get("error"):
        raise HTTPException(400, result["error"])
    return result


@app.get("/api/folders/{folder_name}/lesson")
def get_lesson_state(folder_name: str, user: User = Depends(get_current_user)):
    """Get current lesson state — outline, progress, current section."""
    src_uid = _curated_uid(folder_name) if _curated_uid(folder_name) is not None else user.id
    state = lesson.get_lesson_state(user.id, folder_name, source_user_id=src_uid)
    if isinstance(state, dict) and state.get("has_outline"):
        # Lets any device say "Continue lesson" once the current section has begun.
        from database import ChatMessage
        with SessionLocal() as db:
            state["section_started"] = db.query(ChatMessage.id).filter(
                ChatMessage.user_id == user.id, ChatMessage.context_type == "lesson",
                ChatMessage.context_id == folder_name,
                ChatMessage.section_index == state.get("current_section", 0),
            ).first() is not None
    return state


@app.get("/api/lessons/summary")
def lesson_summaries(user: User = Depends(get_current_user)):
    """One small library/continue response, without per-course OMA scans."""
    from database import CourseOutline, SectionRewardClaim, SectionVerification, ChatMessage
    from sqlalchemy import func
    with SessionLocal() as db:
        outlines = db.query(CourseOutline).filter_by(user_id=user.id).all()
        claims = {(r.folder_name, r.section_index) for r in db.query(SectionRewardClaim).filter_by(user_id=user.id).all()}
        verified = {(r.folder_name, r.section_index) for r in db.query(SectionVerification).filter_by(user_id=user.id, is_active=True).all()}
        recent = dict(db.query(ChatMessage.context_id, func.max(ChatMessage.created_at)).filter(
            ChatMessage.user_id == user.id, ChatMessage.context_type == "lesson"
        ).group_by(ChatMessage.context_id).all())
        result = {}
        for outline in outlines:
            sections = json.loads(outline.outline_json or "[]")
            progress = []
            for i in range(len(sections)):
                done = i < outline.current_section or (outline.folder_name, i) in claims or (outline.folder_name, i) in verified
                progress.append({"mastery_pct": 100 if done else 0, "mastered": done, "attempted": done or i == outline.current_section})
            complete = bool(sections) and all(p["mastered"] for p in progress)
            last = recent.get(outline.folder_name)
            result[outline.folder_name] = {"has_outline": True, "current_section": outline.current_section,
                "total_sections": len(sections), "is_complete": complete, "ever_mastered": bool(outline.ever_mastered) or complete,
                "section_progress": progress, "current_section_title": sections[outline.current_section].get("title", "") if outline.current_section < len(sections) else "",
                "last_studied_at": last.isoformat() if last else None,
                "updated_at": outline.updated_at.isoformat() if outline.updated_at else None}
        return result


@app.get("/api/map")
def get_world_map(user: User = Depends(get_current_user), compact: bool = False):
    """Exploration map with fog of war."""
    import map_world
    return map_world.get_map_state(user.id, compact=compact)


class MapMoveRequest(BaseModel):
    dx: int = 0
    dy: int = 0


@app.post("/api/map/move")
def move_on_map(req: MapMoveRequest, user: User = Depends(get_current_user)):
    import map_world
    return map_world.move_player(user.id, req.dx, req.dy)


class MapTeleportRequest(BaseModel):
    x: int
    y: int


@app.post("/api/map/teleport")
def teleport_on_map(req: MapTeleportRequest, user: User = Depends(get_current_user)):
    import map_world
    return map_world.teleport_player(user.id, req.x, req.y)


@app.get("/api/map/treasures/{chest_id}/quiz")
def treasure_quiz(chest_id: str, user: User = Depends(get_current_user)):
    import treasure
    result = treasure.build_quiz(user.id, chest_id)
    if result.get("error") and result.get("error") not in ("already_opened",):
        code = 400 if result.get("error") != "not_enough_topics" else 422
        raise HTTPException(code, result.get("message") or result["error"])
    return result


@app.post("/api/map/treasures/{chest_id}/complete")
def treasure_complete(chest_id: str, body: dict, user: User = Depends(get_current_user)):
    import treasure
    answer = body.get("answer") or ""
    result = treasure.complete_treasure(user.id, chest_id, answer)
    if result.get("error") == "already_opened":
        raise HTTPException(409, "Treasure chest already opened")
    if result.get("error"):
        raise HTTPException(400, result["error"])
    return result


@app.post("/api/folders/{folder_name}/lesson/test-out")
def apply_test_out_endpoint(
    folder_name: str,
    user: User = Depends(get_current_user),
    body: dict | None = None,
):
    """Jump to a future section after passing a placement test with Pedro."""
    body = body or {}
    target = body.get("target_section")
    if target is None:
        raise HTTPException(400, "target_section is required")
    result = lesson.apply_test_out(user.id, folder_name, int(target), body.get("conversation_id"))
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.post("/api/folders/{folder_name}/lesson/advance")
def advance_lesson(folder_name: str, user: User = Depends(get_current_user)):
    """Advance to the next lesson section."""
    result = lesson.advance_section(user.id, folder_name)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.post("/api/folders/{folder_name}/lesson/section-reward")
def claim_section_reward_endpoint(
    folder_name: str,
    user: User = Depends(get_current_user),
    body: dict | None = None,
):
    """Grant XP + map expansion when Pedro marks a section complete."""
    body = body or {}
    section_index = body.get("section_index")
    if section_index is not None:
        section_index = int(section_index)
    result = lesson.claim_section_reward(user.id, folder_name, section_index=section_index)
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


@app.get("/api/folders/{folder_name}/lesson-notes")
def get_lesson_notes(folder_name: str, user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        row = db.query(LessonNotes).filter(
            LessonNotes.user_id == user.id,
            LessonNotes.folder_name == folder_name,
        ).first()
        return {"content_html": sanitize_notes(row.content_html if row else ""),
                "revision": notes_revision(row.content_html if row else "")}
    finally:
        db.close()


@app.put("/api/folders/{folder_name}/lesson-notes")
def save_lesson_notes(folder_name: str, body: dict, user: User = Depends(get_current_user)):
    html = sanitize_notes(body.get("content_html", ""))
    if "revision" not in body:
        raise HTTPException(428, "Reload notes before saving: a revision is required")
    db = SessionLocal()
    try:
        from sqlalchemy import text
        db.execute(text("BEGIN IMMEDIATE"))
        row = db.query(LessonNotes).filter(
            LessonNotes.user_id == user.id,
            LessonNotes.folder_name == folder_name,
        ).first()
        current_html = row.content_html if row else ""
        if body.get("revision") != notes_revision(current_html) and html != current_html:
            raise HTTPException(409, "Notes changed in another tab. Your draft was not overwritten.")
        if row:
            row.content_html = html
            row.updated_at = datetime.now(timezone.utc)
        else:
            row = LessonNotes(
                user_id=user.id,
                folder_name=folder_name,
                content_html=html,
            )
            db.add(row)
        db.commit()
        return {"ok": True, "revision": notes_revision(html)}
    finally:
        db.close()


@app.get("/api/lesson-notes/all")
def list_all_lesson_notes(user: User = Depends(get_current_user)):
    """All saved lesson notes for the current user (notes workspace)."""
    db = SessionLocal()
    try:
        rows = (
            db.query(LessonNotes)
            .filter(LessonNotes.user_id == user.id)
            .order_by(LessonNotes.updated_at.desc())
            .all()
        )
        return {
            "notes": [
                {
                    "folder_name": r.folder_name,
                    "content_html": sanitize_notes(r.content_html or ""),
                    "revision": notes_revision(r.content_html or ""),
                    "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                }
                for r in rows
            ],
        }
    finally:
        db.close()


@app.get("/api/folders/{folder_name}/all-feedback")
def get_all_feedback(folder_name: str, user: User = Depends(get_current_user)):
    return lesson.get_all_section_feedback(user.id, folder_name)


class SectionFeedbackRequest(BaseModel):
    section_index: int
    section_title: str = ""


@app.post("/api/folders/{folder_name}/section-feedback")
def post_section_feedback(
    folder_name: str,
    body: SectionFeedbackRequest,
    user: User = Depends(get_current_user),
):
    """Generate structured feedback for a completed lesson section."""
    return lesson.generate_section_feedback(
        user.id,
        folder_name,
        int(body.section_index),
        body.section_title or "",
    )


@app.get("/api/folders/{folder_name}/section-chat/{section_index}")
def get_section_chat(folder_name: str, section_index: int, user: User = Depends(get_current_user), resume: bool = False):
    db = SessionLocal()
    try:
        rows = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.user_id == user.id,
                ChatMessage.context_type == "lesson",
                ChatMessage.context_id == folder_name,
                ChatMessage.section_index == section_index,
            )
            .order_by(ChatMessage.created_at.asc())
            .all()
        )
        if resume:
            from database import CourseChatEpoch
            epoch = db.get(CourseChatEpoch, (user.id, folder_name))
            rows = [r for r in rows if r.id > (epoch.through_message_id if epoch else 0)]
            # A section can contain retries/review conversations. Resume the latest one.
            if rows:
                conversation_id = rows[-1].conversation_id
                rows = [r for r in rows if r.conversation_id == conversation_id]
        from coast_content_oma.student.grading import strip_ui_tags
        # Grading tags stay in the stored transcript for evaluation; students never see them.
        messages = [{"role": r.role, "content": strip_ui_tags(r.content).strip() if r.role == "pedro" else r.content}
                    for r in rows]
        return {"messages": messages, "conversation_id": rows[-1].conversation_id if rows else None}
    finally:
        db.close()


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

        folder = notebook.get("folder", "")
        threading.Thread(
            target=ai_usage.carry(_bg_post_process),
            args=(user.id, folder, nb_id, notebook),
            daemon=True,
        ).start()

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


# Direct source Q&A intentionally bypasses lesson gates and Student OMA.
class SourceQuestionRequest(BaseModel):
    message: str
    request_id: uuid.UUID
    conversation_id: Optional[str] = None


def _source_workspace(folder_name):
    if _curated_uid(folder_name) is not None:
        raise HTTPException(404, "Ask sources is available for your uploaded lessons.")


@app.get("/api/folders/{folder_name}/ask-sources/status")
def source_question_status(folder_name: str, user: User = Depends(get_current_user)):
    _source_workspace(folder_name)
    import source_search
    return source_search.status(user.id, folder_name)


@app.get("/api/folders/{folder_name}/ask-sources/conversations")
def source_conversations(folder_name: str, user: User = Depends(get_current_user)):
    _source_workspace(folder_name)
    import source_chat
    return source_chat.conversations(user.id, folder_name)


@app.get("/api/folders/{folder_name}/ask-sources/history")
def source_conversation_history(folder_name: str, conversation_id: str, user: User = Depends(get_current_user)):
    _source_workspace(folder_name)
    import source_chat
    return source_chat.history(user.id, folder_name, conversation_id)


@app.post("/api/folders/{folder_name}/ask-sources")
def ask_sources(folder_name: str, req: SourceQuestionRequest, user: User = Depends(get_current_user)):
    _source_workspace(folder_name)
    question = req.message.strip()
    if not question or len(question) > 6000:
        raise HTTPException(400, "Enter a question of 1–6,000 characters.")
    import source_chat
    # Retries reuse the original question and do not consume another message.
    from database import SourceChatTurn
    with SessionLocal() as db:
        retry = db.get(SourceChatTurn, (user.id, str(req.request_id))) is not None
    if not retry:
        _check_messages(user)
    claim = source_chat.begin(user.id, folder_name, question, str(req.request_id), req.conversation_id)
    if not retry:
        _count_message(user, str(req.request_id))
    def events():
        for event in source_chat.stream_answer(user.id, folder_name, question, claim):
            yield f"data: {json.dumps(event)}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/folders/{folder_name}/sources/{source_id}/pages/{page_number}")
def source_page_preview(folder_name: str, source_id: str, page_number: int, user: User = Depends(get_current_user)):
    # Rendering a single original PDF page makes citations reliable in all browsers.
    with SessionLocal() as db:
        source = db.query(FolderSource).filter_by(user_id=user.id, folder_name=folder_name, source_id=source_id).first()
        import file_store
        if not source or not source.file_path or not file_store.local(source.file_path).is_file():  # from R2 if the cache cleared it
            raise HTTPException(404, "Source no longer available.")
        path, count, kind = source.file_path, source.page_count, source.source_type
    if not 1 <= page_number <= count:
        raise HTTPException(404, "Page not found.")
    from fastapi.responses import Response
    if kind == 'pdf':
        import fitz
        with fitz.open(path) as doc:
            page = doc.load_page(page_number - 1)
            scale = min(1.6, 1800 / max(page.rect.width, page.rect.height, 1))
            png = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).tobytes('png')
        return Response(png, media_type='image/png', headers={'Cache-Control': 'private, max-age=300'})
    from coast_content_oma.normalized_source import load_pages
    pages = load_pages(path, extract_images=False) or []
    page = next((p for p in pages if p['page_number'] == page_number), None)
    if not page:
        raise HTTPException(404, "Slide preview unavailable. Download the original PowerPoint.")
    return {'text': page.get('text', ''), 'page': page_number, 'source_type': kind}


# ═══════════════════════════════════════════════════════════════════════════
# PEDRO CHAT (AI TUTOR)
# ═══════════════════════════════════════════════════════════════════════════

class ChatSendRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    context_type: str = "global"  # "notebook", "global", "session"
    context_id: Optional[str] = None
    notebook_ids: Optional[list[str]] = None
    section_index: Optional[int] = None
    concept_id: Optional[str] = None


@app.post("/api/chat/send")
def chat_send(req: ChatSendRequest, user: User = Depends(get_current_user)):
    """Send a message to Pedro and get a Socratic response."""
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")
    if req.context_type not in ("notebook", "global", "session", "folder", "lesson", "test_out", "onboarding"):
        raise HTTPException(400, "Invalid context_type")

    if req.context_type in ('lesson', 'test_out') and req.context_id:
        from coast_content_oma.progressive import assert_chat_ready
        assert_chat_ready(user.id,req.context_id,req.section_index,test_out=req.context_type == 'test_out')

    if req.context_type != "onboarding":  # the welcome chat is free
        _check_messages(user)
    counted = _counts_as_message(req)

    try:
        result = tutor.send_message(
            user_id=user.id,
            message=req.message.strip(),
            conversation_id=req.conversation_id,
            context_type=req.context_type,
            context_id=req.context_id,
            notebook_ids=req.notebook_ids,
            section_index=req.section_index,
            concept_id=req.concept_id,
        )
        if counted:
            _count_message(user)
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
    if req.context_type not in ("notebook", "global", "session", "folder", "lesson", "test_out", "onboarding"):
        raise HTTPException(400, "Invalid context_type")

    if req.context_type in ('lesson', 'test_out') and req.context_id:
        from coast_content_oma.progressive import assert_chat_ready
        assert_chat_ready(user.id,req.context_id,req.section_index,test_out=req.context_type == 'test_out')

    if req.context_type != "onboarding":  # the welcome chat is free
        _check_messages(user)
    counted = _counts_as_message(req)

    # The turn is produced on its own thread so it always finishes and is saved,
    # even if the student's connection drops mid-answer; the response only relays it.
    import queue as queue_mod
    events: "queue_mod.Queue[str | None]" = queue_mod.Queue()

    def produce_turn():
        try:
            for token, meta in tutor.send_message_stream(
                user_id=user.id,
                message=req.message.strip(),
                conversation_id=req.conversation_id,
                context_type=req.context_type,
                context_id=req.context_id,
                notebook_ids=req.notebook_ids,
                section_index=req.section_index,
                concept_id=req.concept_id,
            ):
                if token is not None:
                    events.put(f"data: {json.dumps({'token': token})}\n\n")
                if meta is not None:
                    if counted:
                        _count_message(user)  # a reply that failed doesn't use a message
                    meta["usage"] = _get_user_usage(user.id)
                    events.put(f"data: {json.dumps({'done': True, **meta})}\n\n")
        except Exception as e:
            import traceback
            traceback.print_exc()
            events.put(f"data: {json.dumps({'error': str(e)})}\n\n")
        finally:
            events.put(None)

    threading.Thread(target=ai_usage.carry(produce_turn), name="coast-chat-turn", daemon=True).start()

    def event_stream():
        while (item := events.get()) is not None:
            yield item

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/chat/history")
def chat_history(conversation_id: str, user: User = Depends(get_current_user)):
    """Get all messages for a specific conversation."""
    messages = tutor.get_chat_history(conversation_id, user.id)
    return messages


@app.get("/api/chat/conversations")
def chat_conversations(
    context_type: Optional[str] = None,
    user: User = Depends(get_current_user),
):
    """List a user's Pedro conversations, optionally filtered by context_type."""
    if context_type and context_type not in ("notebook", "global", "session", "folder", "lesson", "test_out", "onboarding"):
        raise HTTPException(400, "Invalid context_type")
    return tutor.get_conversations(user.id, context_type=context_type)


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
    if not user:  # runs paid AI models: never for anonymous callers
        raise HTTPException(401, "Not authenticated")

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

    import upload_lifecycle
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp_path = Path(tmp.name)
        size = 0
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > upload_lifecycle.MAX_UPLOAD_BYTES:
                tmp.close()
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(413, upload_lifecycle.TOO_LARGE)
            tmp.write(chunk)
    try:
        _security.check_upload(tmp_path, ext)
    except HTTPException:
        tmp_path.unlink(missing_ok=True)
        raise

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
            from curated_config import get_course_for_folder

            course = get_course_for_folder(folder) if folder else None
            paper_paths = _get_paper_paths(course)
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

                        threading.Thread(
                            target=ai_usage.carry(_bg_post_process),
                            args=(user.id, target_folder, nb_id, result),
                            daemon=True,
                        ).start()
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
        threading.Thread(target=ai_usage.carry(run_pipeline_thread), daemon=True).start()

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
    if not is_admin(user):
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
            streak = study_streak(study_days(db, u.id), datetime.now(timezone.utc).date())

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

            # Folders / lessons / sections
            from database import CourseOutline
            folder_count = db.query(StudyFolder).filter(StudyFolder.user_id == u.id).count()
            user_outlines = db.query(CourseOutline).filter(CourseOutline.user_id == u.id).all()
            lessons_started = len(user_outlines)
            sections_done = sum(o.current_section for o in user_outlines)

            last_msg = (
                db.query(ChatMessage)
                .filter(ChatMessage.user_id == u.id)
                .order_by(ChatMessage.created_at.desc())
                .first()
            )
            last_active = (
                last_msg.created_at.isoformat()
                if last_msg and last_msg.created_at
                else (u.created_at.isoformat() if u.created_at else None)
            )

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
                "folders_created": folder_count,
                "lessons_started": lessons_started,
                "sections_completed": sections_done,
                "last_active": last_active,
            })

        return {"users": result, "total_users": len(result)}
    finally:
        db.close()


@app.get("/api/admin/export-cohort")
def export_cohort(user: User = Depends(get_current_user)):
    """Full data export for the current cohort — quiz answers, chat logs, skill profiles."""
    if not is_admin(user):
        raise HTTPException(403, "Admin access only")

    db = SessionLocal()
    try:
        users = db.query(User).all()
        export = []

        for u in users:
            if u.email in ADMIN_EMAILS:
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
# ACTIVITY TRACKING
# ═══════════════════════════════════════════════════════════════════════════


class ActivityRequest(BaseModel):
    feature: str
    duration_ms: int = 0
    action: str = "session"


@app.post("/api/activity")
def log_activity(body: ActivityRequest, user: User = Depends(get_current_user)):
    """Log a feature usage event (time spent on a feature session)."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    db = SessionLocal()
    try:
        ev = ActivityEvent(
            user_id=user.id,
            feature=body.feature,
            action=body.action,
            duration_ms=max(body.duration_ms, 0),
            event_date=today,
        )
        db.add(ev)
        db.commit()
        return {"ok": True}
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════


@app.get("/api/admin/analytics")
def admin_analytics(user: User = Depends(get_current_user)):
    """Platform-wide aggregate stats + growth data for the pitch deck."""
    if not is_admin(user):
        raise HTTPException(403, "Admin access only")

    from sqlalchemy import func
    from datetime import timedelta
    from database import CourseOutline

    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        thirty_days_ago = now - timedelta(days=30)
        cutoff_str = thirty_days_ago.strftime("%Y-%m-%d")

        total_users = db.query(User).count()
        total_messages = db.query(ChatMessage).count()
        total_notebooks = db.query(SavedNotebook).filter(SavedNotebook.is_premade == False).count()
        total_folders = db.query(StudyFolder).count()
        total_lessons_started = db.query(CourseOutline).count()
        total_sections_completed = (
            db.query(func.coalesce(func.sum(CourseOutline.current_section), 0)).scalar() or 0
        )

        avg_messages_per_user = round(total_messages / max(total_users, 1), 1)
        avg_notebooks_per_user = round(total_notebooks / max(total_users, 1), 1)

        users_with_lesson = db.query(CourseOutline.user_id).distinct().count()
        pct_started_lesson = round(users_with_lesson / max(total_users, 1) * 100, 1)

        users_completed = (
            db.query(CourseOutline.user_id)
            .filter(CourseOutline.current_section >= CourseOutline.total_sections)
            .filter(CourseOutline.total_sections > 0)
            .distinct()
            .count()
        )
        pct_completed_lesson = round(users_completed / max(total_users, 1) * 100, 1)

        lessons_with_sections = (
            db.query(CourseOutline).filter(CourseOutline.total_sections > 0).all()
        )
        if lessons_with_sections:
            avg_sections = round(
                sum(o.current_section for o in lessons_with_sections) / len(lessons_with_sections), 1
            )
        else:
            avg_sections = 0.0

        day_col = func.substr(User.created_at, 1, 10)
        signups_raw = (
            db.query(day_col.label("day"), func.count(User.id).label("cnt"))
            .filter(func.substr(User.created_at, 1, 10) >= cutoff_str)
            .group_by(day_col)
            .order_by(day_col)
            .all()
        )
        signups_per_day = [{"date": r.day, "count": r.cnt} for r in signups_raw]

        msg_day_col = func.substr(ChatMessage.created_at, 1, 10)
        messages_raw = (
            db.query(msg_day_col.label("day"), func.count(ChatMessage.id).label("cnt"))
            .filter(func.substr(ChatMessage.created_at, 1, 10) >= cutoff_str)
            .group_by(msg_day_col)
            .order_by(msg_day_col)
            .all()
        )
        messages_per_day = [{"date": r.day, "count": r.cnt} for r in messages_raw]

        total_feedback = db.query(UserFeedback).count()

        # ── Time on platform ──
        total_duration_ms = (
            db.query(func.coalesce(func.sum(ActivityEvent.duration_ms), 0)).scalar() or 0
        )
        total_hours = round(total_duration_ms / 3_600_000, 1)
        avg_hours_per_user = round(total_hours / max(total_users, 1), 2)

        feature_time_raw = (
            db.query(ActivityEvent.feature, func.sum(ActivityEvent.duration_ms).label("ms"))
            .group_by(ActivityEvent.feature)
            .all()
        )
        time_per_feature = {r.feature: round((r.ms or 0) / 3_600_000, 2) for r in feature_time_raw}

        # ── Build unified user-activity-date map from all sources ──
        today_str = now.strftime("%Y-%m-%d")
        seven_days_ago_str = (now - timedelta(days=7)).strftime("%Y-%m-%d")

        user_activity_dates: dict[int, set[str]] = {}

        for uid, ed in db.query(ActivityEvent.user_id, ActivityEvent.event_date).distinct().all():
            user_activity_dates.setdefault(uid, set()).add(ed)

        chat_day = func.substr(ChatMessage.created_at, 1, 10)
        for uid, ed in db.query(ChatMessage.user_id, chat_day.label("d")).distinct().all():
            if ed:
                user_activity_dates.setdefault(uid, set()).add(ed)

        # ── DAU / WAU / MAU ──
        dau_set = {uid for uid, dates in user_activity_dates.items() if today_str in dates}
        wau_set = {uid for uid, dates in user_activity_dates.items() if any(d >= seven_days_ago_str for d in dates)}
        mau_set = {uid for uid, dates in user_activity_dates.items() if any(d >= cutoff_str for d in dates)}
        dau = len(dau_set)
        wau = len(wau_set)
        mau = len(mau_set)

        all_active_dates: dict[str, set[int]] = {}
        for uid, dates in user_activity_dates.items():
            for d in dates:
                if d >= cutoff_str:
                    all_active_dates.setdefault(d, set()).add(uid)
        dau_trend = sorted(
            [{"date": d, "count": len(uids)} for d, uids in all_active_dates.items()],
            key=lambda x: x["date"],
        )

        # ── Retention ──
        all_users = db.query(User.id, func.substr(User.created_at, 1, 10).label("signup")).all()

        retention = {"day_1": 0.0, "day_7": 0.0, "day_30": 0.0}
        if all_users:
            for offset_days, key in [(1, "day_1"), (7, "day_7"), (30, "day_30")]:
                eligible = 0
                returned = 0
                for uid, signup in all_users:
                    if not signup:
                        continue
                    try:
                        signup_dt = datetime.strptime(signup, "%Y-%m-%d")
                    except (ValueError, TypeError):
                        continue
                    target = (signup_dt + timedelta(days=offset_days)).strftime("%Y-%m-%d")
                    if target > today_str:
                        continue
                    eligible += 1
                    if uid in user_activity_dates and target in user_activity_dates[uid]:
                        returned += 1
                retention[key] = round(returned / max(eligible, 1) * 100, 1)

        return {
            "headline": {
                "total_users": total_users,
                "total_messages": total_messages,
                "total_notebooks": total_notebooks,
                "total_folders": total_folders,
                "total_lessons_started": total_lessons_started,
                "total_sections_completed": int(total_sections_completed),
                "total_feedback": total_feedback,
            },
            "engagement": {
                "avg_messages_per_user": avg_messages_per_user,
                "avg_notebooks_per_user": avg_notebooks_per_user,
                "pct_started_lesson": pct_started_lesson,
                "pct_completed_lesson": pct_completed_lesson,
                "avg_sections_per_lesson": avg_sections,
            },
            "growth": {
                "signups_per_day": signups_per_day,
                "messages_per_day": messages_per_day,
            },
            "time": {
                "total_hours": total_hours,
                "avg_hours_per_user": avg_hours_per_user,
                "per_feature": time_per_feature,
            },
            "active_users": {
                "dau": dau,
                "wau": wau,
                "mau": mau,
                "dau_trend": dau_trend,
            },
            "retention": retention,
        }
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# FEEDBACK
# ═══════════════════════════════════════════════════════════════════════════


class FeedbackRequest(BaseModel):
    category: str = "other"
    message: str
    page: str = ""


@app.post("/api/feedback")
def submit_feedback(body: FeedbackRequest, user: User = Depends(get_current_user)):
    """Any authenticated user can submit feedback."""
    if not body.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    db = SessionLocal()
    try:
        fb = UserFeedback(
            user_id=user.id,
            category=body.category,
            message=body.message.strip(),
            page=body.page,
        )
        db.add(fb)
        db.commit()
        return {"ok": True, "id": fb.id}
    finally:
        db.close()


@app.get("/api/admin/feedback")
def admin_feedback(user: User = Depends(get_current_user)):
    """Admin-only: return all user feedback with user info."""
    if not is_admin(user):
        raise HTTPException(403, "Admin access only")

    db = SessionLocal()
    try:
        rows = (
            db.query(UserFeedback, User)
            .join(User, UserFeedback.user_id == User.id)
            .order_by(UserFeedback.created_at.desc())
            .all()
        )
        return {
            "feedback": [
                {
                    "id": fb.id,
                    "user_name": u.name,
                    "user_email": u.email,
                    "category": fb.category,
                    "message": fb.message,
                    "page": fb.page,
                    "created_at": fb.created_at.isoformat() if fb.created_at else None,
                }
                for fb, u in rows
            ]
        }
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# LIVE PRESENCE
# ═══════════════════════════════════════════════════════════════════════════


class HeartbeatBody(BaseModel):
    feature: str = ""


@app.post("/api/heartbeat")
def heartbeat(body: HeartbeatBody = HeartbeatBody(), user: User = Depends(get_current_user)):
    """Called every ~30s by each active browser tab."""
    _live_users[user.id] = {
        "name": user.name,
        "email": user.email,
        "feature": body.feature or "",
        "last_seen": datetime.now(timezone.utc),
    }
    return {"ok": True}


@app.get("/api/admin/live-users")
def admin_live_users(user: User = Depends(get_current_user)):
    """Return currently active users (heartbeat within last 60s). Admin only."""
    if not is_admin(user):
        raise HTTPException(403, "Admin access only")
    return _collect_live_users()


def _is_loadtest_email(email: str) -> bool:
    return (email or "").lower().endswith("@loadtest.local")


def _user_account_breakdown(db) -> dict:
    from sqlalchemy import func

    total = db.query(User).count()
    loadtest = db.query(User).filter(func.lower(User.email).like("%@loadtest.local")).count()
    google = db.query(User).filter(User.google_id.isnot(None)).count()
    return {
        "total_rows": total,
        "real_users": total - loadtest,
        "loadtest_bots": loadtest,
        "google_auth": google,
        "email_auth": db.query(User).filter(User.google_id.is_(None)).count(),
    }


def _purge_user_data(db, user_id: int) -> None:
    from database import (
        CourseOutline,
        FolderSource,
        LessonNotes,
        MapTileProvenance,
        SectionRewardClaim,
        SectionVerification,
        SourceImage,
        StudyFolder,
        TreasureChestOpen,
        UserMapState,
    )

    import file_store
    from coast_content_oma.normalized_source import cache_dir as _page_copy_dir
    for src in db.query(FolderSource).filter(FolderSource.user_id == user_id).all():
        if src.file_path:
            try:
                file_store.remove([src.file_path, _page_copy_dir(src.file_path)])
            except OSError:
                pass
    for img in db.query(SourceImage).filter(SourceImage.user_id == user_id).all():
        if img.image_path:
            try:
                Path(img.image_path).unlink(missing_ok=True)
            except OSError:
                pass

    db.query(ActivityEvent).filter(ActivityEvent.user_id == user_id).delete()
    db.query(UserFeedback).filter(UserFeedback.user_id == user_id).delete()
    db.query(LessonNotes).filter(LessonNotes.user_id == user_id).delete()
    db.query(CourseOutline).filter(CourseOutline.user_id == user_id).delete()
    db.query(StudyFolder).filter(StudyFolder.user_id == user_id).delete()
    from database import SourceSearchIndex, SourceChatTurn
    db.query(SourceSearchIndex).filter(SourceSearchIndex.source_id.in_(db.query(FolderSource.source_id).filter_by(user_id=user_id))).delete(synchronize_session=False)
    db.query(SourceChatTurn).filter_by(user_id=user_id).delete()
    db.query(FolderSource).filter(FolderSource.user_id == user_id).delete()
    db.query(SourceImage).filter(SourceImage.user_id == user_id).delete()
    db.query(UserMapState).filter(UserMapState.user_id == user_id).delete()
    db.query(SectionVerification).filter(SectionVerification.user_id == user_id).delete()
    db.query(SectionRewardClaim).filter(SectionRewardClaim.user_id == user_id).delete()
    db.query(MapTileProvenance).filter(MapTileProvenance.user_id == user_id).delete()
    db.query(TreasureChestOpen).filter(TreasureChestOpen.user_id == user_id).delete()

    user = db.query(User).filter(User.id == user_id).first()
    if user:
        db.delete(user)


def _collect_live_users() -> dict:
    now = datetime.now(timezone.utc)
    active = []
    stale_ids = []
    for uid, info in _live_users.items():
        age = (now - info["last_seen"]).total_seconds()
        if age <= HEARTBEAT_TIMEOUT:
            active.append({
                "user_id": uid,
                "name": info["name"],
                "email": info["email"],
                "feature": info.get("feature") or "unknown",
                "seconds_ago": int(age),
            })
        else:
            stale_ids.append(uid)

    for uid in stale_ids:
        del _live_users[uid]

    active.sort(key=lambda u: u["seconds_ago"])
    return {"count": len(active), "users": active}


@app.get("/api/admin/ai-usage")
def admin_ai_usage(days: int = 7, user: User = Depends(get_current_user)):
    """Tokens and estimated cost of every AI call, by feature, model, day and student."""
    if not is_admin(user):
        raise HTTPException(403, "Admin access only")
    return ai_usage.summary(max(1, min(days, 90)))


@app.get("/api/admin/backups")
def admin_backups(user: User = Depends(get_current_user)):
    """The last backup's manifest: when, how much, and whether it reached R2."""
    if not is_admin(user):
        raise HTTPException(403, "Admin access only")
    import backups
    return {"last": backups.last_backup(), "offsite_configured": backups._r2()[0] is not None}


@app.post("/api/admin/backups")
def admin_run_backup(user: User = Depends(get_current_user)):
    """Start a backup now, in the background (it takes as long as copying the databases)."""
    if not is_admin(user):
        raise HTTPException(403, "Admin access only")
    import backups
    threading.Thread(target=backups.run_backup, name="coast-backup-now", daemon=True).start()
    return {"started": True}


@app.get("/api/admin/control-center")
def admin_control_center(user: User = Depends(get_current_user)):
    """Aggregated mission-control payload: live users, KPIs, traffic, storage."""
    if not is_admin(user):
        raise HTTPException(403, "Admin access only")

    analytics = admin_analytics(user)
    live = _collect_live_users()
    server = _admin_server_metrics()

    db = SessionLocal()
    try:
        from sqlalchemy import func
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        messages_today = (
            db.query(ChatMessage)
            .filter(func.substr(ChatMessage.created_at, 1, 10) == today_str)
            .count()
        )
        signups_today = (
            db.query(User)
            .filter(func.substr(User.created_at, 1, 10) == today_str)
            .count()
        )
        recent_users = (
            db.query(User)
            .order_by(User.created_at.desc())
            .limit(8)
            .all()
        )
        import beta_codes
        from database import BetaCode
        invite_codes = {
            row.used_by_user_id: beta_codes.display(row.code)
            for row in db.query(BetaCode).filter(BetaCode.used_by_user_id.in_([u.id for u in recent_users])).all()
        }
        recent_signups = [
            {
                "id": u.id,
                "name": u.name,
                "email": u.email,
                "course": u.course,
                "beta_code": invite_codes.get(u.id),
                "created_at": u.created_at.isoformat() if u.created_at else None,
                "is_loadtest": _is_loadtest_email(u.email),
            }
            for u in recent_users
        ]
        user_breakdown = _user_account_breakdown(db)
    finally:
        db.close()

    import oma_provider
    oma_db = oma_provider.OMA_DB_PATH
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "live": live,
        "user_breakdown": user_breakdown,
        "kpis": {
            "online_now": live["count"],
            "dau": analytics["active_users"]["dau"],
            "wau": analytics["active_users"]["wau"],
            "mau": analytics["active_users"]["mau"],
            "total_users": user_breakdown["real_users"],
            "total_rows": user_breakdown["total_rows"],
            "loadtest_bots": user_breakdown["loadtest_bots"],
            "total_messages": analytics["headline"]["total_messages"],
            "messages_today": messages_today,
            "signups_today": signups_today,
            "total_hours": analytics["time"]["total_hours"],
            "retention": analytics["retention"],
        },
        "headline": analytics["headline"],
        "engagement": analytics["engagement"],
        "growth": analytics["growth"],
        "active_users": analytics["active_users"],
        "time": analytics["time"],
        "recent_signups": recent_signups,
        "server": server,
        "oma": {
            "enabled": oma_provider.is_oma_enabled(),
            "student_oma_enabled": oma_provider.is_student_enabled(),
            "rag_provider": oma_provider.get_rag_provider(),
            "db_path": str(oma_db),
            "db_bytes": _path_size_bytes(oma_db),
            "image_dir_bytes": _path_size_bytes(oma_provider.OMA_IMAGE_DIR),
        },
    }


class BetaCodeCreateRequest(BaseModel):
    count: int = 1
    note: str = ""


@app.get("/api/admin/beta-codes")
def admin_list_beta_codes(user: User = Depends(get_current_user)):
    """Every invite code with its status (unused / used / revoked)."""
    if not is_admin(user):
        raise HTTPException(403, "Admin access only")
    import beta_codes
    db = SessionLocal()
    try:
        codes = [beta_codes.as_dict(row) for row in beta_codes.list_all(db)]
    finally:
        db.close()
    counts = {s: sum(1 for c in codes if c["status"] == s) for s in ("unused", "used", "revoked")}
    return {"required": beta_codes.required(), "counts": counts, "codes": codes}


@app.post("/api/admin/beta-codes")
def admin_create_beta_codes(req: BetaCodeCreateRequest, user: User = Depends(get_current_user)):
    if not is_admin(user):
        raise HTTPException(403, "Admin access only")
    import beta_codes
    count = max(1, min(50, int(req.count or 1)))
    db = SessionLocal()
    try:
        rows = beta_codes.create(db, count=count, note=req.note, created_by=user.email)
        return {"codes": [beta_codes.as_dict(row) for row in rows]}
    finally:
        db.close()


@app.post("/api/admin/beta-codes/{code}/revoke")
def admin_revoke_beta_code(code: str, user: User = Depends(get_current_user)):
    if not is_admin(user):
        raise HTTPException(403, "Admin access only")
    import beta_codes
    db = SessionLocal()
    try:
        row = beta_codes.revoke(db, code)
        return {"code": beta_codes.as_dict(row)}
    except beta_codes.BetaCodeError as exc:
        raise HTTPException(400, str(exc))
    finally:
        db.close()


class PlanGrantRequest(BaseModel):
    email: str
    plan: str  # "founder" or "beta"


@app.get("/api/admin/plans")
def admin_list_plans(user: User = Depends(get_current_user)):
    """Founding students, and the students waiting for the pass to go on sale."""
    if not is_admin(user):
        raise HTTPException(403, "Admin access only")
    with SessionLocal() as db:
        rows = db.query(User).filter((User.plan == "founder") | (User.founder_interest_at != None)).all()
        return {"students": [{"id": u.id, "email": u.email, "name": u.name, "plan": u.plan or "beta",
                              "interested_at": u.founder_interest_at.isoformat() if u.founder_interest_at else None}
                             for u in rows]}


@app.post("/api/admin/plans")
def admin_set_plan(req: PlanGrantRequest, user: User = Depends(get_current_user)):
    """Give a student the Founding Student pass (or take it back), until payments do it."""
    if not is_admin(user):
        raise HTTPException(403, "Admin access only")
    import plans
    if req.plan not in plans.PLANS:
        raise HTTPException(400, "Choose founder or beta.")
    from sqlalchemy import func
    with SessionLocal() as db:
        account = db.query(User).filter(func.lower(User.email) == req.email.strip().lower()).first()
        if not account:
            raise HTTPException(404, "No account uses that email.")
        account.plan = req.plan
        db.commit()
        print(f"[plan] admin {user.id} set user {account.id} to {req.plan}")
        return {"id": account.id, "plan": account.plan}


@app.post("/api/admin/cleanup-loadtest-users")
def admin_cleanup_loadtest_users(user: User = Depends(get_current_user)):
    """Delete synthetic load-test accounts (@loadtest.local) and their data."""
    if not is_admin(user):
        raise HTTPException(403, "Admin access only")

    from sqlalchemy import func
    import rag

    db = SessionLocal()
    try:
        bots = (
            db.query(User)
            .filter(func.lower(User.email).like("%@loadtest.local"))
            .all()
        )
        deleted_users = 0
        chroma_collections = 0
        for bot in bots:
            chroma_collections += rag.delete_user_collections(bot.id)
            _purge_user_data(db, bot.id)
            deleted_users += 1
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    return {
        "ok": True,
        "deleted_users": deleted_users,
        "chroma_collections_removed": chroma_collections,
        "message": f"Removed {deleted_users} load-test accounts.",
    }


# ═══════════════════════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
def health():
    """Liveness for Render's health check and uptime monitors: cheap, and says
    nothing about users or file paths (the admin Control Center has the details)."""
    from sqlalchemy import text
    with SessionLocal() as db:
        db.execute(text("SELECT 1"))
    return {"status": "ok"}


_csp_seen: dict[tuple, float] = {}


@app.post("/api/csp-report")
async def csp_report(request: Request):
    """What the app's Content-Security-Policy would block (it is report-only until real use shows
    nothing legitimate is caught). Each distinct violation is logged at most hourly; bounded, so
    a flood of fake reports costs nothing."""
    import time
    body = await request.body()
    try:
        data = json.loads(body) if 0 < len(body) <= 8192 else {}
    except ValueError:
        data = {}
    if isinstance(data, list):  # the Reporting API's format
        data = {"csp-report": (data[0] or {}).get("body") or {}} if data else {}
    report = data.get("csp-report") or {} if isinstance(data, dict) else {}
    if isinstance(report, dict) and report:
        directive = str(report.get("effective-directive") or report.get("violated-directive") or report.get("effectiveDirective") or "?")[:40]
        blocked = str(report.get("blocked-uri") or report.get("blockedURL") or "?")[:120]
        now = time.time()
        if now - _csp_seen.get((directive, blocked), 0) > 3600 and len(_csp_seen) < 500:
            _csp_seen[(directive, blocked)] = now
            page = str(report.get("document-uri") or report.get("documentURL") or "")[:80]
            source = str(report.get("source-file") or report.get("sourceFile") or "")[:100]
            print(f"[csp] would block {blocked} ({directive}) on {page} from {source}")
    return StarletteResponse(status_code=204)


class ContentProviderRequest(BaseModel):
    provider: str  # "oma" | "flat"


@app.get("/api/dev/content-provider")
def get_content_provider():
    """Local dev: read active RAG vs Content OMA mode."""
    if os.getenv("RENDER"):
        raise HTTPException(404, "Not available in production")
    import oma_provider
    return oma_provider.content_provider_status()


@app.post("/api/dev/content-provider")
def set_content_provider(req: ContentProviderRequest):
    """Local dev: switch Pedro / lessons between RAG and Content OMA."""
    if os.getenv("RENDER"):
        raise HTTPException(404, "Not available in production")
    import oma_provider
    mode = "oma" if req.provider.strip().lower() in ("oma", "content_oma", "content oma") else "flat"
    return oma_provider.set_rag_provider(mode)


def _get_paper_paths(course: str | None = None) -> list[Path]:
    """Get past paper JSON paths, optionally filtered by course.

    If course is None, returns nothing — we only match papers
    when we know the course to avoid cross-course contamination.
    """
    if not course:
        return []
    paper_files = []
    if PAPERS_DIR.exists():
        for f in PAPERS_DIR.glob("*.json"):
            if f.name in ("notebooks.json", "notebookContent.json"):
                continue
            try:
                import json as _json
                with open(f) as _fh:
                    data = _json.load(_fh)
                if data.get("course", "").upper() == course.upper():
                    paper_files.append(f)
            except Exception:
                pass
    return paper_files


# ═══════════════════════════════════════════════════════════════════════════
# OMA DEBUG ENDPOINTS  (only useful for local testing)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/oma/status")
def oma_status(user: User = Depends(get_current_user)):
    """Quick sanity check — is OMA enabled, what db, etc."""
    if not is_admin(user):
        raise HTTPException(403, "Admin only")
    import oma_provider
    db_path = oma_provider.OMA_DB_PATH
    return {
        "rag_provider": oma_provider.get_rag_provider(),
        "oma_enabled": oma_provider.is_oma_enabled(),
        "student_oma_enabled": oma_provider.is_student_enabled(),
        "db_path": str(db_path),
        "db_bytes": _path_size_bytes(db_path),
        "image_dir": str(oma_provider.OMA_IMAGE_DIR),
        "image_dir_bytes": _path_size_bytes(oma_provider.OMA_IMAGE_DIR),
        "ingest_active": dict(oma_provider._ingest_active),
    }


@app.get("/api/oma/images/{item_id}")
def serve_oma_image(item_id: str, user: User = Depends(get_image_user)):
    """Serve a Content OMA extracted image by its store item id."""
    import oma_provider
    if not oma_provider.is_oma_enabled():
        raise HTTPException(404, "OMA not enabled")
    item = oma_provider._content_orchestrator().images.get(item_id)
    from curated_config import CURATED_FOLDER_NAMES
    from coast_content_oma.stores import make_namespace
    public_namespaces = {make_namespace(_curated_uid(folder), folder) for folder in CURATED_FOLDER_NAMES}
    if not item or (not item.namespace.startswith(f"u{user.id}__") and item.namespace not in public_namespaces):
        raise HTTPException(404, "Image not found")
    path = oma_provider.get_oma_image_path(item_id)
    if not path:
        raise HTTPException(404, "Image not found")
    from fastapi.responses import FileResponse
    return FileResponse(
        path=str(path),
        media_type=_image_media_type(path),
        headers={"Cache-Control": "private, no-store", "Referrer-Policy": "no-referrer"},
    )


@app.get("/api/oma/folder/{folder_name}/stats")
def oma_folder_stats(folder_name: str, user: User = Depends(get_current_user)):
    """How many concepts / chunks / images does OMA have for this folder?"""
    import oma_provider
    if not oma_provider.is_oma_enabled():
        return {"error": "OMA not enabled", "rag_provider": oma_provider.get_rag_provider()}
    from coast_content_oma.stores import make_namespace
    src_uid = _curated_uid(folder_name) if _curated_uid(folder_name) is not None else user.id
    ns = make_namespace(src_uid, folder_name)
    orch = oma_provider._content_orchestrator()
    result = {
        "namespace": ns,
        "owner_id": src_uid,
        "rag_provider": oma_provider.get_rag_provider(),
        "oma_pages": oma_provider.oma_content_page_count(src_uid, folder_name),
        "concepts": orch.concept.stats(ns),
        "content": orch.content.stats(ns),
        "images": orch.images.stats(ns),
    }
    if oma_provider.is_student_enabled():
        from coast_content_oma.student.stores import course_namespace
        student_orch = oma_provider._student_orchestrator()
        course_ns = course_namespace(user.id, folder_name)
        result["student_oma"] = {
            "namespace": course_ns,
            "episodes": len(student_orch.episodes.all(course_ns)),
            "mastery_records": len(student_orch.mastery.all(course_ns)),
            "patterns": len(student_orch.patterns.all(course_ns)),
        }
    return result


@app.get("/api/oma/folder/{folder_name}/preview")
def oma_folder_preview(folder_name: str, q: str, user: User = Depends(get_current_user)):
    """Preview what OMA would inject into Pedro's prompt for query q."""
    import oma_provider
    src_uid = _curated_uid(folder_name) if _curated_uid(folder_name) is not None else user.id
    block, concept_ids = oma_provider.get_folder_context(src_uid, folder_name, q)
    return {"concept_ids": concept_ids, "prompt_block": block}


@app.get("/api/oma/folder/{folder_name}/compare")
def oma_folder_compare(folder_name: str, q: str, user: User = Depends(get_current_user)):
    """Side-by-side: Content OMA vs flat RAG for the same query.

    Use this to A/B what Pedro would see under each retrieval method.
    Lesson chat uses OMA when RAG_PROVIDER=oma and OMA returns a block.
    """
    import time
    import oma_provider
    import rag
    from curated_config import curated_source_uid

    src_uid = curated_source_uid(folder_name)
    effective_uid = src_uid if src_uid is not None else user.id

    t0 = time.time()
    oma_block, concept_ids = oma_provider.get_folder_context(
        effective_uid, folder_name, q,
        max_chars=14000, max_content=12, max_images=6,
    )
    oma_ms = round((time.time() - t0) * 1000)

    t0 = time.time()
    rag_block = rag.build_folder_context(effective_uid, folder_name, q, max_chars=14000)
    rag_ms = round((time.time() - t0) * 1000)

    if oma_provider.is_oma_enabled() and oma_block:
        lesson_uses = "content_oma"
    elif rag_block:
        lesson_uses = "flat_rag"
    else:
        lesson_uses = "fallback_raw_text"

    return {
        "rag_provider": oma_provider.get_rag_provider(),
        "lesson_would_use": lesson_uses,
        "query": q,
        "content_oma": {
            "chars": len(oma_block),
            "ms": oma_ms,
            "concept_count": len(concept_ids),
            "concept_ids": concept_ids,
            "prompt_block": oma_block,
        },
        "flat_rag": {
            "chars": len(rag_block),
            "ms": rag_ms,
            "prompt_block": rag_block,
        },
    }


@app.post("/api/oma/folder/{folder_name}/ingest-all")
def oma_folder_reingest(folder_name: str, user: User = Depends(get_current_user)):
    """Re-ingest every PDF source already uploaded to this folder into
    Content OMA. Useful when:
      - You enabled RAG_PROVIDER=oma after sources were uploaded
      - The original upload-time OMA ingest failed (rate limits, etc.)
    Returns immediately; ingestion runs in a background thread."""
    import oma_provider
    if not oma_provider.is_oma_enabled():
        return {"error": "OMA not enabled"}

    owner_id = _curated_uid(folder_name) if _curated_uid(folder_name) is not None else user.id
    sources = oma_provider.load_folder_pdf_sources(owner_id, folder_name)
    if not sources:
        return {"queued": [], "count": 0, "owner_id": owner_id}

    import learning_jobs
    queued = learning_jobs.retry_sources(owner_id,folder_name)
    return {"queued": queued, "count": len(queued), "owner_id": owner_id, "started": bool(queued)}


@app.post("/api/admin/oma/backfill-all")
def admin_oma_backfill_all(user: User = Depends(get_current_user)):
    """Queue Content OMA ingest for every folder that has PDFs but no OMA index."""
    if not is_admin(user):
        raise HTTPException(403, "Admin only")
    import oma_provider
    if not oma_provider.is_oma_enabled():
        return {"error": "OMA not enabled", "queued": 0}
    return oma_provider.backfill_all_missing_folders()


@app.get("/api/oma/student/{folder_name}/profile")
def oma_student_profile(folder_name: str, user: User = Depends(get_current_user)):
    """Inspect the student's current cognitive profile for this folder."""
    import oma_provider
    if not oma_provider.is_student_enabled():
        return {"error": "Student OMA not enabled"}
    orch = oma_provider._student_orchestrator()
    profile = orch.build_profile(user.id, folder_name)
    return {
        "profile": profile,
        "prompt_block": orch.to_prompt_block(profile),
    }


@app.get("/api/oma/student/{folder_name}/analysis")
def oma_student_analysis(folder_name: str, user: User = Depends(get_current_user)):
    """Student Analysis screen — profile + concept mind map for a course folder."""
    import oma_provider
    result = oma_provider.build_student_analysis(user.id, folder_name)
    if result.get("error"):
        raise HTTPException(400, result["error"])
    return result


@app.get("/api/oma/student/summary")
def oma_student_summary(user: User = Depends(get_current_user)):
    """Cross-course learning summary for the dashboard."""
    import oma_provider
    return oma_provider.build_student_global_summary(user.id)


@app.get("/api/oma/student/mindmap")
def oma_student_mindmap(user: User = Depends(get_current_user)):
    """Global knowledge graph — all courses aggregated (Obsidian-style)."""
    import oma_provider
    result = oma_provider.build_student_global_mindmap(user.id)
    if result.get("error") and not result.get("graph", {}).get("nodes"):
        raise HTTPException(400, result["error"])
    return result


@app.on_event("shutdown")
def stop_learning_worker():
    import learning_jobs
    learning_jobs.stop()
    ai_usage.flush()  # queued usage rows are written before the process exits
    try:
        import oma_provider
        oma_provider.flush_student_writes(timeout=20)  # don't drop queued student memory on deploy
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("  Coast API Server  (local OMA build)")
    print("=" * 50)
    print(f"  Papers dir:  {PAPERS_DIR}")
    print(f"  Generated:   {GENERATED_DIR}")
    print(f"  Database:    {Path(__file__).parent / 'coast.db'}")
    try:
        import oma_provider
        print(f"  RAG provider: {oma_provider.get_rag_provider()}")
        print(f"  Student OMA:  {oma_provider.is_student_enabled()}")
        import tutor as _tutor, claude_chat as _claude
        if _tutor.CHAT_PROVIDER == "anthropic":
            print(f"  Pedro:        Claude {_claude.PEDRO_MODEL} (key {'set' if _claude.available() else 'MISSING'}; "
                  f"failover {_tutor.HELPER_PROVIDER})")
        else:
            print(f"  Pedro:        {_tutor.CHAT_PROVIDER}")
        print(f"  Evaluator:    {'Claude ' + _claude.EVAL_MODEL if _claude.available() else 'Gemini/OpenAI'}")
        print(f"  OMA db:       {oma_provider.OMA_DB_PATH}")
    except Exception as _e:
        print(f"  OMA provider not available: {_e}")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
