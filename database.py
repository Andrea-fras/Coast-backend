"""SQLite database models for the Coast pilot."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

_PERSISTENT_DISK = Path("/data")
_default_db = str(_PERSISTENT_DISK / "coast.db") if _PERSISTENT_DISK.is_dir() else str(Path(__file__).parent / "coast.db")
DB_PATH = Path(os.environ.get("DATABASE_PATH", _default_db))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=engine)


@event.listens_for(engine, "connect")
def _set_sqlite_wal(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    # Safe with WAL (a crash can lose only the last commits, never corrupt) and
    # avoids an fsync on every commit.
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False, default="")
    google_id = Column(String(255), unique=True, nullable=True, index=True)
    email_verified = Column(Boolean, default=False)
    course = Column(String(100), default="")  # e.g. "QM1", "Data Science"
    learning_preferences = Column(Text, default="")
    onboarding_completed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    sessions = relationship("QuizSession", back_populates="user", cascade="all, delete-orphan")
    notebooks = relationship("SavedNotebook", back_populates="user", cascade="all, delete-orphan")
    chat_messages = relationship("ChatMessage", back_populates="user", cascade="all, delete-orphan")
    tutor_memo = relationship("TutorMemo", back_populates="user", uselist=False, cascade="all, delete-orphan")
    skill_profile = relationship("SkillProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    review_cards = relationship("ReviewCard", back_populates="user", cascade="all, delete-orphan")
    review_history = relationship("ReviewHistory", back_populates="user", cascade="all, delete-orphan")


class QuizSession(Base):
    __tablename__ = "quiz_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    paper_id = Column(String(100), nullable=False)
    paper_title = Column(String(255), default="")
    score = Column(Integer, default=0)
    total = Column(Integer, default=0)
    batch_number = Column(Integer, default=1)  # Which batch of 10 (1, 2, 3...)
    completed = Column(Boolean, default=False)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="sessions")
    answers = relationship("SessionAnswer", back_populates="session", cascade="all, delete-orphan")


class SessionAnswer(Base):
    __tablename__ = "session_answers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("quiz_sessions.id"), nullable=False)
    question_id = Column(String(50), nullable=False)
    question_text = Column(Text, default="")
    user_answer = Column(Text, default="")
    correct_answer = Column(Text, default="")
    is_correct = Column(Boolean, default=False)
    time_spent_ms = Column(Integer, default=0)
    tags_json = Column(Text, default="[]")

    session = relationship("QuizSession", back_populates="answers")


class SavedNotebook(Base):
    __tablename__ = "saved_notebooks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    notebook_id = Column(String(100), nullable=False)
    title = Column(String(255), default="")
    course = Column(String(100), default="")
    notebook_json = Column(Text, nullable=False)  # Full notebook JSON
    is_premade = Column(Boolean, default=False)
    folder = Column(String(100), default="", nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    deleted_at = Column(DateTime, nullable=True, default=None)

    user = relationship("User", back_populates="notebooks")


class Paper(Base):
    """Stores past papers in the database for efficient querying."""
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(String(100), unique=True, nullable=False, index=True)
    title = Column(String(255), default="")
    description = Column(Text, default="")
    course = Column(String(100), default="", index=True)  # e.g. "QM1", "Data Science"
    questions_json = Column(Text, nullable=False)  # Full questions array as JSON
    question_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ChatMessage(Base):
    """Stores every chat message between Pedro and a user."""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    conversation_id = Column(String(100), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # "user" or "pedro"
    content = Column(Text, nullable=False)
    context_type = Column(String(20), nullable=False)  # "notebook", "global", "session"
    context_id = Column(String(100), nullable=True)  # notebook_id or session_id
    section_index = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="chat_messages")


class ConversationDigest(Base):
    """Rolling conversation summary; original messages remain the evidence."""
    __tablename__ = 'conversation_digests'
    user_id = Column(Integer, ForeignKey('users.id'), primary_key=True)
    conversation_id = Column(String(100), primary_key=True)
    through_message_id = Column(Integer, nullable=False)
    summary = Column(Text, nullable=False)


class SourceSearchIndex(Base):
    """Page passages and compact vectors; derived from the shared source extraction."""
    __tablename__ = "source_search_indexes"
    source_id = Column(String(100), ForeignKey("folder_sources.source_id"), primary_key=True)
    stamp = Column(String(100), nullable=False)
    passages_json = Column(Text, nullable=False)
    vectors = Column(LargeBinary, default=b"")
    vector_count = Column(Integer, default=0)
    dimensions = Column(Integer, default=0)
    embedding_model = Column(String(100), default="")
    retry_at = Column(Float, default=0)


class SourceChatTurn(Base):
    """Idempotent source questions; answers/citations are separate from learner evidence."""
    __tablename__ = "source_chat_turns"
    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    request_id = Column(String(100), primary_key=True)
    folder_name = Column(String(100), nullable=False, index=True)
    conversation_id = Column(String(100), nullable=False, index=True)
    question_id = Column(Integer, ForeignKey("chat_messages.id"), nullable=False)
    answer_id = Column(Integer, ForeignKey("chat_messages.id"), nullable=True)
    status = Column(String(20), default="running")
    started_at = Column(Float, nullable=False)
    citations_json = Column(Text, default="[]")
    coverage_json = Column(Text, default="{}")


class TutorMemo(Base):
    """Compact LLM-generated summary of what Pedro knows about a student."""
    __tablename__ = "tutor_memos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)
    memo_text = Column(Text, default="")
    message_count_since_update = Column(Integer, default=0)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="tutor_memo")


class SkillProfile(Base):
    """Per-user topic proficiency scores."""
    __tablename__ = "skill_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)
    profile_json = Column(Text, default="{}")  # {"elasticity": 35, "derivatives": 80}
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="skill_profile")


class ReviewCard(Base):
    """A single reviewable concept tied to a user and notebook section."""
    __tablename__ = "review_cards"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    notebook_id = Column(String(100), nullable=False, index=True)
    section_title = Column(String(255), default="")
    concept = Column(String(255), nullable=False)
    concept_summary = Column(Text, default="")

    interval = Column(Float, default=1.0)
    ease_factor = Column(Float, default=2.5)
    repetitions = Column(Integer, default=0)
    next_review = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_review = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="review_cards")
    history = relationship("ReviewHistory", back_populates="card", cascade="all, delete-orphan")


class ReviewHistory(Base):
    """Log of each spaced-repetition review attempt."""
    __tablename__ = "review_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    card_id = Column(Integer, ForeignKey("review_cards.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    quality = Column(Integer, nullable=False)  # 0-5 SM-2 scale
    reviewed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    card = relationship("ReviewCard", back_populates="history")
    user = relationship("User", back_populates="review_history")


class StudyFolder(Base):
    """Persistent folder for grouping notebooks."""
    __tablename__ = "study_folders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    # "lesson" or "workshop" — decided when the course is created, before any roadmap.
    kind = Column(String(20), default="lesson", nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class FolderSource(Base):
    """Raw uploaded document in a folder (no notebook generation)."""
    __tablename__ = "folder_sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    folder_name = Column(String(100), nullable=False, index=True)
    source_id = Column(String(50), nullable=False, unique=True)
    title = Column(String(200), nullable=False)
    filename = Column(String(200), nullable=False)
    source_type = Column(String(20), nullable=False)
    page_count = Column(Integer, default=0)
    raw_text = Column(Text, nullable=False)
    file_path = Column(String(500), nullable=True)
    # OMA ingest lifecycle: PENDING → INGESTING → READY_FOR_ROADMAP → COMPLETE | FAILED
    oma_ingest_status = Column(String(30), default="PENDING", nullable=False)
    oma_ingest_error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class SourceUpload(Base):
    """A selected file, registered before the browser sends its bytes."""
    __tablename__ = "source_uploads"
    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    folder_name = Column(String(100), primary_key=True)
    upload_id = Column(String(80), primary_key=True)
    filename = Column(String(255), nullable=False)
    size_bytes = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False, default="queued")
    source_id = Column(String(50), nullable=True)
    claim = Column(String(80), nullable=True)
    expires_at = Column(Float, nullable=False)
    error = Column(Text, nullable=True)


class SourceImage(Base):
    """An educationally relevant image extracted from a folder source document."""
    __tablename__ = "source_images"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String(50), nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    folder_name = Column(String(100), nullable=False, index=True)
    page_number = Column(Integer, default=0)
    context_text = Column(Text, default="")
    image_path = Column(String(500), nullable=False)
    width = Column(Integer, default=0)
    height = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class LessonNotes(Base):
    """Personal student notes per lesson (one per user+folder)."""
    __tablename__ = "lesson_notes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    folder_name = Column(String(100), nullable=False, index=True)
    content_html = Column(Text, default="")
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class UserFeedback(Base):
    """User-submitted feedback (bugs, suggestions) for the platform."""
    __tablename__ = "user_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    category = Column(String(50), default="other")
    message = Column(Text, nullable=False)
    page = Column(String(100), default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ActivityEvent(Base):
    """Lightweight event log — one row per feature session (open-to-close)."""
    __tablename__ = "activity_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    feature = Column(String(50), nullable=False)
    action = Column(String(30), default="session")
    duration_ms = Column(Integer, default=0)
    event_date = Column(String(10), nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class AiUsage(Base):
    """One row per AI provider call: who it was for, which feature, and the tokens billed."""
    __tablename__ = "ai_usage"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, nullable=False, index=True)
    user_id = Column(Integer, nullable=True, index=True)  # null for shared/background work
    feature = Column(String(60), nullable=False, default="background")
    provider = Column(String(20), nullable=False)
    model = Column(String(80), nullable=False, default="")
    input_tokens = Column(Integer, default=0)        # all prompt tokens, cached ones included
    cached_tokens = Column(Integer, default=0)       # prompt tokens read from the provider's cache
    cache_write_tokens = Column(Integer, default=0)  # Anthropic cache writes (billed above base input)
    output_tokens = Column(Integer, default=0)       # includes reasoning/thinking tokens
    latency_ms = Column(Integer, default=0)
    ok = Column(Boolean, default=True)


class CourseOutline(Base):
    """Structured lesson outline generated from folder sources."""
    __tablename__ = "course_outlines"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    folder_name = Column(String(100), nullable=False, index=True)
    outline_json = Column(Text, nullable=False)
    total_sections = Column(Integer, default=0)
    current_section = Column(Integer, default=0)
    estimated_minutes = Column(Integer, default=0)
    ever_mastered = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class CourseChatEpoch(Base):
    """Messages at/before this watermark belong to an earlier generated outline."""
    __tablename__ = 'course_chat_epochs'
    user_id = Column(Integer, primary_key=True)
    folder_name = Column(String(255), primary_key=True)
    through_message_id = Column(Integer, nullable=False, default=0)


class CourseIdentity(Base):
    """A course title can change; its OMA key does not."""
    __tablename__ = "course_identities"
    user_id = Column(Integer, primary_key=True)
    folder_name = Column(String(255), primary_key=True)
    namespace_key = Column(String(100), nullable=False, index=True)


class LearningJob(Base):
    """Durable completion outbox; payload is immutable once enqueued."""
    __tablename__ = "learning_jobs"
    id = Column(String(64), primary_key=True)
    payload_json = Column(Text, nullable=False)
    status = Column(String(20), default="queued", nullable=False, index=True)
    attempts = Column(Integer, default=0, nullable=False)
    available_at = Column(Float, default=0, nullable=False)
    lease_until = Column(Float, default=0, nullable=False)
    lease_token = Column(String(64), nullable=True)
    last_error = Column(Text, default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class PlacementTestSession(Base):
    """Server-owned placement result bound to the outline assessed by Pedro."""
    __tablename__ = "placement_test_sessions"
    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    conversation_id = Column(String(100), primary_key=True)
    folder_name = Column(String(100), nullable=False)
    target_section = Column(Integer, nullable=False)
    start_section = Column(Integer, nullable=False)
    outline_digest = Column(String(64), nullable=False)
    passed = Column(Boolean, default=False, nullable=False)
    consumed = Column(Boolean, default=False, nullable=False)
    # Adaptive placement: sections passed so far, evidence since the last pass, finished.
    passed_count = Column(Integer, default=0, nullable=False)
    correct_since = Column(Integer, default=0, nullable=False)
    graded_since = Column(Integer, default=0, nullable=False)
    done = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class UserMapState(Base):
    """Player position on the exploration map."""
    __tablename__ = "user_map_state"

    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    pos_x = Column(Integer, default=32)
    pos_y = Column(Integer, default=32)
    pos_world = Column(String(16), nullable=True)  # which world pos_x/pos_y are in ("lumen", "neon")
    full_unlock = Column(Boolean, default=False)
    total_xp = Column(Integer, default=0)
    bonus_unlock_points = Column(Integer, default=0)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class MapSnapshot(Base):
    """Catalog-normalized, rebuildable map projection; learning records remain authoritative."""
    __tablename__ = 'map_snapshots'
    user_id = Column(Integer, ForeignKey('users.id'), primary_key=True)
    signature = Column(Text, nullable=False)
    payload = Column(Text, nullable=False)


class SectionVerification(Base):
    """Pedro verified the student passed all section practice questions."""
    __tablename__ = "section_verifications"

    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    folder_name = Column(String(100), primary_key=True)
    section_index = Column(Integer, primary_key=True)
    verified_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class SectionRewardClaim(Base):
    """Idempotent XP/map reward when Pedro marks a section complete."""
    __tablename__ = "section_reward_claims"

    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    folder_name = Column(String(100), primary_key=True)
    section_index = Column(Integer, primary_key=True)
    xp_gained = Column(Integer, default=0)
    map_bonus_added = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class MapTileProvenance(Base):
    """Which lesson section unlocked each map tile, per map level (world)."""
    __tablename__ = "map_tile_provenance"

    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    map_level = Column(Integer, primary_key=True, default=1)
    x = Column(Integer, primary_key=True)
    y = Column(Integer, primary_key=True)
    folder_name = Column(String(100), nullable=False)
    section_index = Column(Integer, nullable=False)
    section_title = Column(String(255), default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class TreasureChallenge(Base):
    """The exact question/answer rubric presented to this student, across reloads."""
    __tablename__ = 'treasure_challenges'
    user_id = Column(Integer, ForeignKey('users.id'), primary_key=True)
    chest_id = Column(String(32), primary_key=True)
    challenge_json = Column(Text, nullable=False)


class TreasureChestOpen(Base):
    """One-time treasure chest claim per user."""
    __tablename__ = "treasure_chest_opens"

    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    chest_id = Column(String(32), primary_key=True)
    xp_gained = Column(Integer, default=0)
    correct_count = Column(Integer, default=0)
    opened_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class EmailVerification(Base):
    """Pending email verification codes for signup."""
    __tablename__ = "email_verifications"

    email = Column(String(255), primary_key=True)
    code = Column(String(8), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class BetaCode(Base):
    """Single-use invite codes: creating an account consumes one (see beta_codes.py)."""
    __tablename__ = "beta_codes"

    code = Column(String(32), primary_key=True)  # compact form, e.g. COAST7KQ4M9XP
    note = Column(String(255), default="")  # who it was given to
    created_by = Column(String(255), default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    used_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    used_email = Column(String(255), nullable=True)  # kept even if the account is deleted later
    used_at = Column(DateTime, nullable=True)
    revoked = Column(Boolean, default=False, nullable=False)


def _run_migrations():
    """Add columns that may be missing from existing tables."""
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    if "folder_sources" in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns("folder_sources")]
        if "file_path" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE folder_sources ADD COLUMN file_path TEXT"))
        cols = [c["name"] for c in insp.get_columns("folder_sources")]
        if "oma_ingest_status" not in cols:
            with engine.begin() as conn:
                conn.execute(text(
                    "ALTER TABLE folder_sources ADD COLUMN oma_ingest_status VARCHAR(30) DEFAULT 'PENDING'"
                ))
                conn.execute(text(
                    "UPDATE folder_sources SET oma_ingest_status = 'PENDING' "
                    "WHERE oma_ingest_status IS NULL"
                ))
        cols = [c["name"] for c in insp.get_columns("folder_sources")]
        if "oma_ingest_error" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE folder_sources ADD COLUMN oma_ingest_error TEXT"))
    if "session_answers" in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns("session_answers")]
        if "tags_json" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE session_answers ADD COLUMN tags_json TEXT DEFAULT '[]'"))
    if "study_folders" in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns("study_folders")]
        if "kind" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE study_folders ADD COLUMN kind VARCHAR(20) NOT NULL DEFAULT 'lesson'"))
                # Workshops generated before folders had a kind keep being workshops.
                import json as _json
                from workshops import decorate_sections, is_workshop
                rows = conn.execute(text("SELECT user_id, folder_name, outline_json FROM course_outlines")).fetchall()
                for uid, name, outline_json in rows:
                    try:
                        sections = decorate_sections(name, _json.loads(outline_json or "[]"))
                    except ValueError:
                        continue
                    if is_workshop(sections):
                        conn.execute(text("UPDATE study_folders SET kind='workshop' WHERE user_id=:u AND name=:n"),
                                     {"u": uid, "n": name})
    if "placement_test_sessions" in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns("placement_test_sessions")]
        with engine.begin() as conn:
            for name, ddl in (("passed_count", "INTEGER NOT NULL DEFAULT 0"),
                              ("correct_since", "INTEGER NOT NULL DEFAULT 0"),
                              ("graded_since", "INTEGER NOT NULL DEFAULT 0"),
                              ("done", "BOOLEAN NOT NULL DEFAULT 0")):
                if name not in cols:
                    conn.execute(text(f"ALTER TABLE placement_test_sessions ADD COLUMN {name} {ddl}"))
    if "chat_messages" in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns("chat_messages")]
        if "section_index" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE chat_messages ADD COLUMN section_index INTEGER"))
        # Section-level recall over years of history ("what did we do in lecture 3?").
        with engine.begin() as conn:
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chat_user_course_section "
                              "ON chat_messages(user_id, context_id, section_index, id)"))
    if "users" in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns("users")]
        if "learning_preferences" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE users ADD COLUMN learning_preferences TEXT DEFAULT ''"))
        if "onboarding_completed" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE users ADD COLUMN onboarding_completed BOOLEAN DEFAULT 0"))
        if "google_id" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE users ADD COLUMN google_id VARCHAR(255)"))
        if "email_verified" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE users ADD COLUMN email_verified BOOLEAN DEFAULT 0"))
                conn.execute(text("UPDATE users SET email_verified = 1 WHERE email_verified IS NULL OR email_verified = 0"))
    if "user_map_state" in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns("user_map_state")]
        if "full_unlock" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE user_map_state ADD COLUMN full_unlock BOOLEAN DEFAULT 0"))
        if "total_xp" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE user_map_state ADD COLUMN total_xp INTEGER DEFAULT 0"))
        if "bonus_unlock_points" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE user_map_state ADD COLUMN bonus_unlock_points INTEGER DEFAULT 0"))
        if "pos_world" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE user_map_state ADD COLUMN pos_world VARCHAR(16)"))
    if "map_tile_provenance" in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns("map_tile_provenance")]
        if "map_level" not in cols:
            # A rebuildable projection of reward claims: recreate it with the level in its key.
            # Map snapshots are keyed on a versioned signature, so they rebuild on next load.
            with engine.begin() as conn:
                conn.execute(text("DROP TABLE map_tile_provenance"))
            MapTileProvenance.__table__.create(engine)
    if "course_outlines" in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns("course_outlines")]
        if "ever_mastered" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE course_outlines ADD COLUMN ever_mastered BOOLEAN DEFAULT 0"))
                conn.execute(text(
                    "UPDATE course_outlines SET ever_mastered = 1 "
                    "WHERE total_sections > 0 AND current_section >= total_sections"
                ))


def init_db():
    """Create all tables."""
    Base.metadata.create_all(engine)
    _run_migrations()


def get_db() -> Session:
    """Get a database session."""
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.close()
        raise


def load_papers_from_json(data_dir: str | Path):
    """Load all paper JSON files into the database (idempotent)."""
    data_dir = Path(data_dir)
    db = SessionLocal()

    try:
        for f in data_dir.glob("*.json"):
            if f.name in ("notebooks.json", "notebookContent.json", "curatedLessons.json"):
                continue

            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            if not isinstance(data, dict):
                continue

            paper_id = data.get("id", f.stem)
            existing = db.query(Paper).filter(Paper.paper_id == paper_id).first()

            if existing:
                existing.title = data.get("title", "")
                existing.description = data.get("description", "")
                existing.course = data.get("course", existing.course or "")
                existing.questions_json = json.dumps(data.get("questions", []))
                existing.question_count = len(data.get("questions", []))
            else:
                paper = Paper(
                    paper_id=paper_id,
                    title=data.get("title", ""),
                    description=data.get("description", ""),
                    course=data.get("course", ""),
                    questions_json=json.dumps(data.get("questions", [])),
                    question_count=len(data.get("questions", [])),
                )
                db.add(paper)

        db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    print(f"Database created at: {DB_PATH}")
