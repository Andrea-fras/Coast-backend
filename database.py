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
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

DB_PATH = Path(os.environ.get("DATABASE_PATH", str(Path(__file__).parent / "coast.db")))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=engine)


@event.listens_for(engine, "connect")
def _set_sqlite_wal(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)
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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="chat_messages")


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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def _run_migrations():
    """Add columns that may be missing from existing tables."""
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    if "folder_sources" in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns("folder_sources")]
        if "file_path" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE folder_sources ADD COLUMN file_path TEXT"))
    if "users" in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns("users")]
        if "learning_preferences" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE users ADD COLUMN learning_preferences TEXT DEFAULT ''"))
        if "onboarding_completed" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE users ADD COLUMN onboarding_completed BOOLEAN DEFAULT 0"))


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
            if f.name in ("notebooks.json", "notebookContent.json"):
                continue

            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            paper_id = data.get("id", f.stem)
            existing = db.query(Paper).filter(Paper.paper_id == paper_id).first()

            if existing:
                # Update
                existing.title = data.get("title", "")
                existing.description = data.get("description", "")
                existing.questions_json = json.dumps(data.get("questions", []))
                existing.question_count = len(data.get("questions", []))
            else:
                # Insert
                paper = Paper(
                    paper_id=paper_id,
                    title=data.get("title", ""),
                    description=data.get("description", ""),
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
