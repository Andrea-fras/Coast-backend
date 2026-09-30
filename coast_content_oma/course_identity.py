"""Immutable namespace keys, with a non-destructive legacy registration step."""
from contextlib import closing
import hashlib
import os
import re
import sqlite3
from pathlib import Path


def new_key(folder_name):
    return 'c_' + hashlib.sha256(str(folder_name).encode('utf-8')).hexdigest()


def _app_db_path():
    default = Path('/data/coast.db') if Path('/data').is_dir() else Path(__file__).resolve().parents[1] / 'coast.db'
    return Path(os.environ.get('DATABASE_PATH', str(default)))


def namespace_key(user_id, folder_name):
    # Read only: pure store/fixture callers never create an application DB.
    path = _app_db_path()
    if path.exists():
        try:
            with closing(sqlite3.connect(f'file:{path}?mode=ro', uri=True)) as conn:
                row = conn.execute('SELECT namespace_key FROM course_identities WHERE user_id=? AND folder_name=?',
                    (int(user_id), str(folder_name))).fetchone()
                if row:
                    return row[0]
        except sqlite3.OperationalError:
            pass  # Before the first application migration, new names remain collision-safe.
    return new_key(folder_name)


def display_name(user_id, key):
    path = _app_db_path()
    if path.exists():
        try:
            with closing(sqlite3.connect(f'file:{path}?mode=ro', uri=True)) as conn:
                row = conn.execute('SELECT folder_name FROM course_identities WHERE user_id=? AND namespace_key=?',
                    (int(user_id), key)).fetchone()
                if row:
                    return row[0]
        except sqlite3.OperationalError:
            pass
    return key


def register(db, user_id, folder_name):
    from database import CourseIdentity
    row = db.get(CourseIdentity, (int(user_id), str(folder_name)))
    if row is None:
        row = CourseIdentity(user_id=int(user_id), folder_name=str(folder_name), namespace_key=new_key(folder_name))
        db.add(row)
        db.flush()
    return row.namespace_key


def initialize(oma_path):
    """Register existing namespaces without moving/deleting any learning evidence.

    Ambiguous legacy collisions stop migration instead of assigning one course's
    evidence to another. Resolve those explicitly before deploying this migration.
    """
    from collections import defaultdict
    from database import SessionLocal, CourseIdentity, StudyFolder, CourseOutline, FolderSource, ChatMessage
    existing = set()
    oma_path = Path(oma_path)
    if oma_path.exists():
        with closing(sqlite3.connect(f'file:{oma_path}?mode=ro', uri=True)) as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table in ('concept_items','content_items','image_items','episode_items','concept_mastery_items','pattern_items'):
                if table in tables:
                    existing.update(r[0] for r in conn.execute(f'SELECT DISTINCT namespace FROM {table}'))
    with SessionLocal() as db:
        names = set(db.query(StudyFolder.user_id, StudyFolder.name).all())
        names.update(db.query(CourseOutline.user_id, CourseOutline.folder_name).all())
        names.update(db.query(FolderSource.user_id, FolderSource.folder_name).all())
        names.update(db.query(ChatMessage.user_id, ChatMessage.context_id).filter(ChatMessage.context_type.in_(['lesson','folder','test_out'])).distinct().all())
        groups = defaultdict(list)
        for uid, name in names:
            if not name:
                continue
            slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
            groups[(uid,slug)].append(name)
        for (uid, slug), titles in groups.items():
            legacy_exists = f'u{uid}__{slug}' in existing or f'u{uid}__student__{slug}' in existing
            unregistered = [title for title in titles if not db.get(CourseIdentity,(uid,title))]
            if legacy_exists and len(titles) > 1 and unregistered:
                raise RuntimeError(f'Legacy OMA namespace collision for student {uid}: {len(titles)} course titles require explicit migration')
            for title in unregistered:
                db.add(CourseIdentity(user_id=uid, folder_name=title,
                    namespace_key=slug if legacy_exists else new_key(title)))
        db.commit()


def content_namespace_for_student(user_id, folder):
    from curated_config import curated_source_uid
    from .stores.base import make_namespace
    owner = curated_source_uid(folder)
    return make_namespace(owner if owner is not None else user_id, folder)
