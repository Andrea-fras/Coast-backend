"""SQLite completion outbox with recoverable leases and idempotent OMA projection."""
import hashlib
import json
import logging
import threading
import time
import uuid
from sqlalchemy import text, func
from sqlalchemy.dialects.sqlite import insert
from database import SessionLocal, LearningJob, CourseOutline, ChatMessage
import ai_usage

log = logging.getLogger(__name__)
_wake = threading.Event()
_stop = threading.Event()
_thread = None
_source_threads = []
_start_lock = threading.Lock()
LEASE_SECONDS = 120
MAX_ATTEMPTS = 8

def enqueue(db, user_id, folder, section_index, section_title='', completion_type='taught'):
    outline = db.query(CourseOutline).filter_by(user_id=user_id, folder_name=folder).first()
    if not outline:
        raise ValueError('Cannot record completion without an outline')
    from coast_content_oma.course_identity import register
    register(db, user_id, folder)
    sections = json.loads(outline.outline_json)
    if not 0 <= section_index < len(sections):
        raise ValueError('Invalid completion section')
    section = sections[section_index]
    digest = hashlib.sha256(outline.outline_json.encode()).hexdigest()
    job_id = hashlib.sha256(json.dumps([user_id, outline.id, digest, section_index]).encode()).hexdigest()
    through_id = db.query(func.max(ChatMessage.id)).filter_by(user_id=user_id,
        context_type='lesson', context_id=folder, section_index=section_index).scalar() or 0
    from database import CourseChatEpoch
    epoch = db.get(CourseChatEpoch, (int(user_id), folder))
    payload = dict(after_message_id=epoch.through_message_id if epoch else 0,
        user_id=int(user_id), folder=folder, section_index=section_index,
        section_title=section_title or section.get('title', ''), completion_type=completion_type,
        outline_digest=digest, through_message_id=through_id,
        lesson_complete=section_index + 1 >= len(sections),
        section_minutes=max(int(section.get('estimated_minutes') or 25), 25))
    db.execute(insert(LearningJob).values(id=job_id, payload_json=json.dumps(payload),
        status='queued', attempts=0, available_at=0, lease_until=0).on_conflict_do_nothing(index_elements=['id']))
    return job_id

def enqueue_source(db, source):
    payload = {'kind': 'source_ingest', 'user_id': source.user_id, 'folder': source.folder_name,
               'source_id': source.source_id, 'path': source.file_path}
    job_id = hashlib.sha256(('source:' + source.source_id).encode()).hexdigest()
    db.execute(insert(LearningJob).values(id=job_id, payload_json=json.dumps(payload), status='queued',
        attempts=0, available_at=0, lease_until=0).on_conflict_do_nothing(index_elements=['id']))
    return job_id


def retry_sources(user_id, folder):
    """Explicit retry: keep finished work and live leases, requeue incomplete sources."""
    from database import FolderSource
    from coast_content_oma.ingest_status import CONTENT_DONE
    import file_store  # on the disk, or in R2 if the disk cache cleared it
    queued=[]
    with SessionLocal() as db:
        for source in db.query(FolderSource).filter_by(user_id=int(user_id),folder_name=folder):
            if source.oma_ingest_status in CONTENT_DONE or not source.file_path or not file_store.available(source.file_path):
                continue
            job_id=enqueue_source(db,source)
            job=db.get(LearningJob,job_id)
            if job.status == 'running' and job.lease_until > time.time():
                continue
            job.status='queued'
            job.attempts=0
            job.available_at=job.lease_until=0
            job.last_error=''
            job.lease_token=None
            source.oma_ingest_status='PENDING'
            source.oma_ingest_error=None
            queued.append(source.source_id)
        db.commit()
    wake()
    return queued


def retry_failed_completions(user_id=None):
    """Requeue completion jobs that exhausted their retries (e.g. after a bug fix).

    Safe to run repeatedly: projection is idempotent per job id, and section
    rewards are keyed per student/course/section, so replays never duplicate.
    Returns the requeued job ids."""
    requeued = []
    with SessionLocal() as db:
        for job in db.query(LearningJob).filter(LearningJob.status == 'failed'):
            payload = json.loads(job.payload_json)
            if payload.get('kind') or (user_id is not None and int(payload.get('user_id', -1)) != int(user_id)):
                continue
            job.status, job.attempts, job.available_at, job.lease_until = 'queued', 0, 0, 0
            job.lease_token, job.last_error = None, ''
            requeued.append(job.id)
        db.commit()
    wake()
    return requeued


def recover_sources():
    import oma_provider
    if not oma_provider.is_oma_enabled():
        return
    from database import FolderSource
    from coast_content_oma.ingest_status import CONTENT_DONE
    from pathlib import Path
    import file_store  # on the disk, or in R2 if the disk cache cleared it
    folders = set()
    with SessionLocal() as db:
        for source in db.query(FolderSource).all():
            if not source.file_path or not file_store.available(source.file_path):
                continue
            if Path(source.file_path).suffix.lower() not in ('.pdf', '.pptx'):
                continue
            if source.oma_ingest_status in CONTENT_DONE:
                # Finished historical courses need no startup rebuild. Resume
                # interrupted finalization; optional refinement is also lazy
                # when the learner opens a saved roadmap.
                if source.oma_ingest_status == 'CONTENT_INDEXED':
                    folders.add((source.user_id, source.folder_name))
            else:
                job_id = hashlib.sha256(('source:' + source.source_id).encode()).hexdigest()
                if source.oma_ingest_status == 'INGESTING' and not db.get(LearningJob, job_id):
                    source.oma_ingest_status = 'FAILED'
                enqueue_source(db, source)
        db.commit()
    for uid, folder in folders:
        oma_provider.maybe_finalize_folder_concepts(uid, folder)
    wake()


def enqueue_committed(user_id, folder, section_index, section_title=''):
    with SessionLocal() as db:
        job_id = enqueue(db, user_id, folder, section_index, section_title)
        db.commit()
    wake()
    return job_id

def claim(kind=None):
    now = time.time()
    with SessionLocal() as db:
        db.execute(text('BEGIN IMMEDIATE'))
        query = db.query(LearningJob).filter(
            ((LearningJob.status == 'queued') & (LearningJob.available_at <= now)) |
            ((LearningJob.status == 'running') & (LearningJob.lease_until < now))
        )
        job_kind = func.json_extract(LearningJob.payload_json, '$.kind')
        if kind == 'source': query = query.filter(job_kind == 'source_ingest')
        elif kind == 'completion': query = query.filter(job_kind.is_(None))
        query = query.order_by(job_kind.is_not(None), LearningJob.created_at, LearningJob.id)
        if kind == 'source':
            from coast_content_oma.progressive import source_priority
            candidates=query.limit(100).all()
            # Pages a student is about to be taught come first; among equally urgent files the
            # student with the fewest files being indexed goes next, so nobody waits behind
            # someone else's whole upload; then first come, first served.
            busy={}
            for (payload,) in db.query(LearningJob.payload_json).filter(
                    LearningJob.status == 'running', LearningJob.lease_until >= now, job_kind == 'source_ingest'):
                uid=json.loads(payload).get('user_id')
                busy[uid]=busy.get(uid,0)+1
            def turn(item):
                order,job=item
                payload=json.loads(job.payload_json)
                return (source_priority(payload.get('source_id','')), busy.get(payload.get('user_id'),0), order)
            row=min(enumerate(candidates),key=turn,default=(None,None))[1]
        else:
            row=query.first()
        if row is None:
            return None
        row.status = 'running'
        row.attempts += 1
        row.lease_token = uuid.uuid4().hex
        row.lease_until = now + LEASE_SECONDS
        result = (row.id, row.lease_token, json.loads(row.payload_json))
        db.commit()
        return result

def finish(job_id, token, error=None):
    with SessionLocal() as db:
        row = db.query(LearningJob).filter_by(id=job_id, lease_token=token, status='running').first()
        if not row:
            return
        row.status = ('failed' if row.attempts >= MAX_ATTEMPTS else 'queued') if error else 'done'
        row.last_error = str(error)[:1200] if error else ''
        row.available_at = time.time() + min(300, 2 ** row.attempts) if error else 0
        row.lease_until = 0
        db.commit()

def project(job_id, payload):
    ai_usage.tag(feature='source_ingest' if payload.get('kind') == 'source_ingest' else 'section_memory',
                 user_id=payload.get('user_id'))
    if payload.get('kind') == 'source_ingest':
        import oma_provider
        from coast_content_oma.ingest_status import get_status, CONTENT_DONE
        from database import FolderSource
        with SessionLocal() as db:
            source = db.query(FolderSource).filter_by(source_id=payload['source_id'], user_id=payload['user_id']).first()
            if not source:
                return  # Source was intentionally deleted before it was processed.
            job = db.get(LearningJob, job_id)
            if job and job.attempts > 1 and source.oma_ingest_status == 'INGESTING':
                source.oma_ingest_status = 'FAILED'
                db.commit()
        oma_provider.ingest_pdf_into_oma(payload['user_id'], payload['folder'], payload['path'], source_id=payload['source_id'])
        if get_status(payload['source_id']) not in CONTENT_DONE:
            raise RuntimeError('Source ingestion did not reach an indexed state')
        oma_provider.maybe_finalize_folder_concepts(payload['user_id'], payload['folder'])
        return
    # Reward claims have their own unique student/course/section key. This
    # repairs a missing browser reward request, including the final section.
    import map_world
    map_world.claim_section_reward(payload['user_id'], payload['folder'], payload['section_index'],
        section_title=payload['section_title'], lesson_complete=payload['lesson_complete'],
        section_minutes=payload['section_minutes'])
    import oma_provider
    if not oma_provider.is_student_enabled():
        return
    import evaluator
    from coast_content_oma.stores.db import transaction
    from coast_content_oma.student.stores import course_namespace
    orch = oma_provider._student_orchestrator()
    rec = oma_provider._student_recorder_singleton()
    ns = course_namespace(payload['user_id'], payload['folder'])
    # Initialize this schema before entering the projection transaction.
    from coast_content_oma.stores.db import connect_db
    with connect_db(orch.episodes.db_path) as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS applied_learning_events (event_id TEXT PRIMARY KEY, applied_at REAL NOT NULL)')
        if conn.execute('SELECT 1 FROM applied_learning_events WHERE event_id=?', (job_id,)).fetchone():
            return
    evaluation = None
    with SessionLocal() as db:
        current = db.query(CourseOutline).filter_by(user_id=payload['user_id'], folder_name=payload['folder']).first()
        if not current:
            raise ValueError('Course removed before completion was projected')
        current_digest = hashlib.sha256(current.outline_json.encode()).hexdigest()
    if current_digest != payload['outline_digest']:
        # Retain the immutable completion; skip concept verdicts from a different version.
        payload = {**payload, 'completion_type': 'completed_previous_version'}
    if payload['completion_type'] == 'taught' and payload['through_message_id']:
        _, transcript = evaluator.fetch_section_transcript(payload['user_id'], payload['folder'],
            payload['section_index'], through_message_id=payload['through_message_id'], after_message_id=payload.get('after_message_id',0))
        import lesson
        from curated_config import curated_source_uid
        concepts = lesson.get_section_concept_refs(payload['user_id'], payload['folder'], payload['section_index'], source_user_id=curated_source_uid(payload['folder']))
        if transcript:
            from workshops import decorate_sections
            with SessionLocal() as db:
                outline_row = db.query(CourseOutline).filter_by(user_id=payload['user_id'], folder_name=payload['folder']).first()
                sections = decorate_sections(payload['folder'], json.loads(outline_row.outline_json)) if outline_row else []
            contract = (sections[payload['section_index']].get('workshop')
                        if payload['section_index'] < len(sections) else None)
            evaluation = evaluator.evaluate_section_transcript(transcript, payload['section_index'], payload['section_title'],
                                                               concepts, require_model=True, workshop=contract)
            evaluation['evidence'] = {'through_message_id': payload['through_message_id'],
                'outline_digest': payload['outline_digest'], 'after_message_id': payload.get('after_message_id',0), 'completion_type': payload['completion_type']}
    # No LLM/embedding calls inside this transaction. A crash rolls back the
    # completion, verdicts, analogy and dedupe marker together.
    with transaction(orch.episodes.db_path) as conn:
        if conn.execute('SELECT 1 FROM applied_learning_events WHERE event_id=?', (job_id,)).fetchone():
            return
        rec.record_episode(payload['user_id'], payload['folder'], 'section_completed',
            summary=('Tested out of: ' if payload['completion_type'] == 'tested_out' else 'Completed section: ') + payload['section_title'],
            outcome='neutral', concept_refs=[], source='lesson_player',
            section_index=payload['section_index'], section_title=payload['section_title'],
            signals={'completion_type': payload['completion_type'], 'event_id': job_id,
                     'outline_digest': payload['outline_digest'], 'through_message_id': payload['through_message_id']})
        if evaluation:
            evaluator.apply_evaluation(payload['user_id'], payload['folder'], payload['section_index'], payload['section_title'], evaluation)
        conn.execute('INSERT INTO applied_learning_events VALUES (?,?)', (job_id, time.time()))
    oma_provider._run_course_consolidation(payload['user_id'], payload['folder'])

def run_one(handler=project, *, kind=None):
    job = claim(kind)
    if job is None:
        return False
    job_id, token, payload = job
    heartbeat_stop = threading.Event()
    def renew():
        while not heartbeat_stop.wait(LEASE_SECONDS / 3):
            try:
                with SessionLocal() as db:
                    db.query(LearningJob).filter_by(id=job_id, lease_token=token, status='running').update({'lease_until': time.time() + LEASE_SECONDS})
                    db.commit()
            except Exception:
                log.exception('Could not renew completion lease %s', job_id)
    heartbeat = threading.Thread(target=renew, daemon=True)
    heartbeat.start()
    try:
        handler(job_id, payload)
        finish(job_id, token)
    except Exception as exc:
        log.exception('Completion job failed %s', job_id)
        finish(job_id, token, exc)
    finally:
        heartbeat_stop.set()
        heartbeat.join(timeout=1)
    return True

def _run(kind=None):
    import memory_budget
    while not _stop.is_set():
        try:
            if kind == 'source' and not memory_budget.has_room(memory_budget.index_file_mb()):
                _stop.wait(2)  # another file finishes first; the queue keeps its order
                continue
            if run_one(kind=kind):
                continue
        except Exception:
            log.exception('Completion worker iteration failed')
        _wake.wait(2)
        _wake.clear()

def start():
    global _thread
    with _start_lock:
        if _thread and _thread.is_alive():
            return
        _stop.clear()
        _thread = threading.Thread(target=_run, args=('completion',), name='coast-completion-worker', daemon=True)
        _thread.start()
        import os
        _source_threads[:] = [thread for thread in _source_threads if thread.is_alive()]
        from coast_content_oma import remote
        most = 16 if remote.enabled() else 4  # in containers the server only stores the results
        count = max(1,min(most,int(os.getenv('COAST_SOURCE_WORKERS','2'))))
        for i in range(len(_source_threads),count):
            thread=threading.Thread(target=_run,args=('source',),name=f'coast-source-worker-{i}',daemon=True)
            _source_threads.append(thread)
            thread.start()

def wake():
    _wake.set()

def stop():
    _stop.set()
    _wake.set()
