"""Course-scoped passage RAG, independent of OMA concepts and student memory.

Original page text is shared with OMA. SQLite persists resumable embeddings;
keyword retrieval works immediately, including while vectors are being built.
"""
import json
import logging
import math
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import defer
from database import SessionLocal, FolderSource, SourceSearchIndex, SourceUpload
import provider_capacity

log = logging.getLogger(__name__)
MODEL = os.getenv('EMBED_MODEL', 'text-embedding-3-small')
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='coast-source-search')
_pending = set()
_lock = threading.Lock()
STOP = set('a an the and or of to for is are was were be in on at by from with this that these those how what why when where which can do does it its me my you your please explain tell about'.split())


def tokens(value):
    return [w for w in re.findall(r'\w+', value.casefold()) if w not in STOP]


def page_passages(pages):
    out = []
    for page in pages:
        content = (page.get('text') or '').strip()
        start = 0
        while start < len(content):
            end = min(start + 1500, len(content))
            if end < len(content):
                boundary = content.rfind(' ', start + 1000, end)
                if boundary > start:
                    end = boundary
            out.append({'page': int(page['page_number']), 'text': content[start:end]})
            if end == len(content):
                break
            start = end - 180
    return out


def ensure_index(source, with_vectors=True):
    import file_store
    from coast_content_oma.normalized_source import VERSION, cache_dir
    path = Path(source.file_path or '')
    if not source.file_path or not file_store.available(path):
        return None
    # A source's file never changes (a new upload is a new source), so its passages change only
    # with extraction; never with the files' times, which the disk cache refreshes and re-fetches.
    stamp = f'passages-v2:{VERSION}'
    with SessionLocal() as db:
        row = db.get(SourceSearchIndex, source.source_id, options=[] if with_vectors else [defer(SourceSearchIndex.vectors)])
        if row and row.stamp == stamp and row.embedding_model == MODEL:
            db.expunge(row)
            return row
    # Uploads have already extracted these pages; legacy uploads backfill only once.
    manifest = file_store.local(cache_dir(path) / 'manifest.json')
    if manifest.is_file():
        pages = json.loads(manifest.read_text())['pages']
    else:
        from coast_content_oma.extraction import extract_pages
        pages = extract_pages(path, extract_images=False)
    passages = page_passages(pages)
    with SessionLocal() as db:
        db.execute(text('BEGIN IMMEDIATE'))
        if not db.query(FolderSource).filter_by(source_id=source.source_id, user_id=source.user_id).first():
            return None
        row = db.get(SourceSearchIndex, source.source_id)
        if not row:
            row = SourceSearchIndex(source_id=source.source_id)
            db.add(row)
        if row.stamp != stamp or row.embedding_model != MODEL:
            row.stamp, row.passages_json = stamp, json.dumps(passages)
            row.vectors, row.vector_count, row.dimensions = b'', 0, 0
            row.embedding_model, row.retry_at = MODEL, 0
        db.commit()
        db.refresh(row)
        db.expunge(row)
        return row


def embed(texts, priority='background'):
    from openai import OpenAI
    with OpenAI(api_key=os.getenv('OPENAI_API_KEY', ''), timeout=20, max_retries=0) as client:
        result = provider_capacity.call('openai', lambda: client.embeddings.create(model=MODEL, input=texts), priority=priority, lane='embed')
    return np.asarray([r.embedding for r in sorted(result.data, key=lambda r: r.index)], dtype='<f4')


def schedule(source_id):
    if not os.getenv('OPENAI_API_KEY'):
        return  # Keyword index is built on first access without API credentials.
    with _lock:
        if source_id in _pending or len(_pending) >= 32:
            return
        _pending.add(source_id)
    _executor.submit(_build_vectors, source_id)


def _build_vectors(source_id):
    try:
        with SessionLocal() as db:
            source = db.query(FolderSource).filter_by(source_id=source_id).first()
            if not source:
                return
            db.expunge(source)
        row = ensure_index(source)
        if not row or row.retry_at > time.time() or not os.getenv('OPENAI_API_KEY'):
            return
        passages = json.loads(row.passages_json)
        while row.vector_count < len(passages):
            start = row.vector_count
            batch = passages[start:start + 48]
            vectors = embed([p['text'] for p in batch])
            if len(vectors) != len(batch) or vectors.ndim != 2 or not np.isfinite(vectors).all():
                raise ValueError('Invalid embedding batch')
            with SessionLocal() as db:
                # Optimistic compare-and-swap also protects concurrent processes and deletion.
                count = db.query(SourceSearchIndex).filter_by(source_id=source_id, stamp=row.stamp,
                    vector_count=start, embedding_model=MODEL).update({
                        'vectors': row.vectors + vectors.tobytes(), 'vector_count': start + len(batch),
                        'dimensions': vectors.shape[1], 'retry_at': 0})
                db.commit()
                if not count:
                    return
                row = db.get(SourceSearchIndex, source_id)
                db.expunge(row)
    except Exception as exc:
        log.warning('Source search preparation failed source=%s kind=%s', source_id, type(exc).__name__)
        with SessionLocal() as db:
            db.query(SourceSearchIndex).filter_by(source_id=source_id).update({'retry_at': time.time() + 60})
            db.commit()
    finally:
        with _lock:
            _pending.discard(source_id)


def snapshot(user_id, folder, schedule_vectors=True, with_vectors=True):
    with SessionLocal() as db:
        pending_uploads = db.query(SourceUpload).filter_by(user_id=user_id, folder_name=folder).filter(SourceUpload.status.in_(['queued', 'processing', 'failed'])).count()
        sources = db.query(FolderSource).filter_by(user_id=user_id, folder_name=folder).order_by(FolderSource.created_at, FolderSource.id).all()
        for source in sources:
            db.expunge(source)
    rows, info = [], []
    for source in sources:
        try:
            index = ensure_index(source, with_vectors=with_vectors)
        except (OSError, ValueError, KeyError):
            index = None
        passages = json.loads(index.passages_json) if index else []
        ready = bool(passages)
        semantic = ready and index.vector_count == len(passages)
        info.append({'source_id': source.source_id, 'title': source.title, 'ready': ready,
                     'semantic_ready': semantic, 'search_delayed': bool(index and index.retry_at > time.time()),
                     'pages': len({p['page'] for p in passages})})
        if ready:
            rows.append((source, index, passages))
            if schedule_vectors and not semantic and index.retry_at <= time.time():
                schedule(source.source_id)
    return rows, {'sources': info, 'ready_sources': sum(s['ready'] for s in info), 'total_sources': len(info),
                  'pending_uploads': pending_uploads,
                  'search_delayed': not bool(os.getenv('OPENAI_API_KEY')) or any(s['search_delayed'] for s in info),
                  'semantic_ready': bool(info) and all(s['semantic_ready'] for s in info)}


def status(user_id, folder):
    return snapshot(user_id, folder, with_vectors=False)[1]


def retrieve(user_id, folder, query, previous_question='', previous_pages=()):
    rows, coverage = snapshot(user_id, folder)
    passages, vector_blocks = [], []
    for source, index, chunks in rows:
        offset = len(passages)
        for chunk in chunks:
            passages.append({**chunk, 'source_id': source.source_id, 'title': source.title,
                'filename': source.filename, 'source_type': source.source_type, 'page_count': source.page_count})
        if index.vector_count and index.dimensions:
            matrix = np.frombuffer(index.vectors, dtype='<f4').reshape(index.vector_count, index.dimensions)
            vector_blocks.append((offset, matrix))
    if not passages:
        return [], coverage
    # Short follow-ups carry their immediately preceding question and cited pages.
    followup = len(tokens(query)) <= 4 or bool(re.search(r'\b(it|that|those|this|they|them)\b', query, re.I))
    search_query = (previous_question[-700:] + '\n' + query) if followup and previous_question else query
    terms = Counter(tokens(search_query))
    docs = [Counter(tokens(p['title'] + ' ' + p['text'])) for p in passages]
    df = Counter(t for doc in docs for t in doc)
    avglen = sum(sum(doc.values()) for doc in docs) / max(len(docs), 1)
    scores = []
    for doc in docs:
        size = sum(doc.values())
        score = sum(math.log(1 + (len(docs) - df[t] + .5) / (df[t] + .5)) *
            doc[t] * 2.2 / (doc[t] + 1.2 * (.25 + .75 * size / max(avglen, 1))) for t in terms if doc[t])
        scores.append(score)
    ranking = Counter()
    for rank, i in enumerate(sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)):
        if scores[i] > 0:
            ranking[i] += 1 / (40 + rank)
    semantic_used = False
    if vector_blocks and os.getenv('OPENAI_API_KEY'):
        try:
            q = embed([search_query], priority='interactive')[0]
            similarities = []
            for offset, matrix in vector_blocks:
                if matrix.shape[1] != len(q):
                    continue
                sims = matrix @ q / (np.maximum(np.linalg.norm(matrix, axis=1), 1e-8) * max(np.linalg.norm(q), 1e-8))
                similarities.extend((offset + i, float(s)) for i, s in enumerate(sims) if s > .20)
            for rank, (i, _) in enumerate(sorted(similarities, key=lambda item: item[1], reverse=True)):
                ranking[i] += 1 / (40 + rank)
            semantic_used = True
        except Exception as exc:
            log.info('Source search using keywords kind=%s', type(exc).__name__)
    coverage['retrieval'] = 'hybrid' if semantic_used else 'keyword'
    if followup:
        for i, p in enumerate(passages):
            if (p['source_id'], p['page']) in previous_pages:
                ranking[i] += .015
    # Broad summaries need coverage across files, not just the highest-scoring lecture.
    broad = bool(re.search(r'\b(summari[sz]e|overview|main topics|key topics|compare|contrast|all (?:the )?(?:sources|lectures))\b', query, re.I))
    chosen = []
    if broad:
        for source, _, _ in rows:
            candidates = [i for i, p in enumerate(passages) if p['source_id'] == source.source_id]
            chosen.append(max(candidates, key=lambda i: ranking[i]))
    for i in sorted(ranking, key=ranking.get, reverse=True):
        if i not in chosen:
            chosen.append(i)
    # Budget each answer; retrieved evidence never claims to cover the whole corpus.
    selected, chars, per_page = [], 0, Counter()
    for i in chosen:
        p = passages[i]
        key = (p['source_id'], p['page'])
        if per_page[key] >= 2 or chars + len(p['text']) > 18000 or len(selected) >= 12:
            continue
        selected.append({**p, 'id': f'S{len(selected) + 1}'})
        chars += len(p['text'])
        per_page[key] += 1
    coverage['retrieved_sources'] = len({p['source_id'] for p in selected})
    return selected, coverage
