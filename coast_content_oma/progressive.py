"""A page-addressed planning pass and incremental teaching readiness for uploaded courses."""
import json
import os
import threading
from functools import lru_cache
from pathlib import Path
from .stores.db import connect_db

UNIT_PAGES = 8
_ranks = {}
_lock = threading.Lock()

@lru_cache(maxsize=128)
def _read_manifest(filename, mtime_ns, size):
    return json.loads(Path(filename).read_text())

@lru_cache(maxsize=128)
def _source_fingerprint(path,mtime_ns,size):
    from .normalized_source import fingerprint
    return fingerprint(path)

def manifest(source):
    import file_store
    path = file_store.local(source.file_path or '') if source.file_path else Path('')
    cached = file_store.local(Path(str(path) + '.pages') / 'manifest.json')
    try:
        source_stat = path.stat()
        stat = cached.stat()
        value = _read_manifest(str(cached), stat.st_mtime_ns, stat.st_size)
        if value.get('sha256') != _source_fingerprint(str(path),source_stat.st_mtime_ns,source_stat.st_size):
            return None
        if len(value.get('pages', [])) != source.page_count:
            return None
        return value
    except (OSError, ValueError, TypeError):
        return None

def overview(sources, max_chars=70000):
    """Represent every source/page, without waiting for classification or vision APIs."""
    manifests = [(s, manifest(s)) for s in sources]
    if not manifests or any(m is None for _, m in manifests):
        return None
    units, parts, excerpts = {}, ['SOURCE OVERVIEW: excerpts for planning only. Full pages and diagrams will be indexed for teaching.'], []
    for source, data in manifests:
        rows = data['pages']
        for start in range(0,len(rows),UNIT_PAGES):
            group=rows[start:start+UNIT_PAGES]
            uid=f'{source.source_id}:{group[0]["page_number"]}-{group[-1]["page_number"]}'
            units[uid]={'source_id':source.source_id, 'source_title':source.title,
                'source_filename':source.filename, 'sha256':data['sha256'],
                'pages':[p['page_number'] for p in group], 'text_chars':sum(len(p.get('text','')) for p in group),
                'page_chars':{p['page_number']:len(p.get('text','')) for p in group}}
            parts.append(f'UNIT {uid} | source {json.dumps(source.title)}')
            for row in group:
                text=' '.join(row.get('text','').split()) or "Image/blank page; inspect with neighboring pages."
                parts.append(f'p.{row["page_number"]} ({len(row.get("images",[]))} images): ')
                excerpts.append((len(parts)-1,text))

    # Budget the actual headers, then distribute remaining space fairly. Short
    # slides release their unused allowance to denser pages; no page is omitted.
    remaining = max_chars - len('\n'.join(parts))
    if remaining < len(excerpts):
        raise ValueError('Source/page references exceed the roadmap context budget. Reduce the upload set or source title lengths.')
    allocations = {}
    for position,text in sorted(excerpts,key=lambda item:len(item[1])):
        allowance = min(len(text),1000,remaining // (len(excerpts)-len(allocations)))
        allocations[position] = allowance
        remaining -= allowance
    for position,text in excerpts:
        allowance = allocations[position]
        if len(text) <= allowance:
            excerpt = text
        elif allowance < 5:
            excerpt = text[:allowance]
        else:
            head = (allowance-3)*2//3
            tail = allowance-3-head
            excerpt = text[:head]+' … '+text[-tail:]
        parts[position] += excerpt
    return '\n'.join(parts), units

def bind_sections(sections, units, spread=False, skipped=None):
    """Bind each section to exact source pages.

    References are unit ids ("src_x:1-8") or page ranges inside a source ("src_x:17",
    "src_x:17-36"); a range binds exactly those pages, so a section can start and end
    where the topic does rather than where a planning unit does. Invented references
    are dropped. Every page is taught: a page no section claimed joins the section
    holding its nearest page from the same source, unless the planner listed it in
    `skipped` (course logistics, title and reading-list slides). With spread=True a
    roadmap with no usable references at all gets the units in source order, split
    evenly across its sections, instead of failing. Sections bound to exactly the
    same pages are merged."""
    if not sections or not isinstance(sections,list):
        raise ValueError('The roadmap response did not contain sections.')
    order=list(units)  # source order, then page order
    unit_rank={u:i for i,u in enumerate(order)}
    meta={}   # (source, page) -> (unit id, chars)
    for u in order:
        chars=units[u].get('page_chars') or {}
        for page in units[u]['pages']:
            meta[(units[u]['source_id'],page)]=(u,chars.get(page, chars.get(str(page),
                units[u].get('text_chars',0)//max(1,len(units[u]['pages'])))))
    rank={key:(unit_rank[u],key[1]) for key,(u,_) in meta.items()}

    def pages(ref):
        # "src_x:3" or "src_x:1-4" → (source, first, last)
        source,_,span=ref.rpartition(':')
        first,_,last=span.partition('-')
        try:
            return source,int(first),int(last or first)
        except ValueError:
            return None

    def resolve(ref):
        if ref in units:
            return [(units[ref]['source_id'],page) for page in units[ref]['pages']]
        want=pages(ref.strip()) if isinstance(ref,str) else None
        if not want or want[2]<want[1] or want[2]-want[1]>2000:
            return []
        return [(want[0],page) for page in range(want[1],want[2]+1) if (want[0],page) in meta]

    for section in sections:
        if not isinstance(section,dict) or not section.get('title'):
            raise ValueError('The roadmap response contained an invalid section.')
        selected=section.get('source_units') if isinstance(section.get('source_units'),list) else []
        section['_pages']=list(dict.fromkeys(key for ref in selected if isinstance(ref,str) for key in resolve(ref)))
        section['_chosen']=bool(section['_pages'])  # pages the planner picked, not borrowed or spread
    if not any(section['_pages'] for section in sections):
        if not spread:
            for section in sections:
                section.pop('_pages',None);section.pop('_chosen',None)
            raise ValueError('The roadmap did not provide valid source/page references. Please retry generation.')
        for i,section in enumerate(sections):
            lo=min(i*len(order)//len(sections),len(order)-1)
            section['_pages']=[key for u in order[lo:max(lo+1,(i+1)*len(order)//len(sections))]
                               for key in resolve(u)]
            section['_chosen']=False
    # A section whose references were all invented teaches from its nearest neighbour's pages.
    for i,section in enumerate(sections):
        if section['_pages']:
            continue
        for j in sorted(range(len(sections)),key=lambda j:(abs(j-i),j)):
            if j!=i and sections[j]['_pages']:
                section['_pages']=list(sections[j]['_pages'])
                break
    skip={key for ref in (skipped or []) if isinstance(ref,str) for key in resolve(ref)}
    # Measure distance to the planner's own choices only, so one section can't snowball
    # through a gap page by page; a tie goes to the earlier section.
    anchors=[list(section['_pages']) for section in sections]
    assigned={key for pages_ in anchors for key in pages_}
    for key in sorted(meta,key=rank.get):
        if key in assigned or key in skip:
            continue
        same_source=[(abs(other[1]-key[1]),i) for i,pages_ in enumerate(anchors)
                     for other in pages_ if other[0]==key[0]]
        target=min(same_source)[1] if same_source else len(sections)-1
        sections[target]['_pages'].append(key)
    merged,seen=[],{}
    for section in sections:
        keys=tuple(sorted(set(section['_pages']),key=rank.get))
        chosen=section.pop('_chosen')
        if chosen and keys in seen:  # the planner gave two sections the same pages: teach them once, with both goals
            first=seen[keys]
            for field in ('learning_objectives','key_topics'):
                first[field]=list(dict.fromkeys(list(first.get(field) or [])+list(section.get(field) or [])))
            continue
        if chosen:
            seen.setdefault(keys,section)
        section['_pages']=keys
        merged.append(section)
    for section in merged:
        keys=section.pop('_pages')
        if sum(meta[key][1] for key in keys) > 50000:
            raise ValueError('A section has too much material. Split its pages into smaller sections.')
        refs={}
        for source,page in keys:
            unit=units[meta[(source,page)][0]]
            ref=refs.setdefault(source,{'source_id':source,'source_title':unit['source_title'],
                                        'source_filename':unit.get('source_filename'),'sha256':unit.get('sha256'),
                                        'pages':[],'text_chars':0})
            ref['pages'].append(page)
            ref['text_chars']+=meta[(source,page)][1]
        section['source_refs']=list(refs.values())
        section['source_units']=[f"{r['source_id']}:{span}" for r in section['source_refs'] for span in _spans(r['pages'])]
        section['source_notebooks']=list(dict.fromkeys(r['source_title'] for r in section['source_refs']))
        section['preparation_version']=1
    sections[:]=merged
    return sections


def _spans(pages):
    """[17,18,19,24] -> ['17-19','24']"""
    out,start=[],None
    for i,page in enumerate(pages):
        start=page if start is None else start
        if i+1==len(pages) or pages[i+1]!=page+1:
            out.append(str(start) if start==page else f'{start}-{page}')
            start=None
    return out

def set_priority(namespace, sections, current=0):
    ranks={}
    ordered=list(enumerate(sections))[max(0,current):]+list(enumerate(sections))[:max(0,current)]
    for rank,(_,section) in enumerate(ordered):
        for ref in section.get('source_refs',[]):
            for page in ref['pages']:
                ranks.setdefault(('doc_'+ref['source_id'],page),rank)
    with _lock:
        if len(_ranks)>256: _ranks.pop(next(iter(_ranks)))
        _ranks[namespace]=ranks

def page_rank(namespace, source_id, page):
    with _lock:
        return (_ranks.get(namespace,{}).get((source_id,page),100000),page)

def init_progress(path):
    with connect_db(path) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS source_page_progress (
            namespace TEXT NOT NULL, source_id TEXT NOT NULL, page INTEGER NOT NULL,
            text_ready INTEGER NOT NULL DEFAULT 0, image_ids TEXT NOT NULL,
            PRIMARY KEY(namespace,source_id,page))''')

def register_pages(path,namespace,source_id,pages,saved_images):
    import hashlib
    init_progress(path)
    images={}
    for image in saved_images:
        iid='ima_'+hashlib.sha256(f'{namespace}:{source_id}:{Path(image["file_path"]).name}'.encode()).hexdigest()
        images.setdefault(image['page_number'],[]).append(iid)
    with connect_db(path) as conn:
        conn.executemany('''INSERT INTO source_page_progress VALUES (?,?,?,0,?)
            ON CONFLICT(namespace,source_id,page) DO UPDATE SET image_ids=excluded.image_ids''',
            [(namespace,source_id,p['page_number'],json.dumps(images.get(p['page_number'],[]))) for p in pages])

def mark_text(path,namespace,source_id,pages):
    with connect_db(path) as conn:
        conn.executemany('UPDATE source_page_progress SET text_ready=1 WHERE namespace=? AND source_id=? AND page=?',
            [(namespace,source_id,p['page_number']) for p in pages])

def section_status(orch, namespace, section):
    init_progress(orch.content.db_path)
    required={('doc_'+r['source_id'],p) for r in section.get('source_refs',[]) for p in r['pages']}
    if not required: return {'ready':False,'ready_pages':0,'total_pages':0}
    complete=0
    require_images = os.getenv("OMA_DESCRIBE_IMAGES", "true").lower() not in ("false","0","no") and os.getenv("OMA_SKIP_IMAGES", "false").lower() not in ("true","1","yes")
    with connect_db(orch.content.db_path) as conn:
        for sid,page in required:
            row=conn.execute('SELECT text_ready,image_ids FROM source_page_progress WHERE namespace=? AND source_id=? AND page=?',(namespace,sid,page)).fetchone()
            if not row or not row[0]: continue
            good=True
            for iid in (json.loads(row[1]) if require_images else []):
                img=conn.execute('SELECT content,store_specific FROM image_items WHERE id=? AND namespace=?',(iid,namespace)).fetchone()
                # Only figures still being described block; ones that could not be
                # described are taught around from the page text.
                if not img or json.loads(img[1] or '{}').get('_pending_vision'):
                    good=False;break
            if good: complete+=1
    return {'ready':complete==len(required),'ready_pages':complete,'total_pages':len(required)}

def section_concept_refs(orch,namespace,section):
    """Resolve the section's explicit page links locally, without semantic API calls."""
    from .llm import normalize_concept_name
    required={('doc_'+ref['source_id'],page) for ref in section.get('source_refs',[]) for page in ref['pages']}
    doc_ids=sorted({sid for sid,_ in required})
    terms={normalize_concept_name(str(topic)) for topic in section.get('key_topics',[]) if topic}
    ids=set()
    with connect_db(orch.content.db_path) as conn:
        if doc_ids:
            placeholders=','.join('?' for _ in doc_ids)
            for table in ('content_items','image_items'):
                for sid,entities,raw in conn.execute(f'SELECT source_doc_id,entities,store_specific FROM {table} WHERE namespace=? AND source_doc_id IN ({placeholders})',(namespace,*doc_ids)):
                    metadata=json.loads(raw or '{}')
                    if (sid,metadata.get('page_number')) not in required:
                        continue
                    names=json.loads(entities or '[]') + (metadata.get('concept_mentions_raw') or [])
                    for name in names:
                        if isinstance(name,str):
                            ids.add(name);terms.add(normalize_concept_name(name))
        concepts={row[0]:(row[1],json.loads(row[2] or '{}')) for row in conn.execute(
            'SELECT id,content,store_specific FROM concept_items WHERE namespace=?',(namespace,))}
    matched={cid for cid,(_,meta) in concepts.items() if cid in ids or any(
        normalize_concept_name(name) in terms for name in [meta.get('name',''),*(meta.get('aliases') or [])] if isinstance(name,str) and name)}
    prerequisites={pid for cid in matched for pid in concepts[cid][1].get('prerequisite_concept_ids',[]) if pid in concepts}
    return [{'concept_id':cid,'concept_name':concepts[cid][1].get('name') or concepts[cid][0][:80] or cid}
            for cid in sorted(matched|prerequisites)]


def section_context(orch,namespace,section,max_chars=60000):
    """Exact source pages avoid broad semantic retrieval for each section opener."""
    required={('doc_'+r['source_id'],p) for r in section['source_refs'] for p in r['pages']}
    titles={'doc_'+r['source_id']:r['source_title'] for r in section['source_refs']}
    contents=[it for it in orch.content.all(namespace) if (it.source_doc_id,(it.store_specific or {}).get('page_number')) in required]
    contents.sort(key=lambda it:(it.source_doc_id,(it.store_specific or {}).get('page_number',0)))
    parts=['--- ASSIGNED SECTION MATERIAL (Content OMA; exact source pages) ---']
    for it in contents:
        parts.append(f'Source: {titles[it.source_doc_id]} | page {it.store_specific["page_number"]}\n{it.content}')
    present={(it.source_doc_id,it.store_specific.get('page_number')) for it in contents}
    uid=int(namespace.split('__')[0][1:])
    from database import SessionLocal,FolderSource
    with SessionLocal() as db:
        for ref in section['source_refs']:
            source=db.query(FolderSource).filter_by(user_id=uid,source_id=ref['source_id']).first()
            data=manifest(source) if source else None
            for row in (data or {}).get('pages',[]):
                if row['page_number'] in ref['pages'] and ('doc_'+ref['source_id'],row['page_number']) not in present and row.get('text'):
                    parts.append(f"Source: {ref['source_title']} | page {row['page_number']}\n{row['text']}")
    images=[it for it in orch.images.all(namespace) if (it.source_doc_id,(it.store_specific or {}).get('page_number')) in required and not it.store_specific.get('_pending_vision')]
    import oma_provider
    from types import SimpleNamespace
    oma_provider._record_retrieved_images([SimpleNamespace(item=it,why='Assigned section page') for it in images])
    for it in images:
        parts.append(f'Diagram: {titles[it.source_doc_id]}, page {it.store_specific["page_number"]}: {it.content}\nImage URL: {oma_provider._oma_image_base_url()}/{it.id}')
    result='\n\n'.join(parts)
    if len(result)>max_chars:
        raise ValueError('This section contains too much source material. Regenerate with smaller sections.')
    return result

def _sources(user_id,folder):
    from database import SessionLocal,FolderSource
    with SessionLocal() as db:
        return db.query(FolderSource).filter_by(user_id=int(user_id),folder_name=folder).order_by(FolderSource.id).all()

_rebuilding=set()


def rebuild_page_copy(path):
    """Re-make a PDF's page copy in the background (once at a time per file), the way an upload
    does: in its own process, then the cached manifest is read again."""
    path=str(path)
    with _lock:
        if path in _rebuilding: return
        _rebuilding.add(path)
    def run():
        import subprocess, sys
        try:
            subprocess.run([sys.executable,'-m','coast_content_oma.read_upload',path],capture_output=True,timeout=600,
                           cwd=str(Path(__file__).resolve().parents[1]))
            _read_manifest.cache_clear()
            import file_store
            file_store.publish_tree(Path(path + '.pages'))
        finally:
            with _lock: _rebuilding.discard(path)
    threading.Thread(target=run,name='coast-page-copy',daemon=True).start()


def status_for_section(user_id,folder,section):
    import oma_provider
    from .stores import make_namespace
    from .ingest_status import TERMINAL_OK, CONTENT_DONE
    sources={s.source_id:s for s in _sources(user_id,folder)}
    for ref in section['source_refs']:
        source=sources.get(ref['source_id'])
        data=manifest(source) if source else None
        total=sum(len(r['pages']) for r in section['source_refs'])
        if source and data is None and source.file_path and Path(source.file_path).is_file():
            # The PDF is here but its page copy is not (a restore, a cleared cache): it is rebuilt
            # from the PDF, which gives the same fingerprint, and the section is being prepared.
            rebuild_page_copy(source.file_path)
            return {'ready':False,'ready_pages':0,'total_pages':total}
        if not data or data['sha256'] != ref['sha256']:
            return {'ready':False,'ready_pages':0,'total_pages':total,
                'error':'A source changed or was removed. Regenerate this roadmap.'}
    orch=oma_provider._content_orchestrator()
    ns=make_namespace(user_id,folder)
    init_progress(orch.content.db_path)
    # Adopt already-indexed sources from before page receipts were introduced.
    for sid in {r['source_id'] for r in section['source_refs']}:
        source=sources[sid]
        if source.oma_ingest_status not in TERMINAL_OK: continue
        data=manifest(source)
        with connect_db(orch.content.db_path) as conn:
            if conn.execute('SELECT 1 FROM source_page_progress WHERE namespace=? AND source_id=? LIMIT 1',(ns,'doc_'+sid)).fetchone(): continue
        images=orch.images.all(ns)
        by_page={}
        for image in images:
            if image.source_doc_id=='doc_'+sid: by_page.setdefault(image.store_specific.get('page_number'),[]).append(image.id)
        with connect_db(orch.content.db_path) as conn:
            conn.executemany('INSERT OR IGNORE INTO source_page_progress VALUES (?,?,?,1,?)',
                [(ns,'doc_'+sid,row['page_number'],json.dumps(by_page.get(row['page_number'],[]))) for row in data['pages']])
    result=section_status(orch,ns,section)
    failed=[s for s in sources.values() if s.source_id in {r['source_id'] for r in section['source_refs']}
            and s.oma_ingest_status=='FAILED' and not _retrying(s.source_id)]
    if failed: result['error']='Some source material could not be prepared. Retry source processing.'
    elif not result['ready'] and all(
            sources[sid].oma_ingest_status in CONTENT_DONE for sid in {r['source_id'] for r in section['source_refs']}):
        # Text is indexed but figures are still pending: make sure a sweep is running
        # (no-op when one already is), so a restart or lost thread can't stall the section.
        oma_provider.kickoff_background_vision_async(user_id, folder)
    return result

def folder_progress(user_id,folder):
    import oma_provider
    from .stores import make_namespace
    if not oma_provider.is_oma_enabled(): return None
    sources=_sources(user_id,folder)
    if not sources or any(manifest(s) is None for s in sources): return None
    orch=oma_provider._content_orchestrator()
    ns=make_namespace(user_id,folder)
    init_progress(orch.content.db_path)
    with connect_db(orch.content.db_path) as conn:
        rows=conn.execute('SELECT source_id,sum(text_ready) FROM source_page_progress WHERE namespace=? GROUP BY source_id',(ns,)).fetchall()
    counts=dict(rows)
    expected=sum(s.page_count for s in sources)
    indexed=sum(min(s.page_count,counts.get('doc_'+s.source_id, s.page_count if s.oma_ingest_status in ('COMPLETE','READY_FOR_ROADMAP','CONTENT_INDEXED') else 0)) for s in sources)
    from database import SessionLocal,CourseOutline
    preparation=None
    with SessionLocal() as db:
        outline=db.query(CourseOutline).filter_by(user_id=int(user_id),folder_name=folder).first()
        if outline:
            sections=json.loads(outline.outline_json)
            idx=min(outline.current_section,len(sections)-1)
            if sections and sections[idx].get('preparation_version')==1:
                set_priority(ns,sections,idx)
                preparation=status_for_section(user_id,folder,sections[idx])
    return {'folder':folder,'oma_enabled':True,'ready_for_roadmap':True,'roadmap_from_overview':True,
        'phase':'section_ready' if preparation and preparation['ready'] else 'roadmap_ready',
        'pages_expected':expected,'pages_indexed':indexed,'section_preparation':preparation,
        'sources':[{'source_id':s.source_id,'filename':s.filename,'page_count':s.page_count,'status':s.oma_ingest_status,
            'ready':s.oma_ingest_status in ('COMPLETE','READY_FOR_ROADMAP')} for s in sources],
        'concepts':orch.concept.count(ns),'images_indexed':orch.images.count(ns),
        'ingest_threads_active':sum(s.oma_ingest_status=='INGESTING' for s in sources)}

def _retrying(source_id):
    """A file whose indexing failed for a passing reason (a busy database, a provider hiccup) is
    queued to try again; until its retries run out it is still being prepared, not failed."""
    import hashlib
    from database import SessionLocal, LearningJob
    with SessionLocal() as db:
        job=db.get(LearningJob,hashlib.sha256(('source:'+source_id).encode()).hexdigest())
        return bool(job and job.status in ('queued','running'))


def assert_chat_ready(user_id,folder,section_index=None,*,test_out=False):
    """Enforce readiness on the server, before starting an SSE response or saving a turn."""
    from database import SessionLocal,CourseOutline
    from fastapi import HTTPException
    with SessionLocal() as db:
        outline=db.query(CourseOutline).filter_by(user_id=user_id,folder_name=folder).first()
        if not outline: return
        sections=json.loads(outline.outline_json)
        idx=outline.current_section if section_index is None else section_index
        if not 0<=idx<=len(sections) or (idx==len(sections) and not test_out): return
        if test_out and idx <= outline.current_section:
            return  # Placement authorization reports invalid targets.
        # A placement checks sections one at a time from the current one, which is also
        # the first prepared; later sections finish preparing while the check runs.
        required = [sections[outline.current_section]] if test_out else [sections[idx]]
        from .stores import make_namespace
        set_priority(make_namespace(user_id,folder),sections,outline.current_section if test_out else idx)
        for section in required:
            if section.get('preparation_version') != 1:
                continue
            status=status_for_section(user_id,folder,section)
            if not status['ready']:
                detail = ('Pedro is still preparing the source pages for the sections you want to test out of. Please try again shortly.'
                          if test_out else 'Pedro is still preparing this section’s source pages and diagrams. Please try again shortly.')
                raise HTTPException(409,status.get('error') or detail)


def source_priority(source_id):
    with _lock:
        return min((rank for ranks in _ranks.values() for (sid,_),rank in ranks.items() if sid=='doc_'+source_id), default=100000)
