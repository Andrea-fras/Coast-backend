#!/usr/bin/env python3
"""Refresh extraction for explicitly selected courses, preserving outlines and student evidence.

Reuses descriptions for unchanged images and for identical image/context pairs within
this invocation. Requires --apply to write; run with a database backup before applying.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pixels(image):
    rgb = image.convert('RGB')
    return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def refresh(source, orch, pipeline, apply, cache):
    from PIL import Image
    from database import SessionLocal, FolderSource
    from coast_content_oma.extraction import extract_pages
    from coast_content_oma.normalized_source import load_pages, save_pages, VERSION
    from coast_content_oma.stores import make_namespace
    from coast_content_oma.stores.base import MemoryItem
    from coast_content_oma.stores.db import connect_db
    from coast_content_oma import progressive

    started = time.monotonic()
    path = Path(source.file_path)
    pages = load_pages(path) or extract_pages(path, use_cache=False)
    ns = make_namespace(source.user_id, source.folder_name)
    doc_id = 'doc_' + source.source_id
    old_images = {Path(i.store_specific['file_path']).name: i for i in orch.images.all(ns)
                  if i.source_doc_id == doc_id}
    old_content = {i.store_specific['page_number']: i for i in orch.content.all(ns) if i.source_doc_id == doc_id}
    contents = []
    for page in pages:
        item = old_content.get(page['page_number'])
        if not item and page['text'].strip():
            item = MemoryItem(id='con_'+hashlib.sha256(f"{ns}:{doc_id}:page:{page['page_number']}".encode()).hexdigest(),
                namespace=ns, store='content', content='', source_doc_id=doc_id,
                store_specific={'page_number': page['page_number'], 'source_filename': source.filename,
                                'content_types': ['unclassified'], 'concept_mentions_raw': []})
            old_content[page['page_number']] = item
        if item and item.content != page['text']:
            item.content = page['text']
            # Old prose summaries may contain flattened mathematical notation.
            item.store_specific.update(summary='', extraction_version=VERSION)
            contents.append(item)
    report = {'source_id': source.source_id, 'folder': source.folder_name,
              'title': source.title, 'pages': len(pages), 'text_pages_changed': len(contents)}

    with tempfile.TemporaryDirectory(prefix='coast-source-quality-') as tmp:
        stage = Path(tmp)
        candidates, metadata, changed = [], [], []
        for page in pages:
            for img in page['images']:
                name = f"p{page['page_number']}_i{img['idx']}.png"
                old = old_images.get(name)
                digest = pixels(img['pil_image'])
                old_digest = (old.store_specific or {}).get('pixel_sha256') if old else None
                if old and not old_digest:
                    try:
                        with Image.open(old.store_specific['file_path']) as original:
                            old_digest = pixels(original)
                    except OSError:
                        pass
                meta = {'page_number': page['page_number'], 'file_path': str(stage/name),
                        'width': img['width'], 'height': img['height'],
                        'context_hint': page['text'][:500], 'pixel_sha256': digest,
                        'extraction_version': VERSION, 'extraction_kind': img.get('extraction_kind', 'raster')}
                metadata.append(meta)
                needs_description = (not old or old_digest != digest
                    or old.store_specific.get('_pending_vision')
                    or (meta['extraction_kind'] == 'composited' and old.store_specific.get('extraction_version', 0) < VERSION))
                if needs_description:
                    img['pil_image'].save(meta['file_path'], 'PNG')
                    changed.append((meta, old))
                    key = (digest, meta['context_hint'])
                    if key not in cache:
                        candidates.append(meta)

        report.update(images_to_describe=len(candidates), images_changed=len(changed))
        print('SOURCE_PLAN', json.dumps(report), flush=True)
        if not apply:
            return report
        described = pipeline._describe_saved_images_batched(candidates)
        if len(described) != len(candidates) or any(not d.get('description') or d.get('_pending_vision') for d in described):
            raise RuntimeError(f'Incomplete image descriptions for {source.source_id}; source was not updated.')
        for meta, result in zip(candidates, described):
            cache[(meta['pixel_sha256'], meta['context_hint'])] = result

        updated_images = []
        destination = pipeline.image_save_dir / doc_id
        destination.mkdir(parents=True, exist_ok=True)
        for meta, old in changed:
            result = cache[(meta['pixel_sha256'], meta['context_hint'])]
            name = Path(meta['file_path']).name
            final_path = destination / name
            temporary = destination / (name + '.quality.tmp')
            shutil.copyfile(meta['file_path'], temporary)
            os.replace(temporary, final_path)
            item = old or MemoryItem(id='ima_'+hashlib.sha256(f'{ns}:{doc_id}:{name}'.encode()).hexdigest(),
                namespace=ns, store='image', content='', source_doc_id=doc_id)
            item.content = result['description']
            item.tags = [result.get('image_type') or 'figure']
            item.store_specific.update({k:v for k,v in meta.items() if k != 'context_hint'})
            item.store_specific.update(file_path=str(final_path), source_filename=source.filename,
                concept_mentions_raw=result.get('concepts') or [], image_type=item.tags[0])
            item.store_specific.pop('_pending_vision', None)
            updated_images.append(item)

        orch.images.write_items_bulk(updated_images)
        orch.content.write_items_bulk(contents)
        names = {Path(m['file_path']).name for m in metadata}
        with connect_db(orch.images.db_path) as conn:
            for name, old in old_images.items():
                if name not in names:
                    # Hide discarded blank assets from retrieval, retain historical URLs.
                    conn.execute('UPDATE image_items SET superseded_by=? WHERE id=?', ('extraction-v'+str(VERSION), old.id))
            for meta in metadata:
                name = Path(meta['file_path']).name
                if name in old_images:
                    conn.execute("UPDATE image_items SET store_specific=json_set(store_specific,'$.pixel_sha256',?,'$.extraction_version',?) WHERE id=?",
                                 (meta['pixel_sha256'], VERSION, old_images[name].id))
        saved = [{**m, 'file_path': str(destination/Path(m['file_path']).name)} for m in metadata]
        progressive.register_pages(orch.content.db_path, ns, doc_id, pages, saved)
        progressive.mark_text(orch.content.db_path, ns, doc_id, pages)
        by_page = {}
        for meta in saved:
            iid = 'ima_'+hashlib.sha256(f"{ns}:{doc_id}:{Path(meta['file_path']).name}".encode()).hexdigest()
            by_page.setdefault(meta['page_number'], []).append(iid)
        for num, item in old_content.items():
            orch.content.update_store_specific(item.id, {'image_ids': by_page.get(num, [])})
        save_pages(path, pages)
        with SessionLocal() as db:
            row = db.query(FolderSource).filter_by(user_id=source.user_id, source_id=source.source_id).one()
            row.raw_text = '\n\n'.join(p['text'] for p in pages if p['text'])
            db.commit()
    report['seconds'] = round(time.monotonic()-started, 2)
    print('SOURCE_REPAIRED', json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--user-id', type=int, required=True)
    parser.add_argument('--folder', action='append', required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1]/'.env')
    from database import SessionLocal, FolderSource, LearningJob
    import oma_provider
    with SessionLocal() as db:
        sources = db.query(FolderSource).filter(FolderSource.user_id == args.user_id,
            FolderSource.folder_name.in_(args.folder)).order_by(FolderSource.id).all()
        running = db.query(LearningJob).filter_by(status='running').all()
        ids = {s.source_id for s in sources}
        if any(json.loads(j.payload_json).get('source_id') in ids and j.lease_until > time.time() for j in running):
            raise RuntimeError('A selected source is currently indexing; finish that job first.')
    if not sources:
        raise RuntimeError('No matching sources.')
    results, cache = [], {}
    for source in sources:
        results.append(refresh(source, oma_provider._content_orchestrator(),
                               oma_provider._content_ingest_pipeline(), args.apply, cache))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
