"""Reusable normalized pages for PDF/PPTX ingestion, keyed by source bytes."""
import hashlib
import json
import os
from pathlib import Path
from PIL import Image

VERSION = 2  # Baseline-aware math, composited masks, and vector diagram previews.

def cache_dir(path):
    return Path(str(path) + '.pages')

def fingerprint(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def save_pages(path,pages):
    directory=cache_dir(path)
    directory.mkdir(parents=True,exist_ok=True)
    rows=[]
    for page in pages:
        images=[]
        for image in page.get('images') or []:
            name=f"p{page['page_number']}_i{image['idx']}.png"
            image['pil_image'].save(directory/name,'PNG')
            images.append({k:v for k,v in image.items() if k!='pil_image'} | {'file':name})
        rows.append({'page_number':page['page_number'],'text':page.get('text',''),'images':images})
    manifest={'version':VERSION,'sha256':fingerprint(path),'pages':rows}
    temporary=directory/'manifest.tmp'
    temporary.write_text(json.dumps(manifest),encoding='utf-8')
    os.replace(temporary,directory/'manifest.json')

def load_pages(path,extract_images=True):
    directory=cache_dir(path)
    try:
        manifest=json.loads((directory/'manifest.json').read_text())
        if manifest.get('version')!=VERSION or manifest.get('sha256')!=fingerprint(path):
            return None
        pages=[]
        for row in manifest['pages']:
            images=[]
            if extract_images:
                for image in row.get('images') or []:
                    with Image.open(directory/image['file']) as img:
                        images.append({k:v for k,v in image.items() if k!='file'} | {'pil_image':img.convert('RGB')})
            pages.append({**row,'images':images})
        return pages
    except (OSError,ValueError,KeyError):
        return None
