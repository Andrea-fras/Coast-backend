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

# Figures are kept as WebP: an 82-page lecture's 97 figures took 17.7 MB as PNG, 5.4 MB as WebP,
# with no difference the AI or a student can see. Older caches hold PNG and keep working.
FIGURE_EXT = "webp"
FIGURE_QUALITY = 85


def save_figure(pil, target):
    pil.convert("RGB").save(target, "WEBP", quality=FIGURE_QUALITY, method=4)


def store_once(source, target):
    """Give a figure a second name without a second copy: a hard link on the same disk (the
    page copy and OMA's image folder both live on it), a copy only across disks."""
    target = Path(target)
    if target.exists():
        if target.samefile(source):
            return target
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        import shutil
        shutil.copyfile(source, target)
    return target


class LazyImage:
    """A figure saved in the page cache, standing in for a PIL image. Its pixels are read only
    for the moment an operation needs them (a fingerprint, a save), so a file's figures never
    sit in memory together: decoded, 97 figures of an 82-page lecture took 360 MB."""

    def __init__(self, path, width=None, height=None):
        self.path = Path(path)
        if not (width and height):
            import file_store
            file_store.local(self.path)
        if not (width and height):
            with Image.open(self.path) as img:  # reads the header only, not the pixels
                width, height = img.size
        self.width, self.height = int(width), int(height)

    @property
    def size(self):
        return (self.width, self.height)

    def load(self):
        """The pixels, decoded now and owned by the caller (nothing is kept here)."""
        import file_store
        with Image.open(file_store.local(self.path)) as img:
            return img.convert('RGB')

    def convert(self, mode):
        return self.load().convert(mode)

    def save(self, fp, format=None, **params):
        """Saving as PNG copies the cached file as it is: no decoding, no re-encoding."""
        import file_store
        file_store.local(self.path)
        if (format or '').upper() == 'PNG' and not params and isinstance(fp, (str, os.PathLike)):
            if Path(fp).resolve() == self.path.resolve():
                return  # already there: extraction wrote it straight into the cache
            import shutil
            shutil.copyfile(self.path, fp)
        else:
            self.load().save(fp, format, **params)

    def __getattr__(self, name):  # anything else a PIL image offers (resize, histogram, ...)
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(self.load(), name)


def save_pages(path,pages):
    directory=cache_dir(path)
    directory.mkdir(parents=True,exist_ok=True)
    rows=[]
    for page in pages:
        images=[]
        for image in page.get('images') or []:
            pil=image['pil_image']
            if isinstance(pil,LazyImage):  # already written by extraction, in its own format
                name=f"p{page['page_number']}_i{image['idx']}{pil.path.suffix}"
                store_once(pil.path,directory/name)
            else:
                name=f"p{page['page_number']}_i{image['idx']}.{FIGURE_EXT}"
                save_figure(pil,directory/name)
            images.append({k:v for k,v in image.items() if k!='pil_image'} | {'file':name})
        rows.append({'page_number':page['page_number'],'text':page.get('text',''),'images':images})
    manifest={'version':VERSION,'sha256':fingerprint(path),'pages':rows}
    temporary=directory/'manifest.tmp'
    temporary.write_text(json.dumps(manifest),encoding='utf-8')
    os.replace(temporary,directory/'manifest.json')
    # Extraction writes figures as it goes; ones figure selection then dropped are not kept.
    named={image['file'] for row in rows for image in row['images']}
    for leftover in directory.glob('p*_i*.*'):
        if leftover.name not in named:
            leftover.unlink(missing_ok=True)

def load_pages(path,extract_images=True):
    directory=cache_dir(path)
    import file_store
    file_store.local(directory/'manifest.json')  # the page copy's manifest, from R2 if the cache cleared it
    try:
        manifest=json.loads((directory/'manifest.json').read_text())
        if manifest.get('version')!=VERSION or manifest.get('sha256')!=fingerprint(path):
            return None
        pages=[]
        for row in manifest['pages']:
            images=[]
            if extract_images:
                for image in row.get('images') or []:
                    lazy=LazyImage(directory/image['file'],image.get('width'),image.get('height'))
                    images.append({k:v for k,v in image.items() if k!='file'} | {'pil_image':lazy})
            pages.append({**row,'images':images})
        return pages
    except (OSError,ValueError,KeyError):
        return None
