"""PDF text + image extraction.

Reuses pdfplumber for text and PyMuPDF (fitz) for images. Produces a
list of page dicts: [{page_number, text, images: [{idx, pil_image, bbox}]}, ...]

This is intentionally separate from ingestion.py so we can swap the
extractor later (e.g. use Coast's existing extractor.py) without
touching the ingestion logic.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def extract_pages(pdf_path: str | Path, extract_images: bool = True, use_cache: bool = True) -> list[dict[str, Any]]:
    """Extract pages with text (and optionally PIL images) from a PDF.

    Set extract_images=False to skip PyMuPDF image extraction entirely —
    useful for fast text-only ingestion."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    if use_cache:
        from .normalized_source import load_pages
        cached = load_pages(pdf_path, extract_images)
        if cached is not None:
            return cached
    if pdf_path.suffix.lower() == '.pptx':
        return _extract_pptx(pdf_path, extract_images)

    text_by_page = _extract_text_pdfplumber(pdf_path)
    # PDF plain text loses baseline information: 3 superscript 150 becomes 3150.
    # Only replace pages with detected scripts; retain the existing layout path elsewhere.
    import fitz
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            structured, has_scripts = _text_with_scripts(page)
            if has_scripts and i < len(text_by_page):
                text_by_page[i] = structured
    # OCR only pages without usable text; preserve the existing text extractor.
    if any(not text.strip() for text in text_by_page):
        import fitz
        with fitz.open(pdf_path) as doc:
            for i, text in enumerate(text_by_page):
                if text.strip():
                    continue
                try:
                    import os
                    page = doc[i]  # Keep the TextPage's parent alive through extraction.
                    textpage = page.get_textpage_ocr(language=os.getenv('OMA_OCR_LANGUAGE', 'eng'), full=True)
                    text_by_page[i] = page.get_text(textpage=textpage)
                except Exception as exc:
                    logger.warning('OCR unavailable for page %s: %s', i + 1, type(exc).__name__)

    images_by_page = _extract_images_pymupdf(pdf_path) if extract_images else []

    n_pages = max(len(text_by_page), len(images_by_page))
    pages: list[dict[str, Any]] = []
    for i in range(n_pages):
        pages.append({
            "page_number": i + 1,
            "text": text_by_page[i] if i < len(text_by_page) else "",
            "images": images_by_page[i] if i < len(images_by_page) else [],
        })
    return pages


def _text_with_scripts(page) -> tuple[str, bool]:
    """Preserve PDF superscripts and inline subscripts without guessing formulas."""
    lines = []
    changed = False
    for block in page.get_text('dict', flags=0, sort=True)['blocks']:
        for line in block.get('lines', []):
            spans = line['spans']
            normal = [s for s in spans if not s['flags'] & 1 and s['text'].strip()]
            base = max(normal, key=lambda s: s['size'], default=None)
            runs = []
            for span in spans:
                text = span['text']
                kind = ''
                if text.strip() and span['flags'] & 1:
                    kind = '^'
                elif (base and text.strip() and span['size'] < base['size'] * .9
                      and span['origin'][1] - base['origin'][1] > base['size'] * .12):
                    kind = '_'
                if runs and kind and runs[-1][0] == kind:
                    runs[-1] = (kind, runs[-1][1] + text)
                else:
                    runs.append((kind, text))
                changed |= bool(kind)
            lines.append(''.join(f'{kind}({text.strip()})' if kind else text for kind, text in runs))
    return '\n'.join(lines), changed


def _extract_text_pdfplumber(pdf_path: Path) -> list[str]:
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed; falling back to pypdf.")
        return _extract_text_pypdf(pdf_path)
    out: list[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                out.append(text)
                page.close()  # drop the page's cached layout objects (they add up to hundreds of MB)
    except Exception as e:
        logger.warning(f"pdfplumber failed ({e}); falling back to pypdf.")
        return _extract_text_pypdf(pdf_path)
    return out


def _extract_text_pypdf(pdf_path: Path) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return []
    out: list[str] = []
    try:
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            out.append(page.extract_text() or "")
    except Exception as e:
        logger.warning(f"pypdf extraction failed: {e}")
    return out


def _extract_images_pymupdf(pdf_path: Path) -> list[list[dict[str, Any]]]:
    """Return per-page list of {idx, pil_image, bbox} entries. Filters
    out tiny/decorative images at the geometry level (less than 80x80
    or area < 10K)."""
    try:
        import fitz  # PyMuPDF
        from PIL import Image, ImageStat
    except ImportError:
        logger.warning("PyMuPDF or Pillow not installed; image extraction disabled.")
        return []

    out: list[list[dict[str, Any]]] = []
    try:
        doc = fitz.open(str(pdf_path))
        for page_idx, page in enumerate(doc):
            page_imgs: list[dict[str, Any]] = []
            needs_render = False
            for img_idx, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                try:
                    base = doc.extract_image(xref)
                    img_bytes = base.get("image")
                    if not img_bytes:
                        continue
                    # Rendering the occurrence also preserves its soft mask and actual
                    # slide background (which is not necessarily white).
                    rects = page.get_image_rects(xref) if base.get('smask') else []
                    if rects:
                        pil_img = _render_clip(page, rects[0], max_size=min(1200, max(img[2:4])))
                    else:
                        with Image.open(io.BytesIO(img_bytes)) as original:
                            pil_img = original.convert("RGB")
                        # Same cap as rendered clips and PPTX images: a camera photo decodes
                        # to tens of MB, and every page's images are held until saved.
                        pil_img.thumbnail((1200, 1200))
                    w, h = pil_img.size
                    if w < 80 or h < 80:
                        continue
                    if w * h < 10000:
                        continue
                    stats = ImageStat.Stat(pil_img)
                    if max(stats.stddev) < .5 and (min(stats.mean) > 250 or max(stats.mean) < 5):
                        # Empty black/white assets aren't usable diagrams. Colored
                        # patches may be meaningful (for example in color science).
                        continue
                    # Resize for downstream vision/cost.
                    MAX_DIM = 1200
                    if w > MAX_DIM or h > MAX_DIM:
                        ratio = min(MAX_DIM / w, MAX_DIM / h)
                        pil_img = pil_img.resize((int(w * ratio), int(h * ratio)))
                        w, h = pil_img.size
                    page_imgs.append({
                        "idx": img_idx,
                        "extraction_kind": "composited" if rects else "raster",
                        "pil_image": pil_img,
                        "width": w,
                        "height": h,
                    })
                except Exception as e:
                    logger.debug(f"failed to extract image {img_idx} on page {page_idx}: {e}")
                    needs_render = True
                    continue
            if needs_render or _has_vector_diagram(page):
                # At most one extra image per page; full-page rendering retains labels
                # that often sit outside the drawing's bounding box.
                preview = _render_clip(page, page.rect)
                page_imgs.append({'idx': 1000000, 'pil_image': preview,
                                  'extraction_kind': 'page_diagram',
                                  'width': preview.width, 'height': preview.height})
            out.append(page_imgs)
        doc.close()
    except Exception as e:
        logger.warning(f"PyMuPDF failed: {e}")
        return []
    return out


def _render_clip(page, rect, max_size=1600):
    import fitz
    from PIL import Image
    clip = fitz.Rect(rect) & page.rect
    scale = max_size / max(clip.width, clip.height)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip,
                          colorspace=fitz.csRGB, alpha=False)
    return Image.frombytes('RGB', (pix.width, pix.height), pix.samples)


def _has_vector_diagram(page) -> bool:
    """Ignore slide backgrounds and simple borders, retain compound slide drawings."""
    import fitz
    paths = [d for d in page.get_drawings()
             if not (d['rect'].get_area() > page.rect.get_area() * .9 and len(d['items']) <= 2)]
    if sum(len(d['items']) for d in paths) < 8:
        return False
    bounds = fitz.Rect()
    for path in paths:
        bounds |= path['rect']
    return bounds.get_area() > page.rect.get_area() * .02


def _extract_pptx(path, extract_images=True):
    from pptx import Presentation
    from PIL import Image
    prs = Presentation(str(path))
    pages=[]
    for number, slide in enumerate(prs.slides,1):
        texts=[]
        images=[]
        def visit(shapes):
            for shape in shapes:
                if getattr(shape,'has_text_frame',False):
                    texts.append(shape.text_frame.text)
                if getattr(shape,'has_table',False):
                    texts.extend(' | '.join(cell.text for cell in row.cells) for row in shape.table.rows)
                if hasattr(shape,'shapes'):
                    visit(shape.shapes)
                if extract_images and hasattr(shape,'image'):
                    try:
                        with Image.open(io.BytesIO(shape.image.blob)) as original:
                            image=original.convert('RGB')
                        if image.width < 80 or image.height < 80:
                            continue
                        image.thumbnail((1200,1200))
                        images.append({'idx':len(images),'pil_image':image,'width':image.width,'height':image.height})
                    except Exception as exc:
                        logger.warning('PPTX image extraction failed on slide %s: %s', number,type(exc).__name__)
        visit(slide.shapes)
        if slide.has_notes_slide:
            notes=slide.notes_slide.notes_text_frame
            if notes and notes.text.strip():
                texts.append('Speaker notes: '+notes.text)
        pages.append({'page_number':number,'text':'\n'.join(texts),'images':images})
    return pages


def extract_and_cache(pdf_path: str) -> list[dict[str, Any]]:
    """Read a PDF or PowerPoint once (text and figures), save the normalized copy that later
    steps load in about a second, and return the pages' text without images.

    Uploads run this in a separate process: the PDF readers are pure Python and would
    otherwise hold the server's interpreter lock for tens of seconds per file."""
    from .normalized_source import save_pages
    pages = extract_pages(pdf_path, extract_images=True, use_cache=False)
    save_pages(Path(pdf_path), pages)
    return [{"page_number": p["page_number"], "text": p.get("text", "")} for p in pages]

