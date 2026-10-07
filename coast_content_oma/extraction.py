"""PDF text + image extraction.

PyMuPDF (fitz) reads both text and images (pdfplumber and pypdf are fallbacks). Produces
a list of page dicts: [{page_number, text, images: [{idx, pil_image, bbox}]}, ...]
Figures are then thinned by select_figures: slide templates, backgrounds, icons and
repeats are not worth describing.

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


def extract_pages(pdf_path: str | Path, extract_images: bool = True, use_cache: bool = True,
                  cache_to: Path | None = None) -> list[dict[str, Any]]:
    """Extract pages with text (and optionally PIL images) from a PDF.

    Set extract_images=False to skip PyMuPDF image extraction entirely —
    useful for fast text-only ingestion. With cache_to, each figure is written there as soon
    as it is extracted and stands in as a LazyImage, so no file's figures are held together."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    if use_cache:
        from .normalized_source import load_pages
        cached = load_pages(pdf_path, extract_images)
        if cached is not None:
            return select_figures(cached) if extract_images else cached
    if pdf_path.suffix.lower() == '.pptx':
        pages = _extract_pptx(pdf_path, extract_images)
        return select_figures(pages) if extract_images else pages

    import fitz
    try:
        with fitz.open(pdf_path) as doc:
            # PDF plain text loses baseline information: 3 superscript 150 becomes 3150,
            # so pages with scripts keep them marked (^(...), _(...)).
            text_by_page = []
            for page in doc:
                structured, has_scripts = _text_with_scripts(page)
                text_by_page.append(structured if has_scripts else _page_text(page))
    except Exception as e:  # a PDF PyMuPDF can't read: fall back to the pure-Python readers
        logger.warning(f"PyMuPDF text extraction failed ({e}); falling back to pdfplumber.")
        text_by_page = _extract_text_pdfplumber(pdf_path)
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

    images_by_page = _extract_images_pymupdf(pdf_path, cache_to) if extract_images else []

    n_pages = max(len(text_by_page), len(images_by_page))
    pages: list[dict[str, Any]] = []
    for i in range(n_pages):
        pages.append({
            "page_number": i + 1,
            "text": text_by_page[i] if i < len(text_by_page) else "",
            "images": images_by_page[i] if i < len(images_by_page) else [],
        })
    return select_figures(pages) if extract_images else pages


def _page_text(page) -> str:
    """A page's text in reading lines, about 15x faster than pdfplumber.

    PyMuPDF's own lines keep the PDF's spacing; fragments that sit on the same visual
    line (inline maths is often drawn a little higher, as separate lines) are joined
    left to right. Only text drawn inside the page is read: text hidden behind a cropped
    image or outside the slide never reaches a lesson."""
    import fitz
    frags = []
    # TEXTFLAGS_TEXT: text only (the default dict mode also decodes every image on the page).
    for block in page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            if text:
                x0, y0, x1, y1 = line["bbox"]
                frags.append((x0, y0, x1, y1, text, max(span["size"] for span in line["spans"])))
    frags.sort(key=lambda f: ((f[1] + f[3]) / 2, f[0]))
    rows: list[list] = []  # [top, bottom, fragments]
    for f in frags:
        height = max(f[3] - f[1], 0.1)
        for row in reversed(rows[-4:]):
            if min(row[1], f[3]) - max(row[0], f[1]) >= 0.5 * min(height, row[1] - row[0]):
                row[2].append(f)
                row[0], row[1] = min(row[0], f[1]), max(row[1], f[3])
                break
        else:
            rows.append([f[1], f[3], [f]])
    rows.sort(key=lambda row: row[0])
    out = []
    for _, _, fs in rows:
        fs.sort(key=lambda f: f[0])
        text = fs[0][4]
        for a, b in zip(fs, fs[1:]):
            # A smaller fragment touching the one before it is a sub- or superscript.
            script = min(a[5], b[5]) < 0.85 * max(a[5], b[5])
            text += ("" if script and b[0] - a[2] < 1.0 else " ") + b[4]
        out.append(text)
    return "\n".join(out)


def _dhash(img) -> int:
    """A 256-bit difference hash: near-identical images (the same logo rendered at another
    size or crop) differ in only a few bits."""
    from PIL import Image
    g = img.convert("L").resize((17, 16), Image.BILINEAR)
    px = g.tobytes()  # one byte per pixel in L mode
    bits = 0
    for y in range(16):
        for x in range(16):
            bits = (bits << 1) | (px[y * 17 + x] > px[y * 17 + x + 1])
    return bits


def select_figures(pages: list[dict[str, Any]], report: list | None = None) -> list[dict[str, Any]]:
    """Keep only figures worth describing, before any AI sees them.

    - Slide templates: an image (or near-copies of it) on at least a quarter of the
      slides, a banner strip along the top or bottom on three or more, or a full-page
      background on five or more.
    - Icons: shown at under 0.3% of the page.
    - Repeats: an image used on several slides (build-up slides) is kept once, on its first
      page, with the others listed in also_on_pages.
    Full-page renders of drawn diagrams (page_diagram) are always kept."""
    from PIL import ImageChops, ImageStat
    n_pages = max(1, len(pages))
    items = []
    for page in pages:
        for img in page.get("images") or []:
            if img.get("extraction_kind") == "page_diagram" or "pil_image" not in img:
                continue
            pil = img["pil_image"]
            items.append({"page": page["page_number"], "img": img, "hash": _dhash(pil),
                          "aspect": pil.width / max(pil.height, 1), "thumb": pil.convert("RGB").resize((24, 24))})

    def same(a, b):
        """The same picture (re-encoded, rescaled or re-rendered), not merely a similar one:
        two road signs in the same red circle differ in their pictogram, and both matter."""
        if abs(a["aspect"] - b["aspect"]) > 0.06 * max(a["aspect"], b["aspect"]):
            return False
        if bin(a["hash"] ^ b["hash"]).count("1") > 16:
            return False
        return max(ImageStat.Stat(ImageChops.difference(a["thumb"], b["thumb"])).mean) <= 6

    # Each group is compared through its first member, so similar images never chain
    # together into one group.
    groups: list[list] = []
    for item in items:
        for members in groups:
            if same(members[0], item):
                members.append(item)
                break
        else:
            groups.append([item])

    drop: set[int] = set()
    for members in groups:
        on_pages = sorted({m["page"] for m in members})
        boxes = [m["img"].get("bbox") for m in members if m["img"].get("bbox")]
        strip = any((b[3] - b[1]) < 0.2 and (b[2] - b[0]) > 0.5 and (b[1] < 0.15 or b[3] > 0.85) for b in boxes)
        background = any((b[2] - b[0]) * (b[3] - b[1]) > 0.7 for b in boxes)
        # A full-page photo on a few slides is usually content (the lecture's subject);
        # the same full-page image on five or more is the slide background.
        template = (len(on_pages) >= max(4, n_pages / 4) or (len(on_pages) >= 3 and strip)
                    or (len(on_pages) >= 5 and background))
        if report is not None and (template or len(members) > 1):
            report.append({"action": "template" if template else "kept once", "pages": on_pages,
                           "strip": strip, "background": background, "image": members[0]["img"]})
        if template:
            drop.update(id(m["img"]) for m in members)
            continue
        first = members[0]["img"]  # items are in page order
        drop.update(id(m["img"]) for m in members[1:])
        others = sorted(set(first.get("also_on_pages") or []) | set(p for p in on_pages if p != members[0]["page"]))
        if others:
            first["also_on_pages"] = others
    for item in items:
        b = item["img"].get("bbox")
        if b and (b[2] - b[0]) * (b[3] - b[1]) < 0.003:
            drop.add(id(item["img"]))
    kept = total = 0
    for page in pages:
        imgs = page.get("images") or []
        total += len(imgs)
        page["images"] = [img for img in imgs if id(img) not in drop]
        kept += len(page["images"])
    if total != kept:
        logger.info("figures: kept %d of %d (dropped slide templates, icons and repeats)", kept, total)
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


def _extract_images_pymupdf(pdf_path: Path, cache_to: Path | None = None) -> list[list[dict[str, Any]]]:
    """Return per-page list of {idx, pil_image, bbox} entries. Filters
    out tiny/decorative images at the geometry level (less than 80x80
    or area < 10K). With cache_to, every kept image is saved there at once and returned as a
    LazyImage; a reused image (the same object on several slides) points at its first file."""
    try:
        import fitz  # PyMuPDF
        from PIL import Image, ImageStat
    except ImportError:
        logger.warning("PyMuPDF or Pillow not installed; image extraction disabled.")
        return []

    out: list[list[dict[str, Any]]] = []
    try:
        doc = fitz.open(str(pdf_path))
        decoded: dict[int, Any] = {}  # xref -> image: a logo on every slide is decoded once
        from .normalized_source import FIGURE_EXT, LazyImage, save_figure
        if cache_to is not None:
            Path(cache_to).mkdir(parents=True, exist_ok=True)

        def keep(pil, name):
            """With a cache folder, the image goes to disk now and only its stand-in stays."""
            if cache_to is None or isinstance(pil, LazyImage):
                return pil
            target = Path(cache_to) / name
            save_figure(pil, target)
            return LazyImage(target, pil.width, pil.height)
        # The same image object on a quarter or more of the slides is part of the slide
        # template (a logo, a header banner): skip it before decoding anything.
        listed = [page.get_images(full=True) for page in doc]
        on_pages: dict[int, int] = {}
        for imgs in listed:
            for xref in {img[0] for img in imgs}:
                on_pages[xref] = on_pages.get(xref, 0) + 1
        template = {x for x, n in on_pages.items() if n >= max(4, len(listed) / 4)}
        for page_idx, page in enumerate(doc):
            page_imgs: list[dict[str, Any]] = []
            needs_render = False
            pw, ph = page.rect.width or 1, page.rect.height or 1
            # Where each image sits. A slide with many images is scanned once; otherwise each
            # is found by name (get_image_rects checksums the pixels and is far slower).
            placements = None
            if len(listed[page_idx]) > 8:
                placements = {}
                for info in page.get_image_info(xrefs=True):
                    if info.get("xref"):
                        placements.setdefault(info["xref"], fitz.Rect(info["bbox"]))
            for img_idx, img in enumerate(listed[page_idx]):
                xref = img[0]
                if xref in template:
                    continue
                try:
                    if placements is not None:
                        placed = placements.get(xref)
                    else:
                        try:
                            placed = page.get_image_bbox(img)
                        except Exception:
                            placed = None
                    shown = [placed] if placed is not None and placed.is_valid and not placed.is_empty and not placed.is_infinite else []
                    if not shown and img[1]:  # a masked image must be rendered from its exact spot
                        shown = page.get_image_rects(xref)
                    bbox = None
                    if shown:
                        r = shown[0] & page.rect
                        bbox = [round(r.x0 / pw, 3), round(r.y0 / ph, 3), round(r.x1 / pw, 3), round(r.y1 / ph, 3)]
                    # Rendering the occurrence also preserves its soft mask and actual
                    # slide background (which is not necessarily white).
                    rects = shown if img[1] else []
                    if rects:
                        if xref not in decoded:  # rendered once, on its first slide
                            decoded[xref] = _render_clip(page, rects[0], max_size=min(1200, max(img[2:4])))
                        pil_img = decoded[xref]
                    else:
                        if xref not in decoded:
                            img_bytes = doc.extract_image(xref).get("image")
                            if not img_bytes:
                                decoded[xref] = None
                                continue
                            with Image.open(io.BytesIO(img_bytes)) as original:
                                pil_img = original.convert("RGB")
                            # Same cap as rendered clips and PPTX images: a camera photo decodes
                            # to tens of MB, and every page's images are held until saved.
                            pil_img.thumbnail((1200, 1200))
                            decoded[xref] = pil_img
                        pil_img = decoded[xref]
                        if pil_img is None:
                            continue
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
                    pil_img = keep(pil_img, f"p{page_idx + 1}_i{img_idx}.{FIGURE_EXT}")
                    if decoded.get(xref) is not None and not isinstance(decoded[xref], type(pil_img)):
                        decoded[xref] = pil_img  # later slides reuse the saved file, not the pixels
                    page_imgs.append({
                        "idx": img_idx,
                        "extraction_kind": "composited" if rects else "raster",
                        "pil_image": pil_img,
                        "width": w,
                        "height": h,
                        "bbox": bbox,  # where it sits on the page, as fractions of its size
                    })
                except Exception as e:
                    logger.debug(f"failed to extract image {img_idx} on page {page_idx}: {e}")
                    needs_render = True
                    continue
            if needs_render or _has_vector_diagram(page):
                # At most one extra image per page; full-page rendering retains labels
                # that often sit outside the drawing's bounding box.
                preview = keep(_render_clip(page, page.rect), f"p{page_idx + 1}_i1000000.{FIGURE_EXT}")
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
    from .normalized_source import cache_dir, save_pages
    pages = extract_pages(pdf_path, extract_images=True, use_cache=False, cache_to=cache_dir(Path(pdf_path)))
    save_pages(Path(pdf_path), pages)
    return [{"page_number": p["page_number"], "text": p.get("text", "")} for p in pages]

