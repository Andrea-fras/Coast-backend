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


def extract_pages(pdf_path: str | Path, extract_images: bool = True) -> list[dict[str, Any]]:
    """Extract pages with text (and optionally PIL images) from a PDF.

    Set extract_images=False to skip PyMuPDF image extraction entirely —
    useful for fast text-only ingestion."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    text_by_page = _extract_text_pdfplumber(pdf_path)
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
        from PIL import Image
    except ImportError:
        logger.warning("PyMuPDF or Pillow not installed; image extraction disabled.")
        return []

    out: list[list[dict[str, Any]]] = []
    try:
        doc = fitz.open(str(pdf_path))
        for page_idx, page in enumerate(doc):
            page_imgs: list[dict[str, Any]] = []
            for img_idx, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                try:
                    base = doc.extract_image(xref)
                    img_bytes = base.get("image")
                    if not img_bytes:
                        continue
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    w, h = pil_img.size
                    if w < 80 or h < 80:
                        continue
                    if w * h < 10000:
                        continue
                    # Resize for downstream vision/cost.
                    MAX_DIM = 1200
                    if w > MAX_DIM or h > MAX_DIM:
                        ratio = min(MAX_DIM / w, MAX_DIM / h)
                        pil_img = pil_img.resize((int(w * ratio), int(h * ratio)))
                        w, h = pil_img.size
                    page_imgs.append({
                        "idx": img_idx,
                        "pil_image": pil_img,
                        "width": w,
                        "height": h,
                    })
                except Exception as e:
                    logger.debug(f"failed to extract image {img_idx} on page {page_idx}: {e}")
                    continue
            out.append(page_imgs)
        doc.close()
    except Exception as e:
        logger.warning(f"PyMuPDF failed: {e}")
        return []
    return out
