"""Extract educationally relevant images from PDF/PPTX sources and store them."""

from __future__ import annotations

import traceback
from pathlib import Path


def extract_and_store_images(
    file_path: str | Path,
    ext: str,
    source_id: str,
    user_id: int,
    folder_name: str,
    images_dir: Path | None = None,
):
    """Extract educationally relevant images from a PDF/PPTX and store them on disk + DB.

    Filters out tiny/decorative images using the existing extractor pipeline.
    Resizes large images for web display.
    """
    try:
        from extractor import extract_content_from_pdf, extract_content_from_pptx
        from database import SessionLocal, SourceImage

        if images_dir is None:
            _pd = Path("/data")
            _base = _pd / "folder_uploads" if _pd.is_dir() else Path(__file__).parent / "folder_uploads"
            images_dir = _base / "images"

        if ext == ".pdf":
            pages = extract_content_from_pdf(str(file_path))
        elif ext == ".pptx":
            pages = extract_content_from_pptx(str(file_path))
        else:
            return

        if not pages:
            return

        img_dir = images_dir / source_id
        img_dir.mkdir(parents=True, exist_ok=True)

        db = SessionLocal()
        try:
            count = 0
            for page in pages:
                page_images = page.get("images", [])
                page_text = page.get("text", "")
                page_num = page.get("slide_number", 0)

                for img_idx, img in enumerate(page_images):
                    w, h = img.size
                    if w < 80 or h < 80:
                        continue
                    if w * h < 10000:
                        continue

                    MAX_DIM = 1200
                    if w > MAX_DIM or h > MAX_DIM:
                        ratio = min(MAX_DIM / w, MAX_DIM / h)
                        img = img.resize((int(w * ratio), int(h * ratio)), 1)
                        w, h = img.size

                    fname = f"p{page_num}_i{img_idx}.png"
                    img_path = img_dir / fname
                    img.save(str(img_path), "PNG", optimize=True)

                    context_snippet = page_text[:500] if page_text else ""

                    si = SourceImage(
                        source_id=source_id,
                        user_id=user_id,
                        folder_name=folder_name,
                        page_number=page_num,
                        context_text=context_snippet,
                        image_path=str(img_path),
                        width=w,
                        height=h,
                    )
                    db.add(si)
                    count += 1

            db.commit()
            print(f"[images] Extracted {count} images from {source_id}")
        finally:
            db.close()

    except Exception:
        traceback.print_exc()
