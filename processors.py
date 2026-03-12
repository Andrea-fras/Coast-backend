"""Processors for converting input files (PDFs, images) into a list of PIL images."""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image


def load_images_from_path(file_path: str | Path) -> list[Image.Image]:
    """Load a file and return a list of PIL Images (one per page for PDFs)."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.suffix.lower()

    if ext == ".pdf":
        return _pdf_to_images(file_path)
    elif ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp", ".gif"}:
        return [Image.open(file_path).convert("RGB")]
    else:
        raise ValueError(
            f"Unsupported file format: {ext}. "
            "Supported formats: .pdf, .png, .jpg, .jpeg, .tiff, .bmp, .webp, .gif"
        )


def _pdf_to_images(pdf_path: Path, dpi: int = 150) -> list[Image.Image]:
    """Convert a PDF file to a list of PIL images (one per page)."""
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError(
            "pdf2image is required for PDF processing. "
            "Install it with: pip install pdf2image\n"
            "You also need poppler installed:\n"
            "  macOS: brew install poppler\n"
            "  Ubuntu: sudo apt-get install poppler-utils\n"
            "  Windows: download from https://github.com/oschwartz10612/poppler-windows"
        )

    images = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        fmt="png",
        thread_count=4,
    )
    return [img.convert("RGB") for img in images]


def extract_diagram_regions(
    page_image: Image.Image,
    regions: list[dict],
    output_dir: str | Path,
    prefix: str = "diagram",
) -> list[str]:
    """
    Crop diagram regions from a page image and save them.

    Args:
        page_image: The full page image.
        regions: List of dicts with keys: x, y, width, height (in pixels).
        output_dir: Directory to save cropped images.
        prefix: Filename prefix for saved images.

    Returns:
        List of saved file paths (relative to output_dir).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    for i, region in enumerate(regions):
        x, y, w, h = region["x"], region["y"], region["width"], region["height"]
        cropped = page_image.crop((x, y, x + w, y + h))
        filename = f"{prefix}_{i + 1}.png"
        filepath = output_dir / filename
        cropped.save(filepath, "PNG")
        saved_paths.append(filename)

    return saved_paths
