"""Reading uploaded PDFs: fast PyMuPDF text and the figure filter; offline, no model calls.

    python3 -m unittest scripts.test_extraction
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fitz  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from coast_content_oma.extraction import _page_text, select_figures  # noqa: E402


def picture(kind, size=(200, 150)):
    """Small synthetic images: a sign with a pictogram, a logo, a banner, a background."""
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    if kind.startswith("sign"):
        d.ellipse([20, 10, 180, 140], outline="red", width=14)
        if kind == "sign-bike":
            d.ellipse([60, 80, 90, 110], outline="black", width=5); d.ellipse([110, 80, 140, 110], outline="black", width=5)
        else:  # a pedestrian
            d.line([100, 40, 100, 100], fill="black", width=6); d.line([100, 100, 80, 125], fill="black", width=6)
    elif kind == "logo":
        d.text((20, 60), "macsbio", fill="orange"); d.rectangle([10, 10, 60, 50], fill="orange")
    elif kind == "chalk":
        im = Image.new("RGB", size, (40, 45, 48))
    elif kind == "diagram":
        for i in range(6):
            d.rectangle([10 + 30 * i, 20 + 10 * i, 40 + 30 * i, 50 + 10 * i], outline="blue", width=3)
    return im


def page(n, *imgs):
    return {"page_number": n, "text": "", "images": [
        {"idx": i, "pil_image": im, "bbox": bbox, "width": im.width, "height": im.height, **extra}
        for i, (im, bbox, extra) in enumerate(imgs)]}


MID = [0.3, 0.3, 0.7, 0.7]          # an ordinary figure in the middle of the slide
TOP_STRIP = [0.0, 0.0, 1.0, 0.12]   # a banner along the top
FULL = [0.0, 0.0, 1.0, 1.0]         # a full-page background
ICON = [0.5, 0.5, 0.53, 0.53]       # 0.09% of the page


def kept(pages):
    return [(p["page_number"], i["idx"]) for p in pages for i in p["images"]]


class Figures(unittest.TestCase):
    def test_logo_on_most_slides_is_dropped(self):
        pages = [page(n, (picture("logo"), MID, {})) for n in range(1, 13)]
        self.assertEqual(kept(select_figures(pages)), [])

    def test_repeated_figure_is_kept_once_with_its_other_pages(self):
        fig = picture("diagram")
        pages = [page(n, (fig.copy(), MID, {})) for n in (1, 2, 3)] + [page(n) for n in range(4, 20)]
        out = select_figures(pages)
        self.assertEqual(kept(out), [(1, 0)])
        self.assertEqual(out[0]["images"][0]["also_on_pages"], [2, 3])

    def test_similar_but_different_pictures_are_both_kept(self):
        pages = [page(1, (picture("sign-bike"), MID, {})), page(2, (picture("sign-walker"), MID, {}))] + [page(n) for n in range(3, 12)]
        self.assertEqual(kept(select_figures(pages)), [(1, 0), (2, 0)])

    def test_banner_strip_on_three_slides_is_dropped(self):
        banner = picture("logo", (800, 90))
        pages = [page(n, (banner.copy(), TOP_STRIP, {})) for n in (1, 2, 3)] + [page(n) for n in range(4, 30)]
        self.assertEqual(kept(select_figures(pages)), [])

    def test_full_page_photo_on_few_slides_is_content_but_a_background_on_many_is_not(self):
        photo = picture("diagram", (400, 300))
        few = [page(n, (photo.copy(), FULL, {})) for n in (1, 2, 3)] + [page(n) for n in range(4, 30)]
        self.assertEqual(kept(select_figures(few)), [(1, 0)])
        chalk = picture("chalk", (400, 300))
        many = [page(n, (chalk.copy(), FULL, {})) for n in range(1, 8)] + [page(n) for n in range(8, 40)]
        self.assertEqual(kept(select_figures(many)), [])

    def test_icons_are_dropped(self):
        pages = [page(1, (picture("diagram"), ICON, {}), (picture("sign-bike"), MID, {}))]
        self.assertEqual(kept(select_figures(pages)), [(1, 1)])

    def test_rendered_diagram_pages_are_always_kept(self):
        render = picture("diagram", (400, 300))
        pages = [page(n, (render.copy(), FULL, {"extraction_kind": "page_diagram"})) for n in range(1, 10)]
        self.assertEqual(len(kept(select_figures(pages))), 9)


class Text(unittest.TestCase):
    def test_inline_maths_stays_in_its_sentence_and_hidden_text_is_skipped(self):
        doc = fitz.open()
        p = doc.new_page(width=600, height=400)
        p.insert_text((40, 100), "A directed graph is a tuple", fontsize=14)
        p.insert_text((250, 97), "G := (V, E)", fontsize=14)  # drawn a little higher, as maths often is
        p.insert_text((340, 100), ", where V is a set of nodes.", fontsize=14)
        p.insert_text((40, 700), "hidden caption outside the slide", fontsize=14)
        text = _page_text(p)
        self.assertEqual(text, "A directed graph is a tuple G := (V, E) , where V is a set of nodes.")
        doc.close()

    def test_lines_keep_their_order_and_spacing(self):
        doc = fitz.open()
        p = doc.new_page(width=600, height=400)
        p.insert_text((40, 60), "Title of the slide", fontsize=20)
        p.insert_text((40, 120), "fietsers en voetgangers", fontsize=12)
        self.assertEqual(_page_text(p).splitlines(), ["Title of the slide", "fietsers en voetgangers"])
        doc.close()


if __name__ == "__main__":
    unittest.main()
