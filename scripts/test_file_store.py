#!/usr/bin/env python3
"""The file store: R2 holds every uploaded file, the server's disk is a cache in front of it.
Offline: an in-memory stand-in for R2 and a temporary data disk."""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DISK = Path(tempfile.mkdtemp(prefix="coast-store-"))
os.environ.update(BACKUP_DATA_ROOT=str(DISK), FILE_STORE="on", R2_PREFIX="")
import file_store  # noqa: E402


class FakeR2:
    """Just what the store uses of an S3 client."""

    def __init__(self):
        self.objects = {}

    def upload_file(self, path, bucket, key):
        self.objects[key] = Path(path).read_bytes()

    def download_file(self, bucket, key, path):
        if key not in self.objects:
            raise FileNotFoundError(key)
        Path(path).write_bytes(self.objects[key])

    def copy_object(self, Bucket, Key, CopySource):
        self.objects[Key] = self.objects[CopySource["Key"]]

    def delete_objects(self, Bucket, Delete):
        for o in Delete["Objects"]:
            self.objects.pop(o["Key"], None)

    def get_paginator(self, name):
        fake = self

        class Pages:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k, "Size": len(v)} for k, v in fake.objects.items() if k.startswith(Prefix)]}
        return Pages()


class FileStore(unittest.TestCase):
    def setUp(self):
        self.r2 = FakeR2()
        file_store._r2_cache[:] = [(self.r2, "bucket")]
        (DISK / "file_store.db").unlink(missing_ok=True)
        import shutil
        shutil.rmtree(DISK / "folder_uploads", ignore_errors=True)
        self.pdf = DISK / "folder_uploads" / "src_1.pdf"
        self.figure = DISK / "folder_uploads" / "src_1.pdf.pages" / "p3_i0.webp"
        for p, data in ((self.pdf, b"%PDF lecture"), (self.figure, b"RIFF....WEBPfigure")):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)

    def test_a_file_cleared_from_the_disk_comes_back_from_r2(self):
        file_store.publish(self.pdf, wait=True)
        self.assertEqual(self.r2.objects["store/folder_uploads/src_1.pdf"], b"%PDF lecture")
        self.pdf.unlink()
        self.assertEqual(file_store.local(self.pdf).read_bytes(), b"%PDF lecture")

    def test_files_outside_the_disk_and_page_renders_never_go_to_r2(self):
        shipped = ROOT / "requirements.txt"
        file_store.publish(shipped, wait=True)
        render = self.pdf.parent / "src_1.pdf.pages" / "render-1600" / "p1.png"
        render.parent.mkdir(parents=True)
        render.write_bytes(b"png")
        file_store.publish_tree(self.pdf.parent / "src_1.pdf.pages", wait=True)
        self.assertEqual(sorted(self.r2.objects), ["store/folder_uploads/src_1.pdf.pages/p3_i0.webp"])
        self.assertEqual(file_store.local(shipped), shipped)

    def test_a_deleted_source_goes_to_the_trash_even_after_the_cache_cleared_it(self):
        file_store.publish(self.pdf, wait=True)
        file_store.publish_tree(self.figure.parent, wait=True)
        self.figure.unlink()  # the cache already cleared the page copy's figure
        file_store.remove([self.pdf, self.figure.parent])
        file_store._pool.submit(lambda: None).result()  # let the background move finish
        time.sleep(0.2)
        self.assertFalse(self.pdf.exists())
        self.assertFalse(any(k.startswith("store/") for k in self.r2.objects))
        trashed = sorted(k.split("/", 2)[2] for k in self.r2.objects if k.startswith("trash/"))
        self.assertEqual(trashed, ["folder_uploads/src_1.pdf", "folder_uploads/src_1.pdf.pages/p3_i0.webp"])
        self.assertFalse(file_store.published("store/folder_uploads/src_1.pdf"))

    def test_a_full_disk_clears_least_recently_used_files_r2_holds(self):
        file_store.publish(self.pdf, wait=True)
        file_store.publish(self.figure, wait=True)
        old = time.time() - 86400 * 3
        os.utime(self.pdf, (old, old))  # not used for days
        unsent = DISK / "folder_uploads" / "src_2.pdf"
        unsent.write_bytes(b"%PDF not in R2 yet")
        usage = type("U", (), {"total": 100, "used": 80})()
        with patch("file_store.shutil.disk_usage", return_value=usage):
            cleared = file_store.evict()
        self.assertGreaterEqual(cleared, 1)
        self.assertFalse(self.pdf.exists())          # the oldest file R2 holds went first
        self.assertTrue(unsent.exists())             # a file R2 does not hold is never cleared
        self.assertEqual(file_store.local(self.pdf).read_bytes(), b"%PDF lecture")  # and comes back on use

    def test_the_sweep_sends_what_r2_lacks_once(self):
        self.assertEqual(file_store.sweep([DISK / "folder_uploads"]), 2)
        self.assertEqual(file_store.sweep([DISK / "folder_uploads"]), 0)

    def test_off_means_the_disk_alone(self):
        with patch.dict(os.environ, {"FILE_STORE": "off"}):
            file_store.publish(self.pdf, wait=True)
            self.pdf.unlink()
            self.assertFalse(file_store.local(self.pdf).exists())
        self.assertEqual(self.r2.objects, {})


if __name__ == "__main__":
    unittest.main(verbosity=1)
