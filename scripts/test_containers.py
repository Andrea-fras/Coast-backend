#!/usr/bin/env python3
"""Indexing in containers (coast_content_oma/remote.py) gives exactly what indexing on the server
gives, opens a section as early, and falls back to the server when a container fails. Offline:
a stand-in runs the container's work in this process, through the same serialisation."""
import pickle
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.test_launch_integrity as fixture  # noqa: E402
import ai_usage  # noqa: E402
from coast_content_oma import progressive, remote  # noqa: E402
from coast_content_oma.ingestion import IngestionPipeline  # noqa: E402
from coast_content_oma.stores.concept import ConceptStore  # noqa: E402
from coast_content_oma.stores.content import ContentStore  # noqa: E402
from coast_content_oma.stores.db import connect_db  # noqa: E402
from coast_content_oma.stores.image import ImageStore  # noqa: E402

PAGES = 16


def pages():
    return [{"page_number": n, "text": ("Topic %d details " % n) * 20, "images": []} for n in range(1, PAGES + 1)]


def classify(batch):
    return [{"page_number": p["page_number"], "content_types": ["definition"], "concepts": ["topic %d" % p["page_number"]],
             "section_title": "", "summary": "page %d" % p["page_number"]} for p in batch]


def describe(batch):
    return [{"description": "Diagram: " + item["png_bytes"].decode(), "image_type": "diagram", "concepts": []} for item in batch]


class Container:
    """modal.Function as remote.py uses it; pickles what crosses, as Modal does."""

    def __init__(self, stream=True, lost_after_pages=None):
        self.stream, self.lost_after_pages = stream, lost_after_pages

    def remote_gen(self, job):
        job = pickle.loads(pickle.dumps(job))
        items = remote.index_work(job)
        if not self.stream:  # all of it first, then hand it over
            items = list(items)
        arrived = 0
        for item in items:
            if self.lost_after_pages is not None and arrived >= self.lost_after_pages:
                raise ConnectionError("container lost")
            arrived += item[0] == "pages"
            yield pickle.loads(pickle.dumps(item))


class Containers(unittest.TestCase):
    setUp = fixture.LaunchIntegrity.setUp

    def ingest(self, d, container=None, classify_fn=classify, describe_fn=describe, figures=2):
        orch = SimpleNamespace(content=ContentStore(Path(d) / "oma.db"), images=ImageStore(Path(d) / "oma.db"),
                               concept=ConceptStore(Path(d) / "oma.db"))
        pipeline = IngestionPipeline(orch.concept, orch.content, orch.images, Path(d) / "images",
                                     classify_workers=2, vision_workers=1)
        pipeline.VISION_BATCH_SIZE = 1
        saved = []
        for page in range(1, figures + 1):
            path = Path(d) / f"fig{page}.webp"
            path.write_bytes(b"figure %d" % page)
            saved.append({"page_number": page, "file_path": str(path), "width": 100, "height": 100})
        local_calls = []

        def counted(batch):
            local_calls.append(threading.current_thread().name)
            return classify_fn(batch)
        with patch("coast_content_oma.ingestion.extract_pages", return_value=pages()), \
                patch.object(pipeline, "_save_images_to_disk", return_value=saved), \
                patch.object(IngestionPipeline, "_classify_batch", lambda self, batch: counted(batch)), \
                patch("coast_content_oma.llm.describe_images_batch", side_effect=describe_fn), \
                patch.object(remote, "enabled", return_value=container is not None), \
                patch.object(remote, "_function", side_effect=lambda name: container if not isinstance(container, Exception) else (_ for _ in ()).throw(container)):
            stats = pipeline.ingest_folder("u1__c", [Path(d) / "a.pdf"], source_ids={"a.pdf": "a"}, defer_concepts=True)
        return orch, stats, local_calls

    def snapshot(self, orch):
        content = sorted((i.store_specific.get("page_number") if i.store_specific else None, i.content)
                         for i in orch.content.all("u1__c"))
        images = sorted(((i.store_specific or {}).get("file_path", "").rsplit("/", 1)[-1], i.content)
                        for i in orch.images.all("u1__c"))
        with connect_db(orch.content.db_path) as conn:
            ready = conn.execute("select page, text_ready from source_page_progress where namespace='u1__c' order by page").fetchall()
        return content, images, [tuple(r) for r in ready]

    def test_a_container_indexes_exactly_as_the_server_does(self):
        with tempfile.TemporaryDirectory() as here, tempfile.TemporaryDirectory() as there:
            local, local_stats, _ = self.ingest(here)
            boxed, boxed_stats, calls = self.ingest(there, Container())
            self.assertEqual(self.snapshot(local), self.snapshot(boxed))
            for field in ("content_items", "image_items", "figures_pending", "errors"):
                self.assertEqual(getattr(local_stats, field), getattr(boxed_stats, field), field)
            self.assertEqual(boxed_stats.content_items, PAGES)
            self.assertTrue(all(name != "MainThread" for name in calls))

    def test_the_first_section_opens_while_the_container_still_works(self):
        release, later = threading.Event(), threading.Event()

        def slow(batch):
            if batch[0]["page_number"] > 8:
                later.set()
                release.wait(5)
            return classify(batch)
        with tempfile.TemporaryDirectory() as d:
            result = []
            worker = threading.Thread(target=lambda: result.append(self.ingest(d, Container(), classify_fn=slow)))
            worker.start()
            try:
                self.assertTrue(later.wait(5))
                orch = SimpleNamespace(content=ContentStore(Path(d) / "oma.db"), images=ImageStore(Path(d) / "oma.db"))
                status = {"ready": False}
                for _ in range(200):
                    status = progressive.section_status(orch, "u1__c", {"source_refs": [{"source_id": "a", "pages": list(range(1, 9))}]})
                    if status["ready"]:
                        break
                    threading.Event().wait(0.02)
                self.assertTrue(status["ready"])
                self.assertTrue(worker.is_alive())
            finally:
                release.set()
                worker.join(10)
            self.assertEqual(result[0][1].content_items, PAGES)

    def test_no_container_reachable_means_the_server_does_it_all(self):
        with tempfile.TemporaryDirectory() as here, tempfile.TemporaryDirectory() as there:
            local, _, _ = self.ingest(here)
            fallen, stats, _ = self.ingest(there, ConnectionError("modal unreachable"))
            self.assertEqual(self.snapshot(local), self.snapshot(fallen))
            self.assertEqual(stats.errors, [])

    def test_after_a_failure_the_server_works_alone_for_a_while(self):
        """So memory admission plans for work done here, not in containers, during an outage."""
        env = {"INDEX_IN_CONTAINERS": "on", "MODAL_TOKEN_ID": "x", "MODAL_TOKEN_SECRET": "y"}
        with patch.dict("os.environ", env), patch("file_store.enabled", return_value=True), \
                patch.object(remote, "_down_until", 0.0):
            self.assertTrue(remote.enabled())
            remote._note_down(ConnectionError("modal unreachable"))
            self.assertFalse(remote.enabled())
            import memory_budget
            self.assertEqual(memory_budget.index_file_mb(), memory_budget.INDEX_FILE_MB)

    def test_a_container_lost_midway_leaves_the_server_only_the_rest(self):
        seen = []

        def record(batch):
            seen.append([p["page_number"] for p in batch])
            return classify(batch)
        with tempfile.TemporaryDirectory() as here, tempfile.TemporaryDirectory() as there:
            local, _, _ = self.ingest(here)
            # every batch is classified in the container (unstreamed), but only the first one arrives
            lost, stats, _ = self.ingest(there, Container(stream=False, lost_after_pages=1), classify_fn=record)
            self.assertEqual(self.snapshot(local), self.snapshot(lost))
            container_batches, server_batches = PAGES // 8, len(seen) - PAGES // 8
            self.assertEqual(server_batches, container_batches - 1)  # the batch that arrived is not redone
            self.assertEqual(stats.content_items, PAGES)

    def test_figures_the_container_could_not_describe_go_to_the_background(self):
        described = []

        def failing(batch):
            described.append(threading.current_thread().name)
            if b"2" in batch[0]["png_bytes"]:
                raise RuntimeError("vision unavailable")
            return describe(batch)
        with tempfile.TemporaryDirectory() as d:
            _, stats, _ = self.ingest(d, Container(), describe_fn=failing)
        self.assertEqual(len(described), 2)  # tried once, in the container; never again on the server
        self.assertEqual(stats.figures_pending, 1)

    def test_figures_only_in_r2_are_still_described_in_the_background(self):
        """Read in a container, a file's figures are in R2, not on the server's disk."""
        with tempfile.TemporaryDirectory() as d:
            orch, stats, _ = self.ingest(d, Container(), figures=30)  # 24 now, 6 for the background
            self.assertEqual(stats.figures_pending, 6)
            in_r2 = {}
            for fig in Path(d).glob("fig*.webp"):
                in_r2[str(fig)] = fig.read_bytes()
                fig.unlink()

            def fetch(path):
                Path(path).write_bytes(in_r2[str(path)])
                return Path(path)
            pipeline = IngestionPipeline(orch.concept, orch.content, orch.images, Path(d) / "images", vision_workers=1)
            with patch("file_store.available", side_effect=lambda path: str(path) in in_r2), \
                    patch("file_store.local", side_effect=fetch), \
                    patch("coast_content_oma.llm.describe_images_batch", side_effect=describe):
                pipeline.describe_pending_images("u1__c")
            pending = [i for i in orch.images.all("u1__c") if not (i.content or "").startswith("Diagram:")]
            self.assertEqual(pending, [])

    def test_the_containers_ai_usage_is_recorded_on_the_server(self):
        recorded = []

        def billed(batch):
            ai_usage.record("openai", {"input": 100, "output": 10}, model="gpt-5.6-luna", latency_ms=5)
            return classify(batch)
        with patch.object(ai_usage, "record", side_effect=lambda *a, **k: recorded.append((a, k))):
            with tempfile.TemporaryDirectory() as d:
                self.ingest(d, Container(stream=False), classify_fn=billed)
            self.assertEqual(len(recorded), PAGES // 8)
            self.assertEqual(recorded[0][0][:2], ("openai", {"input": 100, "output": 10}))
            self.assertIsNot(ai_usage.record, None)

    def test_reading_an_upload_in_a_container_gives_the_same_pages(self):
        import asyncio
        import fitz
        from coast_content_oma.extraction import extract_and_cache
        with tempfile.TemporaryDirectory() as d:
            pdf = Path(d) / "lecture.pdf"
            doc = fitz.open()
            for n in range(3):
                doc.new_page().insert_text((72, 72), f"Lecture page {n + 1}: entropy and enthalpy")
            doc.save(pdf)
            here = extract_and_cache(str(pdf))
            import shutil
            shutil.rmtree(str(pdf) + ".pages", ignore_errors=True)

            class Reader:
                class remote:
                    @staticmethod
                    async def aio(path, context):
                        return pickle.loads(pickle.dumps(remote.read_upload_work(path, context)))
            with patch.object(remote, "_function", return_value=Reader):
                there = asyncio.run(remote.read_upload(str(pdf)))
            self.assertEqual([p["text"] for p in here], [p["text"] for p in there])
            self.assertTrue((Path(str(pdf) + ".pages") / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=1)
