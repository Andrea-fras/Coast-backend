#!/usr/bin/env python3
"""Account isolation and persistence, across the whole API.

Two students both have a course called "Physics". Everything the first one owns (a source file,
a conversation, notes, a notebook, an outline, Ask-Sources history, student memory) carries a
marker. Then:
  - every GET route is called as the second student with the first one's ids and names, and the
    marker must never come back (nor the first student's files or page images);
  - every route that edits or deletes is called the same way, and the first student's data must be
    untouched afterwards;
  - a fresh process on the same disk (a restart) must serve both students everything they had.
Offline: temporary databases and files, no AI keys.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEMP = tempfile.TemporaryDirectory(prefix="coast-isolation-")
T = Path(TEMP.name)
ENV = {"DATABASE_PATH": str(T / "coast.db"), "OMA_DB_PATH": str(T / "oma_data" / "oma.db"),
       "OMA_IMAGE_DIR": str(T / "oma_data" / "images"), "FOLDER_UPLOADS_DIR": str(T / "folder_uploads"),
       "GENERATED_DIR": str(T / "generated"), "CHROMA_PATH": str(T / "chroma"), "MEDIA_DIR": str(T / "media"),
       "JWT_SECRET": "isolation-test-secret", "STUDENT_OMA_ENABLED": "true", "BACKUPS": "off"}
os.environ.update(ENV)
for key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "RENDER"):
    os.environ.pop(key, None)
import dotenv  # noqa: E402
dotenv.load_dotenv = lambda *a, **k: None  # never load developer credentials in offline tests

import fitz  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import auth  # noqa: E402
import server  # noqa: E402
from database import (ChatMessage, CourseOutline, FolderSource, LessonNotes, SavedNotebook,  # noqa: E402
                      SessionLocal, SourceUpload, StudyFolder, User, init_db)

A, B = 101, 102
MARK = {A: "ALPHA7731MARK", B: "BRAVO4419MARK"}
FOLDER = "Physics"
IDS = {}  # the first student's identifiers, filled in by seed()


def pdf(path: Path, text: str) -> None:
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), text)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))


def seed() -> None:
    init_db()
    uploads = Path(ENV["FOLDER_UPLOADS_DIR"])
    with SessionLocal() as db:
        for uid in (A, B):
            m = MARK[uid]
            db.add(User(id=uid, email=f"student{uid}@example.invalid", name=f"Student {uid}", email_verified=True))
            db.add(StudyFolder(user_id=uid, name=FOLDER, kind="lesson"))
            sid = f"src_{m.lower()[:10]}"
            pdf(uploads / f"{sid}.pdf", f"Lecture notes {m}")
            db.add(FolderSource(user_id=uid, folder_name=FOLDER, source_id=sid, title=f"Notes {m}",
                                filename=f"{m}.pdf", source_type="pdf", page_count=1, raw_text=f"Lecture notes {m}",
                                file_path=str(uploads / f"{sid}.pdf"), oma_ingest_status="READY_FOR_ROADMAP"))
            db.add(SourceUpload(user_id=uid, folder_name=FOLDER, upload_id=f"up-{m}", filename=f"{m}.pdf",
                                size_bytes=10, status="complete", source_id=sid, expires_at=time.time() + 3600))
            conv = f"conv-{m}"
            db.add(ChatMessage(user_id=uid, conversation_id=conv, role="user", content=f"my question {m}",
                               context_type="lesson", context_id=FOLDER, section_index=0))
            db.add(ChatMessage(user_id=uid, conversation_id=conv, role="pedro", content=f"Pedro's answer {m}",
                               context_type="lesson", context_id=FOLDER, section_index=0))
            db.add(LessonNotes(user_id=uid, folder_name=FOLDER, content_html=f"<p>my notes {m}</p>"))
            nb = SavedNotebook(user_id=uid, notebook_id=f"nb-{m}", title=f"Notebook {m}", course="Physics",
                               notebook_json=json.dumps({"title": f"Notebook {m}", "cells": []}), folder=FOLDER)
            db.add(nb)
            db.add(CourseOutline(user_id=uid, folder_name=FOLDER, current_section=0, total_sections=1,
                                 outline_json=json.dumps([{"title": f"Section {m}", "key_topics": [m]}])))
            db.flush()
            if uid == A:
                IDS.update(source_id=sid, conversation_id=conv, notebook_id=nb.id, saved_id=nb.id,
                           upload_id=f"up-{m}", file_bytes=(uploads / f"{sid}.pdf").read_bytes())
        db.commit()
    import oma_provider
    for uid in (A, B):
        oma_provider.apply_capture_tags(uid, FOLDER, [{"trait_type": "goal", "description": f"pass the exam {MARK[uid]}"}],
                                        [f"the bridge example {MARK[uid]}"])
    oma_provider.flush_student_writes()


def token(uid: int) -> dict:
    return {"Authorization": "Bearer " + auth.create_access_token(uid, f"student{uid}@example.invalid")}


def fill(path: str) -> str:
    values = {"folder_name": FOLDER, "source_id": IDS["source_id"], "page_number": "1", "section_index": "0",
              "notebook_id": str(IDS["notebook_id"]), "saved_id": str(IDS["saved_id"]), "upload_id": IDS["upload_id"],
              "conversation_id": IDS["conversation_id"], "image_id": "1", "item_id": "1", "chest_id": "1",
              "session_id": "1", "paper_id": "1", "code": "X"}
    for name, value in values.items():
        path = path.replace("{" + name + "}", value)
    return path


QUERY = {"conversation_id": "conv-ALPHA7731MARK", "folder": FOLDER, "folder_name": FOLDER, "context_id": FOLDER,
         "source_id": "src_alpha7731m", "notebook_id": "nb-ALPHA7731MARK"}
# Routes that only create or sign in (no id of anyone else's to act on), or that stream an AI reply.
SKIP_WRITES = ("/api/chat", "/api/auth/", "/api/folders/{folder_name}/ask-sources")


class Isolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed()
        cls.client = TestClient(server.app)  # no startup hooks: no background jobs during the sweep

    def routes(self, methods):
        for r in server.app.routes:
            if hasattr(r, "methods") and r.path.startswith("/api") and r.methods & methods:
                if methods == {"GET"} or not r.path.startswith(SKIP_WRITES):
                    yield r.path, sorted(r.methods & methods)[0]

    def test_no_route_shows_one_student_anothers_data(self):
        leaks, seen_by_owner = [], 0
        for path, method in self.routes({"GET"}):
            url = fill(path)
            as_owner = self.client.get(url, params=QUERY, headers=token(A))
            seen_by_owner += MARK[A] in as_owner.text
            r = self.client.get(url, params=QUERY, headers=token(B))
            body_type = r.headers.get("content-type", "")
            if MARK[A] in r.text:
                leaks.append(f"GET {url}: the other student's data in the response")
            elif r.status_code == 200 and not body_type.startswith(("application/json", "text/")) and (
                    r.content == IDS["file_bytes"] or IDS["source_id"] in url):
                leaks.append(f"GET {url}: the other student's file or page image ({body_type})")
        self.assertEqual(leaks, [])
        self.assertGreaterEqual(seen_by_owner, 6, "the sweep should reach the owner's data through several routes")

    def test_no_route_lets_one_student_change_anothers_data(self):
        before = self.owner_state()
        for path, method in self.routes({"PUT", "PATCH", "DELETE", "POST"}):
            if method == "POST" and "{" not in path:
                continue  # creating your own things; the routes that act on an id are the risk
            self.client.request(method, fill(path), params=QUERY, headers=token(B), json={"html": "x", "content_html": "x",
                                "new_name": "Stolen", "folder": "Stolen", "revision": None})
        self.assertEqual(self.owner_state(), before)

    def owner_state(self) -> dict:
        with SessionLocal() as db:
            return {
                "folders": sorted(f.name for f in db.query(StudyFolder).filter_by(user_id=A)),
                "sources": [(s.source_id, s.folder_name) for s in db.query(FolderSource).filter_by(user_id=A)],
                "file": (Path(ENV["FOLDER_UPLOADS_DIR"]) / f"{IDS['source_id']}.pdf").exists(),
                "messages": [m.content for m in db.query(ChatMessage).filter_by(user_id=A).order_by(ChatMessage.id)],
                "notes": [n.content_html for n in db.query(LessonNotes).filter_by(user_id=A)],
                "notebooks": [(n.title, n.folder, n.deleted_at) for n in db.query(SavedNotebook).filter_by(user_id=A)],
                "outline": [(o.folder_name, o.current_section) for o in db.query(CourseOutline).filter_by(user_id=A)],
                "uploads": [u.upload_id for u in db.query(SourceUpload).filter_by(user_id=A)],
            }

    def test_everything_survives_a_restart(self):
        # A new interpreter on the same disk is what a redeploy or crash leaves behind.
        check = r'''
import json, os, sys
sys.path.insert(0, os.environ["ROOT"])
import dotenv; dotenv.load_dotenv = lambda *a, **k: None
from fastapi.testclient import TestClient
import auth, server
c = TestClient(server.app)
out = {}
for uid in (101, 102):
    h = {"Authorization": "Bearer " + auth.create_access_token(uid, f"student{uid}@example.invalid")}
    conv = c.get("/api/chat/conversations", headers=h).text
    cid = "conv-" + ("ALPHA7731MARK" if uid == 101 else "BRAVO4419MARK")
    history = c.get("/api/chat/history", params={"conversation_id": cid}, headers=h).text
    notes = c.get("/api/folders/Physics/lesson-notes", headers=h).text
    sources = c.get("/api/folders/Physics/sources", headers=h).json()
    items = sources if isinstance(sources, list) else sources.get("sources", [])
    sid = next(x["source_id"] for x in items if x.get("source_id"))  # notebooks are listed too
    file_ok = c.get(f"/api/folders/Physics/sources/{sid}/file", headers=h).status_code == 200
    profile = c.get("/api/oma/student/Physics/profile", headers=h).text
    out[uid] = {"history": history, "notes": notes, "file": file_ok, "profile": profile, "conversations": conv}
print(json.dumps(out))
'''
        env = {**os.environ, **ENV, "ROOT": str(ROOT)}
        r = subprocess.run([sys.executable, "-c", check], env=env, capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        out = {int(k): v for k, v in json.loads(r.stdout.strip().splitlines()[-1]).items()}
        for uid, other in ((A, B), (B, A)):
            got = out[uid]
            self.assertIn(f"Pedro's answer {MARK[uid]}", got["history"])
            self.assertIn(f"my notes {MARK[uid]}", got["notes"])
            self.assertTrue(got["file"])
            self.assertIn(MARK[uid], got["profile"], "student memory should survive a restart")
            self.assertNotIn(MARK[other], json.dumps(got))


if __name__ == "__main__":
    unittest.main(verbosity=1)
