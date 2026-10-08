#!/usr/bin/env python3
"""Deleting an account removes everything of that student's and nothing of anyone else's (user 1's
deletion leaves user 12, whose namespaces share the "u1" start, untouched); downloading your data
gives it all back without secrets. Isolated databases and files."""
import json, os, sys, tempfile, unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TMP = tempfile.TemporaryDirectory(prefix='coast-delete-')
for key, name in {'DATABASE_PATH': 'app.db', 'OMA_DB_PATH': 'oma.db', 'CHROMA_PATH': 'chroma',
                  'GENERATED_DIR': 'generated', 'FOLDER_UPLOADS_DIR': 'sources', 'OMA_IMAGE_DIR': 'images'}.items():
    os.environ[key] = str(Path(TMP.name) / name)
for key in ('OPENAI_API_KEY', 'GEMINI_API_KEY', 'ANTHROPIC_API_KEY', 'RENDER', 'FILE_STORE'):
    os.environ.pop(key, None)
import dotenv
dotenv.load_dotenv = lambda *a, **k: None
from fastapi.testclient import TestClient
import server
from auth import create_access_token, hash_password
from database import (AiUsage, Base, BetaCode, ChatMessage, FolderSource, LessonNotes, SessionLocal, StudyFolder,
                      User, engine)
from coast_content_oma.stores.base import MemoryItem
from coast_content_oma.stores.content import ContentStore
from coast_content_oma.stores.image import ImageStore
import oma_provider

client = TestClient(server.app)


def item(store, ns, n, **extra):
    return MemoryItem(id=f"{store[:3]}_{ns}_{n}", namespace=ns, store=store, content=f"{ns} item {n}",
                      source_doc_id="doc_s", store_specific=extra)


class AccountDeletion(unittest.TestCase):
    def setUp(self):
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
        from security import limiter
        limiter._hits.clear()
        Path(oma_provider.OMA_DB_PATH).unlink(missing_ok=True)
        self.content = ContentStore(oma_provider.OMA_DB_PATH)
        self.images = ImageStore(oma_provider.OMA_DB_PATH)
        root = Path(TMP.name)
        self.files = {}
        with SessionLocal() as db:
            for uid in (1, 12):
                db.add(User(id=uid, email=f"u{uid}@example.com", name=f"User {uid}", password_hash=hash_password('secret123')))
                pdf = root / f"src_{uid}.pdf"
                pdf.write_bytes(b"%PDF fixture")
                pages = Path(str(pdf) + ".pages")
                pages.mkdir(exist_ok=True)
                (pages / "manifest.json").write_text("{}")
                figure = pages / "p1_i0.webp"
                figure.write_bytes(b"figure")
                self.files[uid] = (pdf, pages, figure)
                db.add(StudyFolder(user_id=uid, name="Physics"))
                db.add(FolderSource(user_id=uid, folder_name="Physics", source_id=f"src_{uid}", title="Lecture",
                                    filename="l.pdf", source_type="pdf", page_count=1, raw_text="secret lecture text",
                                    file_path=str(pdf)))
                db.add(ChatMessage(user_id=uid, conversation_id="c", role="user", content=f"hello from {uid}",
                                   context_type="lesson", context_id="Physics"))
                db.add(LessonNotes(user_id=uid, folder_name="Physics", content_html=f"<p>notes {uid}</p>"))
                db.add(AiUsage(user_id=uid, feature="lesson", provider="openai", model="m", input_tokens=10,
                               created_at=datetime.now(timezone.utc)))
                self.content.write_items_bulk([item("content", f"u{uid}__Physics", 1)], embed=False)
                self.images.write_items_bulk([item("image", f"u{uid}__Physics", 1, file_path=str(figure))], embed=False)
            db.add(BetaCode(code="COASTABCD1234", used_by_user_id=1, used_email="u1@example.com", revoked=False,
                            used_at=datetime.now(timezone.utc)))
            db.commit()

    def token(self, uid):
        return {'Authorization': 'Bearer ' + create_access_token(uid, f"u{uid}@example.com")}

    def test_deleting_an_account_removes_everything_of_theirs_and_nothing_else(self):
        wrong = client.post('/api/account/delete', headers=self.token(1), json={'confirm_email': 'someone@else.com'})
        self.assertEqual(wrong.status_code, 400)
        r = client.post('/api/account/delete', headers=self.token(1), json={'confirm_email': 'U1@example.com '})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()['deleted'])
        with SessionLocal() as db:
            self.assertIsNone(db.get(User, 1))
            for model in (FolderSource, ChatMessage, LessonNotes, StudyFolder):
                self.assertEqual(db.query(model).filter_by(user_id=1).count(), 0, model.__name__)
                self.assertEqual(db.query(model).filter_by(user_id=12).count(), 1, model.__name__)
            code = db.get(BetaCode, "COASTABCD1234")
            self.assertEqual((code.used_email, code.used_by_user_id), ('', None))  # still used, no longer theirs
            self.assertIsNotNone(code.used_at)  # so it can't open a second account
            usage = db.query(AiUsage).all()
            self.assertEqual(sorted(u.user_id for u in usage if u.user_id), [12])  # totals kept, the student gone
            self.assertEqual(len(usage), 2)
        self.assertEqual([i.namespace for i in self.content.all("u1__Physics")], [])
        self.assertEqual([i.namespace for i in self.content.all("u12__Physics")], ["u12__Physics"])
        self.assertEqual(len(self.images.all("u12__Physics")), 1)
        pdf, pages, figure = self.files[1]
        self.assertFalse(pdf.exists() or pages.exists() or figure.exists())
        self.assertTrue(all(p.exists() for p in self.files[12]))
        self.assertEqual(client.get('/api/auth/me', headers=self.token(1)).status_code, 401)  # the session ends

    def test_downloading_your_data_gives_it_back_without_secrets(self):
        r = client.get('/api/account/export', headers=self.token(1))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn('attachment', r.headers['content-disposition'])
        data = r.json()
        self.assertEqual(data['users'][0]['email'], 'u1@example.com')
        self.assertNotIn('password_hash', data['users'][0])
        self.assertNotIn('raw_text', data['folder_sources'][0])
        self.assertEqual([m['content'] for m in data['chat_messages']], ['hello from 1'])
        self.assertEqual(data['lesson_notes'][0]['content_html'], '<p>notes 1</p>')
        self.assertNotIn('ai_usage', data)
        self.assertNotIn('hello from 12', json.dumps(data))


if __name__ == '__main__':
    unittest.main(verbosity=1)
