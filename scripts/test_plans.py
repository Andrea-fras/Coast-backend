#!/usr/bin/env python3
"""Monthly allowances: 5 lessons and workshops together, 20 files and 600 messages to Pedro;
double for founding students, none for admins; reset on the 1st. Isolated database; Pedro and
file reading are stand-ins."""
import os, sys, tempfile, unittest, uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TMP = tempfile.TemporaryDirectory(prefix='coast-plans-')
for key, name in {'DATABASE_PATH': 'app.db', 'OMA_DB_PATH': 'oma.db', 'CHROMA_PATH': 'chroma',
                  'GENERATED_DIR': 'generated', 'FOLDER_UPLOADS_DIR': 'sources', 'OMA_IMAGE_DIR': 'images'}.items():
    os.environ[key] = str(Path(TMP.name) / name)
for key in ('OPENAI_API_KEY', 'GEMINI_API_KEY', 'ANTHROPIC_API_KEY', 'RENDER', 'FILE_STORE'):
    os.environ.pop(key, None)
import dotenv
dotenv.load_dotenv = lambda *a, **k: None
from fastapi.testclient import TestClient
import server
import plans
import upload_lifecycle
from auth import create_access_token, hash_password
from database import Base, ChatMessage, CourseOutline, FolderSource, SessionLocal, StudyFolder, UsageEvent, User, engine

client = TestClient(server.app)
ADMIN = 'andreaf.fraschetti@gmail.com'


def fake_stream(**kwargs):
    yield 'Hi', None
    yield None, {'conversation_id': 'c1'}


def broken_stream(**kwargs):
    raise RuntimeError('provider down')
    yield


class Plans(unittest.TestCase):
    def setUp(self):
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
        from security import limiter
        limiter._hits.clear()
        with SessionLocal() as db:
            db.add(User(id=1, email='sam@example.com', name='Sam', password_hash=hash_password('secret123')))
            db.add(User(id=2, email='fay@example.com', name='Fay', password_hash='x', plan='founder'))
            db.add(User(id=3, email=ADMIN, name='Andrea', password_hash='x', email_verified=True))
            db.commit()

    def h(self, uid):
        email = {1: 'sam@example.com', 2: 'fay@example.com', 3: ADMIN}[uid]
        return {'Authorization': 'Bearer ' + create_access_token(uid, email)}

    def make(self, uid, name, kind='lesson'):
        return client.post('/api/notebooks/folders', headers=self.h(uid), json={'name': name, 'kind': kind})

    def spend(self, uid, kind, n, when=None):
        with SessionLocal() as db:
            for _ in range(n):
                db.add(UsageEvent(user_id=uid, kind=kind, created_at=when or datetime.now(timezone.utc)))
            db.commit()

    def plan(self, uid):
        r = client.get('/api/plan', headers=self.h(uid))
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    # Lessons and workshops

    def test_five_lessons_and_workshops_together_then_the_sixth_waits_for_next_month(self):
        for i, kind in enumerate(['lesson'] * 4 + ['workshop']):
            self.assertEqual(self.make(1, f'Course {i}', kind).status_code, 200)
        r = self.make(1, 'One too many')
        self.assertEqual(r.status_code, 429)
        self.assertEqual(r.headers[plans.LIMIT_HEADER], 'lessons')
        self.assertIn("You've made 5 lessons and workshops this month", r.json()['detail'])
        self.assertIn(f"You can make more from {plans.next_reset().day} ", r.json()['detail'])
        # Opening one they already have, or a premade one, is free.
        self.assertEqual(self.make(1, 'Course 0').status_code, 200)
        self.assertEqual(self.make(1, 'Build a Rocket', 'workshop').status_code, 200)
        self.assertEqual(self.plan(1)['usage']['lessons'], {'used': 5, 'limit': 5})

    def test_a_lesson_deleted_before_its_roadmap_gives_its_slot_back_a_studied_one_does_not(self):
        self.make(1, 'Typo'); self.make(1, 'Studied')
        with SessionLocal() as db:
            db.add(CourseOutline(user_id=1, folder_name='Studied', outline_json='[]'))
            db.commit()
        client.delete('/api/notebooks/folders/Typo', headers=self.h(1))
        client.delete('/api/notebooks/folders/Studied', headers=self.h(1))
        self.assertEqual(self.plan(1)['usage']['lessons']['used'], 1)

    def test_a_renamed_lesson_keeps_its_count_and_can_still_give_it_back(self):
        self.make(1, 'Old name')
        r = client.put('/api/notebooks/folders/Old name/rename', headers=self.h(1), json={'new_name': 'New name'})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.plan(1)['usage']['lessons']['used'], 1)
        client.delete('/api/notebooks/folders/New name', headers=self.h(1))
        self.assertEqual(self.plan(1)['usage']['lessons']['used'], 0)

    def test_last_months_use_does_not_count(self):
        self.spend(1, 'lessons', 5, when=plans.month_start() - timedelta(seconds=1))
        self.assertEqual(self.make(1, 'Fresh month').status_code, 200)

    def test_founders_get_double_and_admins_have_no_limit(self):
        self.spend(2, 'lessons', 9)
        self.assertEqual(self.make(2, 'Tenth').status_code, 200)
        self.assertEqual(self.make(2, 'Eleventh').status_code, 429)
        self.spend(3, 'lessons', 50)
        self.assertEqual(self.make(3, 'Admin course').status_code, 200)
        self.assertEqual(self.plan(3)['plan'], 'unlimited')
        self.assertTrue(self.plan(3)['founder'])  # the admin account wears the founder badge
        self.assertTrue(client.get('/api/auth/me', headers=self.h(3)).json()['founder'])
        self.assertFalse(client.get('/api/auth/me', headers=self.h(1)).json()['founder'])
        self.assertIsNone(self.plan(3)['usage']['lessons']['limit'])

    # Files

    def reserve(self, uid, folder, n):
        files = [{'upload_id': uuid.uuid4().hex, 'filename': f'l{i}.pdf', 'size_bytes': 100} for i in range(n)]
        return files, client.post(f'/api/folders/{folder}/uploads', headers=self.h(uid), json={'files': files})

    def finish(self, uid, folder, file):
        claim, _ = upload_lifecycle.begin(uid, folder, file['upload_id'], file['filename'], file['size_bytes'])
        source_id = f"src_{uuid.uuid4().hex[:10]}"
        with SessionLocal() as db:
            upload_lifecycle.finish(db, uid, folder, file['upload_id'], claim, source_id)
            db.add(FolderSource(user_id=uid, folder_name=folder, source_id=source_id, title='t', filename=file['filename'],
                                source_type='pdf', raw_text='x'))
            db.commit()

    def test_twenty_files_a_month_counting_ones_still_on_their_way(self):
        self.spend(1, 'uploads', 15)
        on_the_way, r = self.reserve(1, 'Physics', 3)
        self.assertEqual(r.status_code, 200, r.text)
        _, r = self.reserve(1, 'Maths', 3)  # 15 uploaded + 3 coming + 3 = 21
        self.assertEqual(r.status_code, 429)
        self.assertEqual(r.headers[plans.LIMIT_HEADER], 'uploads')
        self.assertEqual(r.json()['detail'], 'You can upload 2 more files this month. Choose fewer files.')
        # Asking again for the files already on their way doesn't count them twice.
        self.assertEqual(client.post('/api/folders/Physics/uploads', headers=self.h(1),
                                     json={'files': on_the_way}).status_code, 200)
        for f in on_the_way:
            self.finish(1, 'Physics', f)
        self.assertEqual(self.plan(1)['usage']['uploads']['used'], 18)
        _, r = self.reserve(1, 'Maths', 2)
        self.assertEqual(r.status_code, 200, r.text)
        _, r = self.reserve(1, 'Maths', 1)
        self.assertEqual(r.status_code, 429)
        self.assertIn("You've uploaded 20 files this month", r.json()['detail'])

    def test_a_file_that_never_arrives_does_not_count(self):
        files, r = self.reserve(1, 'Physics', 2)
        upload_lifecycle.cancel(1, 'Physics', files[0]['upload_id'])
        self.finish(1, 'Physics', files[1])
        self.assertEqual(self.plan(1)['usage']['uploads']['used'], 1)

    # Messages

    def chat(self, uid, context='global', stream=fake_stream):
        with patch('tutor.send_message_stream', side_effect=stream):
            r = client.post('/api/chat/stream', headers=self.h(uid), json={'message': 'hello', 'context_type': context})
            return r, r.text

    def test_six_hundred_messages_then_pedro_waits_for_next_month(self):
        self.spend(1, 'messages', 599)
        r, body = self.chat(1)
        self.assertEqual(r.status_code, 200)
        self.assertIn('"chat_messages_remaining": 0', body)
        r, _ = self.chat(1)
        self.assertEqual(r.status_code, 429)
        self.assertEqual(r.headers[plans.LIMIT_HEADER], 'messages')
        self.assertIn("You've sent Pedro 600 messages this month", r.json()['detail'])
        # The welcome chat never counts.
        self.assertEqual(self.chat(1, context='onboarding')[0].status_code, 200)

    def test_the_automatic_section_opener_is_not_a_message(self):
        with patch('coast_content_oma.progressive.assert_chat_ready'), \
             patch('tutor.send_message_stream', side_effect=fake_stream):
            for text in ('I\'m ready to learn about "Queues". Please teach me this section.', 'Why FIFO?'):
                r = client.post('/api/chat/stream', headers=self.h(1), json={
                    'message': text, 'context_type': 'lesson', 'context_id': 'Physics', 'section_index': 0})
                self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.plan(1)['usage']['messages']['used'], 1)

    def test_a_reply_that_fails_does_not_use_a_message(self):
        r, body = self.chat(1, stream=broken_stream)
        self.assertIn('provider down', body)
        self.assertEqual(self.plan(1)['usage']['messages']['used'], 0)
        self.chat(1)
        self.assertEqual(self.plan(1)['usage']['messages']['used'], 1)

    def test_a_retried_source_question_is_one_message(self):
        from database import SourceChatTurn
        with SessionLocal() as db:
            db.add(StudyFolder(user_id=1, name='Physics'))
            db.add(FolderSource(user_id=1, folder_name='Physics', source_id='src_p', title='t', filename='f.pdf',
                                source_type='pdf', raw_text='x'))
            db.commit()
        rid = str(uuid.uuid4())
        with patch('server._source_workspace'), patch('source_chat.stream_answer', side_effect=lambda *a: iter([{'done': True}])):
            for _ in range(2):
                r = client.post('/api/folders/Physics/ask-sources', headers=self.h(1), json={'message': 'Why?', 'request_id': rid})
                self.assertEqual(r.status_code, 200, r.text)
                with SessionLocal() as db:  # the answer broke off: the student retries
                    db.query(SourceChatTurn).update({'status': 'failed'})
                    db.commit()
        self.assertEqual(self.plan(1)['usage']['messages']['used'], 1)

    # The plan panel, the founder pass, admin

    def test_the_plan_panel_shows_the_month_and_the_founder_offer(self):
        self.spend(1, 'messages', 12)
        p = self.plan(1)
        self.assertEqual(p['plan'], 'beta')
        self.assertEqual(p['usage'], {'lessons': {'used': 0, 'limit': 5}, 'uploads': {'used': 0, 'limit': 20},
                                      'messages': {'used': 12, 'limit': 600}})
        self.assertEqual(p['founder_offer']['price_eur'], 14.99)
        self.assertEqual(p['founder_offer']['limits'], {'lessons': 10, 'uploads': 40, 'messages': 1200})
        self.assertTrue(p['period']['resets_at'].startswith(plans.next_reset().strftime('%Y-%m-01')))
        f = self.plan(2)
        self.assertTrue(f['founder'])
        self.assertEqual(f['usage']['messages']['limit'], 1200)
        me = client.get('/api/auth/me', headers=self.h(2)).json()
        self.assertEqual((me.get('user') or me)['plan'], 'founder')

    def test_wanting_the_pass_is_noted_and_an_admin_can_grant_it(self):
        r = client.post('/api/plan/founder-interest', headers=self.h(1))
        self.assertTrue(r.json()['founder_offer']['interested'])
        self.assertEqual(client.get('/api/admin/plans', headers=self.h(1)).status_code, 403)
        listed = client.get('/api/admin/plans', headers=self.h(3)).json()['students']
        self.assertEqual(sorted(s['email'] for s in listed), ['fay@example.com', 'sam@example.com'])
        r = client.post('/api/admin/plans', headers=self.h(3), json={'email': 'Sam@Example.com', 'plan': 'founder'})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.plan(1)['usage']['uploads']['limit'], 40)

    def test_this_months_earlier_use_counts_when_allowances_start(self):
        now = datetime.now(timezone.utc)
        with SessionLocal() as db:
            db.add(ChatMessage(user_id=1, conversation_id='c', role='user', content='q', context_type='lesson', created_at=now))
            db.add(ChatMessage(user_id=1, conversation_id='c', role='assistant', content='a', context_type='lesson', created_at=now))
            db.add(ChatMessage(user_id=1, conversation_id='o', role='user', content='hi', context_type='onboarding', created_at=now))
            db.add(ChatMessage(user_id=1, conversation_id='c', role='user', content='I\'m ready to learn about "Queues".',
                               context_type='lesson', created_at=now))
            db.add(ChatMessage(user_id=1, conversation_id='c', role='user', content='old', context_type='lesson',
                               created_at=plans.month_start() - timedelta(days=1)))
            db.add(StudyFolder(user_id=1, name='Physics', created_at=now))
            db.add(StudyFolder(user_id=1, name='Memory Palace', created_at=now))
            db.add(FolderSource(user_id=1, folder_name='Physics', source_id='src_1', title='t', filename='f.pdf',
                                source_type='pdf', raw_text='x', created_at=now))
            db.commit()
        import database
        with engine.begin() as conn:
            database._count_this_month_so_far(conn)
        self.assertEqual(self.plan(1)['usage'], {'lessons': {'used': 1, 'limit': 5}, 'uploads': {'used': 1, 'limit': 20},
                                                 'messages': {'used': 1, 'limit': 600}})


if __name__ == '__main__':
    unittest.main(verbosity=1)
