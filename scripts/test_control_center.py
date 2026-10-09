#!/usr/bin/env python3
"""Control Center numbers: students only (no admin accounts, no test bots), activity only from
things done on purpose, retention and funnel counted right, feedback readable and resolvable,
reported study time capped. Isolated database."""
import os, sys, tempfile, unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TMP = tempfile.TemporaryDirectory(prefix='coast-cc-')
for key, name in {'DATABASE_PATH': 'app.db', 'OMA_DB_PATH': 'oma.db', 'CHROMA_PATH': 'chroma',
                  'GENERATED_DIR': 'generated', 'FOLDER_UPLOADS_DIR': 'sources', 'OMA_IMAGE_DIR': 'images'}.items():
    os.environ[key] = str(Path(TMP.name) / name)
for key in ('OPENAI_API_KEY', 'GEMINI_API_KEY', 'ANTHROPIC_API_KEY', 'RENDER', 'FILE_STORE'):
    os.environ.pop(key, None)
import dotenv
dotenv.load_dotenv = lambda *a, **k: None
from fastapi.testclient import TestClient
import server
import growth_metrics
from auth import create_access_token
from database import (ActivityEvent, Base, ChatMessage, CourseOutline, SectionRewardClaim, SessionLocal, UsageEvent,
                      User, UserFeedback, engine)

client = TestClient(server.app)
ADMIN = 'andreaf.fraschetti@gmail.com'
NOW = datetime.now(timezone.utc)


def ago(days, hours=0):
    return NOW - timedelta(days=days, hours=hours)


class ControlCenter(unittest.TestCase):
    def setUp(self):
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
        growth_metrics.clear_cache()
        from security import limiter
        limiter._hits.clear()
        with SessionLocal() as db:
            db.add(User(id=1, email=ADMIN, name='Andrea', password_hash='x', email_verified=True, created_at=ago(40)))
            db.add(User(id=2, email='bot1@loadtest.local', name='Bot', password_hash='x', created_at=ago(1)))
            # Ada: joined 10 days ago, studied, came back on day 1, finished a section.
            db.add(User(id=10, email='ada@example.com', name='Ada', password_hash='x', onboarding_completed=True,
                        created_at=ago(10)))
            # Ben: joined 3 days ago, only got Coast's automatic opener in a lesson.
            db.add(User(id=11, email='ben@example.com', name='Ben', password_hash='x', onboarding_completed=True,
                        created_at=ago(3)))
            # Cy: joined today, did nothing yet, asked for the founder pass.
            db.add(User(id=12, email='cy@example.com', name='Cy', password_hash='x', created_at=NOW,
                        founder_interest_at=NOW))
            msg = lambda uid, text, when, ctx='lesson': ChatMessage(user_id=uid, conversation_id='c', role='user',
                                                                     content=text, context_type=ctx, created_at=when)
            db.add_all([
                msg(10, 'What is a queue?', ago(10)),
                msg(10, 'And a stack?', ago(9)),          # came back on day 1
                msg(10, 'Thanks', NOW),                   # active today
                msg(11, "I'm ready to learn about \"Queues\". Please teach me this section.", ago(2)),  # automatic
                msg(1, 'admin testing', NOW),             # the admin never counts
                msg(2, 'bot noise', NOW),
            ])
            db.add(ChatMessage(user_id=10, conversation_id='c', role='assistant', content='reply', context_type='lesson',
                               created_at=NOW))
            db.add(SectionRewardClaim(user_id=10, folder_name='Physics', section_index=0, created_at=ago(9)))
            db.add(CourseOutline(user_id=10, folder_name='Physics', outline_json='[]', total_sections=3))
            db.add(UsageEvent(user_id=11, kind='lessons', ref='Maths', created_at=ago(3)))
            db.add(ActivityEvent(user_id=10, feature='notebook', duration_ms=30 * 60000, event_date=NOW.strftime('%Y-%m-%d')))
            db.add(ActivityEvent(user_id=10, feature='notebook', duration_ms=20 * 3600000,  # a forgotten tab, old data
                                 event_date=ago(9).strftime('%Y-%m-%d')))
            db.add(UserFeedback(user_id=10, category='bug', message='Slides vanished', page='notebook', client='Chrome 141 · macOS'))
            db.add(UserFeedback(user_id=11, category='suggestion', message='Dark notes please', page='map'))
            db.commit()

    def h(self, uid=1, email=ADMIN):
        return {'Authorization': 'Bearer ' + create_access_token(uid, email)}

    def growth(self):
        with SessionLocal() as db:
            return growth_metrics.compute(db)

    def test_only_students_count_and_only_on_purpose(self):
        g = self.growth()
        head = g['headline']
        self.assertEqual(head['students'], 3)            # not the admin, not the bot
        self.assertEqual(head['new_today'], 1)
        self.assertEqual(head['dau'], 1)                 # Ada today; the admin and the bot don't count
        self.assertEqual(head['wau'], 2)                 # Ada, and Ben for making a lesson (not for the opener)
        self.assertEqual(head['messages_today'], 1)      # Ada's question, not Pedro's reply
        self.assertEqual(g['totals']['messages'], 3)     # Ada's three; the automatic opener isn't one
        self.assertEqual(g['totals']['sections'], 1)
        self.assertEqual(head['activation_pct'], 33.3)   # 1 of 3 finished a section

    def test_retention_and_the_funnel(self):
        g = self.growth()
        self.assertEqual(g['retention']['d1'], {'pct': 50.0, 'of': 2})          # Ada yes, Ben no; Cy is too new
        self.assertEqual(g['retention']['first_week'], {'pct': 100.0, 'of': 1})  # only Ada joined 7+ days ago
        steps = {s['step']: s['count'] for s in g['funnel']}
        self.assertEqual(steps, {'Signed up': 3, 'Finished the welcome': 2, 'Started a lesson or workshop': 2,
                                 'Studied with Pedro': 1, 'Finished a section': 1, 'Came back another day': 1})
        self.assertEqual(g['monetization']['wants_pass'], 1)
        self.assertEqual(g['monetization']['pass_intent_pct'], 33.3)

    def test_a_forgotten_tab_cannot_make_a_six_hour_day_longer(self):
        g = self.growth()
        self.assertEqual(g['totals']['study_hours'], 6.5)  # 6 h cap on the old day + 30 min today

    def test_a_reported_minute_is_capped_at_three(self):
        r = client.post('/api/activity', headers=self.h(10, 'ada@example.com'),
                        json={'feature': 'notebook', 'duration_ms': 10 * 3600000})
        self.assertEqual(r.status_code, 200)
        with SessionLocal() as db:
            self.assertEqual(db.query(ActivityEvent).order_by(ActivityEvent.id.desc()).first().duration_ms, 180000)
        # A finished 25-minute focus block counts whole.
        client.post('/api/activity', headers=self.h(10, 'ada@example.com'),
                    json={'feature': 'focus', 'action': 'complete', 'duration_ms': 25 * 60000})
        with SessionLocal() as db:
            self.assertEqual(db.query(ActivityEvent).order_by(ActivityEvent.id.desc()).first().duration_ms, 25 * 60000)

    def test_the_control_center_payload(self):
        r = client.get('/api/admin/control-center', headers=self.h())
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data['kpis']['total_users'], 3)
        self.assertEqual(data['kpis']['loadtest_bots'], 1)
        self.assertEqual(data['feedback_new'], {'bug': 1, 'suggestion': 1, 'other': 0})
        self.assertIn('cohorts', data['growth'])
        self.assertEqual(len(data['growth']['weekly']), 12)
        self.assertEqual(client.get('/api/admin/control-center', headers=self.h(10, 'ada@example.com')).status_code, 403)

    def test_feedback_is_listed_with_its_screen_and_can_be_resolved(self):
        r = client.post('/api/feedback', headers=self.h(11, 'ben@example.com'),
                        json={'category': 'bug', 'message': 'Upload stuck', 'page': 'notebook', 'client': 'Safari 19 · macOS · 1280×720'})
        self.assertEqual(r.status_code, 200)
        items = client.get('/api/admin/feedback', headers=self.h()).json()['feedback']
        newest = items[0]
        self.assertEqual((newest['message'], newest['page'], newest['client'], newest['status']),
                         ('Upload stuck', 'notebook', 'Safari 19 · macOS · 1280×720', 'new'))
        r = client.post(f"/api/admin/feedback/{newest['id']}", headers=self.h(), json={'status': 'resolved'})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(client.get('/api/admin/feedback', headers=self.h()).json()['feedback'][0]['status'], 'resolved')
        self.assertEqual(client.post(f"/api/admin/feedback/{newest['id']}", headers=self.h(11, 'ben@example.com'),
                                     json={'status': 'new'}).status_code, 403)

    def test_odd_feedback_categories_become_other(self):
        client.post('/api/feedback', headers=self.h(11, 'ben@example.com'), json={'category': '<script>', 'message': 'hi'})
        self.assertEqual(client.get('/api/admin/feedback', headers=self.h()).json()['feedback'][0]['category'], 'other')

    def test_bot_cleanup_deletes_the_whole_account(self):
        r = client.post('/api/admin/cleanup-loadtest-users', headers=self.h())
        self.assertEqual(r.json()['deleted_users'], 1)
        with SessionLocal() as db:
            self.assertIsNone(db.get(User, 2))
            self.assertEqual(db.query(ChatMessage).filter_by(user_id=2).count(), 0)
            self.assertIsNotNone(db.get(User, 10))


if __name__ == '__main__':
    unittest.main(verbosity=1)
