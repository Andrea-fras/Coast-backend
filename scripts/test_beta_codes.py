#!/usr/bin/env python3
"""Beta invite codes: every new account needs an unused code, and a code opens exactly one account."""
import os, sys, tempfile, unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TMP = tempfile.TemporaryDirectory(prefix='coast-beta-')
for key, name in {'DATABASE_PATH': 'app.db', 'OMA_DB_PATH': 'oma.db', 'CHROMA_PATH': 'chroma',
                  'GENERATED_DIR': 'generated', 'FOLDER_UPLOADS_DIR': 'sources', 'OMA_IMAGE_DIR': 'images'}.items():
    os.environ[key] = str(Path(TMP.name) / name)
for key in ('OPENAI_API_KEY', 'GEMINI_API_KEY', 'ANTHROPIC_API_KEY', 'RENDER', 'AUTH_EMAIL_VERIFICATION',
            'COAST_REQUIRE_BETA_CODE'):
    os.environ.pop(key, None)
import dotenv
dotenv.load_dotenv = lambda *a, **k: None
from fastapi.testclient import TestClient
import server
import beta_codes
from auth import create_access_token
from database import Base, BetaCode, SessionLocal, User, engine

client = TestClient(server.app)  # No lifespan: no startup jobs.
ADMIN = next(iter(server.ADMIN_EMAILS))


def new_code(note=''):
    with SessionLocal() as db:
        return beta_codes.display(beta_codes.create(db, note=note)[0].code)


def register(email, code='', name='Student'):
    return client.post('/api/auth/register', json={'email': email, 'name': name, 'password': 'secret123',
                                                   'beta_code': code})


def users():
    with SessionLocal() as db:
        return sorted(u.email for u in db.query(User).all())


class BetaCodes(unittest.TestCase):
    def setUp(self):
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
        from security import limiter
        limiter._hits.clear()  # sign-ups are rate-limited per address; each test starts fresh

    def test_signup_needs_a_code(self):
        r = register('ada@example.com')
        self.assertEqual(r.status_code, 400)
        self.assertIn('beta code', r.json()['detail'])
        r = register('ada@example.com', 'COAST-2222-2222')
        self.assertEqual(r.status_code, 400)
        self.assertIn("isn't valid", r.json()['detail'])
        self.assertEqual(users(), [])

    def test_code_opens_exactly_one_account(self):
        code = new_code('Ada')
        typed = code.lower().replace('-', ' ')  # students type codes loosely
        r = register('ada@example.com', typed)
        self.assertEqual(r.status_code, 200, r.text)
        with SessionLocal() as db:
            row = db.get(BetaCode, beta_codes.normalize(code))
            self.assertEqual(row.used_email, 'ada@example.com')
            self.assertEqual(row.used_by_user_id, r.json()['user']['id'])
        again = register('grace@example.com', code)
        self.assertEqual(again.status_code, 400)
        self.assertIn('already been used', again.json()['detail'])
        self.assertEqual(users(), ['ada@example.com'])

    def test_short_form_without_prefix_is_accepted(self):
        code = new_code()
        self.assertEqual(register('ada@example.com', code.split('-', 1)[1]).status_code, 200)

    def test_existing_email_does_not_burn_a_code(self):
        first = new_code()
        self.assertEqual(register('ada@example.com', first).status_code, 200)
        second = new_code()
        r = register('ada@example.com', second)
        self.assertEqual(r.status_code, 400)
        self.assertIn('already registered', r.json()['detail'])
        with SessionLocal() as db:
            self.assertIsNone(db.get(BetaCode, beta_codes.normalize(second)).used_at)

    def test_revoked_code_is_refused(self):
        code = new_code()
        with SessionLocal() as db:
            beta_codes.revoke(db, code)
        r = register('ada@example.com', code)
        self.assertEqual(r.status_code, 400)
        self.assertIn('no longer active', r.json()['detail'])

    def test_simultaneous_signups_with_one_code(self):
        code = new_code()
        emails = [f'student{i}@example.com' for i in range(6)]
        with ThreadPoolExecutor(len(emails)) as pool:
            results = list(pool.map(lambda e: register(e, code), emails))
        self.assertEqual(sorted(r.status_code for r in results), [200] + [400] * 5, [r.text for r in results])
        self.assertEqual(len(users()), 1)

    def test_login_is_unchanged(self):
        self.assertEqual(register('ada@example.com', new_code()).status_code, 200)
        r = client.post('/api/auth/login', json={'email': 'ada@example.com', 'password': 'secret123'})
        self.assertEqual(r.status_code, 200)

    def test_can_be_switched_off(self):
        with patch.dict(os.environ, {'COAST_REQUIRE_BETA_CODE': '0'}):
            self.assertFalse(client.get('/api/auth/config').json()['beta_code_required'])
            self.assertEqual(register('ada@example.com').status_code, 200)
        self.assertTrue(client.get('/api/auth/config').json()['beta_code_required'])

    def test_local_load_test_accounts_are_exempt(self):
        self.assertEqual(register('bot@loadtest.local').status_code, 200)
        with patch.dict(os.environ, {'RENDER': 'true'}):
            self.assertEqual(register('smoke@test.local').status_code, 400)

    def test_google_signup_needs_a_code_but_google_login_does_not(self):
        info = {'email': 'lin@example.com', 'sub': 'g-123', 'name': 'Lin', 'email_verified': True}
        with patch.dict(os.environ, {'GOOGLE_CLIENT_ID': 'test-client'}), \
                patch('google.oauth2.id_token.verify_oauth2_token', return_value=info):
            r = client.post('/api/auth/google', json={'credential': 'x'})
            self.assertEqual(r.status_code, 403)
            self.assertEqual(users(), [])
            code = new_code()
            r = client.post('/api/auth/google', json={'credential': 'x', 'beta_code': code})
            self.assertEqual(r.status_code, 200, r.text)
            again = client.post('/api/auth/google', json={'credential': 'x'})  # returning user
            self.assertEqual(again.status_code, 200)
        with SessionLocal() as db:
            self.assertEqual(db.get(BetaCode, beta_codes.normalize(code)).used_email, 'lin@example.com')

    def test_admin_manages_codes(self):
        with SessionLocal() as db:
            db.add_all([User(id=1, email=ADMIN, name='Admin', email_verified=True), User(id=2, email='kid@example.com', name='Kid')])
            db.commit()
        admin = {'Authorization': 'Bearer ' + create_access_token(1, ADMIN)}
        student = {'Authorization': 'Bearer ' + create_access_token(2, 'kid@example.com')}
        self.assertEqual(client.get('/api/admin/beta-codes', headers=student).status_code, 403)
        self.assertEqual(client.post('/api/admin/beta-codes', headers=student, json={'count': 1}).status_code, 403)

        made = client.post('/api/admin/beta-codes', headers=admin, json={'count': 3, 'note': 'ML society'}).json()['codes']
        self.assertEqual(len(made), 3)
        self.assertTrue(all(c['code'].startswith('COAST-') and c['status'] == 'unused' for c in made))
        self.assertEqual(register('ada@example.com', made[0]['code']).status_code, 200)
        revoked = client.post(f"/api/admin/beta-codes/{made[1]['code']}/revoke", headers=admin)
        self.assertEqual(revoked.json()['code']['status'], 'revoked')
        used = client.post(f"/api/admin/beta-codes/{made[0]['code']}/revoke", headers=admin)
        self.assertEqual(used.status_code, 400)

        listing = client.get('/api/admin/beta-codes', headers=admin).json()
        self.assertEqual(listing['counts'], {'unused': 1, 'used': 1, 'revoked': 1})
        by_code = {c['code']: c for c in listing['codes']}
        self.assertEqual(by_code[made[0]['code']]['used_email'], 'ada@example.com')
        self.assertEqual(by_code[made[0]['code']]['note'], 'ML society')

        signups = client.get('/api/admin/control-center', headers=admin).json()['recent_signups']
        self.assertEqual({u['email']: u['beta_code'] for u in signups}['ada@example.com'], made[0]['code'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
