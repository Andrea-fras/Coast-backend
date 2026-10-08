#!/usr/bin/env python3
"""Sign-up with an emailed code, Google sign-in during the invite-only beta, and forgot password.
Isolated database; emails and Google are stand-ins."""
import os, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TMP = tempfile.TemporaryDirectory(prefix='coast-auth-')
for key, name in {'DATABASE_PATH': 'app.db', 'OMA_DB_PATH': 'oma.db', 'CHROMA_PATH': 'chroma',
                  'GENERATED_DIR': 'generated', 'FOLDER_UPLOADS_DIR': 'sources', 'OMA_IMAGE_DIR': 'images'}.items():
    os.environ[key] = str(Path(TMP.name) / name)
for key in ('OPENAI_API_KEY', 'GEMINI_API_KEY', 'ANTHROPIC_API_KEY', 'RENDER', 'COAST_REQUIRE_BETA_CODE',
            'RESEND_API_KEY', 'AUTH_DEV_EXPOSE_CODES'):
    os.environ.pop(key, None)
os.environ['AUTH_EMAIL_VERIFICATION'] = '1'
os.environ['GOOGLE_CLIENT_ID'] = 'test-client.apps.googleusercontent.com'
import dotenv
dotenv.load_dotenv = lambda *a, **k: None
from fastapi.testclient import TestClient
import server
import beta_codes
from database import Base, BetaCode, SessionLocal, User, engine

client = TestClient(server.app)
SENT = []


def fake_send(email, code, purpose='verify'):
    SENT.append((email, code, purpose))
    return True, 'sent'


def new_code():
    with SessionLocal() as db:
        return beta_codes.display(beta_codes.create(db)[0].code)


def google_token(email, sub, verified=True):
    return {'email': email, 'sub': sub, 'name': 'Grace Hopper', 'email_verified': verified}


class AuthFlows(unittest.TestCase):
    def setUp(self):
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
        from security import limiter
        limiter._hits.clear()
        SENT.clear()
        self.send = patch('auth_email.send_code_email', side_effect=fake_send)
        self.send.start()
        self.addCleanup(self.send.stop)
        self.mx = patch('auth_email.validate_email_address', return_value=(True, ''))  # no DNS lookups
        self.mx.start()
        self.addCleanup(self.mx.stop)

    def test_signup_emails_a_code_only_to_someone_with_a_beta_code(self):
        r = client.post('/api/auth/verify-email/send', json={'email': 'ada@example.com', 'beta_code': ''})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(SENT, [])  # no invite, no email
        code = new_code()
        r = client.post('/api/auth/verify-email/send', json={'email': 'ada@example.com', 'beta_code': code})
        self.assertEqual(r.status_code, 200, r.text)
        (email, emailed, purpose), = SENT
        self.assertEqual((email, purpose), ('ada@example.com', 'verify'))
        # Without the emailed code there is no account; with it, the account opens and the invite is used.
        r = client.post('/api/auth/register', json={'email': email, 'name': 'Ada', 'password': 'secret123', 'beta_code': code})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(client.post('/api/auth/verify-email/check', json={'email': email, 'code': emailed}).status_code, 200)
        r = client.post('/api/auth/register', json={'email': email, 'name': 'Ada', 'password': 'secret123', 'beta_code': code})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()['user']['email_verified'])
        with SessionLocal() as db:
            self.assertIsNotNone(db.get(BetaCode, beta_codes.normalize(code)).used_at)

    def test_a_new_google_account_is_asked_for_its_beta_code_and_then_opens(self):
        with patch('google.oauth2.id_token.verify_oauth2_token', return_value=google_token('grace@example.com', 'g-1')):
            r = client.post('/api/auth/google', json={'credential': 'id-token'})
            self.assertEqual(r.status_code, 403)
            self.assertTrue(r.json()['detail']['needs_beta_code'])
            r = client.post('/api/auth/google', json={'credential': 'id-token', 'beta_code': 'COAST-2222-2222'})
            self.assertEqual(r.status_code, 403)
            self.assertIn("isn't valid", r.json()['detail']['message'])
            r = client.post('/api/auth/google', json={'credential': 'id-token', 'beta_code': new_code()})
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json()['user']['auth_provider'], 'google')
            # Next time: straight in, no code.
            self.assertEqual(client.post('/api/auth/google', json={'credential': 'id-token'}).status_code, 200)

    def test_google_on_an_existing_email_account_signs_in_and_proves_the_address(self):
        with SessionLocal() as db:
            db.add(User(email='andreaf.fraschetti@gmail.com', name='Andrea', password_hash='x', email_verified=False))
            db.commit()
        with patch('google.oauth2.id_token.verify_oauth2_token',
                   return_value=google_token('andreaf.fraschetti@gmail.com', 'g-admin')):
            r = client.post('/api/auth/google', json={'credential': 'id-token'})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()['user']['is_admin'])  # admin needs a proven address: Google proves it

    def test_an_account_made_before_verification_still_signs_in(self):
        with SessionLocal() as db:
            db.add(User(email='old@example.com', name='Old', password_hash=server.hash_password('secret123'), email_verified=False))
            db.commit()
        r = client.post('/api/auth/login', json={'email': 'old@example.com', 'password': 'secret123'})
        self.assertEqual(r.status_code, 200, r.text)

    def test_forgot_password_resets_with_the_emailed_code(self):
        with SessionLocal() as db:
            db.add(User(email='lin@example.com', name='Lin', password_hash=server.hash_password('oldpassword')))
            db.commit()
        r = client.post('/api/auth/password/forgot', json={'email': 'lin@example.com'})
        self.assertEqual(r.status_code, 200)
        (_, emailed, purpose), = SENT
        self.assertEqual(purpose, 'reset')
        self.assertEqual(client.post('/api/auth/password/reset', json={'email': 'lin@example.com', 'code': '000000',
                                                                        'password': 'newpassword'}).status_code, 400)
        r = client.post('/api/auth/password/reset', json={'email': 'lin@example.com', 'code': emailed, 'password': 'newpassword'})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn('token', r.json())
        self.assertEqual(client.post('/api/auth/login', json={'email': 'lin@example.com', 'password': 'oldpassword'}).status_code, 401)
        self.assertEqual(client.post('/api/auth/login', json={'email': 'lin@example.com', 'password': 'newpassword'}).status_code, 200)
        # A used code can't be used again.
        self.assertEqual(client.post('/api/auth/password/reset', json={'email': 'lin@example.com', 'code': emailed,
                                                                        'password': 'another123'}).status_code, 400)

    def test_forgot_password_never_reveals_whether_an_account_exists(self):
        r = client.post('/api/auth/password/forgot', json={'email': 'nobody@example.com'})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()['message'], "If an account uses this email, we've sent it a code.")
        self.assertEqual(SENT, [])

    def test_reset_codes_cannot_be_guessed(self):
        with SessionLocal() as db:
            db.add(User(email='kim@example.com', name='Kim', password_hash=server.hash_password('oldpassword')))
            db.commit()
        client.post('/api/auth/password/forgot', json={'email': 'kim@example.com'})
        (_, emailed, _), = SENT
        wrong = [client.post('/api/auth/password/reset', json={'email': 'kim@example.com', 'code': f'99999{i}',
                                                                'password': 'newpassword'}).status_code for i in range(6)]
        self.assertEqual(wrong[-1], 429)  # five wrong guesses burn the code
        r = client.post('/api/auth/password/reset', json={'email': 'kim@example.com', 'code': emailed, 'password': 'newpassword'})
        self.assertEqual(r.status_code, 429)


if __name__ == '__main__':
    unittest.main(verbosity=1)
