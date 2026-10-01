#!/usr/bin/env python3
"""AI usage accounting: every provider call is recorded with its student, feature, tokens and cost."""
import os, sys, tempfile, threading, unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TMP = tempfile.TemporaryDirectory(prefix='coast-usage-')
for key, name in {'DATABASE_PATH': 'app.db', 'OMA_DB_PATH': 'oma.db', 'CHROMA_PATH': 'chroma',
                  'GENERATED_DIR': 'generated', 'FOLDER_UPLOADS_DIR': 'sources', 'OMA_IMAGE_DIR': 'images'}.items():
    os.environ[key] = str(Path(TMP.name) / name)
for key in ('OPENAI_API_KEY', 'GEMINI_API_KEY', 'ANTHROPIC_API_KEY', 'RENDER', 'COAST_REQUIRE_BETA_CODE'):
    os.environ.pop(key, None)
import dotenv
dotenv.load_dotenv = lambda *a, **k: None
from fastapi.testclient import TestClient
import server
import ai_usage
import provider_capacity
from auth import create_access_token
from database import AiUsage, SessionLocal, User, init_db

init_db()
client = TestClient(server.app)
ADMIN = next(iter(server.ADMIN_EMAILS))


def openai_response(prompt, cached, completion, model='gpt-4o-mini-2024-07-18'):
    return NS(model=model, usage=NS(prompt_tokens=prompt, completion_tokens=completion,
                                    prompt_tokens_details=NS(cached_tokens=cached)))


def rows():
    ai_usage.flush()
    with SessionLocal() as db:
        return db.query(AiUsage).order_by(AiUsage.id).all()


class UsageTests(unittest.TestCase):
    def setUp(self):
        import provider_capacity
        provider_capacity._down_until.clear()  # another suite may have simulated a provider out of credits
        ai_usage.flush()
        with SessionLocal() as db:
            db.query(AiUsage).delete()
            db.commit()

    def test_reads_each_providers_usage(self):
        self.assertEqual(ai_usage.read_usage('openai', openai_response(1000, 600, 50)),
                         {'input': 1000, 'cached': 600, 'cache_write': 0, 'output': 50, 'model': 'gpt-4o-mini-2024-07-18'})
        gemini = NS(model_version='gemini-3-flash-preview', usage_metadata=NS(
            prompt_token_count=900, cached_content_token_count=0, candidates_token_count=40, thoughts_token_count=60))
        self.assertEqual(ai_usage.read_usage('gemini', gemini)['output'], 100)  # thinking is billed as output
        claude = NS(model='claude-sonnet-5', usage=NS(input_tokens=100, cache_read_input_tokens=5000,
                                                      cache_creation_input_tokens=200, output_tokens=80))
        self.assertEqual(ai_usage.read_usage('anthropic', claude),
                         {'input': 5300, 'cached': 5000, 'cache_write': 200, 'output': 80, 'model': 'claude-sonnet-5'})

    def test_costs_use_cached_and_output_prices(self):
        # gpt-4o-mini: $0.15 in, $0.075 cached, $0.60 out per million
        cost = ai_usage.cost_usd('gpt-4o-mini-2024-07-18', 1_000_000, 400_000, 0, 100_000)
        self.assertAlmostEqual(cost, 0.6 * 0.15 + 0.4 * 0.075 + 0.1 * 0.60)
        self.assertIsNone(ai_usage.cost_usd('some-new-model', 10, 0, 0, 10))
        self.assertEqual(ai_usage.price_for('gpt-4o-2024-08-06'), ai_usage.AI_PRICES['gpt-4o'])  # not gpt-4o-mini

    def test_calls_are_recorded_with_student_and_feature(self):
        token = ai_usage.begin(7, 'folders/outline')
        try:
            provider_capacity.call('openai', lambda: openai_response(2000, 0, 300))
            ai_usage.tag(feature='pedro:lesson')
            # Work handed to another thread keeps the request's attribution.
            worker = threading.Thread(target=ai_usage.carry(
                lambda: provider_capacity.call('openai', lambda: openai_response(10, 0, 1))))
            worker.start(); worker.join()
        finally:
            ai_usage.end(token)
        first, second = rows()
        self.assertEqual((first.user_id, first.feature, first.input_tokens, first.output_tokens), (7, 'folders/outline', 2000, 300))
        self.assertEqual((second.user_id, second.feature), (7, 'pedro:lesson'))

    def test_failed_calls_are_counted(self):
        def fail():
            raise RuntimeError('boom')
        with self.assertRaises(RuntimeError):
            provider_capacity.call('openai', fail)
        self.assertFalse(rows()[0].ok)

    def test_streams_record_final_usage_and_hide_the_usage_chunk(self):
        chunks = [NS(model='gpt-4o', choices=[NS(delta=NS(content='Hi'))], usage=None),
                  NS(model='gpt-4o', choices=[], usage=NS(prompt_tokens=500, completion_tokens=20, prompt_tokens_details=None))]
        seen = list(provider_capacity.stream('openai', lambda: iter(chunks)))
        self.assertEqual(len(seen), 1)  # callers index choices[0]; the usage-only chunk would break them
        self.assertEqual((rows()[0].input_tokens, rows()[0].output_tokens), (500, 20))
        events = [NS(type='message_start', message=NS(model='claude-sonnet-5', usage=NS(
                      input_tokens=40, cache_read_input_tokens=3000, cache_creation_input_tokens=0, output_tokens=1))),
                  NS(type='content_block_delta'), NS(type='message_delta', usage=NS(output_tokens=250))]
        list(provider_capacity.stream('anthropic', lambda: iter(events)))
        claude = rows()[-1]
        self.assertEqual((claude.input_tokens, claude.cached_tokens, claude.output_tokens), (3040, 3000, 250))

    def test_a_student_leaving_mid_answer_still_records_tokens(self):
        chunks = iter([NS(model_version='gemini-3-flash-preview', usage_metadata=NS(prompt_token_count=800, candidates_token_count=5)),
                       NS(model_version='gemini-3-flash-preview', usage_metadata=NS(prompt_token_count=800, candidates_token_count=9))])
        stream = provider_capacity.stream('gemini', lambda: chunks)
        next(stream)
        stream.close()
        self.assertEqual(rows()[0].input_tokens, 800)

    def test_admin_summary(self):
        with SessionLocal() as db:
            admin = db.query(User).filter_by(email=ADMIN).first() or User(email=ADMIN, name='Admin', password_hash='x', email_verified=True)
            student = db.query(User).filter_by(email='student@example.com').first() or User(email='student@example.com', name='S', password_hash='x')
            db.add_all([admin, student]); db.commit()
            admin_id, student_id = admin.id, student.id
        token = ai_usage.begin(student_id, 'pedro:lesson')
        provider_capacity.call('openai', lambda: openai_response(1_000_000, 0, 0))
        ai_usage.end(token)
        res = client.get('/api/admin/ai-usage?days=7', headers={'Authorization': 'Bearer ' + create_access_token(admin_id, ADMIN)})
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertAlmostEqual(body['total']['cost_usd'], 0.15)
        self.assertEqual(body['total']['students'], 1)
        self.assertEqual(body['by_feature'][0]['feature'], 'pedro:lesson')
        self.assertEqual(body['top_students'][0]['email'], 'student@example.com')
        student_res = client.get('/api/admin/ai-usage', headers={'Authorization': 'Bearer ' + create_access_token(student_id, 'student@example.com')})
        self.assertEqual(student_res.status_code, 403)

    def test_requests_are_attributed_by_the_middleware(self):
        seen = {}
        original = server.health

        @server.app.get('/api/_usage_probe/{name}')
        def probe(name: str):
            seen.update(ai_usage._scope.get() or {})
            return original()
        client.get('/api/_usage_probe/x', headers={'Authorization': 'Bearer ' + create_access_token(42, 'probe@example.com')})
        self.assertEqual(seen, {'user_id': 42, 'feature': '_usage_probe/x'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
