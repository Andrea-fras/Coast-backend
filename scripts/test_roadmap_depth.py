#!/usr/bin/env python3
"""Roadmap depth: "essentials" keeps the important topics in fewer sections of the usual size and
leaves the rest out (still searchable); "complete" is the roadmap as it always was. Isolated
database; the planner is a stand-in."""
import json, os, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TMP = tempfile.TemporaryDirectory(prefix='coast-depth-')
for key, name in {'DATABASE_PATH': 'app.db', 'OMA_DB_PATH': 'oma.db', 'CHROMA_PATH': 'chroma',
                  'GENERATED_DIR': 'generated', 'FOLDER_UPLOADS_DIR': 'sources', 'OMA_IMAGE_DIR': 'images'}.items():
    os.environ[key] = str(Path(TMP.name) / name)
for key in ('OPENAI_API_KEY', 'GEMINI_API_KEY', 'ANTHROPIC_API_KEY', 'RENDER', 'FILE_STORE'):
    os.environ.pop(key, None)
import dotenv
dotenv.load_dotenv = lambda *a, **k: None
import lesson
from coast_content_oma import progressive
from database import Base, CourseOutline, FolderSource, SessionLocal, StudyFolder, User, engine


def unit(source, first, last):
    pages = list(range(first, last + 1))
    return {'source_id': source, 'source_title': source.upper(), 'source_filename': source + '.pdf', 'sha256': 'x',
            'pages': pages, 'text_chars': 400 * len(pages), 'page_chars': {p: 400 for p in pages}}


UNITS = {'src_a:1-8': unit('src_a', 1, 8), 'src_a:9-16': unit('src_a', 9, 16), 'src_b:1-8': unit('src_b', 1, 8)}
SECTIONS = [
    {'title': 'Core A', 'learning_objectives': ['a'], 'key_topics': ['a'], 'source_notebooks': ['SRC_A'],
     'estimated_minutes': 20, 'source_units': ['src_a:2-6']},
    {'title': 'Core B', 'learning_objectives': ['b'], 'key_topics': ['b'], 'source_notebooks': ['SRC_B'],
     'estimated_minutes': 20, 'source_units': ['src_b:1-5']},
]


def taught(sections):
    return {(r['source_id'], p) for s in sections for r in s['source_refs'] for p in r['pages']}


class RoadmapDepth(unittest.TestCase):
    def setUp(self):
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
        with SessionLocal() as db:
            db.add(User(id=1, email='ada@example.com', name='Ada', password_hash='x'))
            db.add(StudyFolder(user_id=1, name='Physics', kind='lesson'))
            db.add(StudyFolder(user_id=1, name='Build', kind='workshop'))
            for folder in ('Physics', 'Build'):
                for sid, n in (('src_a', 16), ('src_b', 8)):
                    db.add(FolderSource(user_id=1, folder_name=folder, source_id=f'{sid}{"" if folder == "Physics" else "_w"}',
                                        title=sid.upper(), filename=sid + '.pdf', source_type='pdf', page_count=n, raw_text='x'))
            db.commit()
        self.prompts = []

    def plan(self, left_out=None):
        def fake(system, format_line, context, course_format, essentials=False):
            self.prompts.append((system, essentials))
            return json.loads(json.dumps(SECTIONS)), ['src_a:1'], (left_out or []) if essentials else []
        return fake

    def generate(self, folder='Physics', depth=None, left_out=None, kind='lesson'):
        with patch('oma_provider.is_oma_enabled', return_value=True), \
             patch.object(progressive, 'overview', return_value=('UNIT src_a:1-8 | source "SRC_A"\np.1 (0 images): x', UNITS)), \
             patch.object(lesson, '_plan_outline', side_effect=self.plan(left_out)):
            return lesson.generate_outline(1, folder, course_format=kind, depth=depth)

    def test_essentials_leaves_topics_out_instead_of_growing_sections(self):
        result = self.generate(depth='essentials', left_out=[{'topic': 'History of A', 'source_units': ['src_a:9-16']}])
        self.assertEqual(result['depth'], 'essentials')
        system, essentials = self.prompts[-1]
        self.assertTrue(essentials)
        self.assertIn('DEPTH: ESSENTIALS', system)
        self.assertIn('each section covers at most 14 pages', system)
        # Sections keep exactly the pages the planner chose: nothing left over is pushed into them.
        self.assertEqual(taught(result['sections']), {('src_a', p) for p in range(2, 7)} | {('src_b', p) for p in range(1, 6)})
        self.assertEqual(result['left_out'], [
            {'topic': 'History of A', 'pages': 8, 'source_units': [f'src_a:{p}' for p in range(9, 17)]},
            {'topic': 'Other pages', 'pages': 5, 'source_units': ['src_a:7', 'src_a:8', 'src_b:6', 'src_b:7', 'src_b:8']},
        ])  # the administrative title page (src_a:1) is neither taught nor "left out"
        state = lesson.get_lesson_state(1, 'Physics')
        self.assertEqual((state['depth'], [t['topic'] for t in state['left_out']]), ('essentials', ['History of A', 'Other pages']))

    def test_complete_is_the_roadmap_as_before(self):
        result = self.generate(depth='complete')
        system, essentials = self.prompts[-1]
        self.assertFalse(essentials)
        self.assertNotIn('DEPTH: ESSENTIALS', system)
        self.assertIn('Every page belongs to exactly one section', system)
        everything = {(u['source_id'], p) for u in UNITS.values() for p in u['pages']} - {('src_a', 1)}
        self.assertEqual(taught(result['sections']), everything)  # every page taught, as always
        self.assertEqual(result['left_out'], [])

    def test_essentials_asks_for_about_half_the_sections(self):
        self.generate(depth='complete')
        self.generate(depth='essentials')
        import re
        complete, essentials = (re.search(r'Create (\d+)-(\d+) sections', s).groups() for s, _ in self.prompts[-2:])
        self.assertLess(int(essentials[1]), int(complete[1]))

    def test_a_new_lesson_starts_at_essentials_and_a_regenerate_keeps_the_choice(self):
        self.assertEqual(self.generate()['depth'], 'essentials')
        self.generate(depth='complete')
        self.assertEqual(self.generate()['depth'], 'complete')  # regenerate without a choice keeps it
        with SessionLocal() as db:
            self.assertEqual(db.query(CourseOutline).filter_by(user_id=1, folder_name='Physics').one().depth, 'complete')

    def test_workshops_have_no_depth(self):
        units = {k.replace('src_a', 'src_a_w').replace('src_b', 'src_b_w'): {**v, 'source_id': v['source_id'] + '_w'}
                 for k, v in UNITS.items()}
        sections = [{**s, 'source_units': [u.replace('src_a', 'src_a_w').replace('src_b', 'src_b_w') for u in s['source_units']]}
                    for s in SECTIONS]
        with patch('oma_provider.is_oma_enabled', return_value=True), \
             patch.object(progressive, 'overview', return_value=('x', units)), \
             patch.object(lesson, '_plan_outline', side_effect=lambda system, f, c, cf, essentials=False:
                          (self.prompts.append((system, essentials)) or json.loads(json.dumps(sections)), [], [])):
            result = lesson.generate_outline(1, 'Build', course_format='workshop', depth='essentials')
        self.assertIsNone(result['depth'])
        self.assertFalse(self.prompts[-1][1])

    def test_an_oversized_essentials_section_is_sent_back_then_split(self):
        big = [{**SECTIONS[0], 'title': 'Everything about A', 'source_units': ['src_a:2-16', 'src_b:1-8']}]
        calls = []

        def fake(system, format_line, context, course_format, essentials=False):
            calls.append(context)
            return json.loads(json.dumps(big)), ['src_a:1'], []  # ignores the repair note, too big again

        with patch('oma_provider.is_oma_enabled', return_value=True), \
             patch.object(progressive, 'overview', return_value=('x', UNITS)), \
             patch.object(lesson, '_plan_outline', side_effect=fake):
            result = lesson.generate_outline(1, 'Physics', course_format='lesson', depth='essentials')
        self.assertEqual(len(calls), 2)
        self.assertIn('broke the size rule', calls[1])
        sizes = [sum(len(r['pages']) for r in s['source_refs']) for s in result['sections']]
        self.assertEqual(sum(sizes), 23)                      # nothing taught twice or dropped by the split
        self.assertTrue(all(n <= 14 for n in sizes), sizes)   # and no section is bigger than a normal one
        self.assertEqual([s['title'] for s in result['sections']], ['Everything about A (1/2)', 'Everything about A (2/2)'])

    def finish_first_section(self):
        from database import ChatMessage, SectionRewardClaim, SectionVerification
        from datetime import datetime, timedelta, timezone
        earlier = datetime.now(timezone.utc) - timedelta(minutes=5)
        with SessionLocal() as db:
            db.add(SectionVerification(user_id=1, folder_name='Physics', section_index=0, is_active=True, verified_at=earlier))
            db.add(SectionRewardClaim(user_id=1, folder_name='Physics', section_index=0, xp_gained=50, created_at=earlier))
            db.add(ChatMessage(user_id=1, conversation_id='c', role='pedro', content='Great work! [SECTION_COMPLETE]',
                               context_type='lesson', context_id='Physics', section_index=0, created_at=earlier))
            outline = db.query(CourseOutline).filter_by(user_id=1, folder_name='Physics').one()
            outline.current_section = 1
            db.commit()

    def assert_starts_fresh(self):
        from database import SectionRewardClaim
        self.assertFalse(lesson.can_advance_from_section(1, 'Physics', 0))  # the new section 1 is not "done"
        state = lesson.get_lesson_state(1, 'Physics')
        self.assertEqual(state['current_section'], 0)
        self.assertFalse(state['section_verified'])
        import server
        from fastapi.testclient import TestClient
        from auth import create_access_token
        summary = TestClient(server.app).get('/api/lessons/summary', headers={
            'Authorization': 'Bearer ' + create_access_token(1, 'ada@example.com')}).json()['Physics']
        self.assertFalse(summary['section_progress'][0]['mastered'])
        with SessionLocal() as db:  # the XP and map tiles earned stay earned
            self.assertEqual(db.query(SectionRewardClaim).filter_by(user_id=1, folder_name='Physics').count(), 1)

    def test_a_regenerated_roadmap_starts_fresh(self):
        self.generate(depth='complete')
        self.finish_first_section()
        self.assertTrue(lesson.can_advance_from_section(1, 'Physics', 0))
        self.generate(depth='essentials')
        self.assert_starts_fresh()

    def test_a_reset_course_starts_fresh(self):
        self.generate(depth='complete')
        self.finish_first_section()
        lesson.reset_lesson(1, 'Physics')
        self.assert_starts_fresh()

    def test_binding_without_filling_gaps(self):
        sections = json.loads(json.dumps(SECTIONS))
        progressive.bind_sections(sections, UNITS, fill_gaps=False)
        self.assertEqual(len(taught(sections)), 10)
        sections = json.loads(json.dumps(SECTIONS))
        progressive.bind_sections(sections, UNITS)
        self.assertEqual(len(taught(sections)), 24)


if __name__ == '__main__':
    unittest.main(verbosity=1)
