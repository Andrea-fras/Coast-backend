#!/usr/bin/env python3
"""Workshop integration using isolated SQLite; no real student/provider writes."""
import json
import unittest
from unittest.mock import patch
import test_http_integrity as fixture
from database import SessionLocal, CourseOutline, ChatMessage, CourseChatEpoch
from curated_config import get_static_outline, ensure_curated_outline, get_lesson_structure
from workshops import decorate_sections, prior_work, LEGACY_MEMORY_TITLES
import lesson


class Workshops(unittest.TestCase):
    def setUp(self):
        fixture.HttpIntegrity.setUp(self)
        self.sections = get_static_outline('Memory Palace')
        with SessionLocal() as db:
            db.add(CourseOutline(user_id=1, folder_name='Memory Palace', current_section=0,
                                total_sections=5, outline_json=json.dumps(self.sections)))
            db.commit()

    def prompt(self, index=0):
        with patch.object(lesson, '_fetch_section_material', return_value=('Memory Palace source: spatial cues and distinctive associations.', True)):
            return lesson.build_lesson_prompt(1, 'Memory Palace', source_user_id=0,
                structure=get_lesson_structure('Memory Palace'), section_index=index)

    def test_new_outline_has_complete_outcomes_and_all_five_milestones(self):
        self.assertEqual(len(self.sections), 5)
        for sec in self.sections:
            self.assertTrue(sec['workshop']['outcome'])
            self.assertEqual(len(sec['workshop']['criteria']), 2)
            self.assertEqual(sec['learning_objectives'], sec['workshop']['criteria'])
        self.assertEqual(sum(s['estimated_minutes'] for s in self.sections), 45)

    def test_workshop_prompt_has_one_coherent_pedagogy(self):
        prompt = self.prompt()
        self.assertIsNotNone(prompt)
        self.assertIn('WORKSHOP COMPLETION GATE', prompt)
        self.assertIn('The student creates the work', prompt)
        self.assertIn('Simonides story is optional colour', prompt)
        self.assertNotIn('You MUST run a practice/verification round', prompt)
        self.assertNotIn('BEFORE completing the section, you MUST do a PRACTICE ROUND', prompt)
        self.assertNotIn('Retell the Simonides', prompt)
        self.assertNotIn('problems and exam-style questions', prompt)
        self.assertIn('partial', prompt.lower())

    def test_recall_requires_actual_attempt_not_done_or_pedro_answer(self):
        prompt = self.prompt(3)
        self.assertIn("Do not infer success from 'done'", prompt)
        self.assertIn('unaided attempt', prompt)
        self.assertIn('without Pedro displaying the answers', prompt)
        self.assertIn('not proof they did not look', prompt)

    def test_unknown_and_academic_outlines_keep_their_rules(self):
        custom = [{'title':'Custom'}]
        self.assertEqual(decorate_sections('Memory Palace', custom), custom)
        # An ordinary academic outline (no stored contracts) stays a lesson; a stored
        # contract is how a generated workshop is defined, so only valid ones are kept.
        academic = [{k: v for k, v in s.items() if k != 'workshop'} for s in self.sections]
        self.assertTrue(all('workshop' not in s for s in decorate_sections('Physics', academic)))
        broken = [{**s, 'workshop': {'title': s['title']}} for s in academic]
        self.assertTrue(all('workshop' not in s for s in decorate_sections('Physics', broken)))
        with patch.object(lesson, '_fetch_section_material', return_value=('F=ma', True)):
            prompt = lesson.build_lesson_prompt(1, 'Physics')
        self.assertIn('SECTION VERIFICATION GATE', prompt)
        self.assertIn('PRACTICE ROUND', prompt)
        self.assertNotIn('GUIDED WORKSHOP', prompt)

    def test_legacy_progress_is_preserved_and_practical_contract_is_returned(self):
        legacy = [{**s, 'title':title} for s, title in zip(self.sections, LEGACY_MEMORY_TITLES)]
        for sec in legacy:
            sec.pop('workshop')
        original = json.dumps(legacy)
        with SessionLocal() as db:
            row = db.query(CourseOutline).filter_by(user_id=1,folder_name='Memory Palace').one()
            row.outline_json = original; row.current_section = 2; db.commit()
        self.assertTrue(ensure_curated_outline(1, 'Memory Palace'))
        with patch('curated_config.is_curated_content_ready', return_value=True), \
             patch.object(lesson, 'get_section_mastery_list', return_value=[]):
            state = lesson.get_lesson_state(1, 'Memory Palace', source_user_id=0)
        self.assertEqual(state['current_section'], 2)
        self.assertEqual(state['sections'][2]['title'], 'Building & Encoding')
        self.assertEqual(state['sections'][2]['workshop']['title'], 'Build your five-stop route')
        with SessionLocal() as db:
            self.assertEqual(db.query(CourseOutline).filter_by(user_id=1,folder_name='Memory Palace').one().outline_json, original)

    def test_prior_work_is_scoped_to_student_lesson_epoch_and_earlier_sections(self):
        with SessionLocal() as db:
            def add(uid, folder, section, text):
                row=ChatMessage(user_id=uid, conversation_id='test', role='user', content=text,
                                context_type='lesson', context_id=folder, section_index=section)
                db.add(row);db.flush();return row.id
            old=add(1,'Memory Palace',0,'OLD RUN')
            db.add(CourseChatEpoch(user_id=1,folder_name='Memory Palace',through_message_id=old))
            add(1,'Memory Palace',0,'My route starts at the blue door')
            add(2,'Memory Palace',0,'OTHER STUDENT')
            add(1,'Physics',0,'OTHER COURSE')
            add(1,'Memory Palace',4,'FUTURE SECTION')
            db.commit()
            text = prior_work(db,1,'Memory Palace',2)
        self.assertIn('blue door', text)
        for forbidden in ('OLD RUN','OTHER STUDENT','OTHER COURSE','FUTURE SECTION'):
            self.assertNotIn(forbidden, text)
        self.assertIn('blue door', self.prompt(2))

    def test_workshops_cannot_be_skipped(self):
        # Workshops are built milestone by milestone; placement is refused server-side.
        import placement
        with self.assertRaises(ValueError):
            placement.begin(1, 'Memory Palace', 3, 'conv_workshop_skip')

    def test_server_still_rejects_unverified_advance(self):
        result=lesson.advance_section(1,'Memory Palace')
        self.assertIn('error',result)
        with SessionLocal() as db:
            self.assertEqual(db.query(CourseOutline).filter_by(user_id=1,folder_name='Memory Palace').one().current_section,0)


if __name__ == '__main__':
    unittest.main()
