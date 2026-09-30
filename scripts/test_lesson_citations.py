#!/usr/bin/env python3
"""Lesson reference integration against a temporary DB; no provider requests."""
import unittest
from unittest.mock import patch

import test_source_chat as fixture
from database import SessionLocal, FolderSource
from lesson_sources import source_catalog, citation_instructions
import lesson


class LessonCitations(unittest.TestCase):
    def setUp(self):
        fixture.SourceChatTests.setUp(self)

    def test_catalogue_is_owned_folder_metadata_only(self):
        with SessionLocal() as db:
            db.add(FolderSource(user_id=1, folder_name='Other', source_id='elsewhere',
                               title='Other lecture', filename='other.pdf', source_type='pdf', page_count=3,
                               raw_text='Other folder text'))
            db.commit()
            refs = source_catalog(db, 1, 'Physics')
        self.assertEqual([r['source_id'] for r in refs], ['physics'])
        self.assertEqual(refs[0]['page_count'], 3)
        self.assertNotIn('raw_text', refs[0])
        self.assertNotIn('file_path', refs[0])

    def test_lesson_response_contains_clickable_targets(self):
        with patch('oma_provider.is_oma_enabled', return_value=False), \
             patch.object(lesson, 'get_section_mastery_list', return_value=[]):
            result = fixture.client.get('/api/folders/Physics/lesson', headers=self.h)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()['source_references'][0]['source_id'], 'physics')
        self.assertEqual(result.json()['source_references'][0]['filename'], 'lecture.pdf')

    def test_prompt_uses_same_stable_ids_without_inventing_pages(self):
        with patch.object(lesson, '_fetch_section_material', return_value=('Source: Mechanics | page 2\nF = ma', True)), \
             patch.object(lesson, 'get_verification_prompt_block', return_value=''), \
             patch.object(lesson, '_build_section_opening_block', return_value=''):
            prompt = lesson.build_lesson_prompt(1, 'Physics')
        self.assertIn('#lesson-source/SOURCE_ID/N', prompt)
        self.assertIn('"source_id": "physics"', prompt)
        self.assertNotIn('"source_id": "private"', prompt)
        self.assertIn('Only cite pages actually present', prompt)
        self.assertIn('ONE answer', prompt)

    def test_premade_teaching_keeps_existing_attributions(self):
        with patch.object(lesson, '_fetch_section_material', return_value=('Shared material', True)), \
             patch.object(lesson, 'get_verification_prompt_block', return_value=''), \
             patch.object(lesson, '_build_section_opening_block', return_value=''):
            prompt = lesson.build_lesson_prompt(1, 'Physics', source_user_id=0, structure={'pedagogy': 'Workshop'})
        self.assertNotIn('CLICKABLE LECTURE REFERENCES', prompt)
        self.assertEqual(citation_instructions([]), '')

    def test_preview_stays_owned_and_page_bounded(self):
        base = '/api/folders/Physics/sources/physics/pages/'
        self.assertEqual(fixture.client.get(base + '2', headers=self.h).headers['content-type'], 'image/png')
        self.assertEqual(fixture.client.get(base + '4', headers=self.h).status_code, 404)
        self.assertEqual(fixture.client.get(base + '2', headers=self.other).status_code, 404)
        self.assertEqual(fixture.client.get('/api/folders/Other/sources/physics/pages/2', headers=self.h).status_code, 404)


if __name__ == '__main__':
    unittest.main()
