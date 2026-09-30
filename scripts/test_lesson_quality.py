#!/usr/bin/env python3
"""Offline regressions for source fidelity and bounded student evidence."""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fitz
from PIL import Image, ImageDraw, ImageStat
from coast_content_oma.extraction import extract_pages, _extract_images_pymupdf
from coast_content_oma.normalized_source import save_pages, load_pages, cache_dir
from coast_content_oma.student.prompt_budget import course_profile
from coast_content_oma.student.mastery_aggregate import aggregate_mastery_rows


class LessonQuality(unittest.TestCase):
    def test_superscripts_and_subscripts_survive_extraction_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'math.pdf'
            doc = fitz.open(); page = doc.new_page()
            page.insert_text((40, 80), 'Configurations: 3', fontsize=18)
            x = 40 + fitz.get_text_length('Configurations: 3', fontsize=18)
            page.insert_text((x, 74), '150', fontsize=11)
            page.insert_text((40, 120), 'Q', fontsize=18)
            page.insert_text((54, 125), 'next', fontsize=11)
            doc.save(path); doc.close()
            pages = extract_pages(path, use_cache=False)
            self.assertIn('3^(150)', pages[0]['text'])
            self.assertIn('Q_(next)', pages[0]['text'])
            save_pages(path, pages)
            self.assertEqual(load_pages(path)[0]['text'], pages[0]['text'])
            manifest = cache_dir(path) / 'manifest.json'
            data = json.loads(manifest.read_text()); data['version'] = 1
            manifest.write_text(json.dumps(data))
            self.assertIsNone(load_pages(path))

    def test_soft_mask_preserves_visible_foreground(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'mask.pdf'
            original = Image.new('RGBA', (240, 140), (0, 0, 0, 0))
            ImageDraw.Draw(original).rectangle((20, 20, 100, 100), fill=(0, 0, 0, 255))
            stream = io.BytesIO(); original.save(stream, 'PNG')
            doc = fitz.open(); page = doc.new_page(width=400, height=300)
            page.insert_image(fitz.Rect(40, 40, 280, 180), stream=stream.getvalue())
            doc.save(path); doc.close()
            images = _extract_images_pymupdf(path)[0]
            self.assertEqual(len(images), 1)
            pixels = images[0]['pil_image']
            self.assertGreater(min(ImageStat.Stat(pixels).stddev), 30)
            self.assertEqual(pixels.getpixel((0, 0)), (255, 255, 255))

    def test_vector_diagram_available_but_plain_slide_has_no_extra_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'vectors.pdf'
            doc = fitz.open(); page = doc.new_page(width=600, height=400)
            page.insert_text((20, 30), 'Vector diagram')
            for x in range(40, 500, 100):
                page.draw_rect(fitz.Rect(x, 100, x+60, 180))
                page.draw_line((x, 200), (x+60, 240))
            page = doc.new_page(width=600, height=400)
            page.draw_rect(page.rect, fill=(1, 1, 1))
            page.insert_text((20, 30), 'Plain slide')
            doc.save(path); doc.close()
            pages = _extract_images_pymupdf(path)
            self.assertEqual(len(pages[0]), 1)
            self.assertGreater(max(ImageStat.Stat(pages[0][0]['pil_image']).stddev), 5)
            self.assertLessEqual(pages[0][0]['width'], 1600)
            self.assertEqual(pages[1], [])

    def test_active_state_survives_alias_aggregation(self):
        agg = aggregate_mastery_rows([
            {'concept_name': 'Q-learning', 'mastery_score': .25, 'struggles': 3,
             'last_eval_state': 'ACTIVE_MISCONCEPTION', 'misconception_type': 'Uses the chosen action.'},
            {'mastery_score': .95, 'successes': 6, 'last_eval_state': 'RESOLVED'},
        ], canonical_id='q')
        self.assertEqual(agg['misconception_state'], 'ACTIVE_MISCONCEPTION')
        self.assertEqual(agg['misconception_type'], 'Uses the chosen action.')

    def test_alias_state_keeps_metadata_from_winning_evidence(self):
        agg = aggregate_mastery_rows([
            {'last_eval_state': 'UNDER_OBSERVATION', 'misconception_type': 'Earlier slip',
             'last_eval_section': 2},
            {'last_eval_state': 'ACTIVE_MISCONCEPTION'},
        ], canonical_id='q')
        self.assertEqual(agg['misconception_state'], 'ACTIVE_MISCONCEPTION')
        self.assertNotIn('misconception_type', agg)
        self.assertNotIn('last_eval_section', agg)

    def test_profile_finds_old_relevant_alias_memory_before_capping(self):
        from unittest.mock import patch
        from coast_content_oma.student.orchestrator import build_student_orchestrator
        from coast_content_oma.student.stores import course_namespace
        from coast_content_oma.stores.concept_alias import ConceptAliasStore
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'oma.db'
            orch = build_student_orchestrator(path)
            ns = course_namespace(991, 'Fixture')
            with patch('coast_content_oma.course_identity.content_namespace_for_student', return_value='fixture'):
                orch.mastery.record_evidence(ns, 'canonical', 'Q-learning', 'success')
                for i in range(8):
                    orch.patterns.upsert(ns, 'golden_moment', f'Unrelated memory {i}', .99, 2,
                                         related_concept_ids=[f'other{i}'], dedupe_key=f'other{i}')
                orch.patterns.upsert(ns, 'golden_moment', 'Relevant old analogy', .90, 2,
                                     related_concept_ids=['alias'], dedupe_key='relevant')
                ConceptAliasStore(path).append_merge('fixture', 'alias', 'canonical', reason='test')
                profile = orch.build_profile(991, 'Fixture', ['canonical', 'alias', 'canonical'])
                self.assertEqual(profile['requested_concept_count'], 1)
                self.assertEqual(len(profile['focused_mastery']), 1)
                self.assertEqual(len(profile['golden_moments']), 1)
                self.assertEqual(profile['golden_moments'][0]['concept_ids'], ['canonical'])
                self.assertIn('Relevant old analogy', course_profile(profile))

    def profile(self):
        concepts = [dict(concept_id=str(i), name=f'Other concept {i}', score=.96,
                         confidence=.95, successes=9, last_eval_state='RESOLVED') for i in range(44)]
        concepts[-1].update(concept_id='q', name='Q-learning', score=.25, successes=0,
                            struggles=3, last_eval_state='ACTIVE_MISCONCEPTION')
        return {'focused_mastery': concepts, 'requested_concept_count': 45,
                'current_query': 'Teach Q-learning',
                'identity_traits': [{'text': 'One numerical step at a time.'}],
                'active_context': {'last_unresolved': {'text': 'Uses chosen action instead of max.'}},
                'golden_moments': [{'concept_ids': ['q'], 'text': 'Route planner analogy worked.'}],
                'progress_ledger': {'completed_sections': [{'title': 'x'*1000}]*100}}

    def test_target_misconception_and_preference_survive_large_profile(self):
        block = course_profile(self.profile())
        for text in ('Current concept: Q-learning', 'ACTIVE_MISCONCEPTION', 'One numerical step',
                     'instead of max', 'Route planner', '1 unassessed', 'confidence', 'never invent'):
            self.assertIn(text, block)
        self.assertLessEqual(len(block), 1200)

    def test_advanced_summary_accounts_for_unknown_concepts(self):
        p = self.profile(); p['focused_mastery'][-1].update(score=.96, successes=9, last_eval_state='RESOLVED')
        block = course_profile(p)
        self.assertIn('44 strong/resolved', block)
        self.assertIn('1 unassessed', block)

    def test_explicit_cross_course_recall_and_unrelated_golden_filter(self):
        p = self.profile(); p['current_query'] = 'Remember Probability last semester?'
        p['cross_course_memories'] = [{'folder': 'Probability', 'text': 'Weather forecast update helped.'}]
        p['golden_moments'] = [{'concept_ids': ['unrelated'], 'text': 'Private irrelevant story.'}]
        block = course_profile(p)
        self.assertIn('Weather forecast', block)
        self.assertNotIn('Private irrelevant', block)

    def test_unassessed_target_does_not_receive_unrelated_analogy(self):
        p = {'requested_concept_ids': ['q'], 'requested_concept_count': 1,
             'golden_moments': [{'concept_ids': ['biology'], 'text': 'Mitochondria story'}]}
        self.assertNotIn('Mitochondria', course_profile(p))


if __name__ == '__main__':
    unittest.main()
