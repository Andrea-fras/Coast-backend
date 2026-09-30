"""Regression cases for grouped source markers and verified destinations."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source_citations import normalize_citations, strip_citations


class SourceCitations(unittest.TestCase):
    def test_group_spellings(self):
        for marker in ('[[S6], [S7]]', '[[S6, S7]]', '[S6, S7]', '[[ S6 ], [ S7 ]]', '[[S6]; [S7]]'):
            with self.subTest(marker=marker):
                answer, used = normalize_citations('Before '+marker+'. After', {'S6', 'S7'})
                self.assertEqual(answer, 'Before [[S6]] [[S7]]. After')
                self.assertEqual(used, {'S6', 'S7'})

    def test_known_markers_and_duplicate_group(self):
        answer, used = normalize_citations('**[[S6]]** then [[S6], [S6], [S7]].', {'S6','S7'})
        self.assertEqual(answer, '**[[S6]]** then [[S6]] [[S7]].')
        self.assertEqual(used, {'S6','S7'})

    def test_unknown_ids_never_acquire_a_target(self):
        answer, used = normalize_citations('Evidence [[S6], [S999]].', {'S6'})
        self.assertEqual(answer, 'Evidence [[S6]].')
        self.assertEqual(used, {'S6'})
        self.assertEqual(normalize_citations('[[S999]]', {'S6'}), ('',set()))

    def test_code_and_math_stay_literal(self):
        for text in ('`[[S6], [S7]]`', '```text\n[[S6], [S7]]\n```',
                     '~~~text\n[[S6], [S7]]\n~~~', '$[S6]$', '$$[S6]$$'):
            with self.subTest(text=text):
                self.assertEqual(normalize_citations(text, {'S6','S7'}), (text,set()))

    def test_old_history_ids_are_stripped_for_new_retrieval(self):
        self.assertEqual(strip_citations('Old [[S6], [S7]]. Example `[[S6]]`.'), 'Old . Example `[[S6]]`.')


if __name__ == '__main__': unittest.main()
