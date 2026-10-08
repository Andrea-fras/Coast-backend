#!/usr/bin/env python3
"""A workshop a student makes from their own files is an assignment: the roadmap follows its exercises
(none merged or invented) and Pedro guides them through doing it, never adding tasks of his own."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pedro_context  # noqa: E402
from coast_content_oma import progressive  # noqa: E402
from workshops import decorate_sections, workshop_instructions  # noqa: E402


def unit_pages(source="src_a", pages=(1, 2)):
    return {f"{source}:1-8": {"source_id": source, "source_title": "RL TA01", "pages": list(pages),
                              "page_chars": {p: 1000 for p in pages}, "text_chars": 1000 * len(pages)}}


def milestone(title, outcome, *parts):
    return {"title": title, "source_units": ["src_a:1"], "learning_objectives": list(parts),
            "workshop": {"title": title, "outcome": outcome, "criteria": list(parts), "coaching": "Hint, never answer.",
                         "course_outcome": "The assignment, completed"}}


class AssignmentWorkshops(unittest.TestCase):
    def test_exercises_on_one_page_stay_separate_milestones(self):
        sections = [milestone("Exercise 1 · What does the policy need to know?", "Answers to 1a-1b", "1a", "1b"),
                    milestone("Exercise 2 · Immediate reward or future opportunity?", "Answers to 2a", "2a")]
        page = unit_pages(pages=(1,))  # both exercises are set on page 1
        kept = progressive.bind_sections([dict(s) for s in sections], page, merge_same=False)
        self.assertEqual([s["title"] for s in kept], [s["title"] for s in sections])
        merged = progressive.bind_sections([dict(s) for s in sections], page)  # lessons still merge
        self.assertEqual(len(merged), 1)

    def test_a_students_own_workshop_is_an_assignment_and_a_curated_one_is_not(self):
        own = decorate_sections("Week1 RL", [milestone("Exercise 1", "Answers to 1a", "1a")])
        self.assertEqual(own[0]["workshop"]["kind"], "assignment")
        curated = decorate_sections("Build a Rocket", [milestone("Some step", "A rocket", "It flies")])
        self.assertNotIn("kind", curated[0].get("workshop") or {})

    def test_pedro_states_the_exercise_and_adds_nothing(self):
        sections = decorate_sections("Week1 RL", [
            milestone("Exercise 1 · What does the policy need to know?", "Written answers to 1a and 1b", "1a", "1b"),
            milestone("Exercise 2 · Immediate reward or future opportunity?", "Answers to 2a", "2a")])
        frame = pedro_context.workshop_frame("Week1 RL", sections, 0)
        self.assertIn("# This assignment: Week1 RL", frame)
        self.assertIn("Exercise 1 · What does the policy need to know?   ← current exercise", frame)
        self.assertIn("The brief asks for: Written answers to 1a and 1b", frame)
        opener = pedro_context._workshop_opener_note(sections, 0)[0]
        self.assertIn("state what the first exercise asks, as the brief words it", opener)
        core = pedro_context.ASSIGNMENT_CORE
        self.assertIn("Never add work the brief doesn't ask for", core)
        self.assertIn("Their answers stay theirs", core)
        self.assertNotIn("Pitch it for a beginner", core)  # the project workshops' pitch
        self.assertIn("GUIDED ASSIGNMENT", workshop_instructions(sections[0]["workshop"]))


if __name__ == "__main__":
    unittest.main(verbosity=1)
