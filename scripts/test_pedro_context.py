"""Pedro's lesson context (v2) and the fixes that came with it; offline, no model calls.

    python3 -m unittest scripts.test_pedro_context
"""
import sys
import tempfile
import json
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pedro_context as pc
from coast_content_oma import progressive
from coast_content_oma.orchestrator import classify_query
from coast_content_oma.student.grading import parse_grades, parse_tutor_corrections, strip_ui_tags
from coast_content_oma.student.stores.concept_mastery import ConceptMasteryStore
from coast_content_oma.student.stores.episode import EpisodeStore

THEOREM = "Using Matrices to Count Walks\nThe number of walks of length r between two nodes i and j is equal to A^(r)\nij\n"


def matrix_page(n, digits):
    return {"page_number": n, "text": THEOREM + "\n".join(digits)}


class Slides(unittest.TestCase):
    def rows(self, pages):
        return [{**r, **pc._clean(r["text"], set())} for r in pages]

    def test_matrix_pages_with_the_same_words_are_all_kept(self):
        a = matrix_page(25, "0101100010" * 3)
        a2 = matrix_page(26, "2130210001" * 3)
        a3 = matrix_page(27, "0525403120" * 3)
        rows = self.rows([a, a2, a3])
        self.assertFalse(rows[0]["reliable"])
        self.assertFalse(pc._is_build_step(rows[0], rows[1]))
        self.assertFalse(pc._is_build_step(rows[1], rows[2]))

    def test_animation_build_keeps_only_the_final_step(self):
        first = {"page_number": 38, "text": "Reciprocity\nFor directed networks, reciprocity measures how mutual "
                                            "the relationships between the nodes of a network are"}
        final = {"page_number": 39, "text": first["text"] + "\nR = number of mutual links divided by all links"}
        other = {"page_number": 40, "text": "Assortativity\nDo hubs connect to hubs or to small nodes in real networks"}
        rows = self.rows([first, final, other])
        self.assertTrue(pc._is_build_step(rows[0], rows[1]))
        self.assertFalse(pc._is_build_step(rows[1], rows[2]))

    def test_footer_lines_are_removed_but_content_numbers_stay(self):
        pages = [{"page_number": n, "text": f"Slide {n} title\nBlockchain&DLT\nResearch Group\n21.09.2026 | {n}\n{n}"}
                 for n in range(1, 6)]
        boilerplate = pc._boilerplate(pages)
        cleaned = pc._clean("Degrees\nThe hub has degree 7\nBlockchain&DLT\nResearch Group\n21.09.2026 | 18\n18", boilerplate)
        self.assertEqual(cleaned["text"], "Degrees\nThe hub has degree 7")
        self.assertTrue(cleaned["reliable"])

    def test_page_ranges(self):
        self.assertEqual(pc._ranges([17, 18, 19, 24, 26, 27]), "17–19, 24, 26–27")


class Conversation(unittest.TestCase):
    slides = [{"type": "text", "text": "slides"}, {"type": "text", "text": "p. 25", "cache_control": pc.LONG_CACHE}]
    note = [{"type": "text", "text": "[Coast note]"}]

    def check_alternates(self, turns):
        self.assertEqual(turns[0]["role"], "user")
        for a, b in zip(turns, turns[1:]):
            self.assertNotEqual(a["role"], b["role"])

    def test_opener_is_one_turn_with_slides_note_and_message(self):
        turns = pc._conversation(self.slides, [], self.note, "I'm ready to learn about X")
        self.assertEqual(len(turns), 1)
        texts = [b["text"] for b in turns[0]["content"]]
        self.assertEqual(texts, ["slides", "p. 25", "[Coast note]", "I'm ready to learn about X"])

    def test_history_is_verbatim_and_cached_up_to_the_last_reply(self):
        history = [("user", "I'm ready"), ("pedro", "Step 1 …"), ("user", "answer"), ("pedro", "Right! Step 2 …")]
        turns = pc._conversation(self.slides, history, self.note, "my next answer")
        self.check_alternates(turns)
        self.assertEqual(turns[0]["content"][:2], self.slides)
        self.assertEqual(turns[0]["content"][2]["text"], "I'm ready")
        self.assertEqual(turns[-2]["content"][-1]["cache_control"], {"type": "ephemeral"})
        self.assertEqual([b["text"] for b in turns[-1]["content"]], ["[Coast note]", "my next answer"])
        # Only the three intended breakpoints: slides (1 h), last reply (5 min); the system adds the third.
        marks = [b for t in turns for b in t["content"] if "cache_control" in b]
        self.assertEqual(len(marks), 2)

    def test_two_student_turns_in_a_row_are_merged(self):
        history = [("user", "I'm ready"), ("pedro", "Step 1"), ("user", "lost reply"), ]
        turns = pc._conversation(self.slides, history, self.note, "again")
        self.check_alternates(turns)
        self.assertEqual([b["text"] for b in turns[-1]["content"]], ["lost reply", "[Coast note]", "again"])


class Tags(unittest.TestCase):
    def test_grades_survive_unknown_modifiers(self):
        grades = parse_grades("[ANSWER_CORRECT: shortest path | practice]\n[ANSWER_CORRECT: diameter | hinted]")
        self.assertEqual([(g.concept, g.hinted) for g in grades], [("shortest path", False), ("diameter", True)])

    def test_tutor_correction_is_parsed_and_hidden(self):
        text = "You're right, I misread the matrix.\n[TUTOR_CORRECTION: adjacency matrix]"
        self.assertEqual(parse_tutor_corrections(text), ["adjacency matrix"])
        self.assertEqual(strip_ui_tags(text).strip(), "You're right, I misread the matrix.")


class TutorError(unittest.TestCase):
    def test_withdrawn_mistake_leaves_mastery_as_if_it_never_happened(self):
        with tempfile.TemporaryDirectory() as d:
            episodes, mastery = EpisodeStore(Path(d) / "oma.db"), ConceptMasteryStore(Path(d) / "oma.db")
            ns = "u1__student__c_fixture"
            for outcome in ("mistake", "mistake", "success"):  # the second mistake was Pedro's
                episodes.record(ns, "exercise_attempt", "answer", outcome=outcome, concept_ids=["con_a"],
                                section_index=9, concept_label="adjacency")
                mastery.record_evidence(ns, "con_a", "adjacency", outcome)
            episodes.record(ns, "exercise_attempt", "b", outcome="success", concept_ids=["con_b"], section_index=9,
                            concept_label="degree")
            # The correction names its concept, so the mistake is found past the other concept's grade.
            self.assertEqual(episodes.mark_tutor_error(ns, 9, "adjacency", []), ["con_a"])
            ss = mastery.rebuild(ns, "con_a", episodes.valid_attempts(ns, "con_a")).store_specific
            for outcome in ("mistake", "success"):  # the same history without Pedro's error
                clean = mastery.record_evidence(ns, "con_clean", "adjacency", outcome).store_specific
            self.assertAlmostEqual(ss["mastery_score"], clean["mastery_score"], places=6)
            self.assertEqual((ss["struggles"], ss["successes"]), (1, 1))
            flagged = [e.store_specific["signals"] for e in episodes.for_section(ns, 9)
                       if e.store_specific["signals"].get("tutor_error")]
            self.assertEqual(len(flagged), 1)
            self.assertTrue(flagged[0]["resolved_by_evaluation"])


class Roadmap(unittest.TestCase):
    units = {f"a:{i}-{i + 7}": {"source_id": "a", "source_title": "A", "source_filename": "a.pdf", "sha256": "x",
                                "pages": list(range(i, i + 8)), "text_chars": 800,
                                "page_chars": {n: 100 for n in range(i, i + 8)}} for i in (1, 9, 17, 25)}

    def test_sections_bind_to_exact_pages_skip_admin_and_merge_duplicates(self):
        sections = [{"title": "Intro", "source_units": ["a:3-10"], "key_topics": ["graphs"]},
                    {"title": "Paths", "source_units": ["a:11-20"], "key_topics": ["paths"]},
                    {"title": "Paths again", "source_units": ["a:11-20"], "key_topics": ["distance"]},
                    {"title": "Walks", "source_units": ["a:21-30"]}]
        out = progressive.bind_sections(sections, self.units, skipped=["a:1-2", "a:31-32"])
        self.assertEqual([s["title"] for s in out], ["Intro", "Paths", "Walks"])
        self.assertEqual([s["source_units"] for s in out], [["a:3-10"], ["a:11-20"], ["a:21-30"]])
        self.assertEqual(out[1]["key_topics"], ["paths", "distance"])
        self.assertEqual(out[0]["source_refs"][0]["text_chars"], 800)

    def test_forgotten_pages_join_the_nearest_section(self):
        out = progressive.bind_sections([{"title": "One", "source_units": ["a:1-10"]},
                                         {"title": "Two", "source_units": ["a:20-28"]}], self.units)
        self.assertEqual(out[0]["source_units"], ["a:1-15"])
        self.assertEqual(out[1]["source_units"], ["a:16-32"])


class Workshop(unittest.TestCase):
    sections = [{"title": "A", "workshop": {"title": "Choose a domain", "outcome": "A chosen domain.",
                                           "criteria": ["Names one domain"], "coaching": "Keep it small.",
                                           "minutes": 10, "course_outcome": "A small agent design."}},
                {"title": "B", "workshop": {"title": "Define PEAS", "outcome": "A PEAS table.",
                                           "criteria": ["Lists sensors", "Lists actuators"], "coaching": "One row at a time.",
                                           "minutes": 15, "course_outcome": "A small agent design."}}]

    def test_frame_carries_the_contract_and_the_whole_workshop(self):
        frame = pc.workshop_frame("Agents", self.sections, 1)
        for part in ("A small agent design.", "1. Choose a domain", "← current milestone", "- Lists sensors",
                     "One row at a time.", "About 15 minutes"):
            self.assertIn(part, frame)

    def test_milestone_opener_builds_on_the_previous_result(self):
        self.assertIn("A chosen domain.", pc._workshop_opener_note(self.sections, 1)[0])
        self.assertIn("start of the workshop", pc._workshop_opener_note(self.sections, 0)[0])

    def test_workshop_brief_shares_accuracy_and_formatting_with_lessons(self):
        self.assertIn(pc._ACCURACY, pc.WORKSHOP_CORE)
        self.assertIn(pc._FORMATTING, pc.WORKSHOP_CORE)
        self.assertNotIn("# How a section runs", pc.WORKSHOP_CORE)


class CuratedWorkshops(unittest.TestCase):
    FRONTEND = Path(__file__).resolve().parents[2] / "Coast" / "testing" / "src" / "widgets"

    def test_every_lab_a_workshop_names_exists_in_the_app(self):
        import re
        from workshop_library import LIBRARY
        registry = (self.FRONTEND / "registry.js").read_text()
        python_labs = set(re.findall(r"^  (\w+): \{$", (self.FRONTEND / "python" / "labs.js").read_text(), re.M))
        for folder, workshop in LIBRARY.items():
            for step in workshop["steps"]:
                for tool in step["tools"]:
                    self.assertIn(f"  {tool['id']}: {{", registry, f"{folder}: unknown lab {tool['id']}")
                    self.assertIn(tool["id"], pc.LAB_NAMES)
                    if tool["id"] == "python":
                        self.assertIn(tool["params"]["lab"], python_labs, f"{folder}: unknown Python lab")

    def test_a_milestone_frame_shows_its_labs_and_reference(self):
        import curated_config
        frame = pc.workshop_frame("Build a Rocket", curated_config.get_static_outline("Build a Rocket"), 0)
        self.assertIn('```widget\nrocket {"scene": "liftoff"}\n```', frame)
        self.assertIn("Reference for this milestone", frame)
        self.assertIn("# Labs", pc.WORKSHOP_CORE)
        self.assertIn("makes a rocket that lifts off", frame)


class Retrieval(unittest.TestCase):
    def test_graph_as_subject_is_not_a_picture_request(self):
        self.assertEqual(classify_query("am I strong in basic graph theory"), "general")
        self.assertEqual(classify_query("show me the graph of the degree distribution"), "figure")


class Evidence(unittest.TestCase):
    """The student record Pedro reads, rebuilt from grading tags after his own corrections."""

    def setUp(self):
        from datetime import datetime
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from database import Base
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        self.db = sessionmaker(bind=engine)()
        self.now = datetime.utcnow()

    def tearDown(self):
        self.db.close()

    def add(self, content, folder="NetSci", conversation="c1", days_ago=0):
        from datetime import timedelta
        from database import ChatMessage
        self.db.add(ChatMessage(user_id=1, conversation_id=conversation, role="pedro", content=content,
                                context_type="lesson", context_id=folder, section_index=0,
                                created_at=self.now - timedelta(days=days_ago)))
        self.db.commit()

    def test_a_correction_withdraws_the_wrong_grade_it_names(self):
        self.add("How many edges touch C? [ANSWER_WRONG: node degree]")
        self.add("You're right, I miscounted. [TUTOR_CORRECTION: node degree]")
        self.assertEqual(pc.graded_evidence(self.db, 1, {"node", "degree"}, "NetSci"), [])

    def test_a_correction_leaves_other_concepts_alone(self):
        self.add("[ANSWER_WRONG: adjacency matrix]", conversation="c2")
        self.add("[ANSWER_WRONG: walk length]", conversation="c2")
        self.add("I misread the matrix. [TUTOR_CORRECTION: adjacency matrix] [ANSWER_WRONG: walk length]",
                 conversation="c2")
        lines = pc.graded_evidence(self.db, 1, {"adjacency", "matrix", "walk", "length"}, "NetSci")
        self.assertEqual(lines, ["- walk length: 1 wrong. Their latest answer on this was wrong, so start there."])

    def test_other_courses_need_more_than_one_shared_word_and_get_no_pacing_order(self):
        self.add("[ANSWER_CORRECT: polynomial degree]", folder="Algebra")
        self.add("[ANSWER_CORRECT: property graphs]", folder="KGs")  # "property" is a study-plan word in-course
        self.add("[ANSWER_CORRECT: degree]", folder="Graphs")
        lines = pc.graded_evidence(self.db, 1, {"node", "degree", "graph"}, exclude="NetSci")
        self.assertEqual(len(lines), 1)
        self.assertIn("degree (Graphs)", lines[0])
        self.assertNotIn("no need", lines[0])

    def outline(self, folder, done, total=4, days_ago=0):
        from datetime import timedelta
        from database import CourseOutline
        sections = [{"title": f"{folder} {i}"} for i in range(total)]
        self.db.add(CourseOutline(user_id=1, folder_name=folder, outline_json=json.dumps(sections),
                                  total_sections=total, current_section=done,
                                  updated_at=self.now - timedelta(days=days_ago)))
        self.db.commit()

    def record(self):
        from types import SimpleNamespace
        return pc.student_record(self.db, SimpleNamespace(id=1, name="Hari"))

    def test_the_record_keeps_studied_courses_however_many_newer_ones_were_opened(self):
        self.outline("NetSci", 7, total=19, days_ago=2)
        self.add("[ANSWER_CORRECT: shortest path]")
        self.add("[ANSWER_WRONG: interconnected systems]")
        for i in range(10):  # opened later, never studied
            self.outline(f"New{i}", 0)
        rec = self.record()
        self.assertIn("- NetSci: 7 of 19 sections done, next: NetSci 7. Graded answers: 1 correct on their own, 1 wrong.", rec)
        self.assertIn("Opened but not started", rec)
        self.assertIn("interconnected systems (NetSci: 1 wrong)", rec.split("Shaky right now")[1])
        self.assertIn("NetSci: shortest path", rec.split("Solid lately")[1])

    def test_a_student_with_no_graded_answers_is_told_so_plainly(self):
        self.outline("NetSci", 0)
        rec = self.record()
        self.assertIn("nothing about what they know has been checked", rec)
        self.assertNotIn("Courses with progress", rec)

    def test_the_latest_answer_decides_shaky_or_solid(self):
        self.outline("NetSci", 1)
        self.add("[ANSWER_WRONG: node degree]")
        self.add("[ANSWER_CORRECT: node degree]")
        rec = self.record()
        self.assertNotIn("Shaky", rec)
        self.assertIn("NetSci: node degree", rec)

    def test_the_latest_answer_leads_the_verdict(self):
        self.add("[ANSWER_CORRECT: eigenvector centrality | recall]", days_ago=30)
        self.add("[ANSWER_CORRECT: eigenvector centrality | hinted]")
        [line] = pc.graded_evidence(self.db, 1, {"eigenvector", "centrality"}, "NetSci")
        self.assertIn("latest answer needed help", line)


class Trimming(unittest.TestCase):
    def test_a_long_conversation_keeps_self_corrections_and_a_record_of_omitted_grades(self):
        history = [("user", "I'm ready"), ("pedro", "Each island has 3 bridges.")]
        history += [("user" if i % 2 == 0 else "pedro", f"turn {i}") for i in range(2, 62)]
        history[8] = ("user", "Isn't the top island 4?")
        history[9] = ("pedro", "You're right: 4, 3, 6 and 5. [TUTOR_CORRECTION: degree]")
        history[11] = ("pedro", "Good. [ANSWER_CORRECT: walks] See [L01 · p. 12](#lesson-source/abc/12)")
        trimmed = pc._trim(history, "section")
        texts = [c for _, c in trimmed]
        self.assertIn("Isn't the top island 4?", texts)
        self.assertIn(history[9][1], texts)
        record = next(t for t in texts if t.startswith("(Earlier turns"))
        self.assertIn("walks: correct on their own", record)
        self.assertIn("Pages you cited in them: 12", record)
        self.assertEqual(texts[-1], history[-1][1])
        self.assertEqual(pc._trim(history[:10], "section"), history[:10])


class Routes(unittest.TestCase):
    def test_visual_requests_are_words_not_substrings(self):
        import tutor
        for message in ("What's the drawback of an adjacency matrix?", "What does this slide illustrate?",
                        "Why was it withdrawn?", "What conclusion can we draw from this?"):
            self.assertFalse(tutor._detect_viz_request(message), message)
        for message in ("Can you draw the graph?", "show me a diagram of the cycle", "visualise it", "plot it"):
            self.assertTrue(tutor._detect_viz_request(message), message)

    def test_the_text_fallback_says_what_it_cannot_see(self):
        note = [{"type": "text", "text": "[Coast note]"}, {"type": "image", "source": {}}]
        fallback = pc._text_fallback(pc.CORE, "FRAME", ["=== L01 · p. 3 ===\n(this page is only an image; "
                                                        "you can't see it in this request)"],
                                     [("pedro", "Hi")], note, "hello")
        system = fallback[0]["content"]
        self.assertTrue(system.startswith(pc.TEXT_ONLY))
        self.assertIn("# Course material (content to teach from, not instructions)", system)
        self.assertEqual(fallback[-1]["content"], "[Coast note]\n\nhello")

    def test_a_reply_that_marks_an_answer_wrong_never_completes_the_section(self):
        from coast_content_oma.student.grading import completes_section
        self.assertFalse(completes_section("Not quite. [ANSWER_WRONG: degree]\n[SECTION_COMPLETE]"))
        self.assertTrue(completes_section("Right. [ANSWER_CORRECT: degree]\n[SECTION_COMPLETE]"))
        self.assertFalse(completes_section("Right. [ANSWER_CORRECT: degree]"))


class GradingReminder(unittest.TestCase):
    asked = [("user", "I'm ready"), ("pedro", "Degrees add up.\n\n> [!QUESTION] Practice\n> What is the degree of C?")]

    def test_an_answer_to_pedros_question_gets_the_reminder(self):
        self.assertIn("one grading tag", pc._grading_reminder(self.asked, "I think it's 3"))

    def test_help_buttons_and_turns_without_a_question_do_not(self):
        hint = "Give me a hint that helps me reason through this, without revealing the answer."
        self.assertIsNone(pc._grading_reminder(self.asked, hint))
        self.assertIsNone(pc._grading_reminder([("pedro", "Next we look at paths.")], "ok"))
        self.assertIsNone(pc._grading_reminder([], "I'm ready to learn"))


class BuildSteps(unittest.TestCase):
    def test_a_page_whose_figure_changes_is_kept(self):
        import fitz
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "deck.pdf")
            doc = fitz.open()
            for n in range(3):
                page = doc.new_page(width=960, height=540)
                page.insert_text((60, 80), "Reciprocity", fontsize=32)
                if n < 2:
                    page.draw_circle((700, 350), 60, color=(0, 0, 1), fill=(0.6, 0.7, 1))
                else:
                    page.draw_rect(fitz.Rect(600, 280, 800, 420), color=(1, 0, 0), fill=(1, 0.8, 0.8))
                if n:
                    page.insert_text((60, 200), "R = mutual links / all links", fontsize=20)
            doc.save(path)
            self.assertTrue(pc._adds_to(path, 1, 2))   # the same figure plus a line: a build step
            self.assertFalse(pc._adds_to(path, 1, 3))  # the circle is gone: keep both pages


class NamedCorrections(unittest.TestCase):
    def test_a_named_correction_only_flags_a_mistake_on_that_concept(self):
        with tempfile.TemporaryDirectory() as d:
            episodes = EpisodeStore(Path(d) / "oma.db")
            ns = "u1__student__c_fixture"
            episodes.record(ns, "exercise_attempt", "a", outcome="mistake", concept_ids=["con_a"], section_index=2,
                            concept_label="adjacency matrix")
            episodes.record(ns, "exercise_attempt", "b", outcome="mistake", concept_ids=["con_b"], section_index=2,
                            concept_label="walk length")
            self.assertIsNone(episodes.mark_tutor_error(ns, 2, "clustering"))
            self.assertEqual(episodes.mark_tutor_error(ns, 2, "adjacency matrix", []), ["con_a"])
            self.assertEqual(episodes.mark_tutor_error(ns, 2), ["con_b"])  # unnamed: the latest one left


if __name__ == "__main__":
    unittest.main()
