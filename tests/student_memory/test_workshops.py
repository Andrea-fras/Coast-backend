"""Workshops: practical, student-built work (memory palace today, 'build a tiny LLM' next).
The student's own work must be remembered like any lesson, and contracts must not be
hard-coded to one course."""
from harness import chat, episodes, make_course, make_student, run_learning_jobs, stub_evaluator, lesson
from workshops import MEMORY_STEPS
import tutor

# The v2 context (pedro_context) states the contract in its frame and the gate in its brief.
V2 = tutor.PEDRO_CONTEXT == "v2"
WORKSHOP = "# This workshop:" if V2 else "GUIDED WORKSHOP"
GATE = "Finish the milestone when every criterion is shown" if V2 else "WORKSHOP COMPLETION GATE"

ITEMS = "my five items: mitochondria, ribosome, Golgi, lysosome, nucleus; my palace is my grandma's flat"


def _memory_palace(uid):
    make_course(uid, "Memory Palace", [{"title": s["title"], "key_topics": ["method of loci"]} for s in MEMORY_STEPS])


def test_memory_palace_contract_is_applied():
    """Regression: today's only workshop still gets its milestone contract and gate."""
    uid = make_student()
    _memory_palace(uid)
    res = chat(uid, "Let's start", "Pick five things to remember.", context_id="Memory Palace", section_index=0)
    assert WORKSHOP in res["system_prompt"]
    assert GATE in res["system_prompt"]


def test_workshop_work_is_recorded():
    """Step 4. What the student builds in a workshop is stored in Student OMA."""
    uid = make_student()
    _memory_palace(uid)
    chat(uid, ITEMS, "Great choices. [ANSWER_CORRECT: method of loci] [SECTION_COMPLETE]",
         context_id="Memory Palace", section_index=0)
    texts = " ".join((e.store_specific or {}).get("user_message", "") + " " + e.content
                     for e in episodes(uid, "Memory Palace"))
    assert "grandma" in texts, "student's workshop work not in Student OMA"


def test_workshop_work_is_recallable_later():
    """Step 5. 'What did I pick for my memory palace?' in general chat finds their actual choices."""
    uid = make_student()
    _memory_palace(uid)
    chat(uid, ITEMS, "Great choices. [SECTION_COMPLETE]", context_id="Memory Palace", section_index=0)
    res = chat(uid, "What did I pick for my memory palace?", "You picked...", context_type="global")
    assert "grandma" in res["system_prompt"]


def test_custom_workshop_contract_is_honoured():
    """Workshops step. A non-Memory-Palace course (e.g. 'Build a tiny LLM') can carry its own
    milestone contract in the outline and gets the same guided workshop + gate."""
    uid = make_student()
    contract = {
        "title": "Tokenize your corpus", "outcome": "A working character-level tokenizer.",
        "criteria": ["Write encode/decode functions", "Show a round-trip on a sample sentence"],
        "coaching": "Let the student write the code; review it.", "minutes": 20, "version": 1,
        "course_outcome": "Train a tiny character-level language model.",
    }
    make_course(uid, "Build a tiny LLM", [{"title": "Tokenize your corpus", "key_topics": ["tokenization"],
                                           "workshop": contract}])
    res = chat(uid, "Let's start", "First, write encode().", context_id="Build a tiny LLM", section_index=0)
    assert WORKSHOP in res["system_prompt"], "custom workshop contract was dropped"
    assert "Write encode/decode functions" in res["system_prompt"]


def test_milestone_work_is_saved_and_built_on():
    """What the student built in a finished milestone is saved and shown to Pedro next."""
    uid = make_student()
    _memory_palace(uid)
    stub_evaluator({"method of loci": "mastered"},
                   work_summary="Chose mitochondria, ribosome, Golgi, lysosome, nucleus; palace = grandma's flat.")
    chat(uid, ITEMS, "Great choices. [ANSWER_CORRECT: method of loci] [SECTION_COMPLETE]",
         context_id="Memory Palace", section_index=0)
    run_learning_jobs()
    lesson.advance_section(uid, "Memory Palace")
    prompt = chat(uid, "Next milestone please", "Let's make your first scene.",
                  context_id="Memory Palace", section_index=1)["system_prompt"]
    assert "what the student built" in prompt and "grandma's flat" in prompt, "saved work not offered to Pedro"


def test_generated_workshop_contracts_are_validated():
    """Generated contracts are kept when complete and dropped when malformed."""
    from workshops import decorate_sections, is_workshop
    good = {"title": "Tokenize", "outcome": "A tokenizer", "criteria": ["encode/decode round-trip"]}
    sections = decorate_sections("My LLM course", [{"title": "Tokenize", "workshop": good},
                                                   {"title": "Train", "workshop": {"title": "Train"}}])
    assert sections[0]["workshop"]["criteria"] == ["encode/decode round-trip"]
    assert "workshop" not in sections[1], "a contract without outcome/criteria must be dropped"
    assert not is_workshop(sections)


def test_workshop_roadmaps_always_have_complete_contracts():
    """Workshop format: every milestone gets a usable contract (rebuilt if the planner forgot);
    lesson format: no stray contracts."""
    from lesson import _apply_course_format
    planned = [
        {"title": "Tokenize", "learning_objectives": ["Write encode and decode"],
         "workshop": {"outcome": "A tokenizer", "criteria": ["Round-trip a sentence"], "course_outcome": "A tiny LLM"}},
        {"title": "Train", "learning_objectives": ["Run a training loop", "Plot the loss"]},
    ]
    workshop = _apply_course_format([dict(s) for s in planned], "workshop")
    assert all(s["workshop"]["criteria"] for s in workshop)
    assert workshop[0]["workshop"]["title"] == "Tokenize"
    assert workshop[1]["workshop"]["course_outcome"] == "A tiny LLM", "course outcome not shared"
    lesson_format = _apply_course_format([dict(s) for s in planned], "lesson")
    assert not any("workshop" in s for s in lesson_format)


def _workshop_folder(uid, name):
    from harness import SessionLocal
    from database import StudyFolder
    with SessionLocal() as db:
        db.add(StudyFolder(user_id=uid, name=name, kind="workshop"))
        db.commit()


def test_workshop_folder_is_a_workshop_before_its_roadmap():
    """Created from the Workshops page, a course is a workshop from the start."""
    uid = make_student()
    _workshop_folder(uid, "Tiny LLM")
    assert lesson.folder_kind(uid, "Tiny LLM") == "workshop"
    assert lesson.get_lesson_state(uid, "Tiny LLM").get("format") == "workshop"
    assert lesson.folder_kind(uid, "Some lecture course") == "lesson"


def test_generated_workshops_cannot_be_skipped():
    """No placement checks in workshops — refused by the server, not just hidden in the UI."""
    import placement
    uid = make_student()
    contract = {"title": "Tokenize", "outcome": "A tokenizer", "criteria": ["Round-trip a sentence"]}
    make_course(uid, "Tiny LLM", [{"title": f"Milestone {i}", "key_topics": ["tokenization"],
                                   "workshop": {**contract, "title": f"Milestone {i}"}} for i in range(3)])
    try:
        placement.begin(uid, "Tiny LLM", 2, "conv_skip")
    except ValueError as exc:
        assert "no skipping" in str(exc)
    else:
        raise AssertionError("placement started for a workshop")
