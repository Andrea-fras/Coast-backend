"""Found by the 10-lesson eval: Pedro must *act* on what he knows when a new course
starts — the student's stated learning style (in their own words) and concrete
links to related earlier courses — without forcing links that aren't there."""
from harness import chat, make_course, make_student, seed_answers

SAID = "Could you show me some examples of each type first? I learn better that way."


def test_learning_style_reaches_next_course_in_students_words():
    uid = make_student()
    make_course(uid, "What is Intelligence", [{"title": "Kinds of intelligence", "key_topics": ["intelligence"]}])
    make_course(uid, "Intelligent Agents", [{"title": "Agents and environments", "key_topics": ["intelligent agent"]}])
    chat(uid, SAID, "Sure! [REMEMBER: learning_style: prefers worked examples before theory]",
         context_id="What is Intelligence", section_index=0)
    prompt = chat(uid, "Let's start", "OK", context_id="Intelligent Agents", section_index=0)["system_prompt"]
    assert "HOW TO TEACH THIS STUDENT" in prompt, "learning style not turned into a teaching instruction"
    assert SAID[:40] in prompt, "student's own words not carried with the trait"


def test_new_course_links_to_related_earlier_course():
    uid = make_student()
    ids = make_course(uid, "Markov Chains", [{"title": "Steady state", "key_topics": ["markov chains"]}])
    seed_answers(uid, "Markov Chains", "markov chains", ids["markov chains"], [True, True, True])
    make_course(uid, "Queueing Theory", [{"title": "Birth-death queues", "key_topics": ["markov chains", "arrival rate"]}])
    prompt = chat(uid, "Let's start", "OK", context_id="Queueing Theory", section_index=0)["system_prompt"]
    assert "Builds on Markov Chains" in prompt, "related earlier course not offered to Pedro"


def test_unrelated_course_gets_no_forced_link():
    uid = make_student()
    ids = make_course(uid, "Markov Chains", [{"title": "Steady state", "key_topics": ["markov chains"]}])
    seed_answers(uid, "Markov Chains", "markov chains", ids["markov chains"], [True, True, True])
    make_course(uid, "Driving Theory", [{"title": "Right of way", "key_topics": ["priority rules", "roundabouts"]}])
    prompt = chat(uid, "Let's start", "OK", context_id="Driving Theory", section_index=0)["system_prompt"]
    assert "Builds on Markov" not in prompt, "forced a link to an unrelated course"
