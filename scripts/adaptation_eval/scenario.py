"""The 10-lesson adaptation scenario: who the student is, which courses they take,
and the moments planted along the way that Pedro should later remember and use.

A "lesson" is a whole course built from its own uploaded sources, with however
many sections (comprehension gates) its outline has.
"""

STUDENT = {
    "id": "eval_sam",
    "name": "Sam",
    "learning_style": "examples_first",
    "motivation": 4,
    "confidence": 3,
    "base_error_rate": 0.35,
    "verbosity": "low",
    "ask_questions": True,
    "give_up_threshold": 6,
    "prior_knowledge": 0.2,
    "description": "Second-year student. Learns best from a worked example before the theory; "
                   "honest about confusion; wants to pass exams next month.",
}

# (source user id, source folder) -> the course the eval student takes, in order.
# related_to: earlier lessons Pedro could sensibly build on (None = nothing to link).
LESSONS = [
    {"n": 1, "name": "What is Intelligence", "src": (14, "intsys"), "related_to": None},
    {"n": 2, "name": "Intelligent Agents", "src": (14, "ioioioi"), "related_to": [1]},
    {"n": 3, "name": "Markov Chains", "src": (14, "doajdoiadow"), "related_to": None},
    {"n": 4, "name": "Queueing Theory", "src": (14, "test321"), "related_to": [3]},
    {"n": 5, "name": "Expert Systems", "src": (14, "dddddaaaa"), "related_to": [1, 2]},
    {"n": 6, "name": "Transportation Problems", "src": (14, "daowdapok"), "related_to": None},
    {"n": 7, "name": "Genetic Algorithms", "src": (14, "aaaddddd"), "related_to": [2]},
    {"n": 8, "name": "OpenMP Parallel Programming", "src": (14, "gagagag"), "related_to": None},
    {"n": 9, "name": "Systems Architecture", "src": (14, "SystemsArchitecture"), "related_to": [8]},
    {"n": 10, "name": "Driving Theory", "src": (11, "driving"), "related_to": None},
]

# Planted moments. The simulated student performs each once, naturally, when it
# fits (section=None: any section of that lesson). `memory` is what Pedro should
# carry forward; `expect_from` is the lesson from which we check it is used.
BEATS = [
    {"id": "prefers_examples", "lesson": 1, "section": 0,
     "instruction": "Early on, tell Pedro plainly that you learn much better when he shows you a "
                    "worked example FIRST and explains the theory after.",
     "memory": "Sam learns best from a worked example before the theory.",
     "expect_from": 2},
    {"id": "markov_misconception", "lesson": 3, "section": None,
     "instruction": "You genuinely believe every ROW and every COLUMN of a transition matrix must each "
                    "sum to 1. When transition matrices come up, say so and answer a check question "
                    "using that belief. Only give it up if Pedro explicitly corrects it.",
     "memory": "In Markov Chains Sam believed columns of a transition matrix must also sum to 1.",
     "expect_from": 4},
    {"id": "steady_state_analogy", "lesson": 3, "section": 1,
     "instruction": "Ask Pedro for an everyday analogy for steady-state probabilities. If his analogy "
                    "helps, say explicitly that it made it click and name the analogy in a few words.",
     "memory": "An everyday analogy for steady state clicked for Sam in Markov Chains.",
     "expect_from": 4},
    {"id": "exam_goal", "lesson": 5, "section": 0,
     "instruction": "Mention that you have exams in about a month and you are most worried about "
                    "Markov chains and queueing.",
     "memory": "Sam has exams in about a month and worries most about Markov chains and queueing.",
     "expect_from": 6},
    {"id": "crossover_confusion", "lesson": 7, "section": None,
     "instruction": "You keep mixing up crossover and mutation (you think mutation combines two "
                    "parents). Get at least two check questions on this wrong before it sinks in.",
     "memory": "In Genetic Algorithms Sam repeatedly confused crossover with mutation.",
     "expect_from": 8},
]

# Questions the student asks Pedro in general chat after a lesson. `expects` is
# what a good answer is grounded in (checked against what actually happened).
PROBES = [
    {"after": 3, "question": "What did I get wrong in the Markov chains lessons?",
     "expects": ["markov_misconception"]},
    {"after": 5, "question": "Which explanation really helped me with steady state, and what's my exam situation?",
     "expects": ["steady_state_analogy", "exam_goal"]},
    {"after": 7, "question": "What have I found hardest so far across all my courses?",
     "expects": ["markov_misconception", "crossover_confusion"]},
    {"after": 10, "question": "Give me a quick summary of everything we've covered together and what I should "
                              "review before my exams.",
     "expects": ["exam_goal", "markov_misconception", "crossover_confusion"]},
]
