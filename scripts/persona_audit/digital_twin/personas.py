"""Persona bank for the digital twin.

Each persona is a student archetype the simulator role-plays. Personas differ on
the axes that matter for tutoring: learning style, motivation, confidence,
error rate, verbosity, question-asking, give-up threshold, and prior knowledge.

Run one persona per account (or per clean Student OMA state) so the longitudinal
history accumulates for THAT student. Comparing the same lesson across personas
gives the personalization differential.
"""

from __future__ import annotations

PERSONAS: dict[str, dict] = {
    "anxious_step_by_step": {
        "id": "anxious_step_by_step",
        "name": "Maya",
        "learning_style": "step_by_step",
        "motivation": 5,            # 1-5, 5 = very motivated
        "confidence": 2,            # 1-5, low confidence
        "base_error_rate": 0.45,    # prob of getting a fresh exercise wrong
        "verbosity": "medium",
        "ask_questions": True,
        "give_up_threshold": 5,     # turns of sustained confusion before disengaging
        "prior_knowledge": 0.15,    # default mastery for unseen concepts (0-1)
        "description": (
            "Wants every step spelled out, hedges, asks clarifying questions, "
            "gets anxious when explanations are dense or skip steps. Praise and "
            "structure help; being rushed hurts."
        ),
    },
    "overconfident_speedrunner": {
        "id": "overconfident_speedrunner",
        "name": "Leo",
        "learning_style": "examples_first",
        "motivation": 3,
        "confidence": 5,
        "base_error_rate": 0.55,    # commits to wrong answers boldly
        "verbosity": "low",
        "ask_questions": False,
        "give_up_threshold": 3,     # gets bored fast
        "prior_knowledge": 0.35,
        "description": (
            "Wants to move fast, skips basics, commits to answers (often wrong) "
            "with certainty. Gets bored by repetition and basics; engaged by "
            "challenge and being corrected crisply."
        ),
    },
    "visual_learner": {
        "id": "visual_learner",
        "name": "Priya",
        "learning_style": "visual",
        "motivation": 4,
        "confidence": 3,
        "base_error_rate": 0.35,
        "verbosity": "medium",
        "ask_questions": True,
        "give_up_threshold": 4,
        "prior_knowledge": 0.25,
        "description": (
            "Learns from diagrams, analogies, and geometric intuition. Struggles "
            "with dense symbolic manipulation. Engagement jumps when Pedro uses a "
            "visual or a concrete analogy."
        ),
    },
    "foundational_struggler": {
        "id": "foundational_struggler",
        "name": "Sam",
        "learning_style": "plain_language",
        "motivation": 3,
        "confidence": 2,
        "base_error_rate": 0.6,
        "verbosity": "low",
        "ask_questions": True,
        "give_up_threshold": 4,
        "prior_knowledge": 0.05,
        "description": (
            "Weak prior knowledge, easily overwhelmed by jargon. Needs plain "
            "language and scaffolding. Benefits from one idea at a time; "
            "frustration rises fast when Pedro piles on."
        ),
    },
}


def get_persona(persona_id: str) -> dict:
    if persona_id not in PERSONAS:
        raise ValueError(f"unknown persona {persona_id!r}; known: {list(PERSONAS)}")
    return PERSONAS[persona_id]


def list_personas() -> list[str]:
    return list(PERSONAS)
