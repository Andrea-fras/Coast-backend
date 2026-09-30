"""Curated persona audit — seed + run helpers."""

from __future__ import annotations

import json
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
PERSONA_IDS = ("sparse", "medium", "rich")


def load_json(name: str) -> dict:
    path = FIXTURES_DIR / name
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_course() -> dict:
    return load_json("course.json")


def load_persona(persona_id: str) -> dict:
    if persona_id not in PERSONA_IDS:
        raise ValueError(f"Unknown persona {persona_id!r}; expected one of {PERSONA_IDS}")
    return load_json(f"{persona_id}.json")


def load_scenarios() -> dict:
    return load_json("scenarios.json")
