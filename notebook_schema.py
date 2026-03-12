"""Pydantic models for the notebook / lecture-notes JSON schema."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class IntroHighlight(BaseModel):
    """A key topic highlight shown in the notebook intro."""

    label: str = Field(..., description="Short topic label, e.g. 'Supply & Demand'")
    desc: str = Field(..., description="One-line description of the topic")


class NotebookIntro(BaseModel):
    """Introductory section of a generated notebook."""

    text: str = Field(
        ...,
        description="A welcoming paragraph that frames the lecture content and motivates the reader",
    )
    highlights: list[IntroHighlight] = Field(
        ..., description="2-4 key topic highlights"
    )


class Subsection(BaseModel):
    """A subsection within a main section, providing deeper explanation."""

    title: str = Field(..., description="Subsection title")
    content: str = Field(
        ...,
        description="Intuitive, Socratic explanation of the concept — clear enough for any student",
    )
    bullets: Optional[list[str]] = Field(
        None,
        description="Optional bullet points for key takeaways, formulas, or examples",
    )


class Section(BaseModel):
    """A major section of the notebook, covering one core topic."""

    icon: str = Field(..., description="A single emoji that represents this section's topic")
    title: str = Field(
        ...,
        description="Section title — use a pattern like 'Topic Name (Intuitive Subtitle)'",
    )
    tags: list[str] = Field(
        ...,
        description="Lowercase topic tags for matching against past paper questions",
    )
    content: str = Field(
        ...,
        description="Opening paragraph that introduces the section topic intuitively",
    )
    subsections: Optional[list[Subsection]] = Field(
        None,
        description="Deeper dives into specific aspects of the topic",
    )


class ChatResponse(BaseModel):
    """A keyword-triggered chat response for the Pedro AI tutor."""

    keywords: str = Field(
        ...,
        description="Pipe-separated lowercase keywords that trigger this response, e.g. 'elasticity|elastic|inelastic'",
    )
    response: str = Field(
        ...,
        description="Pedro's helpful, Socratic response about this topic",
    )


class Notebook(BaseModel):
    """Top-level notebook representation — a generated study guide from lecture notes."""

    id: str = Field(..., description="Slug identifier, e.g. 'nb_microecon'")
    title: str = Field(
        ..., description="Notebook title, e.g. 'Microeconomics: How Markets Think'"
    )
    course: str = Field(
        ..., description="Course name, e.g. 'Economics 101 — Microeconomics'"
    )
    icon: str = Field(..., description="Single emoji representing the course")
    color: str = Field(
        ..., description="Hex color for UI accent, e.g. '#2ECC71'"
    )
    intro: NotebookIntro = Field(..., description="Introductory section")
    sections: list[Section] = Field(
        ..., description="Ordered list of major sections"
    )
    chatResponses: list[ChatResponse] = Field(
        ...,
        description="5-8 keyword-triggered chat responses covering the main topics",
    )
