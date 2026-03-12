"""Pydantic models for the exam paper JSON schema."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Option(BaseModel):
    """A single option for a multiple-choice question."""

    id: str = Field(..., description="Option identifier (a, b, c, d, …)")
    text: str = Field(..., description="Option text")


class BaseQuestion(BaseModel):
    """Fields shared by every question type."""

    id: str = Field(..., description="Unique question id, e.g. 'e1'")
    number: int = Field(..., description="Question number as it appears on the paper")
    text: str = Field(..., description="Full question text")
    equation: Optional[str] = Field(
        None,
        description="LaTeX equation string if the question contains a formula, else null",
    )
    images: Optional[list[str]] = Field(
        None,
        description="List of image filenames/paths for diagrams associated with this question",
    )


class MultipleChoiceQuestion(BaseQuestion):
    """A multiple-choice question."""

    type: Literal["multiple-choice"] = "multiple-choice"
    options: list[Option] = Field(..., description="List of answer options")
    correctAnswerId: Optional[str] = Field(
        None,
        description="The id of the correct option (if determinable from the paper)",
    )


class MarkPoint(BaseModel):
    """A single marking point in a mark scheme."""

    point: str = Field(..., description="What the student needs to demonstrate")
    marks: int = Field(1, description="How many marks this point is worth")


class OpenEndedQuestion(BaseQuestion):
    """An open-ended / free-response question."""

    type: Literal["open-ended"] = "open-ended"
    modelAnswer: Optional[str] = Field(
        None,
        description="A model answer if one can be inferred or is provided",
    )
    keyTerms: Optional[list[str]] = Field(
        None,
        description="Key terms expected in a good answer",
    )
    markScheme: Optional[list[MarkPoint]] = Field(
        None,
        description="Structured marking points with marks per point",
    )
    totalMarks: Optional[int] = Field(
        None,
        description="Total marks available for this question",
    )


class ExamPaper(BaseModel):
    """Top-level exam paper representation."""

    id: str = Field(..., description="Slug identifier for the paper, e.g. 'econ_101'")
    title: str = Field(..., description="Paper title")
    description: Optional[str] = Field(None, description="Short description of the paper")
    questions: list[MultipleChoiceQuestion | OpenEndedQuestion] = Field(
        ..., description="Ordered list of questions"
    )
