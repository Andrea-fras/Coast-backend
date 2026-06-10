"""Student OMA — per-student cognitive profile.

Architecture: 5 stores, all backed by SQLite (shared db file), each row
namespaced by student.

Per-course stores (namespace = u{user_id}__student__{folder_slug}):
  - ActiveContextStore   "what's happening right now in this course"
  - ConceptMasteryStore  "how well does the student know each concept"
  - EpisodeStore         "immutable chronological event log"
  - PatternStore         "course-specific preferences + struggle clusters"

Cross-course store (namespace = u{user_id}__identity):
  - AcademicIdentityStore "traits that transcend any single course"

The stores are designed to be queryable independently — they are both
the personalization layer for Pedro AND a structured cognitive data
record that supports teacher dashboards and analytics.
"""

from .stores import (
    course_namespace,
    identity_namespace,
    ActiveContextStore,
    ConceptMasteryStore,
    EpisodeStore,
    PatternStore,
    AcademicIdentityStore,
)
from .recorder import StudentRecorder, EpisodeOutcome
from .orchestrator import StudentOrchestrator, build_student_orchestrator
from .consolidator import CourseConsolidator, IdentityConsolidator

__all__ = [
    "course_namespace",
    "identity_namespace",
    "ActiveContextStore",
    "ConceptMasteryStore",
    "EpisodeStore",
    "PatternStore",
    "AcademicIdentityStore",
    "StudentRecorder",
    "EpisodeOutcome",
    "StudentOrchestrator",
    "build_student_orchestrator",
    "CourseConsolidator",
    "IdentityConsolidator",
]
