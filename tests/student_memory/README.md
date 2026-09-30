# Student memory suite (local)

Checks that Pedro has an accurate, recallable picture of each student.

```bash
python3 tests/student_memory/run.py          # all (~6s, no API keys, no network)
python3 tests/student_memory/run.py recall   # filter by file/test name
python3 tests/student_memory/run.py -v       # tracebacks
```

`harness.py` points `coast.db` and `oma.db` at a fresh temp directory (it refuses to run against real data), blanks all API keys, and replaces Pedro's model with a scripted fake that captures the exact system prompt. Everything else is the real code: `tutor.send_message_stream`, `learning_jobs`, `oma_provider`, the evaluator and consolidators. Only the section evaluator's LLM verdict is stubbed, via `stub_evaluator`.

Status: steps 1–5 are implemented and passing. Step 6 (workshop generalisation) is reported as `DEFER` until that phase.

## Plan (local)

| Step | Goal | Tests |
|---|---|---|
| 1 | **Reliable recording.** Fix the evaluator `importance=` crash. Add `learning_jobs.retry_failed_completions(user_id=None)`. Replay failed jobs without duplicating memory. | `test_recording` (evaluator / completion / replay / retry), `test_gating` |
| 2 | **No misleading profile.** Identity strengths need 2+ courses and use concept *names*. Prompt blocks drop malformed traits. "Struggling" uses recent error rate, so recovery clears it but real difficulty stays flagged. Clean up existing junk rows. | `test_profile_accuracy` (identity / struggling / hygiene) |
| 3 | **Record what was demonstrated.** Named grading tags credit one concept. A single success is not "mastered". Wrong answers count as evidence. Hinted success counts less. Correct answers write episodes. | `test_profile_accuracy` (mastery), `test_gating::test_gate_answers_are_kept_as_evidence` |
| 4 | **Every conversation updates the profile:** lesson, folder, general, workshop. Each episode references its `ChatMessage` ids rather than copying the transcript. Behaviour signals feed patterns. | `test_recording` (surfaces), `test_workshops::test_workshop_work_is_recorded` |
| 5 | **Exact recall across courses.** Find the specific lecture or section asked about (not the last 60 messages), with the student's own words and the date. Works from general chat and from inside another course, without trigger words. Golden moments come back. Nothing is invented. | `test_recall`, `test_workshops::test_workshop_work_is_recallable_later` |
| 6 | **Workshops generalised.** Any outline section can carry a `workshop` contract (e.g. "Build a tiny LLM"), not just Memory Palace. | `test_workshops::test_custom_workshop_contract_is_honoured` |

## Contracts the tests assume

- **Grading tags:** `[ANSWER_CORRECT: <concept name>]` / `[ANSWER_WRONG: <concept name>]`, optional `| hinted`. Resolved against the section's concepts. A bare legacy tag credits at most one concept (the only concept in the section, or none).
- **Episodes:** every recorded chat turn stores `store_specific.chat_message_ids = [student_msg_id, pedro_msg_id]`.
- **General chat:** filed under a course namespace when the message clearly concerns that course. Otherwise it is still recorded for the student (namespace free to choose, but prefixed `u{id}__`).
- **Recall:** the student's verbatim words from the matched section, the section title, and the session date appear in Pedro's system prompt.
