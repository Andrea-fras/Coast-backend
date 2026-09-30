# Coast OMA & Pedro Context Architecture

> **Last updated:** June 2026  
> **Purpose:** High-level reference for Content OMA, Student OMA, and how Pedro receives context on each message.  
> **Status:** Lessons with Content OMA work well today. The next phase is making Student OMA **accurate enough** that personalization is **noticeable** — small details in Pedro's replies that show he knows what the student has done.

## September 2026 — memory reliability pass

Verified by `tests/student_memory/` (`python3 tests/student_memory/run.py`, ~6s, isolated, no network):

- **Recording:** every conversation surface records every turn. The section evaluator no longer crashes on open questions. Failed completion jobs can be replayed with `learning_jobs.retry_failed_completions()`.
- **Accuracy:** mastery credits only the concept that was tested. "Struggling" is based on recent answers. Identity strengths and weaknesses need 2+ courses and use concept names. Traits that are malformed or no longer supported are removed.
- **Recall:** `student_history.recall_block()` finds the exact course/section a student asks about (by title, key topics and a search over their own past words) and quotes the real transcript with dates. It works from general chat and from inside another course.
- **Scale:** at ~3 years (15k turns, 6 courses) the profile build takes ~95–130 ms and recall ~30 ms per message.
- **Existing data:** `scripts/rebuild_student_memory.py` does the following. It is a dry run by default; `--apply` backs up oma.db first.
  - Backfills past turns, keeping original timestamps.
  - Re-attributes legacy mistakes.
  - Rebuilds mastery.
  - Removes junk traits.

---

## 1. Mental model

Coast splits “what to teach” from “who to teach it to”:

| System | Question it answers | Retrieval style today |
|--------|---------------------|---------------------|
| **Content OMA** | What is in the student's course materials? | **Query-time search** (semantic + FTS + concept graph) |
| **Student OMA** | Who is this student and what have they done? | **Static snapshot** per message (`get_student_profile_block`) |
| **Pedro** | How do I teach this person right now? | Assembles both into a system prompt + conversation thread |

Both systems share one SQLite database (`oma_data/oma.db` by default) but use **different namespaces** and **different stores**.

```
Student message
       │
       ├─► Content OMA.retrieve(query)     → PDF chunks, diagrams, concepts
       │
       ├─► Student OMA.build_profile()       → progress, mastery, mistakes, traits
       │
       └─► build_system_prompt()             → Pedro identity + blocks above + thread history
```

**Bridge file:** `OCR/oma_provider.py` — wires upload, chat, and lesson flows into both OMA systems.

---

## 2. Configuration

| Env var | Values | Effect |
|---------|--------|--------|
| `RAG_PROVIDER` | `flat` \| `oma` \| `shadow` | `flat` = legacy RAG only; `oma` = Content OMA; `shadow` = both (compare) |
| `STUDENT_OMA_ENABLED` | `true` \| `false` | Default on when `RAG_PROVIDER=oma` |
| `OMA_DB_PATH` | path | SQLite DB (default `./oma_data/oma.db`) |
| `OMA_IMAGE_DIR` | path | Extracted diagram images |

Pedro chat requires **`RAG_PROVIDER=oma`** and **`STUDENT_OMA_ENABLED=true`** for full OMA behavior.

---

## 3. Content OMA

Content OMA indexes a student's uploaded PDFs into searchable knowledge: concepts, text chunks, and images.

### 3.1 Namespaces

Per user + folder:

```
u{user_id}__{folder_slug}
```

Example: user `14`, folder `Operations` → `u14__operations`

### 3.2 Stores (3 tables)

All extend `SemanticStoreBase` (embeddings + FTS5 + reciprocal rank fusion).

| Store | Table prefix | What it holds |
|-------|--------------|---------------|
| **ConceptStore** | `concept_items` | Canonical concepts, prerequisites, aliases |
| **ContentStore** | `content_items` | Text chunks: definitions, examples, exercises, narrative |
| **ImageStore** | `image_items` | Diagrams/figures linked to concepts and source pages |

**Orchestrator:** `coast_content_oma/orchestrator.py` — classifies queries, routes to stores, merges results.

### 3.2 Query classes

The orchestrator regex-classifies each query:

- `definitional`, `example_request`, `exercise`, `how_to`, `prerequisite`, `figure`, `explanation`, `general`

Each class pulls different chunk types (e.g. exercises → exercise chunks; prerequisite → concept graph walk).

### 3.3 Write path — when and how data is written

| Trigger | Function | What happens |
|---------|----------|--------------|
| **PDF upload** | `oma_provider.ingest_pdf_into_oma()` | Background thread; calls `IngestionPipeline` |
| **Folder backfill** | `queue_folder_oma_backfill()` | Re-processes PDFs missing from OMA |
| **After section advance** | `kickoff_background_vision_async()` | Pre-warms image/description for upcoming sections |

**Ingestion pipeline** (`coast_content_oma/ingestion/`):

1. Extract text + pages from PDF  
2. LLM classifies page content types  
3. Vision model describes diagrams  
4. Concepts extracted and linked  
5. Chunks + embeddings written to Concept / Content / Image stores  

Ingest is **async** and **per-folder**. Outline generation can wait for active ingests via `_wait_for_oma_if_indexing()`.

### 3.4 Read path — when and how data is queried

| Call site | Function | Query built from |
|-----------|----------|------------------|
| **Folder chat** | `resolve_folder_content()` → `get_folder_context()` | Student's message |
| **Lesson chat** | `lesson.build_lesson_prompt()` | Section title + objectives + key topics + student message |
| **Lesson follow-up** | Extra `get_folder_context(student_message)` | Student's question only |
| **Global chat (folder detected)** | `_resolve_global_lesson_context()` | Student message → folder name match |
| **Outline generation** | `build_outline_context()` | Full course structure |
| **Map treasures** | Content OMA concept sampling | Completed sections |

**Core retrieval:**

```python
orch.retrieve(namespace, query, max_content=8, max_images=4)
# → RetrievalResult → .to_prompt_block(max_chars=...)
# → also returns concept_ids surfaced
```

Returns: concept candidates, ranked text chunks with provenance (source PDF, page), image URLs.

**Fallback:** If OMA is empty, `rag.build_folder_context()` (legacy flat RAG).

### 3.5 What Pedro gets from Content OMA

- Structured block: `--- COURSE MATERIAL (retrieved via Content OMA) ---`  
- Source filenames and page numbers  
- Up to ~8–16k chars depending on context (folder vs lesson vs recap)  
- Diagram markdown links (`/api/oma/images/{id}`)  
- Lesson mode also injects: course outline, current section objectives, teaching rules, verification block  

**Content OMA is query-time.** This is why Pedro cites the right PDFs and diagrams for the current question.

---

## 4. Student OMA

Student OMA is the **long-term memory of the learner** — progress, mastery, mistakes, patterns, and identity.

### 4.1 Namespaces

| Scope | Pattern | Example |
|-------|---------|---------|
| Per course | `u{user_id}__student__{folder_slug}` | `u14__student__operations` |
| Cross-course identity | `u{user_id}__identity` | `u14__identity` |

### 4.2 Stores (5 tables)

| Store | Purpose | Update style |
|-------|---------|--------------|
| **EpisodeStore** | Immutable event log | Append-only |
| **ConceptMasteryStore** | Per-concept score | In-place Bayesian EMA updates |
| **ActiveContextStore** | “What's happening now” | Singleton fragments supersede previous |
| **PatternStore** | Derived teaching patterns + golden moments | Upsert with confidence |
| **AcademicIdentityStore** | Cross-course traits (learning style) | Upsert with confidence |

**Orchestrator:** `coast_content_oma/student/orchestrator.py`  
**Single write entry point:** `StudentRecorder.record_episode()` in `student/recorder.py`

### 4.3 Episode types

Recorded in `EpisodeStore.store_specific.episode_type`:

| Type | Meaning |
|------|---------|
| `qa` | Student asked, Pedro answered (API exists; rarely used from chat today) |
| `exercise_attempt` | Practice answer (especially wrong answers) |
| `section_completed` | Finished a lesson section (authoritative) |
| `section_evaluation` | Post-section LLM assessment |
| `lesson_started` / `lesson_completed` / `lesson_dropoff` | Lesson lifecycle |
| `concept_reviewed` / `self_assessment` / `external_event` | Other |

Each episode can carry: `user_message`, `assistant_response`, `concept_ids`, `section_index`, `outcome`, `signals`.

### 4.4 Write path — when and how data is written

#### A. Onboarding (once per student)

| Step | Where | Writes to |
|------|-------|-----------|
| Pedro onboarding chat | `onboarding.py` | `AcademicIdentityStore` (traits) |
| Finalize onboarding | `server.py` + `onboarding.py` | Also `users.learning_preferences` (legacy JSON — **no longer injected into Pedro prompt**) |
| Onboarding episodes | `record_onboarding_episode()` | `EpisodeStore` in profile namespace |

#### B. Every Pedro turn — lesson, workshop, folder, general chat

`tutor` → `oma_provider.record_conversation_turn()` (queued on one ordered background writer; never delays the reply).

| Pedro's reply contains | Writes |
|------|--------|
| `[ANSWER_CORRECT: <concept>]` / `[ANSWER_WRONG: <concept>]` (optional `\| hinted`) | `exercise_attempt` episode + mastery evidence for **that one concept** |
| Legacy bare `[ANSWER_CORRECT]` / `[ANSWER_WRONG]` | Episode; mastery only if the section has exactly one concept |
| No grading tag | `qa` episode with behaviour signals (asked for example, step-by-step, …) |
| `[CLICKED: …]` | Golden moment linked to the concept(s) it explains |
| `[REMEMBER: …]` | Identity trait |

Every episode stores `chat_message_ids` pointing at `chat_messages` — the canonical transcript. Episodes index it, they never copy Pedro's reply. General chat with no course goes to `u{id}__general`; general chat clearly about a course is filed under that course. Automatic section openers are not recorded.

**Mastery model** (`concept_mastery.py`): recency-weighted success rate with a neutral prior, `(1 + s) / (2 + s + f)`. Older evidence is discounted ×0.85 per new answer. One correct answer gives 0.67 (not mastered). Three independent correct answers give ~0.78. Hinted successes count half. Section-evaluator verdicts raise or lower the evidence itself, so they persist.

**Struggling** (`struggles.py`): at least 2 wrong among the last 5 graded answers, and wrong answers are at least half of them. Recovery clears it.

#### C. Section completion (authoritative)

| Event | Function | Writes to |
|-------|----------|-----------|
| Student clicks **Next Section** | `lesson.advance_section()` → `record_section_completed_authoritative()` | `EpisodeStore` (`section_completed`) + `ActiveContextStore` (`current_focus`) |
| Background pipeline | `kickoff_post_section_pipeline_async()` | See D + E |

#### D. Post-section evaluator (background, async)

| Step | File | Writes to |
|------|------|-----------|
| Fetch section chat transcript | `evaluator.fetch_section_transcript()` | reads `coast.db` ChatMessage |
| LLM/heuristic evaluation | `evaluator.evaluate_section_transcript()` | — |
| Apply verdict | `evaluator.apply_evaluation()` | `ConceptMasteryStore`, `PatternStore` (golden moments), `EpisodeStore` (`section_evaluation`), `ActiveContextStore` (open questions), marks mistakes resolved |

Runs once per section (skips if already evaluated).

#### E. Course consolidator (background, after evaluator)

| Step | File | Writes to |
|------|------|-----------|
| Rule-based pattern detection | `CourseConsolidator.run()` | `PatternStore` (format prefs, struggle clusters, pace, give-up) |
| Cross-course roll-up | `IdentityConsolidator` | `AcademicIdentityStore` |

Detectors need **signal counts** from episodes (e.g. `asked_for_example` ≥ 3). Signals are sparsely populated today.

#### F. Progress ledger (authoritative, not in OMA DB)

| Source | Table | Used for |
|--------|-------|----------|
| Course outline pointer | `coast.db` → `CourseOutline` | `current_section`, finished sections |

Loaded at read time via `_load_progress_ledger()` — not stored in EpisodeStore but treated as ground truth in the profile block.

### 4.5 Read path — when and how data is queried

| Call site | Function | What it returns |
|-----------|----------|-----------------|
| **Every folder/lesson chat** | `get_student_profile_block(user, folder, concept_ids)` | Compact text block (~1200–5000 chars) |
| **Global chat** | `get_global_student_profile_block(user)` | Cross-course summary (~1600 chars) |
| **First section intro** | `get_course_intro_student_block()` | Onboarding + identity traits |
| **Section intro (in-lesson)** | `get_section_intro_student_block()` | Mastery tier for section concepts |
| **Admin / analysis UI** | `build_student_analysis()` | Full structured JSON + graph |
| **Recap requests** | `lesson._fetch_lesson_conversation_recap()` | Raw chat from `coast.db` (keyword-triggered only) |

**Profile block contents** (`StudentOrchestrator.to_prompt_block()`):

1. Lesson progress (authoritative ledger — finished sections + topics)  
2. Practice mistakes (wrong answer quotes by section)  
3. Struggling topics (persistent — >50% wrong or repeated mistakes)  
4. Active context: current focus, goal, unresolved, open questions  
5. Focused mastery (concepts in current query/section) OR strongest/struggling summary  
6. Course patterns + golden moments (filtered to query concepts if possible)  
7. Fading mastery (spaced repetition hints)  
8. Other-course history + cross-course identity traits  
9. Last 7 days episode stats  

**Important:** `EpisodeStore.search()` exists (semantic search over episodes) but is **not called at chat time** today. The profile is a **pre-built summary**, not query-conditioned retrieval.

### 4.6 Active context — why Pedro often “feels” personalized here

`ActiveContextStore` is updated on **every section complete**:

```
current_focus → "Just completed section: Linear Programming in MDPs"
```

This is small, salient, and appears near the top of the profile block. Combined with the progress ledger (32 finished section titles), Pedro reliably knows **where the student is** — but not deeply **what they said** in past sections unless:

- A mistake quote survived truncation, or  
- A recap keyword triggered raw chat injection, or  
- An evaluator ran and wrote golden moments (rare on real accounts today)

---

## 5. Pedro prompt injection

### 5.1 Entry point

All chat flows through `tutor.send_message()` or `tutor.send_message_stream()`.

### 5.2 Assembly order (`build_system_prompt`)

```
1. PEDRO_IDENTITY          — teaching rules, tone, tag protocol
2. ONBOARDING_MODE_BLOCK   — only if context_type == "onboarding"
3. Student name + course
4. STUDENT OMA block       — get_student_profile_block() or global variant
5. Context-specific material:
   - lesson  → build_lesson_prompt() (outline + Content OMA + teaching rules)
   - folder  → Content OMA folder context
   - global  → matched folder material OR notebook snippets
   - notebook → notebook text
6. Conversation thread     — recent messages (+ summary if long)
```

### 5.3 Legacy layers removed (June 2026)

These are **no longer injected** into Pedro's prompt:

| Removed | Was | Why removed |
|---------|-----|-------------|
| `users.learning_preferences` | Duplicate of onboarding traits | Identity lives in `AcademicIdentityStore` inside Student OMA block |
| `SkillProfile` | Quiz topic scores 0–100 | QuestionPage removed; mastery is in `ConceptMasteryStore` |
| `TutorMemo` | Free-text LLM notes | Unstructured, unsearchable, contradicted Student OMA |

Admin APIs for skill profile and tutor memo still exist for dashboards; they are not Pedro's source of truth.

### 5.4 By UI surface

#### Lesson chat (`context_type=lesson`)

| Layer | Source |
|-------|--------|
| Student profile | `get_student_profile_block()` — concept IDs from **current section** |
| Course material | Section query + optional question-targeted Content OMA |
| Structure | Full outline, objectives, verification block, section opener block |
| Thread | This conversation's messages |

#### Folder chat (`context_type=folder`)

| Layer | Source |
|-------|--------|
| Student profile | `get_student_profile_block()` — concept IDs from **Content OMA hits** |
| Course material | `resolve_folder_content(message)` |
| Thread | This conversation's messages |

#### Global chat (`context_type=global`)

| Layer | Source |
|-------|--------|
| Student profile | Cross-course block OR single-folder if folder name detected in message |
| Course material | Folder match → Content OMA; else notebook snippets |
| Recap | If message matches recap regex → `_fetch_lesson_conversation_recap()` |

#### Onboarding (`context_type=onboarding`)

Pedro extracts traits at end → `AcademicIdentityStore`. No Content OMA required.

### 5.5 Pedro machine-readable tags (write path hooks)

Pedro can append hidden tags in replies (stripped in UI):

| Tag | Effect on Student OMA |
|-----|------------------------|
| `[ANSWER_WRONG: concept]` | `exercise_attempt` (mistake) + mastery evidence on that concept |
| `[ANSWER_CORRECT: concept \| hinted]` | `exercise_attempt` (success) + mastery evidence (half weight if hinted) |
| `[SECTION_COMPLETE]` | Triggers section verification flow |
| `[TEST_OUT_PASSED]` | Test-out advancement |

Grading tags name the concept they tested (see §4.4 B). Every turn is recorded whether or not it carries a tag. Tags are stripped from anything shown to the student: the history API (`tutor.get_chat_history`) and the frontend (`src/utils/pedroTags.js`). They stay in `chat_messages` so the evaluator can read them.

### 5.6 End-to-end diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        STUDENT MESSAGE                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
   Content OMA         Student OMA          coast.db
   retrieve(query)     profile block        chat history
         │                   │                   │
         │              progress ledger          │
         │              (CourseOutline)          │
         ▼                   ▼                   ▼
   PDF chunks +         WHO / WHERE /          thread
   concept_ids           mastery / mistakes
         │                   │
         └─────────┬─────────┘
                   ▼
           build_system_prompt()
                   ▼
              LLM (Pedro)
                   ▼
           strip tags → UI reply
                   ▼
         record_chat_episode()  ──► Student OMA write
         (if graded tags)
```

---

## 6. What works well today

| Capability | Why it works |
|------------|--------------|
| **Structured lessons** | Content OMA + outline + section material give Pedro accurate course-specific teaching |
| **PDF citations** | Query-time Content OMA retrieval with source + page |
| **Diagrams in replies** | ImageStore + vision ingest |
| **Section-aware teaching** | Lesson prompt knows current section, objectives, prior/next sections |
| **Wrong-answer logging** | `[ANSWER_WRONG]` → mistake episodes with student quote |
| **Progress tracking** | CourseOutline + section_completed episodes |
| **Onboarding traits** | Learning style captured once → identity store |

Students get **high-quality, material-grounded lessons**. That foundation is solid.

---

## 7. Personalization gap — what students should notice but often don't

**Goal:** Pedro replies include small, specific details that prove he knows the student:

- *"Since you nailed ILP back in section 5, the LP formulation here will feel familiar."*  
- *"Last time you said ρ could exceed 1 — we cleared that up; quick check…"*  
- *"The coin-flip analogy worked for you on Poisson — same idea here."*  

### 7.1 Root causes today

| Gap | Detail |
|-----|--------|
| **Static profile, not search** | Full progress ledger truncates; deep history drops off |
| **Thin episode log** | Free chat not recorded; only mistakes + section completions |
| **Evaluator under-run** | `section_evaluation` + golden moments missing on most real sections |
| **Signals unused** | `asked_for_example`, `prefers_step_by_step` etc. rarely set → consolidator idle |
| **No paired retrieval** | Content OMA concept IDs don't drive episode/pattern search |
| **Active context dominates** | Pedro knows *where*; less about *what they said* |

### 7.2 Personalization dimensions vs data availability

| Pedro should… | Needs | Populated for typical student? |
|---------------|-------|-------------------------------|
| Match learning style | Identity traits | ⚠️ Onboarding yes; later sparse |
| Skip known material | Concept mastery | ⚠️ Only tagged exercise concepts |
| Reference past mistake | Episode `user_message` | ⚠️ Mistakes only, truncated in block |
| Reuse winning analogy | Golden moment patterns | ❌ Evaluator rarely run |
| Accurate struggle tone | Evaluator `resolved` states | ❌ |
| Cite old section by name | Section index + evaluation | ❌ Not in prompt today |

---

## 8. Target architecture (planned)

Not implemented yet — documented here as the agreed direction.

### 8.1 Paired retrieval

```
1. Content OMA.retrieve(message) → material + concept_ids
2. retrieve_student_context(user, folder, message, concept_ids, intent)
   → Tier 0: anchor (section pointer, 1–2 traits)
   → Tier 1: query-conditioned episodes, mastery, golden moments
   → Tier 2 (recall intent): section evaluation + chat recap
3. Replace static ledger dump with fixed-budget retrieved history
```

### 8.2 Write path improvements (GIGO)

Before search helps, stores must contain the truth:

1. Log substantive `qa` episodes from folder/lesson chat  
2. Run post-section evaluator on every section complete (backfill existing users)  
3. Populate behavioral `signals` on episodes for consolidator  
4. Backfill episode embeddings for semantic search  

### 8.3 Noticeable personalization checklist

A reply is **noticeably personalized** when it includes at least one of:

- [ ] Student's name used naturally (already works)  
- [ ] Reference to a **specific completed section** by title  
- [ ] Reference to a **past answer they gave** (quote or paraphrase)  
- [ ] Adaptation to **learning style trait** (exercises first, visual, etc.)  
- [ ] Reuse of a **golden moment** analogy that worked before  
- [ ] Correct struggle tone (not harsh about resolved mistakes)  
- [ ] Bridge from **their** prior mastery, not generic "you may know"  

---

## 9. Key files reference

| Area | Path |
|------|------|
| Coast ↔ OMA bridge | `OCR/oma_provider.py` |
| Content retrieval | `OCR/coast_content_oma/orchestrator.py` |
| PDF ingest | `OCR/coast_content_oma/ingestion/` |
| Student orchestrator | `OCR/coast_content_oma/student/orchestrator.py` |
| Student recorder | `OCR/coast_content_oma/student/recorder.py` |
| Post-section evaluator | `OCR/evaluator.py` |
| Course consolidator | `OCR/coast_content_oma/student/consolidator.py` |
| Pedro chat + prompt | `OCR/tutor.py` |
| Lesson engine | `OCR/lesson.py` |
| Onboarding traits | `OCR/onboarding.py` |
| Persona audit scripts | `OCR/scripts/persona_audit/` |

---

## 10. Summary

- **Content OMA** = search engine over course materials. **Write on ingest. Read on every query.** Works well.  
- **Student OMA** = five stores for learner memory. **Write on episodes, section complete, evaluator, consolidator. Read as static profile block.** Architecture is sound; **write fidelity and read retrieval** are the gaps.  
- **Pedro** = identity + Student OMA block + Content OMA material + thread. Legacy quiz/memo/prefs layers removed.  

**Coast already delivers fantastic OMA-grounded lessons.** The product shift ahead is making Student OMA behave like Content OMA — **query-time, paired, accurate** — so students feel Pedro is genuinely getting to know them, one small detail at a time.
