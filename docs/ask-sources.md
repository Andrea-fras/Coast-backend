# Ask sources — implementation and validation

Implemented 15 September 2026. Open an uploaded lesson and select **Ask sources** beside **Guided lesson**. No roadmap is required. Premade lesson behavior is unchanged.

## Data flow

The upload pipeline still extracts each PDF/PPTX once into its normalized page cache. `source_search.py` creates overlapping passages within page boundaries, retaining the original source ID and page/slide number. SQLite `source_search_indexes` stores passages and compact float32 embeddings from the existing OpenAI embedding configuration. Background preparation has two workers and a bounded pending set. Completed embedding batches persist and resume on access after a restart. Uploads schedule preparation without waiting for it.

Retrieval filters by authenticated student and exact folder first. It combines BM25-style keyword ranking with cosine embedding ranking using reciprocal ranks. Keyword search is usable before vectors finish, including during OMA concept ingestion. A failed embedding request leaves keyword search available. Short follow-ups reuse the preceding question and cited pages; broad summaries/comparisons distribute evidence across files. Each answer has a maximum of 12 passages / 18,000 evidence characters. This is a selected evidence window, not proof that every page was inspected.

`source_chat.py` uses Pedro's existing configured model and provider admission control, with OpenAI fallback before any Gemini tokens have been emitted. Its dedicated prompt asks for direct source-grounded answers, citations, explicit evidence gaps and no comprehension checks. It does not invoke Content OMA graph traversal, Student OMA, lesson evaluation or map rewards. Generic memo generation excludes source conversations.

Questions/answers remain in `chat_messages` with `context_type=sources` for existing usage accounting. `source_chat_turns` records retry identity, conversation/folder, state, citation evidence and source availability at answer time. One active question per conversation is enforced with a short SQLite transaction. Repeating a request ID returns the saved answer or retries the same question; it cannot insert another user message. Attempts have a five-minute recovery window, with attempt identity preventing an older run overwriting a retry. HTTP streams require an explicit completion event; interrupted replies are displayed as incomplete and can be retried.

## Citations and lifecycle

Only supplied citation IDs become clickable. Citation source ID, original filename, page, excerpt and availability are saved with each answer. Invalid citation IDs are removed; an answer with evidence but no valid citation is replaced by an unsupported-answer response. This validates targets, not factual entailment; answer quality still requires model evaluation.

Citations open an authenticated side panel. PDF pages are rendered from the original file at the requested page, avoiding browser PDF-fragment inconsistencies. PPTX shows extracted slide text plus original download; it does not promise a faithful slide layout. Exact passage highlighting is not implemented. Existing guided-lesson references have not been retrofitted in this change.

Removing a source removes its search index and excludes it from future retrieval. Saved answers retain provenance but mark removed source buttons unavailable. Renaming a lesson carries source conversations with it, and waits for an active source answer. User purge removes the new tables' data. Source data in prompts is treated as untrusted material, and generated source answers cannot load arbitrary images or navigate model-generated links.

No new infrastructure or Python/npm dependencies were added. Existing `OPENAI_API_KEY`, `EMBED_MODEL`, `PEDRO_PROVIDER`, `GEMINI_CHAT_MODEL`, `OPENAI_MODEL` and provider capacity settings apply. Tables are created by the existing `init_db()` at server startup. This implementation was launched locally, not deployed to Render.

## Validation

- 16 new backend tests: real authenticated HTTP, lexical/hybrid retrieval, page ownership, course isolation, saved citations/history, idempotent retries, failed/stale runs, no learner-memory writes, source removal, pending-file coverage and actual PPTX upload→search→slide preview.
- 14 browser checks against the real FolderView/AskSources components with a mock transport: no-roadmap access, source readiness, streamed math, page navigation, history restoration, retries and mode switching.
- 3 stream parser tests including Unicode and byte boundaries, truncated streams and provider errors.
- Existing 12 HTTP integrity and 7 upload regressions pass.
- Production frontend build passes. New components/utilities lint clean. FolderView retains three existing hook-dependency warnings in unrelated lesson/curated callbacks.

Live provider smoke test against the 13-page parallel-programming source in `jkjb`: CRS arrays/boundaries answer correctly cited pages 3 and 4, completed in 10.92 seconds (first token 9.84 seconds), persisted all four conversation messages including the out-of-scope follow-up, and returned an authenticated PNG preview. The unrelated medieval-poetry question returned no supporting evidence in 0.21 seconds without a teaching-model call. Cached readiness took 0.18 seconds; the initial cold source embedding preparation took about 3.16 seconds. These are individual local samples, not load-test results or latency guarantees.

Commands:

```sh
# Backend directory
python3 scripts/test_source_chat.py
python3 scripts/test_http_integrity.py
python3 scripts/test_upload_flow.py

# Frontend directory
node --test scripts/test-source-chat-stream.mjs
npm run build
# Open http://127.0.0.1:5173/scripts/source-chat-regressions.html for the isolated browser fixture.
```

Future load-driven work: vector indexing beyond a single course's in-memory ranking, paginated long conversations, diagram-heavy retrieval and exact passage highlights. Repeated page text such as lecture-template instructions remains an extraction-quality issue shared with the existing ingestion pipeline.
