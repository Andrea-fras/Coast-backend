# Concept alias ledger — read/write rules and ops policy

Append-only ledger: `concept_alias_items` maps `alias_id → canonical_id` per **content** namespace (`make_namespace(user_id, folder)`). Student stores (mastery, episodes, patterns) use **course** namespace (`course_namespace`) but reference content concept IDs.

## Read path

- **`ConceptResolver.resolve(content_ns, id)`** — walk active alias chain to terminal canonical (cycle-safe, max 16 hops).
- **`ConceptMasteryStore.aggregate_for_concept(course_ns, content_ns, id, resolver)`** — collect mastery rows for every ID that resolves to the same terminal; fold via `aggregate_mastery_rows()` (pessimistic-wins on eval state).
- **Content edges** — `prerequisites_of` / `traverse_prereq_chain` resolve edge IDs through the ledger before traversal.
- **Never physically fold mastery rows on merge** — aggregation is virtual at read time only.

## Write path

- **`StudentRecorder`** — `stamp_concept_ids()`: episodes store `concept_ids` (resolved canonical) and `matched_concept_ids` (raw retrieval match). Mastery writes use resolved canonical ID.
- **`merge_into(..., use_ledger=True)`** — append ledger entry, supersede src concept node, skip `remap_student_concept_ids`.

## Reversal and post-merge attribution

When **`ConceptAliasStore.reverse(content_ns, alias_id)`** sets `reversed_at`:

1. **Mastery / virtual aggregation** — `ids_for_canonical()` excludes reversed aliases on the next read; evidence on the un-merged alias ID no longer appears in the canonical's aggregated view.
2. **Episodes written during the merged period** — `concept_ids` points at canonical; `matched_concept_ids` holds the raw ID Pedro/retrieval matched. For audit, dispute resolution, or future re-partition after reversal, **prefer `matched_concept_ids` when present** over inferring from `concept_ids` alone.
3. **Pre-merge rows** — fully recoverable via reversal (history never rewritten).
4. **Post-merge rows without `matched_concept_ids`** — ambiguous after reversal; cannot be auto-partitioned. New writes always stamp both fields (v1+).

Past **destructive** merges (before the ledger) are sunk — no archaeological reconstruction.

## Cycle guard

- **`append_merge`** rejects when `resolve(canonical_id) == alias_id` (would create a loop).
- **`resolve`** returns early on cycles rather than looping forever.

## Dedup trigger policy (explicit)

| Trigger | When | Rationale |
|---------|------|-----------|
| **On ingest** | New PDF into an existing course namespace | The case the ledger exists for; safe and append-only |
| **Manual / dry-run** | `dedupe_folder_concepts(..., dry_run=True)` for inspection | Review before applying |
| **Cross-course / global** | Manual only until map UX defines island moves | Safe ≠ free — merges change Pedro's concept names and map nodes |

Set `OMA_ALIAS_LEDGER_ENABLED=false` to freeze **all** live merges (embedding dedup threshold → 0). Fail-safe: off means stop merging, not fall back to destructive remap.

## Profile block — progress ledger cap (temporary)

`to_prompt_block()` caps the progress ledger at first 2 + "… N more …" + last 3 when >8 sections. This is **recency-based**, not priority-based. Supersede with a budget-ordered assembly (active misconceptions → due_for_review → golden moments → traits → capped ledger) when the priority budget lands.

## Tests

- Unit: `scripts/test_mastery_aggregate.py`, `scripts/test_concept_alias_ledger.py`
- Integration arc: `scripts/test_alias_ledger_integration.py`
