# Chunking and retrieval

How chunks are made, how alternatives are compared, and what the numbers say so far.

## Why chunking is the lever

The embedding model available on Free Edition (`databricks-gte-large-en`, also
`bge-large-en`) embeds at most **512 tokens**. `ai_prep_search` produces chunks of
~4,400 characters (~1,000+ tokens), so roughly the back half of every chunk was never
part of its embedding — a semantic query could only ever match the first half. That is
the strongest single argument in this project for treating chunking as a first-class,
measured decision rather than a default.

A second, smaller effect: `ai_prep_search` prepends 7–12 `Key: value` metadata lines to
every chunk (about 14% of the text). Some describe the chunk (`Sections`, `Tables`,
`Contains`) and are worth keeping at the top, inside the embedded window; the rest is
per-document boilerplate (Act, Minister, publication date, title, page header).

## Strategies

Configured in `chunking.strategy`; implemented in `retrieval/chunking.py` (SQL) and
`retrieval/chunkers.py` (Python).

| Strategy | Cost | Chunk shape | Notes |
|---|---|---|---|
| `ai_prep_search` | AI call per document | ~1,000 tokens, metadata header | Databricks-native; `header_keys_to_keep` filters the header |
| `section` | none | elements grouped under their section header, packed to `max_chars`, title prefixed | Keeps legal structure; tables flattened to `a \| b \| c` rows; `section_reference` and `pages` populated |
| `fixed` | none | sliding windows of `max_chars` with `overlap_chars` | The naive baseline the others should beat |

All strategies write the same columns, so search, Q&A, eval and the app are agnostic.
On the two WSPs, `section` at `max_chars: 1600` (~400 tokens) yields 228 chunks averaging
~800 characters versus 54 chunks at ~4,400 for `ai_prep_search`.

## Variants: running alternatives side by side

`chunking.variants` names alternative settings. Selecting one with `--variant` (CLI) or
`with_chunk_variant()` (code) redirects the chunk table and index:

```
--variant section_400  →  wsp_chunks__section_400 / wsp_chunks_index__section_400
```

Workflow for an A/B:

```bash
doc-intel --variant section_400 chunk       # free for section/fixed
doc-intel --variant section_400 index       # new index on the shared endpoint (~5 min first time)
doc-intel --variant section_400 eval --retriever vs_hybrid --verbose
doc-intel eval --retriever vs_hybrid --verbose            # baseline for comparison
```

Results carry `variant` and `index_name`; `summarise_results` labels rows
`vs_hybrid@section_400` vs `vs_hybrid@base`. Add `--write` to keep runs in
`<type>_eval_results`.

## Retrievers

| Name | What it does |
|---|---|
| `vs_ann` | Vector Search, semantic only |
| `vs_hybrid` | Vector Search built-in hybrid (semantic + keyword) — the production default |
| `bm25` | Local BM25 over the chunk corpus (`retrieval/bm25.py`); no Databricks calls |
| `rrf` | ANN candidates fused with BM25 by reciprocal rank fusion (`retrieval/hybrid.py`) |

The client-side ones exist to test hypotheses, not to ship: if `rrf` beats `vs_hybrid`,
that is a signal about the index's keyword weighting, and the fix belongs in the index
configuration, not in the app.

## Metrics

Plan-level (need only `expected_plans` on a question): plan recall, reciprocal rank,
hit@k. Chunk-level (need `expected_snippet`, a phrase copied from the PDF): snippet rank
and snippet hit@k. Answer-level (need `expected_answer`, cost an `ai_query` each): LLM
judge score. Plan-level metrics saturate quickly with two documents; snippet-level ones
are what actually discriminate chunkings.

## What the numbers say so far (two WSPs, seven draft questions)

- `vs_hybrid` and `rrf` were perfect on plan-level metrics; `vs_ann` missed one document
  on a comparative question; `bm25` ranked the wrong document first once. Hybrid stays
  the default.
- Stripping the metadata header moved one factual question's MRR from 1.0 to 0.75. Best
  guess: the `Title: <plan name>` line was helping identify the document. Worth testing
  `Title` in `header_keys_to_keep` — with the harness, not by argument.
- First A/B, `vs_hybrid` on both: `section_400` beat the `ai_prep_search` baseline —
  factual MRR 0.75 → 1.0 (the cease-to-pump question moved from rank 4 to rank 1) and
  the "198 ML/day" snippet from rank 3 to rank 2, with every other metric unchanged.
  Small sample, but it points the same way as the 512-token argument. `section_400` is
  the leading candidate to become the default once more questions carry snippets.

## Adding a strategy

1. Implement `def my_chunks(elements, cfg: ChunkingConfig) -> list[Chunk]` in
   `retrieval/chunkers.py` and wire it into `chunk_elements`.
2. Add the name to `CHUNKING_STRATEGIES` in `config.py`.
3. Unit-test it on the element fixtures in `tests/test_chunkers.py`; add a read-only
   integration test over the real parsed documents if it has data-dependent behaviour.
4. Add a variant to the config and run the A/B above.

## Next steps this design is set up for

- **Small-to-big**: retrieve small chunks, expand to the parent section
  (`section_reference` is already on every chunk) before prompting the LLM.
- **LLM reranking**: retrieve 20–30 candidates, score them with `ai_query`, keep five.
  Cheap on short chunks and usually worth more than a better embedding model.
- **Model A/B**: `models.embedding: databricks-bge-large-en` plus a rebuilt index is a
  ten-minute experiment once the harness is trusted.
