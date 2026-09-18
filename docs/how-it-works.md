# How it works

A stage-by-stage walkthrough of the pipeline, the tables it produces, and where each
decision lives. Read top to bottom the first time; afterwards the section headings work
as a map.

## 1. The shape of the system

Three layers:

| Layer | Lives in | Role |
|---|---|---|
| Config | `configs/<type>.yaml` | Everything specific to a document type: where the PDFs are, table names, extraction fields, rule taxonomy, unit spellings, prompts, chunking strategy, eval questions |
| Logic | `doc_intelligence/` (Python, tested) | Builds SQL, calls Databricks APIs, does the deterministic work in Python |
| Runtime | Databricks workspace | Unity Catalog tables and volumes, Vector Search, serverless compute, jobs, the Streamlit app |

The rule of thumb: if you would change it when onboarding a new document type, it is
config; if you would change it to make the framework better, it is code; if it is a
thing that exists in the workspace, it is runtime and the code should be able to recreate it.

`DocumentTypeConfig` (`config.py`) is the typed view of one YAML. Nearly every function
in the package takes `cfg` as an argument and derives table names, prompts and column
lists from it. `with_chunk_variant(cfg, "section_400")` returns a copy pointing at an
alternative chunk table and index — that is how chunking experiments coexist.

## 2. The pipeline, stage by stage

Each stage is one CLI command and one package function. Stages marked **$** call an AI
function and consume DBU.

### Parse — `doc-intel parse` → `<type>_parsed_docs` ($)

`parsing/pdf_parser.py`. One SQL statement reads every PDF matching `source.file_pattern`
from `source.volume` with `READ_FILES(... binaryFile)` and runs `ai_parse_document` on the
bytes. The result is one row per PDF with the raw `content` and a `parsed_content` VARIANT.

Inside `parsed_content:document:elements` is an ordered list of elements, each with `id`,
`type` (`text`, `table`, `section_header`, `title`, `figure`, `page_header`, `page_footer`,
`page_number`), `content` (tables arrive as HTML `<table>`), `bbox[0].page_id` and a
`confidence`. `parsing/elements.py` exposes that list as a SQL query
(`build_elements_sql`) that also tags every element with the most recent section header —
this is the backbone of both table extraction and section chunking.

The `plan_name` column is the document identifier (file name minus `.pdf`). The name is
a WSP hangover kept for compatibility with the live tables and index.

### Chunk — `doc-intel chunk` → `<type>_chunks` (($) W)

`retrieval/chunking.py` dispatches on `chunking.strategy`:

- `ai_prep_search` ($): Databricks' own chunker, run in SQL. Produces ~1,000-token chunks
  with 7–12 `Key: value` metadata lines prepended. `header_keys_to_keep` controls which of
  those lines stay in the embedded text; the whole header is preserved in `chunk_header`.
- `section` / `fixed` (free): Python chunkers in `retrieval/chunkers.py` over the parsed
  elements. `section` groups elements under their section header and packs to `max_chars`,
  prefixing the section title; `fixed` is a sliding window baseline.

Every strategy writes the same columns: `chunk_id, plan_name, file_name, chunk_index,
chunk_text, chunk_header, pages, section_reference, strategy`. `chunk_text` is what gets
embedded. See [chunking-and-retrieval.md](chunking-and-retrieval.md) for why chunk size
matters more than model choice here.

Rebuilding the chunks table breaks the Vector Search sync pipeline; the next stage repairs it.

### Index — `doc-intel index` (W)

`retrieval/vector_index.py`. Ensures the Vector Search endpoint and a Delta Sync index
with managed embeddings (`models.embedding`, currently `databricks-gte-large-en`) over the
chunks table exist, detects a failed sync pipeline and recreates the index if needed, then
triggers a sync. Idempotent; safe to re-run.

### Extract rules — `doc-intel extract-rules` → `<type>_rules` ($ W)

Two extraction paths, unioned into one table with a shared canonical schema
(`extraction/schema.py`: `plan_name, extraction_source, rule_type, <config fields...>,
unit_raw, page_id, citation_ids, confidence_score, source_id`).

- **Tables, deterministic** (`extraction/rules_from_tables.py`, free): parses each HTML
  table, forward-fills spanning cells, inherits headers across multi-page tables, aligns
  short continuation rows on the unit-bearing column, and emits a rule for every number
  paired with a known unit. Label columns map onto rule fields via
  `extraction.table_field_hints`; the rest fold into `condition`. `rule_type` comes from
  keyword classification (`extraction/taxonomy.py`). `doc-intel refresh-table-rules`
  re-runs just this path.
- **Prose, LLM** (`extraction/rules_from_text.py`, $): `ai_extract` v2.1 with a JSON
  schema generated from the config fields and taxonomy labels, over chunks that pass the
  `chunk_keyword_filter`. Returns citations and confidence scores. Units are normalised
  inline through the same alias table as the deterministic path.

The `Taxonomy` class is the one place both paths agree on vocabulary: rule type names,
keyword classification order (specific before generic), and unit aliases
(`megalitres per day` → `ML/day`).

### Retrieval and Q&A — `search`, `ask` ($ for `ask`)

`retrieval/search.py` queries the index (`ANN` semantic or `HYBRID` semantic + keyword,
with plan filters). `retrieval/hybrid.py` adds a local BM25 index and reciprocal-rank
fusion so the harness can compare the index's hybrid mode against a client-side one.
`retrieval/qa.py` builds a grounded prompt from the hits and calls `ai_query`; the answer
carries its sources.

### Comparison — `matrix`, `compare`

`comparison/cross_plan.py`. `rule_matrix` pivots rule counts per document × rule type.
`compare_rule_type` pulls every rule of one canonical type and `numeric_comparison` lays
the values out with one column per document — this is where the "canonical taxonomy"
pays off. `find_similar_provisions` uses the index to find each chunk's nearest
neighbours in *other* documents.

### Eval — `eval`

`eval/`. Questions live in `configs/<type>_eval_questions.yaml` with a category
(`factual | comparative | edge_case`), expected documents, optionally an expected snippet
(chunk-level ground truth) and a reference answer. `evaluate_retrieval` scores plan
recall, MRR, hit@k and snippet hit@k with no LLM calls; `evaluate_answers` adds generated
answers and an `ai_query` judge where a reference exists. Results can be appended to
`<type>_eval_results` with `--write`. Runs record the retriever and chunking variant, so
the summary table is the arbiter for retrieval/chunking decisions.

### Change detection — `registry`, `check-source`, `proposals`

Watch → propose → approve → apply. Deterministic where reliability matters, one bounded
LLM opinion where judgement helps, and a human gate before anything touches the corpus.

- **Watch** (`monitoring/source_scan.py`): the publisher's hub page links to region
  pages; each region page has one heading per plan with a "Read:" link to the instrument
  on legislation.nsw.gov.au (instrument id and consolidation date parsed from the URL —
  a free version label) and links to supporting documents (background, rule summary
  sheets, "changes" fact sheets, maps, gazette notices). The site layout is config
  (`source.region_link_pattern`, `instrument_link_pattern`, `supporting_kinds`); the
  scanner is generic. Plans listed on several region pages are merged, with the
  instrument link the majority of pages agree on.
- **Registry** (`monitoring/registry.py`, `<type>_document_registry`): the canonical
  identity of every document the site lists — plan key, current instrument/version,
  file name in the volume once ingested, status, confirmed flag. `registry seed` records
  the baseline (85 WSPs from 14 pages on first run); `source.known_documents` marks the
  ones already ingested. Every scan appends the links it saw to
  `<type>_source_observations` — the evidence trail.
- **Propose** (`check-source`): deltas between the site and the registry become rows in
  `<type>_change_proposals`: `new_plan`, `new_version` (instrument id/date changed),
  `draft`, `supporting_doc` (an unseen link), `withdrawn`. Each carries evidence and a
  recommended action; a dedupe key stops the same proposal being raised twice. A fuzzy
  match suggests which known document a new heading probably supersedes; `--llm` adds one
  `ai_query` opinion per new-document proposal (`monitoring/triage.py`).
- **Approve**: `proposals list|show|approve|reject` on the CLI, or the app's Changes tab.
- **Apply** (`monitoring/apply.py`): supporting documents are fetched into
  `<volume>/supporting/<plan_key>/`. Instruments live on legislation.nsw.gov.au, which
  blocks scripted downloads (including its documented export endpoints), so `apply` opens
  the URL in your browser, waits for the PDF to land in Downloads, uploads it under a
  canonical name, archives the superseded file, records the version and updates the
  registry. One click; then `pipeline` re-ingests.

`diff_texts` + `build_change_summary_sql` (`monitoring/change_detection.py`) turn two
versions of a document's text into an LLM-written change narrative; the impact step that
maps a diff onto affected rules is next.

### Cost — `cost`

`monitoring/cost.py` reads `system.billing.usage` (AI functions bill as serverless
real-time inference DBU) and `system.ai_gateway.usage` (per-endpoint requests/tokens).

## 3. Tables at a glance

| Table | Grain | Produced by |
|---|---|---|
| `<type>_parsed_docs` | one row per PDF | `parse` |
| `<type>_chunks` (+ `__<variant>`) | one row per chunk | `chunk` |
| `<type>_chunks_index` (+ `__<variant>`) | Vector Search index over chunks | `index` |
| `<type>_rules` | one row per extracted rule | `extract-rules` |
| `<type>_eval_results` | one row per question per run | `eval --write` |
| `<type>_document_versions` | one row per observed document version | `check-source --apply` |
| `<type>_change_log` | one row per detected change | change detection glue (pending) |

## 4. Runtime: how code reaches Databricks

`runtime.py` has two ways to run SQL behind one `SqlRunner` type:

- **Databricks Connect** (`get_spark`): a Spark session on serverless compute. Used by
  the CLI and tests locally (profile `wps_doc_intel`) and by jobs in the workspace.
- **Statement Execution API** (`warehouse_sql_runner`): for the Streamlit app, which has
  no Spark session. Values come back as strings.

Functions that only need to read take a `run_sql`; pipeline stages that write take
`spark`. The Databricks SDK `WorkspaceClient` covers Vector Search, files and pipelines.

## 5. Reads, writes and money

- Read-only: `status`, `search`, `matrix`, `compare`, `eval` (without `--answers`),
  `cost`, `check-source` (without `--apply`), every test.
- Workspace writes: `parse`, `chunk`, `index`, `extract-rules`, `refresh-table-rules`,
  `check-source --apply`, `eval --write`, `bundle deploy`. Tables are Delta, so an
  overwrite can be undone with `RESTORE TABLE … TO VERSION AS OF`.
- DBU: `ai_parse_document` (per PDF), `ai_prep_search` (per document), `ai_extract`
  (per filtered chunk), `ai_query` (per question/summary/judge). On the two WSPs a full
  pipeline is a few dollars at list price; `cost` shows the actuals.

## 6. Testing philosophy

Unit tests check generated SQL text and pure-Python logic on fixtures (HTML tables,
element lists, fake hits). Integration tests are marked and read real workspace state:
the parsed documents, chunks, rules, index, system tables. They never call an AI
function, so the suite is cheap to run after every change (~2 minutes). Read-only
integration tests are also how SQL expressions are checked against their Python twins
(header stripping) and how chunkers are exercised on real parsed documents without
writing anything.

## 7. What is deliberately not done yet

- Change-detection glue into one command; `source.listing_url` is unset.
- Small-to-big retrieval (retrieve small chunks, hand the LLM the parent section) and
  LLM reranking — both natural next steps after the chunking experiments.
- Incremental parsing (`parse` re-parses every PDF in the volume).
- The `plan_name` → `document_name` rename (schema change on live tables).
- Deploying the bundle (job + app) — `databricks bundle validate` passes.
