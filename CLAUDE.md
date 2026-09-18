# CLAUDE.md — working notes for AI sessions

Read this before touching anything. Humans: see README.md and docs/.

## What this is

A config-driven document intelligence framework on Databricks (`doc_intelligence/`).
First document type: NSW Water Sharing Plans (`configs/wsp.yaml`). Personal project on
the owner's own machine — do not attribute it to any employer or organisation.
Prototyped on Databricks Free Edition (no outbound network from the workspace, so all
LLM/embedding calls go through Databricks-native `ai_*` SQL functions and served models).

## Ground rules

1. **Reads vs writes.** Running SQL `SELECT`s, Vector Search queries, `workspace export`,
   local tests and `doc-intel status|search|matrix|compare|eval|cost` is fine without
   asking. Anything that changes workspace state — `parse`, `chunk`, `index`,
   `extract-rules`, `refresh-table-rules`, `check-source --apply`, `bundle deploy`,
   `workspace import`, dropping/creating tables or indexes — needs the owner's go-ahead.
   Say what will change and roughly what it costs before asking.
2. **Commits.** Commit only when asked (the owner has been happy with "commit and push
   each finished chunk of work", but confirm). Remote is `origin` → GitHub `main`.
   Attribution trailer per the session's system reminder.
3. **Tests never re-run AI functions.** Unit tests cover SQL text and pure Python;
   integration tests (`@pytest.mark.integration`) read tables/indexes that earlier
   pipeline runs produced. Keep it that way — `ai_parse_document`/`ai_extract` runs cost
   real DBU and take minutes.
4. **Architecture is settled:** logic in the package with tests; YAML per document type;
   Databricks objects are runtime; notebooks and the Streamlit app hold no logic.
5. Default to no comments; explain *why*, never *what*.

## Layout

```
doc_intelligence/
  config.py        typed DocumentTypeConfig from configs/<type>.yaml; with_chunk_variant()
  runtime.py       get_spark (Connect), get_workspace_client, SqlRunner (Spark or warehouse), sql_literal
  parsing/         pdf_parser (ai_parse_document SQL); elements (shared parsed-element query)
  retrieval/       chunking (strategy dispatch + ai_prep_search SQL), chunkers (section/fixed),
                   vector_index (endpoint/index/sync/self-repair), search (SearchHit, retrievers),
                   bm25, hybrid (RRF), qa (grounded answer), summaries
  extraction/      taxonomy (rule types, unit aliases), schema (rules columns, ai_extract schema),
                   rules_from_tables (deterministic HTML tables), rules_from_text (ai_extract), rules_table
  comparison/      cross_plan: rule matrix, side-by-side numeric comparison, similar provisions
  eval/            questions (YAML), metrics (plan + snippet level, LLM judge), runner
  monitoring/      change_detection (discover/hash/diff/upload/log), cost (system tables)
  cli.py           `doc-intel` — thin dispatch; scaffold.py — `new-config`
app/app.py         Streamlit; config switcher in the sidebar; uses the warehouse SqlRunner
configs/           wsp.yaml, wsp_eval_questions.yaml, _template.yaml
databricks.yml     DAB: doc_intel_pipeline (job parameter config_path), wsp_check_source, wsp_app
```

## How to run things

```bash
.venv/Scripts/python.exe -m pytest -q -p no:warnings          # ~2 min incl. integration
.venv/Scripts/doc-intel.exe --profile wps_doc_intel status    # read-only
databricks bundle validate -t dev                             # read-only
```

The `databricks` CLI lives at
`%LOCALAPPDATA%\Microsoft\WinGet\Packages\Databricks.DatabricksCLI_Microsoft.Winget.Source_8wekyb3d8bbwe`
and must be on PATH for SDK auth (`export PATH="$PATH:/c/Users/Owner/AppData/Local/..."` in Bash).
Profile: `wps_doc_intel`. Workspace: `workspace.default`. Warehouse: "Serverless Starter Warehouse".

## Gotchas learned the hard way

- **Spark SQL escaping:** regex arguments need `\\n` in the SQL text; plain string
  separators need `\n`. `regexp_replace(..., '\\.{3,}.*$', '')` is written as `'\\\\.{{3,}}.*$'`
  inside a Python f-string.
- **Bash heredocs in this tool collapse `\\` to `\`.** Write any file containing backslashes
  with the Write tool, not `cat <<EOF`.
- **`CREATE OR REPLACE` / overwrite of the chunks table breaks the Delta Sync index**
  ("Failed to resolve flow '__online_index_view'"). `build_index` detects the failed
  pipeline and drops/recreates the index (~5 min). Expect this after every `chunk`.
- **`ai_parse_document` elements** expose `id`, `type`, `content` (tables as HTML),
  `bbox[0].page_id`, `confidence` — not `element_id`/`text`. Multi-page tables carry a
  header only on the first page and drop the spanning leading cell on later pages;
  `rules_from_tables` inherits headers and re-aligns rows on the unit column.
- **`ai_prep_search`** prepends 7–12 `Key: value` lines to every chunk and produces
  ~1,000-token chunks; `databricks-gte-large-en` embeds 512 tokens. Hence header
  stripping (`chunking.header_keys_to_keep`) and the `section` strategy.
- **Variant tables/indexes** are named `<chunks>__<variant>` / `<index>__<variant>`;
  `--variant` on the CLI or `with_chunk_variant()` in code.
- The document identifier column is `plan_name` everywhere (WSP naming kept for
  compatibility with live tables/index). Eval YAML accepts `expected_plans` or
  `expected_documents`.
- Free Edition: `system.billing.usage` works; SQL `ai_*` calls appear there as
  `PREMIUM_SERVERLESS_REAL_TIME_INFERENCE` with no per-token rows.
- Vector Search filters: `{"plan_name": [...]}` include, `{"plan_name NOT": [...]}` exclude.

## Where the owner's context lives

Auto-memory for this project (outside the repo) holds the roadmap, decisions and
session state. docs/ holds the durable explanations. When they disagree, the code wins.
