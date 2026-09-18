# doc-intelligence

A config-driven document intelligence framework on Databricks. Point it at a folder of
PDFs and a YAML config describing the document type, and it parses, chunks, indexes,
extracts structured rules, answers grounded questions, compares documents, watches the
source for amendments and tracks what it all costs.

The first document type is the **NSW Water Sharing Plans (WSP)** — regulatory
instruments that set water extraction rules. Everything WSP-specific lives in
`configs/wsp.yaml`; onboarding a second document type means writing another YAML, not
touching the pipeline.

Personal project; prototyped on Databricks Free Edition using the native `ai_*` SQL
functions (`ai_parse_document`, `ai_prep_search`, `ai_extract`, `ai_query`).

## How it fits together

```
PDFs in a UC volume
   │  ai_parse_document                       parsing/
   ▼
<type>_parsed_docs ──► elements (text, tables, section headers, pages)
   │                                           │
   │  chunking strategy (ai_prep_search |      │  deterministic table rules  extraction/
   │  section | fixed)          retrieval/     │  + ai_extract text rules
   ▼                                           ▼
<type>_chunks ──► Vector Search index     <type>_rules  (canonical taxonomy, units)
   │                                           │
   ├─ search / hybrid retrieval / grounded Q&A │  comparison/  (rule matrix, side-by-side)
   ├─ eval harness (plan + snippet metrics)    │
   └─ change detection + cost tracking      monitoring/
```

Business logic is the Python package `doc_intelligence/`. Databricks objects (tables,
Vector Search, jobs, the Streamlit app) are the runtime. Notebooks under `notebooks/`
are exported snapshots of the original exploration and are not part of the pipeline.

Read [docs/how-it-works.md](docs/how-it-works.md) for the walkthrough.

## Quick start

Prerequisites: Python 3.11+, the Databricks CLI, and an authenticated profile
(`databricks auth login --host <workspace-url> --profile wps_doc_intel`).

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -r requirements-dev.txt                 # package + app + dev extras, editable
pytest                                              # unit + read-only integration tests
doc-intel --profile wps_doc_intel status            # read-only health check
```

## Command cheat-sheet

Everything is `doc-intel [--config configs/<type>.yaml] [--profile <p>] [--variant <v>] <command>`.
Commands marked **$** call AI functions and consume DBU; commands marked **W** write to the
workspace. Everything else is read-only.

| Command | What it does | |
|---|---|---|
| `status` | Parse status, chunk stats, rules summary, index state | |
| `parse` | `ai_parse_document` over the volume → `<type>_parsed_docs` | $ W |
| `chunk` | Run the configured chunking strategy → `<type>_chunks` | ($) W |
| `index` | Create/repair the Vector Search index and sync it | W |
| `extract-rules` | Table rules + `ai_extract` text rules → `<type>_rules` | $ W |
| `refresh-table-rules` | Re-run only the deterministic table extraction | W |
| `pipeline` | `parse` → `chunk` → `index` → `extract-rules` | $ W |
| `search "q" [--type ANN\|HYBRID]` | Query the index | |
| `ask "q"` | Grounded, cited answer via `ai_query` | $ |
| `summarise` | One LLM summary per document | $ |
| `matrix` | Rule counts per document × rule type | |
| `compare <rule_type>` | One rule type across documents, values side by side | |
| `eval [--retriever …] [--answers] [--write]` | Run the question set; `--answers` adds LLM answers + judge | ($) (W) |
| `check-source [--apply]` | Poll the listing page for new/changed PDFs; `--apply` uploads them | (W) |
| `cost [--days N]` | AI-function DBU/cost and token usage from system tables | |
| `new-config <type> --display-name "…"` | Scaffold a config for a new document type | |

`--variant section_400` points chunk/index/search/eval at an alternative chunking (own
table + index) so strategies can be compared with the eval harness.

## Repository layout

```
configs/            one YAML per document type (+ its eval questions); _template.yaml
doc_intelligence/   the package: config, runtime, parsing/, retrieval/, extraction/,
                    comparison/, eval/, monitoring/, cli, scaffold
app/                Streamlit Databricks App (thin: calls package functions only)
tests/              pytest; integration tests are read-only against the workspace
databricks.yml      Asset Bundle: pipeline job (any config), check-source job, app
notebooks/          exported snapshots of the original exploration (not run)
docs/               walkthroughs
```

## Docs

- [How it works](docs/how-it-works.md) — stage by stage, tables, config vs code, costs, tests
- [Onboarding a document type](docs/onboarding-a-document-type.md)
- [Chunking and retrieval](docs/chunking-and-retrieval.md) — strategies, variants, the eval loop

## Ground rules

- Reads are free to run. Anything that writes to the workspace (`parse`, `chunk`,
  `index`, `extract-rules`, `check-source --apply`, `bundle deploy`) is deliberate.
- Nothing is "done" until it is in the package with a test around it.
- Tests never re-run AI functions; they read what previous pipeline runs produced.
