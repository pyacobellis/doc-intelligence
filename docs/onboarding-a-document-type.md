# Onboarding a new document type

The framework claim is that a new document type needs a new config, not new code. This
is the recipe, followed by an honest list of what does and does not generalise today.

Good candidates for a second type have **tables plus numeric rules with units** — an
enterprise agreement or award (rates, allowances, hours), a council fees-and-charges
schedule, an insurance PDS (limits and excesses), a set of strata by-laws. Two or three
PDFs is enough.

## 1. Scaffold the config

```bash
doc-intel new-config award --display-name "Modern Awards" --file-pattern "AWARD_%"
```

This writes `configs/award.yaml` (from `configs/_template.yaml`) and
`configs/award_eval_questions.yaml`, and checks the result parses. Options:
`--volume` (default `/Volumes/workspace/default/raw`, shared with WSP — file patterns
must not overlap), `--force` to overwrite.

## 2. Edit the parts that carry domain knowledge

Open `configs/award.yaml`. The template ships generic placeholders in five places; these
are where the document type's vocabulary goes:

| Section | What to write |
|---|---|
| `extraction.fields` | The columns you want in the rules table (each becomes an `ai_extract` field). Keep `value` and `unit` if the documents have numbers with units. |
| `extraction.chunk_keyword_filter` | Words that mark a chunk as worth sending to `ai_extract`. Cost control, not correctness — err on the inclusive side. |
| `extraction.table_field_hints` | Header keywords that map a table's label columns onto rule fields (e.g. `Classification` → `applies_to`). |
| `taxonomy.rule_types` | The canonical rule vocabulary, specific types first, `other` last. Keywords drive deterministic classification and the `ai_extract` enum. |
| `taxonomy.unit_aliases` | Every spelling of every unit you expect, mapped to one canonical form. |
| `qa.*`, `eval.judge_prompt` | Prompts; mention the document type by name. |
| `chunking` | `section` is the default for a new type (no AI call, fits the embedding window). `ai_prep_search` is available as a variant. |

You can iterate on all of this without paying for `ai_extract`: the deterministic table
path is free (`refresh-table-rules`) and the taxonomy is unit-tested against fixture text.

## 3. Put the PDFs in the volume

Upload via the Catalog UI, or:

```bash
databricks fs cp AWARD_Clerks_2020.pdf dbfs:/Volumes/workspace/default/raw/ --profile wps_doc_intel
```

File names must match `source.file_pattern`; the name without `.pdf` becomes the
document identifier (`plan_name` column).

## 4. Run the pipeline

```bash
doc-intel --config configs/award.yaml --profile wps_doc_intel pipeline
doc-intel --config configs/award.yaml --profile wps_doc_intel status
```

`pipeline` = `parse` (one `ai_parse_document` per PDF) → `chunk` → `index` (creates
`award_chunks_index` on the shared endpoint; first build takes a few minutes) →
`extract-rules`. Expect the first run to surface taxonomy gaps: check
`doc-intel --config configs/award.yaml matrix` for a large `other` bucket and
`compare other` to see what fell through, then tighten the keywords and
`refresh-table-rules`.

## 5. Look at it

```bash
doc-intel --config configs/award.yaml search "minimum hourly rate for a level 2 clerk" --type HYBRID
doc-intel --config configs/award.yaml compare rate
doc-intel --config configs/award.yaml eval --retriever vs_hybrid --verbose
```

The Streamlit app has a document-type selector in the sidebar (`streamlit run app/app.py`
locally with `DATABRICKS_CONFIG_PROFILE=wps_doc_intel`). The bundle's pipeline job takes
`config_path` as a job parameter:
`databricks bundle run doc_intel_pipeline --params config_path=configs/award.yaml`.

## 6. Write real eval questions

Replace the three placeholders in `configs/award_eval_questions.yaml`. Retrieval metrics
need only `expected_plans` (document names) and, ideally, an `expected_snippet` copied
from the PDF; answer grading needs `expected_answer`.

## What generalises today, and what does not

Generalises (config only): source location and file pattern, all table/index names,
extraction fields and their SQL types, table-to-field hints, rule taxonomy and unit
normalisation, keyword pre-filter, prompts, chunking strategy and variants, eval
questions, models.

Does not, yet:

- The document identifier column is called `plan_name` in every table (kept for the
  live WSP tables and index). Cosmetic, but visible.
- `check-source` assumes one listing page of PDF links per document type; other
  discovery mechanisms need code.
- The Vector Search endpoint is shared; each type adds an index (fine on Free Edition so
  far, but there will be a limit).
- `parse` re-parses everything in the volume that matches the pattern; there is no
  incremental mode.
- The app shows one document type at a time; cross-type views do not exist.

Anything that turns out to be WSP-specific during onboarding is a bug in the framework —
fix it in code and add it to this list only if it genuinely cannot be config.
