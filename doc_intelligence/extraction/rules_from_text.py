from __future__ import annotations

from doc_intelligence.config import DocumentTypeConfig
from doc_intelligence.extraction.schema import extract_schema_json, rule_columns, sql_type_for
from doc_intelligence.extraction.taxonomy import Taxonomy
from doc_intelligence.runtime import sql_literal


def build_keyword_filter_sql(cfg: DocumentTypeConfig, column: str = "chunk_text") -> str:
    keywords = cfg.extraction.chunk_keyword_filter
    if not keywords:
        return "true"
    return " OR ".join(f"lower({column}) LIKE '%{kw.lower()}%'" for kw in keywords)


def build_text_extraction_sql(cfg: DocumentTypeConfig) -> str:
    """ai_extract over rule-bearing chunks. The keyword pre-filter keeps LLM cost down."""
    schema = sql_literal(extract_schema_json(cfg))
    instructions = sql_literal(cfg.extraction.instructions)
    return f"""
        SELECT
          plan_name,
          chunk_id,
          chunk_index,
          ai_extract(
            chunk_text,
            '{schema}',
            MAP(
              'version', '{cfg.extraction.ai_extract_version}',
              'enableCitations', 'true',
              'enableConfidenceScores', 'true',
              'instructions', '{instructions}'
            )
          ) AS extracted
        FROM {cfg.chunks_full_name}
        WHERE ({build_keyword_filter_sql(cfg)})
    """


def build_text_rules_flatten_sql(cfg: DocumentTypeConfig, extracted_view: str) -> str:
    """Flatten the ai_extract v2.1 response into one row per rule with the canonical columns."""
    taxonomy = Taxonomy(cfg.taxonomy)
    field_exprs = []
    for field in cfg.extraction.fields:
        raw = f"rule:{field.name}:value::{sql_type_for(field.type)}"
        if field.name == "unit":
            field_exprs.append(f"{taxonomy.unit_case_sql(raw)} AS unit")
        else:
            field_exprs.append(f"{raw} AS {field.name}")
    fields_sql = ",\n          ".join(field_exprs)
    unit_raw = "rule:unit:value::STRING" if "unit" in cfg.extraction.field_names else "CAST(NULL AS STRING)"
    return f"""
        SELECT
          plan_name,
          'text' AS extraction_source,
          rule:rule_type:value::STRING AS rule_type,
          {fields_sql},
          {unit_raw} AS unit_raw,
          CAST(NULL AS INT) AS page_id,
          try_cast(rule:rule_type:citation_ids AS ARRAY<INT>) AS citation_ids,
          rule:rule_type:confidence_score::DOUBLE AS confidence_score,
          chunk_id AS source_id
        FROM {extracted_view}
        LATERAL VIEW explode(try_cast(extracted:response:rules AS ARRAY<VARIANT>)) AS rule
        WHERE is_variant_null(extracted:error_message)
          AND rule IS NOT NULL
    """


def build_extraction_status_sql(extracted_view: str) -> str:
    return f"""
        SELECT
          plan_name,
          count(*) AS chunks_processed,
          sum(CASE WHEN is_variant_null(extracted:error_message) THEN 1 ELSE 0 END) AS successful_extractions,
          sum(CASE WHEN NOT is_variant_null(extracted:error_message) THEN 1 ELSE 0 END) AS failed_extractions
        FROM {extracted_view}
        GROUP BY plan_name
        ORDER BY plan_name
    """


def extract_text_rules(spark, cfg: DocumentTypeConfig):
    """LLM rule extraction. Calls ai_extract for real (costs DBU)."""
    view = f"{cfg.document_type}_text_rules_extracted"
    spark.sql(build_text_extraction_sql(cfg)).createOrReplaceTempView(view)
    return spark.sql(build_text_rules_flatten_sql(cfg, view)).select(*rule_columns(cfg))
