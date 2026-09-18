import json

from doc_intelligence.extraction.rules_from_text import (
    build_extraction_status_sql,
    build_keyword_filter_sql,
    build_text_extraction_sql,
    build_text_rules_flatten_sql,
)
from doc_intelligence.extraction.schema import extract_schema_json
from doc_intelligence.runtime import sql_literal


def test_text_extraction_sql_embeds_schema_options_and_filter(wsp_config):
    sql = build_text_extraction_sql(wsp_config)
    assert "ai_extract(" in sql
    assert "FROM workspace.default.wsp_chunks" in sql
    assert "'version', '2.1'" in sql
    assert "'enableCitations', 'true'" in sql and "'enableConfidenceScores', 'true'" in sql
    assert '"rule_type": {"type": "enum"' in sql
    assert "cease_to_pump" in sql
    assert "lower(chunk_text) LIKE '%cease%'" in sql


def test_keyword_filter_is_true_when_no_keywords(wsp_config):
    assert build_keyword_filter_sql(wsp_config).count(" OR ") == len(wsp_config.extraction.chunk_keyword_filter) - 1
    no_keywords = type(wsp_config.extraction)(
        ai_extract_version="2.1", chunk_keyword_filter=(), instructions="x", fields=wsp_config.extraction.fields,
    )
    cfg = type(wsp_config)(**{**wsp_config.__dict__, "extraction": no_keywords})
    assert build_keyword_filter_sql(cfg) == "true"


def test_flatten_sql_emits_canonical_columns_with_unit_normalisation(wsp_config):
    sql = build_text_rules_flatten_sql(wsp_config, "v")
    assert "FROM v" in sql
    assert "rule:value:value::DOUBLE AS value" in sql
    assert "rule:water_source:value::STRING AS water_source" in sql
    assert "WHEN 'megalitres per day' THEN 'ML/day'" in sql
    assert "rule:unit:value::STRING AS unit_raw" in sql
    assert "explode(try_cast(extracted:response:rules AS ARRAY<VARIANT>))" in sql
    assert "rule:rule_type:confidence_score::DOUBLE AS confidence_score" in sql
    assert "chunk_id AS source_id" in sql


def test_extraction_status_sql_groups_by_plan():
    sql = build_extraction_status_sql("v")
    assert "failed_extractions" in sql and "GROUP BY plan_name" in sql


def test_extract_schema_json_is_valid_with_taxonomy_labels(wsp_config):
    schema = json.loads(extract_schema_json(wsp_config))
    properties = schema["rules"]["items"]["properties"]
    assert properties["rule_type"]["labels"] == list(wsp_config.taxonomy.rule_type_names)
    assert set(wsp_config.extraction.field_names) <= set(properties)
    assert properties["value"]["type"] == "number"


def test_sql_literal_escapes_quotes_and_backslashes():
    assert sql_literal("it's") == "it''s"
    assert sql_literal("a\\b") == "a\\\\b"
