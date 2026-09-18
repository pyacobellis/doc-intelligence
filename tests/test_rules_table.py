import pytest

from doc_intelligence.extraction.rules_table import build_rules_quality_sql, build_rules_summary_sql
from doc_intelligence.extraction.schema import rule_columns, rules_spark_schema


def test_rules_summary_sql_targets_rules_table(wsp_config):
    sql = build_rules_summary_sql(wsp_config)
    assert "FROM workspace.default.wsp_rules" in sql
    assert "count(CASE WHEN value IS NOT NULL THEN 1 END)" in sql


def test_rules_quality_sql_includes_unit_and_value_metrics(wsp_config):
    sql = build_rules_quality_sql(wsp_config)
    assert sql.count("UNION ALL") == 3
    assert "Rules with normalised unit" in sql


def test_spark_schema_matches_canonical_columns(wsp_config):
    assert rules_spark_schema(wsp_config).fieldNames() == rule_columns(wsp_config)


@pytest.mark.integration
def test_existing_rules_table_is_readable(spark, wsp_config):
    df = spark.table(wsp_config.rules_full_name)
    assert {"plan_name", "rule_type", "value", "unit", "confidence_score"} <= set(df.columns)
