from __future__ import annotations

from doc_intelligence.config import DocumentTypeConfig
from doc_intelligence.extraction.rules_from_tables import extract_table_rules
from doc_intelligence.extraction.rules_from_text import extract_text_rules
from doc_intelligence.extraction.schema import rule_columns, rules_spark_schema


def build_rules_table(spark, cfg: DocumentTypeConfig):
    """Union deterministic table rules with LLM text rules and overwrite the rules table.
    Calls ai_extract (costs DBU)."""
    columns = rule_columns(cfg)
    table_rules = spark.createDataFrame(extract_table_rules(spark, cfg), schema=rules_spark_schema(cfg))
    text_rules = extract_text_rules(spark, cfg).select(*columns)
    all_rules = table_rules.select(*columns).unionByName(text_rules)
    all_rules.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(cfg.rules_full_name)
    return spark.sql(build_rules_summary_sql(cfg))


def refresh_table_rules(spark, cfg: DocumentTypeConfig):
    """Replace only the deterministic (table) rules, leaving LLM-extracted rows untouched.
    No AI calls, so the deterministic path can be iterated cheaply."""
    columns = rule_columns(cfg)
    table_rules = spark.createDataFrame(extract_table_rules(spark, cfg), schema=rules_spark_schema(cfg))
    spark.sql(f"DELETE FROM {cfg.rules_full_name} WHERE extraction_source = 'table'")
    table_rules.select(*columns).write.mode("append").saveAsTable(cfg.rules_full_name)
    return spark.sql(build_rules_summary_sql(cfg))


def _numeric_value_expr(cfg: DocumentTypeConfig) -> str:
    if "value" in cfg.extraction.field_names:
        return "count(CASE WHEN value IS NOT NULL THEN 1 END)"
    return "CAST(NULL AS BIGINT)"


def build_rules_summary_sql(cfg: DocumentTypeConfig) -> str:
    return f"""
        SELECT
          plan_name,
          rule_type,
          extraction_source,
          count(*)                          AS rule_count,
          {_numeric_value_expr(cfg)}        AS rules_with_numeric_value,
          round(avg(confidence_score), 3)   AS avg_confidence_score
        FROM {cfg.rules_full_name}
        GROUP BY plan_name, rule_type, extraction_source
        ORDER BY plan_name, rule_count DESC
    """


def build_rules_quality_sql(cfg: DocumentTypeConfig) -> str:
    rules = cfg.rules_full_name
    parts = [
        f"SELECT 'Total rules' AS metric, count(*) AS count FROM {rules}",
        f"SELECT 'Rules with confidence score', count(*) FROM {rules} WHERE confidence_score IS NOT NULL",
    ]
    if "value" in cfg.extraction.field_names:
        parts.append(f"SELECT 'Rules with numeric value', count(*) FROM {rules} WHERE value IS NOT NULL")
    if "unit" in cfg.extraction.field_names:
        parts.append(f"SELECT 'Rules with normalised unit', count(*) FROM {rules} WHERE unit IS NOT NULL")
    return "\nUNION ALL\n".join(parts)
