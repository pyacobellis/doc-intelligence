from __future__ import annotations

import json

from doc_intelligence.config import DocumentTypeConfig

CORE_PREFIX_COLUMNS = ("plan_name", "extraction_source", "rule_type")
CORE_SUFFIX_COLUMNS = ("unit_raw", "page_id", "citation_ids", "confidence_score", "source_id")

_SQL_TYPES = {"string": "STRING", "number": "DOUBLE", "integer": "BIGINT", "boolean": "BOOLEAN"}


def rule_columns(cfg: DocumentTypeConfig) -> list[str]:
    """Canonical column order of the rules table: fixed core + config-driven fields."""
    return [*CORE_PREFIX_COLUMNS, *cfg.extraction.field_names, *CORE_SUFFIX_COLUMNS]


def sql_type_for(field_type: str) -> str:
    return _SQL_TYPES.get(field_type, "STRING")


def rules_spark_schema(cfg: DocumentTypeConfig):
    from pyspark.sql import types as T

    spark_types = {
        "string": T.StringType(),
        "number": T.DoubleType(),
        "integer": T.LongType(),
        "boolean": T.BooleanType(),
    }
    fields = [T.StructField(name, T.StringType()) for name in CORE_PREFIX_COLUMNS]
    fields += [T.StructField(f.name, spark_types.get(f.type, T.StringType())) for f in cfg.extraction.fields]
    fields += [
        T.StructField("unit_raw", T.StringType()),
        T.StructField("page_id", T.IntegerType()),
        T.StructField("citation_ids", T.ArrayType(T.IntegerType())),
        T.StructField("confidence_score", T.DoubleType()),
        T.StructField("source_id", T.StringType()),
    ]
    return T.StructType(fields)


def build_extract_schema(cfg: DocumentTypeConfig) -> dict:
    """ai_extract v2.1 schema (lowercase JSON types). The rule_type enum is driven by the
    taxonomy so the LLM and the deterministic path share one vocabulary."""
    properties = {
        "rule_type": {
            "type": "enum",
            "labels": list(cfg.taxonomy.rule_type_names),
            "description": "Type of rule. "
            + " ".join(f"{rt.name}: {rt.description}." for rt in cfg.taxonomy.rule_types if rt.description),
        }
    }
    for field in cfg.extraction.fields:
        properties[field.name] = {"type": field.type, "description": field.description}
    return {
        "rules": {
            "type": "array",
            "description": "List of rules found in this text",
            "items": {"type": "object", "properties": properties},
        }
    }


def extract_schema_json(cfg: DocumentTypeConfig) -> str:
    return json.dumps(build_extract_schema(cfg))
