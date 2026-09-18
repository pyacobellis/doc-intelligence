from doc_intelligence.config import DocumentTypeConfig


def build_parse_sql(cfg: DocumentTypeConfig) -> str:
    """SQL that parses every matching PDF in the source volume via ai_parse_document
    and (re)creates the parsed-docs table for this document type."""
    return f"""
        CREATE OR REPLACE TABLE {cfg.parsed_docs_full_name} AS
        SELECT
          _metadata.file_name                                   AS file_name,
          regexp_replace(_metadata.file_name, '\\\\.pdf$', '')  AS plan_name,
          content,
          ai_parse_document(content, MAP('version', '{cfg.ai_parse_version}')) AS parsed_content
        FROM READ_FILES('{cfg.source.volume}/*.pdf', format => 'binaryFile')
        WHERE _metadata.file_name LIKE '{cfg.source.file_pattern}'
    """


def build_parse_status_sql(cfg: DocumentTypeConfig) -> str:
    """Per-document parse status and how many structural elements were extracted."""
    return f"""
        SELECT
          plan_name,
          CASE WHEN is_variant_null(parsed_content:error_status) THEN 'OK' ELSE 'ERROR' END AS parse_status,
          size(try_cast(parsed_content:document:elements AS ARRAY<VARIANT>)) AS element_count
        FROM {cfg.parsed_docs_full_name}
        ORDER BY plan_name
    """


def build_element_type_counts_sql(cfg: DocumentTypeConfig) -> str:
    """Breakdown of parsed element types (text, table, section_header, ...) per document."""
    return f"""
        SELECT plan_name, element:type::STRING AS element_type, count(*) AS element_count
        FROM {cfg.parsed_docs_full_name}
        LATERAL VIEW explode(try_cast(parsed_content:document:elements AS ARRAY<VARIANT>)) AS element
        GROUP BY plan_name, element_type
        ORDER BY plan_name, element_count DESC
    """


def parse_documents(spark, cfg: DocumentTypeConfig):
    """Run the parse step (calls ai_parse_document; costs DBU) and return the status report."""
    spark.sql(build_parse_sql(cfg))
    return spark.sql(build_parse_status_sql(cfg))
