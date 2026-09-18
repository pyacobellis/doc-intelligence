from __future__ import annotations

import re
from typing import Sequence

from doc_intelligence.config import DocumentTypeConfig

# "Key: value" lines that ai_prep_search prepends to every chunk's embed text.
_HEADER_LINE = re.compile(r"^[^\n:]{1,40}: [^\n]*$")
_HEADER_BLOCK_SQL_REGEX = r"^((?:[^\\n:]{1,40}: [^\\n]*\\n)+)"


def split_metadata_header(text: str | None) -> tuple[list[str], str]:
    """Leading 'Key: value' lines vs. the rest of the chunk."""
    lines = (text or "").split("\n")
    count = 0
    while count < len(lines) and _HEADER_LINE.match(lines[count]):
        count += 1
    if count == len(lines):  # header only, nothing after it: treat as body
        return [], text or ""
    return lines[:count], "\n".join(lines[count:])


def clean_chunk_text(text: str | None, keep_keys: Sequence[str] | None) -> str:
    """Python twin of build_clean_chunk_text_sql; keeps only the named header keys."""
    if keep_keys is None:
        return text or ""
    header, body = split_metadata_header(text)
    kept = [line for line in header if any(line.startswith(f"{key}: ") for key in keep_keys)]
    return "\n".join([*kept, body]) if kept else body


def build_clean_chunk_text_sql(column: str, keep_keys: Sequence[str] | None) -> str:
    """Spark SQL expression producing the same result as clean_chunk_text."""
    if keep_keys is None:
        return column
    header = f"regexp_extract({column}, '{_HEADER_BLOCK_SQL_REGEX}', 1)"
    body = f"substring({column}, length({header}) + 1)"
    if not keep_keys:
        return body
    alternation = "|".join(re.escape(key) for key in keep_keys)
    # split() takes a regex ('\\n' -> \n pattern); the join separators must be a real newline ('\n')
    kept = f"filter(split({header}, '\\\\n'), line -> line RLIKE '^({alternation}): ')"
    return f"CASE WHEN size({kept}) > 0 THEN concat(array_join({kept}, '\\n'), '\\n', {body}) ELSE {body} END"


def build_chunk_sql(cfg: DocumentTypeConfig) -> str:
    """ai_prep_search strategy: explode each parsed document into chunks in SQL and
    (re)create the chunks table with Change Data Feed enabled (required for a Vector
    Search Delta Sync index). chunk_text is what gets embedded."""
    embed_text = "chunk:chunk_to_embed::STRING"
    return f"""
        CREATE OR REPLACE TABLE {cfg.chunks_full_name}
        TBLPROPERTIES (delta.enableChangeDataFeed = true)
        AS
        SELECT
          md5(concat(plan_name, '_', cast(chunk:chunk_position AS STRING))) AS chunk_id,
          plan_name,
          file_name,
          chunk:chunk_position::INT AS chunk_index,
          {build_clean_chunk_text_sql(embed_text, cfg.chunking.header_keys_to_keep)} AS chunk_text,
          regexp_extract({embed_text}, '{_HEADER_BLOCK_SQL_REGEX}', 1) AS chunk_header,
          array_join(transform(try_cast(chunk:pages AS ARRAY<VARIANT>), p -> p:page_id::STRING), ',') AS pages,
          CAST(NULL AS STRING) AS section_reference,
          'ai_prep_search' AS strategy
        FROM (
          SELECT
            plan_name,
            file_name,
            explode(try_cast(ai_prep_search(parsed_content):document:contents AS ARRAY<VARIANT>)) AS chunk
          FROM {cfg.parsed_docs_full_name}
          WHERE is_variant_null(parsed_content:error_status)
        )
    """


def build_chunk_summary_sql(cfg: DocumentTypeConfig) -> str:
    return f"""
        SELECT
          plan_name,
          any_value(strategy) AS strategy,
          count(*) AS chunk_count,
          round(avg(length(chunk_text))) AS avg_chunk_len,
          max(length(chunk_text)) AS max_chunk_len
        FROM {cfg.chunks_full_name}
        GROUP BY plan_name
        ORDER BY plan_name
    """


def build_vector_search_prereqs_sql(cfg: DocumentTypeConfig) -> list[str]:
    """DDL a Vector Search Delta Sync index needs: NOT NULL + PK on the key column."""
    return [
        f"ALTER TABLE {cfg.chunks_full_name} ALTER COLUMN chunk_id SET NOT NULL",
        f"ALTER TABLE {cfg.chunks_full_name} ADD CONSTRAINT {cfg.tables.chunks}_pk PRIMARY KEY (chunk_id)",
    ]


def write_chunks(spark, cfg: DocumentTypeConfig, chunks) -> None:
    """Overwrite the chunks table from Python-built chunks (section/fixed strategies)."""
    from doc_intelligence.retrieval.chunkers import chunks_frame, chunks_spark_schema

    df = spark.createDataFrame(chunks_frame(chunks), schema=chunks_spark_schema())
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(cfg.chunks_full_name)
    spark.sql(f"ALTER TABLE {cfg.chunks_full_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")


def chunk_documents(spark, cfg: DocumentTypeConfig):
    """Run the configured chunking strategy and return the per-plan summary.
    ai_prep_search calls an AI function (costs DBU); section/fixed are pure Python over
    the already-parsed elements. Either way the table is recreated, which breaks the
    Delta Sync index until `build_index` repairs it."""
    if cfg.chunking.strategy == "ai_prep_search":
        spark.sql(build_chunk_sql(cfg))
    else:
        from doc_intelligence.parsing.elements import load_elements
        from doc_intelligence.retrieval.chunkers import chunk_elements

        write_chunks(spark, cfg, chunk_elements(load_elements(spark, cfg), cfg.chunking))
    for statement in build_vector_search_prereqs_sql(cfg):
        spark.sql(statement)
    return spark.sql(build_chunk_summary_sql(cfg))
