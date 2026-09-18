from doc_intelligence.config import DocumentTypeConfig


def build_chunk_sql(cfg: DocumentTypeConfig) -> str:
    """SQL that explodes each parsed document into chunks via ai_prep_search and
    (re)creates the chunks table, with Change Data Feed enabled (required for a
    Vector Search Delta Sync index)."""
    return f"""
        CREATE OR REPLACE TABLE {cfg.chunks_full_name}
        TBLPROPERTIES (delta.enableChangeDataFeed = true)
        AS
        SELECT
          md5(concat(plan_name, '_', cast(chunk:chunk_position AS STRING))) AS chunk_id,
          plan_name,
          file_name,
          chunk:chunk_position::INT    AS chunk_index,
          chunk:chunk_to_embed::STRING AS chunk_text,
          array_join(try_cast(chunk:pages AS ARRAY<STRING>), ',') AS pages
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
        SELECT plan_name, count(*) AS chunk_count, round(avg(length(chunk_text))) AS avg_chunk_len
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


def chunk_documents(spark, cfg: DocumentTypeConfig):
    """Run the chunking step (calls ai_prep_search; costs DBU) and return the per-plan summary."""
    spark.sql(build_chunk_sql(cfg))
    for statement in build_vector_search_prereqs_sql(cfg):
        spark.sql(statement)
    return spark.sql(build_chunk_summary_sql(cfg))
