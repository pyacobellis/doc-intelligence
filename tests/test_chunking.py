import pytest

from doc_intelligence.retrieval.chunking import (
    build_chunk_sql,
    build_chunk_summary_sql,
    build_vector_search_prereqs_sql,
)


def test_build_chunk_sql_reads_parsed_docs_and_writes_chunks(wsp_config):
    sql = build_chunk_sql(wsp_config)

    assert "FROM workspace.default.wsp_parsed_docs" in sql
    assert "CREATE OR REPLACE TABLE workspace.default.wsp_chunks" in sql
    assert "ai_prep_search" in sql
    assert "delta.enableChangeDataFeed = true" in sql


def test_build_chunk_summary_sql_targets_chunks_table(wsp_config):
    sql = build_chunk_summary_sql(wsp_config)

    assert "FROM workspace.default.wsp_chunks" in sql
    assert "chunk_count" in sql


def test_build_vector_search_prereqs_sets_not_null_and_pk(wsp_config):
    statements = build_vector_search_prereqs_sql(wsp_config)

    assert any("SET NOT NULL" in s for s in statements)
    assert any("PRIMARY KEY (chunk_id)" in s for s in statements)


@pytest.mark.integration
def test_chunks_table_matches_config_schema(spark, wsp_config):
    """Read-only check against the table the notebook already populated —
    does not re-run ai_prep_search (that's a real DBU cost).
    """
    df = spark.table(wsp_config.chunks_full_name)
    columns = set(df.columns)

    assert {"chunk_id", "plan_name", "file_name", "chunk_index", "chunk_text"} <= columns
    assert df.count() > 0
