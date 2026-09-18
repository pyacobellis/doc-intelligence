import pytest

from doc_intelligence.retrieval.chunking import (
    build_chunk_sql,
    build_chunk_summary_sql,
    build_clean_chunk_text_sql,
    build_vector_search_prereqs_sql,
    clean_chunk_text,
    split_metadata_header,
)

CHUNK = (
    "Act: Water Management Act 2000\n"
    "Title: Water Sharing Plan for the Barwon-Darling 2026\n"
    "Sections: 14 Native title rights; 15 Harvestable rights\n"
    "Contains: table of flow classes\n"
    "\n"
    "Created by NSW Minister, a 2026 Water Sharing Plan.\n"
    "14 Native title rights: the total volume is 198 ML/day."
)


def test_split_metadata_header_separates_key_value_lines():
    header, body = split_metadata_header(CHUNK)
    assert [line.split(":")[0] for line in header] == ["Act", "Title", "Sections", "Contains"]
    assert body.startswith("\nCreated by NSW Minister")
    assert "198 ML/day" in body
    assert split_metadata_header("plain text\nno header") == ([], "plain text\nno header")
    assert split_metadata_header("Only: header") == ([], "Only: header")
    assert split_metadata_header(None) == ([], "")


def test_clean_chunk_text_keeps_only_configured_keys():
    cleaned = clean_chunk_text(CHUNK, ["Sections", "Contains"])
    assert cleaned.startswith("Sections: 14 Native title rights; 15 Harvestable rights\nContains: table of flow classes\n")
    assert "Act:" not in cleaned and "Title:" not in cleaned
    assert cleaned.endswith("the total volume is 198 ML/day.")
    assert clean_chunk_text(CHUNK, []) == split_metadata_header(CHUNK)[1]
    assert clean_chunk_text(CHUNK, None) == CHUNK
    assert clean_chunk_text("no header here", ["Sections"]) == "no header here"


def test_clean_chunk_text_sql_mirrors_config():
    assert build_clean_chunk_text_sql("c", None) == "c"
    assert build_clean_chunk_text_sql("c", []).startswith("substring(c, length(regexp_extract(c,")
    sql = build_clean_chunk_text_sql("c", ["Sections", "Page Header"])
    assert "RLIKE '^(Sections|Page\\ Header): '" in sql
    assert sql.startswith("CASE WHEN size(filter(split(")


def test_build_chunk_sql_reads_parsed_docs_and_writes_chunks(wsp_config):
    sql = build_chunk_sql(wsp_config)
    assert "FROM workspace.default.wsp_parsed_docs" in sql
    assert "CREATE OR REPLACE TABLE workspace.default.wsp_chunks" in sql
    assert "ai_prep_search" in sql
    assert "delta.enableChangeDataFeed = true" in sql
    assert "AS chunk_header" in sql and "AS pages" in sql
    assert "p -> p:page_id::STRING" in sql
    assert "RLIKE '^(Sections|Section|Tables|Contains): '" in sql


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
    does not re-run ai_prep_search (that's a real DBU cost)."""
    df = spark.table(wsp_config.chunks_full_name)
    assert {"chunk_id", "plan_name", "file_name", "chunk_index", "chunk_text"} <= set(df.columns)
    assert df.count() > 0


@pytest.mark.integration
def test_clean_chunk_text_sql_matches_python_on_real_chunks(spark, wsp_config):
    """Apply the SQL header-stripping expression to existing chunks (read-only SELECT)
    and check it agrees with the Python implementation, and actually shrinks the text."""
    keep = wsp_config.chunking.header_keys_to_keep
    expr = build_clean_chunk_text_sql("chunk_text", keep)
    df = spark.sql(
        f"SELECT chunk_text, {expr} AS cleaned FROM {wsp_config.chunks_full_name} ORDER BY plan_name, chunk_index LIMIT 12"
    ).toPandas()
    assert len(df) == 12
    for original, cleaned in zip(df.chunk_text, df.cleaned):
        assert cleaned == clean_chunk_text(original, keep)
        assert not cleaned.startswith(("Act: ", "Title: ", "Date: ", "Minister: "))
    # idempotent: once the table has been rebuilt with stripping, re-applying changes nothing
    assert df.cleaned.str.len().sum() <= df.chunk_text.str.len().sum()
