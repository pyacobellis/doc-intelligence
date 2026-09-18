import pytest

from doc_intelligence.parsing.pdf_parser import build_parse_sql, build_parse_status_sql


def test_build_parse_sql_targets_configured_table_and_volume(wsp_config):
    sql = build_parse_sql(wsp_config)

    assert "workspace.default.wsp_parsed_docs" in sql
    assert "READ_FILES('/Volumes/workspace/default/raw/*.pdf'" in sql  # top level only: supporting/ and archive/ are not plans
    assert "ai_parse_document" in sql
    assert "LIKE 'WSP_%'" in sql


def test_build_parse_status_sql_targets_configured_table(wsp_config):
    sql = build_parse_status_sql(wsp_config)

    assert "FROM workspace.default.wsp_parsed_docs" in sql
    assert "parse_status" in sql


@pytest.mark.integration
def test_parsed_docs_table_matches_config_schema(spark, wsp_config):
    """Cheap read-only check against the table the notebook already populated.
    Does NOT re-run ai_parse_document (that's a real DBU cost) — just confirms
    the config points at a real table with the columns build_parse_sql expects.
    """
    df = spark.table(wsp_config.parsed_docs_full_name)
    columns = set(df.columns)

    assert {"file_name", "plan_name", "content", "parsed_content"} <= columns
    assert df.count() > 0
