import pytest

from doc_intelligence.extraction.rules_from_tables import (
    build_table_elements_sql,
    extract_table_rules,
    parse_table_html,
    rules_from_table,
)
from doc_intelligence.extraction.schema import rule_columns
from doc_intelligence.extraction.taxonomy import Taxonomy

FLOW_TABLE = """<table>
<tr><td>Flow class</td><td>Commence to pump (ML/day)</td><td>Cease to pump (ML/day)</td></tr>
<tr><td>A class</td><td>1,000</td><td>500</td></tr>
<tr><td>B class</td><td>2,500.5</td><td>1,200</td></tr>
</table>"""

TOC_TABLE = (
    "<table><tr><td>1</td><td>Name of Plan</td><td>5</td></tr>"
    "<tr><td>2</td><td>Commencement</td><td>5</td></tr></table>"
)


@pytest.fixture
def taxonomy(wsp_config):
    return Taxonomy(wsp_config.taxonomy)


def _rules(html, wsp_config, taxonomy, **overrides):
    kwargs = dict(
        plan_name="WSP_Test", element_id=11, page_id=3, section_reference="Part 5 Access rules",
        table_html=html, cfg=wsp_config, taxonomy=taxonomy,
    )
    kwargs.update(overrides)
    return rules_from_table(**kwargs)


def test_parse_table_html_returns_rows_of_cells():
    rows = parse_table_html(FLOW_TABLE)
    assert rows[0] == ["Flow class", "Commence to pump (ML/day)", "Cease to pump (ML/day)"]
    assert rows[1] == ["A class", "1,000", "500"]
    assert parse_table_html(None) == []


def test_rules_from_table_pairs_numbers_with_header_units(wsp_config, taxonomy):
    rules = _rules(FLOW_TABLE, wsp_config, taxonomy)
    assert len(rules) == 4
    first = rules[0]
    assert first["rule_type"] == "flow_class_threshold"
    assert first["value"] == 1000.0
    assert first["unit"] == "ML/day" and first["unit_raw"] == "ML/day"
    assert first["condition"] == "A class"
    assert first["applies_to"] == "Commence to pump (ML/day)"
    assert first["section_reference"] == "Part 5 Access rules"
    assert first["citation_ids"] == [11] and first["page_id"] == 3 and first["source_id"] == "11"
    assert first["extraction_source"] == "table" and first["confidence_score"] is None
    assert rules[1]["rule_type"] == "cease_to_pump" and rules[1]["value"] == 500.0
    assert rules[2]["value"] == 2500.5
    assert set(first) == set(rule_columns(wsp_config))


def test_rules_from_table_skips_tables_without_units(wsp_config, taxonomy):
    assert _rules(TOC_TABLE, wsp_config, taxonomy) == []
    assert _rules("<table><tr><td>only header</td></tr></table>", wsp_config, taxonomy) == []


def test_rules_from_table_uses_inline_units_and_normalises_them(wsp_config, taxonomy):
    html = (
        "<table><tr><th>Item</th><th>Limit</th></tr>"
        "<tr><td>Annual extraction limit</td><td>12.5 gigalitres</td></tr></table>"
    )
    rules = _rules(html, wsp_config, taxonomy, section_reference=None)
    assert len(rules) == 1
    assert rules[0]["value"] == 12.5
    assert rules[0]["unit"] == "GL" and rules[0]["unit_raw"] == "gigalitres"
    assert rules[0]["rule_type"] == "extraction_limit"
    assert rules[0]["section_reference"] is None


ROWSPAN_TABLE = """<table><thead>
<tr><th>Column 1 Management Zone</th><th>Column 2 Flow class</th><th>Column 3 Flow class thresholds (ML/day)</th></tr>
</thead><tbody>
<tr><td>Mungindi Zone</td><td>No Flow Class</td><td>0 ML/day at Mungindi gauge</td></tr>
<tr><td></td><td>A Class</td><td>1. More than 198 ML/day at Mungindi gauge, and</td></tr>
<tr><td></td><td></td><td>2. Less than or equal to 1,500 ML/day at Presbury gauge</td></tr>
</tbody></table>"""


def test_rules_from_table_forward_fills_spanning_cells_and_maps_hinted_columns(wsp_config, taxonomy):
    rules = _rules(ROWSPAN_TABLE, wsp_config, taxonomy, section_reference="TABLE A")
    assert [r["value"] for r in rules] == [0.0, 198.0, 1500.0]
    assert {r["water_source"] for r in rules} == {"Mungindi Zone"}
    assert {r["rule_type"] for r in rules} == {"flow_class_threshold"}
    assert rules[1]["condition"] == "A Class | 1. More than 198 ML/day at Mungindi gauge, and"
    assert rules[2]["condition"] == "A Class | 2. Less than or equal to 1,500 ML/day at Presbury gauge"
    assert all(r["unit"] == "ML/day" for r in rules)


def test_build_table_elements_sql_targets_parsed_docs(wsp_config):
    sql = build_table_elements_sql(wsp_config)
    assert "FROM workspace.default.wsp_parsed_docs" in sql
    assert "element_type = 'table'" in sql
    assert "section_reference" in sql
    assert "ai_extract" not in sql and "ai_query" not in sql


@pytest.mark.integration
def test_table_elements_sql_runs_against_parsed_docs(spark, wsp_config):
    """Read-only SQL over the already-parsed documents; no AI functions involved."""
    df = spark.sql(build_table_elements_sql(wsp_config)).toPandas()
    assert len(df) > 0
    assert df["table_html"].str.startswith("<table>").all()
    assert df["page_id"].notna().any()
    assert df["section_reference"].notna().any()


@pytest.mark.integration
def test_extract_table_rules_end_to_end(spark, wsp_config):
    rules = extract_table_rules(spark, wsp_config)
    assert list(rules.columns) == rule_columns(wsp_config)
    preview = rules[["plan_name", "rule_type", "condition", "value", "unit", "section_reference", "page_id"]]
    print("\n" + preview.head(25).to_string())
