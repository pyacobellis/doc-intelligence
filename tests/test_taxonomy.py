import pytest

from doc_intelligence.extraction.taxonomy import Taxonomy


@pytest.fixture
def taxonomy(wsp_config):
    return Taxonomy(wsp_config.taxonomy)


def test_classify_prefers_specific_types_over_generic_access(taxonomy):
    assert taxonomy.classify("Cease to pump when flow falls below") == "cease_to_pump"
    assert taxonomy.classify("Flow class A commence to pump") == "flow_class_threshold"
    assert taxonomy.classify("Long-term average annual extraction limit") == "extraction_limit"
    assert taxonomy.classify("Daily access rules") == "access_condition"
    assert taxonomy.classify("Nothing relevant here") == "other"
    assert taxonomy.classify(None) == "other"


def test_normalize_unit_maps_aliases_and_keeps_unknown(taxonomy):
    assert taxonomy.normalize_unit("megalitres per day") == "ML/day"
    assert taxonomy.normalize_unit("ML/day") == "ML/day"
    assert taxonomy.normalize_unit("percent") == "%"
    assert taxonomy.normalize_unit("  Gigalitres ") == "GL"
    assert taxonomy.normalize_unit("null") is None
    assert taxonomy.normalize_unit("") is None
    assert taxonomy.normalize_unit(None) is None
    assert taxonomy.normalize_unit("furlongs") == "furlongs"


def test_find_unit_matches_whole_tokens_only(taxonomy):
    assert taxonomy.find_unit("Flow (ML/day)") == "ML/day"
    assert taxonomy.find_unit("Share (%)") == "%"
    assert taxonomy.find_unit("html content") is None
    assert taxonomy.find_unit("Name of Plan") is None
    assert taxonomy.find_unit(None) is None


def test_unit_case_sql_covers_aliases(taxonomy):
    sql = taxonomy.unit_case_sql("unit")
    assert sql.startswith("CASE lower(trim(unit))")
    assert "WHEN 'megalitres per day' THEN 'ML/day'" in sql
    assert "ELSE nullif(trim(unit), '') END" in sql


def test_rule_type_case_sql_orders_specific_types_first(taxonomy):
    sql = taxonomy.rule_type_case_sql("txt")
    assert sql.index("'cease_to_pump'") < sql.index("'access_condition'")
    assert "lower(txt) LIKE '%cease to pump%'" in sql
    assert sql.rstrip().endswith("ELSE 'other' END")
