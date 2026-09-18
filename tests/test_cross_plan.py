import pandas as pd
import pytest

from doc_intelligence.comparison.cross_plan import (
    build_rule_detail_sql,
    build_rule_matrix_sql,
    build_similar_provisions_sql,
    find_similar_provisions,
    numeric_comparison,
    pivot_rule_counts,
    rule_matrix,
)
from doc_intelligence.retrieval.search import load_chunks
from doc_intelligence.runtime import get_workspace_client, spark_sql_runner


def test_rule_matrix_sql_groups_by_plan_and_type(wsp_config):
    sql = build_rule_matrix_sql(wsp_config)
    assert "GROUP BY plan_name, rule_type" in sql
    assert "min(value) AS min_value" in sql and "collect_set(unit)" in sql


def test_pivot_rule_counts_fills_missing_cells_with_zero():
    matrix = pd.DataFrame(
        {"plan_name": ["A", "A", "B"], "rule_type": ["x", "y", "x"], "rule_count": [3, 1, 2]}
    )
    pivot = pivot_rule_counts(matrix)
    assert pivot.loc["x", "A"] == 3 and pivot.loc["x", "B"] == 2
    assert pivot.loc["y", "B"] == 0
    assert pivot_rule_counts(pd.DataFrame()).empty


def test_rule_detail_sql_applies_filters(wsp_config):
    sql = build_rule_detail_sql(wsp_config, "cease_to_pump", plan_names=["P1", "O'Brien"], min_confidence=0.8,
                                extraction_source="text")
    assert "rule_type = 'cease_to_pump'" in sql
    assert "plan_name IN ('P1', 'O''Brien')" in sql
    assert "confidence_score >= 0.8" in sql and "extraction_source = 'text'" in sql
    assert sql.endswith("ORDER BY plan_name, value")


def test_numeric_comparison_puts_plans_side_by_side():
    detail = pd.DataFrame(
        {
            "plan_name": ["A", "B", "A", "B"],
            "condition": ["A class", "A class", "B class", "B class"],
            "unit": ["ML/day"] * 4,
            "value": [198.0, 150.0, 1500.0, None],
        }
    )
    table = numeric_comparison(detail)
    assert list(table.columns) == ["condition", "unit", "A", "B"]
    a_class = table[table.condition == "A class"].iloc[0]
    assert a_class["A"] == "198" and a_class["B"] == "150"
    b_class = table[table.condition == "B class"].iloc[0]
    assert b_class["A"] == "1500" and pd.isna(b_class["B"])
    assert numeric_comparison(pd.DataFrame()).empty


def test_similar_provisions_sql_is_cross_plan_only(wsp_config):
    sql = build_similar_provisions_sql(wsp_config, threshold=0.75, limit=5)
    assert "a.plan_name < b.plan_name" in sql and "> 0.75" in sql and "LIMIT 5" in sql


@pytest.mark.integration
def test_rule_matrix_over_existing_rules(spark, wsp_config):
    pivot = rule_matrix(spark_sql_runner(spark), wsp_config)
    assert "WSP_Barwon-Darling_Unregulated_2026" in pivot.columns
    assert "cease_to_pump" in pivot.index
    assert int(pivot.values.sum()) > 100


@pytest.mark.integration
def test_find_similar_provisions_uses_index_across_plans(spark, wsp_config):
    chunks = load_chunks(spark_sql_runner(spark), wsp_config)[:2]
    similar = find_similar_provisions(get_workspace_client("wps_doc_intel"), wsp_config, chunks, num_results=2)
    assert not similar.empty
    assert (similar.plan_a != similar.plan_b).all()
    assert similar.score.is_monotonic_decreasing
