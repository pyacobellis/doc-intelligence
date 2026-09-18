import pytest

from doc_intelligence.monitoring.cost import (
    build_ai_dbu_usage_sql,
    build_gateway_token_usage_sql,
    build_usage_by_sku_sql,
    cost_report,
)
from doc_intelligence.runtime import spark_sql_runner


def test_cost_sql_windows_by_days():
    assert "current_date() - 30" in build_ai_dbu_usage_sql(30)
    assert "LIKE '%REAL_TIME_INFERENCE%'" in build_ai_dbu_usage_sql()
    assert "system.billing.usage" in build_usage_by_sku_sql(3)
    assert "system.ai_gateway.usage" in build_gateway_token_usage_sql(3)


@pytest.mark.integration
def test_cost_report_reads_system_tables(spark):
    report = cost_report(spark_sql_runner(spark), days=14)
    assert set(report) == {"ai_functions", "all_skus", "gateway_tokens"}
    assert not report["all_skus"].empty
    assert {"usage_date", "sku_name", "total_dbus"} <= set(report["all_skus"].columns)
    print("\n" + report["ai_functions"].head(10).to_string())
