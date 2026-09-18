from __future__ import annotations

import pandas as pd

from doc_intelligence.runtime import SqlRunner


def build_ai_dbu_usage_sql(days: int = 7) -> str:
    """Daily DBU + list-price cost of AI functions (billed as serverless real-time
    inference). Billing data lags 12-24h."""
    return f"""
        SELECT
          u.usage_date,
          u.sku_name,
          u.usage_type,
          round(sum(u.usage_quantity), 4) AS total_dbus,
          round(sum(u.usage_quantity * p.pricing.effective_list.default), 4) AS estimated_cost_usd
        FROM system.billing.usage u
        LEFT JOIN system.billing.list_prices p
          ON u.sku_name = p.sku_name AND u.cloud = p.cloud
         AND u.usage_start_time >= p.price_start_time
         AND (p.price_end_time IS NULL OR u.usage_start_time < p.price_end_time)
        WHERE u.usage_date >= current_date() - {int(days)}
          AND u.sku_name LIKE '%REAL_TIME_INFERENCE%'
        GROUP BY u.usage_date, u.sku_name, u.usage_type
        ORDER BY u.usage_date DESC, u.usage_type
    """


def build_usage_by_sku_sql(days: int = 7) -> str:
    """All consumption by SKU family (compute, inference, storage, ...)."""
    return f"""
        SELECT
          usage_date,
          sku_name,
          round(sum(usage_quantity), 4) AS total_dbus
        FROM system.billing.usage
        WHERE usage_date >= current_date() - {int(days)}
        GROUP BY usage_date, sku_name
        ORDER BY usage_date DESC, total_dbus DESC
    """


def build_gateway_token_usage_sql(days: int = 7) -> str:
    """Per-endpoint request and token counts from the AI Gateway log. Only calls routed
    through a serving endpoint appear here; SQL ai_* functions show up in billing only."""
    return f"""
        SELECT
          date(event_time) AS usage_date,
          endpoint_name,
          requester,
          count(*) AS requests,
          sum(input_tokens) AS input_tokens,
          sum(output_tokens) AS output_tokens,
          round(avg(latency_ms)) AS avg_latency_ms
        FROM system.ai_gateway.usage
        WHERE event_time >= current_date() - {int(days)}
        GROUP BY usage_date, endpoint_name, requester
        ORDER BY usage_date DESC, requests DESC
    """


def cost_report(run_sql: SqlRunner, days: int = 7) -> dict[str, pd.DataFrame]:
    return {
        "ai_functions": run_sql(build_ai_dbu_usage_sql(days)),
        "all_skus": run_sql(build_usage_by_sku_sql(days)),
        "gateway_tokens": run_sql(build_gateway_token_usage_sql(days)),
    }
