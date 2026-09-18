from __future__ import annotations

import pandas as pd

from doc_intelligence.config import DocumentTypeConfig
from doc_intelligence.runtime import SqlRunner, sql_literal


def build_plan_summaries_sql(cfg: DocumentTypeConfig, max_chars: int = 12000) -> str:
    """One LLM summary per document from its chunks in reading order. Calls ai_query."""
    return f"""
        WITH plan_text AS (
          SELECT
            plan_name,
            array_join(
              transform(sort_array(collect_list(struct(chunk_index, chunk_text))), x -> x.chunk_text),
              '\\n\\n'
            ) AS full_text
          FROM {cfg.chunks_full_name}
          GROUP BY plan_name
        )
        SELECT
          plan_name,
          ai_query(
            '{cfg.models.llm}',
            concat('{sql_literal(cfg.qa.summary_prompt)}', '\\n\\n', substring(full_text, 1, {int(max_chars)}))
          ) AS summary
        FROM plan_text
        ORDER BY plan_name
    """


def summarise_plans(run_sql: SqlRunner, cfg: DocumentTypeConfig, max_chars: int = 12000) -> pd.DataFrame:
    return run_sql(build_plan_summaries_sql(cfg, max_chars))
