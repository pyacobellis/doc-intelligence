from __future__ import annotations

from typing import Sequence

import pandas as pd

from doc_intelligence.config import DocumentTypeConfig
from doc_intelligence.retrieval.search import SearchHit, search_chunks
from doc_intelligence.runtime import SqlRunner, sql_literal

SIMILAR_PROVISION_COLUMNS = ["plan_a", "chunk_a", "plan_b", "chunk_b", "score", "preview_a", "preview_b"]


def build_rule_matrix_sql(cfg: DocumentTypeConfig) -> str:
    """Per plan x rule type: how many rules were found, and the numeric range where present."""
    fields = cfg.extraction.field_names
    extras = []
    if "value" in fields:
        extras.append("count(value) AS numeric_rules, min(value) AS min_value, max(value) AS max_value")
    if "unit" in fields:
        extras.append("array_sort(collect_set(unit)) AS units")
    extra_sql = "".join(f",\n          {expr}" for expr in extras)
    return f"""
        SELECT
          plan_name,
          rule_type,
          count(*) AS rule_count{extra_sql},
          round(avg(confidence_score), 3) AS avg_confidence
        FROM {cfg.rules_full_name}
        GROUP BY plan_name, rule_type
        ORDER BY plan_name, rule_type
    """


def pivot_rule_counts(matrix: pd.DataFrame) -> pd.DataFrame:
    """Rule types down, plans across, rule counts in the cells."""
    if matrix.empty:
        return pd.DataFrame()
    pivot = matrix.pivot_table(index="rule_type", columns="plan_name", values="rule_count", aggfunc="sum", fill_value=0)
    pivot.columns.name = None
    return pivot.astype(int)


def rule_matrix(run_sql: SqlRunner, cfg: DocumentTypeConfig) -> pd.DataFrame:
    return pivot_rule_counts(run_sql(build_rule_matrix_sql(cfg)))


def build_rule_detail_sql(
    cfg: DocumentTypeConfig,
    rule_type: str,
    *,
    plan_names: Sequence[str] | None = None,
    min_confidence: float | None = None,
    extraction_source: str | None = None,
) -> str:
    fields = cfg.extraction.field_names
    where = [f"rule_type = '{sql_literal(rule_type)}'"]
    if plan_names:
        where.append("plan_name IN (" + ", ".join(f"'{sql_literal(p)}'" for p in plan_names) + ")")
    if min_confidence is not None:
        where.append(f"(confidence_score IS NULL OR confidence_score >= {float(min_confidence)})")
    if extraction_source:
        where.append(f"extraction_source = '{sql_literal(extraction_source)}'")
    order = "plan_name" + (", value" if "value" in fields else "")
    # SELECT * so the query also works on rules tables built before provenance columns were added
    return f"SELECT * FROM {cfg.rules_full_name} WHERE {' AND '.join(where)} ORDER BY {order}"


def compare_rule_type(run_sql: SqlRunner, cfg: DocumentTypeConfig, rule_type: str, **filters) -> pd.DataFrame:
    """Every rule of one canonical type across plans, ready for side-by-side review."""
    return run_sql(build_rule_detail_sql(cfg, rule_type, **filters))


def numeric_comparison(detail: pd.DataFrame, by: Sequence[str] = ("condition", "unit")) -> pd.DataFrame:
    """Plans side by side: one row per (condition, unit), one column per plan listing its values."""
    if detail.empty or "value" not in detail.columns:
        return pd.DataFrame()
    numeric = detail.dropna(subset=["value"]).copy()
    numeric["value"] = pd.to_numeric(numeric["value"], errors="coerce")
    numeric = numeric.dropna(subset=["value"])
    keys = [key for key in by if key in numeric.columns]
    if not keys or numeric.empty:
        return pd.DataFrame()
    table = numeric.pivot_table(
        index=keys,
        columns="plan_name",
        values="value",
        aggfunc=lambda values: ", ".join(f"{v:g}" for v in sorted(values)),
    )
    table.columns.name = None
    return table.reset_index()


def build_similar_provisions_sql(cfg: DocumentTypeConfig, threshold: float = 0.7, limit: int = 20) -> str:
    """Cross-plan provisions with similar wording via ai_similarity. O(n^2) and calls the
    model per pair; prefer find_similar_provisions (index-backed) beyond a few plans."""
    return f"""
        SELECT
          a.plan_name AS plan_a, b.plan_name AS plan_b,
          a.chunk_index AS chunk_a, b.chunk_index AS chunk_b,
          substring(a.chunk_text, 1, 150) AS preview_a,
          substring(b.chunk_text, 1, 150) AS preview_b,
          ai_similarity(a.chunk_text, b.chunk_text) AS score
        FROM {cfg.chunks_full_name} a
        CROSS JOIN {cfg.chunks_full_name} b
        WHERE a.plan_name < b.plan_name
          AND ai_similarity(a.chunk_text, b.chunk_text) > {float(threshold)}
        ORDER BY score DESC
        LIMIT {int(limit)}
    """


def find_similar_provisions(
    w,
    cfg: DocumentTypeConfig,
    chunks: Sequence[SearchHit],
    *,
    num_results: int = 3,
    min_score: float = 0.0,
    max_query_chars: int = 2000,
) -> pd.DataFrame:
    """For each chunk, its nearest neighbours in *other* plans via the Vector Search index."""
    rows = []
    for chunk in chunks:
        matches = search_chunks(
            w, cfg, chunk.chunk_text[:max_query_chars],
            num_results=num_results, query_type="ANN", exclude_plan_names=[chunk.plan_name],
        )
        for match in matches:
            if match.score >= min_score:
                rows.append(
                    {
                        "plan_a": chunk.plan_name, "chunk_a": chunk.chunk_index,
                        "plan_b": match.plan_name, "chunk_b": match.chunk_index,
                        "score": match.score,
                        "preview_a": chunk.chunk_text[:150], "preview_b": match.chunk_text[:150],
                    }
                )
    df = pd.DataFrame(rows, columns=SIMILAR_PROVISION_COLUMNS)
    return df.sort_values("score", ascending=False, ignore_index=True)
