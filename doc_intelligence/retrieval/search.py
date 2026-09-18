from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Sequence

from doc_intelligence.config import DocumentTypeConfig
from doc_intelligence.runtime import SqlRunner

SEARCH_COLUMNS = ("chunk_id", "plan_name", "chunk_index", "chunk_text")


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    plan_name: str
    chunk_index: int
    chunk_text: str
    score: float


Retriever = Callable[[str], list[SearchHit]]


def build_filters_json(
    plan_names: Sequence[str] | None = None, exclude_plan_names: Sequence[str] | None = None
) -> str | None:
    filters: dict = {}
    if plan_names:
        filters["plan_name"] = list(plan_names)
    if exclude_plan_names:
        filters["plan_name NOT"] = list(exclude_plan_names)
    return json.dumps(filters) if filters else None


def hits_from_rows(rows: Sequence[Sequence]) -> list[SearchHit]:
    """Vector Search returns the requested columns in order, with the score appended last."""
    hits = []
    for row in rows:
        chunk_id, plan_name, chunk_index, chunk_text = row[:4]
        hits.append(SearchHit(str(chunk_id), str(plan_name), int(chunk_index), str(chunk_text), float(row[-1])))
    return hits


def search_chunks(
    w,
    cfg: DocumentTypeConfig,
    query: str,
    *,
    num_results: int | None = None,
    query_type: str | None = None,
    plan_names: Sequence[str] | None = None,
    exclude_plan_names: Sequence[str] | None = None,
) -> list[SearchHit]:
    """Query the Vector Search index. query_type ANN = semantic only; HYBRID adds keyword matching."""
    kwargs = dict(
        index_name=cfg.chunks_index_full_name,
        columns=list(SEARCH_COLUMNS),
        query_text=query,
        num_results=num_results or cfg.vector_search.num_results,
        query_type=(query_type or cfg.vector_search.query_type).upper(),
    )
    filters = build_filters_json(plan_names, exclude_plan_names)
    if filters:
        kwargs["filters_json"] = filters
    resp = w.vector_search_indexes.query_index(**kwargs)
    rows = resp.result.data_array if resp.result and resp.result.data_array else []
    return hits_from_rows(rows)


def vector_retriever(
    w,
    cfg: DocumentTypeConfig,
    *,
    query_type: str | None = None,
    num_results: int | None = None,
    plan_names: Sequence[str] | None = None,
) -> Retriever:
    return lambda query: search_chunks(
        w, cfg, query, query_type=query_type, num_results=num_results, plan_names=plan_names
    )


def load_chunks(run_sql: SqlRunner, cfg: DocumentTypeConfig) -> list[SearchHit]:
    """Whole chunk corpus (for local lexical indexes and cross-plan comparison)."""
    df = run_sql(
        f"SELECT chunk_id, plan_name, chunk_index, chunk_text FROM {cfg.chunks_full_name} "
        "ORDER BY plan_name, chunk_index"
    )
    return [
        SearchHit(str(r.chunk_id), str(r.plan_name), int(r.chunk_index), str(r.chunk_text), 0.0)
        for r in df.itertuples(index=False)
    ]
