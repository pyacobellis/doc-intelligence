from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from doc_intelligence.config import DocumentTypeConfig
from doc_intelligence.retrieval.bm25 import BM25Index
from doc_intelligence.retrieval.search import Retriever, SearchHit, search_chunks


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked id lists: score(d) = sum over lists of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def bm25_retriever(chunks: Sequence[SearchHit], num_results: int = 5) -> Retriever:
    """Lexical-only retriever over an in-memory corpus (no Databricks calls)."""
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    index = BM25Index({chunk.chunk_id: chunk.chunk_text for chunk in chunks})

    def retrieve(query: str) -> list[SearchHit]:
        return [replace(by_id[doc_id], score=score) for doc_id, score in index.search(query, top_k=num_results)]

    return retrieve


def rrf_hybrid_retriever(
    w,
    cfg: DocumentTypeConfig,
    chunks: Sequence[SearchHit],
    *,
    num_results: int | None = None,
    candidate_multiplier: int = 3,
) -> Retriever:
    """Client-side hybrid: ANN vector hits fused with local BM25 via RRF. Exists so the
    eval harness can compare it against the index's built-in HYBRID mode."""
    n = num_results or cfg.vector_search.num_results
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    lexical = BM25Index({chunk.chunk_id: chunk.chunk_text for chunk in chunks})

    def retrieve(query: str) -> list[SearchHit]:
        vector_hits = search_chunks(w, cfg, query, num_results=n * candidate_multiplier, query_type="ANN")
        by_id.update({hit.chunk_id: hit for hit in vector_hits})
        lexical_hits = lexical.search(query, top_k=n * candidate_multiplier)
        fused = reciprocal_rank_fusion(
            [[hit.chunk_id for hit in vector_hits], [doc_id for doc_id, _ in lexical_hits]],
            k=cfg.vector_search.rrf_k,
        )
        return [replace(by_id[doc_id], score=score) for doc_id, score in fused[:n] if doc_id in by_id]

    return retrieve
