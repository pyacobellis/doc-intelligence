from doc_intelligence.retrieval.bm25 import BM25Index, tokenize
from doc_intelligence.retrieval.hybrid import bm25_retriever, reciprocal_rank_fusion
from doc_intelligence.retrieval.search import SearchHit

DOCS = {
    "a": "Cease to pump when flow at Mungindi gauge is less than 198 ML/day.",
    "b": "Trading of access licences is permitted between management zones.",
    "c": "Environmental water provisions and planned environmental releases.",
    "d": "A class flow: more than 198 ML/day at Mungindi gauge and 176 ML/day at Presbury.",
}


def test_tokenize_keeps_units_and_numbers_together():
    assert tokenize("198 ML/day at Mungindi") == ["198", "ml/day", "at", "mungindi"]
    assert tokenize("cease-to-pump 3.2") == ["cease-to-pump", "3.2"]
    assert tokenize(None) == []


def test_bm25_ranks_exact_term_matches_first():
    index = BM25Index(DOCS)
    ranked = index.search("198 ML/day Mungindi")
    assert [doc_id for doc_id, _ in ranked[:2]] == ["d", "a"] or [doc_id for doc_id, _ in ranked[:2]] == ["a", "d"]
    assert index.search("trading licences")[0][0] == "b"
    assert index.search("zzz unknown") == []
    assert len(index) == 4


def test_reciprocal_rank_fusion_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    order = [doc_id for doc_id, _ in fused]
    assert set(order[:2]) == {"a", "b"}
    assert order[-1] in {"c", "d"}
    assert fused[0][1] > fused[-1][1]


def test_bm25_retriever_returns_hits_with_scores():
    chunks = [SearchHit(doc_id, "PlanX", i, text, 0.0) for i, (doc_id, text) in enumerate(DOCS.items())]
    retrieve = bm25_retriever(chunks, num_results=2)
    hits = retrieve("environmental releases")
    assert hits[0].chunk_id == "c" and hits[0].score > 0
    assert len(hits) <= 2
