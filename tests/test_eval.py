import pandas as pd
import pytest

from doc_intelligence.eval.metrics import (
    build_judge_sql,
    hit_at_k,
    parse_judge_response,
    plan_recall,
    reciprocal_rank,
    retrieved_plans,
)
from doc_intelligence.eval.questions import EvalQuestion, load_eval_questions
from doc_intelligence.eval.runner import evaluate_retrieval, results_frame, summarise_results
from doc_intelligence.retrieval.search import SearchHit, vector_retriever
from doc_intelligence.runtime import get_workspace_client

HITS = [
    SearchHit("1", "WSP_B", 0, "b text", 0.9),
    SearchHit("2", "WSP_A", 1, "a text", 0.8),
    SearchHit("3", "WSP_B", 2, "more b", 0.7),
]


def test_seed_question_file_loads(wsp_config):
    questions = load_eval_questions(wsp_config.eval.questions_file)
    assert len(questions) >= 5
    assert {q.category for q in questions} == {"factual", "comparative", "edge_case"}
    assert all(q.status == "draft" and not q.gradable for q in questions)
    assert len({q.id for q in questions}) == len(questions)


def test_load_rejects_unknown_category(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text("questions:\n  - id: x\n    question: q\n    category: weird\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_eval_questions(path)


def test_retrieval_metrics():
    assert retrieved_plans(HITS) == ["WSP_B", "WSP_A"]
    assert plan_recall(HITS, ["WSP_A", "WSP_C"]) == 0.5
    assert reciprocal_rank(HITS, ["WSP_A"]) == 0.5
    assert reciprocal_rank(HITS, ["WSP_C"]) == 0.0
    assert hit_at_k(HITS, ["WSP_A"], 1) is False and hit_at_k(HITS, ["WSP_A"], 2) is True
    assert plan_recall(HITS, []) is None and reciprocal_rank(HITS, []) is None and hit_at_k(HITS, [], 3) is None


def test_judge_sql_and_response_parsing(wsp_config):
    sql = build_judge_sql(wsp_config, "Q?", "ref", "cand's")
    assert "Reference answer: ref" in sql and "cand''s" in sql
    assert parse_judge_response('Sure: {"score": 0.8, "reason": "close"}') == (0.8, "close")
    assert parse_judge_response('{"score": 7}') == (1.0, "")
    assert parse_judge_response("no json here")[0] is None
    assert parse_judge_response(None) == (None, "")


def test_evaluate_retrieval_and_summary_without_llm():
    questions = [
        EvalQuestion("q1", "one", "factual", expected_plans=("WSP_A",)),
        EvalQuestion("q2", "two", "edge_case"),
    ]
    results = evaluate_retrieval(questions, lambda q: HITS, "stub", k=2)
    frame = results_frame(results)
    assert list(frame.question_id) == ["q1", "q2"]
    assert frame.loc[0, "plan_recall"] == 1.0 and frame.loc[0, "reciprocal_rank"] == 0.5
    assert pd.isna(frame.loc[1, "plan_recall"])
    summary = summarise_results(frame)
    assert set(summary.retriever) == {"stub"} and summary.questions.sum() == 2
    assert summarise_results(pd.DataFrame()).empty


@pytest.mark.integration
def test_retrieval_eval_runs_on_seed_questions(wsp_config):
    """Vector Search queries only (no LLM). Prints the per-category summary."""
    questions = load_eval_questions(wsp_config.eval.questions_file)
    retriever = vector_retriever(get_workspace_client("wps_doc_intel"), wsp_config, query_type="HYBRID")
    frame = results_frame(evaluate_retrieval(questions, retriever, "vs_hybrid", k=5))
    assert len(frame) == len(questions)
    scored = frame.dropna(subset=["plan_recall"])
    assert scored.plan_recall.mean() > 0.5
    print("\n" + summarise_results(frame).to_string())
