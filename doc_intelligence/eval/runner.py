from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Sequence

import pandas as pd

from doc_intelligence.config import DocumentTypeConfig
from doc_intelligence.eval.metrics import (
    build_judge_sql,
    hit_at_k,
    parse_judge_response,
    plan_recall,
    reciprocal_rank,
    retrieved_plans,
)
from doc_intelligence.eval.questions import EvalQuestion
from doc_intelligence.retrieval.qa import answer_question
from doc_intelligence.retrieval.search import Retriever
from doc_intelligence.runtime import SqlRunner

METRIC_COLUMNS = ["plan_recall", "reciprocal_rank", "hit_at_k", "judge_score"]


@dataclass
class EvalResult:
    question_id: str
    category: str
    retriever: str
    retrieved_plans: list[str]
    plan_recall: float | None
    reciprocal_rank: float | None
    hit_at_k: bool | None
    answer: str | None = None
    judge_score: float | None = None
    judge_reason: str | None = None
    run_at: str = ""


def evaluate_retrieval(
    questions: Sequence[EvalQuestion], retriever: Retriever, retriever_name: str, k: int = 5
) -> list[EvalResult]:
    """Retrieval-only metrics. Vector Search queries only; no LLM calls."""
    run_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results = []
    for question in questions:
        hits = retriever(question.question)
        results.append(
            EvalResult(
                question_id=question.id,
                category=question.category,
                retriever=retriever_name,
                retrieved_plans=retrieved_plans(hits),
                plan_recall=plan_recall(hits, question.expected_plans),
                reciprocal_rank=reciprocal_rank(hits, question.expected_plans),
                hit_at_k=hit_at_k(hits, question.expected_plans, k),
                run_at=run_at,
            )
        )
    return results


def evaluate_answers(
    run_sql: SqlRunner,
    w,
    cfg: DocumentTypeConfig,
    questions: Sequence[EvalQuestion],
    retriever: Retriever,
    retriever_name: str,
    k: int = 5,
) -> list[EvalResult]:
    """Retrieval metrics plus generated answers, LLM-judged where a reference answer exists.
    Calls ai_query twice per gradable question (costs DBU)."""
    results = evaluate_retrieval(questions, retriever, retriever_name, k)
    by_id = {question.id: question for question in questions}
    for result in results:
        question = by_id[result.question_id]
        answer = answer_question(run_sql, w, cfg, question.question, retriever)
        result.answer = answer.answer
        if question.gradable:
            verdict = run_sql(build_judge_sql(cfg, question.question, question.expected_answer, answer.answer))
            result.judge_score, result.judge_reason = parse_judge_response(str(verdict.iloc[0, 0]))
    return results


def results_frame(results: Sequence[EvalResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(result) for result in results])


def summarise_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per retriever x category, plus question counts."""
    if frame.empty:
        return pd.DataFrame()
    metrics = [column for column in METRIC_COLUMNS if column in frame.columns]
    numeric = frame.copy()
    for column in metrics:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    summary = numeric.groupby(["retriever", "category"])[metrics].mean().round(3)
    summary["questions"] = frame.groupby(["retriever", "category"]).size()
    return summary.reset_index()


def write_results(spark, cfg: DocumentTypeConfig, frame: pd.DataFrame) -> None:
    """Append a run to the eval results table."""
    spark.createDataFrame(frame).write.mode("append").option("mergeSchema", "true").saveAsTable(
        cfg.eval_results_full_name
    )
