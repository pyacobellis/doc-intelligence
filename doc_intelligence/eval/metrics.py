from __future__ import annotations

import json
import re
from typing import Sequence

from doc_intelligence.config import DocumentTypeConfig
from doc_intelligence.retrieval.search import SearchHit
from doc_intelligence.runtime import sql_literal

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def retrieved_plans(hits: Sequence[SearchHit]) -> list[str]:
    seen: list[str] = []
    for hit in hits:
        if hit.plan_name not in seen:
            seen.append(hit.plan_name)
    return seen


def plan_recall(hits: Sequence[SearchHit], expected_plans: Sequence[str]) -> float | None:
    """Share of the expected plans that appear anywhere in the hits."""
    if not expected_plans:
        return None
    found = set(retrieved_plans(hits))
    return sum(plan in found for plan in expected_plans) / len(expected_plans)


def reciprocal_rank(hits: Sequence[SearchHit], expected_plans: Sequence[str]) -> float | None:
    if not expected_plans:
        return None
    expected = set(expected_plans)
    for rank, hit in enumerate(hits, start=1):
        if hit.plan_name in expected:
            return 1.0 / rank
    return 0.0


def hit_at_k(hits: Sequence[SearchHit], expected_plans: Sequence[str], k: int) -> bool | None:
    if not expected_plans:
        return None
    expected = set(expected_plans)
    return any(hit.plan_name in expected for hit in hits[:k])


def build_judge_prompt(cfg: DocumentTypeConfig, question: str, reference_answer: str, candidate_answer: str) -> str:
    return (
        f"{cfg.eval.judge_prompt}\n\nQuestion: {question}\n\n"
        f"Reference answer: {reference_answer}\n\nCandidate answer: {candidate_answer}"
    )


def build_judge_sql(cfg: DocumentTypeConfig, question: str, reference_answer: str, candidate_answer: str) -> str:
    prompt = build_judge_prompt(cfg, question, reference_answer, candidate_answer)
    return f"SELECT ai_query('{cfg.models.llm}', '{sql_literal(prompt)}') AS verdict"


def parse_judge_response(text: str | None) -> tuple[float | None, str]:
    """Pull {"score": x, "reason": "..."} out of a judge reply; tolerant of surrounding prose."""
    match = _JSON_RE.search(text or "")
    if not match:
        return None, (text or "").strip()[:500]
    try:
        data = json.loads(match.group(0))
        score = float(data.get("score"))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None, (text or "").strip()[:500]
    return max(0.0, min(1.0, score)), str(data.get("reason", "")).strip()
