from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from doc_intelligence.config import DocumentTypeConfig
from doc_intelligence.retrieval.search import Retriever, SearchHit, vector_retriever
from doc_intelligence.runtime import SqlRunner, sql_literal


@dataclass(frozen=True)
class Answer:
    question: str
    answer: str
    sources: tuple[SearchHit, ...]
    model: str


def build_context(hits: Sequence[SearchHit]) -> str:
    return "\n---\n".join(f"Plan: {hit.plan_name} (chunk {hit.chunk_index})\n{hit.chunk_text}" for hit in hits)


def build_prompt(cfg: DocumentTypeConfig, question: str, hits: Sequence[SearchHit]) -> str:
    return f"{cfg.qa.system_prompt}\n\nQuestion: {question}\n\nExcerpts:\n{build_context(hits)}"


def build_ai_query_sql(model: str, prompt: str) -> str:
    return f"SELECT ai_query('{model}', '{sql_literal(prompt)}') AS answer"


def answer_question(
    run_sql: SqlRunner,
    w,
    cfg: DocumentTypeConfig,
    question: str,
    retriever: Retriever | None = None,
) -> Answer:
    """Retrieve, then ask the LLM for a grounded, cited answer. Calls ai_query (costs DBU)."""
    hits = (retriever or vector_retriever(w, cfg))(question)
    if not hits:
        return Answer(question, "No relevant passages found in the indexed documents.", (), cfg.models.llm)
    df = run_sql(build_ai_query_sql(cfg.models.llm, build_prompt(cfg, question, hits)))
    return Answer(question, str(df.iloc[0, 0]), tuple(hits), cfg.models.llm)
