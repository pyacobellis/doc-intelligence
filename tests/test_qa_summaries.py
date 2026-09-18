from doc_intelligence.retrieval.qa import Answer, answer_question, build_ai_query_sql, build_context, build_prompt
from doc_intelligence.retrieval.search import SearchHit
from doc_intelligence.retrieval.summaries import build_plan_summaries_sql

HITS = [
    SearchHit("c1", "WSP_A", 4, "Cease to pump below 198 ML/day.", 0.9),
    SearchHit("c2", "WSP_B", 1, "It's a rule about access.", 0.8),
]


def test_context_and_prompt_cite_plan_and_chunk(wsp_config):
    context = build_context(HITS)
    assert "Plan: WSP_A (chunk 4)" in context and "\n---\n" in context
    prompt = build_prompt(wsp_config, "What is the threshold?", HITS)
    assert prompt.startswith(wsp_config.qa.system_prompt)
    assert "Question: What is the threshold?" in prompt


def test_ai_query_sql_escapes_quotes():
    sql = build_ai_query_sql("databricks-llama-4-maverick", "It's here")
    assert sql == "SELECT ai_query('databricks-llama-4-maverick', 'It''s here') AS answer"


def test_answer_question_short_circuits_without_hits(wsp_config):
    calls = []
    answer = answer_question(lambda sql: calls.append(sql), None, wsp_config, "q", retriever=lambda q: [])
    assert isinstance(answer, Answer) and answer.sources == ()
    assert "No relevant passages" in answer.answer
    assert calls == []


def test_answer_question_uses_runner_result(wsp_config):
    import pandas as pd

    seen = {}

    def run_sql(sql):
        seen["sql"] = sql
        return pd.DataFrame({"answer": ["Grounded answer."]})

    answer = answer_question(run_sql, None, wsp_config, "What?", retriever=lambda q: HITS)
    assert answer.answer == "Grounded answer." and answer.sources == tuple(HITS)
    assert "It''s a rule" in seen["sql"] and wsp_config.models.llm in seen["sql"]


def test_plan_summaries_sql_orders_chunks_and_bounds_text(wsp_config):
    sql = build_plan_summaries_sql(wsp_config, max_chars=5000)
    assert "sort_array(collect_list(struct(chunk_index, chunk_text)))" in sql
    assert "substring(full_text, 1, 5000)" in sql
    assert f"ai_query(\n            '{wsp_config.models.llm}'" in sql
