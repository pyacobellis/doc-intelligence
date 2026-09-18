import json

import pytest

from doc_intelligence.retrieval.search import (
    SearchHit,
    build_filters_json,
    hits_from_rows,
    load_chunks,
    search_chunks,
    vector_retriever,
)
from doc_intelligence.runtime import get_workspace_client, spark_sql_runner


def test_build_filters_json_supports_include_and_exclude():
    assert build_filters_json() is None
    assert json.loads(build_filters_json(plan_names=["a", "b"])) == {"plan_name": ["a", "b"]}
    assert json.loads(build_filters_json(exclude_plan_names=["a"])) == {"plan_name NOT": ["a"]}


def test_hits_from_rows_uses_column_order_and_trailing_score():
    rows = [["id1", "PlanA", "3", "some text", 0.91], ["id2", "PlanB", 7, "other", "0.5"]]
    hits = hits_from_rows(rows)
    assert hits[0] == SearchHit("id1", "PlanA", 3, "some text", 0.91)
    assert hits[1].chunk_index == 7 and hits[1].score == 0.5


@pytest.fixture(scope="session")
def workspace():
    return get_workspace_client("wps_doc_intel")


@pytest.mark.integration
def test_semantic_and_hybrid_queries_hit_the_index(workspace, wsp_config):
    query = "cease to pump when flow falls below the threshold"
    semantic = search_chunks(workspace, wsp_config, query, num_results=3, query_type="ANN")
    hybrid = search_chunks(workspace, wsp_config, query, num_results=3, query_type="HYBRID")
    assert len(semantic) == 3 and len(hybrid) == 3
    assert all(hit.chunk_text for hit in semantic)
    assert {hit.plan_name for hit in semantic} <= {
        "WSP_Barwon-Darling_Unregulated_2026", "WSP_Intersecting_Streams_Unregulated_2024",
    }


@pytest.mark.integration
def test_plan_filter_restricts_results(workspace, wsp_config):
    only = "WSP_Intersecting_Streams_Unregulated_2024"
    hits = vector_retriever(workspace, wsp_config, plan_names=[only], num_results=4)("daily access rules")
    assert hits and {hit.plan_name for hit in hits} == {only}


@pytest.mark.integration
def test_load_chunks_reads_whole_corpus(spark, wsp_config):
    chunks = load_chunks(spark_sql_runner(spark), wsp_config)
    assert len(chunks) > 40
    assert chunks[0].chunk_index == 0 and chunks[0].score == 0.0
