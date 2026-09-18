"""Databricks App (Streamlit): thin presentation layer over the doc_intelligence package.
No business logic here; every panel calls a package function and renders the result."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from doc_intelligence.comparison import compare_rule_type, numeric_comparison, rule_matrix  # noqa: E402
from doc_intelligence.config import load_document_type_config  # noqa: E402
from doc_intelligence.monitoring import cost_report  # noqa: E402
from doc_intelligence.retrieval import answer_question, search_chunks, vector_retriever  # noqa: E402
from doc_intelligence.runtime import get_workspace_client, warehouse_sql_runner  # noqa: E402

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_CONFIG = os.environ.get("DOC_INTEL_CONFIG", "configs/wsp.yaml")


def available_configs() -> list[Path]:
    """One entry per document type: configs/*.yaml minus the template and question files."""
    return sorted(
        p for p in CONFIGS_DIR.glob("*.yaml")
        if not p.name.startswith("_") and not p.name.endswith("_eval_questions.yaml")
    )


@st.cache_resource
def connection():
    w = get_workspace_client(os.environ.get("DATABRICKS_CONFIG_PROFILE"))
    warehouse_id = os.environ.get("DATABRICKS_WAREHOUSE_ID") or next(iter(w.warehouses.list())).id
    return w, warehouse_sql_runner(w, warehouse_id)


@st.cache_resource
def config_for(path: str):
    return load_document_type_config(path)


st.set_page_config(page_title="Document intelligence", layout="wide")
choices = {p.stem: str(p) for p in available_configs()}
default_key = Path(DEFAULT_CONFIG).stem if Path(DEFAULT_CONFIG).stem in choices else next(iter(choices))
selected = st.sidebar.selectbox("Document type", list(choices), index=list(choices).index(default_key))
cfg = config_for(choices[selected])
w, run_sql = connection()
if st.session_state.get("active_config") != selected:  # chat history belongs to one document type
    st.session_state.messages = []
    st.session_state.active_config = selected

st.title(f"{cfg.display_name} — document intelligence")
st.sidebar.caption(f"{cfg.chunks_full_name} · {cfg.rules_full_name}")

ask_tab, rules_tab, changes_tab, cost_tab = st.tabs(["Ask", "Rules", "Changes", "Costs"])

with ask_tab:
    query_type = st.radio("Retrieval", ["HYBRID", "ANN"], horizontal=True, help="HYBRID adds keyword matching")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for source in message.get("sources", []):
                with st.expander(f"{source['plan']} · chunk {source['chunk']} · score {source['score']:.3f}"):
                    st.text(source["text"][:1200])
    if question := st.chat_input(f"Ask about the {cfg.display_name}..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and answering..."):
                answer = answer_question(run_sql, w, cfg, question, vector_retriever(w, cfg, query_type=query_type))
            st.markdown(answer.answer)
            sources = [
                {"plan": hit.plan_name, "chunk": hit.chunk_index, "score": hit.score, "text": hit.chunk_text}
                for hit in answer.sources
            ]
            for source in sources:
                with st.expander(f"{source['plan']} · chunk {source['chunk']} · score {source['score']:.3f}"):
                    st.text(source["text"][:1200])
        st.session_state.messages.append({"role": "assistant", "content": answer.answer, "sources": sources})

with rules_tab:
    st.subheader("Rules per plan and type")
    try:
        st.dataframe(rule_matrix(run_sql, cfg), use_container_width=True)
    except Exception as error:
        st.warning(f"Rules table not available yet: {error}")
    rule_type = st.selectbox("Compare a rule type across plans", cfg.taxonomy.rule_type_names)
    min_confidence = st.slider("Minimum confidence (LLM-extracted rules)", 0.0, 1.0, 0.0, 0.05)
    if st.button("Compare"):
        detail = compare_rule_type(run_sql, cfg, rule_type, min_confidence=min_confidence or None)
        side_by_side = numeric_comparison(detail)
        if not side_by_side.empty:
            st.markdown("**Numeric values side by side**")
            st.dataframe(side_by_side, use_container_width=True)
        st.markdown(f"**All {len(detail)} rules**")
        st.dataframe(detail, use_container_width=True)

with changes_tab:
    st.subheader("Detected document changes")
    try:
        log = run_sql(f"SELECT * FROM {cfg.change_log_full_name} ORDER BY detected_at DESC LIMIT 50")
        st.dataframe(log, use_container_width=True) if not log.empty else st.info("No changes recorded yet.")
    except Exception:
        st.info("No change log yet — run `doc-intel check-source --apply` once a listing URL is configured.")

with cost_tab:
    days = st.slider("Window (days)", 1, 30, 7)
    report = cost_report(run_sql, days)
    st.markdown("**AI function consumption (DBU, list price)**")
    st.dataframe(report["ai_functions"], use_container_width=True)
    st.markdown("**All SKUs**")
    st.dataframe(report["all_skus"], use_container_width=True)
    st.markdown("**AI Gateway requests/tokens**")
    st.dataframe(report["gateway_tokens"], use_container_width=True)
