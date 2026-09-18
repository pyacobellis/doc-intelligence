from __future__ import annotations

import time

from databricks.sdk.errors.platform import NotFound
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    EndpointType,
    PipelineType,
    VectorIndexType,
)

from doc_intelligence.config import DocumentTypeConfig


def ensure_endpoint(w, cfg: DocumentTypeConfig, wait_seconds: int = 900) -> str:
    name = cfg.vector_search.endpoint_name
    try:
        w.vector_search_endpoints.get_endpoint(name)
        return name
    except NotFound:
        w.vector_search_endpoints.create_endpoint(name=name, endpoint_type=EndpointType.STANDARD)
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        state = str(w.vector_search_endpoints.get_endpoint(name).endpoint_status.state)
        if "ONLINE" in state:
            return name
        time.sleep(10)
    raise TimeoutError(f"Vector Search endpoint '{name}' did not come online within {wait_seconds}s")


def ensure_index(w, cfg: DocumentTypeConfig, wait_seconds: int = 1800) -> str:
    """Delta Sync index with managed embeddings over the chunks table (created once)."""
    index_name = cfg.chunks_index_full_name
    try:
        w.vector_search_indexes.get_index(index_name=index_name)
        return index_name
    except NotFound:
        pass
    w.vector_search_indexes.create_index(
        name=index_name,
        endpoint_name=cfg.vector_search.endpoint_name,
        primary_key="chunk_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=cfg.chunks_full_name,
            embedding_source_columns=[
                EmbeddingSourceColumn(name="chunk_text", embedding_model_endpoint_name=cfg.models.embedding)
            ],
            pipeline_type=PipelineType.TRIGGERED,
        ),
    )
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if index_status(w, cfg)["ready"]:
            return index_name
        time.sleep(15)
    raise TimeoutError(f"Vector Search index '{index_name}' was not ready within {wait_seconds}s")


def sync_index(w, cfg: DocumentTypeConfig) -> None:
    """Trigger a sync so new/changed chunks get embedded."""
    w.vector_search_indexes.sync_index(index_name=cfg.chunks_index_full_name)


def index_status(w, cfg: DocumentTypeConfig) -> dict:
    idx = w.vector_search_indexes.get_index(index_name=cfg.chunks_index_full_name)
    status = idx.status
    return {
        "name": idx.name,
        "endpoint": cfg.vector_search.endpoint_name,
        "ready": bool(status.ready) if status else False,
        "indexed_rows": status.indexed_row_count if status else None,
        "message": status.message if status else None,
    }


def build_index(w, cfg: DocumentTypeConfig) -> dict:
    """Idempotent: create endpoint + index if missing, then trigger a sync."""
    ensure_endpoint(w, cfg)
    ensure_index(w, cfg)
    sync_index(w, cfg)
    return index_status(w, cfg)
