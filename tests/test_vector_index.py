import pytest

from doc_intelligence.retrieval.vector_index import index_status
from doc_intelligence.runtime import get_workspace_client


@pytest.mark.integration
def test_index_status_reports_ready_index(wsp_config):
    status = index_status(get_workspace_client("wps_doc_intel"), wsp_config)
    assert status["name"] == wsp_config.chunks_index_full_name
    assert status["ready"] is True
    assert status["indexed_rows"] and status["indexed_rows"] > 0
