import pytest

from doc_intelligence.config import load_document_type_config


@pytest.fixture(scope="session")
def wsp_config():
    return load_document_type_config("configs/wsp.yaml")


@pytest.fixture(scope="session")
def spark():
    from databricks.connect import DatabricksSession

    return DatabricksSession.builder.profile("wps_doc_intel").serverless(True).getOrCreate()
