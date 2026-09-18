from __future__ import annotations

import time
from typing import Callable

import pandas as pd

SqlRunner = Callable[[str], pd.DataFrame]


def get_spark(profile: str | None = None):
    """Databricks Connect session. With a profile this targets serverless compute from a
    local machine; without one it picks up the ambient session inside a Databricks job."""
    from databricks.connect import DatabricksSession

    builder = DatabricksSession.builder
    if profile:
        builder = builder.profile(profile).serverless(True)
    return builder.getOrCreate()


def get_workspace_client(profile: str | None = None):
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient(profile=profile) if profile else WorkspaceClient()


def spark_sql_runner(spark) -> SqlRunner:
    return lambda sql: spark.sql(sql).toPandas()


def warehouse_sql_runner(w, warehouse_id: str) -> SqlRunner:
    """Runs SQL through the Statement Execution API, for contexts with no Spark session
    (Databricks Apps). Values come back as strings."""
    from databricks.sdk.service.sql import StatementState

    def run(sql: str) -> pd.DataFrame:
        resp = w.statement_execution.execute_statement(
            statement=sql, warehouse_id=warehouse_id, wait_timeout="50s"
        )
        while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
            time.sleep(1)
            resp = w.statement_execution.get_statement(resp.statement_id)
        if resp.status.state != StatementState.SUCCEEDED:
            detail = resp.status.error.message if resp.status.error else "unknown error"
            raise RuntimeError(f"SQL failed ({resp.status.state}): {detail}")
        columns = [c.name for c in resp.manifest.schema.columns]
        rows = resp.result.data_array if resp.result and resp.result.data_array else []
        return pd.DataFrame(rows, columns=columns)

    return run


def sql_literal(text: str) -> str:
    """Escape a Python string for use inside a single-quoted Spark SQL literal."""
    return text.replace("\\", "\\\\").replace("'", "''")
