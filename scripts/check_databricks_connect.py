"""Smoke test: confirm Databricks Connect can reach serverless compute
using the `wps_doc_intel` CLI profile.
"""

from databricks.connect import DatabricksSession

def main() -> None:
    spark = DatabricksSession.builder.profile("wps_doc_intel").serverless(True).getOrCreate()
    df = spark.sql("SELECT current_user() AS user, count(*) AS chunk_count FROM workspace.default.wsp_chunks")
    df.show(truncate=False)

if __name__ == "__main__":
    main()
