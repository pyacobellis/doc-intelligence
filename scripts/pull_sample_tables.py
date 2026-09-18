"""Pull a small sample of rows from each wsp_* table in workspace.default
via the SQL warehouse, using the `wps_doc_intel` CLI profile for auth.
"""

import pandas as pd
from databricks import sql
from databricks.sdk import WorkspaceClient

PROFILE = "wps_doc_intel"
CATALOG = "workspace"
SCHEMA = "default"
TABLES = ["wsp_chunks", "wsp_parsed_docs", "wsp_rules"]  # wsp_chunks_index is a Vector Search index, not a plain table
SAMPLE_ROWS = 100

OUT_DIR = "data/samples"


def get_http_path(w: WorkspaceClient) -> str:
    warehouse = next(iter(w.warehouses.list()))
    return warehouse.odbc_params.path


def main() -> None:
    w = WorkspaceClient(profile=PROFILE)
    http_path = get_http_path(w)
    cfg = w.config

    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    with sql.connect(
        server_hostname=cfg.host.replace("https://", ""),
        http_path=http_path,
        credentials_provider=lambda: cfg.authenticate,
    ) as conn:
        for table in TABLES:
            full_name = f"{CATALOG}.{SCHEMA}.{table}"
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT * FROM {full_name} LIMIT {SAMPLE_ROWS}")
                df = cursor.fetchall_arrow().to_pandas()

            print(f"\n=== {full_name} ({len(df)} rows sampled) ===")
            print(df.dtypes)
            print(df.head(5))

            out_path = f"{OUT_DIR}/{table}.parquet"
            df.to_parquet(out_path)
            print(f"Saved sample to {out_path}")


if __name__ == "__main__":
    main()
