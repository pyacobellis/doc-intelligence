"""Smoke test: confirm the databricks-sdk can authenticate against the workspace
using the CLI profile set up via `databricks auth login --profile wps_doc_intel`.
"""

from databricks.sdk import WorkspaceClient

def main() -> None:
    w = WorkspaceClient(profile="wps_doc_intel")
    me = w.current_user.me()
    print(f"Authenticated as: {me.user_name}")

    print("\nSQL warehouses:")
    warehouses = list(w.warehouses.list())
    if not warehouses:
        print("  (none found)")
    for wh in warehouses:
        print(f"  - {wh.name} ({wh.state})")

    print("\nClusters:")
    clusters = list(w.clusters.list())
    if not clusters:
        print("  (none found)")
    for c in clusters:
        print(f"  - {c.cluster_name} ({c.state})")

if __name__ == "__main__":
    main()
