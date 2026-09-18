"""Known documents (registry), the observation log, and change proposals (the inbox).

The registry is the canonical identity of every document the site lists; the first
`seed` records the current state so later scans only raise *deltas*: new headings,
changed instrument versions, new supporting documents, withdrawn listings. Nothing
here downloads or changes the corpus; proposals wait for a human decision."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Mapping, Sequence

import pandas as pd

from doc_intelligence.config import DocumentTypeConfig
from doc_intelligence.monitoring.source_scan import PlanListing, SourceSnapshot
from doc_intelligence.runtime import SqlRunner, sql_literal

REGISTRY_COLUMNS = [
    "plan_key", "display_name", "page_url", "instrument_id", "version_label", "instrument_url",
    "plan_name", "status", "confirmed", "first_seen", "last_seen",
]
OBSERVATION_COLUMNS = [
    "observed_at", "plan_key", "display_name", "page_url", "kind", "text", "url", "instrument_id", "version_label",
]
PROPOSAL_COLUMNS = [
    "proposal_id", "dedupe_key", "created_at", "kind", "plan_key", "display_name", "confidence", "evidence",
    "recommended_action", "status", "decided_at", "decision_note", "applied_at", "result",
]
PROPOSAL_KINDS = ("new_plan", "new_version", "draft", "supporting_doc", "withdrawn")
PROPOSAL_STATUSES = ("proposed", "approved", "rejected", "applied")


@dataclass
class RegistryEntry:
    plan_key: str
    display_name: str
    page_url: str
    instrument_id: str | None
    version_label: str | None
    instrument_url: str | None
    plan_name: str | None  # file stem in the volume once ingested
    status: str  # active | draft | withdrawn
    confirmed: bool
    first_seen: str
    last_seen: str


@dataclass
class Proposal:
    proposal_id: str
    dedupe_key: str
    created_at: str
    kind: str
    plan_key: str
    display_name: str
    confidence: float
    evidence: dict
    recommended_action: str
    status: str = "proposed"
    decided_at: str | None = None
    decision_note: str | None = None
    applied_at: str | None = None
    result: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def entry_from_listing(listing: PlanListing, cfg: DocumentTypeConfig, now: str | None = None) -> RegistryEntry:
    now = now or _now()
    current = listing.current_instrument(cfg.source.instrument_anchor_prefix)
    return RegistryEntry(
        plan_key=listing.plan_key,
        display_name=listing.display_name,
        page_url=listing.page_url,
        instrument_id=current.instrument_id if current else None,
        version_label=current.version_label if current else None,
        instrument_url=current.url if current else None,
        plan_name=cfg.source.known_documents.get(listing.plan_key),
        status="draft" if listing.is_draft else "active",
        confirmed=listing.plan_key in cfg.source.known_documents,
        first_seen=now,
        last_seen=now,
    )


def seed_registry(snapshot: SourceSnapshot, cfg: DocumentTypeConfig) -> list[RegistryEntry]:
    """Baseline: every listing becomes an entry; known documents are marked confirmed."""
    return [entry_from_listing(listing, cfg, snapshot.observed_at) for listing in snapshot.plans]


def observations_from_snapshot(snapshot: SourceSnapshot) -> pd.DataFrame:
    rows = [
        {
            "observed_at": snapshot.observed_at,
            "plan_key": listing.plan_key,
            "display_name": listing.display_name,
            "page_url": listing.page_url,
            "kind": link.kind,
            "text": link.text,
            "url": link.url,
            "instrument_id": link.instrument_id,
            "version_label": link.version_label,
        }
        for listing in snapshot.plans
        for link in listing.links
    ]
    return pd.DataFrame(rows, columns=OBSERVATION_COLUMNS)


def _proposal(kind: str, listing_key: str, display_name: str, dedupe_key: str, confidence: float,
              evidence: dict, action: str, now: str) -> Proposal:
    return Proposal(
        proposal_id=uuid.uuid4().hex[:12],
        dedupe_key=dedupe_key,
        created_at=now,
        kind=kind,
        plan_key=listing_key,
        display_name=display_name,
        confidence=confidence,
        evidence=evidence,
        recommended_action=action,
    )


def detect_proposals(
    snapshot: SourceSnapshot,
    registry: Mapping[str, RegistryEntry],
    seen_urls: set[str],
    existing_dedupe_keys: set[str],
    cfg: DocumentTypeConfig,
) -> list[Proposal]:
    """Deterministic deltas between what the site lists now and what the registry knows."""
    now = snapshot.observed_at
    prefix = cfg.source.instrument_anchor_prefix
    proposals: list[Proposal] = []

    def add(proposal: Proposal) -> None:
        if proposal.dedupe_key not in existing_dedupe_keys:
            existing_dedupe_keys.add(proposal.dedupe_key)
            proposals.append(proposal)

    for listing in snapshot.plans:
        current = listing.current_instrument(prefix)
        entry = registry.get(listing.plan_key)
        base_evidence = {
            "display_name": listing.display_name,
            "page_url": listing.page_url,
            "instrument_url": current.url if current else None,
            "instrument_id": current.instrument_id if current else None,
            "version_label": current.version_label if current else None,
            "supporting": [{"kind": link.kind, "text": link.text, "url": link.url} for link in listing.supporting],
        }
        if entry is None:
            kind = "draft" if listing.is_draft else "new_plan"
            action = "add_to_registry" if listing.is_draft or not current else "add_to_registry_and_download"
            add(_proposal(kind, listing.plan_key, listing.display_name,
                          f"{kind}:{listing.plan_key}", 0.9, base_evidence, action, now))
            continue
        if current and current.instrument_id and (current.instrument_id, current.version_label) != (
            entry.instrument_id, entry.version_label
        ):
            evidence = {**base_evidence, "previous_instrument_id": entry.instrument_id,
                        "previous_version_label": entry.version_label, "previous_plan_name": entry.plan_name}
            add(_proposal("new_version", listing.plan_key, listing.display_name,
                          f"new_version:{listing.plan_key}:{current.instrument_id}:{current.version_label}",
                          0.95, evidence, "download_instrument", now))
        if entry.status == "draft" and not listing.is_draft:
            add(_proposal("new_plan", listing.plan_key, listing.display_name,
                          f"final:{listing.plan_key}", 0.9, base_evidence, "add_to_registry_and_download", now))
        new_supporting = [link for link in listing.supporting if link.url not in seen_urls]
        if new_supporting:
            evidence = {**base_evidence,
                        "new_supporting": [{"kind": l.kind, "text": l.text, "url": l.url} for l in new_supporting]}
            add(_proposal("supporting_doc", listing.plan_key, listing.display_name,
                          "supporting:" + "|".join(sorted(l.url for l in new_supporting)),
                          0.9, evidence, "fetch_supporting", now))

    listed = {listing.plan_key for listing in snapshot.plans}
    for plan_key, entry in registry.items():
        if entry.status != "withdrawn" and plan_key not in listed:
            add(_proposal("withdrawn", plan_key, entry.display_name, f"withdrawn:{plan_key}", 0.7,
                          {"last_seen": entry.last_seen, "page_url": entry.page_url}, "review", now))
    return proposals


# --- persistence -------------------------------------------------------------------

def _table_missing(error: Exception) -> bool:
    text = str(error)
    return "TABLE_OR_VIEW_NOT_FOUND" in text or "does not exist" in text.lower()


def table_exists(run_sql: SqlRunner, cfg: DocumentTypeConfig, table_name: str) -> bool:
    """Cheap existence check so first runs do not trip (and log) a not-found error."""
    try:
        frame = run_sql(f"SHOW TABLES IN {cfg.catalog}.{cfg.schema} LIKE '{sql_literal(table_name)}'")
    except Exception:
        return True  # fall back to the SELECT-and-catch path
    return not frame.empty


def registry_frame(entries: Sequence[RegistryEntry]) -> pd.DataFrame:
    return pd.DataFrame([asdict(e) for e in entries], columns=REGISTRY_COLUMNS)


def proposals_frame(proposals: Sequence[Proposal]) -> pd.DataFrame:
    rows = []
    for proposal in proposals:
        row = asdict(proposal)
        row["evidence"] = json.dumps(row["evidence"])
        rows.append(row)
    return pd.DataFrame(rows, columns=PROPOSAL_COLUMNS)


def load_registry(run_sql: SqlRunner, cfg: DocumentTypeConfig) -> dict[str, RegistryEntry]:
    if not table_exists(run_sql, cfg, cfg.tables.registry):
        return {}
    try:
        df = run_sql(f"SELECT {', '.join(REGISTRY_COLUMNS)} FROM {cfg.registry_full_name}")
    except Exception as error:
        if _table_missing(error):
            return {}
        raise
    entries = {}
    for r in df.itertuples(index=False):
        values = {c: (None if pd.isna(v) else v) for c, v in zip(REGISTRY_COLUMNS, r)}
        values["confirmed"] = str(values["confirmed"]).lower() in ("true", "1")
        entries[str(values["plan_key"])] = RegistryEntry(**values)
    return entries


def load_seen_urls(run_sql: SqlRunner, cfg: DocumentTypeConfig) -> set[str]:
    if not table_exists(run_sql, cfg, cfg.tables.observations):
        return set()
    try:
        df = run_sql(f"SELECT DISTINCT url FROM {cfg.observations_full_name}")
    except Exception as error:
        if _table_missing(error):
            return set()
        raise
    return set(df["url"].astype(str))


def load_proposals(run_sql: SqlRunner, cfg: DocumentTypeConfig, status: str | None = None) -> pd.DataFrame:
    where = f" WHERE status = '{sql_literal(status)}'" if status else ""
    if not table_exists(run_sql, cfg, cfg.tables.proposals):
        return pd.DataFrame(columns=PROPOSAL_COLUMNS)
    try:
        return run_sql(f"SELECT * FROM {cfg.proposals_full_name}{where} ORDER BY created_at DESC")
    except Exception as error:
        if _table_missing(error):
            return pd.DataFrame(columns=PROPOSAL_COLUMNS)
        raise


def _string_schema(columns: Sequence[str], overrides: Mapping[str, object] | None = None):
    """Explicit schema: pandas columns that are entirely None would otherwise be written as
    VOID, which later UPDATEs cannot fill."""
    from pyspark.sql import types as T

    overrides = overrides or {}
    return T.StructType([T.StructField(name, overrides.get(name, T.StringType()), True) for name in columns])


def registry_spark_schema():
    from pyspark.sql import types as T

    return _string_schema(REGISTRY_COLUMNS, {"confirmed": T.BooleanType()})


def observations_spark_schema():
    return _string_schema(OBSERVATION_COLUMNS)


def proposals_spark_schema():
    from pyspark.sql import types as T

    return _string_schema(PROPOSAL_COLUMNS, {"confidence": T.DoubleType()})


def write_registry(spark, cfg: DocumentTypeConfig, entries: Sequence[RegistryEntry]) -> None:
    """The registry is small; rewrite it whole."""
    df = spark.createDataFrame(registry_frame(entries), schema=registry_spark_schema())
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(cfg.registry_full_name)


def append_observations(spark, cfg: DocumentTypeConfig, frame: pd.DataFrame) -> None:
    if not frame.empty:
        df = spark.createDataFrame(frame, schema=observations_spark_schema())
        df.write.mode("append").option("mergeSchema", "true").saveAsTable(cfg.observations_full_name)


def append_proposals(spark, cfg: DocumentTypeConfig, proposals: Sequence[Proposal]) -> None:
    if proposals:
        df = spark.createDataFrame(proposals_frame(proposals), schema=proposals_spark_schema())
        df.write.mode("append").option("mergeSchema", "true").saveAsTable(cfg.proposals_full_name)


def set_proposal_status(spark, cfg: DocumentTypeConfig, proposal_id: str, status: str,
                        note: str | None = None, result: str | None = None) -> None:
    if status not in PROPOSAL_STATUSES:
        raise ValueError(f"status must be one of {PROPOSAL_STATUSES}")
    column = "applied_at" if status == "applied" else "decided_at"
    sets = [f"status = '{status}'", f"{column} = '{_now()}'"]
    if note is not None:
        sets.append(f"decision_note = '{sql_literal(note)}'")
    if result is not None:
        sets.append(f"result = '{sql_literal(result)}'")
    spark.sql(f"UPDATE {cfg.proposals_full_name} SET {', '.join(sets)} WHERE proposal_id = '{sql_literal(proposal_id)}'")


def touch_registry(entries: Mapping[str, RegistryEntry], snapshot: SourceSnapshot) -> list[RegistryEntry]:
    """Update last_seen for listings still present; entries are returned for rewrite."""
    listed = {listing.plan_key for listing in snapshot.plans}
    updated = []
    for entry in entries.values():
        if entry.plan_key in listed:
            entry.last_seen = snapshot.observed_at
        updated.append(entry)
    return updated
