from __future__ import annotations

import difflib
import hashlib
import io
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Sequence
from urllib.parse import unquote, urljoin, urlparse

import pandas as pd

from doc_intelligence.config import DocumentTypeConfig
from doc_intelligence.runtime import SqlRunner, sql_literal

VERSION_COLUMNS = ["file_name", "content_hash", "size_bytes", "source_url", "observed_at", "full_text"]
CHANGE_LOG_COLUMNS = ["file_name", "previous_hash", "current_hash", "detected_at", "change_kind", "summary"]


@dataclass(frozen=True)
class RemoteDocument:
    url: str
    file_name: str


@dataclass(frozen=True)
class DocumentVersion:
    file_name: str
    content_hash: str
    size_bytes: int
    source_url: str | None
    observed_at: str
    full_text: str | None = None


@dataclass(frozen=True)
class ChangeSet:
    new: tuple[DocumentVersion, ...]
    updated: tuple[tuple[DocumentVersion, DocumentVersion], ...]  # (previous, current)
    unchanged: tuple[str, ...]

    @property
    def has_changes(self) -> bool:
        return bool(self.new or self.updated)


def sql_like_to_regex(pattern: str) -> str:
    """Translate a SQL LIKE pattern (WSP_%) into an anchored regex."""
    escaped = re.escape(pattern).replace("%", ".*").replace("_", ".")
    return f"^{escaped}$"


def discover_documents(
    listing_html: str, base_url: str, link_pattern: str, file_pattern: str | None = None
) -> list[RemoteDocument]:
    """Links on a listing page whose href matches link_pattern (and whose file name
    matches the config's SQL LIKE file_pattern, if given)."""
    from bs4 import BeautifulSoup

    link_re = re.compile(link_pattern, re.IGNORECASE)
    name_re = re.compile(sql_like_to_regex(file_pattern), re.IGNORECASE) if file_pattern else None
    found: dict[str, RemoteDocument] = {}
    for anchor in BeautifulSoup(listing_html, "html.parser").find_all("a", href=True):
        href = anchor["href"].strip()
        if not link_re.search(href):
            continue
        url = urljoin(base_url, href)
        file_name = unquote(urlparse(url).path.rsplit("/", 1)[-1])
        if name_re and not name_re.match(file_name):
            continue
        found.setdefault(file_name, RemoteDocument(url=url, file_name=file_name))
    return list(found.values())


def fetch_text(url: str, session=None, timeout: int = 30) -> str:
    import requests

    response = (session or requests).get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def fetch_bytes(url: str, session=None, timeout: int = 120) -> bytes:
    import requests

    response = (session or requests).get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def version_from_bytes(file_name: str, data: bytes, source_url: str | None) -> DocumentVersion:
    return DocumentVersion(
        file_name=file_name,
        content_hash=content_hash(data),
        size_bytes=len(data),
        source_url=source_url,
        observed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def detect_changes(
    remote: Sequence[DocumentVersion], known: Mapping[str, DocumentVersion]
) -> ChangeSet:
    new, updated, unchanged = [], [], []
    for version in remote:
        previous = known.get(version.file_name)
        if previous is None:
            new.append(version)
        elif previous.content_hash != version.content_hash:
            updated.append((previous, version))
        else:
            unchanged.append(version.file_name)
    return ChangeSet(tuple(new), tuple(updated), tuple(unchanged))


def build_latest_versions_sql(cfg: DocumentTypeConfig) -> str:
    return f"""
        SELECT file_name, content_hash, size_bytes, source_url, observed_at, full_text
        FROM {cfg.document_versions_full_name}
        QUALIFY row_number() OVER (PARTITION BY file_name ORDER BY observed_at DESC) = 1
    """


def load_known_versions(run_sql: SqlRunner, cfg: DocumentTypeConfig) -> dict[str, DocumentVersion]:
    """Latest recorded version per file; empty on first run (table does not exist yet)."""
    try:
        df = run_sql(build_latest_versions_sql(cfg))
    except Exception as error:  # table missing on first run
        if "TABLE_OR_VIEW_NOT_FOUND" in str(error) or "does not exist" in str(error).lower():
            return {}
        raise
    return {
        str(r.file_name): DocumentVersion(
            file_name=str(r.file_name),
            content_hash=str(r.content_hash),
            size_bytes=int(r.size_bytes),
            source_url=None if pd.isna(r.source_url) else str(r.source_url),
            observed_at=str(r.observed_at),
            full_text=None if pd.isna(r.full_text) else str(r.full_text),
        )
        for r in df.itertuples(index=False)
    }


def build_plan_text_sql(cfg: DocumentTypeConfig, plan_name: str) -> str:
    """Full text of one plan from its chunks, in reading order."""
    return f"""
        SELECT array_join(
                 transform(sort_array(collect_list(struct(chunk_index, chunk_text))), x -> x.chunk_text),
                 '\\n\\n') AS full_text
        FROM {cfg.chunks_full_name}
        WHERE plan_name = '{sql_literal(plan_name)}'
    """


def plan_name_for(file_name: str) -> str:
    return re.sub(r"[.]pdf$", "", file_name, flags=re.IGNORECASE)


def upload_to_volume(w, cfg: DocumentTypeConfig, file_name: str, data: bytes) -> str:
    """Write a document into the source volume (overwrites; the parse step re-reads it)."""
    path = f"{cfg.source.volume}/{file_name}"
    w.files.upload(path, io.BytesIO(data), overwrite=True)
    return path


def diff_texts(previous: str | None, current: str | None) -> list[dict]:
    """Paragraph-level diff: which passages were added, removed or replaced."""
    old = [p.strip() for p in (previous or "").split("\n") if p.strip()]
    new = [p.strip() for p in (current or "").split("\n") if p.strip()]
    changes = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        changes.append({"kind": tag, "previous": "\n".join(old[i1:i2]), "current": "\n".join(new[j1:j2])})
    return changes


def format_diff(changes: Sequence[dict], max_chars: int = 12000) -> str:
    parts = []
    for change in changes:
        parts.append(f"[{change['kind'].upper()}]")
        if change["previous"]:
            parts.append("Previous:\n" + change["previous"])
        if change["current"]:
            parts.append("Current:\n" + change["current"])
        parts.append("")
    return "\n".join(parts)[:max_chars]


def build_change_summary_sql(cfg: DocumentTypeConfig, plan_name: str, diff_text: str) -> str:
    """LLM narrative of what changed between versions. Calls ai_query (costs DBU)."""
    prompt = f"{cfg.qa.change_summary_prompt}\n\nPlan: {plan_name}\n\nChanges:\n{diff_text}"
    return f"SELECT ai_query('{cfg.models.llm}', '{sql_literal(prompt)}') AS summary"


def versions_frame(versions: Sequence[DocumentVersion]) -> pd.DataFrame:
    return pd.DataFrame([v.__dict__ for v in versions], columns=VERSION_COLUMNS)


def change_log_frame(changeset: ChangeSet, summaries: Mapping[str, str] | None = None) -> pd.DataFrame:
    summaries = summaries or {}
    detected_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = [
        {"file_name": v.file_name, "previous_hash": None, "current_hash": v.content_hash,
         "detected_at": detected_at, "change_kind": "new", "summary": summaries.get(v.file_name)}
        for v in changeset.new
    ] + [
        {"file_name": cur.file_name, "previous_hash": prev.content_hash, "current_hash": cur.content_hash,
         "detected_at": detected_at, "change_kind": "updated", "summary": summaries.get(cur.file_name)}
        for prev, cur in changeset.updated
    ]
    return pd.DataFrame(rows, columns=CHANGE_LOG_COLUMNS)


def record_versions(spark, cfg: DocumentTypeConfig, versions: Sequence[DocumentVersion]) -> None:
    if not versions:
        return
    spark.createDataFrame(versions_frame(versions)).write.mode("append").option("mergeSchema", "true").saveAsTable(
        cfg.document_versions_full_name
    )


def record_change_log(spark, cfg: DocumentTypeConfig, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    spark.createDataFrame(frame).write.mode("append").option("mergeSchema", "true").saveAsTable(
        cfg.change_log_full_name
    )


@dataclass(frozen=True)
class SourceCheck:
    changes: ChangeSet
    payloads: Mapping[str, bytes]  # file_name -> downloaded bytes, for applying the changes


def check_source(run_sql: SqlRunner, cfg: DocumentTypeConfig, session=None) -> SourceCheck:
    """Poll the configured listing page and compare each document against the last
    recorded version. Read-only: nothing is uploaded or written here."""
    if not cfg.source.listing_url:
        raise ValueError("source.listing_url is not set in the document type config")
    listing = fetch_text(cfg.source.listing_url, session=session)
    remote_docs = discover_documents(listing, cfg.source.listing_url, cfg.source.link_pattern, cfg.source.file_pattern)
    payloads = {doc.file_name: fetch_bytes(doc.url, session=session) for doc in remote_docs}
    remote_versions = [version_from_bytes(doc.file_name, payloads[doc.file_name], doc.url) for doc in remote_docs]
    return SourceCheck(detect_changes(remote_versions, load_known_versions(run_sql, cfg)), payloads)


def apply_changes(w, spark, cfg: DocumentTypeConfig, check: SourceCheck) -> list[str]:
    """Upload new/updated documents into the source volume and record their versions.
    Writes to the workspace; the pipeline must be re-run afterwards to re-parse."""
    versions = [*check.changes.new, *(current for _, current in check.changes.updated)]
    uploaded = [upload_to_volume(w, cfg, version.file_name, check.payloads[version.file_name]) for version in versions]
    record_versions(spark, cfg, versions)
    return uploaded
