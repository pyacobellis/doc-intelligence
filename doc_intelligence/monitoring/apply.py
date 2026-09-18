"""Applying approved proposals. Supporting documents are fetched directly (the department
site allows it). Instruments live on legislation.nsw.gov.au, which blocks scripted
downloads, so `apply_instrument` opens the URL in the user's own browser, waits for the
PDF to land in their Downloads folder, and takes it from there — one click, nothing to
copy or rename."""

from __future__ import annotations

import io
import re
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from doc_intelligence.config import DocumentTypeConfig
from doc_intelligence.monitoring.change_detection import content_hash, fetch_bytes, version_from_bytes, versions_frame
from doc_intelligence.monitoring.registry import Proposal, RegistryEntry
from doc_intelligence.monitoring.source_scan import domain_allowed


@dataclass(frozen=True)
class ApplyResult:
    uploaded: tuple[str, ...]
    archived: tuple[str, ...]
    note: str


def canonical_file_name(cfg: DocumentTypeConfig, plan_key: str) -> str:
    """Volume file name for an instrument: the pattern's literal prefix + plan key."""
    prefix = cfg.source.file_pattern.split("%")[0].split("_")[0] or cfg.document_type.upper()
    return f"{prefix}_{plan_key}.pdf"


def _safe_name(url: str) -> str:
    name = url.rstrip("/").rsplit("/", 1)[-1].split("?")[0] or "document.pdf"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def apply_supporting(w, cfg: DocumentTypeConfig, proposal: Proposal, session=None) -> ApplyResult:
    """Fetch each new supporting document into <volume>/supporting/<plan_key>/."""
    links = proposal.evidence.get("new_supporting") or proposal.evidence.get("supporting") or []
    uploaded = []
    for link in links[: cfg.source.max_downloads]:
        url = link["url"]
        if not domain_allowed(url, cfg.source):
            continue
        try:
            data = fetch_bytes(url, session=session)
        except Exception as error:  # a gated or dead link should not sink the whole proposal
            uploaded.append(f"SKIPPED {url} ({type(error).__name__})")
            continue
        path = f"{cfg.source.volume}/supporting/{proposal.plan_key}/{_safe_name(url)}"
        w.files.upload(path, io.BytesIO(data), overwrite=True)
        uploaded.append(path)
    return ApplyResult(tuple(uploaded), (), f"fetched {sum(not u.startswith('SKIPPED') for u in uploaded)} supporting documents")


def wait_for_download(downloads_dir: Path, started_at: float, timeout_seconds: int, poll: float = 2.0) -> Path | None:
    """Newest .pdf that appeared in downloads_dir after started_at (ignoring in-progress files)."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        candidates = [
            p for p in downloads_dir.glob("*.pdf")
            if p.stat().st_mtime >= started_at and not (p.with_suffix(p.suffix + ".crdownload")).exists()
        ]
        if candidates:
            newest = max(candidates, key=lambda p: p.stat().st_mtime)
            size = newest.stat().st_size
            time.sleep(1)
            if newest.stat().st_size == size and size > 0:  # finished writing
                return newest
        time.sleep(poll)
    return None


def archive_previous(w, cfg: DocumentTypeConfig, plan_name: str | None) -> str | None:
    """Move the superseded instrument out of the active corpus (parse only sees the volume root)."""
    if not plan_name:
        return None
    source = f"{cfg.source.volume}/{plan_name}.pdf"
    target = f"{cfg.source.volume}/archive/{plan_name}.pdf"
    try:
        data = w.files.download(source).contents.read()
    except Exception:
        return None
    w.files.upload(target, io.BytesIO(data), overwrite=True)
    w.files.delete(source)
    return target


def apply_instrument(
    w,
    spark,
    cfg: DocumentTypeConfig,
    proposal: Proposal,
    entry: RegistryEntry | None,
    *,
    downloads_dir: Path | None = None,
    timeout_seconds: int = 300,
    opener: Callable[[str], object] = webbrowser.open,
) -> tuple[ApplyResult, RegistryEntry]:
    """One-click ingest of a new/updated instrument via the user's browser."""
    url = proposal.evidence.get("instrument_url")
    if not url:
        raise ValueError("proposal has no instrument_url to fetch")
    downloads_dir = downloads_dir or (Path.home() / "Downloads")
    started_at = time.time() - 1
    opener(url)
    downloaded = wait_for_download(downloads_dir, started_at, timeout_seconds)
    if downloaded is None:
        raise TimeoutError(f"no new PDF appeared in {downloads_dir} within {timeout_seconds}s")
    data = downloaded.read_bytes()
    file_name = canonical_file_name(cfg, proposal.plan_key)
    plan_name = file_name[:-4]

    archived = archive_previous(w, cfg, entry.plan_name if entry and entry.plan_name != plan_name else None)
    target = f"{cfg.source.volume}/{file_name}"
    w.files.upload(target, io.BytesIO(data), overwrite=True)

    version = version_from_bytes(file_name, data, url)
    spark.createDataFrame(versions_frame([version])).write.mode("append").option("mergeSchema", "true").saveAsTable(
        cfg.document_versions_full_name
    )
    updated = RegistryEntry(
        plan_key=proposal.plan_key,
        display_name=proposal.display_name,
        page_url=proposal.evidence.get("page_url") or (entry.page_url if entry else ""),
        instrument_id=proposal.evidence.get("instrument_id"),
        version_label=proposal.evidence.get("version_label"),
        instrument_url=url,
        plan_name=plan_name,
        status="active",
        confirmed=True,
        first_seen=entry.first_seen if entry else proposal.created_at,
        last_seen=proposal.created_at,
    )
    note = f"uploaded {target} (sha256 {content_hash(data)[:12]}); run `doc-intel pipeline` to ingest"
    return ApplyResult((target,), (archived,) if archived else (), note), updated
