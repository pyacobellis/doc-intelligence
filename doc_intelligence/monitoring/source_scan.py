"""Deterministic watcher over the publishing site: hub -> region pages -> one heading per
document, with links classified as the instrument itself or supporting material. No
downloads and no LLM calls; this produces the evidence that triage reasons over."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Sequence
from urllib.parse import unquote, urljoin, urlparse

from bs4 import BeautifulSoup

from doc_intelligence.config import DocumentTypeConfig, SourceConfig

_DASHES = str.maketrans({"–": "-", "—": "-", "‑": "-", " ": " ", " ": " "})
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_INSTRUMENT_PATTERNS = (
    (re.compile(r"/asmade/(sl-\d{4}-\d+)"), "asmade"),
    (re.compile(r"/file/(\d{4}-\d+)[\s-]*(\d{8})?"), None),
    (re.compile(r"/view/regulation/(\d{4})/(\d+)"), "regulation"),
    (re.compile(r"/asi/[^/]+/([a-z0-9-]+)/?$"), "information"),
)


@dataclass(frozen=True)
class SourceLink:
    text: str
    url: str
    kind: str  # "instrument" | "supporting:<kind>" | "other"
    instrument_id: str | None = None
    version_label: str | None = None


@dataclass(frozen=True)
class PlanListing:
    display_name: str
    plan_key: str
    page_url: str
    is_draft: bool
    links: tuple[SourceLink, ...]

    @property
    def instruments(self) -> list[SourceLink]:
        return [link for link in self.links if link.kind == "instrument"]

    @property
    def supporting(self) -> list[SourceLink]:
        return [link for link in self.links if link.kind.startswith("supporting:")]

    def current_instrument(self, anchor_prefix: str | None) -> SourceLink | None:
        """The link the site presents as 'the' document: the anchor with the configured
        prefix ("Read: ...") if any, otherwise the first instrument link."""
        instruments = self.instruments
        if anchor_prefix:
            for link in instruments:
                if link.text.lower().startswith(anchor_prefix.lower()):
                    return link
        return instruments[0] if instruments else None


@dataclass(frozen=True)
class SourceSnapshot:
    observed_at: str
    hub_url: str
    region_urls: tuple[str, ...]
    plans: tuple[PlanListing, ...]


def normalise_title(text: str) -> str:
    return " ".join(text.translate(_DASHES).split())


def plan_key_for(title: str) -> str:
    """Stable identity from a heading: lowercase, dashes/spaces -> underscores, year kept."""
    lowered = normalise_title(title).lower()
    lowered = re.sub(r"^(draft\s+)?(water sharing plan for the\s+)?", "", lowered)
    lowered = re.sub(r"\s*-\s*draft.*$", "", lowered)
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", lowered)).strip("_")


def parse_instrument_ref(url: str) -> tuple[str | None, str | None]:
    """(instrument id, version label) from a legislation URL, e.g.
    /view/pdf/asmade/sl-2026-64 -> ("sl-2026-64", "asmade");
    /file/2024-279 20251128.pdf -> ("2024-279", "20251128")."""
    parsed = urlparse(url)
    path = unquote(parsed.path) + ("#" + unquote(parsed.fragment) if parsed.fragment else "")
    for pattern, label in _INSTRUMENT_PATTERNS:
        match = pattern.search(path)
        if not match:
            continue
        if label == "regulation":
            return f"{match.group(1)}-{match.group(2)}", None
        if label is None:  # /file/<id>[ <yyyymmdd>].pdf
            return match.group(1), match.group(2)
        return match.group(1), label
    return None, None


def classify_link(text: str, url: str, cfg: SourceConfig) -> SourceLink:
    text = normalise_title(text)
    if cfg.instrument_link_pattern and re.search(cfg.instrument_link_pattern, url, re.IGNORECASE):
        instrument_id, version = parse_instrument_ref(url)
        return SourceLink(text, url, "instrument", instrument_id, version)
    lowered = text.lower()
    for kind, keywords in cfg.supporting_kinds.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            return SourceLink(text, url, f"supporting:{kind}")
    if re.search(cfg.link_pattern, url, re.IGNORECASE):
        return SourceLink(text, url, "supporting:other")
    return SourceLink(text, url, "other")


def domain_allowed(url: str, cfg: SourceConfig) -> bool:
    if not cfg.allowed_domains:
        return True
    return urlparse(url).netloc.lower() in {d.lower() for d in cfg.allowed_domains}


def region_urls(hub_html: str, hub_url: str, cfg: SourceConfig) -> list[str]:
    if not cfg.region_link_pattern:
        return [hub_url]
    found: list[str] = []
    for anchor in BeautifulSoup(hub_html, "html.parser").find_all("a", href=True):
        url = urljoin(hub_url, anchor["href"]).split("#")[0]
        if re.search(cfg.region_link_pattern, url) and url not in found and domain_allowed(url, cfg):
            found.append(url)
    return found


def scan_listing_page(html: str, page_url: str, cfg: SourceConfig) -> list[PlanListing]:
    """One PlanListing per <h2> that names a document (headings with a year); the
    listing's links are every anchor between that heading and the next one."""
    soup = BeautifulSoup(html, "html.parser")
    root = soup.find("main") or soup
    listings: list[PlanListing] = []
    for heading in root.find_all("h2"):
        title = normalise_title(heading.get_text(" ", strip=True))
        if not _YEAR.search(title):
            continue
        links: dict[str, SourceLink] = {}
        for element in heading.find_all_next(["a", "h2"]):
            if element.name == "h2":
                break
            if not element.get("href"):
                continue
            url = urljoin(page_url, element["href"]).split("#")[0] if "#/" not in element["href"] else urljoin(page_url, element["href"])
            text = element.get_text(" ", strip=True)
            if not text or url in links or not domain_allowed(url, cfg):
                continue
            link = classify_link(text, url, cfg)
            if link.kind != "other":
                links[url] = link
        listings.append(
            PlanListing(
                display_name=title,
                plan_key=plan_key_for(title),
                page_url=page_url,
                is_draft="draft" in title.lower(),
                links=tuple(links.values()),
            )
        )
    return listings


def fetch_html(url: str, session=None, timeout: int = 60) -> str:
    import requests

    headers = {"User-Agent": "Mozilla/5.0 (doc-intelligence watcher)"}
    response = (session or requests).get(url, timeout=timeout, headers=headers)
    response.raise_for_status()
    return response.text


def scan_source(cfg: DocumentTypeConfig, fetch: Callable[[str], str] = fetch_html) -> SourceSnapshot:
    """Read-only: fetch the hub and its region pages, return every document listing."""
    source = cfg.source
    if not source.listing_url:
        raise ValueError("source.listing_url is not set in the document type config")
    hub_html = fetch(source.listing_url)
    regions = region_urls(hub_html, source.listing_url, source)
    plans: list[PlanListing] = []
    for url in regions:
        html = hub_html if url == source.listing_url else fetch(url)
        plans.extend(scan_listing_page(html, url, source))
    return SourceSnapshot(
        observed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        hub_url=source.listing_url,
        region_urls=tuple(regions),
        plans=tuple(plans),
    )


def listings_by_key(plans: Sequence[PlanListing]) -> dict[str, PlanListing]:
    return {plan.plan_key: plan for plan in plans}
