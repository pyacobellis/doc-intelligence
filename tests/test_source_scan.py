from pathlib import Path

import pytest

from doc_intelligence.monitoring.source_scan import (
    classify_link,
    listings_by_key,
    parse_instrument_ref,
    plan_key_for,
    region_urls,
    scan_listing_page,
    scan_source,
)

FIXTURES = Path("tests/fixtures")
HUB_URL = "https://www.water.dcceew.nsw.gov.au/our-work/plans-and-strategies/water-sharing-plans"
REGION_URL = HUB_URL + "/barwon-darling-and-west-region"


@pytest.fixture(scope="module")
def hub_html():
    return (FIXTURES / "wsp_hub.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def region_html():
    return (FIXTURES / "wsp_region_barwon_darling_west.html").read_text(encoding="utf-8")


def test_parse_instrument_ref_handles_every_link_style_seen_on_the_site():
    assert parse_instrument_ref("https://legislation.nsw.gov.au/view/pdf/asmade/sl-2026-64") == ("sl-2026-64", "asmade")
    assert parse_instrument_ref("https://legislation.nsw.gov.au/file/2024-279%2020251128.pdf") == ("2024-279", "20251128")
    assert parse_instrument_ref("https://legislation.nsw.gov.au/file/2024-280%20%2020251128.pdf") == ("2024-280", "20251128")
    assert parse_instrument_ref("https://legislation.nsw.gov.au/file/2012-22-20230825.pdf") == ("2012-22", "20230825")
    assert parse_instrument_ref("https://legislation.nsw.gov.au/file/2011-573.pdf") == ("2011-573", None)
    assert parse_instrument_ref("https://www.legislation.nsw.gov.au/#/view/regulation/2012/488") == ("2012-488", None)
    assert parse_instrument_ref(
        "https://legislation.nsw.gov.au/information/asi/electricity-and-water/wsp-barwon-darling-unregulated"
    ) == ("wsp-barwon-darling-unregulated", "information")
    assert parse_instrument_ref("https://example.test/other.pdf") == (None, None)


def test_plan_key_is_stable_across_dash_and_prefix_variants():
    assert plan_key_for("Barwon–Darling Unregulated River Water Source 2026") == "barwon_darling_unregulated_river_water_source_2026"
    assert plan_key_for("Water Sharing Plan for the Barwon-Darling Unregulated River Water Source 2026") == (
        "barwon_darling_unregulated_river_water_source_2026"
    )
    assert plan_key_for("Draft Water Sharing Plan for the Lachlan Unregulated River Water Sources 2025") == (
        "lachlan_unregulated_river_water_sources_2025"
    )
    assert plan_key_for("Gwydir Unregulated River Water Sources 2025 - Draft for public exhibition") == (
        "gwydir_unregulated_river_water_sources_2025"
    )


def test_classify_link_uses_instrument_pattern_then_supporting_keywords(wsp_config):
    src = wsp_config.source
    inst = classify_link("Read: Something 2026", "https://legislation.nsw.gov.au/view/pdf/asmade/sl-2026-64", src)
    assert inst.kind == "instrument" and inst.instrument_id == "sl-2026-64"
    assert classify_link("Rule summary sheets", "https://x/a.pdf", src).kind == "supporting:rule_summary"
    assert classify_link("Background document (2026)", "https://x/b.pdf", src).kind == "supporting:background"
    assert classify_link("Some other report", "https://x/c.pdf", src).kind == "supporting:other"
    assert classify_link("About us", "https://x/page", src).kind == "other"


def test_region_urls_come_from_the_hub(hub_html, wsp_config):
    urls = region_urls(hub_html, HUB_URL, wsp_config.source)
    assert REGION_URL in urls
    assert len(urls) >= 5
    assert all(url.endswith("-region") for url in urls)


def test_region_page_yields_one_listing_per_plan_heading(region_html, wsp_config):
    plans = scan_listing_page(region_html, REGION_URL, wsp_config.source)
    assert len(plans) == 10  # 11 headings on the page; 'About the region' has no year
    by_key = listings_by_key(plans)

    bd = by_key["barwon_darling_unregulated_river_water_source_2026"]
    current = bd.current_instrument(wsp_config.source.instrument_anchor_prefix)
    assert current.text.startswith("Read:") and current.instrument_id == "sl-2026-64"
    kinds = {link.kind for link in bd.supporting}
    assert {"supporting:changes_fact_sheet", "supporting:rule_summary", "supporting:background", "supporting:map"} <= kinds
    assert not bd.is_draft

    isu = by_key["intersecting_streams_unregulated_river_water_sources_2024"]
    current = isu.current_instrument(wsp_config.source.instrument_anchor_prefix)
    assert (current.instrument_id, current.version_label) == ("2024-279", "20251128")
    # previous versions stay visible as non-current instruments
    assert any(link.instrument_id == "2011-573" for link in isu.instruments)


def test_scan_source_stitches_hub_and_regions_with_an_injected_fetcher(hub_html, region_html, wsp_config):
    pages = {HUB_URL: hub_html, REGION_URL: region_html}

    def fetch(url):
        if url in pages:
            return pages[url]
        return "<main></main>"  # other regions: empty in this test

    snapshot = scan_source(wsp_config, fetch=fetch)
    assert snapshot.hub_url == HUB_URL and REGION_URL in snapshot.region_urls
    assert len(snapshot.plans) == 10
    assert snapshot.observed_at.endswith("+00:00")


@pytest.mark.integration
def test_live_scan_of_the_department_site(wsp_config):
    """Read-only HTTP against the real source; a few seconds and no downloads."""
    snapshot = scan_source(wsp_config)
    assert len(snapshot.region_urls) >= 5
    assert len(snapshot.plans) >= 40
    keys = set(listings_by_key(snapshot.plans))
    assert "barwon_darling_unregulated_river_water_source_2026" in keys
    print(f"\nlive: {len(snapshot.region_urls)} regions, {len(snapshot.plans)} plan listings")
