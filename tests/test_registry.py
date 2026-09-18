import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from doc_intelligence.monitoring.registry import (
    PROPOSAL_COLUMNS,
    REGISTRY_COLUMNS,
    detect_proposals,
    entry_from_listing,
    load_proposals,
    load_registry,
    load_seen_urls,
    observations_from_snapshot,
    proposals_frame,
    registry_frame,
    seed_registry,
    touch_registry,
)
from doc_intelligence.monitoring.source_scan import PlanListing, SourceLink, SourceSnapshot, scan_listing_page

REGION_URL = "https://www.water.dcceew.nsw.gov.au/our-work/plans-and-strategies/water-sharing-plans/barwon-darling-and-west-region"


@pytest.fixture(scope="module")
def snapshot(wsp_config):
    html = Path("tests/fixtures/wsp_region_barwon_darling_west.html").read_text(encoding="utf-8")
    plans = scan_listing_page(html, REGION_URL, wsp_config.source)
    return SourceSnapshot("2026-09-18T12:00:00+00:00", wsp_config.source.listing_url, (REGION_URL,), tuple(plans))


def _registry(snapshot, cfg):
    return {e.plan_key: e for e in seed_registry(snapshot, cfg)}


def _seen(snapshot):
    return set(observations_from_snapshot(snapshot)["url"])


def test_seed_marks_known_documents_confirmed_and_ingested(snapshot, wsp_config):
    entries = seed_registry(snapshot, wsp_config)
    assert len(entries) == 10
    by_key = {e.plan_key: e for e in entries}
    bd = by_key["barwon_darling_unregulated_river_water_source_2026"]
    assert bd.confirmed and bd.plan_name == "WSP_Barwon-Darling_Unregulated_2026"
    assert bd.instrument_id == "sl-2026-64" and bd.status == "active"
    others = [e for e in entries if not e.confirmed]
    assert others and all(e.plan_name is None for e in others)
    assert list(registry_frame(entries).columns) == REGISTRY_COLUMNS


def test_observations_have_one_row_per_link(snapshot):
    frame = observations_from_snapshot(snapshot)
    assert len(frame) == sum(len(p.links) for p in snapshot.plans)
    assert frame.kind.str.startswith("instrument").sum() > 10
    assert frame.kind.str.startswith("supporting:").sum() > 50


def test_seeded_registry_raises_no_proposals(snapshot, wsp_config):
    assert detect_proposals(snapshot, _registry(snapshot, wsp_config), _seen(snapshot), set(), wsp_config) == []


def test_new_heading_becomes_new_plan_proposal(snapshot, wsp_config):
    registry = _registry(snapshot, wsp_config)
    removed = registry.pop("intersecting_streams_unregulated_river_water_sources_2024")
    proposals = detect_proposals(snapshot, registry, _seen(snapshot), set(), wsp_config)
    assert [p.kind for p in proposals] == ["new_plan"]
    p = proposals[0]
    assert p.plan_key == removed.plan_key and p.recommended_action == "add_to_registry_and_download"
    assert p.evidence["instrument_id"] == "2024-279" and p.evidence["supporting"]
    assert p.dedupe_key == "new_plan:intersecting_streams_unregulated_river_water_sources_2024"


def test_changed_instrument_version_becomes_new_version_proposal(snapshot, wsp_config):
    registry = _registry(snapshot, wsp_config)
    key = "intersecting_streams_unregulated_river_water_sources_2024"
    registry[key] = replace(registry[key], version_label="20240601")
    proposals = detect_proposals(snapshot, registry, _seen(snapshot), set(), wsp_config)
    assert [p.kind for p in proposals] == ["new_version"]
    assert proposals[0].evidence["previous_version_label"] == "20240601"
    assert proposals[0].evidence["previous_plan_name"] == "WSP_Intersecting_Streams_Unregulated_2024"
    assert proposals[0].recommended_action == "download_instrument"


def test_new_supporting_document_and_withdrawal_and_dedupe(snapshot, wsp_config):
    registry = _registry(snapshot, wsp_config)
    seen = _seen(snapshot)
    bd = "barwon_darling_unregulated_river_water_source_2026"
    unseen_url = next(l.url for p in snapshot.plans if p.plan_key == bd for l in p.supporting)
    seen.discard(unseen_url)
    registry["ghost_plan_2019"] = replace(registry[bd], plan_key="ghost_plan_2019", display_name="Ghost 2019")
    proposals = detect_proposals(snapshot, registry, seen, set(), wsp_config)
    kinds = sorted(p.kind for p in proposals)
    assert kinds == ["supporting_doc", "withdrawn"]
    supporting = next(p for p in proposals if p.kind == "supporting_doc")
    assert supporting.evidence["new_supporting"][0]["url"] == unseen_url
    # already-raised proposals are not raised again
    existing = {p.dedupe_key for p in proposals}
    assert detect_proposals(snapshot, registry, seen, existing, wsp_config) == []


def test_draft_turning_final_is_proposed(wsp_config):
    draft = PlanListing("Lachlan Unregulated River Water Sources 2025 - Draft", "lachlan_2025", "u", True, ())
    final = PlanListing("Lachlan Unregulated River Water Sources 2025", "lachlan_2025", "u", False,
                        (SourceLink("Read: Lachlan 2025", "https://legislation.nsw.gov.au/view/pdf/asmade/sl-2025-9",
                                    "instrument", "sl-2025-9", "asmade"),))
    registry = {"lachlan_2025": entry_from_listing(draft, wsp_config, "t0")}
    assert registry["lachlan_2025"].status == "draft"
    snap = SourceSnapshot("t1", "hub", ("u",), (final,))
    proposals = detect_proposals(snap, registry, set(), set(), wsp_config)
    assert {p.kind for p in proposals} == {"new_version", "new_plan"}


def test_proposals_frame_serialises_evidence_and_loaders_tolerate_missing_tables(snapshot, wsp_config):
    registry = _registry(snapshot, wsp_config)
    registry.pop("intersecting_streams_unregulated_river_water_sources_2024")
    proposals = detect_proposals(snapshot, registry, _seen(snapshot), set(), wsp_config)
    frame = proposals_frame(proposals)
    assert list(frame.columns) == PROPOSAL_COLUMNS
    assert json.loads(frame.loc[0, "evidence"])["display_name"].startswith("Intersecting Streams")

    def missing(sql):
        raise RuntimeError("[TABLE_OR_VIEW_NOT_FOUND] nope")

    assert load_registry(missing, wsp_config) == {}
    assert load_seen_urls(missing, wsp_config) == set()
    assert load_proposals(missing, wsp_config).empty

    def other_error(sql):
        raise RuntimeError("PERMISSION_DENIED")

    with pytest.raises(RuntimeError):
        load_registry(other_error, wsp_config)


def test_explicit_spark_schemas_match_column_lists():
    from doc_intelligence.monitoring.registry import (
        OBSERVATION_COLUMNS,
        observations_spark_schema,
        proposals_spark_schema,
        registry_spark_schema,
    )

    assert registry_spark_schema().fieldNames() == REGISTRY_COLUMNS
    assert observations_spark_schema().fieldNames() == OBSERVATION_COLUMNS
    assert proposals_spark_schema().fieldNames() == PROPOSAL_COLUMNS
    types = {f.name: f.dataType.simpleString() for f in proposals_spark_schema().fields}
    assert types["confidence"] == "double" and types["decided_at"] == "string"  # never VOID
    assert {f.name: f.dataType.simpleString() for f in registry_spark_schema().fields}["confirmed"] == "boolean"


def test_load_registry_round_trips_frame_values(snapshot, wsp_config):
    entries = seed_registry(snapshot, wsp_config)
    frame = registry_frame(entries)
    loaded = load_registry(lambda sql: frame, wsp_config)
    assert set(loaded) == {e.plan_key for e in entries}
    assert loaded["barwon_darling_unregulated_river_water_source_2026"].confirmed is True
    touched = touch_registry(loaded, replace(snapshot, observed_at="2026-09-19T00:00:00+00:00"))
    assert all(e.last_seen == "2026-09-19T00:00:00+00:00" for e in touched)
