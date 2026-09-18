import io
import json
import time
from pathlib import Path

import pytest

from doc_intelligence.monitoring import apply as apply_mod
from doc_intelligence.monitoring.apply import (
    apply_instrument,
    apply_supporting,
    canonical_file_name,
    wait_for_download,
)
from doc_intelligence.monitoring.registry import Proposal, RegistryEntry
from doc_intelligence.monitoring.triage import (
    build_triage_prompt,
    enrich_with_candidates,
    llm_triage,
    parse_triage_response,
    supersedes_candidates,
)


def _proposal(kind="new_version", **evidence):
    base = {"instrument_url": "https://legislation.nsw.gov.au/file/2024-279%2020260101.pdf",
            "instrument_id": "2024-279", "version_label": "20260101", "page_url": "https://x/region"}
    base.update(evidence)
    return Proposal("p1", "k", "2026-09-19T00:00:00+00:00", kind, "intersecting_streams_unregulated_river_water_sources_2024",
                    "Intersecting Streams Unregulated River Water Sources 2024", 0.95, base, "download_instrument")


class FakeFiles:
    def __init__(self):
        self.uploads = {}
        self.deleted = []

    def upload(self, path, data, overwrite=False):
        self.uploads[path] = data.read()

    def download(self, path):
        if path not in self.uploads:
            raise FileNotFoundError(path)
        return type("R", (), {"contents": io.BytesIO(self.uploads[path])})()

    def delete(self, path):
        self.deleted.append(path)
        self.uploads.pop(path, None)


class FakeWorkspace:
    def __init__(self):
        self.files = FakeFiles()


class FakeWriter:
    def __init__(self, sink):
        self.sink = sink

    def mode(self, *_):
        return self

    def option(self, *_):
        return self

    def saveAsTable(self, name):
        self.sink.append(name)


class FakeSpark:
    def __init__(self):
        self.saved = []

    def createDataFrame(self, frame, schema=None):
        return type("DF", (), {"write": FakeWriter(self.saved)})()


def test_canonical_file_name_uses_pattern_prefix(wsp_config):
    assert canonical_file_name(wsp_config, "lachlan_unregulated_2025") == "WSP_lachlan_unregulated_2025.pdf"


def test_wait_for_download_ignores_old_and_in_progress_files(tmp_path):
    old = tmp_path / "old.pdf"
    old.write_bytes(b"%PDF-old")
    started = time.time() + 1  # anything before this is ignored
    assert wait_for_download(tmp_path, started, timeout_seconds=1, poll=0.2) is None
    new = tmp_path / "new.pdf"
    new.write_bytes(b"%PDF-new")
    import os
    os.utime(new, (started + 5, started + 5))
    assert wait_for_download(tmp_path, started, timeout_seconds=3, poll=0.2) == new


def test_apply_instrument_uploads_archives_and_updates_registry(tmp_path, wsp_config):
    w, spark = FakeWorkspace(), FakeSpark()
    previous = "WSP_Intersecting_Streams_Unregulated_2024"
    w.files.uploads[f"{wsp_config.source.volume}/{previous}.pdf"] = b"%PDF-previous"
    entry = RegistryEntry("intersecting_streams_unregulated_river_water_sources_2024", "Intersecting Streams ...",
                          "https://x/region", "2024-279", "20251128", "https://old", previous, "active", True, "t0", "t0")
    opened = []

    def opener(url):
        opened.append(url)
        target = tmp_path / "2024-279 20260101.pdf"
        target.write_bytes(b"%PDF-new-version")
        return True

    result, updated = apply_instrument(w, spark, wsp_config, _proposal(), entry, downloads_dir=tmp_path,
                                       timeout_seconds=5, opener=opener)
    assert opened == [_proposal().evidence["instrument_url"]]
    new_path = f"{wsp_config.source.volume}/WSP_intersecting_streams_unregulated_river_water_sources_2024.pdf"
    assert w.files.uploads[new_path] == b"%PDF-new-version"
    assert result.uploaded == (new_path,)
    assert result.archived == (f"{wsp_config.source.volume}/archive/{previous}.pdf",)
    assert f"{wsp_config.source.volume}/{previous}.pdf" in w.files.deleted
    assert spark.saved == [wsp_config.document_versions_full_name]
    assert updated.plan_name == "WSP_intersecting_streams_unregulated_river_water_sources_2024"
    assert updated.version_label == "20260101" and updated.status == "active" and updated.confirmed
    assert "run `doc-intel pipeline`" in result.note


def test_apply_instrument_times_out_without_a_download(tmp_path, wsp_config):
    with pytest.raises(TimeoutError):
        apply_instrument(FakeWorkspace(), FakeSpark(), wsp_config, _proposal(), None, downloads_dir=tmp_path,
                         timeout_seconds=1, opener=lambda url: True)


def test_apply_supporting_fetches_allowed_links_and_skips_failures(monkeypatch, wsp_config):
    fetched = []

    def fake_fetch(url, session=None, timeout=120):
        fetched.append(url)
        if "gated" in url:
            raise RuntimeError("403")
        return b"%PDF-supporting"

    monkeypatch.setattr(apply_mod, "fetch_bytes", fake_fetch)
    w = FakeWorkspace()
    proposal = _proposal("supporting_doc", new_supporting=[
        {"kind": "supporting:rule_summary", "text": "Rule summary sheets",
         "url": "https://www.water.dcceew.nsw.gov.au/sites/default/files/2025-08/rules.pdf"},
        {"kind": "supporting:other", "text": "gated", "url": "https://publications.water.nsw.gov.au/gated.pdf"},
        {"kind": "supporting:other", "text": "elsewhere", "url": "https://evil.example/x.pdf"},
    ])
    result = apply_supporting(w, cfg=wsp_config, proposal=proposal)
    assert len(fetched) == 2  # the disallowed domain is never fetched
    assert list(w.files.uploads) == [f"{wsp_config.source.volume}/supporting/{proposal.plan_key}/rules.pdf"]
    assert any(u.startswith("SKIPPED") for u in result.uploaded)
    assert "fetched 1 supporting" in result.note


def test_supersedes_candidates_and_prompt(wsp_config):
    registry = {
        "barwon_2012": RegistryEntry("barwon_2012", "Barwon-Darling Unregulated and Alluvial River Water Sources 2012",
                                     "u", None, None, None, None, "active", True, "t", "t"),
        "gwydir_2020": RegistryEntry("gwydir_2020", "Gwydir Regulated River Water Source 2020",
                                     "u", None, None, None, None, "active", True, "t", "t"),
    }
    candidates = supersedes_candidates("Barwon–Darling Unregulated River Water Source 2026", registry)
    assert candidates and candidates[0][0] == "barwon_2012" and candidates[0][1] > 0.6
    assert all(key != "gwydir_2020" for key, _ in candidates)

    proposal = _proposal("new_plan", supporting=[{"kind": "supporting:map", "text": "m", "url": "u"}])
    proposal.display_name = "Barwon–Darling Unregulated River Water Source 2026"
    enrich_with_candidates([proposal], registry)
    assert proposal.evidence["supersedes_candidates"][0][0] == "barwon_2012"
    prompt = build_triage_prompt(proposal, registry)
    assert "Closest known documents" in prompt and "barwon_2012" in prompt and "supporting_kinds" in prompt


def test_parse_and_llm_triage_adjusts_confidence(wsp_config):
    assert parse_triage_response('{"kind": "new_version", "supersedes_plan_key": "barwon_2012", "confidence": 0.8, "reason": "same source, new year"}') == {
        "kind": "new_version", "supersedes_plan_key": "barwon_2012", "confidence": 0.8, "reason": "same source, new year",
    }
    assert parse_triage_response("nonsense") is None
    assert parse_triage_response('{"kind": "x", "confidence": "high"}')["confidence"] is None

    import pandas as pd

    proposal = _proposal("new_plan")
    proposal.confidence = 0.9
    llm_triage(lambda sql: pd.DataFrame({"verdict": ['{"kind":"new_version","confidence":0.5,"reason":"r"}']}),
               wsp_config, [proposal], {})
    assert proposal.confidence == 0.7 and proposal.evidence["llm_triage"]["kind"] == "new_version"
    untouched = _proposal("supporting_doc")
    llm_triage(lambda sql: (_ for _ in ()).throw(AssertionError("must not be called")), wsp_config, [untouched], {})
