import pandas as pd

from doc_intelligence.monitoring.change_detection import (
    ChangeSet,
    DocumentVersion,
    build_change_summary_sql,
    change_log_frame,
    content_hash,
    detect_changes,
    diff_texts,
    discover_documents,
    format_diff,
    plan_name_for,
    sql_like_to_regex,
    version_from_bytes,
)

LISTING = """
<html><body>
  <a href="/plans/WSP_Barwon-Darling_Unregulated_2026.pdf">Barwon-Darling</a>
  <a href="https://example.test/files/WSP_Intersecting%20Streams_2024.pdf">Intersecting</a>
  <a href="/plans/Other_Plan_2024.pdf">Not a WSP</a>
  <a href="/plans/WSP_Barwon-Darling_Unregulated_2026.pdf">duplicate link</a>
  <a href="/about">About</a>
</body></html>
"""


def test_sql_like_to_regex():
    assert sql_like_to_regex("WSP_%") == "^WSP..*$"
    import re
    assert re.match(sql_like_to_regex("WSP_%"), "WSP_Barwon.pdf")
    assert not re.match(sql_like_to_regex("WSP_%"), "Other.pdf")


def test_discover_documents_filters_and_dedupes():
    docs = discover_documents(LISTING, "https://example.test/list", "[.]pdf$", "WSP_%")
    assert [d.file_name for d in docs] == [
        "WSP_Barwon-Darling_Unregulated_2026.pdf", "WSP_Intersecting Streams_2024.pdf",
    ]
    assert docs[0].url == "https://example.test/plans/WSP_Barwon-Darling_Unregulated_2026.pdf"
    assert len(discover_documents(LISTING, "https://example.test/", "[.]pdf$")) == 3


def test_detect_changes_classifies_new_updated_unchanged():
    old = DocumentVersion("a.pdf", content_hash(b"v1"), 2, None, "t0")
    remote = [
        version_from_bytes("a.pdf", b"v2", "u"),
        version_from_bytes("b.pdf", b"new", "u"),
        version_from_bytes("c.pdf", b"same", "u"),
    ]
    known = {"a.pdf": old, "c.pdf": DocumentVersion("c.pdf", content_hash(b"same"), 4, None, "t0")}
    changes = detect_changes(remote, known)
    assert [v.file_name for v in changes.new] == ["b.pdf"]
    assert [(p.file_name, c.size_bytes) for p, c in changes.updated] == [("a.pdf", 2)]
    assert changes.unchanged == ("c.pdf",)
    assert changes.has_changes
    assert not detect_changes([], {}).has_changes


def test_diff_texts_reports_paragraph_changes():
    previous = "Part 1\nCease to pump below 198 ML/day.\nPart 3"
    current = "Part 1\nCease to pump below 250 ML/day.\nPart 3\nNew clause"
    changes = diff_texts(previous, current)
    assert [c["kind"] for c in changes] == ["replace", "insert"]
    assert "198" in changes[0]["previous"] and "250" in changes[0]["current"]
    assert diff_texts("same", "same") == []
    rendered = format_diff(changes)
    assert rendered.startswith("[REPLACE]") and "New clause" in rendered


def test_change_log_frame_and_summary_sql(wsp_config):
    old = DocumentVersion("a.pdf", "h1", 1, None, "t0")
    new = DocumentVersion("a.pdf", "h2", 1, None, "t1")
    frame = change_log_frame(ChangeSet((), ((old, new),), ()), {"a.pdf": "limits changed"})
    assert frame.loc[0, "change_kind"] == "updated" and frame.loc[0, "previous_hash"] == "h1"
    assert frame.loc[0, "summary"] == "limits changed"
    assert change_log_frame(ChangeSet((), (), ("x",))).empty
    sql = build_change_summary_sql(wsp_config, "WSP_A", "[REPLACE] it's changed")
    assert "Plan: WSP_A" in sql and "it''s changed" in sql
    assert plan_name_for("WSP_A.PDF") == "WSP_A"
    assert isinstance(frame, pd.DataFrame)
