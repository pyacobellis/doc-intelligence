import pytest

from doc_intelligence.config import ChunkingConfig
from doc_intelligence.parsing.elements import ParsedElement, load_elements
from doc_intelligence.retrieval.chunkers import (
    CHUNK_COLUMNS,
    body_elements,
    chunk_elements,
    chunks_frame,
    element_text,
    fixed_chunks,
    section_chunks,
    split_text,
)


def _el(i, etype, content, section, page=1, plan="WSP_A"):
    return ParsedElement(plan, f"{plan}.pdf", i, etype, content, page, section)


ELEMENTS = [
    _el(1, "page_header", "Water Sharing Plan [NSW]", None),
    _el(2, "section_header", "Part 1 Introduction", "Part 1 Introduction"),
    _el(3, "text", "This Plan commences on 1 July 2026.", "Part 1 Introduction"),
    _el(4, "text", "It applies to the Barwon-Darling water source.", "Part 1 Introduction", page=2),
    _el(5, "section_header", "Part 5 Access rules", "Part 5 Access rules"),
    _el(6, "table", "<table><tr><td>Flow class</td><td>ML/day</td></tr><tr><td>A</td><td>198</td></tr></table>",
        "Part 5 Access rules", page=3),
    _el(7, "text", "Cease to pump below the A class threshold. " * 40, "Part 5 Access rules", page=3),
    _el(8, "page_footer", "Page 3", "Part 5 Access rules"),
    _el(9, "text", "Second plan text.", "Part 1", plan="WSP_B"),
]


def test_element_text_flattens_tables_and_whitespace():
    assert element_text(ELEMENTS[5]) == "Flow class | ML/day\nA | 198"
    assert element_text(_el(1, "text", "  spaced   out\n text ", None)) == "spaced out text"


def test_body_elements_drops_furniture_and_headers():
    kept = [e.element_id for e in body_elements(ELEMENTS)]
    assert kept == [3, 4, 6, 7, 9]


def test_split_text_respects_sentence_boundaries_and_hard_limits():
    text = "First sentence. Second sentence is here. Third one."
    assert split_text(text, 100) == [text]
    pieces = split_text(text, 25)
    assert pieces == ["First sentence.", "Second sentence is here.", "Third one."]
    long_word = "x" * 50
    assert all(len(p) <= 20 for p in split_text(long_word, 20))
    assert "".join(split_text(long_word, 20)) == long_word


def test_section_chunks_group_by_section_and_prefix_title():
    chunks = section_chunks(ELEMENTS, ChunkingConfig(strategy="section", max_chars=400))
    by_plan = {}
    for c in chunks:
        by_plan.setdefault(c.plan_name, []).append(c)
    a = by_plan["WSP_A"]
    assert a[0].chunk_text.startswith("Part 1 Introduction\n\nThis Plan commences")
    assert a[0].pages == "1,2" and a[0].section_reference == "Part 1 Introduction"
    assert a[1].section_reference == "Part 5 Access rules" and "Flow class | ML/day" in a[1].chunk_text
    assert all(len(c.chunk_text) <= 400 for c in a)
    assert len(a) >= 4  # the long text element is split into several chunks
    assert [c.chunk_index for c in a] == list(range(len(a)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert by_plan["WSP_B"][0].chunk_text == "Part 1\n\nSecond plan text."
    assert {c.strategy for c in chunks} == {"section"}


def test_fixed_chunks_slide_with_overlap():
    cfg = ChunkingConfig(strategy="fixed", max_chars=300, overlap_chars=50)
    chunks = [c for c in fixed_chunks(ELEMENTS, cfg) if c.plan_name == "WSP_A"]
    assert len(chunks) > 3
    assert all(len(c.chunk_text) <= 300 for c in chunks)
    assert chunks[0].chunk_text.startswith("This Plan commences")
    assert chunks[0].section_reference == "Part 1 Introduction" and chunks[0].chunk_header is None
    # consecutive windows share text
    tail = chunks[0].chunk_text[-20:]
    assert tail in chunks[1].chunk_text
    assert chunks[-1].pages.endswith("3")


def test_chunk_elements_dispatch_and_frame():
    cfg = ChunkingConfig(strategy="section", max_chars=400)
    frame = chunks_frame(chunk_elements(ELEMENTS, cfg))
    assert list(frame.columns) == CHUNK_COLUMNS
    assert frame.chunk_text.str.len().max() <= 400
    with pytest.raises(ValueError):
        chunk_elements(ELEMENTS, ChunkingConfig(strategy="ai_prep_search"))


@pytest.mark.integration
def test_section_chunking_over_real_parsed_docs(spark, wsp_config):
    """Chunk the already-parsed WSPs in Python (read-only; no AI calls, nothing written)."""
    elements = load_elements(spark, wsp_config)
    assert len(elements) > 500
    chunks = section_chunks(elements, ChunkingConfig(strategy="section", max_chars=1600))
    frame = chunks_frame(chunks)
    print(f"\nsection_400: {len(frame)} chunks, avg {frame.chunk_text.str.len().mean():.0f} chars, "
          f"max {frame.chunk_text.str.len().max()}")
    assert frame.chunk_text.str.len().max() <= 1600
    assert frame.plan_name.nunique() == 2
    assert frame.section_reference.notna().mean() > 0.9
    assert (frame.pages != "").mean() > 0.9
