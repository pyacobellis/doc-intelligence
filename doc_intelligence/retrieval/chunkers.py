"""Deterministic chunking strategies over parsed elements (no AI calls).

`section`: elements grouped under their section header, packed up to max_chars, with the
section title prefixed so it sits inside the embedding window.
`fixed`:   sliding character windows with overlap over the document text — the naive
baseline the others should beat.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from doc_intelligence.config import ChunkingConfig
from doc_intelligence.extraction.rules_from_tables import parse_table_html
from doc_intelligence.parsing.elements import LAYOUT_TYPES, ParsedElement

CHUNK_COLUMNS = [
    "chunk_id", "plan_name", "file_name", "chunk_index", "chunk_text",
    "chunk_header", "pages", "section_reference", "strategy",
]

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.;:!?])\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    plan_name: str
    file_name: str
    chunk_index: int
    chunk_text: str
    chunk_header: str | None
    pages: str
    section_reference: str | None
    strategy: str


def chunk_id_for(plan_name: str, strategy: str, index: int) -> str:
    return hashlib.md5(f"{plan_name}_{strategy}_{index}".encode()).hexdigest()


def element_text(element: ParsedElement) -> str:
    """Plain text for an element; HTML tables become one line per row."""
    if element.element_type == "table":
        rows = parse_table_html(element.content)
        return "\n".join(" | ".join(cell for cell in row if cell) for row in rows)
    return " ".join(element.content.split())


def body_elements(elements: Sequence[ParsedElement]) -> list[ParsedElement]:
    """Content-bearing elements only: no page furniture, no section headers (those are
    carried as section_reference), nothing empty."""
    return [
        e for e in elements
        if e.element_type not in LAYOUT_TYPES and e.element_type != "section_header" and element_text(e)
    ]


def split_text(text: str, max_chars: int) -> list[str]:
    """Split oversized text at sentence boundaries, then hard at whitespace if a single
    sentence is still too long."""
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_BOUNDARY.split(text):
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


def _pages(elements: Sequence[ParsedElement]) -> str:
    return ",".join(str(p) for p in sorted({e.page_id for e in elements if e.page_id is not None}))


def _group_by_plan(elements: Sequence[ParsedElement]) -> dict[str, list[ParsedElement]]:
    grouped: dict[str, list[ParsedElement]] = {}
    for element in elements:
        grouped.setdefault(element.plan_name, []).append(element)
    return grouped


def section_chunks(elements: Sequence[ParsedElement], cfg: ChunkingConfig, strategy: str = "section") -> list[Chunk]:
    chunks: list[Chunk] = []
    for plan_name, plan_elements in _group_by_plan(elements).items():
        file_name = plan_elements[0].file_name
        index = 0

        def emit(section: str | None, parts: list[tuple[str, ParsedElement]]) -> None:
            nonlocal index
            if not parts:
                return
            body = "\n\n".join(text for text, _ in parts)
            text = f"{section}\n\n{body}" if section else body
            chunks.append(
                Chunk(
                    chunk_id=chunk_id_for(plan_name, strategy, index),
                    plan_name=plan_name,
                    file_name=file_name,
                    chunk_index=index,
                    chunk_text=text,
                    chunk_header=section,
                    pages=_pages([e for _, e in parts]),
                    section_reference=section,
                    strategy=strategy,
                )
            )
            index += 1

        current_section: str | None = None
        buffer: list[tuple[str, ParsedElement]] = []
        buffer_len = 0
        for element in body_elements(plan_elements):
            if element.section_reference != current_section and buffer:
                emit(current_section, buffer)
                buffer, buffer_len = [], 0
            current_section = element.section_reference
            budget = cfg.max_chars - (len(current_section) + 2 if current_section else 0)
            for piece in split_text(element_text(element), max(budget, 200)):
                if buffer and buffer_len + len(piece) + 2 > budget:
                    emit(current_section, buffer)
                    buffer, buffer_len = [], 0
                buffer.append((piece, element))
                buffer_len += len(piece) + 2
        emit(current_section, buffer)
    return chunks


def fixed_chunks(elements: Sequence[ParsedElement], cfg: ChunkingConfig, strategy: str = "fixed") -> list[Chunk]:
    chunks: list[Chunk] = []
    overlap = min(cfg.overlap_chars, cfg.max_chars // 2)
    for plan_name, plan_elements in _group_by_plan(elements).items():
        file_name = plan_elements[0].file_name
        spans: list[tuple[int, int, ParsedElement]] = []
        text = ""
        for element in body_elements(plan_elements):
            piece = element_text(element)
            if text:
                text += "\n\n"
            spans.append((len(text), len(text) + len(piece), element))
            text += piece
        start, index = 0, 0
        while start < len(text):
            end = min(start + cfg.max_chars, len(text))
            if end < len(text):
                cut = text.rfind(" ", start + cfg.max_chars // 2, end)
                end = cut if cut > start else end
            covered = [e for s, e_, e in spans if s < end and e_ > start]
            chunks.append(
                Chunk(
                    chunk_id=chunk_id_for(plan_name, strategy, index),
                    plan_name=plan_name,
                    file_name=file_name,
                    chunk_index=index,
                    chunk_text=text[start:end].strip(),
                    chunk_header=None,
                    pages=_pages(covered),
                    section_reference=covered[0].section_reference if covered else None,
                    strategy=strategy,
                )
            )
            index += 1
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


def chunk_elements(elements: Sequence[ParsedElement], cfg: ChunkingConfig) -> list[Chunk]:
    if cfg.strategy == "section":
        return section_chunks(elements, cfg)
    if cfg.strategy == "fixed":
        return fixed_chunks(elements, cfg)
    raise ValueError(f"strategy {cfg.strategy!r} is not a Python chunker (ai_prep_search runs in SQL)")


def chunks_frame(chunks: Sequence[Chunk]) -> pd.DataFrame:
    return pd.DataFrame([c.__dict__ for c in chunks], columns=CHUNK_COLUMNS)


def chunks_spark_schema():
    from pyspark.sql import types as T

    return T.StructType(
        [
            T.StructField("chunk_id", T.StringType(), False),
            T.StructField("plan_name", T.StringType()),
            T.StructField("file_name", T.StringType()),
            T.StructField("chunk_index", T.IntegerType()),
            T.StructField("chunk_text", T.StringType()),
            T.StructField("chunk_header", T.StringType()),
            T.StructField("pages", T.StringType()),
            T.StructField("section_reference", T.StringType()),
            T.StructField("strategy", T.StringType()),
        ]
    )
