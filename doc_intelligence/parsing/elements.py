from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from doc_intelligence.config import DocumentTypeConfig

# Page furniture from ai_parse_document that carries no content of its own.
LAYOUT_TYPES = ("page_header", "page_footer", "page_number")


@dataclass(frozen=True)
class ParsedElement:
    plan_name: str
    file_name: str
    element_id: int
    element_type: str
    content: str
    page_id: int | None
    section_reference: str | None


def build_elements_sql(
    cfg: DocumentTypeConfig, element_types: Sequence[str] | None = None, ordered: bool = True
) -> str:
    """Every parsed element in document order, tagged with the most recent section header
    (with table-of-contents dot leaders stripped). Pure SQL, no AI calls. Shared by the
    deterministic table extraction and the section/fixed chunkers."""
    if not element_types:
        type_filter = ""
    elif len(element_types) == 1:
        type_filter = f"WHERE element_type = '{element_types[0]}'"
    else:
        type_filter = "WHERE element_type IN (" + ", ".join(f"'{t}'" for t in element_types) + ")"
    order = "ORDER BY plan_name, element_id" if ordered else ""
    return f"""
        WITH elements AS (
          SELECT
            plan_name,
            file_name,
            element:id::BIGINT                        AS element_id,
            element:type::STRING                      AS element_type,
            element:content::STRING                   AS content,
            try_cast(element:bbox[0].page_id AS INT)  AS page_id,
            try_cast(element:confidence AS DOUBLE)    AS parse_confidence
          FROM {cfg.parsed_docs_full_name}
          LATERAL VIEW explode(try_cast(parsed_content:document:elements AS ARRAY<VARIANT>)) AS element
          WHERE is_variant_null(parsed_content:error_status)
        ),
        with_section AS (
          SELECT *,
            last_value(
              CASE WHEN element_type = 'section_header'
                   THEN trim(regexp_replace(content, '\\\\.{{3,}}.*$', '')) END,
              true
            ) OVER (PARTITION BY plan_name ORDER BY element_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS section_reference
          FROM elements
        )
        SELECT plan_name, file_name, element_id, element_type, content, page_id, parse_confidence, section_reference
        FROM with_section
        {type_filter}
        {order}
    """


def elements_from_frame(df: pd.DataFrame) -> list[ParsedElement]:
    return [
        ParsedElement(
            plan_name=str(r.plan_name),
            file_name=str(r.file_name),
            element_id=int(r.element_id),
            element_type=str(r.element_type),
            content="" if pd.isna(r.content) else str(r.content),
            page_id=None if pd.isna(r.page_id) else int(r.page_id),
            section_reference=None if pd.isna(r.section_reference) else str(r.section_reference),
        )
        for r in df.itertuples(index=False)
    ]


def load_elements(spark, cfg: DocumentTypeConfig, element_types: Sequence[str] | None = None) -> list[ParsedElement]:
    return elements_from_frame(spark.sql(build_elements_sql(cfg, element_types)).toPandas())
