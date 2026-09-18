from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Mapping

import pandas as pd

from doc_intelligence.config import DocumentTypeConfig
from doc_intelligence.extraction.schema import rule_columns
from doc_intelligence.extraction.taxonomy import Taxonomy

# A number followed by ". " is a list marker ("1. More than ..."), not a value.
_NUMBER = r"(?P<value>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![.]\s)"


def build_table_elements_sql(cfg: DocumentTypeConfig) -> str:
    """Table elements from the parsed documents, each tagged with the most recent
    section header so extracted rules carry a section reference. Pure SQL, no AI calls."""
    return f"""
        WITH elements AS (
          SELECT
            plan_name,
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
        SELECT plan_name, element_id, page_id, parse_confidence, section_reference, content AS table_html
        FROM with_section
        WHERE element_type = 'table'
        ORDER BY plan_name, element_id
    """


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def parse_table_html(html: str | None) -> list[list[str]]:
    parser = _TableHTMLParser()
    parser.feed(html or "")
    parser.close()
    return [row for row in parser.rows if any(cell for cell in row)]


def _hint_field(header: str, hints: Mapping[str, tuple[str, ...]]) -> str | None:
    lowered = header.lower()
    for field, keywords in hints.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            return field
    return None


def _forward_fill_labels(rows: list[list[str]], header_units: list[str | None]) -> list[list[str]]:
    """Blank cells in label columns inherit the value above them (PDF tables render a
    spanning cell once and leave the rows beneath it empty)."""
    width = max(len(row) for row in rows)
    carry = [""] * width
    filled = []
    for row in rows:
        new_row = []
        for col, cell in enumerate(row):
            is_label_column = col >= len(header_units) or header_units[col] is None
            if not cell.strip() and is_label_column:
                new_row.append(carry[col])
            else:
                new_row.append(cell)
                carry[col] = cell
        filled.append(new_row)
    return filled


def _has_prose(cell: str, taxonomy: Taxonomy) -> bool:
    stripped = re.sub(taxonomy.unit_alternation, " ", cell, flags=re.IGNORECASE) if taxonomy.unit_alternation else cell
    return re.search(r"[A-Za-z]{3,}", stripped) is not None


def rules_from_table(
    *,
    plan_name: str,
    element_id: int | None,
    page_id: int | None,
    section_reference: str | None,
    table_html: str | None,
    cfg: DocumentTypeConfig,
    taxonomy: Taxonomy,
) -> list[dict]:
    """Turn one parsed HTML table into rule rows. A cell yields a rule only when a number
    is paired with a known unit (in the cell or its column header), which keeps tables of
    contents and section numbering out of the rules table. Label columns are mapped onto
    rule fields via the config's table_field_hints; the rest fold into `condition`."""
    rows = parse_table_html(table_html)
    if len(rows) < 2:
        return []
    header = rows[0]
    header_units = [taxonomy.find_unit(cell) for cell in header]
    rows = _forward_fill_labels(rows, header_units)
    pattern = re.compile(_NUMBER + rf"\s*(?P<unit>{taxonomy.unit_alternation})?(?![A-Za-z])", re.IGNORECASE)
    field_names = cfg.extraction.field_names
    hints = cfg.extraction.table_field_hints

    def is_label_column(col: int) -> bool:
        return col >= len(header_units) or header_units[col] is None

    rules: list[dict] = []
    for row in rows[1:]:
        for col, cell in enumerate(row):
            matches = list(pattern.finditer(cell))
            if not matches:
                continue
            column_header = header[col] if col < len(header) else ""
            labels = [(j, text) for j, text in enumerate(row) if j != col and text and is_label_column(j)]
            assigned: dict[str, list[str]] = {}
            for j, text in labels:
                field = _hint_field(header[j] if j < len(header) else "", hints) or "condition"
                assigned.setdefault(field, []).append(text)
            if _has_prose(cell, taxonomy):
                assigned.setdefault("condition", []).append(cell)
            context = " ".join(part for part in (section_reference, column_header, *(t for _, t in labels)) if part)

            for match in matches:
                unit_raw = match.group("unit") or (header_units[col] if col < len(header_units) else None)
                if not unit_raw:
                    continue
                derived = {
                    "value": float(match.group("value").replace(",", "")),
                    "unit": taxonomy.normalize_unit(unit_raw),
                    "applies_to": column_header or None,
                    "section_reference": section_reference,
                }
                derived.update({field: " | ".join(parts) for field, parts in assigned.items()})
                rule = {name: None for name in field_names}
                rule.update({k: v for k, v in derived.items() if k in field_names})
                rule.update(
                    {
                        "plan_name": plan_name,
                        "extraction_source": "table",
                        "rule_type": taxonomy.classify(context),
                        "unit_raw": unit_raw,
                        "page_id": page_id,
                        "citation_ids": [int(element_id)] if element_id is not None else [],
                        "confidence_score": None,
                        "source_id": None if element_id is None else str(element_id),
                    }
                )
                rules.append(rule)
    return rules


def extract_table_rules(spark, cfg: DocumentTypeConfig) -> pd.DataFrame:
    """Deterministic rule extraction from parsed tables (SQL + Python, no AI functions)."""
    taxonomy = Taxonomy(cfg.taxonomy)
    elements = spark.sql(build_table_elements_sql(cfg)).toPandas()
    rules: list[dict] = []
    for element in elements.itertuples(index=False):
        rules.extend(
            rules_from_table(
                plan_name=element.plan_name,
                element_id=None if pd.isna(element.element_id) else int(element.element_id),
                page_id=None if pd.isna(element.page_id) else int(element.page_id),
                section_reference=element.section_reference,
                table_html=element.table_html,
                cfg=cfg,
                taxonomy=taxonomy,
            )
        )
    df = pd.DataFrame(rules, columns=rule_columns(cfg))
    df = df.loc[~df.drop(columns=["citation_ids"]).duplicated()].reset_index(drop=True)
    df["page_id"] = df["page_id"].astype("Int64")
    return df
