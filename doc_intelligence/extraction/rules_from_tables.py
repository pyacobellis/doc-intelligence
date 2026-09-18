from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Mapping, Sequence

import pandas as pd

from doc_intelligence.config import DocumentTypeConfig
from doc_intelligence.extraction.schema import rule_columns
from doc_intelligence.extraction.taxonomy import Taxonomy
from doc_intelligence.parsing.elements import build_elements_sql

# A number followed by ". " is a list marker ("1. More than ..."), and a number preceded
# by "(" is a reference code such as a gauge id ("(416001)"); neither is a value.
_NUMBER = r"(?<![(\w])(?P<value>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![.]\s)"


def build_table_elements_sql(cfg: DocumentTypeConfig) -> str:
    """Table elements in document order with their section reference. Pure SQL, no AI calls."""
    inner = build_elements_sql(cfg, ("table",), ordered=False)
    return f"""
        SELECT plan_name, element_id, page_id, parse_confidence, section_reference, content AS table_html
        FROM ({inner})
        ORDER BY plan_name, element_id
    """


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.header_rows: list[int] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._in_thead = False
        self._row_has_th = False

    def handle_starttag(self, tag, attrs):
        if tag == "thead":
            self._in_thead = True
        elif tag == "tr":
            self._row = []
            self._row_has_th = False
        elif tag in ("td", "th"):
            self._cell = []
            if tag == "th":
                self._row_has_th = True

    def handle_endtag(self, tag):
        if tag == "thead":
            self._in_thead = False
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._in_thead or self._row_has_th:
                self.header_rows.append(len(self.rows))
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def parse_table(html: str | None) -> tuple[list[list[str]], bool]:
    """Rows of cell text, plus whether the first row is an explicit header (<thead>/<th>)."""
    parser = _TableHTMLParser()
    parser.feed(html or "")
    parser.close()
    kept = [(index, row) for index, row in enumerate(parser.rows) if any(cell for cell in row)]
    rows = [row for _, row in kept]
    explicit_header = bool(kept) and kept[0][0] in parser.header_rows
    return rows, explicit_header


def parse_table_html(html: str | None) -> list[list[str]]:
    return parse_table(html)[0]


def _looks_like_header(row: Sequence[str], taxonomy: Taxonomy) -> bool:
    """A header row names things; a data row carries numbers with units."""
    if not taxonomy.unit_alternation:
        return True
    value_with_unit = re.compile(rf"\d\s*(?:{taxonomy.unit_alternation})(?![A-Za-z])", re.IGNORECASE)
    return not any(value_with_unit.search(cell) for cell in row)


def resolve_table_header(
    rows: list[list[str]],
    explicit_header: bool,
    inherited_header: Sequence[str] | None,
    taxonomy: Taxonomy,
) -> tuple[list[str], list[list[str]]]:
    """Header row and data rows. Multi-page tables only carry their header on the first
    page, so a header-less continuation inherits the previous table's header."""
    if not rows:
        return [], []
    if explicit_header:
        return rows[0], rows[1:]
    if inherited_header and not _looks_like_header(rows[0], taxonomy):
        return list(inherited_header), rows
    return rows[0], rows[1:]


def _hint_field(header: str, hints: Mapping[str, tuple[str, ...]]) -> str | None:
    lowered = header.lower()
    for field, keywords in hints.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            return field
    return None


def _align_to_header(rows: list[list[str]], header_units: list[str | None], taxonomy: Taxonomy) -> list[list[str]]:
    """Continuation pages often drop leading cells that span from the previous page, so a
    row comes back one or more columns short and shifted left. Re-align each short row so
    its number-with-unit cell sits under the header's unit-bearing column."""
    unit_columns = [col for col, unit in enumerate(header_units) if unit]
    if not unit_columns or not taxonomy.unit_alternation:
        return rows
    target = unit_columns[0]
    value_with_unit = re.compile(rf"\d\s*(?:{taxonomy.unit_alternation})(?![A-Za-z])", re.IGNORECASE)
    aligned = []
    for row in rows:
        if len(row) < len(header_units):
            found = next((col for col, cell in enumerate(row) if value_with_unit.search(cell)), None)
            shift = target - found if found is not None else 0
            if 0 < shift <= len(header_units) - len(row):
                row = [""] * shift + row
        aligned.append(row)
    return aligned


def _forward_fill_labels(rows: list[list[str]], header_units: list[str | None]) -> list[list[str]]:
    """Blank cells in label columns inherit the value above them (PDF tables render a
    spanning cell once and leave the rows beneath it empty)."""
    if not rows:
        return rows
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
    inherited_header: Sequence[str] | None = None,
) -> list[dict]:
    """Turn one parsed HTML table into rule rows. A cell yields a rule only when a number
    is paired with a known unit (in the cell or its column header), which keeps tables of
    contents and section numbering out of the rules table. Label columns are mapped onto
    rule fields via the config's table_field_hints; the rest fold into `condition`."""
    rows, explicit_header = parse_table(table_html)
    header, data_rows = resolve_table_header(rows, explicit_header, inherited_header, taxonomy)
    if not header or not data_rows:
        return []
    header_units = [taxonomy.find_unit(cell) for cell in header]
    data_rows = _forward_fill_labels(_align_to_header(data_rows, header_units, taxonomy), header_units)
    pattern = re.compile(_NUMBER + rf"\s*(?P<unit>{taxonomy.unit_alternation})?(?![A-Za-z])", re.IGNORECASE)
    field_names = cfg.extraction.field_names
    hints = cfg.extraction.table_field_hints

    def is_label_column(col: int) -> bool:
        return col >= len(header_units) or header_units[col] is None

    rules: list[dict] = []
    for row in data_rows:
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
    """Deterministic rule extraction from parsed tables (SQL + Python, no AI functions).
    Tables are processed in document order so continuation pages inherit headers."""
    taxonomy = Taxonomy(cfg.taxonomy)
    elements = spark.sql(build_table_elements_sql(cfg)).toPandas()
    headers: dict[tuple[str, str | None], list[str]] = {}
    rules: list[dict] = []
    for element in elements.itertuples(index=False):
        key = (element.plan_name, element.section_reference)
        inherited = headers.get(key)
        rules.extend(
            rules_from_table(
                plan_name=element.plan_name,
                element_id=None if pd.isna(element.element_id) else int(element.element_id),
                page_id=None if pd.isna(element.page_id) else int(element.page_id),
                section_reference=element.section_reference,
                table_html=element.table_html,
                cfg=cfg,
                taxonomy=taxonomy,
                inherited_header=inherited,
            )
        )
        rows, explicit_header = parse_table(element.table_html)
        header, _ = resolve_table_header(rows, explicit_header, inherited, taxonomy)
        if header:
            headers[key] = header
    df = pd.DataFrame(rules, columns=rule_columns(cfg))
    df = df.loc[~df.drop(columns=["citation_ids"]).duplicated()].reset_index(drop=True)
    df["page_id"] = df["page_id"].astype("Int64")
    return df
