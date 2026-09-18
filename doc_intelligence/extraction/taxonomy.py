from __future__ import annotations

import re

from doc_intelligence.config import TaxonomyConfig


class Taxonomy:
    """Canonical rule types and unit normalisation for one document type.

    Both the deterministic (table) and LLM (text) extraction paths go through this so
    every plan's idiosyncratic wording lands on the same vocabulary."""

    def __init__(self, cfg: TaxonomyConfig):
        self.cfg = cfg
        self._unit_lookup: dict[str, str] = {}
        for canonical, aliases in cfg.unit_aliases.items():
            self._unit_lookup[canonical.lower()] = canonical
            for alias in aliases:
                self._unit_lookup[alias.lower()] = canonical
        self._fallback = "other" if "other" in cfg.rule_type_names else cfg.rule_type_names[-1]
        alternation = self.unit_alternation
        self._unit_re = (
            re.compile(rf"(?<![A-Za-z])({alternation})(?![A-Za-z])", re.IGNORECASE) if alternation else None
        )

    @property
    def rule_type_names(self) -> tuple[str, ...]:
        return self.cfg.rule_type_names

    @property
    def unit_alternation(self) -> str:
        """Regex alternation of every known unit spelling, longest first."""
        return "|".join(re.escape(k) for k in sorted(self._unit_lookup, key=len, reverse=True))

    def classify(self, text: str | None) -> str:
        lowered = (text or "").lower()
        for rule_type in self.cfg.rule_types:
            if any(keyword.lower() in lowered for keyword in rule_type.keywords):
                return rule_type.name
        return self._fallback

    def normalize_unit(self, unit: str | None) -> str | None:
        if unit is None:
            return None
        cleaned = " ".join(str(unit).split())
        if not cleaned or cleaned.lower() == "null":
            return None
        return self._unit_lookup.get(cleaned.lower(), cleaned)

    def find_unit(self, text: str | None) -> str | None:
        if not text or self._unit_re is None:
            return None
        match = self._unit_re.search(text)
        return match.group(1) if match else None

    def unit_case_sql(self, column: str) -> str:
        whens = "\n".join(
            f"          WHEN '{alias}' THEN '{canonical}'" for alias, canonical in self._unit_lookup.items()
        )
        return f"CASE lower(trim({column}))\n{whens}\n          ELSE nullif(trim({column}), '') END"

    def rule_type_case_sql(self, column: str) -> str:
        whens = []
        for rule_type in self.cfg.rule_types:
            if not rule_type.keywords:
                continue
            conditions = " OR ".join(f"lower({column}) LIKE '%{kw.lower()}%'" for kw in rule_type.keywords)
            whens.append(f"          WHEN {conditions} THEN '{rule_type.name}'")
        return "CASE\n" + "\n".join(whens) + f"\n          ELSE '{self._fallback}' END"
