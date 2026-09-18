from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml


@dataclass(frozen=True)
class SourceConfig:
    volume: str
    file_pattern: str
    listing_url: str | None
    link_pattern: str


@dataclass(frozen=True)
class TablesConfig:
    parsed_docs: str
    chunks: str
    rules: str
    eval_results: str
    document_versions: str
    change_log: str


@dataclass(frozen=True)
class ModelsConfig:
    llm: str
    embedding: str


@dataclass(frozen=True)
class VectorSearchConfig:
    endpoint_name: str
    index_name: str
    num_results: int
    query_type: str
    rrf_k: int


CHUNKING_STRATEGIES = ("ai_prep_search", "section", "fixed")


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = "ai_prep_search"
    # None = leave ai_prep_search's "Key: value" header untouched; () = drop it entirely;
    # otherwise keep only these keys in the embedded text (full header kept in chunk_header)
    header_keys_to_keep: tuple[str, ...] | None = None
    # section / fixed strategies: target chunk size in characters (~4 chars per token)
    max_chars: int = 1600
    # fixed strategy only: characters shared between consecutive windows
    overlap_chars: int = 200
    # named alternative settings for A/B runs; each variant gets its own chunk table + index
    variants: Mapping[str, Mapping] = MappingProxyType({})

    def with_overrides(self, overrides: Mapping) -> "ChunkingConfig":
        allowed = {"strategy", "header_keys_to_keep", "max_chars", "overlap_chars"}
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(f"unknown chunking keys: {sorted(unknown)}")
        values = {k: overrides[k] for k in allowed if k in overrides}
        if "header_keys_to_keep" in values and values["header_keys_to_keep"] is not None:
            values["header_keys_to_keep"] = tuple(values["header_keys_to_keep"])
        if "strategy" in values and values["strategy"] not in CHUNKING_STRATEGIES:
            raise ValueError(f"chunking.strategy must be one of {CHUNKING_STRATEGIES}")
        return replace(self, **values)


@dataclass(frozen=True)
class ExtractionField:
    name: str
    type: str
    description: str


@dataclass(frozen=True)
class ExtractionConfig:
    ai_extract_version: str
    chunk_keyword_filter: tuple[str, ...]
    instructions: str
    fields: tuple[ExtractionField, ...]
    # field name -> header keywords; maps label columns of parsed tables onto rule fields
    table_field_hints: Mapping[str, tuple[str, ...]] = MappingProxyType({})

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)


@dataclass(frozen=True)
class RuleTypeConfig:
    name: str
    description: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class TaxonomyConfig:
    rule_types: tuple[RuleTypeConfig, ...]
    unit_aliases: Mapping[str, tuple[str, ...]]

    @property
    def rule_type_names(self) -> tuple[str, ...]:
        return tuple(rt.name for rt in self.rule_types)


@dataclass(frozen=True)
class QAConfig:
    system_prompt: str
    summary_prompt: str
    change_summary_prompt: str


@dataclass(frozen=True)
class EvalConfig:
    questions_file: str
    judge_prompt: str


@dataclass(frozen=True)
class DocumentTypeConfig:
    document_type: str
    display_name: str
    catalog: str
    schema: str
    source: SourceConfig
    tables: TablesConfig
    models: ModelsConfig
    vector_search: VectorSearchConfig
    ai_parse_version: str
    chunking: ChunkingConfig
    extraction: ExtractionConfig
    taxonomy: TaxonomyConfig
    qa: QAConfig
    eval: EvalConfig

    def qualified_table(self, name: str) -> str:
        return f"{self.catalog}.{self.schema}.{name}"

    @property
    def parsed_docs_full_name(self) -> str:
        return self.qualified_table(self.tables.parsed_docs)

    @property
    def chunks_full_name(self) -> str:
        return self.qualified_table(self.tables.chunks)

    @property
    def rules_full_name(self) -> str:
        return self.qualified_table(self.tables.rules)

    @property
    def eval_results_full_name(self) -> str:
        return self.qualified_table(self.tables.eval_results)

    @property
    def document_versions_full_name(self) -> str:
        return self.qualified_table(self.tables.document_versions)

    @property
    def change_log_full_name(self) -> str:
        return self.qualified_table(self.tables.change_log)

    @property
    def chunks_index_full_name(self) -> str:
        return self.qualified_table(self.vector_search.index_name)


def load_document_type_config(path: str | Path) -> DocumentTypeConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return config_from_dict(raw)


def config_from_dict(raw: dict) -> DocumentTypeConfig:
    source = raw["source"]
    vs = raw["vector_search"]
    extraction = raw["extraction"]
    taxonomy = raw["taxonomy"]
    return DocumentTypeConfig(
        document_type=raw["document_type"],
        display_name=raw.get("display_name", raw["document_type"]),
        catalog=raw["catalog"],
        schema=raw["schema"],
        source=SourceConfig(
            volume=source["volume"].rstrip("/"),
            file_pattern=source["file_pattern"],
            listing_url=source.get("listing_url"),
            link_pattern=source.get("link_pattern", "[.]pdf$"),
        ),
        tables=TablesConfig(**raw["tables"]),
        models=ModelsConfig(**raw["models"]),
        vector_search=VectorSearchConfig(
            endpoint_name=vs["endpoint_name"],
            index_name=vs["index_name"],
            num_results=int(vs.get("num_results", 5)),
            query_type=str(vs.get("query_type", "ANN")).upper(),
            rrf_k=int(vs.get("rrf_k", 60)),
        ),
        ai_parse_version=str(raw.get("parsing", {}).get("ai_parse_version", "2.0")),
        chunking=_chunking_from_dict(raw.get("chunking") or {}),
        extraction=ExtractionConfig(
            ai_extract_version=str(extraction["ai_extract_version"]),
            chunk_keyword_filter=tuple(extraction.get("chunk_keyword_filter", ())),
            instructions=extraction["instructions"].strip(),
            fields=tuple(
                ExtractionField(f["name"], f["type"], f["description"]) for f in extraction["fields"]
            ),
            table_field_hints=MappingProxyType(
                {field: tuple(keywords) for field, keywords in extraction.get("table_field_hints", {}).items()}
            ),
        ),
        taxonomy=TaxonomyConfig(
            rule_types=tuple(
                RuleTypeConfig(name, spec.get("description", ""), tuple(spec.get("keywords", ())))
                for name, spec in taxonomy["rule_types"].items()
            ),
            unit_aliases=MappingProxyType(
                {unit: tuple(aliases) for unit, aliases in taxonomy.get("unit_aliases", {}).items()}
            ),
        ),
        qa=QAConfig(**{k: v.strip() for k, v in raw["qa"].items()}),
        eval=EvalConfig(
            questions_file=raw["eval"]["questions_file"],
            judge_prompt=raw["eval"]["judge_prompt"].strip(),
        ),
    )


def _chunking_from_dict(raw: Mapping) -> ChunkingConfig:
    base = ChunkingConfig().with_overrides({k: v for k, v in raw.items() if k != "variants"})
    variants = MappingProxyType({name: dict(spec or {}) for name, spec in (raw.get("variants") or {}).items()})
    for name, spec in variants.items():
        base.with_overrides(spec)  # fail fast on a bad variant
    return replace(base, variants=variants)


def with_chunk_variant(cfg: DocumentTypeConfig, variant: str | None) -> DocumentTypeConfig:
    """A copy of the config pointing at the variant's own chunk table and index, so
    alternative chunkings coexist and can be evaluated side by side."""
    if not variant:
        return cfg
    if variant not in cfg.chunking.variants:
        raise ValueError(f"unknown chunking variant {variant!r}; configured: {sorted(cfg.chunking.variants)}")
    suffix = f"__{variant}"
    return replace(
        cfg,
        chunking=cfg.chunking.with_overrides(cfg.chunking.variants[variant]),
        tables=replace(cfg.tables, chunks=cfg.tables.chunks + suffix),
        vector_search=replace(cfg.vector_search, index_name=cfg.vector_search.index_name + suffix),
    )
