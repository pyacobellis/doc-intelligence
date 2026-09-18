from doc_intelligence.config import config_from_dict


def test_config_loads_nested_sections(wsp_config):
    assert wsp_config.document_type == "wsp"
    assert wsp_config.source.volume == "/Volumes/workspace/default/raw"
    assert wsp_config.source.link_pattern == "[.]pdf$"
    assert wsp_config.tables.rules == "wsp_rules"
    assert wsp_config.rules_full_name == "workspace.default.wsp_rules"
    assert wsp_config.chunks_index_full_name == "workspace.default.wsp_chunks_index"
    assert wsp_config.vector_search.query_type == "HYBRID"
    assert wsp_config.models.llm.startswith("databricks-")
    assert wsp_config.chunking.header_keys_to_keep == ("Sections", "Section", "Tables", "Contains")


def test_extraction_fields_and_taxonomy_come_from_config(wsp_config):
    assert wsp_config.extraction.field_names == (
        "water_source", "condition", "value", "unit", "applies_to", "section_reference",
    )
    assert wsp_config.taxonomy.rule_type_names[0] == "cease_to_pump"
    assert "other" in wsp_config.taxonomy.rule_type_names
    assert wsp_config.taxonomy.unit_aliases["ML/day"][0] == "ml/day"


def test_defaults_apply_for_optional_keys():
    raw = {
        "document_type": "demo",
        "catalog": "c",
        "schema": "s",
        "source": {"volume": "/Volumes/c/s/raw/", "file_pattern": "%"},
        "tables": {
            "parsed_docs": "p", "chunks": "ch", "rules": "r", "eval_results": "e",
            "document_versions": "dv", "change_log": "cl",
        },
        "models": {"llm": "m", "embedding": "e"},
        "vector_search": {"endpoint_name": "ep", "index_name": "ix"},
        "extraction": {
            "ai_extract_version": "2.1",
            "instructions": " do it ",
            "fields": [{"name": "amount", "type": "number", "description": "d"}],
        },
        "taxonomy": {"rule_types": {"only": {}}},
        "qa": {"system_prompt": "a", "summary_prompt": "b", "change_summary_prompt": "c"},
        "eval": {"questions_file": "q.yaml", "judge_prompt": "j"},
    }
    cfg = config_from_dict(raw)
    assert cfg.display_name == "demo"
    assert cfg.source.volume == "/Volumes/c/s/raw"
    assert cfg.vector_search.query_type == "ANN" and cfg.vector_search.num_results == 5
    assert cfg.ai_parse_version == "2.0"
    assert cfg.chunking.header_keys_to_keep is None
    assert cfg.extraction.instructions == "do it"
    assert cfg.taxonomy.rule_type_names == ("only",)
    assert dict(cfg.taxonomy.unit_aliases) == {}
