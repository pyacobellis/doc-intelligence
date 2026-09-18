import shutil
from pathlib import Path

import pytest

from doc_intelligence.config import load_document_type_config
from doc_intelligence.eval.questions import load_eval_questions
from doc_intelligence.scaffold import new_document_type_config


@pytest.fixture
def configs_dir(tmp_path):
    shutil.copy(Path("configs/_template.yaml"), tmp_path / "_template.yaml")
    return tmp_path


def test_new_config_fills_placeholders_and_parses(configs_dir):
    paths = new_document_type_config("award", "Modern Awards", configs_dir=configs_dir)
    assert [p.name for p in paths] == ["award.yaml", "award_eval_questions.yaml"]
    text = paths[0].read_text(encoding="utf-8")
    assert "__" not in text.replace("award_", "")  # no placeholders left (allow award_parsed_docs etc.)

    cfg = load_document_type_config(paths[0])
    assert cfg.document_type == "award" and cfg.display_name == "Modern Awards"
    assert cfg.source.file_pattern == "AWARD_%"
    assert cfg.tables.chunks == "award_chunks" and cfg.chunks_index_full_name.endswith("award_chunks_index")
    assert cfg.chunking.strategy == "section" and "ai_prep" in cfg.chunking.variants
    assert cfg.taxonomy.rule_type_names[-1] == "other"
    assert cfg.eval.questions_file == "configs/award_eval_questions.yaml"

    questions = load_eval_questions(paths[1])
    assert {q.category for q in questions} == {"factual", "comparative", "edge_case"}
    assert all(q.id.startswith("award-") for q in questions)


def test_new_config_options_and_guards(configs_dir):
    with pytest.raises(ValueError):
        new_document_type_config("Bad Name", "x", configs_dir=configs_dir)
    new_document_type_config("lease", "Leases", configs_dir=configs_dir, file_pattern="LEASE-%", volume="/Volumes/c/s/docs/")
    cfg = load_document_type_config(configs_dir / "lease.yaml")
    assert cfg.source.file_pattern == "LEASE-%" and cfg.source.volume == "/Volumes/c/s/docs"
    with pytest.raises(FileExistsError):
        new_document_type_config("lease", "Leases", configs_dir=configs_dir)
    new_document_type_config("lease", "Leases v2", configs_dir=configs_dir, force=True)
    assert load_document_type_config(configs_dir / "lease.yaml").display_name == "Leases v2"


def test_wsp_config_and_template_share_a_schema():
    """Whatever keys the WSP config uses, the template must have too (and vice versa)."""
    import yaml

    wsp = yaml.safe_load(Path("configs/wsp.yaml").read_text(encoding="utf-8"))
    template = yaml.safe_load(Path("configs/_template.yaml").read_text(encoding="utf-8"))
    assert set(wsp) == set(template)
    for section in ("source", "tables", "models", "vector_search", "chunking", "extraction", "qa", "eval"):
        assert set(wsp[section]) == set(template[section]), section
