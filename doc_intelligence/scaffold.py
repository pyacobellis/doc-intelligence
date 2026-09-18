"""Create the config files for a new document type from configs/_template.yaml."""

from __future__ import annotations

import re
from pathlib import Path

from doc_intelligence.config import load_document_type_config

_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]{1,30}$")

_QUESTIONS_SKELETON = """# Evaluation questions for the {display_name} document type.
# categories: factual | comparative | edge_case
# Fill in expected_answer (and set status: reviewed) once a subject-matter expert has
# verified it; expected_snippet is a phrase that must appear in a retrieved chunk.

questions:
  - id: {document_type}-f-001
    category: factual
    question: REPLACE ME - a question with a single, checkable answer
    expected_plans: []
    expected_snippet: null
    status: draft

  - id: {document_type}-c-001
    category: comparative
    question: REPLACE ME - a question that needs more than one document
    expected_plans: []
    status: draft

  - id: {document_type}-e-001
    category: edge_case
    question: REPLACE ME - something the documents do not cover
    expected_plans: []
    notes: A good answer says the excerpts do not address this.
    status: draft
"""


def new_document_type_config(
    document_type: str,
    display_name: str,
    *,
    configs_dir: str | Path = "configs",
    file_pattern: str | None = None,
    volume: str = "/Volumes/workspace/default/raw",
    force: bool = False,
) -> list[Path]:
    """Write configs/<type>.yaml and configs/<type>_eval_questions.yaml. Returns the paths."""
    if not _TYPE_RE.match(document_type):
        raise ValueError("document_type must be lowercase letters/digits/underscores, e.g. 'award'")
    configs_dir = Path(configs_dir)
    template = configs_dir / "_template.yaml"
    config_path = configs_dir / f"{document_type}.yaml"
    questions_path = configs_dir / f"{document_type}_eval_questions.yaml"
    if not force:
        for path in (config_path, questions_path):
            if path.exists():
                raise FileExistsError(f"{path} already exists (use --force to overwrite)")

    replacements = {
        "__DOCTYPE__": document_type,
        "__DISPLAY_NAME__": display_name,
        "__FILE_PATTERN__": file_pattern or f"{document_type.upper()}_%",
        "__VOLUME__": volume.rstrip("/"),
    }
    lines = template.read_text(encoding="utf-8").splitlines()
    first_content = next(i for i, line in enumerate(lines) if line.strip() and not line.startswith("#"))
    header = [
        f"# {display_name} ({document_type}) — generated from configs/_template.yaml by `doc-intel new-config`.",
        "# Review the extraction fields, table_field_hints and taxonomy before the first pipeline run.",
        "",
    ]
    text = "\n".join(header + lines[first_content:]) + "\n"
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    config_path.write_text(text, encoding="utf-8")
    questions_path.write_text(
        _QUESTIONS_SKELETON.format(document_type=document_type, display_name=display_name), encoding="utf-8"
    )
    load_document_type_config(config_path)  # fail loudly if the result does not parse
    return [config_path, questions_path]
