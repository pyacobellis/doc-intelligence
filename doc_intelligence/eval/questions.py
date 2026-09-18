from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CATEGORIES = ("factual", "comparative", "edge_case")


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    category: str
    expected_answer: str | None = None
    expected_plans: tuple[str, ...] = ()
    expected_rule_type: str | None = None
    # chunk-level ground truth: a phrase that must appear in a retrieved chunk (case-insensitive)
    expected_snippet: str | None = None
    notes: str | None = None
    status: str = "draft"

    @property
    def gradable(self) -> bool:
        return bool(self.expected_answer)


def load_eval_questions(path: str | Path) -> list[EvalQuestion]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    questions: list[EvalQuestion] = []
    for item in raw.get("questions", []):
        category = item["category"]
        if category not in CATEGORIES:
            raise ValueError(f"question {item.get('id')}: category must be one of {CATEGORIES}, got {category!r}")
        questions.append(
            EvalQuestion(
                id=str(item["id"]),
                question=str(item["question"]).strip(),
                category=category,
                expected_answer=(str(item["expected_answer"]).strip() or None) if item.get("expected_answer") else None,
                expected_plans=tuple(item.get("expected_plans") or item.get("expected_documents") or ()),
                expected_rule_type=item.get("expected_rule_type"),
                expected_snippet=(str(item["expected_snippet"]).strip() or None) if item.get("expected_snippet") else None,
                notes=item.get("notes"),
                status=str(item.get("status", "draft")),
            )
        )
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate eval question ids")
    return questions
