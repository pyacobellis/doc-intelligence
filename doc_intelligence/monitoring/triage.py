"""Judgement over proposals, kept small and bounded: a fuzzy match to say which known
document a new heading probably supersedes, and an optional single LLM call that turns
the gathered evidence into a classification with reasons. Never applies anything."""

from __future__ import annotations

import difflib
import json
import re
from typing import Mapping, Sequence

from doc_intelligence.config import DocumentTypeConfig
from doc_intelligence.eval.metrics import _JSON_RE
from doc_intelligence.monitoring.registry import Proposal, RegistryEntry
from doc_intelligence.monitoring.source_scan import normalise_title
from doc_intelligence.runtime import SqlRunner, sql_literal

_YEAR = re.compile(r"\b(19|20)\d{2}\b")

TRIAGE_PROMPT = (
    "You maintain a registry of regulatory documents. Given evidence about a document "
    "listed on the publisher's site and the closest known documents, classify it. Reply "
    "with only a JSON object: {\"kind\": \"new_plan\"|\"new_version\"|\"amendment\"|\"irrelevant\", "
    "\"supersedes_plan_key\": <plan_key or null>, \"confidence\": <0-1>, \"reason\": \"<one sentence>\"}."
)


def _stem(title: str) -> str:
    return _YEAR.sub("", normalise_title(title).lower()).replace("water sharing plan for the", "").strip(" -")


def supersedes_candidates(
    display_name: str, registry: Mapping[str, RegistryEntry], limit: int = 3, min_score: float = 0.8
) -> list[tuple[str, float]]:
    """Known documents whose name (year stripped) resembles this one, best first. The bar is
    high on purpose: plan names share long suffixes ("... River Water Source")."""
    stem = _stem(display_name)
    scored = []
    for plan_key, entry in registry.items():
        score = difflib.SequenceMatcher(None, stem, _stem(entry.display_name)).ratio()
        if score >= min_score and entry.display_name != display_name:
            scored.append((plan_key, round(score, 3)))
    return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]


def enrich_with_candidates(proposals: Sequence[Proposal], registry: Mapping[str, RegistryEntry]) -> None:
    for proposal in proposals:
        if proposal.kind in ("new_plan", "draft"):
            proposal.evidence["supersedes_candidates"] = supersedes_candidates(proposal.display_name, registry)


def build_triage_prompt(proposal: Proposal, registry: Mapping[str, RegistryEntry]) -> str:
    candidates = [
        {"plan_key": key, "display_name": registry[key].display_name, "similarity": score,
         "status": registry[key].status}
        for key, score in proposal.evidence.get("supersedes_candidates", [])
        if key in registry
    ]
    evidence = {k: v for k, v in proposal.evidence.items() if k not in ("supporting", "supersedes_candidates")}
    evidence["supporting_kinds"] = sorted({s["kind"] for s in proposal.evidence.get("supporting", [])})
    return (
        f"{TRIAGE_PROMPT}\n\nDeterministic classification so far: {proposal.kind}\n"
        f"Evidence: {json.dumps(evidence)}\nClosest known documents: {json.dumps(candidates)}"
    )


def build_triage_sql(cfg: DocumentTypeConfig, proposal: Proposal, registry: Mapping[str, RegistryEntry]) -> str:
    return f"SELECT ai_query('{cfg.models.llm}', '{sql_literal(build_triage_prompt(proposal, registry))}') AS verdict"


def parse_triage_response(text: str | None) -> dict | None:
    match = _JSON_RE.search(text or "")
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence"))))
    except (TypeError, ValueError):
        confidence = None
    return {
        "kind": data.get("kind"),
        "supersedes_plan_key": data.get("supersedes_plan_key") or None,
        "confidence": confidence,
        "reason": str(data.get("reason", "")).strip(),
    }


def llm_triage(run_sql: SqlRunner, cfg: DocumentTypeConfig, proposals: Sequence[Proposal],
               registry: Mapping[str, RegistryEntry]) -> None:
    """One ai_query per new_plan/draft proposal (costs DBU); results land in the evidence
    and adjust confidence. The deterministic kind is kept; the LLM view is advisory."""
    for proposal in proposals:
        if proposal.kind not in ("new_plan", "draft"):
            continue
        verdict = parse_triage_response(str(run_sql(build_triage_sql(cfg, proposal, registry)).iloc[0, 0]))
        if verdict is None:
            continue
        proposal.evidence["llm_triage"] = verdict
        if verdict["confidence"] is not None:
            proposal.confidence = round((proposal.confidence + verdict["confidence"]) / 2, 3)
