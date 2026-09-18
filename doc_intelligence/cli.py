"""Thin command-line front door over the package. Every command maps onto one or two
package functions; no business logic lives here."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from doc_intelligence.config import DocumentTypeConfig, load_document_type_config

COSTLY = " (calls AI functions; costs DBU)"


def _show(frame) -> None:
    if isinstance(frame, pd.DataFrame):
        print(frame.to_string(index=False) if not frame.empty else "(no rows)")
    elif hasattr(frame, "toPandas"):
        _show(frame.toPandas())
    else:
        print(frame)


def _configured_profiles() -> list[str]:
    import configparser

    parser = configparser.ConfigParser()
    parser.read(Path.home() / ".databrickscfg")
    return [s for s in parser.sections() if s != "__settings__"]


def _check_profile(args) -> None:
    if args.profile and args.profile not in _configured_profiles():
        available = ", ".join(_configured_profiles()) or "(none)"
        raise SystemExit(
            f"unknown Databricks profile {args.profile!r}; available in ~/.databrickscfg: {available}\n"
            "tip: export DATABRICKS_CONFIG_PROFILE=<profile> to stop passing --profile"
        )


def _spark(args):
    from doc_intelligence.runtime import get_spark

    _check_profile(args)
    return get_spark(args.profile)


def _client(args):
    from doc_intelligence.runtime import get_workspace_client

    _check_profile(args)
    return get_workspace_client(args.profile)


def _runner(args):
    from doc_intelligence.runtime import spark_sql_runner

    return spark_sql_runner(_spark(args))


def _resolve(path: str, config_path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return Path(config_path).resolve().parent.parent / path


def _retriever(args, cfg: DocumentTypeConfig):
    from doc_intelligence.retrieval import bm25_retriever, load_chunks, rrf_hybrid_retriever, vector_retriever

    name = args.retriever
    if name == "vs_hybrid":
        return vector_retriever(_client(args), cfg, query_type="HYBRID", num_results=args.k)
    if name == "vs_ann":
        return vector_retriever(_client(args), cfg, query_type="ANN", num_results=args.k)
    chunks = load_chunks(_runner(args), cfg)
    if name == "bm25":
        return bm25_retriever(chunks, num_results=args.k)
    return rrf_hybrid_retriever(_client(args), cfg, chunks, num_results=args.k)


def cmd_parse(args, cfg):
    from doc_intelligence.parsing import parse_documents

    _show(parse_documents(_spark(args), cfg))


def cmd_chunk(args, cfg):
    from doc_intelligence.retrieval import chunk_documents

    _show(chunk_documents(_spark(args), cfg))


def cmd_index(args, cfg):
    from doc_intelligence.retrieval import build_index

    print(build_index(_client(args), cfg))


def cmd_extract_rules(args, cfg):
    from doc_intelligence.extraction import build_rules_table

    _show(build_rules_table(_spark(args), cfg))


def cmd_refresh_table_rules(args, cfg):
    from doc_intelligence.extraction import refresh_table_rules

    _show(refresh_table_rules(_spark(args), cfg))


def cmd_pipeline(args, cfg):
    for step in (cmd_parse, cmd_chunk, cmd_index, cmd_extract_rules):
        print(f"== {step.__name__.removeprefix('cmd_')} ==")
        step(args, cfg)


def cmd_status(args, cfg):
    from doc_intelligence.extraction.rules_table import build_rules_summary_sql
    from doc_intelligence.parsing.pdf_parser import build_parse_status_sql
    from doc_intelligence.retrieval import index_status
    from doc_intelligence.retrieval.chunking import build_chunk_summary_sql

    spark = _spark(args)
    print(f"config: {cfg.document_type} | chunking: {cfg.chunking.strategy}"
          f"{f' (variant {args.variant})' if args.variant else ''} -> {cfg.chunks_full_name}")
    for title, sql in (
        ("parsed documents", build_parse_status_sql(cfg)),
        ("chunks", build_chunk_summary_sql(cfg)),
        ("rules", build_rules_summary_sql(cfg)),
    ):
        print(f"== {title} ==")
        try:
            _show(spark.sql(sql))
        except Exception as error:  # a step that has not run yet
            print(f"unavailable: {str(error).splitlines()[0][:200]}")
    print("== vector index ==")
    try:
        print(index_status(_client(args), cfg))
    except Exception as error:
        print(f"unavailable: {str(error).splitlines()[0][:200]}")


def cmd_search(args, cfg):
    from doc_intelligence.retrieval import search_chunks

    hits = search_chunks(_client(args), cfg, args.query, num_results=args.k, query_type=args.type)
    for rank, hit in enumerate(hits, start=1):
        print(f"{rank}. [{hit.score:.4f}] {hit.plan_name} #{hit.chunk_index}\n   {hit.chunk_text[:300]!s}\n")


def cmd_ask(args, cfg):
    from doc_intelligence.retrieval import answer_question, vector_retriever

    w = _client(args)
    answer = answer_question(_runner(args), w, cfg, args.question, vector_retriever(w, cfg, query_type=args.type))
    print(answer.answer)
    print("\nSources:")
    for hit in answer.sources:
        print(f"  - {hit.plan_name} chunk {hit.chunk_index} (score {hit.score:.3f})")


def cmd_summarise(args, cfg):
    from doc_intelligence.retrieval import summarise_plans

    _show(summarise_plans(_runner(args), cfg))


def cmd_matrix(args, cfg):
    from doc_intelligence.comparison import rule_matrix

    print(rule_matrix(_runner(args), cfg).to_string())


def cmd_compare(args, cfg):
    from doc_intelligence.comparison import compare_rule_type, numeric_comparison

    detail = compare_rule_type(
        _runner(args), cfg, args.rule_type, min_confidence=args.min_confidence, extraction_source=args.source
    )
    print(f"== {args.rule_type}: {len(detail)} rules ==")
    _show(detail.head(args.limit))
    print("\n== numeric comparison (plans side by side) ==")
    _show(numeric_comparison(detail))


def cmd_eval(args, cfg):
    from doc_intelligence.eval import (
        evaluate_answers,
        evaluate_retrieval,
        load_eval_questions,
        results_frame,
        summarise_results,
        write_results,
    )

    questions = load_eval_questions(_resolve(cfg.eval.questions_file, args.config))
    retriever = _retriever(args, cfg)
    run_meta = dict(variant=args.variant, index_name=cfg.chunks_index_full_name)
    if args.answers:
        results = evaluate_answers(
            _runner(args), _client(args), cfg, questions, retriever, args.retriever, args.k, **run_meta
        )
    else:
        results = evaluate_retrieval(questions, retriever, args.retriever, args.k, **run_meta)
    frame = results_frame(results)
    _show(summarise_results(frame))
    if args.verbose:
        _show(frame.drop(columns=["answer", "judge_reason"], errors="ignore"))
    if args.write:
        write_results(_spark(args), cfg, frame)
        print(f"appended {len(frame)} rows to {cfg.eval_results_full_name}")


def cmd_registry(args, cfg):
    from doc_intelligence.monitoring import registry as reg
    from doc_intelligence.monitoring.source_scan import scan_source

    runner = _runner(args)
    if args.registry_command == "seed":
        snapshot = scan_source(cfg)
        entries = reg.seed_registry(snapshot, cfg)
        existing = reg.load_registry(runner, cfg)
        if existing and not args.force:
            print(f"registry already has {len(existing)} entries; use --force to rebuild it from the site")
            return
        spark = _spark(args)
        reg.write_registry(spark, cfg, entries)
        reg.append_observations(spark, cfg, reg.observations_from_snapshot(snapshot))
        confirmed = sum(e.confirmed for e in entries)
        print(f"seeded {len(entries)} documents from {len(snapshot.region_urls)} pages "
              f"({confirmed} confirmed/ingested, {len(entries) - confirmed} known but not ingested)")
        return
    entries = reg.load_registry(runner, cfg)
    if args.registry_command == "fetch-supporting":
        from doc_intelligence.monitoring.apply import apply_supporting

        kinds = [k.strip() for k in args.kinds.split(",")] if args.kinds else None
        links = reg.supporting_links_for(runner, cfg, args.plan_key, kinds)
        if not links:
            raise SystemExit(f"no supporting links observed for {args.plan_key!r} (run `registry seed` or `check-source` first)")
        entry = entries.get(args.plan_key)
        proposal = reg.Proposal("manual", f"manual:{args.plan_key}", reg._now(), "supporting_doc", args.plan_key,
                                entry.display_name if entry else args.plan_key, 1.0, {"new_supporting": links},
                                "fetch_supporting")
        result = apply_supporting(_client(args), cfg, proposal)
        for path in result.uploaded:
            print(f"  {path}")
        print(result.note)
        return
    if args.registry_command == "confirm":
        entry = entries.get(args.plan_key)
        if entry is None:
            raise SystemExit(f"unknown plan_key {args.plan_key!r}")
        entry.confirmed = True
        if args.plan_name:
            entry.plan_name = args.plan_name
        reg.write_registry(_spark(args), cfg, list(entries.values()))
        print(f"confirmed {args.plan_key}" + (f" -> {args.plan_name}" if args.plan_name else ""))
        return
    frame = reg.registry_frame(list(entries.values()))
    if args.unconfirmed:
        frame = frame[~frame.confirmed]
    _show(frame[["plan_key", "status", "confirmed", "plan_name", "instrument_id", "version_label", "last_seen"]])


def cmd_check_source(args, cfg):
    from doc_intelligence.monitoring import registry as reg
    from doc_intelligence.monitoring.source_scan import scan_source
    from doc_intelligence.monitoring.triage import enrich_with_candidates, llm_triage

    runner = _runner(args)
    registry = reg.load_registry(runner, cfg)
    if not registry:
        raise SystemExit("registry is empty: run `doc-intel registry seed` first to establish the baseline")
    snapshot = scan_source(cfg)
    existing = reg.load_proposals(runner, cfg)
    dedupe = set(existing["dedupe_key"].astype(str)) if not existing.empty else set()
    proposals = reg.detect_proposals(snapshot, registry, reg.load_seen_urls(runner, cfg), dedupe, cfg)
    enrich_with_candidates(proposals, registry)
    if args.llm and proposals:
        llm_triage(runner, cfg, proposals, registry)
    print(f"scanned {len(snapshot.region_urls)} pages, {len(snapshot.plans)} listings -> {len(proposals)} new proposals")
    for p in proposals:
        extra = ""
        if p.evidence.get("supersedes_candidates"):
            extra = f" | may supersede {p.evidence['supersedes_candidates'][0][0]}"
        if p.evidence.get("llm_triage"):
            extra += f" | llm: {p.evidence['llm_triage']['kind']} ({p.evidence['llm_triage']['reason']})"
        print(f"  [{p.kind:14}] {p.display_name[:60]:60} conf={p.confidence:.2f} -> {p.recommended_action}{extra}")
    if args.dry_run:
        print("dry run: nothing recorded")
        return
    spark = _spark(args)
    reg.append_observations(spark, cfg, reg.observations_from_snapshot(snapshot))
    reg.append_proposals(spark, cfg, proposals)
    reg.write_registry(spark, cfg, reg.touch_registry(registry, snapshot))
    print(f"recorded; review with `doc-intel proposals list`")


def cmd_proposals(args, cfg):
    import json

    from doc_intelligence.monitoring import registry as reg
    from doc_intelligence.monitoring.apply import apply_instrument, apply_supporting

    runner = _runner(args)
    if args.proposals_command == "list":
        frame = reg.load_proposals(runner, cfg, status=args.status)
        _show(frame[["proposal_id", "status", "kind", "display_name", "confidence", "recommended_action", "created_at"]]
              if not frame.empty else frame)
        return
    frame = reg.load_proposals(runner, cfg)
    row = frame[frame.proposal_id == args.proposal_id]
    if row.empty:
        raise SystemExit(f"no proposal {args.proposal_id}")
    record = row.iloc[0].to_dict()
    evidence = json.loads(record["evidence"]) if isinstance(record["evidence"], str) else record["evidence"]
    if args.proposals_command == "show":
        for key in ("proposal_id", "status", "kind", "display_name", "confidence", "recommended_action", "created_at",
                    "decision_note", "result"):
            print(f"{key:20} {record.get(key)}")
        print("evidence:")
        print(json.dumps(evidence, indent=2)[:4000])
        return
    if args.proposals_command in ("approve", "reject"):
        status = "approved" if args.proposals_command == "approve" else "rejected"
        reg.set_proposal_status(_spark(args), cfg, args.proposal_id, status, note=args.note)
        print(f"{args.proposal_id}: {status}")
        return
    # apply
    if record["status"] != "approved":
        raise SystemExit(f"proposal {args.proposal_id} is {record['status']}, not approved")
    proposal = reg.Proposal(
        proposal_id=record["proposal_id"], dedupe_key=record["dedupe_key"], created_at=record["created_at"],
        kind=record["kind"], plan_key=record["plan_key"], display_name=record["display_name"],
        confidence=float(record["confidence"]), evidence=evidence, recommended_action=record["recommended_action"],
    )
    w, spark = _client(args), _spark(args)
    registry = reg.load_registry(runner, cfg)
    if proposal.kind == "supporting_doc":
        result = apply_supporting(w, cfg, proposal)
    elif proposal.kind in ("new_plan", "new_version"):
        print(f"opening {proposal.evidence.get('instrument_url')} in your browser; save the PDF to your Downloads folder")
        result, entry = apply_instrument(w, spark, cfg, proposal, registry.get(proposal.plan_key),
                                         timeout_seconds=args.wait)
        registry[proposal.plan_key] = entry
        reg.write_registry(spark, cfg, list(registry.values()))
    else:
        raise SystemExit(f"nothing to apply for a {proposal.kind} proposal; mark it approved/rejected only")
    reg.set_proposal_status(spark, cfg, proposal.proposal_id, "applied", result=result.note)
    for path in result.uploaded:
        print(f"  {path}")
    for path in result.archived:
        print(f"  archived previous version -> {path}")
    print(result.note)


def cmd_new_config(args, cfg):
    from doc_intelligence.scaffold import new_document_type_config

    paths = new_document_type_config(
        args.document_type,
        args.display_name,
        configs_dir=Path(args.config).resolve().parent,
        file_pattern=args.file_pattern,
        volume=args.volume,
        force=args.force,
    )
    for path in paths:
        print(f"wrote {path}")
    print(
        "next: review the taxonomy/extraction sections, upload PDFs matching the file pattern "
        f"to the volume, then run: doc-intel --config {paths[0]} pipeline"
    )


def cmd_cost(args, cfg):
    from doc_intelligence.monitoring import cost_report

    for title, frame in cost_report(_runner(args), args.days).items():
        print(f"== {title} (last {args.days} days) ==")
        _show(frame)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="doc-intel", description="Config-driven document intelligence on Databricks")
    parser.add_argument("--config", default=os.environ.get("DOC_INTEL_CONFIG", "configs/wsp.yaml"))
    parser.add_argument(
        "--profile",
        default=os.environ.get("DATABRICKS_CONFIG_PROFILE"),
        help="Databricks CLI profile for local runs (omit inside a Databricks job)",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="chunking variant from the config (uses its own chunk table and index), e.g. section_400",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("parse", help="ai_parse_document over the source volume" + COSTLY).set_defaults(run=cmd_parse)
    sub.add_parser("chunk", help="ai_prep_search chunking" + COSTLY).set_defaults(run=cmd_chunk)
    sub.add_parser("index", help="ensure the Vector Search endpoint/index and sync").set_defaults(run=cmd_index)
    sub.add_parser("extract-rules", help="rebuild the rules table" + COSTLY).set_defaults(run=cmd_extract_rules)
    sub.add_parser(
        "refresh-table-rules", help="re-run only the deterministic table extraction (no AI calls)"
    ).set_defaults(run=cmd_refresh_table_rules)
    sub.add_parser("pipeline", help="parse, chunk, index, extract-rules" + COSTLY).set_defaults(run=cmd_pipeline)
    sub.add_parser("status", help="read-only health check of every stage").set_defaults(run=cmd_status)

    p = sub.add_parser("search", help="query the vector index")
    p.add_argument("query")
    p.add_argument("--type", choices=["ANN", "HYBRID"], default=None)
    p.add_argument("-k", type=int, default=None)
    p.set_defaults(run=cmd_search)

    p = sub.add_parser("ask", help="grounded, cited answer" + COSTLY)
    p.add_argument("question")
    p.add_argument("--type", choices=["ANN", "HYBRID"], default=None)
    p.set_defaults(run=cmd_ask)

    sub.add_parser("summarise", help="per-plan summaries" + COSTLY).set_defaults(run=cmd_summarise)
    sub.add_parser("matrix", help="rule counts per plan x rule type").set_defaults(run=cmd_matrix)

    p = sub.add_parser("compare", help="one rule type across plans, side by side")
    p.add_argument("rule_type")
    p.add_argument("--min-confidence", type=float, default=None)
    p.add_argument("--source", choices=["table", "text"], default=None)
    p.add_argument("--limit", type=int, default=40)
    p.set_defaults(run=cmd_compare)

    p = sub.add_parser("eval", help="run the eval question set")
    p.add_argument("--retriever", choices=["vs_hybrid", "vs_ann", "bm25", "rrf"], default="vs_hybrid")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--answers", action="store_true", help="also generate and judge answers" + COSTLY)
    p.add_argument("--write", action="store_true", help="append results to the eval results table")
    p.add_argument("--verbose", action="store_true", help="print per-question rows")
    p.set_defaults(run=cmd_eval)

    p = sub.add_parser("registry", help="known documents on the source site")
    rs = p.add_subparsers(dest="registry_command", required=True)
    q = rs.add_parser("seed", help="scan the site and record every listing as the baseline (writes)")
    q.add_argument("--force", action="store_true", help="rebuild even if the registry already exists")
    q = rs.add_parser("list", help="show the registry")
    q.add_argument("--unconfirmed", action="store_true", help="only entries not yet confirmed")
    q = rs.add_parser("confirm", help="mark an entry confirmed, optionally naming its file in the volume")
    q.add_argument("plan_key")
    q.add_argument("--plan-name", default=None)
    q = rs.add_parser("fetch-supporting", help="fetch a document's supporting PDFs into <volume>/supporting/<plan_key>/")
    q.add_argument("plan_key")
    q.add_argument("--kinds", default=None, help="comma-separated kinds, e.g. rule_summary,changes_fact_sheet")
    p.set_defaults(run=cmd_registry)

    p = sub.add_parser("check-source", help="scan the site for new/changed documents and record proposals")
    p.add_argument("--dry-run", action="store_true", help="print proposals without recording them")
    p.add_argument("--llm", action="store_true", help="add an LLM opinion to new-document proposals" + COSTLY)
    p.set_defaults(run=cmd_check_source)

    p = sub.add_parser("proposals", help="the change inbox")
    ps = p.add_subparsers(dest="proposals_command", required=True)
    q = ps.add_parser("list")
    q.add_argument("--status", choices=["proposed", "approved", "rejected", "applied"], default=None)
    for name in ("show", "approve", "reject", "apply"):
        q = ps.add_parser(name)
        q.add_argument("proposal_id")
        if name in ("approve", "reject"):
            q.add_argument("--note", default=None)
        if name == "apply":
            q.add_argument("--wait", type=int, default=300, help="seconds to wait for the browser download")
    p.set_defaults(run=cmd_proposals)

    p = sub.add_parser("cost", help="AI-function DBU/cost and token usage from system tables")
    p.add_argument("--days", type=int, default=7)
    p.set_defaults(run=cmd_cost)

    p = sub.add_parser("new-config", help="scaffold configs/<type>.yaml for a new document type")
    p.add_argument("document_type", help="short lowercase id, e.g. award")
    p.add_argument("--display-name", required=True, help='human name, e.g. "Modern Awards"')
    p.add_argument("--file-pattern", default=None, help="SQL LIKE pattern for PDFs (default <TYPE>_%%)")
    p.add_argument("--volume", default="/Volumes/workspace/default/raw")
    p.add_argument("--force", action="store_true", help="overwrite existing files")
    p.set_defaults(run=cmd_new_config)
    return parser


def main(argv: list[str] | None = None) -> int:
    from doc_intelligence.config import with_chunk_variant

    args = build_parser().parse_args(argv)
    cfg = with_chunk_variant(load_document_type_config(args.config), args.variant)
    args.run(args, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
