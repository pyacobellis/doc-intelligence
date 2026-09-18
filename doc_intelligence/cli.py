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


def _spark(args):
    from doc_intelligence.runtime import get_spark

    return get_spark(args.profile)


def _client(args):
    from doc_intelligence.runtime import get_workspace_client

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


def cmd_check_source(args, cfg):
    from doc_intelligence.monitoring.change_detection import apply_changes, check_source

    check = check_source(_runner(args), cfg)
    changes = check.changes
    print(f"new: {[v.file_name for v in changes.new]}")
    print(f"updated: {[c.file_name for _, c in changes.updated]}")
    print(f"unchanged: {list(changes.unchanged)}")
    if args.apply and changes.has_changes:
        for path in apply_changes(_client(args), _spark(args), cfg, check):
            print(f"uploaded {path}")
        print("re-run `pipeline` to parse the new versions")


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

    p = sub.add_parser("check-source", help="poll the listing page for new/updated documents")
    p.add_argument("--apply", action="store_true", help="upload changed documents to the volume and record versions")
    p.set_defaults(run=cmd_check_source)

    p = sub.add_parser("cost", help="AI-function DBU/cost and token usage from system tables")
    p.add_argument("--days", type=int, default=7)
    p.set_defaults(run=cmd_cost)
    return parser


def main(argv: list[str] | None = None) -> int:
    from doc_intelligence.config import with_chunk_variant

    args = build_parser().parse_args(argv)
    cfg = with_chunk_variant(load_document_type_config(args.config), args.variant)
    args.run(args, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
