from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - zero-dependency fallback.
    def tqdm(iterable, **_kwargs):
        return iterable

from expected_answer_rag.analysis import (
    compare_methods,
    evaluate_by_leakage_bucket,
    generation_features,
    summarize_generation_features,
    summarize_method_leakage,
)
from expected_answer_rag.cache import JsonCache
from expected_answer_rag.counterfactual import (
    build_entity_counterfactual_dataset,
    export_counterfactual_artifacts,
    load_counterfactual_artifacts,
    resolve_counterfactual_artifact_dir,
)
from expected_answer_rag.datasets import Document, export_local_dataset, load_dataset
from expected_answer_rag.fusion import reciprocal_rank_fusion, weighted_reciprocal_rank_fusion
from expected_answer_rag.generators import (
    CachedTextGenerator,
    HeuristicGenerator,
    MissingGenerator,
    OpenAITextGenerator,
    build_counterfactual_prompt_payload,
    entity_only_mask_answer,
    extract_query_anchors,
    generic_mask_answer,
    length_matched_neutral_filler,
    parse_counterfactual_prompt_query_expansion,
    parse_answer_candidate_template,
    random_span_mask_answer,
    remove_gold_from_text,
    validate_answer_candidate_template,
)
from expected_answer_rag.leakage import extract_concrete_candidates, leakage_bucket_name, normalize_text, score_generation_methods, tokenize
from expected_answer_rag.metrics import evaluate_run, per_query_metrics
from expected_answer_rag.qualitative import select_qualitative_examples
from expected_answer_rag.retrieval import RankedList, make_retriever
from expected_answer_rag.statistics import paired_bootstrap_ci, paired_permutation_test, win_tie_loss


FROZEN_PRIMARY_METRICS = ["ndcg@10", "recall@10", "recall@20", "recall@100", "mrr@10"]
FROZEN_PRIMARY_COMPARISONS = [
    ("concat_query_raw_expected", "concat_query_masked_expected"),
    ("concat_query_raw_expected", "concat_query_answer_candidate_constrained_template"),
    ("hyde_doc_only", "concat_query_answer_candidate_constrained_template"),
    ("query2doc_concat", "concat_query_answer_candidate_constrained_template"),
    ("concat_query_raw_expected", "corpus_steered_expansion_concat"),
    ("safe_rrf_v0", "query2doc_concat"),
    ("safe_rrf_v1", "safe_rrf_v0"),
    ("cf_prompt_query_expansion_rrf", "concat_query_answer_candidate_constrained_template"),
]

SAFE_RRF_V0_WEIGHTS = {
    "query_only": 1.0,
    "generative_relevance_feedback_concat": 0.8,
    "query2doc_concat": 0.55,
    "concat_query_answer_candidate_constrained_template": 0.55,
}

ROUTE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}

CF_PROMPT_BASE_WEIGHTS = {
    "relation": 0.55,
    "evidence": 0.45,
    "answer_slot": 0.35,
    "bridge": 0.5,
    "generic": 0.3,
}

CF_PROMPT_GENERIC_PATTERNS = (
    "information about",
    "documents about",
    "what does",
    "guidance on",
    "example disclosures",
    "illustrative notes",
)

FAWE_DEFAULT_BETAS = {
    "raw_expected": 0.25,
    "masked_expected": 0.25,
    "answer_constrained": 0.5,
    "query2doc": 0.25,
}

MAIN_METHODS = [
    "query_only",
    "bm25_rm3_query_only",
    "hyde_doc_only",
    "query2doc_concat",
    "generative_relevance_feedback_concat",
    "corpus_steered_short_concat",
    "raw_expected_answer_only",
    "concat_query_raw_expected",
    "masked_expected_answer_only",
    "concat_query_masked_expected",
    "answer_candidate_constrained_template_only",
    "concat_query_answer_candidate_constrained_template",
    "random_span_masking",
    "safe_rrf_v0",
    "safe_rrf_v1",
    "cf_prompt_query_expansion_rrf",
    "concat_query_random_span_masking",
    "entity_only_masking",
    "concat_query_entity_only_masking",
    "generic_mask_slot",
    "concat_query_generic_mask_slot",
    "concat_query_wrong_answer",
    "query_repeated",
    "query_repeated_length_matched",
    "query_plus_shuffled_expected",
    "query_plus_neutral_filler",
    "neutral_filler_plus_query",
    "raw_expected_then_query",
    "fawe_raw_expected_beta0p25",
    "fawe_masked_expected_beta0p25",
    "fawe_answer_constrained_beta0p5",
    "fawe_query2doc_beta0p25",
    "fawe_safe_adaptive_beta",
]

SUSPICIOUS_IDENTITY_PAIRS = [
    ("gold_answer_only", "raw_expected_answer_only"),
    ("oracle_answer_masked", "raw_expected_answer_only"),
    ("post_hoc_gold_removed_expected_answer", "raw_expected_answer_only"),
    ("concat_query_oracle_answer_masked", "concat_query_raw_expected"),
    ("concat_query_post_hoc_gold_removed_expected", "concat_query_raw_expected"),
    ("dual_query_masked_expected_rrf", "rrf_query_masked_expected"),
    ("dual_query_answer_candidate_constrained_template_rrf", "rrf_query_answer_constrained"),
    ("concat_query_wrong_answer", "wrong_answer_injection"),
]

RETRIEVAL_AUDIT_METHODS = [
    "query_only",
    "raw_expected_answer_only",
    "concat_query_raw_expected",
    "raw_expected_then_query",
    "query2doc_concat",
    "concat_query_answer_candidate_constrained_template",
    "safe_rrf_v1",
    "query_repeated",
    "query_repeated_length_matched",
    "query_plus_shuffled_expected",
    "query_plus_neutral_filler",
    "neutral_filler_plus_query",
    "length_matched_neutral_filler",
    "concat_query_wrong_answer",
    "fawe_raw_expected_beta0p25",
    "fawe_answer_constrained_beta0p5",
    "fawe_safe_adaptive_beta",
]

CONCAT_RESCUE_PAIRS = [
    ("raw_expected_answer_only", "concat_query_raw_expected"),
    ("masked_expected_answer_only", "concat_query_masked_expected"),
    ("answer_candidate_constrained_template_only", "concat_query_answer_candidate_constrained_template"),
    ("random_span_masking", "concat_query_random_span_masking"),
    ("entity_only_masking", "concat_query_entity_only_masking"),
    ("generic_mask_slot", "concat_query_generic_mask_slot"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage-aware retrieval baselines.")
    parser.add_argument("--dataset", default="toy", help="toy, BEIR dataset name, or local dataset path")
    parser.add_argument("--max-corpus", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--query-metadata", default=None)
    parser.add_argument("--counterfactual", choices=["none", "entity", "entity_and_value"], default="none")
    parser.add_argument("--counterfactual-alias-style", choices=["natural", "coded"], default="natural")
    parser.add_argument("--counterfactual-seed", type=int, default=13)
    parser.add_argument("--counterfactual-export-dir", default=None)
    parser.add_argument("--counterfactual-artifact-root", default=None)
    parser.add_argument("--retriever", choices=["bm25", "dense", "hybrid", "lexical_neural"], default="bm25")
    parser.add_argument("--include-rm3-baseline", action="store_true")
    parser.add_argument("--rm3-fb-docs", type=int, default=10)
    parser.add_argument("--rm3-fb-terms", type=int, default=10)
    parser.add_argument("--rm3-original-query-weight", type=float, default=0.5)
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--embedding-chunk-size", type=int, default=1024)
    parser.add_argument("--embedding-cache", default=None)
    parser.add_argument("--local-files-only", action="store_true", help="Force dense model loading from local cache only.")
    parser.add_argument(
        "--query-prefix",
        default="Represent this sentence for searching relevant passages: ",
        help="Prefix applied only to dense retrieval queries. Use '' to disable.",
    )
    parser.add_argument("--generator", choices=["heuristic", "openai", "openrouter"], default="heuristic")
    parser.add_argument("--model", default="openai/gpt-5-mini")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--token-param", choices=["auto", "max_tokens", "max_completion_tokens", "none"], default="none")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--referer", default=None)
    parser.add_argument("--app-title", default="expected-answer-rag")
    parser.add_argument("--include-reasoning", action="store_true")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--generation-cache", default="outputs/generation_cache.json")
    parser.add_argument("--generation-workers", type=int, default=1, help="Threads for precomputing generations across queries.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--metric-ks",
        default="5,10,20,100",
        help="Comma-separated recall cutoffs to compute from each ranking. Use top-k >= max(metric-ks).",
    )
    parser.add_argument("--method-profile", choices=["full", "main"], default="full")
    parser.add_argument("--include-fawe-controls", action="store_true")
    parser.add_argument("--include-fawe-beta-grid", action="store_true")
    parser.add_argument(
        "--answer-rrf-weights",
        default="0.25,0.5,0.75",
        help="Comma-separated answer-route weights for weighted RRF. Query route weight is 1.0.",
    )
    parser.add_argument(
        "--fawe-betas",
        default="0.25,0.5",
        help="Comma-separated FAWE beta values. Named FAWE methods use the closest configured beta.",
    )
    parser.add_argument("--output", default="outputs/run.json")
    parser.add_argument("--records-output", default="outputs/records.jsonl")
    parser.add_argument("--audit-sample-size", type=int, default=20)
    parser.add_argument("--audit-seed", type=int, default=17)
    parser.add_argument("--clear-generation-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true", help="Use existing generation cache and fail if any generation is missing.")
    parser.add_argument("--cache-namespace", default=None, help="Override generation cache namespace.")
    parser.add_argument("--qualitative-limit", type=int, default=5)
    parser.add_argument("--stats-bootstrap-samples", type=int, default=500)
    parser.add_argument("--stats-permutation-samples", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    run_start = time.perf_counter()
    args = parse_args()
    counterfactual_artifact_dir: Path | None = None
    print(
        f"Phase 0/4: Loading dataset {args.dataset} "
        f"(max_queries={args.max_queries}, max_corpus={args.max_corpus})"
    )
    dataset = load_dataset(
        args.dataset,
        max_corpus=args.max_corpus,
        max_queries=args.max_queries,
        cache_dir=args.cache_dir,
        query_metadata_path=args.query_metadata,
    )
    if args.counterfactual != "none":
        include_values = args.counterfactual == "entity_and_value"
        if args.counterfactual_artifact_root:
            counterfactual_artifact_dir = resolve_counterfactual_artifact_dir(
                _resolve_path(args.counterfactual_artifact_root),
                dataset,
                alias_style=args.counterfactual_alias_style,
                include_values=include_values,
                seed=args.counterfactual_seed,
            )
        if counterfactual_artifact_dir and (counterfactual_artifact_dir / "manifest.json").exists():
            print(
                f"Phase 0.5/4: Reusing counterfactual dataset "
                f"from {counterfactual_artifact_dir}"
            )
            counterfactual = load_counterfactual_artifacts(counterfactual_artifact_dir)
        else:
            print(
                f"Phase 0.5/4: Building counterfactual dataset "
                f"(regime={args.counterfactual}, alias_style={args.counterfactual_alias_style})"
            )
            counterfactual = build_entity_counterfactual_dataset(
                dataset,
                alias_style=args.counterfactual_alias_style,
                include_values=include_values,
                seed=args.counterfactual_seed,
                progress=print,
            )
            if counterfactual_artifact_dir:
                export_local_dataset(counterfactual.dataset, counterfactual_artifact_dir)
                export_counterfactual_artifacts(counterfactual, counterfactual_artifact_dir)
                print(f"Phase 0.5/4: Cached counterfactual artifact at {counterfactual_artifact_dir}")
        dataset = counterfactual.dataset
        if args.counterfactual_export_dir:
            export_local_dataset(dataset, _resolve_path(args.counterfactual_export_dir))
            export_counterfactual_artifacts(counterfactual, _resolve_path(args.counterfactual_export_dir))
        counterfactual_validation = counterfactual.validation
        print(
            f"Phase 0.5/4 complete: Counterfactual corpus={len(dataset.corpus)}, "
            f"queries={len(dataset.queries)}, qrels_queries={len(dataset.qrels)}"
        )
    else:
        counterfactual_validation = None

    print(
        f"Loaded {dataset.name}: corpus={len(dataset.corpus)}, "
        f"queries={len(dataset.queries)}, qrels_queries={len(dataset.qrels)}"
    )
    retriever = make_retriever(
        args.retriever,
        dataset.corpus,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        query_prefix=args.query_prefix,
        embedding_cache=_resolve_path(args.embedding_cache) if args.embedding_cache else None,
        embedding_chunk_size=args.embedding_chunk_size,
        local_files_only=args.local_files_only,
    )
    base_generator = _make_base_generator(args)
    cache_path = _resolve_path(args.generation_cache)
    if args.clear_generation_cache and cache_path.exists():
        cache_path.unlink()
    generator = CachedTextGenerator(
        inner=base_generator,
        cache=JsonCache(cache_path),
        namespace=args.cache_namespace or _default_cache_namespace(args, dataset.name),
        namespace_aliases=_legacy_cache_namespaces(args),
    )
    answer_rrf_weights = _parse_float_list(args.answer_rrf_weights)
    fawe_betas = _parse_float_list(args.fawe_betas)
    metric_ks = [int(value) for value in _parse_float_list(args.metric_ks)]
    selected_methods = _select_methods(
        args.method_profile,
        answer_rrf_weights,
        fawe_betas,
        include_rm3_baseline=args.include_rm3_baseline,
        include_fawe_controls=args.include_fawe_controls,
        include_fawe_beta_grid=args.include_fawe_beta_grid,
    )
    runs = _initialize_runs(
        answer_rrf_weights,
        fawe_betas,
        selected_methods,
        include_rm3_baseline=args.include_rm3_baseline,
        include_fawe_controls=args.include_fawe_controls,
        include_fawe_beta_grid=args.include_fawe_beta_grid,
    )
    records_path = _resolve_path(args.records_output)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    doc_map = {doc.doc_id: doc for doc in dataset.corpus}
    cache_namespace = generator.namespace
    cache_namespace_aliases = _legacy_cache_namespaces(args)
    checkpoint_context = {
        "dataset": dataset.name,
        "retriever": args.retriever,
        "counterfactual_regime": args.counterfactual,
        "method_profile": args.method_profile,
        "top_k": args.top_k,
    }
    checkpoint_records = _load_checkpoint_records(records_path, dataset, runs.keys(), checkpoint_context)
    checkpoint_state = _restore_checkpoint_state(checkpoint_records, runs, dataset, metric_ks=metric_ks)
    completed_query_ids = set(checkpoint_state["completed_query_ids"])
    remaining_queries = [query for query in dataset.queries if query.query_id not in completed_query_ids]
    query_positions = {query.query_id: index for index, query in enumerate(dataset.queries)}
    if checkpoint_records:
        print(
            f"Resume: loaded {len(checkpoint_records)} compatible checkpoint record(s); "
            f"{len(remaining_queries)} querie(s) remaining"
        )
    query_contexts = {}
    print(
        f"Phase 1/4: First-pass retrieval over {len(remaining_queries)} remaining querie(s) "
        f"(retriever={args.retriever}, top_k={args.top_k})"
    )
    for query in tqdm(remaining_queries, desc="First-pass retrieval"):
        query_rank = retriever.search(query.text, top_k=args.top_k)
        query_contexts[query.query_id] = {
            "query_rank": query_rank,
            "cf_prompt_support": _build_cf_prompt_support_context(query.text, query_rank, doc_map),
            "corpus_steered_text": _build_corpus_steered_expansion(query.text, query_rank, doc_map),
            "corpus_steered_short_text": _build_corpus_steered_expansion(
                query.text,
                query_rank,
                doc_map,
                max_docs=2,
                max_words=40,
            ),
        }
    print(
        f"Phase 2/4: Precomputing generations for {len(remaining_queries)} remaining querie(s) "
        f"(workers={args.generation_workers}, generator={args.generator}, model={args.model})"
    )
    generation_bundles = _precompute_generation_bundles(
        args=args,
        dataset_name=dataset.name,
        queries=remaining_queries,
        cache=generator.cache,
        cache_namespace=cache_namespace,
        cache_namespace_aliases=cache_namespace_aliases,
        support_contexts={
            query_id: str(bundle["cf_prompt_support"]["prompt_context"])
            for query_id, bundle in query_contexts.items()
        },
    )
    print(f"Phase 3/4: Running retrieval fusion and evaluation over {len(remaining_queries)} remaining querie(s)")
    features_by_query = dict(checkpoint_state["features_by_query"])
    generations = dict(checkpoint_state["generations"])
    records = list(checkpoint_state["records"])
    leakage_by_method = {
        name: list(checkpoint_state["leakage_by_method"].get(name, []))
        for name in runs
    }
    per_query_method_metrics = {
        name: dict(checkpoint_state["per_query_method_metrics"].get(name, {}))
        for name in runs
    }

    checkpoint_handle = records_path.open("a", encoding="utf-8")
    try:
        for index, query in enumerate(tqdm(remaining_queries, desc="Running queries")):
            processed = _process_query(
                args=args,
                dataset=dataset,
                query=query,
                query_index=query_positions[query.query_id],
                query_context=query_contexts[query.query_id],
                generation_bundle=generation_bundles[query.query_id],
                retriever=retriever,
                doc_map=doc_map,
                answer_rrf_weights=answer_rrf_weights,
                fawe_betas=fawe_betas,
                run_method_names=runs.keys(),
                checkpoint_context=checkpoint_context,
            )
            query_id = query.query_id
            features_by_query[query_id] = processed["features_by_query"]
            generations[query_id] = processed["generation"]
            for method_name, score in processed["leakage_scores"].items():
                if method_name in leakage_by_method:
                    leakage_by_method[method_name].append(score)
            for run_name, ranking in processed["record"]["rankings"].items():
                if run_name not in runs:
                    continue
                runs[run_name][query_id] = ranking
                per_query_method_metrics[run_name][query_id] = per_query_metrics(
                    ranking,
                    dataset.qrels.get(query_id, {}),
                    ks=metric_ks,
                )
            records.append(processed["record"])
            _append_checkpoint_record(checkpoint_handle, processed["record"])
    finally:
        checkpoint_handle.close()

    metrics = {name: evaluate_run(run, dataset.qrels, ks=metric_ks) for name, run in runs.items()}
    metric_error_bars = _compute_metric_error_bars(
        per_query_method_metrics,
        bootstrap_samples=args.stats_bootstrap_samples,
    )
    leakage_metrics = {name: evaluate_by_leakage_bucket(run, dataset.qrels, features_by_query) for name, run in runs.items()}
    stats = _compute_primary_comparisons(
        dataset,
        per_query_method_metrics,
        comparisons=FROZEN_PRIMARY_COMPARISONS,
        bootstrap_samples=args.stats_bootstrap_samples,
        permutation_samples=args.stats_permutation_samples,
    )
    duplicate_method_audit = _duplicate_method_audit(records, SUSPICIOUS_IDENTITY_PAIRS)
    concatenation_rescue_summary = _concatenation_rescue_summary(records, CONCAT_RESCUE_PAIRS)
    query_dominance_summary = _query_dominance_summary(records)
    retrieval_audit_samples = _sample_retrieval_audit(
        records,
        audit_methods=RETRIEVAL_AUDIT_METHODS,
        sample_size=args.audit_sample_size,
        seed=args.audit_seed,
    )
    dense_position_audit = _dense_position_audit(records) if args.retriever in {"dense", "hybrid", "lexical_neural"} else None
    qualitative = {
        "raw_expected_answer_only": select_qualitative_examples(records, "raw_expected_answer_only", limit=args.qualitative_limit),
        "masked_expected_answer_only": select_qualitative_examples(records, "masked_expected_answer_only", limit=args.qualitative_limit),
        "answer_candidate_constrained_template_only": select_qualitative_examples(records, "answer_candidate_constrained_template_only", limit=args.qualitative_limit),
        "cf_prompt_query_expansion_rrf": select_qualitative_examples(records, "cf_prompt_query_expansion_rrf", limit=args.qualitative_limit),
    }
    result = {
        "dataset": dataset.name,
        "dataset_metadata": dict(dataset.metadata),
        "counterfactual_validation": counterfactual_validation,
        "num_corpus": len(dataset.corpus),
        "num_queries": len(dataset.queries),
        "num_qrels_queries": len(dataset.qrels),
        "qrel_coverage": dataset.metadata.get("qrel_coverage"),
        "retriever": args.retriever,
        "embedding_model": args.embedding_model if args.retriever in {"dense", "hybrid", "lexical_neural"} else None,
        "query_prefix": args.query_prefix if args.retriever in {"dense", "hybrid", "lexical_neural"} else None,
        "embedding_cache": str(_resolve_path(args.embedding_cache)) if args.embedding_cache else None,
        "answer_rrf_weights": answer_rrf_weights,
        "fawe_betas": fawe_betas,
        "method_profile": args.method_profile,
        "include_rm3_baseline": args.include_rm3_baseline,
        "include_fawe_controls": args.include_fawe_controls,
        "include_fawe_beta_grid": args.include_fawe_beta_grid,
        "rm3_params": {
            "fb_docs": args.rm3_fb_docs,
            "fb_terms": args.rm3_fb_terms,
            "original_query_weight": args.rm3_original_query_weight,
        } if args.include_rm3_baseline else None,
        "methods_evaluated": list(runs),
        "generator": args.generator,
        "model": args.model if args.generator in {"openai", "openrouter"} else None,
        "generation_cache_path": str(cache_path),
        "generation_cache_namespace": generator.namespace,
        "counterfactual_regime": args.counterfactual,
        "counterfactual_alias_style": args.counterfactual_alias_style if args.counterfactual != "none" else None,
        "counterfactual_artifact_dir": str(counterfactual_artifact_dir) if counterfactual_artifact_dir else None,
        "top_k": args.top_k,
        "metric_ks": metric_ks,
        "runtime_seconds": round(time.perf_counter() - run_start, 3),
        "resume_summary": {
            "checkpoint_records_loaded": len(checkpoint_records),
            "queries_completed_from_checkpoint": len(completed_query_ids),
            "queries_processed_this_run": len(remaining_queries),
        },
        "frozen_primary_metrics": FROZEN_PRIMARY_METRICS,
        "frozen_primary_comparisons": FROZEN_PRIMARY_COMPARISONS,
        "metrics": metrics,
        "metric_error_bars": metric_error_bars,
        "method_ranking": compare_methods(metrics),
        "generation_summary": summarize_generation_features(features_by_query.values()),
        "method_leakage_summary": summarize_method_leakage(leakage_by_method),
        "leakage_bucket_metrics": leakage_metrics,
        "primary_comparison_stats": stats,
        "concatenation_rescue_summary": concatenation_rescue_summary,
        "query_dominance_summary": query_dominance_summary,
        "duplicate_method_audit": duplicate_method_audit,
        "retrieval_audit_samples": retrieval_audit_samples,
        "dense_position_audit": dense_position_audit,
        "qualitative_examples": qualitative,
        "integrity_summary": _summarize_integrity(records, counterfactual_validation),
        "sample_generations": dict(list(generations.items())[:5]),
    }

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    records_path = _resolve_path(args.records_output)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
    print("Method ranking:")
    print(json.dumps(result["method_ranking"], indent=2, ensure_ascii=False))
    print(f"Wrote {output_path}")
    print(f"Wrote {records_path}")


def _make_base_generator(args: argparse.Namespace):
    if args.cache_only:
        return MissingGenerator()
    if args.generator in {"openai", "openrouter"}:
        return OpenAITextGenerator(
            model=args.model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            token_param=args.token_param,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            referer=args.referer,
            app_title=args.app_title,
            include_reasoning=args.include_reasoning,
            reasoning_effort=args.reasoning_effort,
        )
    return HeuristicGenerator()


def _initialize_runs(
    answer_rrf_weights: list[float],
    fawe_betas: list[float],
    selected_methods: list[str],
    include_rm3_baseline: bool = False,
    include_fawe_controls: bool = False,
    include_fawe_beta_grid: bool = False,
) -> Dict[str, Dict[str, RankedList]]:
    base_methods = {
        "query_only": {},
        **({"bm25_rm3_query_only": {}} if include_rm3_baseline else {}),
        "hyde_doc_only": {},
        "query2doc_concat": {},
        "generative_relevance_feedback_concat": {},
        "corpus_steered_expansion_concat": {},
        "corpus_steered_short_concat": {},
        "raw_expected_answer_only": {},
        "concat_query_raw_expected": {},
        "dual_query_raw_expected_rrf": {},
        "masked_expected_answer_only": {},
        "concat_query_masked_expected": {},
        "dual_query_masked_expected_rrf": {},
        "rrf_query_masked_expected": {},
        "answer_candidate_constrained_template_only": {},
        "concat_query_answer_candidate_constrained_template": {},
        "dual_query_answer_candidate_constrained_template_rrf": {},
        "rrf_query_answer_constrained": {},
        "gold_answer_only": {},
        "oracle_answer_masked": {},
        "concat_query_oracle_answer_masked": {},
        "post_hoc_gold_removed_expected_answer": {},
        "concat_query_post_hoc_gold_removed_expected": {},
        "random_span_masking": {},
        "concat_query_random_span_masking": {},
        "entity_only_masking": {},
        "concat_query_entity_only_masking": {},
        "generic_mask_slot": {},
        "concat_query_generic_mask_slot": {},
        "length_matched_neutral_filler": {},
        "wrong_answer_only": {},
        "concat_query_wrong_answer": {},
        "rrf_query_wrong_answer": {},
        "wrong_answer_injection": {},
        "query_repeated": {},
        "query_repeated_length_matched": {},
        "query_plus_shuffled_expected": {},
        "query_plus_neutral_filler": {},
        "neutral_filler_plus_query": {},
        "raw_expected_then_query": {},
        "rrf_query_corpus_steered_short": {},
        "safe_rrf_v0": {},
        "safe_rrf_v1": {},
        "cf_prompt_query_expansion_rrf": {},
        "fawe_raw_expected_beta0p25": {},
        "fawe_masked_expected_beta0p25": {},
        "fawe_answer_constrained_beta0p5": {},
        "fawe_query2doc_beta0p25": {},
        "fawe_safe_adaptive_beta": {},
    }
    if include_fawe_controls:
        base_methods["fawe_shuffled_expected_beta0p25"] = {}
        base_methods["fawe_wrong_answer_beta0p25"] = {}
        base_methods["fawe_neutral_filler_beta0p25"] = {}
        base_methods["fawe_query_repeated_beta0p25"] = {}
        base_methods["fawe_random_terms_from_corpus_beta0p25"] = {}
        base_methods["fawe_idf_matched_random_terms_beta0p25"] = {}
    if include_fawe_beta_grid:
        for method_name in _fawe_beta_grid_method_names(fawe_betas):
            base_methods[method_name] = {}
    for weight in answer_rrf_weights:
        suffix = _weight_suffix(weight)
        base_methods[f"weighted_dual_query_raw_expected_rrf_w{suffix}"] = {}
        base_methods[f"weighted_dual_query_masked_expected_rrf_w{suffix}"] = {}
        base_methods[f"weighted_rrf_query_answer_constrained_w{suffix}"] = {}
    if not fawe_betas and not include_fawe_beta_grid:
        for name in [
            "fawe_raw_expected_beta0p25",
            "fawe_masked_expected_beta0p25",
            "fawe_answer_constrained_beta0p5",
            "fawe_query2doc_beta0p25",
            "fawe_safe_adaptive_beta",
        ]:
            base_methods.pop(name, None)
    if not selected_methods:
        return base_methods
    return {name: base_methods[name] for name in selected_methods if name in base_methods}


def _select_methods(
    method_profile: str,
    answer_rrf_weights: list[float],
    fawe_betas: list[float],
    include_rm3_baseline: bool = False,
    include_fawe_controls: bool = False,
    include_fawe_beta_grid: bool = False,
) -> list[str]:
    all_methods = list(
        _initialize_runs(
            answer_rrf_weights,
            fawe_betas,
            selected_methods=[],
            include_rm3_baseline=include_rm3_baseline,
            include_fawe_controls=include_fawe_controls,
            include_fawe_beta_grid=include_fawe_beta_grid,
        ).keys()
    )
    if method_profile == "full":
        return all_methods
    keep = set(MAIN_METHODS)
    if include_rm3_baseline:
        keep.add("bm25_rm3_query_only")
    if include_fawe_controls:
        keep.update(
            {
                "fawe_shuffled_expected_beta0p25",
                "fawe_wrong_answer_beta0p25",
                "fawe_neutral_filler_beta0p25",
                "fawe_query_repeated_beta0p25",
                "fawe_random_terms_from_corpus_beta0p25",
                "fawe_idf_matched_random_terms_beta0p25",
            }
        )
    if include_fawe_beta_grid:
        keep.update(_fawe_beta_grid_method_names(fawe_betas))
    keep.update({"fawe_safe_adaptive_beta"} if fawe_betas else set())
    return [name for name in all_methods if name in keep]


def _repeat_query_to_match_text(query_text: str, reference_text: str) -> str:
    query_tokens = query_text.split()
    if not query_tokens:
        return query_text
    needed = max(len(reference_text.split()), 1)
    repeated = [query_tokens[index % len(query_tokens)] for index in range(needed)]
    return f"{query_text}\n{' '.join(repeated)}".strip()


def _shuffle_text_tokens(text: str, seed: int) -> str:
    tokens = text.split()
    if len(tokens) <= 1:
        return text
    rng = random.Random(seed)
    shuffled = list(tokens)
    rng.shuffle(shuffled)
    return " ".join(shuffled)


def _prepend_filler_to_query(query_text: str, reference_text: str) -> str:
    filler_tokens = max(len(reference_text.split()), 1)
    filler = " ".join(["relevant"] * filler_tokens)
    return f"{filler}\n{query_text}".strip()


def _resolve_named_fawe_betas(fawe_betas: list[float]) -> dict[str, float]:
    if not fawe_betas:
        fawe_betas = sorted(set(FAWE_DEFAULT_BETAS.values()))
    return {
        name: _closest_beta(fawe_betas, default_beta)
        for name, default_beta in FAWE_DEFAULT_BETAS.items()
    }


def _fawe_beta_grid_method_names(fawe_betas: list[float]) -> list[str]:
    betas = sorted(dict.fromkeys(fawe_betas or sorted(set(FAWE_DEFAULT_BETAS.values()))))
    families = [
        "raw_expected",
        "masked_expected",
        "answer_constrained",
        "query2doc",
    ]
    names: list[str] = []
    for family in families:
        for beta in betas:
            names.append(f"fawe_{family}_beta{_weight_suffix(beta)}")
    return names


def _closest_beta(betas: list[float], target: float) -> float:
    return min(betas, key=lambda value: (abs(value - target), value))


def _fielded_anchor_weighted_search(
    query_ranking: RankedList,
    expansion_ranking: RankedList,
    beta: float,
    top_k: int,
) -> RankedList:
    query_scores = dict(query_ranking)
    expansion_scores = dict(expansion_ranking)
    doc_ids = set(query_scores) | set(expansion_scores)
    combined = [
        (doc_id, float(query_scores.get(doc_id, 0.0)) + float(beta) * float(expansion_scores.get(doc_id, 0.0)))
        for doc_id in doc_ids
    ]
    combined.sort(key=lambda item: item[1], reverse=True)
    return combined[:top_k]


def _bm25_rm3_search(
    retriever,
    query_text: str,
    query_ranking: RankedList,
    top_k: int,
    fb_docs: int,
    fb_terms: int,
    original_query_weight: float,
) -> tuple[RankedList, str]:
    if not hasattr(retriever, "term_freqs") or not hasattr(retriever, "doc_len"):
        return query_ranking, query_text
    feedback_docs = query_ranking[: max(fb_docs, 1)]
    if not feedback_docs:
        return query_ranking, query_text
    max_score = max(score for _doc_id, score in feedback_docs)
    exp_weights = [math.exp(score - max_score) for _doc_id, score in feedback_docs]
    total_weight = sum(exp_weights) or 1.0
    normalized_doc_weights = [weight / total_weight for weight in exp_weights]
    expansion_scores: dict[str, float] = defaultdict(float)
    original_terms = Counter(tokenize(query_text))
    for (doc_id, _score), doc_weight in zip(feedback_docs, normalized_doc_weights, strict=False):
        term_freqs = retriever.term_freqs.get(doc_id, {})
        doc_len = max(retriever.doc_len.get(doc_id, 0), 1)
        for term, tf in term_freqs.items():
            if len(term) <= 2 or term in ROUTE_STOPWORDS:
                continue
            expansion_scores[term] += doc_weight * (tf / doc_len)
    ranked_terms = [
        (term, score)
        for term, score in sorted(expansion_scores.items(), key=lambda item: (-item[1], item[0]))
        if term not in original_terms
    ][: max(fb_terms, 1)]
    if not ranked_terms:
        return query_ranking, query_text
    max_term_score = ranked_terms[0][1] or 1.0
    expansion_terms: list[str] = []
    for term, score in ranked_terms:
        repeats = max(1, round((score / max_term_score) * 3))
        expansion_terms.extend([term] * repeats)
    query_repeat = max(1, round(original_query_weight * 4))
    expanded_query = " ".join(([query_text] * query_repeat) + expansion_terms).strip()
    return retriever.search(expanded_query, top_k=top_k), expanded_query


def _random_corpus_terms(
    retriever,
    doc_map: Mapping[str, Document],
    seed: int,
    count: int,
    exclude_terms: set[str],
) -> str:
    vocabulary = list(getattr(retriever, "vocabulary_terms", []))
    if not vocabulary:
        vocab_set: set[str] = set()
        for doc in doc_map.values():
            vocab_set.update(tokenize(doc.text))
        vocabulary = sorted(vocab_set)
    candidates = [term for term in vocabulary if len(term) > 2 and term not in ROUTE_STOPWORDS and term not in exclude_terms]
    if not candidates:
        return ""
    rng = random.Random(seed)
    if len(candidates) <= count:
        rng.shuffle(candidates)
        return " ".join(candidates)
    return " ".join(rng.sample(candidates, count))


def _idf_matched_random_terms(
    reference_text: str,
    retriever,
    doc_map: Mapping[str, Document],
    seed: int,
    count: int,
    exclude_terms: set[str],
) -> str:
    idf_by_term = getattr(retriever, "idf_by_term", {})
    if not idf_by_term:
        return _random_corpus_terms(retriever, doc_map, seed=seed, count=count, exclude_terms=exclude_terms)
    reference_terms = [term for term in tokenize(reference_text) if term in idf_by_term]
    if not reference_terms:
        return _random_corpus_terms(retriever, doc_map, seed=seed, count=count, exclude_terms=exclude_terms)
    rng = random.Random(seed)
    vocabulary = [term for term in getattr(retriever, "vocabulary_terms", []) if len(term) > 2 and term not in ROUTE_STOPWORDS and term not in exclude_terms]
    if not vocabulary:
        return ""
    chosen: list[str] = []
    used: set[str] = set()
    for index, term in enumerate(reference_terms[:count]):
        target = idf_by_term[term]
        nearby = sorted(
            vocabulary,
            key=lambda candidate: (abs(idf_by_term.get(candidate, 0.0) - target), candidate),
        )
        if not nearby:
            continue
        window = nearby[: min(25, len(nearby))]
        candidate = window[(rng.randint(0, len(window) - 1) + index) % len(window)]
        if candidate in used:
            continue
        used.add(candidate)
        chosen.append(candidate)
    if len(chosen) < count:
        fallback = _random_corpus_terms(retriever, doc_map, seed=seed + 997, count=count - len(chosen), exclude_terms=exclude_terms | set(chosen))
        if fallback:
            chosen.extend(fallback.split())
    return " ".join(chosen[:count])


def _format_fawe_retrieval_text(query_text: str, expansion_text: str, beta: float) -> str:
    return (
        f"FAWE(query_weight=1.0,beta={beta})\n"
        f"[QUERY]\n{query_text}\n"
        f"[EXPANSION]\n{expansion_text}"
    ).strip()


def _safe_adaptive_fawe_beta(
    features: Mapping[str, Mapping[str, float | int]],
    template_validation: Mapping[str, object],
) -> float:
    answer_constrained = features["concat_query_answer_candidate_constrained_template"]
    anchor_coverage = float(answer_constrained["anchor_coverage"])
    unsupported = int(answer_constrained["unsupported_entity_count"])
    support = float(answer_constrained["support"])
    if bool(template_validation.get("valid")) and unsupported == 0 and anchor_coverage >= 0.75 and support >= 0.45:
        return 0.5
    if bool(template_validation.get("has_required_keys")) and unsupported <= 1 and anchor_coverage >= 0.5:
        return 0.35
    return 0.2


def _build_retrieval_diagnostics(
    query_id: str,
    original_query_text: str,
    query_text: str,
    retrieval_strings: Mapping[str, str],
    rankings: Mapping[str, RankedList],
    qrels: Mapping[str, int],
    included_methods: Iterable[str],
) -> dict[str, dict[str, object]]:
    diagnostics: dict[str, dict[str, object]] = {}
    relevant_docids = [doc_id for doc_id, score in qrels.items() if score > 0]
    for method_name in included_methods:
        retrieval_text = str(retrieval_strings.get(method_name, ""))
        ranking = rankings.get(method_name, [])
        diagnostics[method_name] = {
            "query_id": query_id,
            "original_query": original_query_text,
            "counterfactual_query": query_text,
            "final_retrieval_text": retrieval_text,
            "retriever_input_hash": hashlib.sha256(retrieval_text.encode("utf-8")).hexdigest(),
            "top10_docids": [doc_id for doc_id, _score in ranking[:10]],
            "relevant_docids": relevant_docids,
        }
    return diagnostics


def _duplicate_method_audit(
    records: list[dict[str, object]],
    suspicious_pairs: list[tuple[str, str]],
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for left, right in suspicious_pairs:
        text_same = 0
        top10_same = 0
        shared = 0
        for record in records:
            retrieval_strings = record.get("retrieval_strings", {})
            rankings = record.get("rankings", {})
            retrieval_diagnostics = record.get("retrieval_diagnostics", {})
            if left not in retrieval_strings or right not in retrieval_strings:
                continue
            shared += 1
            if _normalize_text(str(retrieval_strings[left])) == _normalize_text(str(retrieval_strings[right])):
                text_same += 1
            left_top = [doc_id for doc_id, _score in rankings.get(left, [])[:10]]
            right_top = [doc_id for doc_id, _score in rankings.get(right, [])[:10]]
            if not left_top:
                left_top = list((retrieval_diagnostics.get(left, {}) or {}).get("top10_docids") or [])
            if not right_top:
                right_top = list((retrieval_diagnostics.get(right, {}) or {}).get("top10_docids") or [])
            if left_top == right_top:
                top10_same += 1
        summary[f"{left}__vs__{right}"] = {
            "shared_queries": shared,
            "identical_retrieval_text_rate": (text_same / shared) if shared else None,
            "identical_top10_rate": (top10_same / shared) if shared else None,
        }
    return summary


def _concatenation_rescue_summary(
    records: list[dict[str, object]],
    pairs: list[tuple[str, str]],
) -> dict[str, dict[str, float | None]]:
    summary: dict[str, dict[str, float | None]] = {}
    for generated_only, query_plus_generated in pairs:
        deltas = []
        for record in records:
            rankings = record.get("rankings", {})
            qrels = record.get("qrels", {})
            if generated_only not in rankings or query_plus_generated not in rankings:
                continue
            generated_score = per_query_metrics(rankings[generated_only], qrels).get("ndcg@10", 0.0)
            anchored_score = per_query_metrics(rankings[query_plus_generated], qrels).get("ndcg@10", 0.0)
            deltas.append(anchored_score - generated_score)
        summary[f"{generated_only}__to__{query_plus_generated}"] = {
            "avg_ndcg@10_delta": (sum(deltas) / len(deltas)) if deltas else None,
            "num_queries": len(deltas),
        }
    return summary


def _query_dominance_summary(records: list[dict[str, object]]) -> dict[str, float | None]:
    method_names = [
        "concat_query_wrong_answer",
        "query_repeated",
        "query_repeated_length_matched",
        "query_plus_shuffled_expected",
        "query_plus_neutral_filler",
        "neutral_filler_plus_query",
    ]
    summary: dict[str, float | None] = {}
    for method_name in method_names:
        deltas = []
        for record in records:
            rankings = record.get("rankings", {})
            qrels = record.get("qrels", {})
            if "query_only" not in rankings or method_name not in rankings:
                continue
            baseline = per_query_metrics(rankings["query_only"], qrels).get("ndcg@10", 0.0)
            method_score = per_query_metrics(rankings[method_name], qrels).get("ndcg@10", 0.0)
            deltas.append(method_score - baseline)
        summary[f"{method_name}_delta_vs_query_only_ndcg@10"] = (sum(deltas) / len(deltas)) if deltas else None
    return summary


def _compute_metric_error_bars(
    per_query_method_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    bootstrap_samples: int,
) -> dict[str, dict[str, dict[str, float]]]:
    error_bars: dict[str, dict[str, dict[str, float]]] = {}
    for method_name, query_metrics in per_query_method_metrics.items():
        metric_names = sorted({metric for metrics in query_metrics.values() for metric in metrics})
        method_error_bars: dict[str, dict[str, float]] = {}
        for metric_name in metric_names:
            values = [float(metrics[metric_name]) for metrics in query_metrics.values() if metric_name in metrics]
            if values:
                method_error_bars[metric_name] = paired_bootstrap_ci(values, num_samples=bootstrap_samples)
        error_bars[method_name] = method_error_bars
    return error_bars


def _sample_retrieval_audit(
    records: list[dict[str, object]],
    audit_methods: list[str],
    sample_size: int,
    seed: int,
) -> list[dict[str, object]]:
    if not records:
        return []
    rng = random.Random(seed)
    sampled_records = list(records)
    rng.shuffle(sampled_records)
    sampled_records = sampled_records[: min(sample_size, len(sampled_records))]
    audit_rows: list[dict[str, object]] = []
    for record in sampled_records:
        diagnostics = record.get("retrieval_diagnostics", {})
        generation = record.get("generation", {})
        for method_name in audit_methods:
            if method_name not in diagnostics:
                continue
            diag = diagnostics[method_name]
            audit_rows.append(
                {
                    "query_id": record.get("query_id"),
                    "original_query": diag.get("original_query"),
                    "counterfactual_query": diag.get("counterfactual_query"),
                    "method": method_name,
                    "generated_text": _generated_text_for_audit(method_name, generation),
                    "final_retrieval_text": diag.get("final_retrieval_text"),
                    "retriever_input_hash": diag.get("retriever_input_hash"),
                    "top10_docids": diag.get("top10_docids"),
                    "relevant_docids": diag.get("relevant_docids"),
                }
            )
    return audit_rows


def _generated_text_for_audit(method_name: str, generation: Mapping[str, object]) -> str | None:
    if method_name.startswith("fawe_raw_expected") or method_name in {
        "raw_expected_answer_only",
        "concat_query_raw_expected",
        "raw_expected_then_query",
        "query_plus_shuffled_expected",
    }:
        return str(generation.get("expected_answer") or "")
    if method_name.startswith("fawe_masked_expected") or method_name in {
        "masked_expected_answer_only",
        "concat_query_masked_expected",
    }:
        return str(generation.get("masked_expected_answer") or "")
    if method_name in {
        "query2doc_concat",
        "fawe_query2doc_beta0p25",
    }:
        return str(generation.get("query2doc_document") or "")
    if method_name in {
        "concat_query_answer_candidate_constrained_template",
        "fawe_answer_constrained_beta0p5",
        "fawe_safe_adaptive_beta",
    }:
        parsed = generation.get("answer_candidate_template_parsed", {})
        return str(parsed.get("retrieval_text") or generation.get("answer_candidate_template") or "")
    controls = generation.get("controls", {})
    if method_name in controls:
        return str(controls.get(method_name) or "")
    return None


def _dense_position_audit(records: list[dict[str, object]]) -> dict[str, float | None]:
    pairs = [
        ("concat_query_raw_expected", "raw_expected_then_query"),
        ("query_plus_neutral_filler", "neutral_filler_plus_query"),
    ]
    summary: dict[str, float | None] = {}
    for first, second in pairs:
        deltas = []
        for record in records:
            rankings = record.get("rankings", {})
            qrels = record.get("qrels", {})
            if first not in rankings or second not in rankings:
                continue
            first_score = per_query_metrics(rankings[first], qrels).get("ndcg@10", 0.0)
            second_score = per_query_metrics(rankings[second], qrels).get("ndcg@10", 0.0)
            deltas.append(first_score - second_score)
        summary[f"{first}__minus__{second}_avg_ndcg@10"] = (sum(deltas) / len(deltas)) if deltas else None
    return summary


def _build_corpus_steered_expansion(
    query_text: str,
    query_rank: RankedList,
    doc_map: Mapping[str, Document],
    max_docs: int = 3,
    max_words: int | None = None,
) -> str:
    snippets = []
    for doc_id, _score in query_rank[:max_docs]:
        doc = doc_map.get(doc_id)
        if not doc:
            continue
        sentence = doc.text.split(".")[0].strip()
        if sentence:
            snippets.append(sentence)
    text = "\n".join(snippets)
    if max_words is not None:
        text = " ".join(text.split()[:max_words]).strip()
    return text


def _build_cf_prompt_support_context(
    query_text: str,
    query_rank: RankedList,
    doc_map: Mapping[str, Document],
    max_docs: int = 3,
    max_terms: int = 8,
) -> dict[str, object]:
    snippets: list[str] = []
    term_counts: dict[str, int] = {}
    query_tokens = {
        token
        for token in tokenize(query_text)
        if len(token) > 2 and token not in ROUTE_STOPWORDS
    }
    query_anchors = {normalize_text(anchor) for anchor in extract_query_anchors(query_text)}
    for doc_id, _score in query_rank[:max_docs]:
        doc = doc_map.get(doc_id)
        if not doc:
            continue
        title = doc.title.strip()
        first_sentence = doc.text.split(".")[0].strip()
        snippet = " - ".join(part for part in [title, first_sentence] if part).strip()
        if snippet:
            snippets.append(snippet)
        source_text = f"{title} {first_sentence}".strip()
        for phrase in extract_query_anchors(source_text):
            normalized = normalize_text(phrase)
            if not normalized or normalized in query_anchors:
                continue
            term_counts[phrase] = term_counts.get(phrase, 0) + 2
        for token in tokenize(source_text):
            if len(token) <= 3 or token in ROUTE_STOPWORDS or token in query_tokens:
                continue
            term_counts[token] = term_counts.get(token, 0) + 1
    support_terms = [
        term
        for term, _count in sorted(
            term_counts.items(),
            key=lambda item: (-item[1], len(item[0]), item[0].lower()),
        )
    ][:max_terms]
    prompt_lines = []
    if support_terms:
        prompt_lines.append("support_terms: " + ", ".join(support_terms))
    if snippets:
        prompt_lines.append("support_snippets:")
        prompt_lines.extend(f"- {snippet}" for snippet in snippets[:max_docs])
    return {
        "support_terms": support_terms,
        "support_snippets": snippets[:max_docs],
        "prompt_context": "\n".join(prompt_lines).strip(),
    }


def _route_reliability_bundle(
    query_text: str,
    query_rank: RankedList,
    route_texts: Mapping[str, str],
    route_rankings: Mapping[str, RankedList],
    doc_map: Mapping[str, Document],
    template_validation: Mapping[str, object],
) -> dict[str, object]:
    features = {
        name: _route_reliability_features(query_text, route_text, query_rank, route_rankings[name], doc_map)
        for name, route_text in route_texts.items()
    }
    weights = _safe_rrf_v1_weights(features, template_validation)
    return {
        "features": features,
        "weights": weights,
    }


def _route_reliability_features(
    query_text: str,
    route_text: str,
    query_rank: RankedList,
    route_rank: RankedList,
    doc_map: Mapping[str, Document],
    top_docs: int = 5,
) -> dict[str, float | int]:
    support_docs = [doc_map[doc_id] for doc_id, _score in query_rank[:top_docs] if doc_id in doc_map]
    support_text = query_text + "\n" + "\n".join(doc.text for doc in support_docs)
    support_tokens = {token for token in tokenize(support_text) if len(token) > 2 and token not in ROUTE_STOPWORDS}
    route_tokens = [token for token in tokenize(route_text) if len(token) > 2 and token not in ROUTE_STOPWORDS]
    token_support = (
        sum(1 for token in route_tokens if token in support_tokens) / len(route_tokens)
        if route_tokens
        else 1.0
    )
    support_candidates = {normalize_text(value) for value in extract_concrete_candidates(support_text)}
    route_candidates = extract_concrete_candidates(route_text)
    candidate_support = (
        sum(1 for candidate in route_candidates if normalize_text(candidate) in support_candidates) / len(route_candidates)
        if route_candidates
        else 1.0
    )
    query_anchor_values = extract_query_anchors(query_text)
    query_anchor_norms = [normalize_text(anchor) for anchor in query_anchor_values if normalize_text(anchor)]
    anchor_coverage = (
        sum(1 for anchor in query_anchor_norms if anchor in normalize_text(route_text)) / len(query_anchor_norms)
        if query_anchor_norms
        else 1.0
    )
    unsupported_candidates = [
        candidate
        for candidate in route_candidates
        if normalize_text(candidate) not in support_candidates
        and normalize_text(candidate) not in query_anchor_norms
    ]
    route_top = {doc_id for doc_id, _score in route_rank[:10]}
    query_top = {doc_id for doc_id, _score in query_rank[:20]}
    route_agreement = len(route_top & query_top) / max(len(route_top), 1)
    return {
        "support": round((token_support + candidate_support) / 2.0, 6),
        "token_support": round(token_support, 6),
        "candidate_support": round(candidate_support, 6),
        "unsupported_entity_count": len(unsupported_candidates),
        "anchor_coverage": round(anchor_coverage, 6),
        "route_agreement": round(route_agreement, 6),
        "answer_form_penalty": round(_answer_form_penalty(route_text), 6),
    }


def _safe_rrf_v1_weights(
    features: Mapping[str, Mapping[str, float | int]],
    template_validation: Mapping[str, object],
) -> dict[str, float]:
    weights = {"query_only": 1.0}

    grf = features["generative_relevance_feedback_concat"]
    grf_support = float(grf["support"])
    grf_unsupported = int(grf["unsupported_entity_count"])
    grf_agreement = float(grf["route_agreement"])
    grf_penalty = float(grf["answer_form_penalty"])
    if grf_support >= 0.45 and grf_unsupported <= 1 and grf_agreement >= 0.15 and grf_penalty < 0.7:
        weights["generative_relevance_feedback_concat"] = 0.8
    elif grf_support >= 0.25 and grf_unsupported <= 2 and grf_agreement >= 0.05:
        weights["generative_relevance_feedback_concat"] = 0.5
    else:
        weights["generative_relevance_feedback_concat"] = 0.25

    query2doc = features["query2doc_concat"]
    q2d_support = float(query2doc["support"])
    q2d_unsupported = int(query2doc["unsupported_entity_count"])
    q2d_agreement = float(query2doc["route_agreement"])
    q2d_penalty = float(query2doc["answer_form_penalty"])
    if q2d_support >= 0.45 and q2d_unsupported <= 1 and q2d_agreement >= 0.10 and q2d_penalty < 0.7:
        weights["query2doc_concat"] = 0.55
    elif q2d_support >= 0.25 and q2d_unsupported <= 2:
        weights["query2doc_concat"] = 0.35
    else:
        weights["query2doc_concat"] = 0.2

    answer_constrained = features["concat_query_answer_candidate_constrained_template"]
    ac_anchor = float(answer_constrained["anchor_coverage"])
    ac_unsupported = int(answer_constrained["unsupported_entity_count"])
    if bool(template_validation.get("valid")) and ac_unsupported == 0 and ac_anchor >= 0.75:
        weights["concat_query_answer_candidate_constrained_template"] = 0.55
    elif bool(template_validation.get("has_required_keys")) and ac_unsupported <= 1 and ac_anchor >= 0.5:
        weights["concat_query_answer_candidate_constrained_template"] = 0.4
    else:
        weights["concat_query_answer_candidate_constrained_template"] = 0.25

    return weights


def _cf_prompt_subquery_bundle(
    query_text: str,
    query_rank: RankedList,
    cf_prompt_payload: Mapping[str, object],
    retriever,
    doc_map: Mapping[str, Document],
    top_k: int,
    support_terms: list[str],
    max_selected: int = 3,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    role_map = []
    role_map.extend(("relation", query) for query in cf_prompt_payload.get("relation_queries") or [])
    role_map.extend(("evidence", query) for query in cf_prompt_payload.get("evidence_queries") or [])
    role_map.append(("answer_slot", str(cf_prompt_payload.get("answer_slot_query") or "").strip()))
    role_map.append(("bridge", str(cf_prompt_payload.get("bridge_query") or "").strip()))
    for role, candidate_query in role_map:
        candidate_text = str(candidate_query or "").strip()
        if not candidate_text:
            continue
        ranking = retriever.search(candidate_text, top_k=top_k)
        features = _route_reliability_features(query_text, candidate_text, query_rank, ranking, doc_map)
        weight, keep, reasons = _cf_prompt_subquery_weight(
            role,
            candidate_text,
            query_text,
            features,
            support_terms=support_terms,
        )
        candidates.append(
            {
                "role": role,
                "query": candidate_text,
                "features": features,
                "contains_support_term": _contains_support_term(candidate_text, support_terms),
                "weight": round(weight, 6),
                "selected": keep,
                "selection_reasons": reasons,
                "ranking": ranking,
            }
        )
    selected = [candidate for candidate in candidates if candidate["selected"] and candidate["weight"] > 0]
    deduped: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    role_caps = {"relation": 2, "evidence": 1, "bridge": 1, "answer_slot": 1}
    role_counts: dict[str, int] = {}
    selected.sort(
        key=lambda item: (
            -float(item["weight"]),
            -int(bool(item["contains_support_term"])),
            -float(item["features"]["route_agreement"]),
            item["query"],
        )
    )
    for candidate in selected:
        key = _cf_query_shape_key(candidate["query"])
        if key in seen_keys:
            continue
        role = str(candidate["role"])
        if role_counts.get(role, 0) >= role_caps.get(role, 1):
            continue
        role_counts[role] = role_counts.get(role, 0) + 1
        seen_keys.add(key)
        deduped.append(candidate)
    selected = deduped
    if not selected and candidates:
        fallback = max(
            candidates,
            key=lambda item: (
                int(bool(item["contains_support_term"])),
                float(item["weight"]),
                float(item["features"]["support"]),
                float(item["features"]["route_agreement"]),
            ),
        )
        fallback = dict(fallback)
        fallback["selected"] = True
        fallback["selection_reasons"] = list(fallback["selection_reasons"]) + ["fallback_best_candidate"]
        selected = [fallback]
    selected = selected[:max_selected]
    rankings = [candidate["ranking"] for candidate in selected]
    weights = [float(candidate["weight"]) for candidate in selected]
    return {
        "candidates": [
            {
                key: value
                for key, value in candidate.items()
                if key != "ranking"
            }
            for candidate in candidates
        ],
        "selected_queries": [candidate["query"] for candidate in selected],
        "selected_roles": [candidate["role"] for candidate in selected],
        "selected_weights": weights,
        "rankings": rankings,
    }


def _cf_prompt_subquery_weight(
    role: str,
    candidate_query: str,
    query_text: str,
    features: Mapping[str, float | int],
    support_terms: list[str],
) -> tuple[float, bool, list[str]]:
    reasons: list[str] = []
    if _normalize_text(candidate_query) == _normalize_text(query_text):
        return 0.0, False, ["duplicate_query_only"]
    base = CF_PROMPT_BASE_WEIGHTS.get(role, CF_PROMPT_BASE_WEIGHTS["generic"])
    support = float(features["support"])
    anchor_coverage = float(features["anchor_coverage"])
    agreement = float(features["route_agreement"])
    unsupported = int(features["unsupported_entity_count"])
    answer_penalty = float(features["answer_form_penalty"])
    has_support_term = _contains_support_term(candidate_query, support_terms)
    generic_penalty = _cf_prompt_generic_penalty(candidate_query)
    weight = base
    if unsupported > 1:
        return 0.0, False, ["unsupported_candidates"]
    if anchor_coverage < 0.34:
        return 0.0, False, ["anchor_dropout"]
    if answer_penalty >= 0.8:
        return 0.0, False, ["answer_form_penalty"]
    if support_terms and role != "answer_slot" and not has_support_term:
        weight -= 0.2
        reasons.append("missing_support_term")
    if generic_penalty:
        weight -= generic_penalty
        reasons.append("generic_phrase_penalty")
    if support >= 0.45:
        weight += 0.1
        reasons.append("supported_by_query_or_docs")
    elif support < 0.2:
        weight -= 0.1
        reasons.append("low_support")
    if agreement >= 0.15:
        weight += 0.1
        reasons.append("agrees_with_query_only")
    elif agreement < 0.05:
        weight -= 0.05
        reasons.append("low_route_agreement")
    if anchor_coverage >= 0.75:
        weight += 0.05
        reasons.append("strong_anchor_preservation")
    if unsupported == 0:
        weight += 0.05
        reasons.append("no_unsupported_candidates")
    keep = weight >= 0.2 and (has_support_term or role == "answer_slot" or not support_terms)
    if not keep and not reasons:
        reasons.append("weight_below_threshold")
    return max(round(weight, 6), 0.0), keep, reasons


def _contains_support_term(text: str, support_terms: list[str]) -> bool:
    normalized_text = _normalize_text(text)
    return any(_normalize_text(term) and _normalize_text(term) in normalized_text for term in support_terms)


def _cf_prompt_generic_penalty(text: str) -> float:
    lowered = _normalize_text(text)
    for pattern in CF_PROMPT_GENERIC_PATTERNS:
        if pattern in lowered:
            return 0.15
    return 0.0


def _cf_query_shape_key(text: str) -> str:
    tokens = [token for token in tokenize(text) if token not in ROUTE_STOPWORDS]
    return " ".join(sorted(dict.fromkeys(tokens)))


def _answer_form_penalty(text: str) -> float:
    lowered = text.strip().lower()
    candidates = extract_concrete_candidates(text)
    token_count = len(tokenize(text))
    if lowered.startswith("the answer is"):
        return 1.0
    if token_count <= 6 and candidates:
        return 0.8
    if candidates and token_count <= 12:
        return 0.6
    return 0.0


def _wrong_answer_candidate(dataset, index: int) -> str:
    other_answers = []
    for offset in range(1, len(dataset.queries)):
        other = dataset.queries[(index + offset) % len(dataset.queries)]
        if other.answers:
            other_answers = list(other.answers)
            break
    return other_answers[0] if other_answers else "Wrong Candidate"


def _precompute_generation_bundles(
    args: argparse.Namespace,
    dataset_name: str,
    queries,
    cache: JsonCache,
    cache_namespace: str,
    cache_namespace_aliases: list[str],
    support_contexts: Mapping[str, str],
) -> dict[str, dict[str, object]]:
    max_workers = max(int(args.generation_workers or 1), 1)
    if max_workers == 1:
        return {
            query.query_id: _generate_query_bundle(
                args=args,
                dataset_name=dataset_name,
                query_text=query.text,
                support_context=str(support_contexts.get(query.query_id, "")),
                cache=cache,
                cache_namespace=cache_namespace,
                cache_namespace_aliases=cache_namespace_aliases,
            )
            for query in tqdm(queries, desc="Precomputing generations")
        }
    results: dict[str, dict[str, object]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _generate_query_bundle,
                args,
                dataset_name,
                query.text,
                str(support_contexts.get(query.query_id, "")),
                cache,
                cache_namespace,
                cache_namespace_aliases,
            ): query.query_id
            for query in queries
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_map),
            total=len(future_map),
            desc="Precomputing generations",
        ):
            query_id = future_map[future]
            results[query_id] = future.result()
    return results


def _generate_query_bundle(
    args: argparse.Namespace,
    dataset_name: str,
    query_text: str,
    support_context: str,
    cache: JsonCache,
    cache_namespace: str,
    cache_namespace_aliases: list[str],
) -> dict[str, object]:
    local_generator = CachedTextGenerator(
        inner=_make_base_generator(args),
        cache=cache,
        namespace=cache_namespace,
        namespace_aliases=cache_namespace_aliases,
    )
    expected = local_generator.expected_answer(query_text)
    expected_artifact = dict(local_generator.last_artifact() or {})
    masked = local_generator.mask_answer(query_text, expected)
    masked_artifact = dict(local_generator.last_artifact() or {})
    hyde_doc = local_generator.hyde_document(query_text)
    hyde_artifact = dict(local_generator.last_artifact() or {})
    query2doc_doc = local_generator.query2doc_document(query_text)
    query2doc_artifact = dict(local_generator.last_artifact() or {})
    relevance_feedback = local_generator.relevance_feedback(query_text)
    feedback_artifact = dict(local_generator.last_artifact() or {})
    template_text = local_generator.answer_candidate_template(query_text)
    template_artifact = dict(local_generator.last_artifact() or {})
    cf_prompt_text = local_generator.counterfactual_prompt_query_expansion(query_text, support_context)
    cf_prompt_artifact = dict(local_generator.last_artifact() or {})
    return {
        "expected_answer": expected,
        "expected_answer_artifact": expected_artifact,
        "masked_expected_answer": masked,
        "masked_expected_answer_artifact": masked_artifact,
        "hyde_document": hyde_doc,
        "hyde_document_artifact": hyde_artifact,
        "query2doc_document": query2doc_doc,
        "query2doc_document_artifact": query2doc_artifact,
        "generative_relevance_feedback": relevance_feedback,
        "generative_relevance_feedback_artifact": feedback_artifact,
        "answer_candidate_template": template_text,
        "answer_candidate_template_artifact": template_artifact,
        "cf_prompt_query_expansion": cf_prompt_text,
        "cf_prompt_query_expansion_artifact": cf_prompt_artifact,
    }


def _process_query(
    args: argparse.Namespace,
    dataset,
    query,
    query_index: int,
    query_context: Mapping[str, object],
    generation_bundle: Mapping[str, object],
    retriever,
    doc_map: Mapping[str, Document],
    answer_rrf_weights: list[float],
    fawe_betas: list[float],
    run_method_names: Iterable[str],
    checkpoint_context: Mapping[str, object],
) -> dict[str, object]:
    selected_run_methods = set(run_method_names)
    relevant_docs = [doc_map[doc_id] for doc_id in dataset.qrels.get(query.query_id, {}) if doc_id in doc_map]
    query_rank = query_context["query_rank"]
    cf_prompt_support = query_context["cf_prompt_support"]
    expected = str(generation_bundle["expected_answer"])
    expected_artifact = generation_bundle["expected_answer_artifact"]
    masked = str(generation_bundle["masked_expected_answer"])
    masked_artifact = generation_bundle["masked_expected_answer_artifact"]
    hyde_doc = str(generation_bundle["hyde_document"])
    hyde_artifact = generation_bundle["hyde_document_artifact"]
    query2doc_doc = str(generation_bundle["query2doc_document"])
    query2doc_artifact = generation_bundle["query2doc_document_artifact"]
    relevance_feedback = str(generation_bundle["generative_relevance_feedback"])
    feedback_artifact = generation_bundle["generative_relevance_feedback_artifact"]
    template_text = str(generation_bundle["answer_candidate_template"])
    template_artifact = generation_bundle["answer_candidate_template_artifact"]
    template_payload = parse_answer_candidate_template(template_text, query.text)
    template_validation = validate_answer_candidate_template(query.text, template_payload)
    template_retrieval_text = str(template_payload.get("retrieval_text") or query.text)
    cf_prompt_text = str(generation_bundle["cf_prompt_query_expansion"])
    cf_prompt_artifact = generation_bundle["cf_prompt_query_expansion_artifact"]
    cf_prompt_payload = parse_counterfactual_prompt_query_expansion(
        cf_prompt_text,
        query.text,
        build_counterfactual_prompt_payload(query.text),
    )
    cf_prompt_queries = list(cf_prompt_payload.get("queries") or [])
    cf_prompt_joined = "\n".join(cf_prompt_queries).strip()

    oracle_masked = remove_gold_from_text(expected, list(query.all_answer_strings))
    posthoc_gold_removed = remove_gold_from_text(expected, list(query.all_answer_strings))
    random_masked = random_span_mask_answer(expected, query.text, seed=args.counterfactual_seed + query_index)
    entity_only_masked = entity_only_mask_answer(expected, query.text)
    generic_masked = generic_mask_answer(masked)
    length_matched_filler = length_matched_neutral_filler(query.text, expected)
    wrong_answer_candidate = _wrong_answer_candidate(dataset, query_index)
    wrong_answer_only = wrong_answer_candidate
    wrong_answer_injection = f"{query.text}\n{wrong_answer_candidate}".strip()
    query_repeated = f"{query.text}\n{query.text}".strip()
    query_repeated_length_matched = _repeat_query_to_match_text(query.text, expected)
    shuffled_expected = _shuffle_text_tokens(expected, seed=args.audit_seed + query_index)
    query_plus_shuffled_expected = f"{query.text}\n{shuffled_expected}".strip()
    query_plus_neutral_filler = length_matched_filler
    neutral_filler_plus_query = _prepend_filler_to_query(query.text, expected)
    raw_expected_then_query = f"{expected}\n{query.text}".strip()
    exclude_terms = set(tokenize(query.text))
    random_corpus_terms = _random_corpus_terms(
        retriever,
        doc_map,
        seed=args.audit_seed + query_index,
        count=max(len(tokenize(expected)), 4),
        exclude_terms=exclude_terms,
    )
    idf_matched_random_terms = _idf_matched_random_terms(
        expected,
        retriever,
        doc_map,
        seed=args.audit_seed + query_index + 101,
        count=max(len(tokenize(expected)), 4),
        exclude_terms=exclude_terms,
    )
    corpus_steered_text = str(query_context["corpus_steered_text"])
    corpus_steered_short_text = str(query_context["corpus_steered_short_text"])

    method_queries = {
        "hyde_doc_only": hyde_doc,
        "query2doc_concat": f"{query.text}\n{query2doc_doc}",
        "generative_relevance_feedback_concat": f"{query.text}\n{relevance_feedback}",
        "corpus_steered_expansion_concat": f"{query.text}\n{corpus_steered_text}",
        "corpus_steered_short_concat": f"{query.text}\n{corpus_steered_short_text}",
        "raw_expected_answer_only": expected,
        "concat_query_raw_expected": f"{query.text}\n{expected}",
        "masked_expected_answer_only": masked,
        "concat_query_masked_expected": f"{query.text}\n{masked}",
        "answer_candidate_constrained_template_only": template_retrieval_text,
        "concat_query_answer_candidate_constrained_template": f"{query.text}\n{template_retrieval_text}",
        "gold_answer_only": " ".join(query.answers) if query.answers else expected,
        "oracle_answer_masked": oracle_masked,
        "concat_query_oracle_answer_masked": f"{query.text}\n{oracle_masked}",
        "post_hoc_gold_removed_expected_answer": posthoc_gold_removed,
        "concat_query_post_hoc_gold_removed_expected": f"{query.text}\n{posthoc_gold_removed}",
        "random_span_masking": random_masked,
        "concat_query_random_span_masking": f"{query.text}\n{random_masked}",
        "entity_only_masking": entity_only_masked,
        "concat_query_entity_only_masking": f"{query.text}\n{entity_only_masked}",
        "generic_mask_slot": generic_masked,
        "concat_query_generic_mask_slot": f"{query.text}\n{generic_masked}",
        "length_matched_neutral_filler": length_matched_filler,
        "wrong_answer_only": wrong_answer_only,
        "concat_query_wrong_answer": wrong_answer_injection,
        "wrong_answer_injection": wrong_answer_injection,
        "query_repeated": query_repeated,
        "query_repeated_length_matched": query_repeated_length_matched,
        "query_plus_shuffled_expected": query_plus_shuffled_expected,
        "query_plus_neutral_filler": query_plus_neutral_filler,
        "neutral_filler_plus_query": neutral_filler_plus_query,
        "raw_expected_then_query": raw_expected_then_query,
    }
    method_queries["query_only"] = query.text

    search_cache: dict[tuple[str, int], RankedList] = {(query.text, args.top_k): query_rank}

    def search_cached(text: str, top_k: int) -> RankedList:
        key = (text, top_k)
        ranking = search_cache.get(key)
        if ranking is None:
            ranking = retriever.search(text, top_k=top_k)
            search_cache[key] = ranking
        return ranking

    needed_single_query_methods = _required_single_query_methods(selected_run_methods)
    rankings = {"query_only": query_rank}
    for method_name in needed_single_query_methods:
        if method_name == "query_only":
            continue
        rankings[method_name] = search_cached(method_queries[method_name], args.top_k)
    rm3_retrieval_text = query.text
    if "bm25_rm3_query_only" in selected_run_methods:
        rankings["bm25_rm3_query_only"], rm3_retrieval_text = _bm25_rm3_search(
            retriever=retriever,
            query_text=query.text,
            query_ranking=query_rank,
            top_k=args.top_k,
            fb_docs=args.rm3_fb_docs,
            fb_terms=args.rm3_fb_terms,
            original_query_weight=args.rm3_original_query_weight,
        )

    needs_cf_prompt = "cf_prompt_query_expansion_rrf" in selected_run_methods
    if needs_cf_prompt:
        cf_prompt_bundle = _cf_prompt_subquery_bundle(
            query_text=query.text,
            query_rank=query_rank,
            cf_prompt_payload=cf_prompt_payload,
            retriever=retriever,
            doc_map=doc_map,
            top_k=args.top_k,
            support_terms=cf_prompt_support["support_terms"],
        )
        cf_prompt_selected_queries = list(cf_prompt_bundle["selected_queries"])
        cf_prompt_selected_weights = list(cf_prompt_bundle["selected_weights"])
        rankings["cf_prompt_query_expansion_rrf"] = weighted_reciprocal_rank_fusion(
            [query_rank, *cf_prompt_bundle["rankings"]] if cf_prompt_bundle["rankings"] else [query_rank],
            [1.0, *cf_prompt_selected_weights] if cf_prompt_bundle["rankings"] else [1.0],
            top_k=args.top_k,
        )
    else:
        cf_prompt_bundle = {"selected_queries": [], "selected_roles": [], "selected_weights": [], "candidates": [], "rankings": []}
        cf_prompt_selected_queries = []
        cf_prompt_selected_weights = []
    cf_prompt_selected_joined = "\n".join(cf_prompt_selected_queries).strip()

    if "dual_query_raw_expected_rrf" in selected_run_methods:
        rankings["dual_query_raw_expected_rrf"] = reciprocal_rank_fusion(
            [query_rank, rankings["raw_expected_answer_only"]],
            top_k=args.top_k,
        )
    if "dual_query_masked_expected_rrf" in selected_run_methods or "rrf_query_masked_expected" in selected_run_methods:
        rankings["dual_query_masked_expected_rrf"] = reciprocal_rank_fusion(
            [query_rank, rankings["masked_expected_answer_only"]],
            top_k=args.top_k,
        )
        if "rrf_query_masked_expected" in selected_run_methods:
            rankings["rrf_query_masked_expected"] = rankings["dual_query_masked_expected_rrf"]
    if (
        "dual_query_answer_candidate_constrained_template_rrf" in selected_run_methods
        or "rrf_query_answer_constrained" in selected_run_methods
    ):
        rankings["dual_query_answer_candidate_constrained_template_rrf"] = reciprocal_rank_fusion(
            [query_rank, rankings["answer_candidate_constrained_template_only"]],
            top_k=args.top_k,
        )
        if "rrf_query_answer_constrained" in selected_run_methods:
            rankings["rrf_query_answer_constrained"] = rankings["dual_query_answer_candidate_constrained_template_rrf"]
    if "rrf_query_wrong_answer" in selected_run_methods:
        rankings["rrf_query_wrong_answer"] = reciprocal_rank_fusion(
            [query_rank, rankings["wrong_answer_only"]],
            top_k=args.top_k,
        )
    if "rrf_query_corpus_steered_short" in selected_run_methods:
        rankings["rrf_query_corpus_steered_short"] = reciprocal_rank_fusion(
            [query_rank, rankings["corpus_steered_short_concat"]],
            top_k=args.top_k,
        )
    if "safe_rrf_v0" in selected_run_methods:
        rankings["safe_rrf_v0"] = weighted_reciprocal_rank_fusion(
            [
                query_rank,
                rankings["generative_relevance_feedback_concat"],
                rankings["query2doc_concat"],
                rankings["concat_query_answer_candidate_constrained_template"],
            ],
            [
                SAFE_RRF_V0_WEIGHTS["query_only"],
                SAFE_RRF_V0_WEIGHTS["generative_relevance_feedback_concat"],
                SAFE_RRF_V0_WEIGHTS["query2doc_concat"],
                SAFE_RRF_V0_WEIGHTS["concat_query_answer_candidate_constrained_template"],
            ],
            top_k=args.top_k,
        )
    for weight in answer_rrf_weights:
        suffix = _weight_suffix(weight)
        raw_rrf_name = f"weighted_dual_query_raw_expected_rrf_w{suffix}"
        masked_rrf_name = f"weighted_dual_query_masked_expected_rrf_w{suffix}"
        answer_rrf_name = f"weighted_rrf_query_answer_constrained_w{suffix}"
        if raw_rrf_name in selected_run_methods:
            rankings[raw_rrf_name] = weighted_reciprocal_rank_fusion(
                [query_rank, rankings["raw_expected_answer_only"]],
                [1.0, weight],
                top_k=args.top_k,
            )
        if masked_rrf_name in selected_run_methods:
            rankings[masked_rrf_name] = weighted_reciprocal_rank_fusion(
                [query_rank, rankings["masked_expected_answer_only"]],
                [1.0, weight],
                top_k=args.top_k,
            )
        if answer_rrf_name in selected_run_methods:
            rankings[answer_rrf_name] = weighted_reciprocal_rank_fusion(
                [query_rank, rankings["answer_candidate_constrained_template_only"]],
                [1.0, weight],
                top_k=args.top_k,
            )
    named_fawe_betas = _resolve_named_fawe_betas(fawe_betas)
    needs_route_reliability = bool({"safe_rrf_v1", "fawe_safe_adaptive_beta"} & selected_run_methods)
    route_reliability = {
        "weights": dict(SAFE_RRF_V0_WEIGHTS),
        "features": {},
    }
    safe_rrf_v1_weights = dict(SAFE_RRF_V0_WEIGHTS)
    fawe_safe_beta = 0.2
    if needs_route_reliability:
        route_reliability = _route_reliability_bundle(
            query_text=query.text,
            query_rank=query_rank,
            route_texts={
                "generative_relevance_feedback_concat": relevance_feedback,
                "query2doc_concat": query2doc_doc,
                "concat_query_answer_candidate_constrained_template": template_retrieval_text,
            },
            route_rankings={
                "generative_relevance_feedback_concat": rankings["generative_relevance_feedback_concat"],
                "query2doc_concat": rankings["query2doc_concat"],
                "concat_query_answer_candidate_constrained_template": rankings["concat_query_answer_candidate_constrained_template"],
            },
            doc_map=doc_map,
            template_validation=template_validation,
        )
        safe_rrf_v1_weights = route_reliability["weights"]
        fawe_safe_beta = _safe_adaptive_fawe_beta(route_reliability["features"], template_validation)
    needs_fawe = bool(
        {
            "fawe_raw_expected_beta0p25",
            "fawe_masked_expected_beta0p25",
            "fawe_answer_constrained_beta0p5",
            "fawe_query2doc_beta0p25",
            "fawe_safe_adaptive_beta",
        }
        & selected_run_methods
    )
    if needs_fawe:
        full_query_ranking = search_cached(query.text, len(dataset.corpus))
        fawe_expansion_rankings: dict[str, RankedList] = {}

        def fawe_expansion_ranking(expansion_text: str) -> RankedList:
            ranking = fawe_expansion_rankings.get(expansion_text)
            if ranking is None:
                ranking = search_cached(expansion_text, len(dataset.corpus))
                fawe_expansion_rankings[expansion_text] = ranking
            return ranking

        if "fawe_raw_expected_beta0p25" in selected_run_methods:
            rankings["fawe_raw_expected_beta0p25"] = _fielded_anchor_weighted_search(
                query_ranking=full_query_ranking,
                expansion_ranking=fawe_expansion_ranking(expected),
                beta=named_fawe_betas["raw_expected"],
                top_k=args.top_k,
            )
        if "fawe_masked_expected_beta0p25" in selected_run_methods:
            rankings["fawe_masked_expected_beta0p25"] = _fielded_anchor_weighted_search(
                query_ranking=full_query_ranking,
                expansion_ranking=fawe_expansion_ranking(masked),
                beta=named_fawe_betas["masked_expected"],
                top_k=args.top_k,
            )
        if "fawe_answer_constrained_beta0p5" in selected_run_methods:
            rankings["fawe_answer_constrained_beta0p5"] = _fielded_anchor_weighted_search(
                query_ranking=full_query_ranking,
                expansion_ranking=fawe_expansion_ranking(template_retrieval_text),
                beta=named_fawe_betas["answer_constrained"],
                top_k=args.top_k,
            )
        if "fawe_query2doc_beta0p25" in selected_run_methods:
            rankings["fawe_query2doc_beta0p25"] = _fielded_anchor_weighted_search(
                query_ranking=full_query_ranking,
                expansion_ranking=fawe_expansion_ranking(query2doc_doc),
                beta=named_fawe_betas["query2doc"],
                top_k=args.top_k,
            )
        if "fawe_safe_adaptive_beta" in selected_run_methods:
            rankings["fawe_safe_adaptive_beta"] = _fielded_anchor_weighted_search(
                query_ranking=full_query_ranking,
                expansion_ranking=fawe_expansion_ranking(template_retrieval_text),
                beta=fawe_safe_beta,
                top_k=args.top_k,
            )
        fawe_control_expansions = {
            "fawe_shuffled_expected_beta0p25": shuffled_expected,
            "fawe_wrong_answer_beta0p25": wrong_answer_only,
            "fawe_neutral_filler_beta0p25": length_matched_filler,
            "fawe_query_repeated_beta0p25": query.text,
            "fawe_random_terms_from_corpus_beta0p25": random_corpus_terms,
            "fawe_idf_matched_random_terms_beta0p25": idf_matched_random_terms,
        }
        for method_name, expansion_text in fawe_control_expansions.items():
            if method_name not in selected_run_methods or not expansion_text.strip():
                continue
            rankings[method_name] = _fielded_anchor_weighted_search(
                query_ranking=full_query_ranking,
                expansion_ranking=fawe_expansion_ranking(expansion_text),
                beta=0.25,
                top_k=args.top_k,
            )
        if args.include_fawe_beta_grid:
            beta_grid_expansions = {
                "raw_expected": expected,
                "masked_expected": masked,
                "answer_constrained": template_retrieval_text,
                "query2doc": query2doc_doc,
            }
            for family, expansion_text in beta_grid_expansions.items():
                for beta in sorted(dict.fromkeys(fawe_betas or sorted(set(FAWE_DEFAULT_BETAS.values())))):
                    method_name = f"fawe_{family}_beta{_weight_suffix(beta)}"
                    if method_name not in selected_run_methods:
                        continue
                    rankings[method_name] = _fielded_anchor_weighted_search(
                        query_ranking=full_query_ranking,
                        expansion_ranking=fawe_expansion_ranking(expansion_text),
                        beta=beta,
                        top_k=args.top_k,
                    )

    base_features = generation_features(query, expected, masked, hyde_doc)
    generated_texts_for_leakage = {
        "bm25_rm3_query_only": rm3_retrieval_text,
        "raw_expected_answer_only": expected,
        "concat_query_raw_expected": f"{query.text}\n{expected}",
        "masked_expected_answer_only": masked,
        "concat_query_masked_expected": f"{query.text}\n{masked}",
        "hyde_doc_only": hyde_doc,
        "query2doc_concat": query2doc_doc,
        "generative_relevance_feedback_concat": relevance_feedback,
        "corpus_steered_expansion_concat": corpus_steered_text,
        "corpus_steered_short_concat": corpus_steered_short_text,
        "answer_candidate_constrained_template_only": template_retrieval_text,
        "concat_query_answer_candidate_constrained_template": f"{query.text}\n{template_retrieval_text}",
        "oracle_answer_masked": oracle_masked,
        "concat_query_oracle_answer_masked": f"{query.text}\n{oracle_masked}",
        "post_hoc_gold_removed_expected_answer": posthoc_gold_removed,
        "concat_query_post_hoc_gold_removed_expected": f"{query.text}\n{posthoc_gold_removed}",
        "random_span_masking": random_masked,
        "concat_query_random_span_masking": f"{query.text}\n{random_masked}",
        "entity_only_masking": entity_only_masked,
        "concat_query_entity_only_masking": f"{query.text}\n{entity_only_masked}",
        "generic_mask_slot": generic_masked,
        "concat_query_generic_mask_slot": f"{query.text}\n{generic_masked}",
        "length_matched_neutral_filler": length_matched_filler,
        "wrong_answer_only": wrong_answer_only,
        "concat_query_wrong_answer": wrong_answer_injection,
        "wrong_answer_injection": wrong_answer_injection,
        "query_repeated": query_repeated,
        "query_repeated_length_matched": query_repeated_length_matched,
        "query_plus_shuffled_expected": query_plus_shuffled_expected,
        "query_plus_neutral_filler": query_plus_neutral_filler,
        "neutral_filler_plus_query": neutral_filler_plus_query,
        "raw_expected_then_query": raw_expected_then_query,
        "fawe_raw_expected_beta0p25": expected,
        "fawe_masked_expected_beta0p25": masked,
        "fawe_answer_constrained_beta0p5": template_retrieval_text,
        "fawe_query2doc_beta0p25": query2doc_doc,
        "fawe_safe_adaptive_beta": template_retrieval_text,
        "fawe_shuffled_expected_beta0p25": shuffled_expected,
        "fawe_wrong_answer_beta0p25": wrong_answer_only,
        "fawe_neutral_filler_beta0p25": length_matched_filler,
        "fawe_query_repeated_beta0p25": query.text,
        "fawe_random_terms_from_corpus_beta0p25": random_corpus_terms,
        "fawe_idf_matched_random_terms_beta0p25": idf_matched_random_terms,
        "cf_prompt_query_expansion_rrf": cf_prompt_selected_joined or cf_prompt_joined,
    }
    leakage_scores = score_generation_methods(query, generated_texts_for_leakage, relevant_docs)
    leakage_labels = {
        method_name: {
            "bucket": leakage_bucket_name(score),
            "is_leakage_positive": leakage_bucket_name(score) != "leakage_negative",
            "has_exact_answer_leakage": bool(score.get("exact_answer_leakage")),
            "has_alias_answer_leakage": bool(score.get("alias_answer_leakage")),
            "has_candidate_injection": bool(score.get("answer_candidate_leakage")),
            "has_wrong_prior_injection": bool(score.get("wrong_prior_candidates")),
            "has_unsupported_injection": bool(score.get("unsupported_candidates")),
        }
        for method_name, score in leakage_scores.items()
    }
    features_entry = {
        **base_features,
        "leakage_bucket": leakage_bucket_name(leakage_scores["raw_expected_answer_only"]),
        "raw_expected_answer_only": leakage_scores["raw_expected_answer_only"],
    }
    if "safe_rrf_v1" in selected_run_methods:
        rankings["safe_rrf_v1"] = weighted_reciprocal_rank_fusion(
            [
                query_rank,
                rankings["generative_relevance_feedback_concat"],
                rankings["query2doc_concat"],
                rankings["concat_query_answer_candidate_constrained_template"],
            ],
            [
                safe_rrf_v1_weights["query_only"],
                safe_rrf_v1_weights["generative_relevance_feedback_concat"],
                safe_rrf_v1_weights["query2doc_concat"],
                safe_rrf_v1_weights["concat_query_answer_candidate_constrained_template"],
            ],
            top_k=args.top_k,
        )
    retrieval_strings = dict(method_queries)
    retrieval_strings["bm25_rm3_query_only"] = rm3_retrieval_text
    retrieval_strings["cf_prompt_query_expansion_rrf"] = cf_prompt_selected_joined or cf_prompt_joined
    retrieval_strings["fawe_raw_expected_beta0p25"] = _format_fawe_retrieval_text(query.text, expected, named_fawe_betas["raw_expected"])
    retrieval_strings["fawe_masked_expected_beta0p25"] = _format_fawe_retrieval_text(query.text, masked, named_fawe_betas["masked_expected"])
    retrieval_strings["fawe_answer_constrained_beta0p5"] = _format_fawe_retrieval_text(
        query.text,
        template_retrieval_text,
        named_fawe_betas["answer_constrained"],
    )
    retrieval_strings["fawe_query2doc_beta0p25"] = _format_fawe_retrieval_text(query.text, query2doc_doc, named_fawe_betas["query2doc"])
    retrieval_strings["fawe_safe_adaptive_beta"] = _format_fawe_retrieval_text(query.text, template_retrieval_text, fawe_safe_beta)
    retrieval_strings["fawe_shuffled_expected_beta0p25"] = _format_fawe_retrieval_text(query.text, shuffled_expected, 0.25)
    retrieval_strings["fawe_wrong_answer_beta0p25"] = _format_fawe_retrieval_text(query.text, wrong_answer_only, 0.25)
    retrieval_strings["fawe_neutral_filler_beta0p25"] = _format_fawe_retrieval_text(query.text, length_matched_filler, 0.25)
    retrieval_strings["fawe_query_repeated_beta0p25"] = _format_fawe_retrieval_text(query.text, query.text, 0.25)
    retrieval_strings["fawe_random_terms_from_corpus_beta0p25"] = _format_fawe_retrieval_text(query.text, random_corpus_terms, 0.25)
    retrieval_strings["fawe_idf_matched_random_terms_beta0p25"] = _format_fawe_retrieval_text(query.text, idf_matched_random_terms, 0.25)
    if args.include_fawe_beta_grid:
        beta_grid_expansions = {
            "raw_expected": expected,
            "masked_expected": masked,
            "answer_constrained": template_retrieval_text,
            "query2doc": query2doc_doc,
        }
        for family, expansion_text in beta_grid_expansions.items():
            for beta in sorted(dict.fromkeys(fawe_betas or sorted(set(FAWE_DEFAULT_BETAS.values())))):
                retrieval_strings[f"fawe_{family}_beta{_weight_suffix(beta)}"] = _format_fawe_retrieval_text(
                    query.text,
                    expansion_text,
                    beta,
                )
    retrieval_specs = _build_retrieval_specs(
        query_text=query.text,
        method_queries=method_queries,
        answer_rrf_weights=answer_rrf_weights,
        named_fawe_betas=named_fawe_betas,
        fawe_safe_beta=fawe_safe_beta,
        fawe_expansions={
            "raw_expected": expected,
            "masked_expected": masked,
            "answer_constrained": template_retrieval_text,
            "query2doc": query2doc_doc,
        },
        cf_prompt_queries=cf_prompt_queries,
        safe_rrf_v1_weights=safe_rrf_v1_weights,
        cf_prompt_bundle=cf_prompt_bundle,
    )
    retrieval_specs["bm25_rm3_query_only"] = {
        "mode": "rm3_query_expansion",
        "fb_docs": args.rm3_fb_docs,
        "fb_terms": args.rm3_fb_terms,
        "original_query_weight": args.rm3_original_query_weight,
        "routes": {"query_only": query.text},
    }
    retrieval_specs["fawe_shuffled_expected_beta0p25"] = {
        "mode": "fielded_anchor_weighted",
        "beta": 0.25,
        "routes": {"query_only": query.text, "expansion": shuffled_expected},
    }
    retrieval_specs["fawe_wrong_answer_beta0p25"] = {
        "mode": "fielded_anchor_weighted",
        "beta": 0.25,
        "routes": {"query_only": query.text, "expansion": wrong_answer_only},
    }
    retrieval_specs["fawe_neutral_filler_beta0p25"] = {
        "mode": "fielded_anchor_weighted",
        "beta": 0.25,
        "routes": {"query_only": query.text, "expansion": length_matched_filler},
    }
    retrieval_specs["fawe_query_repeated_beta0p25"] = {
        "mode": "fielded_anchor_weighted",
        "beta": 0.25,
        "routes": {"query_only": query.text, "expansion": query.text},
    }
    retrieval_specs["fawe_random_terms_from_corpus_beta0p25"] = {
        "mode": "fielded_anchor_weighted",
        "beta": 0.25,
        "routes": {"query_only": query.text, "expansion": random_corpus_terms},
    }
    retrieval_specs["fawe_idf_matched_random_terms_beta0p25"] = {
        "mode": "fielded_anchor_weighted",
        "beta": 0.25,
        "routes": {"query_only": query.text, "expansion": idf_matched_random_terms},
    }
    if args.include_fawe_beta_grid:
        beta_grid_expansions = {
            "raw_expected": expected,
            "masked_expected": masked,
            "answer_constrained": template_retrieval_text,
            "query2doc": query2doc_doc,
        }
        for family, expansion_text in beta_grid_expansions.items():
            for beta in sorted(dict.fromkeys(fawe_betas or sorted(set(FAWE_DEFAULT_BETAS.values())))):
                retrieval_specs[f"fawe_{family}_beta{_weight_suffix(beta)}"] = {
                    "mode": "fielded_anchor_weighted",
                    "beta": beta,
                    "routes": {"query_only": query.text, "expansion": expansion_text},
                }
    wrong_answer_verification = {
        "candidate": wrong_answer_candidate,
        "candidate_present_in_wrong_answer_only": _contains_text(wrong_answer_only, wrong_answer_candidate),
        "candidate_present_in_concat_query_wrong_answer": _contains_text(wrong_answer_injection, wrong_answer_candidate),
        "concat_query_wrong_answer_differs_from_query_only": _normalize_text(wrong_answer_injection) != _normalize_text(query.text),
    }
    retrieval_diagnostics = _build_retrieval_diagnostics(
        query_id=query.query_id,
        original_query_text=str(query.original_text or query.text),
        query_text=query.text,
        retrieval_strings=retrieval_strings,
        rankings=rankings,
        qrels=dataset.qrels.get(query.query_id, {}),
        included_methods=rankings.keys(),
    )
    generation = {
        "query": query.text,
        "original_query": str(query.original_text or query.text),
        "answers": list(query.answers),
        "answer_aliases": list(query.answer_aliases),
        "expected_answer": expected,
        "masked_expected_answer": masked,
        "hyde_document": hyde_doc,
        "query2doc_document": query2doc_doc,
        "generative_relevance_feedback": relevance_feedback,
        "corpus_steered_expansion": corpus_steered_text,
        "corpus_steered_short_expansion": corpus_steered_short_text,
        "answer_candidate_template": template_text,
        "answer_candidate_template_parsed": template_payload,
        "answer_candidate_template_validation": template_validation,
        "cf_prompt_query_expansion": cf_prompt_payload,
        "cf_prompt_query_expansion_selection": {
            "support_context": cf_prompt_support,
            "generation_workers": args.generation_workers,
            "selected_queries": cf_prompt_selected_queries,
            "selected_weights": cf_prompt_selected_weights,
            "candidates": cf_prompt_bundle["candidates"],
        },
        "retrieval_strings": retrieval_strings,
        "retrieval_specs": retrieval_specs,
        "adaptive_fusion": {
            "safe_rrf_v0_weights": SAFE_RRF_V0_WEIGHTS,
            "safe_rrf_v1": route_reliability,
            "fawe_named_betas": named_fawe_betas,
            "fawe_safe_adaptive_beta": fawe_safe_beta,
        },
        "controls": {
            "oracle_answer_masked": oracle_masked,
            "post_hoc_gold_removed_expected_answer": posthoc_gold_removed,
            "random_span_masking": random_masked,
            "entity_only_masking": entity_only_masked,
            "generic_mask_slot": generic_masked,
            "length_matched_neutral_filler": length_matched_filler,
            "wrong_answer_only": wrong_answer_only,
            "concat_query_wrong_answer": wrong_answer_injection,
            "wrong_answer_injection": wrong_answer_injection,
            "query_repeated": query_repeated,
            "query_repeated_length_matched": query_repeated_length_matched,
            "query_plus_shuffled_expected": query_plus_shuffled_expected,
            "query_plus_neutral_filler": query_plus_neutral_filler,
            "neutral_filler_plus_query": neutral_filler_plus_query,
            "raw_expected_then_query": raw_expected_then_query,
            "fawe_shuffled_expected_beta0p25": shuffled_expected,
            "fawe_wrong_answer_beta0p25": wrong_answer_only,
            "fawe_neutral_filler_beta0p25": length_matched_filler,
            "fawe_query_repeated_beta0p25": query.text,
            "fawe_random_terms_from_corpus_beta0p25": random_corpus_terms,
            "fawe_idf_matched_random_terms_beta0p25": idf_matched_random_terms,
        },
        "integrity": {
            "wrong_answer_verification": wrong_answer_verification,
        },
        "retrieval_diagnostics": retrieval_diagnostics,
        "artifacts": {
            "expected_answer": expected_artifact,
            "masked_expected_answer": masked_artifact,
            "hyde_document": hyde_artifact,
            "query2doc_document": query2doc_artifact,
            "generative_relevance_feedback": feedback_artifact,
            "answer_candidate_template": template_artifact,
            "cf_prompt_query_expansion": cf_prompt_artifact,
        },
        "features": base_features,
        "leakage": leakage_scores,
        "leakage_labels": leakage_labels,
    }
    record = {
        "query_id": query.query_id,
        "query": query.text,
        "original_query": str(query.original_text or query.text),
        "checkpoint_context": dict(checkpoint_context),
        "answers": list(query.answers),
        "answer_aliases": list(query.answer_aliases),
        "generation": generation,
        "leakage_labels": leakage_labels,
        "integrity": {
            "wrong_answer_verification": wrong_answer_verification,
        },
        "retrieval_strings": retrieval_strings,
        "retrieval_specs": retrieval_specs,
        "retrieval_diagnostics": retrieval_diagnostics,
        "rankings": {name: rankings[name] for name in run_method_names if name in rankings},
        "qrels": dataset.qrels.get(query.query_id, {}),
    }
    return {
        "record": record,
        "generation": generation,
        "features_by_query": features_entry,
        "leakage_scores": leakage_scores,
    }


def _required_single_query_methods(selected_methods: set[str]) -> set[str]:
    needed = {"query_only"}
    synthetic_methods = {
        "query_only",
        "bm25_rm3_query_only",
        "dual_query_raw_expected_rrf",
        "dual_query_masked_expected_rrf",
        "dual_query_answer_candidate_constrained_template_rrf",
        "rrf_query_masked_expected",
        "rrf_query_answer_constrained",
        "rrf_query_wrong_answer",
        "rrf_query_corpus_steered_short",
        "safe_rrf_v0",
        "safe_rrf_v1",
        "cf_prompt_query_expansion_rrf",
        "fawe_raw_expected_beta0p25",
        "fawe_masked_expected_beta0p25",
        "fawe_answer_constrained_beta0p5",
        "fawe_query2doc_beta0p25",
        "fawe_safe_adaptive_beta",
        "fawe_shuffled_expected_beta0p25",
        "fawe_wrong_answer_beta0p25",
        "fawe_neutral_filler_beta0p25",
        "fawe_query_repeated_beta0p25",
        "fawe_random_terms_from_corpus_beta0p25",
        "fawe_idf_matched_random_terms_beta0p25",
    }
    for method_name in selected_methods:
        if method_name in synthetic_methods:
            continue
        if method_name.startswith("weighted_dual_query_raw_expected_rrf_w"):
            continue
        if method_name.startswith("weighted_dual_query_masked_expected_rrf_w"):
            continue
        if method_name.startswith("weighted_rrf_query_answer_constrained_w"):
            continue
        if method_name.startswith("fawe_raw_expected_beta"):
            continue
        if method_name.startswith("fawe_masked_expected_beta"):
            continue
        if method_name.startswith("fawe_answer_constrained_beta"):
            continue
        if method_name.startswith("fawe_query2doc_beta"):
            continue
        needed.add(method_name)
    if "dual_query_raw_expected_rrf" in selected_methods or any(
        method_name.startswith("weighted_dual_query_raw_expected_rrf_w") for method_name in selected_methods
    ):
        needed.add("raw_expected_answer_only")
    if (
        "dual_query_masked_expected_rrf" in selected_methods
        or "rrf_query_masked_expected" in selected_methods
        or any(method_name.startswith("weighted_dual_query_masked_expected_rrf_w") for method_name in selected_methods)
    ):
        needed.add("masked_expected_answer_only")
    if (
        "dual_query_answer_candidate_constrained_template_rrf" in selected_methods
        or "rrf_query_answer_constrained" in selected_methods
        or any(method_name.startswith("weighted_rrf_query_answer_constrained_w") for method_name in selected_methods)
    ):
        needed.add("answer_candidate_constrained_template_only")
    if "rrf_query_wrong_answer" in selected_methods:
        needed.add("wrong_answer_only")
    if "rrf_query_corpus_steered_short" in selected_methods:
        needed.add("corpus_steered_short_concat")
    if "safe_rrf_v0" in selected_methods or "safe_rrf_v1" in selected_methods or "fawe_safe_adaptive_beta" in selected_methods:
        needed.update(
            {
                "generative_relevance_feedback_concat",
                "query2doc_concat",
                "concat_query_answer_candidate_constrained_template",
            }
        )
    return needed


def _load_checkpoint_records(
    records_path: Path,
    dataset,
    required_methods: Iterable[str],
    checkpoint_context: Mapping[str, object],
) -> list[dict[str, object]]:
    if not records_path.exists():
        return []
    dataset_queries = {query.query_id: query for query in dataset.queries}
    required_method_set = set(required_methods)
    by_query_id: dict[str, dict[str, object]] = {}
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            query_id = str(record.get("query_id") or "")
            query = dataset_queries.get(query_id)
            if query is None:
                continue
            if not _checkpoint_record_is_compatible(record, query, required_method_set, checkpoint_context):
                continue
            by_query_id[query_id] = record
    return [by_query_id[query.query_id] for query in dataset.queries if query.query_id in by_query_id]


def _checkpoint_record_is_compatible(
    record: Mapping[str, object],
    query,
    required_methods: set[str],
    checkpoint_context: Mapping[str, object],
) -> bool:
    record_context = record.get("checkpoint_context", {})
    if not isinstance(record_context, Mapping):
        return False
    for key, value in checkpoint_context.items():
        if record_context.get(key) != value:
            return False
    stored_query = str(record.get("query") or "")
    if stored_query != query.text:
        return False
    rankings = record.get("rankings")
    if not isinstance(rankings, Mapping):
        return False
    return required_methods.issubset(set(rankings.keys()))


def _restore_checkpoint_state(
    checkpoint_records: list[dict[str, object]],
    runs: Mapping[str, Dict[str, RankedList]],
    dataset,
    metric_ks: Iterable[int] = (5, 10, 20),
) -> dict[str, object]:
    features_by_query: dict[str, dict[str, object]] = {}
    generations: dict[str, dict[str, object]] = {}
    leakage_by_method: dict[str, list[dict[str, object]]] = {name: [] for name in runs}
    per_query_method_metrics: dict[str, dict[str, dict[str, float]]] = {name: {} for name in runs}
    completed_query_ids: list[str] = []
    for record in checkpoint_records:
        query_id = str(record["query_id"])
        completed_query_ids.append(query_id)
        generation = record.get("generation", {})
        if isinstance(generation, Mapping):
            generations[query_id] = dict(generation)
            base_features = dict(generation.get("features", {}))
            raw_expected_leakage = dict((generation.get("leakage") or {}).get("raw_expected_answer_only", {}))
            if base_features or raw_expected_leakage:
                features_by_query[query_id] = {
                    **base_features,
                    "leakage_bucket": leakage_bucket_name(raw_expected_leakage) if raw_expected_leakage else None,
                    "raw_expected_answer_only": raw_expected_leakage,
                }
            for method_name, score in (generation.get("leakage") or {}).items():
                if method_name in leakage_by_method and isinstance(score, Mapping):
                    leakage_by_method[method_name].append(dict(score))
        rankings = record.get("rankings", {})
        for run_name in runs:
            ranking = rankings.get(run_name)
            if ranking is None:
                continue
            runs[run_name][query_id] = ranking
            per_query_method_metrics[run_name][query_id] = per_query_metrics(
                ranking,
                dataset.qrels.get(query_id, {}),
                ks=metric_ks,
            )
    return {
        "records": checkpoint_records,
        "features_by_query": features_by_query,
        "generations": generations,
        "leakage_by_method": leakage_by_method,
        "per_query_method_metrics": per_query_method_metrics,
        "completed_query_ids": completed_query_ids,
    }


def _append_checkpoint_record(handle, record: Mapping[str, object]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _compute_primary_comparisons(
    dataset,
    per_query_method_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    comparisons: Iterable[tuple[str, str]],
    bootstrap_samples: int,
    permutation_samples: int,
) -> dict[str, object]:
    results: dict[str, object] = {}
    query_ids = [query.query_id for query in dataset.queries if query.query_id in dataset.qrels]
    for left, right in comparisons:
        if left not in per_query_method_metrics or right not in per_query_method_metrics:
            continue
        metrics_for_pair = {}
        for metric_name in FROZEN_PRIMARY_METRICS:
            deltas = []
            for qid in query_ids:
                left_metrics = per_query_method_metrics[left].get(qid)
                right_metrics = per_query_method_metrics[right].get(qid)
                if not left_metrics or not right_metrics:
                    continue
                deltas.append(left_metrics.get(metric_name, 0.0) - right_metrics.get(metric_name, 0.0))
            metrics_for_pair[metric_name] = {
                "bootstrap_ci": paired_bootstrap_ci(deltas, num_samples=bootstrap_samples),
                "permutation": paired_permutation_test(deltas, num_samples=permutation_samples),
                "win_tie_loss": win_tie_loss(deltas),
            }
        results[f"{left}__vs__{right}"] = metrics_for_pair
    return results


def _resolve_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def _parse_float_list(text: str) -> list[float]:
    if not text.strip():
        return []
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _weight_suffix(weight: float) -> str:
    return str(weight).replace(".", "p")


def _default_cache_namespace(args: argparse.Namespace, dataset_name: str) -> str:
    return ":".join(
        [
            f"generator={args.generator}",
            f"model={args.model}",
            f"temperature={args.temperature}",
            f"dataset={dataset_name}",
            f"counterfactual={args.counterfactual}",
            f"alias_style={args.counterfactual_alias_style}",
        ]
    )


def _legacy_cache_namespaces(args: argparse.Namespace) -> list[str]:
    namespaces = [f"{args.generator}:{args.model}:temp={args.temperature}"]
    if args.cache_namespace:
        namespaces.append(args.cache_namespace)
    return list(dict.fromkeys(namespaces))


def _build_retrieval_specs(
    query_text: str,
    method_queries: Mapping[str, str],
    answer_rrf_weights: list[float],
    named_fawe_betas: Mapping[str, float],
    fawe_safe_beta: float,
    fawe_expansions: Mapping[str, str],
    cf_prompt_queries: list[str],
    safe_rrf_v1_weights: Mapping[str, float],
    cf_prompt_bundle: Mapping[str, object],
) -> dict[str, object]:
    specs: dict[str, object] = {
        method_name: {"mode": "single_query", "query_text": method_query}
        for method_name, method_query in method_queries.items()
    }
    specs["bm25_rm3_query_only"] = {
        "mode": "rm3_query_expansion",
        "fb_docs": None,
        "fb_terms": None,
        "original_query_weight": None,
        "routes": {"query_only": query_text},
    }
    specs["dual_query_raw_expected_rrf"] = {
        "mode": "rrf",
        "routes": {"query_only": query_text, "raw_expected_answer_only": method_queries["raw_expected_answer_only"]},
    }
    specs["dual_query_masked_expected_rrf"] = {
        "mode": "rrf",
        "routes": {"query_only": query_text, "masked_expected_answer_only": method_queries["masked_expected_answer_only"]},
    }
    specs["rrf_query_masked_expected"] = {
        "mode": "rrf",
        "routes": {"query_only": query_text, "masked_expected_answer_only": method_queries["masked_expected_answer_only"]},
    }
    specs["dual_query_answer_candidate_constrained_template_rrf"] = {
        "mode": "rrf",
        "routes": {
            "query_only": query_text,
            "answer_candidate_constrained_template_only": method_queries["answer_candidate_constrained_template_only"],
        },
    }
    specs["rrf_query_answer_constrained"] = specs["dual_query_answer_candidate_constrained_template_rrf"]
    specs["rrf_query_wrong_answer"] = {
        "mode": "rrf",
        "routes": {"query_only": query_text, "wrong_answer_only": method_queries["wrong_answer_only"]},
    }
    specs["rrf_query_corpus_steered_short"] = {
        "mode": "rrf",
        "routes": {"query_only": query_text, "corpus_steered_short_concat": method_queries["corpus_steered_short_concat"]},
    }
    specs["safe_rrf_v0"] = {
        "mode": "weighted_rrf",
        "weights": dict(SAFE_RRF_V0_WEIGHTS),
        "routes": {
            "query_only": query_text,
            "generative_relevance_feedback_concat": method_queries["generative_relevance_feedback_concat"],
            "query2doc_concat": method_queries["query2doc_concat"],
            "concat_query_answer_candidate_constrained_template": method_queries["concat_query_answer_candidate_constrained_template"],
        },
    }
    specs["safe_rrf_v1"] = {
        "mode": "adaptive_weighted_rrf",
        "weights": dict(safe_rrf_v1_weights),
        "routes": {
            "query_only": query_text,
            "generative_relevance_feedback_concat": method_queries["generative_relevance_feedback_concat"],
            "query2doc_concat": method_queries["query2doc_concat"],
            "concat_query_answer_candidate_constrained_template": method_queries["concat_query_answer_candidate_constrained_template"],
        },
    }
    specs["cf_prompt_query_expansion_rrf"] = {
        "mode": "adaptive_weighted_rrf",
        "weights": {
            "query_only": 1.0,
            "cf_prompt_queries": list(cf_prompt_bundle.get("selected_weights") or []),
        },
        "routes": {
            "query_only": query_text,
            "cf_prompt_queries": cf_prompt_queries,
            "selected_cf_prompt_queries": list(cf_prompt_bundle.get("selected_queries") or []),
            "selected_cf_prompt_roles": list(cf_prompt_bundle.get("selected_roles") or []),
        },
    }
    specs["fawe_raw_expected_beta0p25"] = {
        "mode": "fielded_anchor_weighted",
        "beta": named_fawe_betas["raw_expected"],
        "routes": {"query_only": query_text, "raw_expected_answer_only": fawe_expansions["raw_expected"]},
    }
    specs["fawe_masked_expected_beta0p25"] = {
        "mode": "fielded_anchor_weighted",
        "beta": named_fawe_betas["masked_expected"],
        "routes": {"query_only": query_text, "masked_expected_answer_only": fawe_expansions["masked_expected"]},
    }
    specs["fawe_answer_constrained_beta0p5"] = {
        "mode": "fielded_anchor_weighted",
        "beta": named_fawe_betas["answer_constrained"],
        "routes": {
            "query_only": query_text,
            "answer_candidate_constrained_template_only": fawe_expansions["answer_constrained"],
        },
    }
    specs["fawe_query2doc_beta0p25"] = {
        "mode": "fielded_anchor_weighted",
        "beta": named_fawe_betas["query2doc"],
        "routes": {"query_only": query_text, "query2doc_document": fawe_expansions["query2doc"]},
    }
    specs["fawe_safe_adaptive_beta"] = {
        "mode": "fielded_anchor_weighted",
        "beta": fawe_safe_beta,
        "routes": {
            "query_only": query_text,
            "answer_candidate_constrained_template_only": fawe_expansions["answer_constrained"],
        },
    }
    fawe_control_expansions = {
        "fawe_shuffled_expected_beta0p25": method_queries.get("query_plus_shuffled_expected", ""),
        "fawe_wrong_answer_beta0p25": method_queries.get("wrong_answer_only", ""),
        "fawe_neutral_filler_beta0p25": method_queries.get("length_matched_neutral_filler", ""),
        "fawe_query_repeated_beta0p25": query_text,
    }
    for method_name, expansion_text in fawe_control_expansions.items():
        specs[method_name] = {
            "mode": "fielded_anchor_weighted",
            "beta": 0.25,
            "routes": {"query_only": query_text, "expansion": expansion_text},
        }
    for weight in answer_rrf_weights:
        suffix = _weight_suffix(weight)
        specs[f"weighted_dual_query_raw_expected_rrf_w{suffix}"] = {
            "mode": "weighted_rrf",
            "weights": {"query_only": 1.0, "raw_expected_answer_only": weight},
            "routes": {"query_only": query_text, "raw_expected_answer_only": method_queries["raw_expected_answer_only"]},
        }
        specs[f"weighted_dual_query_masked_expected_rrf_w{suffix}"] = {
            "mode": "weighted_rrf",
            "weights": {"query_only": 1.0, "masked_expected_answer_only": weight},
            "routes": {"query_only": query_text, "masked_expected_answer_only": method_queries["masked_expected_answer_only"]},
        }
        specs[f"weighted_rrf_query_answer_constrained_w{suffix}"] = {
            "mode": "weighted_rrf",
            "weights": {"query_only": 1.0, "answer_candidate_constrained_template_only": weight},
            "routes": {
                "query_only": query_text,
                "answer_candidate_constrained_template_only": method_queries["answer_candidate_constrained_template_only"],
            },
        }
    for beta in sorted(dict.fromkeys(named_fawe_betas.values())):
        suffix = _weight_suffix(beta)
        specs.setdefault(
            f"fawe_query2doc_beta{suffix}",
            {
                "mode": "fielded_anchor_weighted",
                "beta": beta,
                "routes": {"query_only": query_text, "query2doc_document": fawe_expansions["query2doc"]},
            },
        )
    return specs


def _summarize_integrity(records: list[dict[str, object]], counterfactual_validation: Mapping[str, object] | None) -> dict[str, object]:
    wrong_answer_rows = [record.get("integrity", {}).get("wrong_answer_verification", {}) for record in records]
    verification = {
        "candidate_present_in_wrong_answer_only_rate": _rate(
            row.get("candidate_present_in_wrong_answer_only") for row in wrong_answer_rows
        ),
        "candidate_present_in_concat_query_wrong_answer_rate": _rate(
            row.get("candidate_present_in_concat_query_wrong_answer") for row in wrong_answer_rows
        ),
        "concat_query_wrong_answer_differs_from_query_only_rate": _rate(
            row.get("concat_query_wrong_answer_differs_from_query_only") for row in wrong_answer_rows
        ),
    }
    if counterfactual_validation:
        verification["counterfactual_validation"] = dict(counterfactual_validation)
    return verification


def _rate(values: Iterable[object]) -> float | None:
    filtered = [bool(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(1 for value in filtered if value) / len(filtered)


def _contains_text(haystack: str, needle: str) -> bool:
    return bool(_normalize_text(needle) and _normalize_text(needle) in _normalize_text(haystack))


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


if __name__ == "__main__":
    main()
