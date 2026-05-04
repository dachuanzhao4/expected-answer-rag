from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping

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
from expected_answer_rag.counterfactual import build_entity_counterfactual_dataset, export_counterfactual_artifacts
from expected_answer_rag.datasets import Document, export_local_dataset, load_dataset
from expected_answer_rag.fusion import reciprocal_rank_fusion, weighted_reciprocal_rank_fusion
from expected_answer_rag.generators import (
    CachedTextGenerator,
    HeuristicGenerator,
    MissingGenerator,
    OpenAITextGenerator,
    entity_only_mask_answer,
    generic_mask_answer,
    length_matched_neutral_filler,
    parse_answer_candidate_template,
    random_span_mask_answer,
    remove_gold_from_text,
    validate_answer_candidate_template,
)
from expected_answer_rag.leakage import leakage_bucket_name, score_generation_methods
from expected_answer_rag.metrics import evaluate_run, per_query_metrics
from expected_answer_rag.qualitative import select_qualitative_examples
from expected_answer_rag.retrieval import RankedList, make_retriever
from expected_answer_rag.statistics import paired_bootstrap_ci, paired_permutation_test, win_tie_loss


FROZEN_PRIMARY_METRICS = ["ndcg@10", "recall@10", "recall@20", "mrr@10"]
FROZEN_PRIMARY_COMPARISONS = [
    ("concat_query_raw_expected", "concat_query_masked_expected"),
    ("concat_query_raw_expected", "concat_query_answer_candidate_constrained_template"),
    ("hyde_doc_only", "concat_query_answer_candidate_constrained_template"),
    ("query2doc_concat", "concat_query_answer_candidate_constrained_template"),
    ("concat_query_raw_expected", "corpus_steered_expansion_concat"),
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
    parser.add_argument("--retriever", choices=["bm25", "dense"], default="bm25")
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
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--answer-rrf-weights",
        default="0.25,0.5,0.75",
        help="Comma-separated answer-route weights for weighted RRF. Query route weight is 1.0.",
    )
    parser.add_argument("--output", default="outputs/run.json")
    parser.add_argument("--records-output", default="outputs/records.jsonl")
    parser.add_argument("--clear-generation-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true", help="Use existing generation cache and fail if any generation is missing.")
    parser.add_argument("--cache-namespace", default=None, help="Override generation cache namespace.")
    parser.add_argument("--qualitative-limit", type=int, default=5)
    parser.add_argument("--stats-bootstrap-samples", type=int, default=500)
    parser.add_argument("--stats-permutation-samples", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(
        args.dataset,
        max_corpus=args.max_corpus,
        max_queries=args.max_queries,
        cache_dir=args.cache_dir,
        query_metadata_path=args.query_metadata,
    )
    if args.counterfactual != "none":
        counterfactual = build_entity_counterfactual_dataset(
            dataset,
            alias_style=args.counterfactual_alias_style,
            include_values=args.counterfactual == "entity_and_value",
            seed=args.counterfactual_seed,
        )
        dataset = counterfactual.dataset
        if args.counterfactual_export_dir:
            export_local_dataset(dataset, _resolve_path(args.counterfactual_export_dir))
            export_counterfactual_artifacts(counterfactual, _resolve_path(args.counterfactual_export_dir))
        counterfactual_validation = counterfactual.validation
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
    runs = _initialize_runs(answer_rrf_weights)
    doc_map = {doc.doc_id: doc for doc in dataset.corpus}
    features_by_query: dict[str, dict[str, object]] = {}
    generations = {}
    records = []
    leakage_by_method: dict[str, list[dict[str, object]]] = {name: [] for name in runs}
    per_query_method_metrics: dict[str, dict[str, dict[str, float]]] = {name: {} for name in runs}

    for index, query in enumerate(tqdm(dataset.queries, desc="Running queries")):
        relevant_docs = [doc_map[doc_id] for doc_id in dataset.qrels.get(query.query_id, {}) if doc_id in doc_map]
        expected = generator.expected_answer(query.text)
        expected_artifact = generator.last_artifact()
        masked = generator.mask_answer(query.text, expected)
        masked_artifact = generator.last_artifact()
        hyde_doc = generator.hyde_document(query.text)
        hyde_artifact = generator.last_artifact()
        query2doc_doc = generator.query2doc_document(query.text)
        query2doc_artifact = generator.last_artifact()
        relevance_feedback = generator.relevance_feedback(query.text)
        feedback_artifact = generator.last_artifact()
        template_text = generator.answer_candidate_template(query.text)
        template_artifact = generator.last_artifact()
        template_payload = parse_answer_candidate_template(template_text, query.text)
        template_validation = validate_answer_candidate_template(query.text, template_payload)
        template_retrieval_text = str(template_payload.get("retrieval_text") or query.text)

        oracle_masked = remove_gold_from_text(expected, list(query.all_answer_strings))
        posthoc_gold_removed = remove_gold_from_text(expected, list(query.all_answer_strings))
        random_masked = random_span_mask_answer(expected, query.text, seed=args.counterfactual_seed + index)
        entity_only_masked = entity_only_mask_answer(expected, query.text)
        generic_masked = generic_mask_answer(masked)
        length_matched_filler = length_matched_neutral_filler(query.text, expected)
        wrong_answer_candidate = _wrong_answer_candidate(dataset, index)
        wrong_answer_only = wrong_answer_candidate
        wrong_answer_injection = f"{query.text}\n{wrong_answer_candidate}".strip()

        query_rank = retriever.search(query.text, top_k=args.top_k)
        corpus_steered_text = _build_corpus_steered_expansion(query.text, query_rank, doc_map)
        corpus_steered_short_text = _build_corpus_steered_expansion(
            query.text,
            query_rank,
            doc_map,
            max_docs=2,
            max_words=40,
        )

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
        }
        method_queries["query_only"] = query.text

        rankings = {"query_only": query_rank}
        for method_name, method_query in method_queries.items():
            if method_name == "query_only":
                continue
            rankings[method_name] = retriever.search(method_query, top_k=args.top_k)

        rankings["dual_query_raw_expected_rrf"] = reciprocal_rank_fusion([query_rank, rankings["raw_expected_answer_only"]], top_k=args.top_k)
        rankings["dual_query_masked_expected_rrf"] = reciprocal_rank_fusion([query_rank, rankings["masked_expected_answer_only"]], top_k=args.top_k)
        rankings["dual_query_answer_candidate_constrained_template_rrf"] = reciprocal_rank_fusion(
            [query_rank, rankings["answer_candidate_constrained_template_only"]],
            top_k=args.top_k,
        )
        rankings["rrf_query_masked_expected"] = rankings["dual_query_masked_expected_rrf"]
        rankings["rrf_query_answer_constrained"] = rankings["dual_query_answer_candidate_constrained_template_rrf"]
        rankings["rrf_query_wrong_answer"] = reciprocal_rank_fusion(
            [query_rank, rankings["wrong_answer_only"]],
            top_k=args.top_k,
        )
        rankings["rrf_query_corpus_steered_short"] = reciprocal_rank_fusion(
            [query_rank, rankings["corpus_steered_short_concat"]],
            top_k=args.top_k,
        )
        for weight in answer_rrf_weights:
            suffix = _weight_suffix(weight)
            rankings[f"weighted_dual_query_raw_expected_rrf_w{suffix}"] = weighted_reciprocal_rank_fusion(
                [query_rank, rankings["raw_expected_answer_only"]],
                [1.0, weight],
                top_k=args.top_k,
            )
            rankings[f"weighted_dual_query_masked_expected_rrf_w{suffix}"] = weighted_reciprocal_rank_fusion(
                [query_rank, rankings["masked_expected_answer_only"]],
                [1.0, weight],
                top_k=args.top_k,
            )
            rankings[f"weighted_rrf_query_answer_constrained_w{suffix}"] = weighted_reciprocal_rank_fusion(
                [query_rank, rankings["answer_candidate_constrained_template_only"]],
                [1.0, weight],
                top_k=args.top_k,
            )

        for run_name in runs:
            if run_name in rankings:
                runs[run_name][query.query_id] = rankings[run_name]
                per_query_method_metrics[run_name][query.query_id] = per_query_metrics(
                    rankings[run_name],
                    dataset.qrels.get(query.query_id, {}),
                )

        base_features = generation_features(query, expected, masked, hyde_doc)
        generated_texts_for_leakage = {
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
        bucket = leakage_bucket_name(leakage_scores["raw_expected_answer_only"])
        features_by_query[query.query_id] = {
            **base_features,
            "leakage_bucket": bucket,
            "raw_expected_answer_only": leakage_scores["raw_expected_answer_only"],
        }
        retrieval_strings = dict(method_queries)
        retrieval_specs = _build_retrieval_specs(
            query_text=query.text,
            method_queries=method_queries,
            answer_rrf_weights=answer_rrf_weights,
        )
        wrong_answer_verification = {
            "candidate": wrong_answer_candidate,
            "candidate_present_in_wrong_answer_only": _contains_text(wrong_answer_only, wrong_answer_candidate),
            "candidate_present_in_concat_query_wrong_answer": _contains_text(wrong_answer_injection, wrong_answer_candidate),
            "concat_query_wrong_answer_differs_from_query_only": _normalize_text(wrong_answer_injection) != _normalize_text(query.text),
        }

        generations[query.query_id] = {
            "query": query.text,
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
            "retrieval_strings": retrieval_strings,
            "retrieval_specs": retrieval_specs,
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
            },
            "integrity": {
                "wrong_answer_verification": wrong_answer_verification,
            },
            "artifacts": {
                "expected_answer": expected_artifact,
                "masked_expected_answer": masked_artifact,
                "hyde_document": hyde_artifact,
                "query2doc_document": query2doc_artifact,
                "generative_relevance_feedback": feedback_artifact,
                "answer_candidate_template": template_artifact,
            },
            "features": base_features,
            "leakage": leakage_scores,
            "leakage_labels": leakage_labels,
        }

        for method_name, score in leakage_scores.items():
            if method_name in leakage_by_method:
                leakage_by_method[method_name].append(score)

        records.append(
            {
                "query_id": query.query_id,
                "query": query.text,
                "answers": list(query.answers),
                "answer_aliases": list(query.answer_aliases),
                "generation": generations[query.query_id],
                "leakage_labels": leakage_labels,
                "integrity": {
                    "wrong_answer_verification": wrong_answer_verification,
                },
                "retrieval_strings": retrieval_strings,
                "retrieval_specs": retrieval_specs,
                "rankings": {name: runs[name][query.query_id] for name in runs if query.query_id in runs[name]},
                "qrels": dataset.qrels.get(query.query_id, {}),
            }
        )

    metrics = {name: evaluate_run(run, dataset.qrels) for name, run in runs.items()}
    leakage_metrics = {name: evaluate_by_leakage_bucket(run, dataset.qrels, features_by_query) for name, run in runs.items()}
    stats = _compute_primary_comparisons(
        dataset,
        per_query_method_metrics,
        comparisons=FROZEN_PRIMARY_COMPARISONS,
        bootstrap_samples=args.stats_bootstrap_samples,
        permutation_samples=args.stats_permutation_samples,
    )
    qualitative = {
        "raw_expected_answer_only": select_qualitative_examples(records, "raw_expected_answer_only", limit=args.qualitative_limit),
        "masked_expected_answer_only": select_qualitative_examples(records, "masked_expected_answer_only", limit=args.qualitative_limit),
        "answer_candidate_constrained_template_only": select_qualitative_examples(records, "answer_candidate_constrained_template_only", limit=args.qualitative_limit),
    }
    result = {
        "dataset": dataset.name,
        "dataset_metadata": dict(dataset.metadata),
        "counterfactual_validation": counterfactual_validation,
        "num_corpus": len(dataset.corpus),
        "num_queries": len(dataset.queries),
        "num_qrels_queries": len(dataset.qrels),
        "retriever": args.retriever,
        "embedding_model": args.embedding_model if args.retriever == "dense" else None,
        "query_prefix": args.query_prefix if args.retriever == "dense" else None,
        "embedding_cache": str(_resolve_path(args.embedding_cache)) if args.embedding_cache else None,
        "answer_rrf_weights": answer_rrf_weights,
        "generator": args.generator,
        "model": args.model if args.generator in {"openai", "openrouter"} else None,
        "generation_cache_path": str(cache_path),
        "generation_cache_namespace": generator.namespace,
        "counterfactual_regime": args.counterfactual,
        "counterfactual_alias_style": args.counterfactual_alias_style if args.counterfactual != "none" else None,
        "top_k": args.top_k,
        "frozen_primary_metrics": FROZEN_PRIMARY_METRICS,
        "frozen_primary_comparisons": FROZEN_PRIMARY_COMPARISONS,
        "metrics": metrics,
        "method_ranking": compare_methods(metrics),
        "generation_summary": summarize_generation_features(features_by_query.values()),
        "method_leakage_summary": summarize_method_leakage(leakage_by_method),
        "leakage_bucket_metrics": leakage_metrics,
        "primary_comparison_stats": stats,
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


def _initialize_runs(answer_rrf_weights: list[float]) -> Dict[str, Dict[str, RankedList]]:
    base_methods = {
        "query_only": {},
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
        "rrf_query_corpus_steered_short": {},
    }
    for weight in answer_rrf_weights:
        suffix = _weight_suffix(weight)
        base_methods[f"weighted_dual_query_raw_expected_rrf_w{suffix}"] = {}
        base_methods[f"weighted_dual_query_masked_expected_rrf_w{suffix}"] = {}
        base_methods[f"weighted_rrf_query_answer_constrained_w{suffix}"] = {}
    return base_methods


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


def _wrong_answer_candidate(dataset, index: int) -> str:
    other_answers = []
    for offset in range(1, len(dataset.queries)):
        other = dataset.queries[(index + offset) % len(dataset.queries)]
        if other.answers:
            other_answers = list(other.answers)
            break
    return other_answers[0] if other_answers else "Wrong Candidate"


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
) -> dict[str, object]:
    specs: dict[str, object] = {
        method_name: {"mode": "single_query", "query_text": method_query}
        for method_name, method_query in method_queries.items()
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
