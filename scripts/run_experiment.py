from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
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
from expected_answer_rag.counterfactual import build_entity_counterfactual_dataset, export_counterfactual_artifacts
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


FROZEN_PRIMARY_METRICS = ["ndcg@10", "recall@10", "recall@20", "mrr@10"]
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
    parser.add_argument("--generation-workers", type=int, default=1, help="Threads for precomputing generations across queries.")
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
        print(
            f"Phase 0.5/4: Building counterfactual dataset "
            f"(regime={args.counterfactual}, alias_style={args.counterfactual_alias_style})"
        )
        counterfactual = build_entity_counterfactual_dataset(
            dataset,
            alias_style=args.counterfactual_alias_style,
            include_values=args.counterfactual == "entity_and_value",
            seed=args.counterfactual_seed,
            progress=print,
        )
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
    runs = _initialize_runs(answer_rrf_weights)
    doc_map = {doc.doc_id: doc for doc in dataset.corpus}
    cache_namespace = generator.namespace
    cache_namespace_aliases = _legacy_cache_namespaces(args)
    query_contexts = {}
    print(
        f"Phase 1/4: First-pass retrieval over {len(dataset.queries)} queries "
        f"(retriever={args.retriever}, top_k={args.top_k})"
    )
    for query in tqdm(dataset.queries, desc="First-pass retrieval"):
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
        f"Phase 2/4: Precomputing generations for {len(dataset.queries)} queries "
        f"(workers={args.generation_workers}, generator={args.generator}, model={args.model})"
    )
    generation_bundles = _precompute_generation_bundles(
        args=args,
        dataset_name=dataset.name,
        queries=dataset.queries,
        cache=generator.cache,
        cache_namespace=cache_namespace,
        cache_namespace_aliases=cache_namespace_aliases,
        support_contexts={
            query_id: str(bundle["cf_prompt_support"]["prompt_context"])
            for query_id, bundle in query_contexts.items()
        },
    )
    print(f"Phase 3/4: Running retrieval fusion and evaluation over {len(dataset.queries)} queries")
    features_by_query: dict[str, dict[str, object]] = {}
    generations = {}
    records = []
    leakage_by_method: dict[str, list[dict[str, object]]] = {name: [] for name in runs}
    per_query_method_metrics: dict[str, dict[str, dict[str, float]]] = {name: {} for name in runs}

    for index, query in enumerate(tqdm(dataset.queries, desc="Running queries")):
        query_context = query_contexts[query.query_id]
        generation_bundle = generation_bundles[query.query_id]
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
        random_masked = random_span_mask_answer(expected, query.text, seed=args.counterfactual_seed + index)
        entity_only_masked = entity_only_mask_answer(expected, query.text)
        generic_masked = generic_mask_answer(masked)
        length_matched_filler = length_matched_neutral_filler(query.text, expected)
        wrong_answer_candidate = _wrong_answer_candidate(dataset, index)
        wrong_answer_only = wrong_answer_candidate
        wrong_answer_injection = f"{query.text}\n{wrong_answer_candidate}".strip()

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
        cf_prompt_selected_joined = "\n".join(cf_prompt_selected_queries).strip()
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
        }
        method_queries["query_only"] = query.text

        rankings = {"query_only": query_rank}
        for method_name, method_query in method_queries.items():
            if method_name == "query_only":
                continue
            rankings[method_name] = retriever.search(method_query, top_k=args.top_k)
        rankings["cf_prompt_query_expansion_rrf"] = weighted_reciprocal_rank_fusion(
            [query_rank, *cf_prompt_bundle["rankings"]] if cf_prompt_bundle["rankings"] else [query_rank],
            [1.0, *cf_prompt_selected_weights] if cf_prompt_bundle["rankings"] else [1.0],
            top_k=args.top_k,
        )

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
        bucket = leakage_bucket_name(leakage_scores["raw_expected_answer_only"])
        features_by_query[query.query_id] = {
            **base_features,
            "leakage_bucket": bucket,
            "raw_expected_answer_only": leakage_scores["raw_expected_answer_only"],
        }
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
        runs["safe_rrf_v1"][query.query_id] = rankings["safe_rrf_v1"]
        per_query_method_metrics["safe_rrf_v1"][query.query_id] = per_query_metrics(
            rankings["safe_rrf_v1"],
            dataset.qrels.get(query.query_id, {}),
        )
        retrieval_strings = dict(method_queries)
        retrieval_strings["cf_prompt_query_expansion_rrf"] = cf_prompt_selected_joined or cf_prompt_joined
        retrieval_specs = _build_retrieval_specs(
            query_text=query.text,
            method_queries=method_queries,
            answer_rrf_weights=answer_rrf_weights,
            cf_prompt_queries=cf_prompt_queries,
            safe_rrf_v1_weights=safe_rrf_v1_weights,
            cf_prompt_bundle=cf_prompt_bundle,
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
                "cf_prompt_query_expansion": cf_prompt_artifact,
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
        "cf_prompt_query_expansion_rrf": select_qualitative_examples(records, "cf_prompt_query_expansion_rrf", limit=args.qualitative_limit),
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
        "safe_rrf_v0": {},
        "safe_rrf_v1": {},
        "cf_prompt_query_expansion_rrf": {},
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
    cf_prompt_queries: list[str],
    safe_rrf_v1_weights: Mapping[str, float],
    cf_prompt_bundle: Mapping[str, object],
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
