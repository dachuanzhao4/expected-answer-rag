from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tqdm import tqdm

from expected_answer_rag.analysis import (
    compare_methods,
    evaluate_by_leakage_bucket,
    generation_features,
    summarize_generation_features,
)
from expected_answer_rag.answer_blanked import (
    build_lf_er_package,
    build_llm_lf_er_package,
    build_relation_keyword_query,
    normalize_for_match,
    validate_answer_blanked_format,
)
from expected_answer_rag.cache import JsonCache
from expected_answer_rag.datasets import load_dataset
from expected_answer_rag.fusion import (
    agreement_adjusted_weights,
    agreement_weighted_reciprocal_rank_fusion,
    weighted_reciprocal_rank_fusion,
)
from expected_answer_rag.generators import CachedTextGenerator, HeuristicGenerator, MissingGenerator, OpenAITextGenerator
from expected_answer_rag.metrics import evaluate_run
from expected_answer_rag.retrieval import RankedList, make_retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Query2Doc masking retrieval baselines.")
    parser.add_argument(
        "--local-cache-root",
        default="D:/hf_cache",
        help="Local cache root for HuggingFace, Transformers, sentence-transformers, and temp files.",
    )
    parser.add_argument("--dataset", default="toy", help="toy, nq, hotpotqa, fiqa, scifact, ...")
    parser.add_argument(
        "--max-corpus",
        type=int,
        default=200000,
        help="Maximum number of corpus documents to load. Default is the stage-2 setting: 200k.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=500,
        help="Maximum number of queries to run. Default is the stage-2 setting: 500.",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--retriever", choices=["bm25", "dense"], default="bm25")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--embedding-chunk-size", type=int, default=1024)
    parser.add_argument("--embedding-cache", default=None)
    parser.add_argument(
        "--embedding-device",
        default=None,
        help="Device for sentence-transformers, e.g. 'cuda' or 'cpu'. Default lets the library choose.",
    )
    parser.add_argument(
        "--query-prefix",
        default="Represent this sentence for searching relevant passages: ",
        help="Prefix applied only to dense retrieval queries. Use '' to disable.",
    )
    parser.add_argument("--generator", choices=["heuristic", "openai", "openrouter"], default="heuristic")
    parser.add_argument("--model", default="openai/gpt-5-mini")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument(
        "--token-param",
        choices=["auto", "max_tokens", "max_completion_tokens", "none"],
        default="none",
    )
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--referer", default=None)
    parser.add_argument("--app-title", default="query2doc-mask-rag")
    parser.add_argument("--include-reasoning", action="store_true")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument(
        "--prompt-style",
        choices=["query2doc_fewshot", "query2doc_zero_shot"],
        default="query2doc_fewshot",
        help="Prompt used to generate pseudo-documents.",
    )
    parser.add_argument("--generation-cache", default="outputs/generation_cache.json")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Write partial metrics every N queries. Set 0 to disable partial checkpoints.",
    )
    parser.add_argument(
        "--query-repeat",
        type=int,
        default=5,
        help="For BM25 Query2Doc expansion, repeat the original query this many times before appending the pseudo-doc.",
    )
    parser.add_argument(
        "--dense-separator",
        default="[SEP]",
        help="Separator between original query and pseudo-doc for dense Query2Doc expansion.",
    )
    parser.add_argument(
        "--include-answer-blanked",
        action="store_true",
        help=(
            "Also run answer-blanked Query2Doc methods: skeleton-only, expanded query, "
            "and query/skeleton/relation-keyword RRF fusion."
        ),
    )
    parser.add_argument(
        "--answer-blanked-fusion-weights",
        default="1.0,0.8,0.6",
        help="Comma-separated RRF weights for query, answer-blanked skeleton, and relation keyword rankings.",
    )
    parser.add_argument(
        "--include-lf-er",
        action="store_true",
        help=(
            "Run leakage-free evidence reformat routes: anchor, relation keywords, "
            "forward/inverse evidence, slotless evidence, and weighted RRF fusion."
        ),
    )
    parser.add_argument(
        "--include-llm-reformat",
        action="store_true",
        help=(
            "Run structured LLM answer-free reformat routes. The LLM emits intent JSON; "
            "local validation renders retrieval views without answer guesses."
        ),
    )
    parser.add_argument(
        "--llm-reformat-version",
        choices=["v1", "v2"],
        default="v2",
        help=(
            "Structured reformat schema. v1 keeps free-form generic cues; "
            "v2 uses a controlled relation-class ontology and local rendering."
        ),
    )
    parser.add_argument(
        "--llm-reformat-fusion-weights",
        default="6.0,1.0,1.0,0.6,0.6",
        help=(
            "Comma-separated RRF weights for LLM reformat routes: query, anchor, "
            "intent terms, corpus style, expanded query."
        ),
    )
    parser.add_argument(
        "--lf-er-fusion-weights",
        default="4.0,0.3,0.8,0.1,0.1,0.6",
        help=(
            "Comma-separated RRF weights for LF-ER routes: original query, anchor, "
            "relation keywords, evidence forward, evidence inverse, slotless evidence."
        ),
    )
    parser.add_argument(
        "--lf-er-bm25-agreement-weights",
        default="4.0,0.9,0.7,0.6,0.0",
        help=(
            "Comma-separated agreement-gated weights for BM25 LF-ER v3 routes: "
            "query, relation keywords, slotless evidence, evidence inverse, template expansion."
        ),
    )
    parser.add_argument(
        "--lf-er-dense-agreement-weights",
        default="4.0,1.0,0.8,0.6,0.4",
        help=(
            "Comma-separated agreement-gated weights for dense LF-ER routes: "
            "query, anchor, dense-safe view, dense-safe expanded query, relation-expanded query."
        ),
    )
    parser.add_argument(
        "--lf-er-inverse-query-weights",
        default="4.0,0.8",
        help="Comma-separated RRF weights for LF-ER v3.1 main route: original query and evidence inverse.",
    )
    parser.add_argument(
        "--lf-er-anchor-gated-fusion-weights",
        default="4.0,1.4,0.8,0.5,0.4",
        help=(
            "Comma-separated agreement-gated weights for LF-ER v4 routes: "
            "query, LF-ER expanded query, corpus-style view, relation keywords, slotless evidence."
        ),
    )
    parser.add_argument(
        "--lf-er-short-fusion-weights",
        default="6.0,1.2,1.0,0.8",
        help=(
            "Comma-separated RRF weights for LF-ER v4.1 short-expanded routes: "
            "query, relation-expanded, slotless-expanded, short-expanded."
        ),
    )
    parser.add_argument(
        "--lf-er-dense-safe-fusion-weights",
        default="6.0,1.2,0.8,0.4",
        help=(
            "Comma-separated agreement-gated weights for LF-ER dense-safe routes: "
            "query, anchor, dense-safe view, dense-safe expanded query."
        ),
    )
    parser.add_argument("--output", default="outputs/run.json")
    parser.add_argument("--records-output", default="outputs/records.jsonl")
    parser.add_argument("--clear-generation-cache", action="store_true")
    parser.add_argument(
        "--api-workers",
        type=int,
        default=1,
        help=(
            "Generate API-backed query artifacts concurrently across queries. "
            "Default 1 preserves the original sequential behavior."
        ),
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Use existing generation cache and fail if any generation is missing.",
    )
    parser.add_argument(
        "--cache-namespace",
        default=None,
        help="Override generation cache namespace, useful for cache-only reruns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_local_cache(args.local_cache_root)
    dataset = load_dataset(
        args.dataset,
        max_corpus=args.max_corpus,
        max_queries=args.max_queries,
        cache_dir=args.cache_dir,
    )
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
        embedding_device=args.embedding_device,
    )
    if args.cache_only:
        base_generator = MissingGenerator()
    elif args.generator in {"openai", "openrouter"}:
        base_generator = OpenAITextGenerator(
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
            prompt_style=args.prompt_style,
            llm_reformat_version=args.llm_reformat_version,
        )
    else:
        base_generator = HeuristicGenerator(llm_reformat_version=args.llm_reformat_version)
    cache_path = _resolve_path(args.generation_cache)
    if args.clear_generation_cache and cache_path.exists():
        cache_path.unlink()
    generator = CachedTextGenerator(
        inner=base_generator,
        cache=JsonCache(cache_path),
        namespace=args.cache_namespace or f"{args.generator}:{args.model}:temp={args.temperature}:prompt={args.prompt_style}",
        llm_reformat_version=args.llm_reformat_version,
    )

    runs: Dict[str, Dict[str, RankedList]] = {
        "query_only": {},
        "query2doc_pseudo_doc_only": {},
        "masked_query2doc_pseudo_doc_only": {},
        "query2doc_expanded_query": {},
        "masked_query2doc_expanded_query": {},
    }
    if args.include_answer_blanked:
        runs.update(
            {
                "answer_blanked_pseudo_doc_only": {},
                "answer_blanked_expanded_query": {},
                "answer_blanked_relation_keywords": {},
                "answer_blanked_fusion": {},
            }
        )
    if args.include_lf_er:
        runs.update(
            {
                "lf_er_anchor_view": {},
                "lf_er_relation_keyword_view": {},
                "lf_er_evidence_forward_view": {},
                "lf_er_evidence_inverse_view": {},
                "lf_er_slotless_evidence_view": {},
                "lf_er_bm25_field_view": {},
                "lf_er_dense_natural_view": {},
                "lf_er_dense_safe_view": {},
                "lf_er_dense_safe_expanded_query": {},
                "lf_er_template_expansion_view": {},
                "lf_er_corpus_style_view": {},
                "lf_er_expanded_query": {},
                "lf_er_relation_expanded_query": {},
                "lf_er_slotless_expanded_query": {},
                "lf_er_short_expanded_query": {},
                "lf_er_fusion": {},
                "lf_er_agreement_fusion": {},
                "lf_er_inverse_query_fusion": {},
                "lf_er_anchor_gated_fusion": {},
                "lf_er_short_expanded_fusion": {},
                "lf_er_dense_safe_fusion": {},
            }
        )
    if args.include_llm_reformat:
        runs.update(
            {
                "llm_reformat_anchor_view": {},
                "llm_reformat_intent_terms_view": {},
                "llm_reformat_dense_view": {},
                "llm_reformat_bm25_view": {},
                "llm_reformat_corpus_style_view": {},
                "llm_reformat_expanded_query": {},
                "llm_reformat_dense_expanded_query": {},
                "llm_reformat_fusion": {},
            }
        )
    answer_blanked_fusion_weights = parse_fusion_weights(args.answer_blanked_fusion_weights)
    lf_er_fusion_weights = parse_lf_er_fusion_weights(args.lf_er_fusion_weights)
    lf_er_bm25_agreement_weights = parse_lf_er_agreement_weights(
        args.lf_er_bm25_agreement_weights,
        "--lf-er-bm25-agreement-weights",
    )
    lf_er_dense_agreement_weights = parse_lf_er_agreement_weights(
        args.lf_er_dense_agreement_weights,
        "--lf-er-dense-agreement-weights",
    )
    lf_er_inverse_query_weights = parse_lf_er_inverse_query_weights(args.lf_er_inverse_query_weights)
    lf_er_anchor_gated_fusion_weights = parse_lf_er_anchor_gated_fusion_weights(
        args.lf_er_anchor_gated_fusion_weights
    )
    lf_er_short_fusion_weights = parse_lf_er_short_fusion_weights(args.lf_er_short_fusion_weights)
    lf_er_dense_safe_fusion_weights = parse_lf_er_dense_safe_fusion_weights(args.lf_er_dense_safe_fusion_weights)
    llm_reformat_fusion_weights = parse_llm_reformat_fusion_weights(args.llm_reformat_fusion_weights)
    generations = {}
    features_by_query = {}
    diagnostics_by_query = {}
    doc_by_id = {doc.doc_id: doc.text for doc in dataset.corpus}
    records_path = _resolve_path(args.records_output)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_text("", encoding="utf-8")
    generation_bundles = precompute_generation_bundles(dataset.queries, args, generator)

    for idx, query in enumerate(tqdm(dataset.queries, desc="Running queries"), start=1):
        bundle = generation_bundles.get(query.query_id)
        if bundle is None:
            bundle = build_generation_bundle(query, args, generator)
        query2doc = bundle["query2doc"]
        masked_query2doc = bundle["masked_query2doc"]
        answer_blanked_query2doc = bundle["answer_blanked_query2doc"]
        answer_blanked_relation_query = bundle["answer_blanked_relation_query"]
        lf_er_package = bundle["lf_er_package"]
        lf_er_views = bundle["lf_er_views"]
        llm_reformat_views = bundle["llm_reformat_views"]
        features_by_query[query.query_id] = bundle["features"]
        generations[query.query_id] = bundle["generation"]

        expanded_query = build_expanded_query(
            query=query.text,
            pseudo_doc=query2doc,
            retriever=args.retriever,
            query_repeat=args.query_repeat,
            dense_separator=args.dense_separator,
        )
        masked_expanded_query = build_expanded_query(
            query=query.text,
            pseudo_doc=masked_query2doc,
            retriever=args.retriever,
            query_repeat=args.query_repeat,
            dense_separator=args.dense_separator,
        )
        if args.include_answer_blanked:
            answer_blanked_expanded_query = build_expanded_query(
                query=query.text,
                pseudo_doc=str(answer_blanked_query2doc),
                retriever=args.retriever,
                query_repeat=args.query_repeat,
                dense_separator=args.dense_separator,
            )
        lf_er_expanded_query = None
        lf_er_relation_expanded_query = None
        lf_er_slotless_expanded_query = None
        lf_er_short_expanded_query = None
        lf_er_dense_safe_expanded_query = None
        llm_reformat_expanded_query = None
        llm_reformat_dense_expanded_query = None
        if args.include_lf_er:
            lf_er_expanded_query = build_expanded_query(
                query=query.text,
                pseudo_doc=lf_er_views["corpus_style_view"],
                retriever=args.retriever,
                query_repeat=args.query_repeat,
                dense_separator=args.dense_separator,
            )
            lf_er_relation_expanded_query = build_expanded_query(
                query=query.text,
                pseudo_doc=lf_er_views["relation_keyword_view"],
                retriever=args.retriever,
                query_repeat=args.query_repeat,
                dense_separator=args.dense_separator,
            )
            lf_er_slotless_expanded_query = build_expanded_query(
                query=query.text,
                pseudo_doc=lf_er_views["slotless_evidence_view"],
                retriever=args.retriever,
                query_repeat=args.query_repeat,
                dense_separator=args.dense_separator,
            )
            lf_er_short_expanded_query = build_expanded_query(
                query=query.text,
                pseudo_doc=build_lf_er_short_expansion(lf_er_views),
                retriever=args.retriever,
                query_repeat=args.query_repeat,
                dense_separator=args.dense_separator,
            )
            lf_er_dense_safe_expanded_query = build_expanded_query(
                query=query.text,
                pseudo_doc=lf_er_views["dense_safe_expansion_view"],
                retriever=args.retriever,
                query_repeat=args.query_repeat,
                dense_separator=args.dense_separator,
            )
        if args.include_llm_reformat:
            llm_reformat_expanded_query = build_expanded_query(
                query=query.text,
                pseudo_doc=llm_reformat_views["llm_corpus_style_view"],
                retriever=args.retriever,
                query_repeat=args.query_repeat,
                dense_separator=args.dense_separator,
            )
            llm_reformat_dense_expanded_query = build_expanded_query(
                query=query.text,
                pseudo_doc=llm_reformat_views["llm_dense_expansion_view"],
                retriever=args.retriever,
                query_repeat=args.query_repeat,
                dense_separator=args.dense_separator,
            )
        query_rank = retriever.search(query.text, top_k=args.top_k)
        query2doc_rank = retriever.search(query2doc, top_k=args.top_k)
        masked_query2doc_rank = retriever.search(masked_query2doc, top_k=args.top_k)
        expanded_query_rank = retriever.search(expanded_query, top_k=args.top_k)
        masked_expanded_query_rank = retriever.search(masked_expanded_query, top_k=args.top_k)
        if args.include_answer_blanked:
            answer_blanked_rank = retriever.search(str(answer_blanked_query2doc), top_k=args.top_k)
            answer_blanked_expanded_rank = retriever.search(str(answer_blanked_expanded_query), top_k=args.top_k)
            answer_blanked_relation_rank = retriever.search(str(answer_blanked_relation_query), top_k=args.top_k)
            answer_blanked_fusion_rank = weighted_reciprocal_rank_fusion(
                [query_rank, answer_blanked_rank, answer_blanked_relation_rank],
                weights=answer_blanked_fusion_weights,
                top_k=args.top_k,
            )
        if args.include_lf_er:
            lf_er_anchor_rank = retriever.search(lf_er_views["anchor_view"], top_k=args.top_k)
            lf_er_keyword_rank = retriever.search(lf_er_views["relation_keyword_view"], top_k=args.top_k)
            lf_er_forward_rank = retriever.search(lf_er_views["evidence_forward_view"], top_k=args.top_k)
            lf_er_inverse_rank = retriever.search(lf_er_views["evidence_inverse_view"], top_k=args.top_k)
            lf_er_slotless_rank = retriever.search(lf_er_views["slotless_evidence_view"], top_k=args.top_k)
            lf_er_bm25_field_rank = retriever.search(lf_er_views["bm25_field_view"], top_k=args.top_k)
            lf_er_dense_natural_rank = retriever.search(lf_er_views["dense_natural_view"], top_k=args.top_k)
            lf_er_dense_safe_rank = retriever.search(lf_er_views["dense_safe_view"], top_k=args.top_k)
            lf_er_template_rank = retriever.search(lf_er_views["template_expansion_view"], top_k=args.top_k)
            lf_er_corpus_style_rank = retriever.search(lf_er_views["corpus_style_view"], top_k=args.top_k)
            lf_er_expanded_rank = retriever.search(str(lf_er_expanded_query), top_k=args.top_k)
            lf_er_relation_expanded_rank = retriever.search(str(lf_er_relation_expanded_query), top_k=args.top_k)
            lf_er_slotless_expanded_rank = retriever.search(str(lf_er_slotless_expanded_query), top_k=args.top_k)
            lf_er_short_expanded_rank = retriever.search(str(lf_er_short_expanded_query), top_k=args.top_k)
            lf_er_dense_safe_expanded_rank = retriever.search(str(lf_er_dense_safe_expanded_query), top_k=args.top_k)
            lf_er_fusion_rank = weighted_reciprocal_rank_fusion(
                [
                    query_rank,
                    lf_er_anchor_rank,
                    lf_er_keyword_rank,
                    lf_er_forward_rank,
                    lf_er_inverse_rank,
                    lf_er_slotless_rank,
                ],
                weights=lf_er_fusion_weights,
                top_k=args.top_k,
            )
            agreement_rankings, agreement_weights, agreement_names = lf_er_agreement_inputs(
                retriever_name=args.retriever,
                query_rank=query_rank,
                route_rankings={
                    "lf_er_relation_keyword_view": lf_er_keyword_rank,
                    "lf_er_slotless_evidence_view": lf_er_slotless_rank,
                    "lf_er_bm25_field_view": lf_er_bm25_field_rank,
                    "lf_er_template_expansion_view": lf_er_template_rank,
                    "lf_er_dense_natural_view": lf_er_dense_natural_rank,
                    "lf_er_anchor_view": lf_er_anchor_rank,
                    "lf_er_dense_safe_view": lf_er_dense_safe_rank,
                    "lf_er_dense_safe_expanded_query": lf_er_dense_safe_expanded_rank,
                    "lf_er_relation_expanded_query": lf_er_relation_expanded_rank,
                    "lf_er_short_expanded_query": lf_er_short_expanded_rank,
                    "lf_er_evidence_forward_view": lf_er_forward_rank,
                    "lf_er_evidence_inverse_view": lf_er_inverse_rank,
                },
                bm25_weights=lf_er_bm25_agreement_weights,
                dense_weights=lf_er_dense_agreement_weights,
            )
            lf_er_agreement_adjusted_weights = agreement_adjusted_weights(
                agreement_rankings,
                agreement_weights,
            )
            lf_er_agreement_fusion_rank = agreement_weighted_reciprocal_rank_fusion(
                agreement_rankings,
                agreement_weights,
                top_k=args.top_k,
            )
            lf_er_inverse_query_fusion_rank = weighted_reciprocal_rank_fusion(
                [query_rank, lf_er_inverse_rank],
                weights=lf_er_inverse_query_weights,
                top_k=args.top_k,
            )
            anchor_gated_rankings = [
                query_rank,
                lf_er_expanded_rank,
                lf_er_corpus_style_rank,
                lf_er_keyword_rank,
                lf_er_slotless_rank,
            ]
            anchor_gated_names = [
                "query_only",
                "lf_er_expanded_query",
                "lf_er_corpus_style_view",
                "lf_er_relation_keyword_view",
                "lf_er_slotless_evidence_view",
            ]
            lf_er_anchor_gated_adjusted_weights = agreement_adjusted_weights(
                anchor_gated_rankings,
                lf_er_anchor_gated_fusion_weights,
            )
            lf_er_anchor_gated_fusion_rank = agreement_weighted_reciprocal_rank_fusion(
                anchor_gated_rankings,
                lf_er_anchor_gated_fusion_weights,
                top_k=args.top_k,
            )
            lf_er_short_expanded_fusion_rank = weighted_reciprocal_rank_fusion(
                [
                    query_rank,
                    lf_er_relation_expanded_rank,
                    lf_er_slotless_expanded_rank,
                    lf_er_short_expanded_rank,
                ],
                weights=lf_er_short_fusion_weights,
                top_k=args.top_k,
            )
            dense_safe_rankings = [
                query_rank,
                lf_er_anchor_rank,
                lf_er_dense_safe_rank,
                lf_er_dense_safe_expanded_rank,
            ]
            dense_safe_names = [
                "query_only",
                "lf_er_anchor_view",
                "lf_er_dense_safe_view",
                "lf_er_dense_safe_expanded_query",
            ]
            lf_er_dense_safe_adjusted_weights = agreement_adjusted_weights(
                dense_safe_rankings,
                lf_er_dense_safe_fusion_weights,
            )
            lf_er_dense_safe_fusion_rank = agreement_weighted_reciprocal_rank_fusion(
                dense_safe_rankings,
                lf_er_dense_safe_fusion_weights,
                top_k=args.top_k,
            )
        if args.include_llm_reformat:
            llm_reformat_anchor_rank = retriever.search(llm_reformat_views["llm_anchor_view"], top_k=args.top_k)
            llm_reformat_intent_rank = retriever.search(
                llm_reformat_views["llm_intent_terms_view"],
                top_k=args.top_k,
            )
            llm_reformat_dense_rank = retriever.search(llm_reformat_views["llm_dense_view"], top_k=args.top_k)
            llm_reformat_bm25_rank = retriever.search(llm_reformat_views["llm_bm25_view"], top_k=args.top_k)
            llm_reformat_corpus_rank = retriever.search(
                llm_reformat_views["llm_corpus_style_view"],
                top_k=args.top_k,
            )
            llm_reformat_expanded_rank = retriever.search(str(llm_reformat_expanded_query), top_k=args.top_k)
            llm_reformat_dense_expanded_rank = retriever.search(
                str(llm_reformat_dense_expanded_query),
                top_k=args.top_k,
            )
            llm_reformat_fusion_rank = weighted_reciprocal_rank_fusion(
                [
                    query_rank,
                    llm_reformat_anchor_rank,
                    llm_reformat_intent_rank,
                    llm_reformat_corpus_rank,
                    llm_reformat_expanded_rank,
                ],
                weights=llm_reformat_fusion_weights,
                top_k=args.top_k,
            )

        runs["query_only"][query.query_id] = query_rank
        runs["query2doc_pseudo_doc_only"][query.query_id] = query2doc_rank
        runs["masked_query2doc_pseudo_doc_only"][query.query_id] = masked_query2doc_rank
        runs["query2doc_expanded_query"][query.query_id] = expanded_query_rank
        runs["masked_query2doc_expanded_query"][query.query_id] = masked_expanded_query_rank
        if args.include_answer_blanked:
            runs["answer_blanked_pseudo_doc_only"][query.query_id] = answer_blanked_rank
            runs["answer_blanked_expanded_query"][query.query_id] = answer_blanked_expanded_rank
            runs["answer_blanked_relation_keywords"][query.query_id] = answer_blanked_relation_rank
            runs["answer_blanked_fusion"][query.query_id] = answer_blanked_fusion_rank
        if args.include_lf_er:
            runs["lf_er_anchor_view"][query.query_id] = lf_er_anchor_rank
            runs["lf_er_relation_keyword_view"][query.query_id] = lf_er_keyword_rank
            runs["lf_er_evidence_forward_view"][query.query_id] = lf_er_forward_rank
            runs["lf_er_evidence_inverse_view"][query.query_id] = lf_er_inverse_rank
            runs["lf_er_slotless_evidence_view"][query.query_id] = lf_er_slotless_rank
            runs["lf_er_bm25_field_view"][query.query_id] = lf_er_bm25_field_rank
            runs["lf_er_dense_natural_view"][query.query_id] = lf_er_dense_natural_rank
            runs["lf_er_dense_safe_view"][query.query_id] = lf_er_dense_safe_rank
            runs["lf_er_dense_safe_expanded_query"][query.query_id] = lf_er_dense_safe_expanded_rank
            runs["lf_er_template_expansion_view"][query.query_id] = lf_er_template_rank
            runs["lf_er_corpus_style_view"][query.query_id] = lf_er_corpus_style_rank
            runs["lf_er_expanded_query"][query.query_id] = lf_er_expanded_rank
            runs["lf_er_relation_expanded_query"][query.query_id] = lf_er_relation_expanded_rank
            runs["lf_er_slotless_expanded_query"][query.query_id] = lf_er_slotless_expanded_rank
            runs["lf_er_short_expanded_query"][query.query_id] = lf_er_short_expanded_rank
            runs["lf_er_fusion"][query.query_id] = lf_er_fusion_rank
            runs["lf_er_agreement_fusion"][query.query_id] = lf_er_agreement_fusion_rank
            runs["lf_er_inverse_query_fusion"][query.query_id] = lf_er_inverse_query_fusion_rank
            runs["lf_er_anchor_gated_fusion"][query.query_id] = lf_er_anchor_gated_fusion_rank
            runs["lf_er_short_expanded_fusion"][query.query_id] = lf_er_short_expanded_fusion_rank
            runs["lf_er_dense_safe_fusion"][query.query_id] = lf_er_dense_safe_fusion_rank
            diagnostics_by_query[query.query_id] = build_lf_er_diagnostics(
                query_text=query.text,
                qrels=dataset.qrels.get(query.query_id, {}),
                doc_by_id=doc_by_id,
                package=lf_er_package.to_dict(),
                rankings={
                    "query_only": query_rank,
                    "lf_er_anchor_view": lf_er_anchor_rank,
                    "lf_er_relation_keyword_view": lf_er_keyword_rank,
                    "lf_er_evidence_forward_view": lf_er_forward_rank,
                    "lf_er_evidence_inverse_view": lf_er_inverse_rank,
                    "lf_er_slotless_evidence_view": lf_er_slotless_rank,
                    "lf_er_bm25_field_view": lf_er_bm25_field_rank,
                    "lf_er_dense_natural_view": lf_er_dense_natural_rank,
                    "lf_er_dense_safe_view": lf_er_dense_safe_rank,
                    "lf_er_dense_safe_expanded_query": lf_er_dense_safe_expanded_rank,
                    "lf_er_template_expansion_view": lf_er_template_rank,
                    "lf_er_corpus_style_view": lf_er_corpus_style_rank,
                    "lf_er_expanded_query": lf_er_expanded_rank,
                    "lf_er_relation_expanded_query": lf_er_relation_expanded_rank,
                    "lf_er_slotless_expanded_query": lf_er_slotless_expanded_rank,
                    "lf_er_short_expanded_query": lf_er_short_expanded_rank,
                    "lf_er_fusion": lf_er_fusion_rank,
                    "lf_er_agreement_fusion": lf_er_agreement_fusion_rank,
                    "lf_er_inverse_query_fusion": lf_er_inverse_query_fusion_rank,
                    "lf_er_anchor_gated_fusion": lf_er_anchor_gated_fusion_rank,
                    "lf_er_short_expanded_fusion": lf_er_short_expanded_fusion_rank,
                    "lf_er_dense_safe_fusion": lf_er_dense_safe_fusion_rank,
                },
                agreement_route_names=agreement_names,
                agreement_adjusted_weights=lf_er_agreement_adjusted_weights,
                anchor_gated_route_names=anchor_gated_names,
                anchor_gated_adjusted_weights=lf_er_anchor_gated_adjusted_weights,
                dense_safe_route_names=dense_safe_names,
                dense_safe_adjusted_weights=lf_er_dense_safe_adjusted_weights,
            )
        if args.include_llm_reformat:
            runs["llm_reformat_anchor_view"][query.query_id] = llm_reformat_anchor_rank
            runs["llm_reformat_intent_terms_view"][query.query_id] = llm_reformat_intent_rank
            runs["llm_reformat_dense_view"][query.query_id] = llm_reformat_dense_rank
            runs["llm_reformat_bm25_view"][query.query_id] = llm_reformat_bm25_rank
            runs["llm_reformat_corpus_style_view"][query.query_id] = llm_reformat_corpus_rank
            runs["llm_reformat_expanded_query"][query.query_id] = llm_reformat_expanded_rank
            runs["llm_reformat_dense_expanded_query"][query.query_id] = llm_reformat_dense_expanded_rank
            runs["llm_reformat_fusion"][query.query_id] = llm_reformat_fusion_rank
        record = {
            "query_id": query.query_id,
            "query": query.text,
            "answers": list(query.answers),
            "generation": generations[query.query_id],
            "expanded_queries": {
                "query2doc_expanded_query": expanded_query,
                "masked_query2doc_expanded_query": masked_expanded_query,
            },
            "rankings": {name: runs[name][query.query_id] for name in runs},
            "qrels": dataset.qrels.get(query.query_id, {}),
        }
        if args.include_answer_blanked:
            record["expanded_queries"]["answer_blanked_expanded_query"] = answer_blanked_expanded_query
            record["expanded_queries"]["answer_blanked_relation_keywords"] = answer_blanked_relation_query
        if args.include_lf_er:
            record["expanded_queries"]["lf_er_expanded_query"] = lf_er_expanded_query
            record["expanded_queries"]["lf_er_relation_expanded_query"] = lf_er_relation_expanded_query
            record["expanded_queries"]["lf_er_slotless_expanded_query"] = lf_er_slotless_expanded_query
            record["expanded_queries"]["lf_er_short_expanded_query"] = lf_er_short_expanded_query
            record["expanded_queries"]["lf_er_dense_safe_expanded_query"] = lf_er_dense_safe_expanded_query
            record["lf_er_views"] = lf_er_views
            record["lf_er_diagnostics"] = diagnostics_by_query[query.query_id]
        if args.include_llm_reformat:
            record["expanded_queries"]["llm_reformat_expanded_query"] = llm_reformat_expanded_query
            record["expanded_queries"]["llm_reformat_dense_expanded_query"] = llm_reformat_dense_expanded_query
            record["llm_reformat_views"] = llm_reformat_views
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        if args.checkpoint_every > 0 and idx % args.checkpoint_every == 0:
            partial_result = build_result(
                args,
                dataset,
                runs,
                features_by_query,
                generations,
                completed_queries=idx,
                diagnostics_by_query=diagnostics_by_query,
            )
            partial_path = _partial_output_path(_resolve_path(args.output))
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text(json.dumps(partial_result, indent=2, ensure_ascii=False), encoding="utf-8")

    result = build_result(
        args,
        dataset,
        runs,
        features_by_query,
        generations,
        completed_queries=len(dataset.queries),
        diagnostics_by_query=diagnostics_by_query,
    )

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
    print("Method ranking:")
    print(json.dumps(result["method_ranking"], indent=2, ensure_ascii=False))
    print(f"Wrote {output_path}")
    print(f"Wrote {records_path}")


def _resolve_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def configure_local_cache(cache_root: str | None) -> None:
    if not cache_root:
        return
    root = Path(cache_root)
    paths = {
        "HF_HOME": root,
        "HF_HUB_CACHE": root / "hub",
        "HF_DATASETS_CACHE": root / "datasets",
        "TRANSFORMERS_CACHE": root / "transformers",
        "SENTENCE_TRANSFORMERS_HOME": root / "sentence_transformers",
        "TMP": root / "tmp",
        "TEMP": root / "tmp",
    }
    for value in paths.values():
        value.mkdir(parents=True, exist_ok=True)
    for key, value in paths.items():
        os.environ.setdefault(key, str(value))


def _partial_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.partial{path.suffix}")


def build_result(
    args,
    dataset,
    runs,
    features_by_query,
    generations,
    completed_queries: int,
    diagnostics_by_query: dict | None = None,
) -> dict:
    qrels = {
        query_id: rels
        for query_id, rels in dataset.qrels.items()
        if any(query_id in run for run in runs.values())
    }
    metrics = {name: evaluate_run(run, qrels) for name, run in runs.items()}
    leakage_metrics = {
        name: evaluate_by_leakage_bucket(run, qrels, features_by_query)
        for name, run in runs.items()
    }
    return {
        "dataset": dataset.name,
        "num_corpus": len(dataset.corpus),
        "num_queries": len(dataset.queries),
        "completed_queries": completed_queries,
        "num_qrels_queries": len(dataset.qrels),
        "retriever": args.retriever,
        "embedding_model": args.embedding_model if args.retriever == "dense" else None,
        "query_prefix": args.query_prefix if args.retriever == "dense" else None,
        "embedding_cache": str(_resolve_path(args.embedding_cache)) if args.embedding_cache else None,
        "embedding_device": args.embedding_device if args.retriever == "dense" else None,
        "generator": args.generator,
        "model": args.model if args.generator in {"openai", "openrouter"} else None,
        "prompt_style": args.prompt_style if args.generator in {"openai", "openrouter"} else None,
        "query_repeat": args.query_repeat if args.retriever == "bm25" else None,
        "dense_separator": args.dense_separator if args.retriever == "dense" else None,
        "include_answer_blanked": args.include_answer_blanked,
        "answer_blanked_fusion_weights": parse_fusion_weights(args.answer_blanked_fusion_weights)
        if args.include_answer_blanked
        else None,
        "include_lf_er": args.include_lf_er,
        "lf_er_fusion_weights": parse_lf_er_fusion_weights(args.lf_er_fusion_weights)
        if args.include_lf_er
        else None,
        "lf_er_bm25_agreement_weights": parse_lf_er_agreement_weights(
            args.lf_er_bm25_agreement_weights,
            "--lf-er-bm25-agreement-weights",
        )
        if args.include_lf_er
        else None,
        "lf_er_dense_agreement_weights": parse_lf_er_agreement_weights(
            args.lf_er_dense_agreement_weights,
            "--lf-er-dense-agreement-weights",
        )
        if args.include_lf_er
        else None,
        "lf_er_inverse_query_weights": parse_lf_er_inverse_query_weights(args.lf_er_inverse_query_weights)
        if args.include_lf_er
        else None,
        "lf_er_anchor_gated_fusion_weights": parse_lf_er_anchor_gated_fusion_weights(
            args.lf_er_anchor_gated_fusion_weights
        )
        if args.include_lf_er
        else None,
        "lf_er_short_fusion_weights": parse_lf_er_short_fusion_weights(args.lf_er_short_fusion_weights)
        if args.include_lf_er
        else None,
        "lf_er_dense_safe_fusion_weights": parse_lf_er_dense_safe_fusion_weights(
            args.lf_er_dense_safe_fusion_weights
        )
        if args.include_lf_er
        else None,
        "include_llm_reformat": args.include_llm_reformat,
        "llm_reformat_version": args.llm_reformat_version if args.include_llm_reformat else None,
        "llm_reformat_fusion_weights": parse_llm_reformat_fusion_weights(args.llm_reformat_fusion_weights)
        if args.include_llm_reformat
        else None,
        "api_workers": args.api_workers,
        "top_k": args.top_k,
        "metrics": metrics,
        "method_ranking": compare_methods(metrics),
        "generation_summary": summarize_generation_features(features_by_query.values()),
        "leakage_bucket_metrics": leakage_metrics,
        "lf_er_diagnostics_summary": summarize_lf_er_diagnostics(diagnostics_by_query or {})
        if args.include_lf_er
        else None,
        "sample_generations": dict(list(generations.items())[:5]),
    }


def precompute_generation_bundles(queries, args, generator) -> dict[str, dict]:
    workers = max(1, int(args.api_workers or 1))
    if workers <= 1 or args.cache_only:
        return {}
    print(f"Precomputing generations with api_workers={workers}", flush=True)
    bundles: dict[str, dict] = {}
    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_by_query_id = {
            executor.submit(build_generation_bundle, query, args, generator): query.query_id
            for query in queries
        }
        for future in tqdm(
            futures.as_completed(future_by_query_id),
            total=len(future_by_query_id),
            desc="Generating",
        ):
            query_id = future_by_query_id[future]
            try:
                bundles[query_id] = future.result()
            except Exception as exc:  # noqa: BLE001 - add query context.
                raise RuntimeError(f"Generation failed for {query_id}") from exc
    return bundles


def build_generation_bundle(query, args, generator) -> dict:
    query2doc = generator.query2doc(query.text)
    masked_query2doc = generator.mask_query2doc(query.text, query2doc)
    answer_blanked_query2doc = None
    answer_blanked_relation_query = None
    answer_blanked_validation = None
    lf_er_package = None
    lf_er_views = {}
    llm_reformat_raw = None
    llm_reformat_package = None
    llm_reformat_views = {}
    if args.include_answer_blanked:
        answer_blanked_query2doc = generator.answer_blanked_query2doc(query.text)
        answer_blanked_validation = validate_answer_blanked_format(query.text, answer_blanked_query2doc)
        if not answer_blanked_validation.ok:
            raise RuntimeError(
                f"Invalid answer-blanked Query2Doc for {query.query_id}: "
                f"{answer_blanked_validation.issues}\n{answer_blanked_query2doc}"
            )
        answer_blanked_relation_query = build_relation_keyword_query(query.text)
    if args.include_lf_er:
        lf_er_package = build_lf_er_package(query.text)
        if not lf_er_package.validation.ok:
            raise RuntimeError(
                f"Invalid LF-ER package for {query.query_id}: "
                f"{lf_er_package.validation.issues}\n{lf_er_package.to_dict()}"
            )
        lf_er_views = {view.name: view.text for view in lf_er_package.retrieval_views}
    if args.include_llm_reformat:
        llm_reformat_raw = generator.llm_reformat_intent(query.text)
        llm_reformat_package = build_llm_lf_er_package(
            query.text,
            llm_reformat_raw,
            version=args.llm_reformat_version,
        )
        if not llm_reformat_package.validation.ok:
            raise RuntimeError(
                f"Invalid LLM reformat package for {query.query_id}: "
                f"{llm_reformat_package.validation.issues}\n{llm_reformat_package.to_dict()}"
            )
        llm_reformat_views = {view.name: view.text for view in llm_reformat_package.retrieval_views}
    features = generation_features(query, query2doc, masked_query2doc)
    generation = {
        "query": query.text,
        "answers": list(query.answers),
        "query2doc": query2doc,
        "masked_query2doc": masked_query2doc,
        "features": features,
    }
    if args.include_answer_blanked:
        generation["answer_blanked_query2doc"] = answer_blanked_query2doc
        generation["answer_blanked_validation"] = {
            "ok": answer_blanked_validation.ok,
            "issues": list(answer_blanked_validation.issues),
        }
        generation["answer_blanked_relation_query"] = answer_blanked_relation_query
    if args.include_lf_er:
        generation["lf_er_package"] = lf_er_package.to_dict()
    if args.include_llm_reformat:
        generation["llm_reformat_raw"] = llm_reformat_raw
        generation["llm_reformat_package"] = llm_reformat_package.to_dict()
    return {
        "query2doc": query2doc,
        "masked_query2doc": masked_query2doc,
        "answer_blanked_query2doc": answer_blanked_query2doc,
        "answer_blanked_relation_query": answer_blanked_relation_query,
        "lf_er_package": lf_er_package,
        "lf_er_views": lf_er_views,
        "llm_reformat_views": llm_reformat_views,
        "features": features,
        "generation": generation,
    }


def build_expanded_query(
    query: str,
    pseudo_doc: str,
    retriever: str,
    query_repeat: int,
    dense_separator: str,
) -> str:
    if retriever == "bm25":
        return " ".join([query] * max(query_repeat, 1) + [pseudo_doc])
    return f"{query} {dense_separator} {pseudo_doc}"


def build_lf_er_short_expansion(views: dict[str, str]) -> str:
    source = " ".join(
        [
            str(views.get("relation_keyword_view", "")),
            str(views.get("slotless_evidence_view", "")),
        ]
    )
    return dedupe_whitespace_terms(source, max_terms=48)


def dedupe_whitespace_terms(text: str, max_terms: int) -> str:
    terms = []
    seen = set()
    for token in str(text).split():
        cleaned = token.strip(" ,.;:!?()[]{}\"'")
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(cleaned)
        if len(terms) >= max_terms:
            break
    return " ".join(terms)


def build_lf_er_diagnostics(
    query_text: str,
    qrels: dict[str, int],
    doc_by_id: dict[str, str],
    package: dict[str, object],
    rankings: dict[str, RankedList],
    agreement_route_names: list[str] | None = None,
    agreement_adjusted_weights: list[float] | None = None,
    anchor_gated_route_names: list[str] | None = None,
    anchor_gated_adjusted_weights: list[float] | None = None,
    dense_safe_route_names: list[str] | None = None,
    dense_safe_adjusted_weights: list[float] | None = None,
) -> dict[str, object]:
    route_best_ranks = {
        name: best_relevant_rank(ranking, qrels)
        for name, ranking in rankings.items()
    }
    oracle_routes = {
        name: rank
        for name, rank in route_best_ranks.items()
        if not is_lf_er_fusion_route(name)
    }
    oracle_best_route = min(oracle_routes, key=oracle_routes.get) if oracle_routes else None
    oracle_best_rank = oracle_routes.get(oracle_best_route, 999) if oracle_best_route else 999
    reformat_routes = {
        name: rank
        for name, rank in route_best_ranks.items()
        if name.startswith("lf_er_") and not is_lf_er_fusion_route(name)
    }
    best_reformat_route = min(reformat_routes, key=reformat_routes.get) if reformat_routes else None
    best_reformat_rank = reformat_routes.get(best_reformat_route, 999) if best_reformat_route else 999
    anchor_coverage = anchor_coverage_in_gold(package, qrels, doc_by_id)
    relation_overlap = relation_overlap_in_gold(package, qrels, doc_by_id)
    failure_type = classify_lf_er_failure(
        route_best_ranks=route_best_ranks,
        best_reformat_rank=best_reformat_rank,
        anchor_coverage=anchor_coverage,
        relation_overlap=relation_overlap,
    )
    diagnostic = {
        "route_best_ranks": route_best_ranks,
        "oracle_best_route": oracle_best_route,
        "oracle_best_rank": oracle_best_rank,
        "best_reformat_route": best_reformat_route,
        "best_reformat_rank": best_reformat_rank,
        "anchor_coverage_in_gold": anchor_coverage,
        "relation_term_overlap_in_gold": relation_overlap,
        "anchor_preservation_ok": package.get("validation", {}).get("ok", False),
        "slot_ok": expected_slot_present(package),
        "failure_type": failure_type,
        "query": query_text,
    }
    if agreement_route_names and agreement_adjusted_weights:
        diagnostic["agreement_adjusted_weights"] = {
            name: weight for name, weight in zip(agreement_route_names, agreement_adjusted_weights)
        }
    if anchor_gated_route_names and anchor_gated_adjusted_weights:
        diagnostic["anchor_gated_adjusted_weights"] = {
            name: weight for name, weight in zip(anchor_gated_route_names, anchor_gated_adjusted_weights)
        }
    if dense_safe_route_names and dense_safe_adjusted_weights:
        diagnostic["dense_safe_adjusted_weights"] = {
            name: weight for name, weight in zip(dense_safe_route_names, dense_safe_adjusted_weights)
        }
    return diagnostic


def best_relevant_rank(ranking: RankedList, qrels: dict[str, int]) -> int:
    if not qrels:
        return 999
    relevant = {doc_id for doc_id, score in qrels.items() if score > 0}
    for rank, (doc_id, _score) in enumerate(ranking, start=1):
        if doc_id in relevant:
            return rank
    return 999


def anchor_coverage_in_gold(package: dict[str, object], qrels: dict[str, int], doc_by_id: dict[str, str]) -> float:
    anchors = [
        str(anchor.get("text", ""))
        for anchor in package.get("known_anchors", [])
        if anchor.get("required", True) and str(anchor.get("text", "")).strip()
    ]
    if not anchors:
        return 1.0
    gold_text = normalized_gold_text(qrels, doc_by_id)
    if not gold_text:
        return 0.0
    covered = sum(1 for anchor in anchors if normalize_for_match(anchor) in gold_text)
    return covered / len(anchors)


def relation_overlap_in_gold(package: dict[str, object], qrels: dict[str, int], doc_by_id: dict[str, str]) -> float:
    relation_frame = package.get("relation_frame", {})
    terms = [
        str(term)
        for term in relation_frame.get("core_relation_terms", [])
        if str(term).strip()
    ]
    if not terms:
        return 1.0
    gold_text = normalized_gold_text(qrels, doc_by_id)
    if not gold_text:
        return 0.0
    covered = sum(1 for term in terms if normalize_for_match(term) in gold_text)
    return covered / len(terms)


def normalized_gold_text(qrels: dict[str, int], doc_by_id: dict[str, str]) -> str:
    texts = [doc_by_id.get(doc_id, "") for doc_id, score in qrels.items() if score > 0]
    return normalize_for_match(" ".join(texts))


def expected_slot_present(package: dict[str, object]) -> bool:
    slot = str(package.get("answer_slot", ""))
    views = package.get("retrieval_views", {})
    return bool(slot) and any(slot in str(text) for text in views.values())


def classify_lf_er_failure(
    route_best_ranks: dict[str, int],
    best_reformat_rank: int,
    anchor_coverage: float,
    relation_overlap: float,
) -> str:
    query_rank = route_best_ranks.get("query_only", 999)
    fusion_rank = min(
        (rank for name, rank in route_best_ranks.items() if is_lf_er_fusion_route(name)),
        default=999,
    )
    if fusion_rank <= 10:
        return "ok"
    if anchor_coverage < 0.5:
        return "anchor_not_in_gold"
    if query_rank <= 10 and best_reformat_rank > 10:
        if relation_overlap < 0.2:
            return "reformat_lost_query_context_or_relation_bridge"
        return "reformat_lost_query_context"
    if best_reformat_rank <= 10 and fusion_rank > best_reformat_rank:
        return "fusion_degraded_best_reformat_route"
    if query_rank <= 10 and fusion_rank > query_rank:
        return "reformat_noise"
    if relation_overlap < 0.2 and best_reformat_rank > 10:
        return "relation_frame_or_lexical_bridge_error"
    return "all_routes_failed"


def is_lf_er_fusion_route(name: str) -> bool:
    return name in {
        "lf_er_fusion",
        "lf_er_agreement_fusion",
        "lf_er_inverse_query_fusion",
        "lf_er_anchor_gated_fusion",
        "lf_er_short_expanded_fusion",
        "lf_er_dense_safe_fusion",
    }


def summarize_lf_er_diagnostics(diagnostics_by_query: dict[str, dict]) -> dict[str, object]:
    if not diagnostics_by_query:
        return {}
    failure_counts: dict[str, int] = {}
    oracle_route_counts: dict[str, int] = {}
    best_reformat_route_counts: dict[str, int] = {}
    anchor_coverages = []
    relation_overlaps = []
    oracle_at_10 = 0
    best_reformat_at_10 = 0
    fusion_at_10 = 0
    query_at_10 = 0
    for diagnostic in diagnostics_by_query.values():
        failure = str(diagnostic.get("failure_type", "unknown"))
        failure_counts[failure] = failure_counts.get(failure, 0) + 1
        route = str(diagnostic.get("oracle_best_route", "unknown"))
        oracle_route_counts[route] = oracle_route_counts.get(route, 0) + 1
        reformat_route = str(diagnostic.get("best_reformat_route", "unknown"))
        best_reformat_route_counts[reformat_route] = best_reformat_route_counts.get(reformat_route, 0) + 1
        anchor_coverages.append(float(diagnostic.get("anchor_coverage_in_gold", 0.0)))
        relation_overlaps.append(float(diagnostic.get("relation_term_overlap_in_gold", 0.0)))
        if int(diagnostic.get("oracle_best_rank", 999)) <= 10:
            oracle_at_10 += 1
        if int(diagnostic.get("best_reformat_rank", 999)) <= 10:
            best_reformat_at_10 += 1
        ranks = diagnostic.get("route_best_ranks", {})
        if min((int(rank) for name, rank in ranks.items() if is_lf_er_fusion_route(name)), default=999) <= 10:
            fusion_at_10 += 1
        if int(ranks.get("query_only", 999)) <= 10:
            query_at_10 += 1
    total = len(diagnostics_by_query)
    return {
        "num_queries": total,
        "failure_counts": failure_counts,
        "oracle_route_counts": oracle_route_counts,
        "best_reformat_route_counts": best_reformat_route_counts,
        "avg_anchor_coverage_in_gold": sum(anchor_coverages) / total,
        "avg_relation_term_overlap_in_gold": sum(relation_overlaps) / total,
        "oracle_route_recall_at_10": oracle_at_10 / total,
        "best_reformat_route_recall_at_10": best_reformat_at_10 / total,
        "fusion_recall_at_10": fusion_at_10 / total,
        "query_recall_at_10": query_at_10 / total,
    }


def lf_er_agreement_inputs(
    retriever_name: str,
    query_rank: RankedList,
    route_rankings: dict[str, RankedList],
    bm25_weights: list[float],
    dense_weights: list[float],
) -> tuple[list[RankedList], list[float], list[str]]:
    if retriever_name == "bm25":
        route_names = [
            "query_only",
            "lf_er_relation_keyword_view",
            "lf_er_slotless_evidence_view",
            "lf_er_evidence_inverse_view",
            "lf_er_template_expansion_view",
        ]
        weights = bm25_weights
    else:
        route_names = [
            "query_only",
            "lf_er_anchor_view",
            "lf_er_dense_safe_view",
            "lf_er_dense_safe_expanded_query",
            "lf_er_relation_expanded_query",
        ]
        weights = dense_weights
    rankings = [query_rank]
    rankings.extend(route_rankings[name] for name in route_names[1:])
    return rankings, weights, route_names


def parse_fusion_weights(value: str) -> list[float]:
    return parse_fixed_weights(value, expected=3, flag_name="--answer-blanked-fusion-weights")


def parse_lf_er_fusion_weights(value: str) -> list[float]:
    return parse_fixed_weights(value, expected=6, flag_name="--lf-er-fusion-weights")


def parse_lf_er_agreement_weights(value: str, flag_name: str) -> list[float]:
    return parse_fixed_weights(value, expected=5, flag_name=flag_name)


def parse_lf_er_inverse_query_weights(value: str) -> list[float]:
    return parse_fixed_weights(value, expected=2, flag_name="--lf-er-inverse-query-weights")


def parse_lf_er_anchor_gated_fusion_weights(value: str) -> list[float]:
    return parse_fixed_weights(value, expected=5, flag_name="--lf-er-anchor-gated-fusion-weights")


def parse_lf_er_short_fusion_weights(value: str) -> list[float]:
    return parse_fixed_weights(value, expected=4, flag_name="--lf-er-short-fusion-weights")


def parse_lf_er_dense_safe_fusion_weights(value: str) -> list[float]:
    return parse_fixed_weights(value, expected=4, flag_name="--lf-er-dense-safe-fusion-weights")


def parse_llm_reformat_fusion_weights(value: str) -> list[float]:
    return parse_fixed_weights(value, expected=5, flag_name="--llm-reformat-fusion-weights")


def parse_fixed_weights(value: str, expected: int, flag_name: str) -> list[float]:
    try:
        weights = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid {flag_name}: {value}") from exc
    if len(weights) != expected:
        raise ValueError(f"{flag_name} must contain exactly {expected} numbers")
    if any(weight < 0 for weight in weights):
        raise ValueError(f"{flag_name} must be non-negative")
    if not any(weight > 0 for weight in weights):
        raise ValueError(f"{flag_name} cannot all be zero")
    return weights


if __name__ == "__main__":
    main()
