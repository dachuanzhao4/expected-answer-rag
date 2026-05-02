from __future__ import annotations

import argparse
import json
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
from expected_answer_rag.cache import JsonCache
from expected_answer_rag.datasets import load_dataset
from expected_answer_rag.fusion import reciprocal_rank_fusion, weighted_reciprocal_rank_fusion
from expected_answer_rag.generators import CachedTextGenerator, HeuristicGenerator, MissingGenerator, OpenAITextGenerator
from expected_answer_rag.metrics import evaluate_run
from expected_answer_rag.retrieval import RankedList, make_retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run expected-answer RAG retrieval baselines.")
    parser.add_argument("--dataset", default="toy", help="toy, nq, hotpotqa, fiqa, scifact, ...")
    parser.add_argument("--max-corpus", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--retriever", choices=["bm25", "dense"], default="bm25")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--embedding-chunk-size", type=int, default=1024)
    parser.add_argument("--embedding-cache", default=None)
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
        )
    else:
        base_generator = HeuristicGenerator()
    cache_path = _resolve_path(args.generation_cache)
    if args.clear_generation_cache and cache_path.exists():
        cache_path.unlink()
    generator = CachedTextGenerator(
        inner=base_generator,
        cache=JsonCache(cache_path),
        namespace=args.cache_namespace or f"{args.generator}:{args.model}:temp={args.temperature}",
    )

    runs: Dict[str, Dict[str, RankedList]] = {
        "query_only": {},
        "hyde_doc_only": {},
        "raw_expected_answer_only": {},
        "masked_expected_answer_only": {},
        "concat_query_raw_expected": {},
        "concat_query_masked_expected": {},
        "dual_query_raw_expected_rrf": {},
        "dual_query_masked_expected_rrf": {},
    }
    answer_rrf_weights = _parse_float_list(args.answer_rrf_weights)
    for weight in answer_rrf_weights:
        suffix = _weight_suffix(weight)
        runs[f"weighted_dual_query_raw_expected_rrf_w{suffix}"] = {}
        runs[f"weighted_dual_query_masked_expected_rrf_w{suffix}"] = {}
    generations = {}
    features_by_query = {}
    records = []

    for query in tqdm(dataset.queries, desc="Running queries"):
        expected = generator.expected_answer(query.text)
        masked = generator.mask_answer(query.text, expected)
        hyde_doc = generator.hyde_document(query.text)
        features = generation_features(query, expected, masked, hyde_doc)
        features_by_query[query.query_id] = features
        generations[query.query_id] = {
            "query": query.text,
            "answers": list(query.answers),
            "expected_answer": expected,
            "masked_expected_answer": masked,
            "hyde_document": hyde_doc,
            "features": features,
        }

        query_rank = retriever.search(query.text, top_k=args.top_k)
        raw_rank = retriever.search(expected, top_k=args.top_k)
        masked_rank = retriever.search(masked, top_k=args.top_k)

        runs["query_only"][query.query_id] = query_rank
        runs["hyde_doc_only"][query.query_id] = retriever.search(hyde_doc, top_k=args.top_k)
        runs["raw_expected_answer_only"][query.query_id] = raw_rank
        runs["masked_expected_answer_only"][query.query_id] = masked_rank
        runs["concat_query_raw_expected"][query.query_id] = retriever.search(
            f"{query.text}\n{expected}",
            top_k=args.top_k,
        )
        runs["concat_query_masked_expected"][query.query_id] = retriever.search(
            f"{query.text}\n{masked}",
            top_k=args.top_k,
        )
        runs["dual_query_raw_expected_rrf"][query.query_id] = reciprocal_rank_fusion(
            [query_rank, raw_rank],
            top_k=args.top_k,
        )
        runs["dual_query_masked_expected_rrf"][query.query_id] = reciprocal_rank_fusion(
            [query_rank, masked_rank],
            top_k=args.top_k,
        )
        for weight in answer_rrf_weights:
            suffix = _weight_suffix(weight)
            runs[f"weighted_dual_query_raw_expected_rrf_w{suffix}"][query.query_id] = weighted_reciprocal_rank_fusion(
                [query_rank, raw_rank],
                [1.0, weight],
                top_k=args.top_k,
            )
            runs[f"weighted_dual_query_masked_expected_rrf_w{suffix}"][query.query_id] = weighted_reciprocal_rank_fusion(
                [query_rank, masked_rank],
                [1.0, weight],
                top_k=args.top_k,
            )
        records.append(
            {
                "query_id": query.query_id,
                "query": query.text,
                "answers": list(query.answers),
                "generation": generations[query.query_id],
                "rankings": {name: runs[name][query.query_id] for name in runs},
                "qrels": dataset.qrels.get(query.query_id, {}),
            }
        )

    metrics = {name: evaluate_run(run, dataset.qrels) for name, run in runs.items()}
    leakage_metrics = {
        name: evaluate_by_leakage_bucket(run, dataset.qrels, features_by_query)
        for name, run in runs.items()
    }
    result = {
        "dataset": dataset.name,
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
        "top_k": args.top_k,
        "metrics": metrics,
        "method_ranking": compare_methods(metrics),
        "generation_summary": summarize_generation_features(features_by_query.values()),
        "leakage_bucket_metrics": leakage_metrics,
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


if __name__ == "__main__":
    main()
