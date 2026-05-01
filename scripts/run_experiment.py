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
from expected_answer_rag.fusion import reciprocal_rank_fusion
from expected_answer_rag.generators import CachedTextGenerator, HeuristicGenerator, OpenAITextGenerator
from expected_answer_rag.metrics import evaluate_run
from expected_answer_rag.retrieval import RankedList, make_retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run expected-answer RAG retrieval baselines.")
    parser.add_argument("--dataset", default="toy", help="toy, nq, hotpotqa, fiqa, scifact, ...")
    parser.add_argument("--max-corpus", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--retriever", choices=["bm25", "dense"], default="bm25")
    parser.add_argument("--generator", choices=["heuristic", "openai"], default="heuristic")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--generation-cache", default="outputs/generation_cache.json")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", default="outputs/run.json")
    parser.add_argument("--records-output", default="outputs/records.jsonl")
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
    retriever = make_retriever(args.retriever, dataset.corpus)
    if args.generator == "openai":
        base_generator = OpenAITextGenerator(
            model=args.model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )
    else:
        base_generator = HeuristicGenerator()
    cache_path = _resolve_path(args.generation_cache)
    generator = CachedTextGenerator(
        inner=base_generator,
        cache=JsonCache(cache_path),
        namespace=f"{args.generator}:{args.model}:temp={args.temperature}",
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
    generations = {}
    features_by_query = {}
    records = []

    for query in tqdm(dataset.queries, desc="Running queries"):
        expected = generator.expected_answer(query.text)
        masked = generator.mask_answer(expected)
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
        "generator": args.generator,
        "model": args.model if args.generator == "openai" else None,
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


if __name__ == "__main__":
    main()
