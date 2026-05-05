from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from expected_answer_rag.datasets import load_dataset
from expected_answer_rag.metrics import evaluate_run, per_query_metrics
from expected_answer_rag.retrieval import make_retriever
from expected_answer_rag.statistics import paired_bootstrap_ci


EXPANSION_FIELDS = {
    "query2doc": "query2doc_document",
    "expected": "expected_answer",
    "hyde": "hyde_document",
    "grf": "generative_relevance_feedback",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generator/corpus cross-regime leakage isolation.")
    parser.add_argument("--public-dataset", required=True)
    parser.add_argument("--cf-dataset", required=True)
    parser.add_argument("--public-max-corpus", type=int, default=None)
    parser.add_argument("--public-max-queries", type=int, default=100)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--public-records", required=True)
    parser.add_argument("--cf-records", required=True)
    parser.add_argument("--retriever", choices=["bm25", "dense", "hybrid", "lexical_neural"], default="bm25")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--embedding-cache-public", default=None)
    parser.add_argument("--embedding-cache-cf", default=None)
    parser.add_argument("--query-prefix", default="Represent this sentence for searching relevant passages: ")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--expansion", choices=sorted(EXPANSION_FIELDS), default="query2doc")
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    run_start = time.perf_counter()
    args = parse_args()
    public_dataset = load_dataset(
        args.public_dataset,
        max_corpus=args.public_max_corpus,
        max_queries=args.public_max_queries,
        cache_dir=args.cache_dir,
    )
    cf_dataset = load_dataset(args.cf_dataset)
    public_records = _load_records(args.public_records)
    cf_records = _load_records(args.cf_records)
    public_retriever = make_retriever(
        args.retriever,
        public_dataset.corpus,
        embedding_model=args.embedding_model,
        embedding_cache=args.embedding_cache_public,
        query_prefix=args.query_prefix,
        local_files_only=args.local_files_only,
    )
    cf_retriever = make_retriever(
        args.retriever,
        cf_dataset.corpus,
        embedding_model=args.embedding_model,
        embedding_cache=args.embedding_cache_cf,
        query_prefix=args.query_prefix,
        local_files_only=args.local_files_only,
    )
    cf_queries = {query.query_id: query.text for query in cf_dataset.queries}
    public_queries = {query.query_id: query.text for query in public_dataset.queries}
    field = EXPANSION_FIELDS[args.expansion]
    query_ids = sorted(set(public_records) & set(cf_records) & set(cf_queries) & set(public_queries))

    runs: dict[str, dict[str, list[tuple[str, float]]]] = {
        "g_pub_on_public": {},
        "q_pub_plus_g_pub_on_public": {},
        "fawe_q_pub_g_pub_on_public": {},
        "g_pub_on_cf": {},
        "g_cf_on_cf": {},
        "q_cf_plus_g_pub_on_cf": {},
        "q_cf_plus_g_cf_on_cf": {},
        "fawe_q_cf_g_pub_on_cf": {},
        "fawe_q_cf_g_cf_on_cf": {},
    }
    records = []
    for qid in query_ids:
        public_query = public_queries[qid]
        cf_query = cf_queries[qid]
        g_pub = _generation_field(public_records[qid], field)
        g_cf = _generation_field(cf_records[qid], field)
        public_query_full = public_retriever.search(public_query, top_k=len(public_dataset.corpus))
        cf_query_full = cf_retriever.search(cf_query, top_k=len(cf_dataset.corpus))
        public_expansion_full = public_retriever.search(g_pub, top_k=len(public_dataset.corpus))
        cf_g_pub_full = cf_retriever.search(g_pub, top_k=len(cf_dataset.corpus))
        cf_g_cf_full = cf_retriever.search(g_cf, top_k=len(cf_dataset.corpus))

        rankings = {
            "g_pub_on_public": public_expansion_full[: args.top_k],
            "q_pub_plus_g_pub_on_public": public_retriever.search(f"{public_query}\n{g_pub}", top_k=args.top_k),
            "fawe_q_pub_g_pub_on_public": _fawe(public_query_full, public_expansion_full, args.beta, args.top_k),
            "g_pub_on_cf": cf_g_pub_full[: args.top_k],
            "g_cf_on_cf": cf_g_cf_full[: args.top_k],
            "q_cf_plus_g_pub_on_cf": cf_retriever.search(f"{cf_query}\n{g_pub}", top_k=args.top_k),
            "q_cf_plus_g_cf_on_cf": cf_retriever.search(f"{cf_query}\n{g_cf}", top_k=args.top_k),
            "fawe_q_cf_g_pub_on_cf": _fawe(cf_query_full, cf_g_pub_full, args.beta, args.top_k),
            "fawe_q_cf_g_cf_on_cf": _fawe(cf_query_full, cf_g_cf_full, args.beta, args.top_k),
        }
        for name, ranking in rankings.items():
            runs[name][qid] = ranking
        records.append(
            {
                "query_id": qid,
                "public_query": public_query,
                "counterfactual_query": cf_query,
                "public_expansion": g_pub,
                "counterfactual_expansion": g_cf,
                "rankings": rankings,
            }
        )

    metrics = {}
    error_bars = {}
    for name, run in runs.items():
        qrels = public_dataset.qrels if name.endswith("_public") else cf_dataset.qrels
        metrics[name] = evaluate_run(run, qrels, ks=(5, 10, 20, 100))
        per_query = [per_query_metrics(ranking, qrels.get(qid, {}), ks=(5, 10, 20, 100)) for qid, ranking in run.items()]
        error_bars[name] = {
            metric: paired_bootstrap_ci([row[metric] for row in per_query], num_samples=500)
            for metric in per_query[0]
        } if per_query else {}

    result = {
        "public_dataset": args.public_dataset,
        "cf_dataset": args.cf_dataset,
        "retriever": args.retriever,
        "embedding_model": args.embedding_model if args.retriever != "bm25" else None,
        "expansion": args.expansion,
        "beta": args.beta,
        "top_k": args.top_k,
        "num_queries": len(query_ids),
        "runtime_seconds": round(time.perf_counter() - run_start, 3),
        "metrics": metrics,
        "metric_error_bars": error_bars,
        "records": records[:20],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {output}")


def _load_records(path: str) -> dict[str, Mapping[str, object]]:
    records = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[str(record["query_id"])] = record
    return records


def _generation_field(record: Mapping[str, object], field: str) -> str:
    generation = record.get("generation", {})
    if not isinstance(generation, Mapping):
        return ""
    return str(generation.get(field) or "")


def _fawe(
    query_ranking: list[tuple[str, float]],
    expansion_ranking: list[tuple[str, float]],
    beta: float,
    top_k: int,
) -> list[tuple[str, float]]:
    scores = {doc_id: float(score) for doc_id, score in query_ranking}
    for doc_id, score in expansion_ranking:
        scores[doc_id] = scores.get(doc_id, 0.0) + beta * float(score)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]


if __name__ == "__main__":
    main()
