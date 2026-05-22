from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


RUN_FILES = {
    "public_nq_bm25": Path("results/public_nq/nq_stage2_bm25_formal_query2doc_mask_run.json"),
    "public_nq_dense": Path("results/public_nq/nq_stage2_dense_bge_base_formal_query2doc_mask_run.json"),
    "opaque_bm25": Path(
        "results/renamed_private_like/final_v2/"
        "renamed_nq_stage2_token_v5_full_llm_reformat_v2_opaque_bm25_formal_query2doc_mask_run.json"
    ),
    "opaque_dense": Path(
        "results/renamed_private_like/final_v2/"
        "renamed_nq_stage2_token_v5_full_llm_reformat_v2_opaque_dense_formal_query2doc_mask_run.json"
    ),
    "plausible_bm25": Path(
        "results/renamed_private_like/final_v2/"
        "renamed_nq_stage2_token_v5_full_llm_reformat_v2_plausible_bm25_formal_query2doc_mask_run.json"
    ),
    "plausible_dense": Path(
        "results/renamed_private_like/final_v2/"
        "renamed_nq_stage2_token_v5_full_llm_reformat_v2_plausible_dense_formal_query2doc_mask_run.json"
    ),
}

RECORD_FILES = {
    "opaque_bm25": Path(
        "results/renamed_private_like/final_v2/"
        "renamed_nq_stage2_token_v5_full_llm_reformat_v2_opaque_bm25_formal_query2doc_mask_records.jsonl"
    ),
    "opaque_dense": Path(
        "results/renamed_private_like/final_v2/"
        "renamed_nq_stage2_token_v5_full_llm_reformat_v2_opaque_dense_formal_query2doc_mask_records.jsonl"
    ),
    "plausible_bm25": Path(
        "results/renamed_private_like/final_v2/"
        "renamed_nq_stage2_token_v5_full_llm_reformat_v2_plausible_bm25_formal_query2doc_mask_records.jsonl"
    ),
    "plausible_dense": Path(
        "results/renamed_private_like/final_v2/"
        "renamed_nq_stage2_token_v5_full_llm_reformat_v2_plausible_dense_formal_query2doc_mask_records.jsonl"
    ),
}

METHODS = [
    "query_only",
    "query2doc_expanded_query",
    "masked_query2doc_expanded_query",
    "llm_reformat_fusion",
    "llm_reformat_anchor_view",
    "llm_reformat_dense_view",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def replacement_tokens(root: Path, mode: str) -> set[str]:
    mapping_path = root / "data_metadata" / "renamed_nq_stage2_token_v5_full" / mode / "mapping.json"
    mapping = load_json(mapping_path)
    tokens = {entry["replacement"] for entry in mapping.values()}
    return {token.lower() for token in tokens}


def changed_query_ids(root: Path, mode: str) -> set[str]:
    replacements = replacement_tokens(root, mode)
    query_path = root / "data_metadata" / "renamed_nq_stage2_token_v5_full" / mode / "queries.jsonl"
    changed: set[str] = set()
    token_re = re.compile(r"[A-Za-z0-9_]+")
    for item in load_jsonl(query_path):
        query_tokens = {token.lower() for token in token_re.findall(item["text"])}
        if query_tokens & replacements:
            changed.add(item["_id"])
    return changed


def dcg(relevances: list[float]) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def evaluate_records(records: list[dict[str, Any]], method: str) -> dict[str, float]:
    recall5 = []
    recall10 = []
    recall20 = []
    mrr10 = []
    ndcg10 = []
    for record in records:
        qrels = record.get("qrels", {})
        rankings = record.get("rankings", {}).get(method, [])
        relevant = {doc_id: float(rel) for doc_id, rel in qrels.items() if float(rel) > 0}
        if not relevant:
            continue
        docs = [doc_id for doc_id, _score in rankings]
        relevant_count = len(relevant)
        recall5.append(len(set(docs[:5]) & set(relevant)) / relevant_count)
        recall10.append(len(set(docs[:10]) & set(relevant)) / relevant_count)
        recall20.append(len(set(docs[:20]) & set(relevant)) / relevant_count)
        reciprocal = 0.0
        for rank, doc_id in enumerate(docs[:10], start=1):
            if doc_id in relevant:
                reciprocal = 1.0 / rank
                break
        mrr10.append(reciprocal)
        ranked_rels = [relevant.get(doc_id, 0.0) for doc_id in docs[:10]]
        ideal_rels = sorted(relevant.values(), reverse=True)[:10]
        ideal = dcg(ideal_rels)
        ndcg10.append(0.0 if ideal == 0 else dcg(ranked_rels) / ideal)

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "recall@5": mean(recall5),
        "recall@10": mean(recall10),
        "recall@20": mean(recall20),
        "mrr@10": mean(mrr10),
        "ndcg@10": mean(ndcg10),
    }


def pick_metrics(metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
    picked: dict[str, dict[str, float]] = {}
    for method in METHODS:
        if method in metrics:
            picked[method] = {
                key: round(float(metrics[method][key]), 6)
                for key in ("recall@10", "mrr@10", "ndcg@10")
                if key in metrics[method]
            }
    return picked


def subset_summary(root: Path, mode: str, retriever: str) -> dict[str, Any]:
    key = f"{mode}_{retriever}"
    records = load_jsonl(root / RECORD_FILES[key])
    changed = changed_query_ids(root, mode)
    changed_records = [record for record in records if record["query_id"] in changed]
    unchanged_records = [record for record in records if record["query_id"] not in changed]
    result: dict[str, Any] = {
        "num_records": len(records),
        "num_changed": len(changed_records),
        "num_unchanged": len(unchanged_records),
        "changed": {},
        "unchanged": {},
    }
    available_methods = [method for method in METHODS if any(method in record.get("rankings", {}) for record in records)]
    for method in available_methods:
        result["changed"][method] = {
            key: round(value, 6)
            for key, value in evaluate_records(changed_records, method).items()
            if key in {"recall@10", "mrr@10", "ndcg@10"}
        }
        result["unchanged"][method] = {
            key: round(value, 6)
            for key, value in evaluate_records(unchanged_records, method).items()
            if key in {"recall@10", "mrr@10", "ndcg@10"}
        }
    return result


def build_summary(root: Path) -> dict[str, Any]:
    runs = {}
    for key, rel_path in RUN_FILES.items():
        run = load_json(root / rel_path)
        runs[key] = {
            "dataset": run.get("dataset"),
            "retriever": run.get("retriever"),
            "num_queries": run.get("num_queries"),
            "metrics": pick_metrics(run["metrics"]),
        }

    private_subsets = {}
    for mode in ("opaque", "plausible"):
        for retriever in ("bm25", "dense"):
            private_subsets[f"{mode}_{retriever}"] = subset_summary(root, mode, retriever)

    return {
        "notes": {
            "primary_metric": "ndcg@10",
            "private_effective_subset": "queries whose text contains at least one corpus-aligned replacement token",
            "final_method": "llm_reformat_v2",
        },
        "runs": runs,
        "private_subsets": private_subsets,
    }


def print_markdown(summary: dict[str, Any]) -> None:
    print("# Final result summary\n")
    print("## Full 500-query runs")
    print("| setting | query | query2doc | masked q2doc | llm reformat fusion |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for key, run in summary["runs"].items():
        metrics = run["metrics"]
        values = [
            metrics.get("query_only", {}).get("ndcg@10"),
            metrics.get("query2doc_expanded_query", {}).get("ndcg@10"),
            metrics.get("masked_query2doc_expanded_query", {}).get("ndcg@10"),
            metrics.get("llm_reformat_fusion", {}).get("ndcg@10"),
        ]
        cells = ["" if value is None else f"{value:.4f}" for value in values]
        print(f"| {key} | {' | '.join(cells)} |")

    print("\n## Private-effective changed subset")
    print("| setting | n | query | query2doc | masked q2doc | llm reformat fusion |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for key, subset in summary["private_subsets"].items():
        changed = subset["changed"]
        values = [
            changed.get("query_only", {}).get("ndcg@10"),
            changed.get("query2doc_expanded_query", {}).get("ndcg@10"),
            changed.get("masked_query2doc_expanded_query", {}).get("ndcg@10"),
            changed.get("llm_reformat_fusion", {}).get("ndcg@10"),
        ]
        cells = ["" if value is None else f"{value:.4f}" for value in values]
        print(f"| {key} | {subset['num_changed']} | {' | '.join(cells)} |")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    summary = build_summary(root)
    print_markdown(summary)
    if args.output is not None:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
