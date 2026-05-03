from __future__ import annotations

import re
from typing import Dict, Iterable, Mapping

from expected_answer_rag.datasets import Query
from expected_answer_rag.leakage import contains_any_alias, leakage_bucket_name, summarize_leakage_scores
from expected_answer_rag.metrics import evaluate_run
from expected_answer_rag.retrieval import RankedList


CAPITALIZED_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b")
SLOT_RE = re.compile(r"\[[A-Z_]+\]")


def generation_features(query: Query, expected: str, masked: str, hyde_doc: str) -> Dict[str, object]:
    return {
        "contains_gold_answer": contains_any_alias(expected, query.answers),
        "contains_answer_alias": contains_any_alias(expected, query.all_answer_strings),
        "masked_contains_gold_answer": contains_any_alias(masked, query.answers),
        "masked_contains_answer_alias": contains_any_alias(masked, query.all_answer_strings),
        "expected_token_count": len(expected.split()),
        "hyde_token_count": len(hyde_doc.split()),
        "expected_capitalized_span_count": len(extract_capitalized_spans(expected)),
        "hyde_capitalized_span_count": len(extract_capitalized_spans(hyde_doc)),
        "mask_slot_count": len(SLOT_RE.findall(masked)),
    }


def extract_capitalized_spans(text: str) -> list[str]:
    keep = {"The", "A", "An", "Question"}
    return [span for span in CAPITALIZED_RE.findall(text) if span not in keep]


def summarize_generation_features(records: Iterable[Mapping[str, object]]) -> Dict[str, float]:
    rows = list(records)
    if not rows:
        return {}
    summary: Dict[str, float] = {}
    numeric_keys = [
        "expected_token_count",
        "hyde_token_count",
        "expected_capitalized_span_count",
        "hyde_capitalized_span_count",
        "mask_slot_count",
    ]
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        summary[f"avg_{key}"] = sum(values) / len(values) if values else 0.0

    for key in [
        "contains_gold_answer",
        "contains_answer_alias",
        "masked_contains_gold_answer",
        "masked_contains_answer_alias",
    ]:
        values = [row.get(key) for row in rows if row.get(key) is not None]
        if values:
            summary[f"rate_{key}"] = sum(1 for value in values if value) / len(values)
    return summary


def evaluate_by_leakage_bucket(
    run: Mapping[str, RankedList],
    qrels: Mapping[str, Mapping[str, int]],
    features_by_query: Mapping[str, Mapping[str, object]],
) -> Dict[str, Dict[str, float]]:
    buckets: dict[str, set[str]] = {}
    for qid, features in features_by_query.items():
        bucket = str(features.get("leakage_bucket") or leakage_bucket_name(features))
        buckets.setdefault(bucket, set()).add(qid)

    results: Dict[str, Dict[str, float]] = {}
    for bucket, qids in buckets.items():
        bucket_run = {qid: ranking for qid, ranking in run.items() if qid in qids}
        bucket_qrels = {qid: rels for qid, rels in qrels.items() if qid in qids}
        if bucket_run and bucket_qrels:
            results[bucket] = evaluate_run(bucket_run, bucket_qrels)
    return results


def compare_methods(metrics: Mapping[str, Mapping[str, float]], primary_metric: str = "ndcg@10") -> list[dict[str, object]]:
    rows = []
    baseline = metrics.get("query_only", {}).get(primary_metric, 0.0)
    for method, values in metrics.items():
        score = values.get(primary_metric, 0.0)
        rows.append(
            {
                "method": method,
                primary_metric: score,
                "delta_vs_query_only": score - baseline,
            }
        )
    return sorted(rows, key=lambda row: float(row[primary_metric]), reverse=True)


def summarize_method_leakage(leakage_records: Mapping[str, Iterable[Mapping[str, object]]]) -> dict[str, dict[str, float]]:
    return {
        method: summarize_leakage_scores(records)
        for method, records in leakage_records.items()
    }
