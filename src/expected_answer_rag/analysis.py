from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, Mapping

from expected_answer_rag.datasets import Query
from expected_answer_rag.metrics import evaluate_run
from expected_answer_rag.retrieval import RankedList


CAPITALIZED_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b")
SLOT_RE = re.compile(r"\[[A-Z_]+\]")


def generation_features(query: Query, expected: str, masked: str, hyde_doc: str) -> Dict[str, object]:
    return {
        "contains_gold_answer": contains_any_answer(expected, query.answers),
        "masked_contains_gold_answer": contains_any_answer(masked, query.answers),
        "expected_token_count": len(expected.split()),
        "hyde_token_count": len(hyde_doc.split()),
        "expected_capitalized_span_count": len(extract_capitalized_spans(expected)),
        "hyde_capitalized_span_count": len(extract_capitalized_spans(hyde_doc)),
        "mask_slot_count": len(SLOT_RE.findall(masked)),
    }


def contains_any_answer(text: str, answers: Iterable[str]) -> bool | None:
    normalized_text = normalize_for_match(text)
    answer_list = [answer for answer in answers if answer]
    if not answer_list:
        return None
    return any(normalize_for_match(answer) in normalized_text for answer in answer_list)


def extract_capitalized_spans(text: str) -> list[str]:
    keep = {"The", "A", "An", "Question"}
    return [span for span in CAPITALIZED_RE.findall(text) if span not in keep]


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


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

    for key in ["contains_gold_answer", "masked_contains_gold_answer"]:
        values = [row.get(key) for row in rows if row.get(key) is not None]
        if values:
            summary[f"rate_{key}"] = sum(1 for value in values if value) / len(values)
    return summary


def evaluate_by_leakage_bucket(
    run: Mapping[str, RankedList],
    qrels: Mapping[str, Mapping[str, int]],
    features_by_query: Mapping[str, Mapping[str, object]],
) -> Dict[str, Dict[str, float]]:
    buckets = {
        "expected_contains_gold": set(),
        "expected_not_contains_gold": set(),
        "unknown_gold_answer": set(),
    }
    for qid, features in features_by_query.items():
        value = features.get("contains_gold_answer")
        if value is True:
            buckets["expected_contains_gold"].add(qid)
        elif value is False:
            buckets["expected_not_contains_gold"].add(qid)
        else:
            buckets["unknown_gold_answer"].add(qid)

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
