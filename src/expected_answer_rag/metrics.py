from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Tuple

from expected_answer_rag.retrieval import RankedList


def evaluate_run(
    run: Mapping[str, RankedList],
    qrels: Mapping[str, Mapping[str, int]],
    ks: Iterable[int] = (5, 10, 20),
) -> Dict[str, float]:
    query_ids = [qid for qid in run if qrels.get(qid)]
    if not query_ids:
        return {f"recall@{k}": 0.0 for k in ks} | {"mrr@10": 0.0, "ndcg@10": 0.0}

    metrics: Dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}"] = sum(_recall_at_k(run[qid], qrels[qid], k) for qid in query_ids) / len(query_ids)
    metrics["mrr@10"] = sum(_mrr_at_k(run[qid], qrels[qid], 10) for qid in query_ids) / len(query_ids)
    metrics["ndcg@10"] = sum(_ndcg_at_k(run[qid], qrels[qid], 10) for qid in query_ids) / len(query_ids)
    return metrics


def _recall_at_k(ranking: RankedList, rels: Mapping[str, int], k: int) -> float:
    relevant = {doc_id for doc_id, score in rels.items() if score > 0}
    if not relevant:
        return 0.0
    retrieved = {doc_id for doc_id, _score in ranking[:k]}
    return len(relevant & retrieved) / len(relevant)


def _mrr_at_k(ranking: RankedList, rels: Mapping[str, int], k: int) -> float:
    for rank, (doc_id, _score) in enumerate(ranking[:k], start=1):
        if rels.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(ranking: RankedList, rels: Mapping[str, int], k: int) -> float:
    dcg = 0.0
    for rank, (doc_id, _score) in enumerate(ranking[:k], start=1):
        gain = rels.get(doc_id, 0)
        if gain > 0:
            dcg += gain / math.log2(rank + 1)

    ideal_gains = sorted((score for score in rels.values() if score > 0), reverse=True)[:k]
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal_gains, start=1))
    if idcg == 0:
        return 0.0
    return dcg / idcg
