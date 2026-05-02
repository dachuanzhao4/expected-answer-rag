from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List, Sequence, Tuple

from expected_answer_rag.retrieval import RankedList


def reciprocal_rank_fusion(rankings: Iterable[RankedList], rrf_k: int = 60, top_k: int = 10) -> RankedList:
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]


def weighted_reciprocal_rank_fusion(
    rankings: Sequence[RankedList],
    weights: Sequence[float],
    rrf_k: int = 60,
    top_k: int = 10,
) -> RankedList:
    if len(rankings) != len(weights):
        raise ValueError("rankings and weights must have the same length")
    scores = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            scores[doc_id] += weight / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
