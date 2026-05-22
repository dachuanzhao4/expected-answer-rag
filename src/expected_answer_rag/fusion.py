from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

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


def agreement_weighted_reciprocal_rank_fusion(
    rankings: Sequence[RankedList],
    weights: Sequence[float],
    query_index: int = 0,
    agreement_depth: int = 5,
    min_route_multiplier: float = 0.2,
    max_route_multiplier: float = 1.0,
    rrf_k: int = 60,
    top_k: int = 10,
) -> RankedList:
    adjusted = agreement_adjusted_weights(
        rankings=rankings,
        weights=weights,
        query_index=query_index,
        agreement_depth=agreement_depth,
        min_route_multiplier=min_route_multiplier,
        max_route_multiplier=max_route_multiplier,
    )
    return weighted_reciprocal_rank_fusion(rankings, adjusted, rrf_k=rrf_k, top_k=top_k)


def agreement_adjusted_weights(
    rankings: Sequence[RankedList],
    weights: Sequence[float],
    query_index: int = 0,
    agreement_depth: int = 5,
    min_route_multiplier: float = 0.2,
    max_route_multiplier: float = 1.0,
) -> list[float]:
    if len(rankings) != len(weights):
        raise ValueError("rankings and weights must have the same length")
    if not rankings:
        return []
    if query_index < 0 or query_index >= len(rankings):
        raise ValueError("query_index is out of range")
    top_sets = [_top_doc_set(ranking, agreement_depth) for ranking in rankings]
    query_docs = top_sets[query_index]
    adjusted: list[float] = []
    for index, (weight, docs) in enumerate(zip(weights, top_sets)):
        if index == query_index or weight == 0:
            adjusted.append(weight)
            continue
        query_agreement = overlap_fraction(docs, query_docs)
        peer_agreement = max(
            (
                overlap_fraction(docs, other_docs)
                for other_index, other_docs in enumerate(top_sets)
                if other_index not in {index, query_index}
            ),
            default=0.0,
        )
        agreement = max(query_agreement, 0.5 * peer_agreement)
        multiplier = min_route_multiplier + (max_route_multiplier - min_route_multiplier) * agreement
        adjusted.append(weight * multiplier)
    return adjusted


def overlap_fraction(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _top_doc_set(ranking: RankedList, depth: int) -> set[str]:
    return {doc_id for doc_id, _score in ranking[: max(depth, 1)]}
