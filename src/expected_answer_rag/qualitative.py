from __future__ import annotations

from typing import Iterable, Mapping


def select_qualitative_examples(
    records: Iterable[Mapping[str, object]],
    method: str,
    counterfactual_method: str | None = None,
    limit: int = 5,
) -> list[dict[str, object]]:
    candidates = []
    for record in records:
        generation = record.get("generation", {})
        leakage = generation.get("leakage", {}).get(method, {})
        if not leakage:
            continue
        exact = bool(leakage.get("exact_answer_leakage"))
        unsupported = bool(leakage.get("unsupported_candidates"))
        if not exact and not unsupported:
            continue
        rankings = record.get("rankings", {})
        method_rank = rankings.get(method, [])
        query_rank = rankings.get("query_only", [])
        method_top = method_rank[0][0] if method_rank else None
        query_top = query_rank[0][0] if query_rank else None
        score = int(exact) * 2 + int(unsupported)
        candidates.append(
            {
                "query_id": record.get("query_id"),
                "query": record.get("query"),
                "answers": record.get("answers"),
                "method": method,
                "generation_text": generation.get(method),
                "leakage": leakage,
                "method_top_doc": method_top,
                "query_top_doc": query_top,
                "score": score,
            }
        )
    candidates.sort(key=lambda row: row["score"], reverse=True)
    return candidates[:limit]
