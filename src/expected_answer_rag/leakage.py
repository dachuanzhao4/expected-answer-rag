from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from expected_answer_rag.datasets import Document, Query


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
CAPITALIZED_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}\b")
QUOTED_TITLE_RE = re.compile(r'"([^"\n]{2,120})"')
DATE_RE = re.compile(
    r"\b(?:\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"January|February|March|April|May|June|July|August|September|October|November|December)\b",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
SLOT_RE = re.compile(r"\[[A-Z_]+\]")


@dataclass(frozen=True)
class LeakageScore:
    exact_answer_leakage: bool | None
    alias_answer_leakage: bool | None
    answer_candidate_leakage: bool
    evidence_support_overlap: float
    evidence_entailment_heuristic: bool
    introduced_candidates: tuple[str, ...]
    unsupported_candidates: tuple[str, ...]
    wrong_prior_candidates: tuple[str, ...]
    bridge_candidates: tuple[str, ...]
    masked_slot_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "exact_answer_leakage": self.exact_answer_leakage,
            "alias_answer_leakage": self.alias_answer_leakage,
            "answer_candidate_leakage": self.answer_candidate_leakage,
            "evidence_support_overlap": self.evidence_support_overlap,
            "evidence_entailment_heuristic": self.evidence_entailment_heuristic,
            "introduced_candidates": list(self.introduced_candidates),
            "unsupported_candidates": list(self.unsupported_candidates),
            "wrong_prior_candidates": list(self.wrong_prior_candidates),
            "bridge_candidates": list(self.bridge_candidates),
            "masked_slot_count": self.masked_slot_count,
        }


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def contains_any_alias(text: str, aliases: Iterable[str]) -> bool | None:
    normalized_text = normalize_text(text)
    alias_values = [alias for alias in aliases if alias]
    if not alias_values:
        return None
    return any(normalize_text(alias) in normalized_text for alias in alias_values)


def extract_concrete_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    seen = set()
    for match in QUOTED_TITLE_RE.findall(text):
        value = match.strip()
        if value and value.lower() not in seen:
            seen.add(value.lower())
            candidates.append(value)
    for match in CAPITALIZED_RE.findall(text):
        value = match.strip()
        if value in {"The", "A", "An"}:
            continue
        if value.lower() not in seen:
            seen.add(value.lower())
            candidates.append(value)
    for pattern in [DATE_RE, NUMBER_RE]:
        for match in pattern.findall(text):
            value = match.strip()
            if value and value.lower() not in seen:
                seen.add(value.lower())
                candidates.append(value)
    return candidates


def extract_query_anchors(text: str) -> set[str]:
    anchors = {normalize_text(match) for match in CAPITALIZED_RE.findall(text)}
    anchors.update(normalize_text(match) for match in QUOTED_TITLE_RE.findall(text))
    anchors.update(tokenize(text))
    anchors.discard("")
    return anchors


def support_overlap_ratio(text: str, evidence_texts: Sequence[str]) -> float:
    text_tokens = set(tokenize(text))
    if not text_tokens:
        return 0.0
    best = 0.0
    for evidence in evidence_texts:
        evidence_tokens = set(tokenize(evidence))
        if not evidence_tokens:
            continue
        overlap = len(text_tokens & evidence_tokens) / max(len(text_tokens), 1)
        best = max(best, overlap)
    return best


def evidence_entailment_heuristic(text: str, evidence_texts: Sequence[str]) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    for evidence in evidence_texts:
        evidence_normalized = normalize_text(evidence)
        if not evidence_normalized:
            continue
        if normalized in evidence_normalized:
            return True
        overlap = support_overlap_ratio(text, [evidence])
        if overlap >= 0.75:
            return True
    return False


def introduced_candidates(query_text: str, expansion_text: str) -> tuple[str, ...]:
    query_anchors = extract_query_anchors(query_text)
    values = []
    seen = set()
    for candidate in extract_concrete_candidates(expansion_text):
        key = normalize_text(candidate)
        if not key or key in query_anchors or key in seen:
            continue
        seen.add(key)
        values.append(candidate)
    return tuple(values)


def score_expansion(
    query: Query,
    expansion_text: str,
    relevant_documents: Sequence[Document],
) -> LeakageScore:
    evidence_texts = [doc.text for doc in relevant_documents]
    exact_answer = contains_any_alias(expansion_text, query.answers)
    alias_answer = contains_any_alias(expansion_text, query.all_answer_strings)
    introduced = introduced_candidates(query.text, expansion_text)
    relevant_candidate_space = {
        normalize_text(candidate)
        for text in [*evidence_texts, *query.all_answer_strings]
        for candidate in extract_concrete_candidates(text)
    }
    relevant_candidate_space.update(normalize_text(value) for value in query.all_answer_strings)
    unsupported = tuple(
        candidate
        for candidate in introduced
        if normalize_text(candidate) not in relevant_candidate_space
    )
    wrong_prior = tuple(
        candidate
        for candidate in introduced
        if normalize_text(candidate) not in {normalize_text(value) for value in query.all_answer_strings}
        and normalize_text(candidate) not in {normalize_text(candidate) for text in evidence_texts for candidate in extract_concrete_candidates(text)}
    )
    bridge_values = tuple(
        candidate
        for candidate in introduced
        if normalize_text(candidate) in {
            normalize_text(candidate)
            for text in evidence_texts
            for candidate in extract_concrete_candidates(text)
        }
        and normalize_text(candidate) not in {normalize_text(value) for value in query.all_answer_strings}
    )
    overlap = support_overlap_ratio(expansion_text, evidence_texts)
    entailment = evidence_entailment_heuristic(expansion_text, evidence_texts)
    return LeakageScore(
        exact_answer_leakage=exact_answer,
        alias_answer_leakage=alias_answer,
        answer_candidate_leakage=bool(introduced),
        evidence_support_overlap=overlap,
        evidence_entailment_heuristic=entailment,
        introduced_candidates=introduced,
        unsupported_candidates=unsupported,
        wrong_prior_candidates=wrong_prior,
        bridge_candidates=bridge_values,
        masked_slot_count=len(SLOT_RE.findall(expansion_text)),
    )


def summarize_leakage_scores(scores: Iterable[Mapping[str, object]]) -> dict[str, float]:
    rows = list(scores)
    if not rows:
        return {}
    summary: dict[str, float] = {}
    boolean_keys = [
        "exact_answer_leakage",
        "alias_answer_leakage",
        "answer_candidate_leakage",
        "evidence_entailment_heuristic",
    ]
    for key in boolean_keys:
        values = [row.get(key) for row in rows if row.get(key) is not None]
        if values:
            summary[f"rate_{key}"] = sum(1 for value in values if value) / len(values)
    numeric_keys = ["evidence_support_overlap", "masked_slot_count"]
    for key in numeric_keys:
        values = [float(row.get(key, 0.0)) for row in rows if row.get(key) is not None]
        if values:
            summary[f"avg_{key}"] = sum(values) / len(values)
    count_keys = ["introduced_candidates", "unsupported_candidates", "wrong_prior_candidates", "bridge_candidates"]
    for key in count_keys:
        values = [len(row.get(key, [])) for row in rows if row.get(key) is not None]
        if values:
            summary[f"avg_{key}_count"] = sum(values) / len(values)
    return summary


def leakage_bucket_name(score: Mapping[str, object]) -> str:
    if score.get("exact_answer_leakage"):
        return "exact_answer_leakage"
    if score.get("alias_answer_leakage"):
        return "alias_answer_leakage"
    if score.get("wrong_prior_candidates"):
        return "wrong_prior_injection"
    if score.get("unsupported_candidates"):
        return "unsupported_injection"
    if score.get("evidence_entailment_heuristic"):
        return "evidence_entailment"
    return "leakage_negative"


def score_generation_methods(
    query: Query,
    generations: Mapping[str, str],
    relevant_documents: Sequence[Document],
) -> dict[str, dict[str, object]]:
    return {
        method: score_expansion(query, text, relevant_documents).to_dict()
        for method, text in generations.items()
    }
