from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol

from expected_answer_rag.cache import JsonCache


class TextGenerator(Protocol):
    def expected_answer(self, query: str) -> str:
        ...

    def hyde_document(self, query: str) -> str:
        ...

    def mask_answer(self, expected_answer: str) -> str:
        ...


@dataclass
class HeuristicGenerator:
    """Zero-dependency generator for pipeline tests.

    This is intentionally simple. Replace it with OpenAITextGenerator or another
    model for real experiments.
    """

    def expected_answer(self, query: str) -> str:
        return f"The answer to the question is a specific entity or value related to: {query}"

    def hyde_document(self, query: str) -> str:
        return (
            "A relevant passage would directly discuss the subject of the question, "
            f"provide the requested fact, and include supporting context. Question: {query}"
        )

    def mask_answer(self, expected_answer: str) -> str:
        return mask_answer_spans(expected_answer)


@dataclass
class OpenAITextGenerator:
    model: str = "gpt-4.1-mini"
    temperature: float = 0.0
    max_output_tokens: int = 512

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install 'openai' to use OpenAITextGenerator.") from exc
        self._client = OpenAI()

    def expected_answer(self, query: str) -> str:
        return self._complete(
            "Write a concise expected answer to the question in one sentence. "
            "Prefer the likely answer form over a long explanation. "
            "Do not cite sources and do not include extra background.\n\n"
            f"Question: {query}"
        )

    def hyde_document(self, query: str) -> str:
        return self._complete(
            "Write a detailed hypothetical passage that would answer the question. "
            "The passage may be approximate, but should look like a real retrieved document. "
            "Include enough context that it resembles a paragraph from Wikipedia or a reference source.\n\n"
            f"Question: {query}"
        )

    def mask_answer(self, expected_answer: str) -> str:
        return self._complete(
            "Replace answer-bearing spans with neutral slots while preserving the relation being asked. "
            "Mask named entities, dates, numbers, locations, organizations, titles, and specific values "
            "that could directly determine the answer. Use slots such as [PERSON], [LOCATION], [DATE], "
            "[NUMBER], [ORGANIZATION], [TITLE], or [ENTITY]. Return only the masked sentence.\n\n"
            f"Expected answer: {expected_answer}"
        )

    def _complete(self, prompt: str) -> str:
        response = self._client.responses.create(
            model=self.model,
            input=prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        return response.output_text.strip()


@dataclass
class CachedTextGenerator:
    inner: TextGenerator
    cache: JsonCache
    namespace: str

    def expected_answer(self, query: str) -> str:
        return self._cached("expected_answer", query, lambda: self.inner.expected_answer(query))

    def hyde_document(self, query: str) -> str:
        return self._cached("hyde_document", query, lambda: self.inner.hyde_document(query))

    def mask_answer(self, expected_answer: str) -> str:
        return self._cached("mask_answer", expected_answer, lambda: self.inner.mask_answer(expected_answer))

    def _cached(self, task: str, text: str, build) -> str:
        key = f"{self.namespace}:{task}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
        cached = self.cache.get(key)
        if cached is not None:
            return str(cached)
        value = str(build()).strip()
        self.cache.set(key, value)
        return value


def mask_answer_spans(text: str) -> str:
    masked = text
    masked = re.sub(r"\b\d{1,4}([-/]\d{1,2})?([-/]\d{1,4})?\b", "[NUMBER]", masked)
    masked = re.sub(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b"
        r"( \d{1,2})?(, \d{4})?",
        "[DATE]",
        masked,
        flags=re.IGNORECASE,
    )
    masked = re.sub(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b",
        _mask_capitalized_span,
        masked,
    )
    return _collapse_repeated_slots(masked)


def _mask_capitalized_span(match: re.Match[str]) -> str:
    value = match.group(1)
    keep = {"The", "A", "An", "Question"}
    if value in keep:
        return value
    return "[ENTITY]"


def _collapse_repeated_slots(text: str) -> str:
    previous = None
    current = text
    while previous != current:
        previous = current
        current = re.sub(r"(\[[A-Z]+\])(\s+\1)+", r"\1", current)
    return current
