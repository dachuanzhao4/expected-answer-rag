from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Protocol

from expected_answer_rag.cache import JsonCache


class TextGenerator(Protocol):
    def expected_answer(self, query: str) -> str:
        ...

    def hyde_document(self, query: str) -> str:
        ...

    def mask_answer(self, query: str, expected_answer: str) -> str:
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

    def mask_answer(self, query: str, expected_answer: str) -> str:
        return mask_answer_spans(expected_answer)


@dataclass
class MissingGenerator:
    """Generator used when all generations must already exist in cache."""

    def expected_answer(self, query: str) -> str:
        raise RuntimeError(f"Missing cached expected answer for query: {query}")

    def hyde_document(self, query: str) -> str:
        raise RuntimeError(f"Missing cached HyDE document for query: {query}")

    def mask_answer(self, query: str, expected_answer: str) -> str:
        raise RuntimeError(f"Missing cached masked answer for query: {query}")


@dataclass
class OpenAITextGenerator:
    model: str = "openai/gpt-5-mini"
    temperature: float | None = None
    max_output_tokens: int = 512
    token_param: str = "none"
    base_url: str | None = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    referer: str | None = None
    app_title: str | None = "expected-answer-rag"
    retries: int = 2
    include_reasoning: bool = False
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install 'openai' to use OpenAITextGenerator.") from exc
        import os

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Set {self.api_key_env} before using OpenAITextGenerator.")
        kwargs = {"api_key": api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)

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

    def mask_answer(self, query: str, expected_answer: str) -> str:
        return self._complete(
            "You are doing query-aware answer masking for retrieval.\n\n"
            "Given a question and a concise expected answer, mask ONLY the span(s) in the expected answer "
            "that directly answer the question. Do not mask words, entities, dates, numbers, titles, or "
            "names that are already present in the question; those are retrieval anchors and must be preserved. "
            "Keep as much surrounding context as possible.\n\n"
            "Use typed neutral slots:\n"
            "- [PERSON] for people\n"
            "- [LOCATION] for cities, countries, regions, addresses, or places\n"
            "- [ORGANIZATION] for companies, bands, schools, agencies, parties, or teams\n"
            "- [DATE] for years, dates, seasons, or time periods\n"
            "- [NUMBER] for counts, measurements, rankings, percentages, or amounts\n"
            "- [TITLE] for songs, books, films, albums, shows, laws, or named works\n"
            "- [EVENT] for named events\n"
            "- [ENTITY] only when no more specific type fits\n\n"
            "Examples:\n"
            "Question: how many episodes are in chicago fire season 4\n"
            "Expected answer: Chicago Fire season 4 consists of 23 episodes.\n"
            "Masked answer: Chicago Fire season 4 consists of [NUMBER] episodes.\n\n"
            "Question: who sings love will keep us alive by the eagles\n"
            "Expected answer: Timothy B. Schmit sings \"Love Will Keep Us Alive\" by the Eagles.\n"
            "Masked answer: [PERSON] sings \"Love Will Keep Us Alive\" by the Eagles.\n\n"
            "Question: where was Marie Curie born\n"
            "Expected answer: Marie Curie was born in Warsaw.\n"
            "Masked answer: Marie Curie was born in [LOCATION].\n\n"
            "Return only the masked expected answer.\n\n"
            f"Question: {query}\n"
            f"Expected answer: {expected_answer}"
        )

    def _complete(self, prompt: str) -> str:
        extra_headers = {}
        if self.referer:
            extra_headers["HTTP-Referer"] = self.referer
        if self.app_title:
            extra_headers["X-OpenRouter-Title"] = self.app_title

        last_error = None
        for attempt in range(self.retries + 1):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "extra_headers": extra_headers or None,
                }
                if self.temperature is not None:
                    kwargs["temperature"] = self.temperature
                if self.max_output_tokens > 0:
                    if self._token_param_name() is None:
                        pass
                    elif self._token_param_name() == "max_completion_tokens":
                        kwargs["max_completion_tokens"] = self.max_output_tokens
                    elif self._token_param_name() == "max_tokens":
                        kwargs["max_tokens"] = self.max_output_tokens
                if self.include_reasoning:
                    kwargs["include_reasoning"] = True
                if self.reasoning_effort:
                    kwargs["reasoning"] = {"effort": self.reasoning_effort}
                response = self._client.chat.completions.create(**kwargs)
                message = response.choices[0].message
                content = message.content
                if isinstance(content, list):
                    content = "".join(str(part.get("text", part)) for part in content)
                text = (content or "").strip()
                if text:
                    return text
                reasoning = getattr(message, "reasoning", None)
                if reasoning:
                    text = str(reasoning).strip()
                    if text:
                        return text
                last_error = RuntimeError("model returned empty content")
            except Exception as exc:  # noqa: BLE001 - retry provider/transient failures.
                last_error = exc
            if attempt < self.retries:
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Generation failed after retries: {last_error}")

    def _token_param_name(self) -> str | None:
        if self.token_param == "none":
            return None
        if self.token_param in {"max_tokens", "max_completion_tokens"}:
            return self.token_param
        if self.model.startswith("openai/gpt-5"):
            return "max_completion_tokens"
        return "max_tokens"


@dataclass
class CachedTextGenerator:
    inner: TextGenerator
    cache: JsonCache
    namespace: str

    def expected_answer(self, query: str) -> str:
        return self._cached("expected_answer", query, lambda: self.inner.expected_answer(query))

    def hyde_document(self, query: str) -> str:
        return self._cached("hyde_document", query, lambda: self.inner.hyde_document(query))

    def mask_answer(self, query: str, expected_answer: str) -> str:
        cache_text = f"Question: {query}\nExpected answer: {expected_answer}"
        return self._cached("query_aware_mask_answer", cache_text, lambda: self.inner.mask_answer(query, expected_answer))

    def _cached(self, task: str, text: str, build) -> str:
        key = f"{self.namespace}:{task}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
        cached = self.cache.get(key)
        if isinstance(cached, str) and cached.strip():
            return str(cached)
        value = str(build()).strip()
        if not value:
            raise RuntimeError(f"Empty generation for task={task}")
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
