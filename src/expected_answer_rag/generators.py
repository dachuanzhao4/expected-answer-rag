from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from expected_answer_rag.cache import JsonCache


class TextGenerator(Protocol):
    def expected_answer(self, query: str) -> str:
        ...

    def hyde_document(self, query: str) -> str:
        ...

    def query2doc_document(self, query: str) -> str:
        ...

    def relevance_feedback(self, query: str) -> str:
        ...

    def mask_answer(self, query: str, expected_answer: str) -> str:
        ...

    def answer_candidate_template(self, query: str) -> str:
        ...

    def last_artifact(self) -> dict[str, Any] | None:
        ...


@dataclass
class HeuristicGenerator:
    """Zero-dependency generator for pipeline tests."""

    _last_artifact: dict[str, Any] | None = None

    def expected_answer(self, query: str) -> str:
        text = f"The answer to the question is a specific entity or value related to: {query}"
        self._record("expected_answer", query, text, prompt="heuristic expected answer", prompt_version="expected_answer_v1")
        return text

    def hyde_document(self, query: str) -> str:
        text = (
            "A relevant passage would directly discuss the subject of the question, "
            f"provide the requested fact, and include supporting context. Question: {query}"
        )
        self._record("hyde_document", query, text, prompt="heuristic hyde document", prompt_version="hyde_document_v1")
        return text

    def query2doc_document(self, query: str) -> str:
        text = (
            "Relevant background and entities for retrieval: "
            f"{query}. The document likely states the requested relation explicitly."
        )
        self._record("query2doc_document", query, text, prompt="heuristic query2doc", prompt_version="query2doc_document_v1")
        return text

    def relevance_feedback(self, query: str) -> str:
        text = f"Important feedback terms and relation clues for retrieval: {query}"
        self._record("relevance_feedback", query, text, prompt="heuristic relevance feedback", prompt_version="relevance_feedback_v1")
        return text

    def mask_answer(self, query: str, expected_answer: str) -> str:
        text = mask_answer_spans(expected_answer, query=query)
        self._record(
            "query_aware_mask_answer",
            f"Question: {query}\nExpected answer: {expected_answer}",
            text,
            prompt="heuristic mask answer",
            prompt_version="query_aware_mask_answer_v1",
        )
        return text

    def answer_candidate_template(self, query: str) -> str:
        slot = infer_answer_slot(query)
        payload = {
            "known_anchors": extract_query_anchors(query),
            "unknown_slot_type": slot.strip("[]"),
            "relation_intent": relation_intent_from_query(query),
            "retrieval_text": f"{query} {slot}".strip(),
        }
        text = json.dumps(payload, ensure_ascii=False)
        self._record("answer_candidate_template", query, text, prompt="heuristic template", prompt_version="answer_candidate_template_v1")
        return text

    def last_artifact(self) -> dict[str, Any] | None:
        return self._last_artifact

    def _record(self, task: str, input_text: str, output_text: str, prompt: str, prompt_version: str) -> None:
        self._last_artifact = {
            "task": task,
            "input_text": input_text,
            "prompt": prompt,
            "prompt_version": prompt_version,
            "output_text": output_text,
            "provider": "heuristic",
            "model": None,
            "raw_response": None,
            "timestamp_utc": _now_utc(),
        }


@dataclass
class MissingGenerator:
    """Generator used when all generations must already exist in cache."""

    _last_artifact: dict[str, Any] | None = None

    def expected_answer(self, query: str) -> str:
        raise RuntimeError(f"Missing cached expected answer for query: {query}")

    def hyde_document(self, query: str) -> str:
        raise RuntimeError(f"Missing cached HyDE document for query: {query}")

    def query2doc_document(self, query: str) -> str:
        raise RuntimeError(f"Missing cached Query2doc document for query: {query}")

    def relevance_feedback(self, query: str) -> str:
        raise RuntimeError(f"Missing cached relevance feedback for query: {query}")

    def mask_answer(self, query: str, expected_answer: str) -> str:
        raise RuntimeError(f"Missing cached masked answer for query: {query}")

    def answer_candidate_template(self, query: str) -> str:
        raise RuntimeError(f"Missing cached answer-candidate template for query: {query}")

    def last_artifact(self) -> dict[str, Any] | None:
        return self._last_artifact


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
    _last_artifact: dict[str, Any] | None = None

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
        prompt = (
            "Write a concise expected answer to the question in one sentence. "
            "Prefer the likely answer form over a long explanation. "
            "Do not cite sources and do not include extra background.\n\n"
            f"Question: {query}"
        )
        return self._complete("expected_answer", query, prompt, prompt_version="expected_answer_v1")

    def hyde_document(self, query: str) -> str:
        prompt = (
            "Write a detailed hypothetical passage that would answer the question. "
            "The passage may be approximate, but should look like a real retrieved document. "
            "Include enough context that it resembles a paragraph from Wikipedia or a reference source.\n\n"
            f"Question: {query}"
        )
        return self._complete("hyde_document", query, prompt, prompt_version="hyde_document_v1")

    def query2doc_document(self, query: str) -> str:
        prompt = (
            "Write a short pseudo-document that would help a retriever find evidence for the question. "
            "Use natural wording and include relation clues, but do not mention uncertainty.\n\n"
            f"Question: {query}"
        )
        return self._complete("query2doc_document", query, prompt, prompt_version="query2doc_document_v1")

    def relevance_feedback(self, query: str) -> str:
        prompt = (
            "Write a compact retrieval expansion with key relevance clues, entities, and relation terms that would help retrieve evidence. "
            "Keep it concise and retrieval-oriented.\n\n"
            f"Question: {query}"
        )
        return self._complete("relevance_feedback", query, prompt, prompt_version="relevance_feedback_v1")

    def mask_answer(self, query: str, expected_answer: str) -> str:
        prompt = (
            "You are doing query-aware answer masking for retrieval.\n\n"
            "Given a question and a concise expected answer, mask ONLY the span(s) in the expected answer "
            "that directly answer the question. Do not mask words, entities, dates, numbers, titles, or "
            "names that are already present in the question; those are retrieval anchors and must be preserved. "
            "Do not introduce any new concrete candidate entity, date, value, or title. "
            "Keep as much surrounding context as possible.\n\n"
            "Use typed neutral slots:\n"
            "- [PERSON]\n"
            "- [LOCATION]\n"
            "- [ORGANIZATION]\n"
            "- [DATE]\n"
            "- [NUMBER]\n"
            "- [TITLE]\n"
            "- [EVENT]\n"
            "- [ENTITY] only when no more specific type fits\n\n"
            "Return only the masked expected answer.\n\n"
            f"Question: {query}\n"
            f"Expected answer: {expected_answer}"
        )
        return self._complete(
            "query_aware_mask_answer",
            f"Question: {query}\nExpected answer: {expected_answer}",
            prompt,
            prompt_version="query_aware_mask_answer_v1",
        )

    def answer_candidate_template(self, query: str) -> str:
        prompt = (
            "You are writing a retrieval query, not answering the question.\n\n"
            f"Question: {query}\n\n"
            "Create a retrieval-oriented reformulation that helps find evidence documents.\n"
            "Rules:\n"
            "1. Preserve concrete entities, dates, titles, and values only if they appear in the original question.\n"
            "2. Do not introduce any new concrete person, organization, location, title, date, number, or answer candidate.\n"
            "3. Represent the unknown answer with a typed slot such as [PERSON], [ORGANIZATION], [LOCATION], [DATE], [NUMBER], [TITLE], or [EVENT].\n"
            "4. Include the relation or evidence need in natural language.\n"
            "5. Return JSON with keys: known_anchors, unknown_slot_type, relation_intent, retrieval_text.\n"
        )
        return self._complete("answer_candidate_template", query, prompt, prompt_version="answer_candidate_template_v1")

    def last_artifact(self) -> dict[str, Any] | None:
        return self._last_artifact

    def _complete(self, task: str, input_text: str, prompt: str, prompt_version: str) -> str:
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
                    token_name = self._token_param_name()
                    if token_name == "max_completion_tokens":
                        kwargs["max_completion_tokens"] = self.max_output_tokens
                    elif token_name == "max_tokens":
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
                if not text:
                    reasoning = getattr(message, "reasoning", None)
                    if reasoning:
                        text = str(reasoning).strip()
                if text:
                    self._last_artifact = {
                        "task": task,
                        "input_text": input_text,
                        "prompt": prompt,
                        "prompt_version": prompt_version,
                        "output_text": text,
                        "provider": "openai-compatible",
                        "model": self.model,
                        "raw_response": _safe_model_dump(response),
                        "timestamp_utc": _now_utc(),
                    }
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
    namespace_aliases: list[str] | None = None
    _last_artifact: dict[str, Any] | None = None

    def expected_answer(self, query: str) -> str:
        return self._cached("expected_answer", query, lambda: self.inner.expected_answer(query))

    def hyde_document(self, query: str) -> str:
        return self._cached("hyde_document", query, lambda: self.inner.hyde_document(query))

    def query2doc_document(self, query: str) -> str:
        return self._cached("query2doc_document", query, lambda: self.inner.query2doc_document(query))

    def relevance_feedback(self, query: str) -> str:
        return self._cached("relevance_feedback", query, lambda: self.inner.relevance_feedback(query))

    def mask_answer(self, query: str, expected_answer: str) -> str:
        cache_text = f"Question: {query}\nExpected answer: {expected_answer}"
        return self._cached("query_aware_mask_answer", cache_text, lambda: self.inner.mask_answer(query, expected_answer))

    def answer_candidate_template(self, query: str) -> str:
        return self._cached("answer_candidate_template", query, lambda: self.inner.answer_candidate_template(query))

    def last_artifact(self) -> dict[str, Any] | None:
        return self._last_artifact

    def _cached(self, task: str, text: str, build) -> str:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        namespaces = [self.namespace, *(self.namespace_aliases or [])]
        cached = None
        key = None
        for namespace in namespaces:
            candidate_key = f"{namespace}:{task}:{text_hash}"
            cached = self.cache.get(candidate_key)
            if cached is not None:
                key = candidate_key
                break
        if key is None:
            key = f"{self.namespace}:{task}:{text_hash}"
        if isinstance(cached, dict) and str(cached.get("output_text", "")).strip():
            self._last_artifact = dict(cached)
            return str(cached["output_text"])
        if isinstance(cached, str) and cached.strip():
            self._last_artifact = {
                "task": task,
                "input_text": text,
                "prompt": None,
                "prompt_version": _default_prompt_version(task),
                "output_text": cached,
                "provider": "legacy-cache",
                "model": None,
                "raw_response": None,
                "timestamp_utc": None,
            }
            return str(cached)
        value = str(build()).strip()
        if not value:
            raise RuntimeError(f"Empty generation for task={task}")
        artifact = dict(self.inner.last_artifact() or {})
        artifact.setdefault("task", task)
        artifact.setdefault("input_text", text)
        artifact.setdefault("output_text", value)
        artifact.setdefault("prompt_version", _default_prompt_version(task))
        artifact.setdefault("timestamp_utc", _now_utc())
        self.cache.set(key, artifact)
        self._last_artifact = artifact
        return value


def _default_prompt_version(task: str) -> str:
    return {
        "expected_answer": "expected_answer_v1",
        "hyde_document": "hyde_document_v1",
        "query2doc_document": "query2doc_document_v1",
        "relevance_feedback": "relevance_feedback_v1",
        "query_aware_mask_answer": "query_aware_mask_answer_v1",
        "answer_candidate_template": "answer_candidate_template_v1",
    }.get(task, f"{task}_v1")


def parse_answer_candidate_template(text: str, query: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("Template output is not a JSON object")
    except Exception:
        slot = infer_answer_slot(query)
        payload = {
            "known_anchors": extract_query_anchors(query),
            "unknown_slot_type": slot.strip("[]"),
            "relation_intent": relation_intent_from_query(query),
            "retrieval_text": text.strip(),
        }
    payload.setdefault("known_anchors", extract_query_anchors(query))
    payload.setdefault("unknown_slot_type", infer_answer_slot(query).strip("[]"))
    payload.setdefault("relation_intent", relation_intent_from_query(query))
    payload.setdefault("retrieval_text", query)
    return payload


def validate_answer_candidate_template(query: str, payload: dict[str, Any]) -> dict[str, Any]:
    query_anchors = {anchor.lower() for anchor in extract_query_anchors(query)}
    retrieval_text = str(payload.get("retrieval_text") or "")
    generated_candidates = [candidate for candidate in extract_query_anchors(retrieval_text) if candidate.lower() not in query_anchors]
    slot_type = str(payload.get("unknown_slot_type") or "").strip()
    return {
        "has_required_keys": all(key in payload for key in ["known_anchors", "unknown_slot_type", "relation_intent", "retrieval_text"]),
        "introduced_candidate_count": len(generated_candidates),
        "introduced_candidates": generated_candidates,
        "uses_slot": bool(slot_type),
        "valid": all(key in payload for key in ["known_anchors", "unknown_slot_type", "relation_intent", "retrieval_text"])
        and not generated_candidates
        and bool(slot_type),
    }


def mask_answer_spans(text: str, query: str = "") -> str:
    masked = text
    anchors = {anchor.lower() for anchor in extract_query_anchors(query)}
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
        lambda match: _mask_capitalized_span(match, anchors),
        masked,
    )
    return _collapse_repeated_slots(masked)


def generic_mask_answer(text: str) -> str:
    return re.sub(r"\[[A-Z_]+\]", "[MASK]", mask_answer_spans(text))


def entity_only_mask_answer(text: str, query: str = "") -> str:
    anchors = {anchor.lower() for anchor in extract_query_anchors(query)}
    masked = re.sub(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b",
        lambda match: _mask_capitalized_span(match, anchors, slot="[ENTITY]"),
        text,
    )
    return _collapse_repeated_slots(masked)


def random_span_mask_answer(text: str, query: str = "", seed: int = 13) -> str:
    slots = re.findall(r"\b\w+\b", text)
    if not slots:
        return text
    rng = random.Random(seed + len(text))
    words = text.split()
    idx = rng.randrange(len(words))
    words[idx] = "[MASK]"
    return " ".join(words)


def remove_gold_from_text(text: str, gold_strings: list[str] | tuple[str, ...]) -> str:
    updated = text
    for gold in sorted([value for value in gold_strings if value], key=len, reverse=True):
        updated = re.sub(re.escape(gold), "[ANSWER]", updated, flags=re.IGNORECASE)
    return _collapse_repeated_slots(updated)


def length_matched_neutral_filler(query: str, reference_text: str) -> str:
    filler_tokens = max(len(reference_text.split()), 1)
    filler = " ".join(["relevant"] * filler_tokens)
    return f"{query}\n{filler}".strip()


def relation_intent_from_query(query: str) -> str:
    lowered = query.lower()
    if lowered.startswith("where"):
        return "location or place relation"
    if lowered.startswith("when"):
        return "time or date relation"
    if lowered.startswith("who"):
        return "person or agent relation"
    if lowered.startswith("how many") or lowered.startswith("how much"):
        return "quantity relation"
    return "retrieve evidence for the query relation"


def infer_answer_slot(query: str) -> str:
    lowered = query.lower()
    if lowered.startswith("where"):
        return "[LOCATION]"
    if lowered.startswith("when"):
        return "[DATE]"
    if lowered.startswith("who"):
        return "[PERSON]"
    if lowered.startswith("how many") or lowered.startswith("how much"):
        return "[NUMBER]"
    return "[ENTITY]"


def extract_query_anchors(query: str) -> list[str]:
    anchors = []
    seen = set()
    for match in re.findall(r'"([^"\n]{2,120})"', query):
        if match.lower() not in seen:
            seen.add(match.lower())
            anchors.append(match)
    for match in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", query):
        if match.lower() not in seen and match not in {"The", "A", "An"}:
            seen.add(match.lower())
            anchors.append(match)
    return anchors


def _mask_capitalized_span(match: re.Match[str], anchors: set[str], slot: str = "[ENTITY]") -> str:
    value = match.group(1)
    keep = {"The", "A", "An", "Question"}
    if value in keep or value.lower() in anchors:
        return value
    return slot


def _collapse_repeated_slots(text: str) -> str:
    previous = None
    current = text
    while previous != current:
        previous = current
        current = re.sub(r"(\[[A-Z_]+\])(\s+\1)+", r"\1", current)
    return current


def _safe_model_dump(response: Any) -> Any:
    try:
        return response.model_dump()
    except Exception:  # noqa: BLE001
        return None


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
