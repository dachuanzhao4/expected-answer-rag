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

    def counterfactual_prompt_query_expansion(self, query: str, support_context: str = "") -> str:
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

    def counterfactual_prompt_query_expansion(self, query: str, support_context: str = "") -> str:
        payload = build_counterfactual_prompt_payload(query)
        anchors = list(payload["known_anchors"])
        slot = f"[{payload['unknown_slot_type']}]"
        relation = str(payload["relation_intent"])
        support_terms = _support_terms_from_context(support_context)
        support_phrase = " ".join(support_terms[:3]).strip()
        relation_queries = [
            f"{' '.join(anchors)} {support_phrase or relation}".strip(),
            f"{' '.join(anchors)} evidence {support_phrase or relation}".strip(),
        ]
        evidence_queries = [
            f"{' '.join(anchors)} {relation} {' '.join(support_terms[:2])}".strip(),
        ]
        answer_slot_query = f"{' '.join(anchors)} {slot}".strip()
        bridge_query = (
            f"{' '.join(anchors)} intermediate evidence {support_phrase or relation}".strip()
            if payload.get("requires_bridge_query")
            else ""
        )
        result = {
            "obfuscated_question": payload["obfuscated_question"],
            "entity_map": payload["entity_map"],
            "known_anchors": payload["known_anchors"],
            "unknown_slot_type": payload["unknown_slot_type"],
            "relation_intent": payload["relation_intent"],
            "relation_queries": _unique_nonempty_queries(relation_queries),
            "evidence_queries": _unique_nonempty_queries(evidence_queries),
            "answer_slot_query": answer_slot_query,
            "bridge_query": bridge_query,
            "queries": _unique_nonempty_queries(
                relation_queries + evidence_queries + [answer_slot_query, bridge_query]
            ),
        }
        text = json.dumps(result, ensure_ascii=False)
        self._record(
            "counterfactual_prompt_query_expansion_v3",
            f"Question: {query}\nSupport context: {support_context}",
            text,
            prompt="heuristic counterfactual prompt expansion",
            prompt_version="counterfactual_prompt_query_expansion_v3",
        )
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

    def counterfactual_prompt_query_expansion(self, query: str, support_context: str = "") -> str:
        raise RuntimeError(f"Missing cached counterfactual prompt expansion for query: {query}")

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

    def counterfactual_prompt_query_expansion(self, query: str, support_context: str = "") -> str:
        payload = build_counterfactual_prompt_payload(query)
        prompt = (
            "You are generating retrieval queries, not answering the question.\n\n"
            "The question below has been obfuscated to hide public entity triggers that could cause answer recall.\n"
            "Generate retrieval queries that preserve relation intent without introducing any new named entities, dates, numbers, titles, or answer candidates.\n\n"
            "Rules:\n"
            "1. Do not answer the question.\n"
            "2. Do not introduce new concrete entities, values, dates, or titles.\n"
            "3. Use only the placeholders shown in the entity map.\n"
            "4. Preserve the known anchors exactly.\n"
            "5. Use typed unknown slots when the answer should remain unspecified.\n"
            "6. Reuse corpus-supported relation terms when they are provided below.\n"
            "7. If the question is multi-hop, include one bridge-style query but do not invent the bridge entity.\n"
            "8. Avoid generic fillers such as 'information about', 'documents about', 'what does X represent', or vague paraphrases unless paired with specific corpus-supported relation terms.\n"
            "9. Every non-slot query should include at least one concrete relation or evidence term from the support context when available.\n"
            "10. Return strict JSON with keys:\n"
            "   - relation_queries: list of 2 to 4 short relation-preserving queries\n"
            "   - evidence_queries: list of 1 to 2 evidence-focused queries\n"
            "   - answer_slot_query: one query using a typed unknown slot\n"
            "   - bridge_query: optional string, empty if not needed\n\n"
            f"Obfuscated question:\n{payload['obfuscated_question']}\n\n"
            "Entity map:\n"
            f"{json.dumps(payload['entity_map'], ensure_ascii=False, indent=2)}\n\n"
            "Support context from first-pass retrieval:\n"
            f"{support_context or 'None'}\n"
        )
        raw_text = self._complete(
            "counterfactual_prompt_query_expansion_v3",
            f"Question: {query}\nSupport context: {support_context}",
            prompt,
            prompt_version="counterfactual_prompt_query_expansion_v3",
        )
        parsed = parse_counterfactual_prompt_query_expansion(raw_text, query, payload)
        final_text = json.dumps(parsed, ensure_ascii=False)
        if self._last_artifact is not None:
            self._last_artifact["raw_model_text"] = raw_text
            self._last_artifact["output_text"] = final_text
            self._last_artifact["obfuscated_question"] = payload["obfuscated_question"]
            self._last_artifact["entity_map"] = payload["entity_map"]
        return final_text

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

    def counterfactual_prompt_query_expansion(self, query: str, support_context: str = "") -> str:
        return self._cached(
            "counterfactual_prompt_query_expansion_v3",
            f"Question: {query}\nSupport context: {support_context}",
            lambda: self.inner.counterfactual_prompt_query_expansion(query, support_context),
        )

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
        "counterfactual_prompt_query_expansion": "counterfactual_prompt_query_expansion_v1",
        "counterfactual_prompt_query_expansion_v2": "counterfactual_prompt_query_expansion_v2",
        "counterfactual_prompt_query_expansion_v3": "counterfactual_prompt_query_expansion_v3",
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


def build_counterfactual_prompt_payload(query: str) -> dict[str, Any]:
    anchors = _extract_obfuscatable_anchors(query)
    obfuscated = query
    entity_map = []
    type_counters: dict[str, int] = {}
    placeholder_to_original: dict[str, str] = {}
    for anchor in sorted(anchors, key=len, reverse=True):
        anchor_type = _infer_obfuscation_type(anchor)
        type_counters[anchor_type] = type_counters.get(anchor_type, 0) + 1
        placeholder = f"{anchor_type}_{type_counters[anchor_type]}"
        obfuscated = re.sub(rf"\b{re.escape(anchor)}\b", placeholder, obfuscated)
        entity_map.append(
            {
                "placeholder": placeholder,
                "original": anchor,
                "type": anchor_type,
                "description": "known anchor from the original question",
            }
        )
        placeholder_to_original[placeholder] = anchor
    return {
        "original_question": query,
        "obfuscated_question": obfuscated,
        "entity_map": entity_map,
        "placeholder_to_original": placeholder_to_original,
        "known_anchors": extract_query_anchors(query),
        "unknown_slot_type": infer_answer_slot(query).strip("[]"),
        "relation_intent": relation_intent_from_query(query),
        "requires_bridge_query": _requires_bridge_query(query),
    }


def parse_counterfactual_prompt_query_expansion(
    text: str,
    query: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    raw_queries: list[str] = []
    relation_queries: list[str] = []
    evidence_queries: list[str] = []
    answer_slot_query = ""
    bridge_query = ""
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            relation_queries = _query_list_from_json(loaded.get("relation_queries"))
            evidence_queries = _query_list_from_json(loaded.get("evidence_queries"))
            answer_slot_query = str(loaded.get("answer_slot_query") or "").strip()
            bridge_query = str(loaded.get("bridge_query") or "").strip()
            candidate_queries = loaded.get("queries", [])
            if isinstance(candidate_queries, list):
                raw_queries = [str(item).strip() for item in candidate_queries]
        elif isinstance(loaded, list):
            raw_queries = [str(item).strip() for item in loaded]
    except Exception:
        raw_queries = []
    raw_queries = _unique_nonempty_queries(
        relation_queries + evidence_queries + [answer_slot_query, bridge_query] + raw_queries
    )
    if not raw_queries:
        raw_queries = [
            re.sub(r"^\s*[-*\d.]+\s*", "", line).strip()
            for line in text.splitlines()
            if line.strip()
        ]
    if not raw_queries:
        slot = infer_answer_slot(query)
        anchors = [item["original"] for item in payload["entity_map"]]
        raw_queries = [
            f"{' '.join(anchors)} {relation_intent_from_query(query)}".strip(),
            f"{' '.join(anchors)} {slot}".strip(),
        ]
    queries = []
    for candidate in raw_queries:
        queries.append(_deobfuscate_query(candidate, payload))
    relation_queries = [_deobfuscate_query(candidate, payload) for candidate in relation_queries]
    evidence_queries = [_deobfuscate_query(candidate, payload) for candidate in evidence_queries]
    answer_slot_query = _deobfuscate_query(answer_slot_query, payload)
    bridge_query = _deobfuscate_query(bridge_query, payload)
    return {
        "obfuscated_question": payload["obfuscated_question"],
        "entity_map": payload["entity_map"],
        "known_anchors": payload["known_anchors"],
        "unknown_slot_type": payload["unknown_slot_type"],
        "relation_intent": payload["relation_intent"],
        "relation_queries": _unique_nonempty_queries(relation_queries),
        "evidence_queries": _unique_nonempty_queries(evidence_queries),
        "answer_slot_query": answer_slot_query,
        "bridge_query": bridge_query,
        "queries": _unique_nonempty_queries(queries),
    }


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
    banned = {"the", "a", "an", "who", "what", "where", "when", "which", "how", "question"}
    for match in re.findall(r'"([^"\n]{2,120})"', query):
        if match.lower() not in seen and match.lower() not in banned:
            seen.add(match.lower())
            anchors.append(match)
    for match in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", query):
        if match.lower() not in seen and match.lower() not in banned:
            seen.add(match.lower())
            anchors.append(match)
    return anchors


def _extract_obfuscatable_anchors(query: str) -> list[str]:
    anchors = []
    seen = set()
    patterns = [
        r'"([^"\n]{2,120})"',
        r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}|of|the|and|for|to|in|on)){0,6}\b",
    ]
    stop_values = {"Who", "What", "Where", "When", "Which", "How", "Question", "The", "A", "An"}
    for pattern in patterns:
        for match in re.findall(pattern, query):
            value = str(match).strip()
            if not value or value in stop_values:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            anchors.append(value)
    return anchors


def _infer_obfuscation_type(anchor: str) -> str:
    if re.search(r"[\"“”']", anchor):
        return "WORK"
    if len(anchor.split()) >= 3:
        return "WORK"
    return "ENTITY"


def _query_list_from_json(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _support_terms_from_context(support_context: str) -> list[str]:
    terms = []
    seen = set()
    for line in str(support_context or "").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip().lower() != "support_terms":
            continue
        for term in value.split(","):
            normalized = " ".join(term.split()).strip()
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            terms.append(normalized)
    return terms


def _deobfuscate_query(text: str, payload: dict[str, Any]) -> str:
    updated = str(text or "").strip()
    if not updated:
        return ""
    for item in payload["entity_map"]:
        updated = updated.replace(item["placeholder"], item["original"])
    return updated.strip()


def _requires_bridge_query(query: str) -> bool:
    lowered = f" {query.lower()} "
    markers = [
        " that ",
        " which ",
        " book that inspired ",
        " author of the book ",
        " nationality of the author ",
        " inspired by ",
        " score for ",
    ]
    return any(marker in lowered for marker in markers)


def _unique_nonempty_queries(values: list[str]) -> list[str]:
    queries = []
    seen = set()
    for value in values:
        normalized = " ".join(value.split())
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(normalized)
    return queries[:6]


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
