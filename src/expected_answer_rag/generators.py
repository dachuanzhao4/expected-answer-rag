from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Protocol

from expected_answer_rag.answer_blanked import (
    build_answer_blanked_query2doc,
    deterministic_llm_reformat_payload_text,
    sanitize_answer_blanked_query2doc,
)
from expected_answer_rag.cache import JsonCache


class TextGenerator(Protocol):
    def query2doc(self, query: str) -> str:
        ...

    def mask_query2doc(self, query: str, query2doc: str) -> str:
        ...

    def answer_blanked_query2doc(self, query: str) -> str:
        ...

    def llm_reformat_intent(self, query: str) -> str:
        ...


@dataclass
class HeuristicGenerator:
    """Zero-dependency generator for pipeline tests.

    This is intentionally simple. Replace it with OpenAITextGenerator or another
    model for real experiments.
    """

    llm_reformat_version: str = "v2"

    def query2doc(self, query: str) -> str:
        return (
            "A relevant passage would directly discuss the subject of the question, "
            f"provide the requested fact, and include supporting context. Question: {query}"
        )

    def mask_query2doc(self, query: str, query2doc: str) -> str:
        return mask_answer_spans(query2doc)

    def answer_blanked_query2doc(self, query: str) -> str:
        return build_answer_blanked_query2doc(query)

    def llm_reformat_intent(self, query: str) -> str:
        return deterministic_llm_reformat_payload_text(query, version=self.llm_reformat_version)


@dataclass
class MissingGenerator:
    """Generator used when all generations must already exist in cache."""

    def query2doc(self, query: str) -> str:
        raise RuntimeError(f"Missing cached Query2Doc document for query: {query}")

    def mask_query2doc(self, query: str, query2doc: str) -> str:
        raise RuntimeError(f"Missing cached masked Query2Doc document for query: {query}")

    def answer_blanked_query2doc(self, query: str) -> str:
        raise RuntimeError(f"Missing cached answer-blanked Query2Doc document for query: {query}")

    def llm_reformat_intent(self, query: str) -> str:
        raise RuntimeError(f"Missing cached LLM reformat intent for query: {query}")


@dataclass
class OpenAITextGenerator:
    model: str = "openai/gpt-5-mini"
    temperature: float | None = None
    max_output_tokens: int = 512
    token_param: str = "none"
    base_url: str | None = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    referer: str | None = None
    app_title: str | None = "query2doc-mask-rag"
    retries: int = 2
    include_reasoning: bool = False
    reasoning_effort: str | None = None
    prompt_style: str = "query2doc_fewshot"
    llm_reformat_version: str = "v2"

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

    def query2doc(self, query: str) -> str:
        if self.prompt_style == "query2doc_fewshot":
            return self._complete(_query2doc_fewshot_prompt(query))
        return self._complete(
            "Write a passage that answers the given query.\n\n"
            f"Query: {query}\n"
            "Passage:"
        )

    def mask_query2doc(self, query: str, query2doc: str) -> str:
        return self._complete(
            "You are doing query-aware masking for Query2Doc retrieval.\n\n"
            "Given a question and a hypothetical document, mask ONLY the span(s) in the document "
            "that directly fill the unknown answer asked by the question. Do not mask words, entities, "
            "dates, numbers, titles, or "
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
            "Query2Doc: Chicago Fire season 4 consists of 23 episodes and aired on NBC.\n"
            "Masked Query2Doc: Chicago Fire season 4 consists of [NUMBER] episodes and aired on NBC.\n\n"
            "Question: who sings love will keep us alive by the eagles\n"
            "Query2Doc: Timothy B. Schmit sings \"Love Will Keep Us Alive\" by the Eagles.\n"
            "Masked Query2Doc: [PERSON] sings \"Love Will Keep Us Alive\" by the Eagles.\n\n"
            "Question: where was Marie Curie born\n"
            "Query2Doc: Marie Curie was born in Warsaw and later became known for her work on radioactivity.\n"
            "Masked Query2Doc: Marie Curie was born in [LOCATION] and later became known for her work on radioactivity.\n\n"
            "Return only the masked Query2Doc text.\n\n"
            f"Question: {query}\n"
            f"Query2Doc: {query2doc}"
        )

    def answer_blanked_query2doc(self, query: str) -> str:
        generated = self._complete(_answer_blanked_query2doc_prompt(query))
        return sanitize_answer_blanked_query2doc(query, generated)

    def llm_reformat_intent(self, query: str) -> str:
        if self.llm_reformat_version == "v1":
            return self._complete(_llm_reformat_intent_prompt_v1(query))
        return self._complete(_llm_reformat_intent_prompt_v2(query))

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
    llm_reformat_version: str = "v2"

    def query2doc(self, query: str) -> str:
        return self._cached("query2doc", query, lambda: self.inner.query2doc(query), fallback_tasks=["hyde_document"])

    def mask_query2doc(self, query: str, query2doc: str) -> str:
        cache_text = f"Question: {query}\nQuery2Doc: {query2doc}"
        legacy_cache_text = f"Question: {query}\nExpected answer: {query2doc}"
        return self._cached(
            "query_aware_mask_query2doc",
            cache_text,
            lambda: self.inner.mask_query2doc(query, query2doc),
            fallback_keys=[
                self._key("query_aware_mask_answer", legacy_cache_text),
            ],
        )

    def answer_blanked_query2doc(self, query: str) -> str:
        return self._cached(
            "answer_blanked_query2doc",
            query,
            lambda: self.inner.answer_blanked_query2doc(query),
        )

    def llm_reformat_intent(self, query: str) -> str:
        return self._cached(
            f"llm_reformat_intent_{self.llm_reformat_version}",
            query,
            lambda: self.inner.llm_reformat_intent(query),
        )

    def _key(self, task: str, text: str) -> str:
        return f"{self.namespace}:{task}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"

    def _cached(
        self,
        task: str,
        text: str,
        build,
        fallback_tasks: list[str] | None = None,
        fallback_keys: list[str] | None = None,
    ) -> str:
        key = self._key(task, text)
        cached = self.cache.get(key)
        if isinstance(cached, str) and cached.strip():
            return str(cached)
        for fallback_task in fallback_tasks or []:
            cached = self.cache.get(self._key(fallback_task, text))
            if isinstance(cached, str) and cached.strip():
                self.cache.set(key, str(cached))
                return str(cached)
        for fallback_key in fallback_keys or []:
            cached = self.cache.get(fallback_key)
            if isinstance(cached, str) and cached.strip():
                self.cache.set(key, str(cached))
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


def _query2doc_fewshot_prompt(query: str) -> str:
    return (
        "Write a passage that answers the given query.\n\n"
        "Query: what causes seasonal allergies\n"
        "Passage: Seasonal allergies happen when the immune system reacts to airborne pollen, mold spores, or other outdoor allergens. Symptoms often include sneezing, itchy eyes, congestion, and a runny nose.\n\n"
        "Query: who wrote pride and prejudice\n"
        "Passage: Pride and Prejudice is a novel by Jane Austen. It was first published in 1813 and is one of the best-known works of English literature.\n\n"
        "Query: how many players are on a soccer team\n"
        "Passage: A standard soccer team has eleven players on the field, including one goalkeeper. Teams may also have substitutes available during a match.\n\n"
        "Query: where is the great barrier reef located\n"
        "Passage: The Great Barrier Reef is located off the northeastern coast of Australia in the Coral Sea. It extends along the coast of Queensland and is the world's largest coral reef system.\n\n"
        f"Query: {query}\n"
        "Passage:"
    )


def _answer_blanked_query2doc_prompt(query: str) -> str:
    return (
        "Write an answer-blanked retrieval passage for private RAG.\n\n"
        "Goal: produce corpus-style retrieval text that preserves known query anchors and relation words, "
        "but never guesses the unknown answer. The unknown answer span must be replaced by one typed slot.\n\n"
        "Rules:\n"
        "- Keep names, titles, dates, numbers, codes, and other anchors that already appear in the query.\n"
        "- Do not introduce any new concrete person, location, organization, title, date, or number as the answer.\n"
        "- Do not say you do not know, cannot find, need context, or that a code is a placeholder.\n"
        "- Use exactly these slot forms when blanking unknown answers: [PERSON], [LOCATION], [ORGANIZATION], "
        "[DATE], [NUMBER], [TITLE], [EVENT], [ENTITY].\n"
        "- Return only the passage text. No JSON, labels, bullets, caveats, or explanation.\n\n"
        "Examples:\n"
        "Query: who sings Love Will Keep Us Alive by the Eagles\n"
        "Passage: [PERSON] is the singer, performer, or vocalist associated with Love Will Keep Us Alive by the Eagles. "
        "Relevant passages discuss the Eagles, Love Will Keep Us Alive, singer, performer, vocals, and credits.\n\n"
        "Query: how many episodes are in Chicago Fire season 4\n"
        "Passage: Chicago Fire season 4 has [NUMBER] episodes. Relevant passages discuss Chicago Fire, season 4, "
        "episode count, aired episodes, and the total number of episodes.\n\n"
        "Query: where was Marie Curie born\n"
        "Passage: Marie Curie was born in [LOCATION]. Relevant biographical passages discuss Marie Curie, born, "
        "birthplace, hometown, city, country, and early life.\n\n"
        "Query: when does Work_AA0074 come out\n"
        "Passage: Work_AA0074 comes out on [DATE]. Relevant passages discuss Work_AA0074, release date, premiere, "
        "came out, announced, aired, and publication timing.\n\n"
        f"Query: {query}\n"
        "Passage:"
    )


def _llm_reformat_intent_prompt_v1(query: str) -> str:
    return (
        "You are extracting an answer-free retrieval intent for a private RAG corpus.\n\n"
        "Important setting: the corpus may contain private or renamed entities that you have never seen. "
        "Tokens such as Entity_123, Work_AB0001, Location_X9, or unusual names are real corpus anchors, "
        "not placeholders. Do not use world knowledge and do not answer the question.\n\n"
        "Return exactly one JSON object with this schema:\n"
        "{\n"
        '  "answer_type": "PERSON|LOCATION|ORGANIZATION|DATE|NUMBER|TITLE|EVENT|ENTITY",\n'
        '  "anchors": [{"text": "exact query substring", "role": "short_generic_role"}],\n'
        '  "relation": {\n'
        '    "name": "short_snake_case_relation",\n'
        '    "confidence": "high|medium|low",\n'
        '    "safe_cues": ["lowercase generic retrieval cue", "..."]\n'
        "  },\n"
        '  "must_keep_terms": ["query substring or query token", "..."],\n'
        '  "safe_expansion_terms": ["lowercase generic relation cue", "..."]\n'
        "}\n\n"
        "Controls:\n"
        "- anchors must be copied from the query, preferably exact entity/title/person/code substrings.\n"
        "- must_keep_terms must be copied from the query or composed only of query tokens.\n"
        "- safe_expansion_terms may add only generic relation/search words, never concrete answers.\n"
        "- Do not introduce new people, places, organizations, titles, dates, years, numbers, or proper nouns.\n"
        "- If the relation is unclear, set confidence to low and use an empty safe_expansion_terms list.\n"
        "- Keep safe_expansion_terms short: at most four items, each one to four lowercase words.\n"
        "- Do not output a passage, an explanation, markdown, or refusal text.\n\n"
        "Examples:\n"
        "Q: who sings Love Will Keep Us Alive by the Eagles\n"
        "JSON: {\"answer_type\":\"PERSON\",\"anchors\":[{\"text\":\"Love Will Keep Us Alive\",\"role\":\"work_title\"},{\"text\":\"the Eagles\",\"role\":\"known_group\"}],\"relation\":{\"name\":\"performer_lookup\",\"confidence\":\"high\",\"safe_cues\":[\"singer\",\"vocalist\",\"performer\",\"credits\"]},\"must_keep_terms\":[\"Love Will Keep Us Alive\",\"the Eagles\",\"sings\"],\"safe_expansion_terms\":[\"singer\",\"vocalist\",\"performer\",\"credits\"]}\n\n"
        "Q: when does Work_AA0074 come out\n"
        "JSON: {\"answer_type\":\"DATE\",\"anchors\":[{\"text\":\"Work_AA0074\",\"role\":\"work_title\"}],\"relation\":{\"name\":\"release_timing\",\"confidence\":\"high\",\"safe_cues\":[\"release date\",\"premiere\",\"came out\"]},\"must_keep_terms\":[\"Work_AA0074\",\"come out\"],\"safe_expansion_terms\":[\"release date\",\"premiere\",\"came out\"]}\n\n"
        "Q: which government had more power under the articles of confederation\n"
        "JSON: {\"answer_type\":\"ENTITY\",\"anchors\":[{\"text\":\"articles of confederation\",\"role\":\"known_context\"}],\"relation\":{\"name\":\"comparative_government_power\",\"confidence\":\"medium\",\"safe_cues\":[\"government power\",\"federal state\",\"confederation\"]},\"must_keep_terms\":[\"government\",\"power\",\"articles of confederation\"],\"safe_expansion_terms\":[\"government power\",\"federal state\",\"confederation\"]}\n\n"
        f"Q: {query}\n"
        "JSON:"
    )


def _llm_reformat_intent_prompt_v2(query: str) -> str:
    return (
        "You are extracting a leakage-free retrieval intent for a private RAG corpus.\n\n"
        "The corpus may contain private or renamed entities that are not in your training data. "
        "Tokens such as Entity_123, Work_AB0001, Person_X9, Location_Z7, or unusual names are real anchors. "
        "Do not answer the question, do not use world knowledge, and do not invent facts.\n\n"
        "Return exactly one JSON object with this schema:\n"
        "{\n"
        '  "answer_type": "PERSON|LOCATION|ORGANIZATION|DATE|NUMBER|TITLE|EVENT|ENTITY",\n'
        '  "anchors": [\n'
        '    {"text": "exact substring copied from the query", "role": "subject|work|person|organization|location|event|context|modifier", "importance": "primary|support"}\n'
        "  ],\n"
        '  "query_focus_terms": ["exact query substring or query token", "..."],\n'
        '  "relation_class": "identity|person_role|performer|cast|creator|release_time|event_time|location|origin|count|definition|ownership|membership|comparison|function|legal_rule|event|other",\n'
        '  "relation_confidence": "high|medium|low",\n'
        '  "retrieval_policy": "anchor_only|anchor_plus_one_cue|query_preserve"\n'
        "}\n\n"
        "Global rules:\n"
        "- anchors and query_focus_terms must be copied from the query text. Prefer the longest non-overlapping anchors.\n"
        "- relation_class must be chosen only from the fixed list. Do not create a new label.\n"
        "- retrieval_policy controls drift: use anchor_plus_one_cue only when a generic relation cue is likely helpful; "
        "use anchor_only when anchors are strong and the relation is obvious from the query; use query_preserve when the relation is ambiguous.\n"
        "- Do not output safe_cues, expanded passages, answers, explanations, markdown, or refusal text.\n"
        "- Never add concrete people, places, organizations, titles, dates, years, numbers, or proper nouns that are not already in the query.\n"
        "- If the query contains private codes or renamed entities, preserve them exactly and treat them as meaningful.\n\n"
        "Examples:\n"
        "Q: who sings Love Will Keep Us Alive by the Eagles\n"
        "JSON: {\"answer_type\":\"PERSON\",\"anchors\":[{\"text\":\"Love Will Keep Us Alive\",\"role\":\"work\",\"importance\":\"primary\"},{\"text\":\"the Eagles\",\"role\":\"organization\",\"importance\":\"support\"}],\"query_focus_terms\":[\"Love Will Keep Us Alive\",\"the Eagles\",\"sings\"],\"relation_class\":\"performer\",\"relation_confidence\":\"high\",\"retrieval_policy\":\"anchor_plus_one_cue\"}\n\n"
        "Q: when does Work_AA0074 come out\n"
        "JSON: {\"answer_type\":\"DATE\",\"anchors\":[{\"text\":\"Work_AA0074\",\"role\":\"work\",\"importance\":\"primary\"}],\"query_focus_terms\":[\"Work_AA0074\",\"come out\"],\"relation_class\":\"release_time\",\"relation_confidence\":\"high\",\"retrieval_policy\":\"anchor_plus_one_cue\"}\n\n"
        "Q: which government had more power under the articles of confederation\n"
        "JSON: {\"answer_type\":\"ENTITY\",\"anchors\":[{\"text\":\"articles of confederation\",\"role\":\"context\",\"importance\":\"primary\"}],\"query_focus_terms\":[\"government\",\"power\",\"articles of confederation\"],\"relation_class\":\"comparison\",\"relation_confidence\":\"medium\",\"retrieval_policy\":\"query_preserve\"}\n\n"
        "Q: who is Entity_P50845 on days of our lives\n"
        "JSON: {\"answer_type\":\"PERSON\",\"anchors\":[{\"text\":\"Entity_P50845\",\"role\":\"person\",\"importance\":\"primary\"},{\"text\":\"days of our lives\",\"role\":\"work\",\"importance\":\"support\"}],\"query_focus_terms\":[\"Entity_P50845\",\"days of our lives\"],\"relation_class\":\"person_role\",\"relation_confidence\":\"low\",\"retrieval_policy\":\"query_preserve\"}\n\n"
        f"Q: {query}\n"
        "JSON:"
    )
