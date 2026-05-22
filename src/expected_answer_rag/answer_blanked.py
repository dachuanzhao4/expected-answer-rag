from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any


SLOT_BY_TYPE = {
    "PERSON": "[PERSON]",
    "LOCATION": "[LOCATION]",
    "ORGANIZATION": "[ORGANIZATION]",
    "DATE": "[DATE]",
    "NUMBER": "[NUMBER]",
    "TITLE": "[TITLE]",
    "EVENT": "[EVENT]",
    "ENTITY": "[ENTITY]",
}

VALID_SLOTS = set(SLOT_BY_TYPE.values())

# Keep this override narrow: phrases such as "I can't help falling in love"
# are valid query text, not refusals.
_LEGACY_REFUSAL_RE = re.compile(
    r"\b("
    r"i\s+(?:do not|don't|don[’']t)\s+know|"
    r"i\s+(?:can not|cannot|can't|can[’']t)|"
    r"(?:can not|cannot|can't|can[’']t)\s+find|"
    r"unable\s+to|"
    r"not\s+enough\s+information|"
    r"no\s+(?:reliable|specific|publicly\s+available)|"
    r"could\s+not|couldn[’']t|"
    r"placeholder|fictional|"
    r"insufficient\s+information|"
    r"not\s+possible\s+to\s+determine|"
    r"need\s+more\s+context|"
    r"please\s+provide|"
    r"is\s+not\s+a\s+recognizable|"
    r"does\s+not\s+correspond\s+to"
    r")\b",
    re.IGNORECASE,
)

REFUSAL_RE = re.compile(
    r"\b("
    r"i\s+(?:do not|don't|don.t)\s+know|"
    r"i\s+(?:can not|cannot|can't|can.t)\s+(?:find|answer|determine|provide|verify|confirm|identify)|"
    r"(?:can not|cannot|can't|can.t)\s+find|"
    r"unable\s+to|"
    r"not\s+enough\s+information|"
    r"no\s+(?:reliable|specific|publicly\s+available)|"
    r"could\s+not|couldn.t|"
    r"placeholder|fictional|"
    r"insufficient\s+information|"
    r"not\s+possible\s+to\s+determine|"
    r"need\s+more\s+context|"
    r"please\s+provide|"
    r"is\s+not\s+a\s+recognizable|"
    r"does\s+not\s+correspond\s+to"
    r")\b",
    re.IGNORECASE,
)

LABEL_RE = re.compile(r"^\s*(?:passage|answer-blanked passage|skeleton|query|output)\s*:\s*", re.IGNORECASE)
SLOT_RE = re.compile(r"\[(PERSON|LOCATION|ORGANIZATION|DATE|NUMBER|TITLE|EVENT|ENTITY)\]")
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
CODE_RE = re.compile(r"\b(?:[A-Z][A-Za-z]+[A-Z][A-Za-z]*\d{3,}|Entity_[A-Z0-9]+|Org_[A-Z0-9]+|Work_[A-Z0-9]+|Person_[A-Z0-9]+|Location_[A-Z0-9]+)\b")
CAPITALIZED_SPAN_RE = re.compile(
    r"\b(?:[A-Z]\.|[A-Z][A-Za-z0-9_]+)(?:\s+(?:[A-Z]\.|[A-Z][A-Za-z0-9_]+)){0,5}\b"
)
NUMBER_RE = re.compile(r"\b\d+(?:[,.]\d+)*(?:st|nd|rd|th)?\b", re.IGNORECASE)
MONTH_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"(?:\s+\d{1,2})?(?:,\s*\d{4})?\b",
    re.IGNORECASE,
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "with",
}

GENERIC_SAFE_CUE_TOKENS = {
    "actor",
    "air",
    "aired",
    "album",
    "amount",
    "announced",
    "appearance",
    "appearances",
    "author",
    "authorship",
    "available",
    "birth",
    "birthplace",
    "book",
    "born",
    "broadcast",
    "came",
    "cast",
    "character",
    "city",
    "company",
    "construction",
    "control",
    "count",
    "country",
    "credit",
    "credited",
    "credits",
    "date",
    "definition",
    "duration",
    "early",
    "episode",
    "episodes",
    "event",
    "federal",
    "film",
    "function",
    "government",
    "group",
    "head",
    "hometown",
    "identity",
    "launch",
    "leader",
    "lead",
    "life",
    "location",
    "meaning",
    "number",
    "office",
    "origin",
    "out",
    "owner",
    "performer",
    "person",
    "place",
    "played",
    "power",
    "premiere",
    "premiered",
    "purpose",
    "region",
    "release",
    "released",
    "role",
    "scheduled",
    "season",
    "seasons",
    "setting",
    "singer",
    "source",
    "state",
    "streaming",
    "team",
    "term",
    "title",
    "total",
    "use",
    "used",
    "vocalist",
    "vocals",
    "writer",
    "written",
    "year",
}

GENERIC_CAPITALIZED = {
    "Answer",
    "Blank",
    "Context",
    "Document",
    "Documents",
    "Evidence",
    "Passage",
    "Passages",
    "Query",
    "Relevant",
    "The",
    "This",
}

ANSWER_ROLE_BY_TYPE = {
    "PERSON": "person, performer, author, leader, actor, or other named individual",
    "LOCATION": "place, city, country, region, address, or other location",
    "ORGANIZATION": "organization, company, agency, school, team, party, or group",
    "DATE": "date, year, season, or time period",
    "NUMBER": "number, count, amount, rank, measurement, or quantity",
    "TITLE": "title, work, song, book, film, album, show, or named creative work",
    "EVENT": "event, competition, incident, or named occurrence",
    "ENTITY": "entity requested by the question",
}

ANSWER_TYPE_TERMS = {
    "PERSON": ["person", "name", "identity"],
    "LOCATION": ["place", "location", "city", "country", "region"],
    "ORGANIZATION": ["organization", "company", "agency", "team", "group"],
    "DATE": ["date", "year", "time", "period"],
    "NUMBER": ["number", "count", "total", "amount"],
    "TITLE": ["title", "work", "song", "book", "film", "album", "show"],
    "EVENT": ["event", "incident", "competition"],
    "ENTITY": ["entity", "answer", "fact"],
}

SLOTLESS_ANSWER_TYPE_TERMS = {
    "PERSON": ["person", "name"],
    "LOCATION": ["place", "location", "city", "country", "region"],
    "ORGANIZATION": ["organization", "company", "agency", "team", "group"],
    "DATE": ["date", "year", "time", "period"],
    "NUMBER": ["number", "count", "total", "amount"],
    "TITLE": ["title", "work", "song", "book", "film", "album", "show"],
    "EVENT": ["event", "incident", "competition"],
    "ENTITY": ["profile", "overview"],
}

RELATION_RULES: list[tuple[re.Pattern[str], str, list[str]]] = [
    (re.compile(r"\b(?:sing|sings|sang|singer|song|vocals?|vocalist)\b", re.I), "performer / singer / vocals", ["singer", "performer", "vocals", "lead vocalist", "credited artist"]),
    (re.compile(r"\b(?:born|birth|birthplace)\b", re.I), "birthplace / born in", ["born", "birthplace", "birth place", "hometown", "early life"]),
    (re.compile(r"\b(?:episodes?|season)\b", re.I), "episode count / season", ["episodes", "season", "episode count", "number of episodes", "aired"]),
    (re.compile(r"\b(?:release|released|come out|came out|premiere|premiered)\b", re.I), "release date / premiere", ["released", "release date", "premiere", "came out", "aired"]),
    (re.compile(r"\b(?:originate|originated|origin|source)\b", re.I), "origin / source location", ["origin", "originate", "source", "came from", "derived from"]),
    (re.compile(r"\b(?:purpose|function|used for|use of)\b", re.I), "purpose / function", ["purpose", "function", "used for", "use", "role"]),
    (re.compile(r"\b(?:games?\s+has\s+.+?\s+played|played\s+in|appearances?)\b", re.I), "games played / appearances", ["games", "played", "played in", "appearances", "career", "season"]),
    (re.compile(r"\b(?:belongs?\s+to\s+which\s+part|part\s+of)\b", re.I), "part of / subdivision", ["part", "part of", "belongs to", "region", "division", "subdivision"]),
    (re.compile(r"\b(?:building|built|construction|constructing|site)\b", re.I), "construction site / location", ["building", "construction", "site", "located", "location", "new"]),
    (re.compile(r"\b(?:setting|set\s+in|takes?\s+place)\b", re.I), "setting / location", ["setting", "set in", "takes place", "location", "place"]),
    (re.compile(r"\b(?:take\s+over|took\s+over|control|annex|ceded|occupied)\b", re.I), "takeover / control date", ["take over", "control", "annexed", "ceded", "occupied", "island"]),
    (re.compile(r"\b(?:removed\s+from|remove\s+from)\b", re.I), "removal date", ["removed", "removed from", "books", "canon", "date"]),
    (re.compile(r"\b(?:stay\s+in\s+office|term|tenure|served)\b", re.I), "term length / tenure", ["term", "tenure", "office", "served", "minister", "length"]),
    (re.compile(r"\b(?:belong|belongs|belonged|owner|owned)\b", re.I), "ownership / belonged to", ["belonged", "owner", "owned by", "previous owner", "possession"]),
    (re.compile(r"\b(?:leader|president|chair|chairman|chief|head)\b", re.I), "leader / head of organization", ["leader", "head", "chair", "president", "party leader"]),
    (re.compile(r"\b(?:wrote|writer|author|written by)\b", re.I), "author / writer", ["author", "writer", "written by", "credited"]),
    (re.compile(r"\b(?:who\s+(?:play|plays|played)|portrays?|actor|actress|cast|character)\b", re.I), "cast / character", ["actor", "cast", "played", "character", "role"]),
    (re.compile(r"\b(?:located|location|where)\b", re.I), "location / located in", ["located", "location", "place", "city", "country"]),
    (re.compile(r"\b(?:meaning|definition|define|what is)\b", re.I), "definition / meaning", ["meaning", "definition", "refers to", "describes"]),
]

RELATION_CLASS_CUES = {
    "identity": ["name", "identity"],
    "person_role": ["role", "person"],
    "performer": ["performer", "singer", "vocalist"],
    "cast": ["cast", "actor", "portrays"],
    "creator": ["author", "writer", "creator"],
    "release_time": ["release date", "premiere", "aired"],
    "event_time": ["date", "year", "time"],
    "location": ["location", "place", "site"],
    "origin": ["origin", "source", "derived from"],
    "count": ["count", "number", "total"],
    "definition": ["meaning", "definition"],
    "ownership": ["owner", "owned by"],
    "membership": ["part of", "belongs to"],
    "comparison": ["comparison", "power"],
    "function": ["function", "purpose", "use"],
    "legal_rule": ["rule", "law", "opinion"],
    "event": ["event", "incident"],
    "other": [],
}

RELATION_CLASS_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:sing|sings|sang|singer|song|vocals?|vocalist|performer)\b", re.I), "performer"),
    (re.compile(r"\b(?:who\s+(?:play|plays|played)|portrays?|actor|actress|cast|character)\b", re.I), "cast"),
    (re.compile(r"\b(?:wrote|writer|author|written by|created|creator)\b", re.I), "creator"),
    (re.compile(r"\b(?:release|released|come out|came out|premiere|premiered|aired|air date)\b", re.I), "release_time"),
    (re.compile(r"\b(?:when|date|year|season|day|month|term|tenure|served)\b", re.I), "event_time"),
    (re.compile(r"\b(?:where|born|birthplace|located|location|city|country|state|site|set in|takes? place)\b", re.I), "location"),
    (re.compile(r"\b(?:originate|originated|origin|source|came from|derived from)\b", re.I), "origin"),
    (re.compile(r"\b(?:how many|how much|number|count|total|episodes?|seasons?|amount)\b", re.I), "count"),
    (re.compile(r"\b(?:meaning|definition|define|what is|refers to)\b", re.I), "definition"),
    (re.compile(r"\b(?:belong|belongs|belonged|owner|owned|possession)\b", re.I), "ownership"),
    (re.compile(r"\b(?:part of|belongs to which part|subdivision|division|member of)\b", re.I), "membership"),
    (re.compile(r"\b(?:more|less|compare|comparison|power|larger|smaller)\b", re.I), "comparison"),
    (re.compile(r"\b(?:purpose|function|used for|use of|role)\b", re.I), "function"),
    (re.compile(r"\b(?:law|legal|court|opinion|dissent|rule|canon|articles of confederation)\b", re.I), "legal_rule"),
    (re.compile(r"\b(?:event|war|battle|incident|competition)\b", re.I), "event"),
    (re.compile(r"\b(?:who|person|leader|president|chair|chief|head)\b", re.I), "person_role"),
]

RELATION_CLASS_ALIASES = {
    "birthplace": "location",
    "birth_place": "location",
    "birth_location": "location",
    "located_in": "location",
    "place": "location",
    "time": "event_time",
    "date": "event_time",
    "year": "event_time",
    "release": "release_time",
    "release_date": "release_time",
    "premiere": "release_time",
    "song_performer": "performer",
    "singer": "performer",
    "vocalist": "performer",
    "actor": "cast",
    "character": "cast",
    "writer": "creator",
    "author": "creator",
    "meaning": "definition",
    "owner": "ownership",
}

VALID_RETRIEVAL_POLICIES = {"anchor_only", "anchor_plus_one_cue", "query_preserve"}


@dataclass(frozen=True)
class QueryIntent:
    query: str
    answer_type: str
    slot: str
    anchors: tuple[str, ...]
    strong_anchors: tuple[str, ...]
    relation: str
    relation_terms: tuple[str, ...]


@dataclass(frozen=True)
class FormatValidation:
    ok: bool
    issues: tuple[str, ...]


@dataclass(frozen=True)
class ReformatAnchor:
    text: str
    role: str
    required: bool = True


@dataclass(frozen=True)
class ReformatView:
    name: str
    text: str
    uses_slot: bool


@dataclass(frozen=True)
class LeakageFreeReformatPackage:
    query: str
    answer_type: str
    answer_slot: str
    known_anchors: tuple[ReformatAnchor, ...]
    relation_name: str
    relation_confidence: str
    core_relation_terms: tuple[str, ...]
    retrieval_views: tuple[ReformatView, ...]
    validation: FormatValidation
    metadata: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        has_slot_view = any(view.uses_slot for view in self.retrieval_views)
        return {
            "query": self.query,
            "answer_type": self.answer_type,
            "answer_slot": self.answer_slot,
            "known_anchors": [asdict(anchor) for anchor in self.known_anchors],
            "relation_frame": {
                "name": self.relation_name,
                "confidence": self.relation_confidence,
                "core_relation_terms": list(self.core_relation_terms),
                "forbidden_answer_generation": True,
            },
            "retrieval_views": {view.name: view.text for view in self.retrieval_views},
            "constraints": {
                "must_preserve_anchors": True,
                "must_include_expected_slot": self.answer_slot if has_slot_view else False,
                "must_not_include_query_external_entities": True,
                "must_not_include_refusal": True,
            },
            "validation": {
                "ok": self.validation.ok,
                "issues": list(self.validation.issues),
            },
            "metadata": self.metadata or {},
        }


def infer_query_intent(query: str) -> QueryIntent:
    cleaned = normalize_space(query)
    answer_type = infer_answer_type(cleaned)
    anchors = extract_anchors(cleaned)
    strong_anchors = extract_strong_anchors(cleaned)
    relation, relation_terms = infer_relation(cleaned)
    return QueryIntent(
        query=cleaned,
        answer_type=answer_type,
        slot=SLOT_BY_TYPE[answer_type],
        anchors=tuple(anchors),
        strong_anchors=tuple(strong_anchors),
        relation=relation,
        relation_terms=tuple(relation_terms),
    )


def infer_answer_type(query: str) -> str:
    lower = query.lower()
    if re.search(r"\bhow\s+(?:many|much|long|far|old)\b", lower):
        return "NUMBER"
    if re.search(r"\b(?:what|which)\s+(?:year|date|season|day|month)\b", lower) or lower.startswith("when "):
        return "DATE"
    if lower.startswith("where ") or re.search(r"\b(?:which|what)\s+(?:city|country|state|province|place|location)\b", lower):
        return "LOCATION"
    if lower.startswith("who ") or re.search(r"\bwho\s+(?:is|was|plays|played|sings|sang|wrote)\b", lower):
        return "PERSON"
    if re.search(r"\b(?:which|what)\s+(?:company|organization|school|agency|team|party|band)\b", lower):
        return "ORGANIZATION"
    if re.search(r"\b(?:which|what)\s+(?:song|book|movie|film|album|show|novel|title)\b", lower):
        return "TITLE"
    if re.search(r"\b(?:which|what)\s+(?:event|war|battle|incident|competition)\b", lower):
        return "EVENT"
    return "ENTITY"


def infer_relation(query: str) -> tuple[str, list[str]]:
    for pattern, relation, terms in RELATION_RULES:
        if pattern.search(query):
            return relation, terms
    content_terms = []
    for token in TOKEN_RE.findall(query.lower()):
        if token not in STOPWORDS and len(token) >= 4 and token not in content_terms:
            content_terms.append(token)
    if not content_terms:
        content_terms = ["relevant", "fact", "answer"]
    return "query relation / requested fact", content_terms[:6]


def extract_anchors(query: str) -> list[str]:
    anchors: list[str] = []
    for value in extract_strong_anchors(query):
        append_unique(anchors, value)
    for token in TOKEN_RE.findall(query):
        lowered = token.lower()
        if lowered in STOPWORDS or len(token) < 4:
            continue
        append_unique(anchors, token)
    return anchors[:12]


def extract_strong_anchors(query: str) -> list[str]:
    anchors: list[str] = []
    for quoted in re.findall(r'"([^"]+)"|' + r"'([^']+)'", query):
        value = quoted[0] or quoted[1]
        if value.strip():
            append_unique(anchors, value.strip())
    for match in CODE_RE.findall(query):
        append_unique(anchors, match)
    for match in CAPITALIZED_SPAN_RE.findall(query):
        value = clean_span(match)
        if value and value not in GENERIC_CAPITALIZED and value.lower() not in STOPWORDS:
            append_unique(anchors, value)
    for match in NUMBER_RE.findall(query):
        append_unique(anchors, match)
    return anchors[:12]


def build_answer_blanked_query2doc(query: str) -> str:
    intent = infer_query_intent(query)
    anchor_text = anchor_phrase(intent)
    topic_text = topic_phrase(intent)
    terms = ", ".join(compact_terms(list(intent.relation_terms) + ANSWER_TYPE_TERMS[intent.answer_type])[:8])
    role = ANSWER_ROLE_BY_TYPE[intent.answer_type]
    slot = intent.slot
    if intent.answer_type == "PERSON":
        first = f"{slot} is the {role} connected to {anchor_text}."
    elif intent.answer_type == "LOCATION":
        first = f"{topic_text} associated with the location {slot}."
    elif intent.answer_type == "DATE":
        first = f"{topic_text} associated with the date or time {slot}."
    elif intent.answer_type == "NUMBER":
        first = f"{topic_text} associated with the number or count {slot}."
    elif intent.answer_type == "TITLE":
        first = f"{topic_text} associated with the title or work {slot}."
    else:
        first = f"{topic_text} associated with the requested answer {slot}."
    second = f"Relevant passages discuss {anchor_text}, {intent.relation}, and terms such as {terms}."
    third = f"The answer-bearing span is intentionally blanked as {slot} while query anchors are preserved."
    return sanitize_answer_blanked_query2doc(query, " ".join([first, second, third]))


def build_relation_keyword_query(query: str) -> str:
    intent = infer_query_intent(query)
    terms = compact_terms(list(intent.anchors) + list(intent.relation_terms) + ANSWER_TYPE_TERMS[intent.answer_type])
    return " ".join(terms) or intent.query


def build_lf_er_package(query: str) -> LeakageFreeReformatPackage:
    """Build a leakage-free evidence reformat package without corpus probing."""

    intent = infer_query_intent(query)
    frame = infer_lf_er_frame(intent)
    views = build_lf_er_views(intent, frame)
    validation = validate_lf_er_package(intent.query, views, intent.slot)
    if validation.ok:
        final_views = views
        final_validation = validation
    else:
        final_views = build_generic_lf_er_views(intent)
        final_validation = validate_lf_er_package(intent.query, final_views, intent.slot)
    return LeakageFreeReformatPackage(
        query=intent.query,
        answer_type=intent.answer_type,
        answer_slot=intent.slot,
        known_anchors=tuple(frame["anchors"]),
        relation_name=str(frame["name"]),
        relation_confidence=str(frame.get("confidence", "high")),
        core_relation_terms=tuple(str(term) for term in frame["terms"]),
        retrieval_views=tuple(final_views),
        validation=final_validation,
    )


def build_llm_lf_er_package(query: str, raw_payload: str, version: str = "v2") -> LeakageFreeReformatPackage:
    """Build an answer-free reformat package from an LLM-produced intent JSON.

    The LLM is allowed to identify query anchors and abstract relation cues, but
    the final retrieval text is rendered locally after strict query-support and
    specificity checks. This keeps the method general without letting a free-form
    pseudo-document introduce answer priors.
    """

    version = normalize_llm_reformat_version(version)
    intent = infer_query_intent(query)
    parse_issues: list[str] = []
    fallback_used = False
    try:
        payload = parse_llm_reformat_json(raw_payload)
    except ValueError as exc:
        payload = deterministic_llm_reformat_payload(query, version=version)
        parse_issues.append(f"json_parse_error:{exc}")
        fallback_used = True
    sanitized = sanitize_llm_reformat_payload(intent, payload, parse_issues, version=version)
    if not sanitized["anchors"] or not sanitized["must_keep_terms"]:
        fallback_payload = deterministic_llm_reformat_payload(query, version=version)
        sanitized = sanitize_llm_reformat_payload(
            intent,
            fallback_payload,
            [*parse_issues, "fallback_used:missing_anchor_or_context"],
            version=version,
        )
        fallback_used = True

    views = build_llm_lf_er_views(
        query=intent.query,
        anchors=list(sanitized["anchors"]),
        must_keep_terms=list(sanitized["must_keep_terms"]),
        safe_expansion_terms=list(sanitized["safe_expansion_terms"]),
        relation_name=str(sanitized["relation_name"]),
        confidence=str(sanitized["confidence"]),
        relation_class=str(sanitized["relation_class"]),
        retrieval_policy=str(sanitized["retrieval_policy"]),
        version=version,
    )
    validation = validate_llm_lf_er_package(intent.query, views)
    if not validation.ok:
        fallback_payload = deterministic_llm_reformat_payload(query, version=version)
        sanitized = sanitize_llm_reformat_payload(
            intent,
            fallback_payload,
            [*parse_issues, *validation.issues, "fallback_used:validation_failed"],
            version=version,
        )
        views = build_llm_lf_er_views(
            query=intent.query,
            anchors=list(sanitized["anchors"]),
            must_keep_terms=list(sanitized["must_keep_terms"]),
            safe_expansion_terms=list(sanitized["safe_expansion_terms"]),
            relation_name=str(sanitized["relation_name"]),
            confidence=str(sanitized["confidence"]),
            relation_class=str(sanitized["relation_class"]),
            retrieval_policy=str(sanitized["retrieval_policy"]),
            version=version,
        )
        validation = validate_llm_lf_er_package(intent.query, views)
        fallback_used = True

    return LeakageFreeReformatPackage(
        query=intent.query,
        answer_type=str(sanitized["answer_type"]),
        answer_slot=SLOT_BY_TYPE[str(sanitized["answer_type"])],
        known_anchors=tuple(sanitized["anchors"]),
        relation_name=str(sanitized["relation_name"]),
        relation_confidence=str(sanitized["confidence"]),
        core_relation_terms=tuple(str(term) for term in sanitized["safe_expansion_terms"]),
        retrieval_views=tuple(views),
        validation=validation,
        metadata={
            "builder": f"llm_structured_reformat_{version}",
            "fallback_used": fallback_used,
            "relation_class": str(sanitized["relation_class"]),
            "retrieval_policy": str(sanitized["retrieval_policy"]),
            "sanitizer_issues": list(sanitized["issues"]),
            "raw_payload": raw_payload,
        },
    )


def deterministic_llm_reformat_payload(query: str, version: str = "v2") -> dict[str, object]:
    intent = infer_query_intent(query)
    frame = infer_lf_er_frame(intent)
    anchors = [
        {"text": anchor.text, "role": anchor.role}
        for anchor in frame["anchors"]
    ]
    query_terms = extract_query_context_terms(intent.query)
    confidence = str(frame.get("confidence", "low"))
    relation_budget = {"high": 4, "medium": 2}.get(confidence, 0)
    safe_terms = dense_safe_relation_terms([str(term) for term in frame["terms"]])[:relation_budget]
    if normalize_llm_reformat_version(version) == "v2":
        relation_class = infer_relation_class(intent.query, str(frame["name"]), [str(term) for term in frame["terms"]])
        retrieval_policy = infer_retrieval_policy(intent, relation_class, confidence)
        return {
            "answer_type": intent.answer_type,
            "anchors": anchors,
            "query_focus_terms": compact_terms([*query_terms, *[anchor["text"] for anchor in anchors]])[:12],
            "relation_class": relation_class,
            "relation_confidence": confidence,
            "retrieval_policy": retrieval_policy,
        }
    return {
        "answer_type": intent.answer_type,
        "anchors": anchors,
        "relation": {
            "name": str(frame["name"]),
            "confidence": confidence,
            "safe_cues": safe_terms,
        },
        "must_keep_terms": compact_terms([*query_terms, *[anchor["text"] for anchor in anchors]])[:12],
        "safe_expansion_terms": safe_terms,
    }


def deterministic_llm_reformat_payload_text(query: str, version: str = "v2") -> str:
    return json.dumps(deterministic_llm_reformat_payload(query, version=version), ensure_ascii=False)


def parse_llm_reformat_json(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("missing_json_object")
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid_json:{exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("json_root_not_object")
    return payload


def normalize_llm_reformat_version(version: str | None) -> str:
    normalized = (version or "v2").strip().lower()
    if normalized not in {"v1", "v2"}:
        return "v2"
    return normalized


def sanitize_llm_reformat_payload(
    intent: QueryIntent,
    payload: dict[str, Any],
    initial_issues: list[str] | None = None,
    version: str = "v2",
) -> dict[str, object]:
    version = normalize_llm_reformat_version(version)
    issues = list(initial_issues or [])
    answer_type = str(payload.get("answer_type", intent.answer_type)).upper()
    if answer_type not in SLOT_BY_TYPE:
        issues.append(f"invalid_answer_type:{answer_type}")
        answer_type = intent.answer_type

    relation_payload = payload.get("relation", {})
    if not isinstance(relation_payload, dict):
        relation_payload = {}
        issues.append("relation_not_object")
    relation_name = normalize_relation_name(str(relation_payload.get("name") or payload.get("relation_name") or "query_relation"))
    confidence = str(
        relation_payload.get("confidence")
        or payload.get("relation_confidence")
        or payload.get("confidence")
        or "low"
    ).lower()
    if confidence not in {"high", "medium", "low"}:
        issues.append(f"invalid_confidence:{confidence}")
        confidence = "low"

    anchors = sanitize_llm_anchors(intent.query, payload.get("anchors"), issues)
    if not anchors:
        anchors = infer_fallback_anchors(intent)
        issues.append("fallback_anchors")

    focus_values = [
        *collect_text_values(payload.get("query_focus_terms")),
        *collect_text_values(payload.get("must_keep_terms")),
    ]
    must_keep_terms = sanitize_query_supported_terms(
        intent.query,
        focus_values,
        issues,
        field_name="must_keep_terms",
        max_terms=14,
    )
    if not must_keep_terms:
        must_keep_terms = compact_terms([*extract_query_context_terms(intent.query), *[anchor.text for anchor in anchors]])[:14]
        issues.append("fallback_must_keep_terms")

    candidate_safe_terms = [
        *collect_text_values(payload.get("safe_expansion_terms")),
        *collect_text_values(relation_payload.get("safe_cues")),
        *collect_text_values(relation_payload.get("evidence_cues")),
    ]
    relation_class = sanitize_relation_class(
        payload.get("relation_class") or relation_payload.get("class") or relation_payload.get("relation_class"),
        intent,
        relation_name,
        candidate_safe_terms,
        issues,
    )
    retrieval_policy = sanitize_retrieval_policy(
        payload.get("retrieval_policy") or relation_payload.get("retrieval_policy"),
        intent,
        relation_class,
        confidence,
        issues,
    )
    if version == "v2":
        safe_terms = controlled_relation_cues(
            relation_class=relation_class,
            confidence=confidence,
            retrieval_policy=retrieval_policy,
        )
        if candidate_safe_terms:
            issues.append("ignored_freeform_safe_terms_v2")
    else:
        safe_terms = sanitize_safe_expansion_terms(
            intent.query,
            candidate_safe_terms,
            issues,
            confidence=confidence,
        )
    if not safe_terms and confidence == "high":
        confidence = "medium"

    return {
        "answer_type": answer_type,
        "relation_name": relation_name,
        "confidence": confidence,
        "relation_class": relation_class,
        "retrieval_policy": retrieval_policy,
        "anchors": tuple(anchors),
        "must_keep_terms": tuple(must_keep_terms),
        "safe_expansion_terms": tuple(safe_terms),
        "issues": tuple(issues),
    }


def sanitize_llm_anchors(query: str, value: Any, issues: list[str]) -> list[ReformatAnchor]:
    anchors: list[ReformatAnchor] = []
    items = value if isinstance(value, list) else []
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("text", ""))
            role_parts = [
                str(item.get("importance", "")),
                str(item.get("role", "query_anchor")),
            ]
            role = normalize_relation_name("_".join(part for part in role_parts if part)) or "query_anchor"
        else:
            text = str(item)
            role = "query_anchor"
        text = clean_anchor_text(text)
        if not text:
            continue
        if not text_supported_by_query(text, query):
            issues.append(f"dropped_query_external_anchor:{text}")
            continue
        append_anchor(anchors, ReformatAnchor(text=text, role=role, required=True))
    pruned = prune_nested_anchors(query, anchors)
    if len(pruned) < len(anchors):
        issues.append("pruned_nested_anchors")
    return pruned[:8]


def prune_nested_anchors(query: str, anchors: list[ReformatAnchor]) -> list[ReformatAnchor]:
    if len(anchors) <= 1:
        return anchors
    ranked = sorted(
        enumerate(anchors),
        key=lambda pair: (
            -len(TOKEN_RE.findall(pair[1].text)),
            -len(pair[1].text),
            pair[0],
        ),
    )
    kept: list[tuple[int, ReformatAnchor]] = []
    kept_norms: list[str] = []
    for index, anchor in ranked:
        normalized = normalize_for_match(anchor.text)
        if not normalized:
            continue
        if any(normalized == existing or normalized in existing for existing in kept_norms):
            continue
        kept.append((index, anchor))
        kept_norms.append(normalized)
    kept.sort(key=lambda pair: query_anchor_position(query, pair[1].text, pair[0]))
    return [anchor for _, anchor in kept]


def query_anchor_position(query: str, anchor: str, fallback: int) -> tuple[int, int]:
    position = normalize_for_match(query).find(normalize_for_match(anchor))
    if position < 0:
        position = 100000 + fallback
    return (position, fallback)


def sanitize_query_supported_terms(
    query: str,
    values: list[str],
    issues: list[str],
    field_name: str,
    max_terms: int,
) -> list[str]:
    terms: list[str] = []
    for value in values:
        cleaned = clean_span(value)
        if not cleaned or len(TOKEN_RE.findall(cleaned)) > 6:
            continue
        if not text_supported_by_query(cleaned, query):
            issues.append(f"dropped_query_external_{field_name}:{cleaned}")
            continue
        append_unique(terms, cleaned)
        if len(terms) >= max_terms:
            break
    return terms


def sanitize_safe_expansion_terms(
    query: str,
    values: list[str],
    issues: list[str],
    confidence: str,
) -> list[str]:
    terms: list[str] = []
    budget = {"high": 4, "medium": 2}.get(confidence, 0)
    if budget <= 0:
        return terms
    for value in values:
        cleaned = clean_span(value)
        if not cleaned:
            continue
        token_count = len(TOKEN_RE.findall(cleaned))
        if token_count == 0 or token_count > 4:
            issues.append(f"dropped_long_safe_term:{cleaned}")
            continue
        if not text_supported_by_query(cleaned, query) and not is_generic_safe_cue(cleaned):
            issues.append(f"dropped_non_generic_safe_term:{cleaned}")
            continue
        if looks_like_specific_answer_candidate(cleaned, query):
            issues.append(f"dropped_specific_safe_term:{cleaned}")
            continue
        if REFUSAL_RE.search(cleaned):
            issues.append(f"dropped_refusal_safe_term:{cleaned}")
            continue
        append_unique(terms, cleaned.lower())
        if len(terms) >= budget:
            break
    return terms


def sanitize_relation_class(
    value: Any,
    intent: QueryIntent,
    relation_name: str,
    cue_terms: list[str],
    issues: list[str],
) -> str:
    if value is not None:
        normalized = normalize_relation_name(str(value))
        normalized = RELATION_CLASS_ALIASES.get(normalized, normalized)
        if normalized in RELATION_CLASS_CUES:
            return normalized
        issues.append(f"invalid_relation_class:{normalized}")
    return infer_relation_class(intent.query, relation_name, cue_terms)


def infer_relation_class(query: str, relation_name: str = "", cue_terms: list[str] | None = None) -> str:
    combined = " ".join([query, relation_name.replace("_", " "), " ".join(cue_terms or [])])
    normalized = normalize_relation_name(relation_name)
    if normalized in RELATION_CLASS_ALIASES:
        return RELATION_CLASS_ALIASES[normalized]
    if normalized in RELATION_CLASS_CUES:
        return normalized
    for pattern, relation_class in RELATION_CLASS_RULES:
        if pattern.search(combined):
            return relation_class
    return "other"


def sanitize_retrieval_policy(
    value: Any,
    intent: QueryIntent,
    relation_class: str,
    confidence: str,
    issues: list[str],
) -> str:
    if value is not None:
        normalized = normalize_relation_name(str(value))
        if normalized in VALID_RETRIEVAL_POLICIES:
            return normalized
        issues.append(f"invalid_retrieval_policy:{normalized}")
    return infer_retrieval_policy(intent, relation_class, confidence)


def infer_retrieval_policy(intent: QueryIntent, relation_class: str, confidence: str) -> str:
    if confidence == "low" or relation_class == "other":
        return "query_preserve"
    if relation_class in {"identity", "definition", "comparison", "legal_rule"}:
        return "query_preserve"
    if len(intent.strong_anchors) >= 2 and relation_class in {"performer", "cast", "creator", "ownership"}:
        return "anchor_plus_one_cue"
    if intent.answer_type in {"DATE", "NUMBER", "LOCATION"}:
        return "anchor_plus_one_cue"
    return "anchor_only"


def controlled_relation_cues(
    relation_class: str,
    confidence: str,
    retrieval_policy: str,
) -> list[str]:
    if confidence == "low" or retrieval_policy in {"anchor_only", "query_preserve"}:
        return []
    budget = 1 if retrieval_policy == "anchor_plus_one_cue" else 0
    return RELATION_CLASS_CUES.get(relation_class, [])[:budget]


def build_llm_lf_er_views(
    query: str,
    anchors: list[ReformatAnchor],
    must_keep_terms: list[str],
    safe_expansion_terms: list[str],
    relation_name: str,
    confidence: str,
    relation_class: str = "other",
    retrieval_policy: str = "query_preserve",
    version: str = "v2",
) -> list[ReformatView]:
    version = normalize_llm_reformat_version(version)
    anchor_texts = [anchor.text for anchor in anchors]
    query_terms = compact_terms(must_keep_terms)
    if version == "v2":
        query_terms = drop_anchor_contained_terms(query_terms, anchor_texts)
    safe_terms = compact_terms(safe_expansion_terms)
    if version == "v2":
        safe_terms = controlled_relation_cues(
            relation_class=relation_class,
            confidence=confidence,
            retrieval_policy=retrieval_policy,
        )
    dense_terms = dense_safe_relation_terms(safe_terms)
    if confidence == "low":
        dense_terms = []

    anchor_view = " ".join(compact_terms([query, *anchor_texts])) or query
    if version == "v2" and retrieval_policy == "query_preserve":
        intent_terms_view = " ".join(compact_terms([query, *anchor_texts[:3], *query_terms[:8]])) or query
        dense_expansion_terms = compact_terms(anchor_texts[:2])
        dense_view_terms = []
        bm25_extra_terms = compact_terms([*anchor_texts[:4], *query_terms[:8]])
        relation_text = ""
    elif version == "v2":
        intent_terms_view = " ".join(compact_terms([query, *anchor_texts[:3], *query_terms[:8], *safe_terms[:1]])) or query
        dense_expansion_terms = compact_terms([*dense_terms[:1], *anchor_texts[:1]])
        dense_view_terms = compact_terms(dense_terms[:1])
        bm25_extra_terms = compact_terms([*anchor_texts[:4], *query_terms[:10], *safe_terms[:1]])
        relation_text = " ".join(RELATION_CLASS_CUES.get(relation_class, [])[:1])
    else:
        intent_terms_view = " ".join(compact_terms([query, *query_terms, *safe_terms])) or query
        dense_expansion_terms = compact_terms([*anchor_texts[:4], *query_terms[:8], *dense_terms[:3]])
        dense_view_terms = dense_expansion_terms
        bm25_extra_terms = compact_terms([*anchor_texts, *query_terms, *safe_terms])
        relation_text = relation_name.replace("_", " ")
    dense_expansion = " ".join(dense_expansion_terms) or " ".join(anchor_texts[:2]) or query
    dense_view = normalize_space(" ".join([query, *dense_view_terms])) or query
    bm25_view = " ".join(compact_terms([query, query, *bm25_extra_terms])) or query
    corpus_safe_terms = safe_terms[:1] if version == "v2" else safe_terms[:4]
    corpus_style = normalize_space(
        " ".join(
            compact_terms(
                [
                    query,
                    *anchor_texts[:4],
                    *query_terms[:8],
                    *corpus_safe_terms,
                    relation_text if confidence != "low" else "",
                ]
            )
        )
    )
    return [
        ReformatView("llm_anchor_view", anchor_view, uses_slot=False),
        ReformatView("llm_intent_terms_view", intent_terms_view, uses_slot=False),
        ReformatView("llm_dense_view", dense_view, uses_slot=False),
        ReformatView("llm_dense_expansion_view", dense_expansion, uses_slot=False),
        ReformatView("llm_bm25_view", bm25_view, uses_slot=False),
        ReformatView("llm_corpus_style_view", corpus_style or intent_terms_view, uses_slot=False),
    ]


def drop_anchor_contained_terms(terms: list[str], anchors: list[str]) -> list[str]:
    anchor_norms = [normalize_for_match(anchor) for anchor in anchors]
    filtered: list[str] = []
    for term in terms:
        normalized = normalize_for_match(term)
        if any(normalized == anchor_norm or normalized in anchor_norm.split() for anchor_norm in anchor_norms):
            continue
        append_unique(filtered, term)
    return filtered


def validate_llm_lf_er_package(query: str, views: list[ReformatView]) -> FormatValidation:
    required = {
        "llm_anchor_view",
        "llm_intent_terms_view",
        "llm_dense_view",
        "llm_dense_expansion_view",
        "llm_bm25_view",
        "llm_corpus_style_view",
    }
    issues: list[str] = []
    names = {view.name for view in views}
    for name in sorted(required - names):
        issues.append(f"missing_view:{name}")
    for view in views:
        text = normalize_space(view.text)
        if not text:
            issues.append(f"empty_view:{view.name}")
        if REFUSAL_RE.search(text):
            issues.append(f"refusal:{view.name}")
        if SLOT_RE.search(text):
            issues.append(f"unexpected_slot:{view.name}")
        for span in find_query_external_specific_spans(query, text):
            issues.append(f"unblanked_specific:{view.name}:{span}")
        if view.name != "llm_dense_expansion_view" and normalize_for_match(query) not in normalize_for_match(text):
            issues.append(f"query_not_preserved:{view.name}")
    return FormatValidation(not issues, tuple(issues))


def build_lf_er_views(intent: QueryIntent, frame: dict[str, object]) -> list[ReformatView]:
    anchors = [anchor.text for anchor in frame["anchors"]]
    frame_terms = [str(term) for term in frame["terms"]]
    slotless_frame_terms = [str(term) for term in frame["slotless_terms"]]
    confidence = str(frame.get("confidence", "high"))
    query_terms = extract_query_context_terms(intent.query)
    answer_terms = ANSWER_TYPE_TERMS[intent.answer_type]
    slotless_answer_terms = SLOTLESS_ANSWER_TYPE_TERMS[intent.answer_type]
    anchor_text = anchor_join([anchor for anchor in frame["anchors"]])
    relation_name = str(frame["name"]).replace("_", " ")
    slot_templates = relation_slot_templates(str(frame["name"]), anchors, intent.slot)
    slotless_templates = relation_slotless_templates(str(frame["name"]), anchors)

    anchor_view = " ".join(compact_terms([intent.query, *anchors])) or intent.query
    keyword_view = " ".join(
        compact_terms([intent.query, *anchors, *query_terms, *frame_terms, *slotless_answer_terms])
    ) or intent.query
    forward = normalize_space(
        f"{intent.query}. "
        f"{anchor_text} {relation_name} {intent.slot}. "
        f"{frame['forward']}"
    )
    inverse = normalize_space(
        f"{intent.query}. "
        f"{relation_name} for {anchor_text}: {intent.slot}. "
        f"{frame['inverse']}"
    )
    slotless_terms = compact_terms(
        [intent.query, *anchors, *query_terms, *slotless_frame_terms, *frame_terms, *slotless_answer_terms]
    )
    slotless = " ".join(slotless_terms) or keyword_view
    bm25_terms = compact_terms(
        [
            intent.query,
            intent.query,
            *anchors,
            *query_terms,
            *frame_terms,
            *slotless_frame_terms,
            *slotless_templates,
            *slotless_answer_terms,
        ]
    )
    bm25_view = " ".join(bm25_terms) or slotless
    corpus_style = build_corpus_style_view(intent.query, anchors, frame_terms, slotless_templates, slotless_answer_terms)
    dense_safe_expansion = build_dense_safe_expansion_view(
        anchors=anchors,
        frame_terms=frame_terms,
        slotless_frame_terms=slotless_frame_terms,
        confidence=confidence,
    )
    dense_safe_view = build_dense_safe_view(intent.query, dense_safe_expansion)
    dense_view = normalize_space(f"{intent.query}. {dense_safe_expansion}.")
    template_view = normalize_space(
        f"{intent.query}. "
        f"{' '.join(slotless_templates)}"
    )
    return [
        ReformatView("anchor_view", anchor_view, uses_slot=False),
        ReformatView("relation_keyword_view", keyword_view, uses_slot=False),
        ReformatView("evidence_forward_view", forward, uses_slot=True),
        ReformatView("evidence_inverse_view", inverse, uses_slot=True),
        ReformatView("slotless_evidence_view", slotless, uses_slot=False),
        ReformatView("bm25_field_view", bm25_view, uses_slot=False),
        ReformatView("dense_natural_view", dense_view, uses_slot=False),
        ReformatView("dense_safe_view", dense_safe_view, uses_slot=False),
        ReformatView("dense_safe_expansion_view", dense_safe_expansion, uses_slot=False),
        ReformatView("template_expansion_view", template_view, uses_slot=False),
        ReformatView("corpus_style_view", corpus_style, uses_slot=False),
    ]


def build_generic_lf_er_views(intent: QueryIntent) -> list[ReformatView]:
    anchors = [anchor.text for anchor in infer_fallback_anchors(intent)]
    anchor_text = anchor_phrase(intent)
    query_terms = extract_query_context_terms(intent.query)
    slotless_answer_terms = SLOTLESS_ANSWER_TYPE_TERMS[intent.answer_type]
    terms = compact_terms([intent.query, *anchors, *query_terms, *intent.relation_terms, *slotless_answer_terms])
    slot_templates = relation_slot_templates("generic_relation", anchors or [anchor_text], intent.slot)
    slotless_templates = relation_slotless_templates("generic_relation", anchors or [anchor_text])
    anchor_view = " ".join(compact_terms([intent.query, *anchors])) or intent.query
    keyword_view = " ".join(terms) or intent.query
    forward = normalize_space(
        f"{intent.query}. "
        f"{anchor_text} {intent.relation} {intent.slot}. "
        f"{anchor_text} is associated with {intent.slot}."
    )
    inverse = normalize_space(
        f"{intent.query}. "
        f"{intent.relation} for {anchor_text}: {intent.slot}."
    )
    slotless = " ".join(compact_terms([intent.query, *anchors, *query_terms, *intent.relation_terms])) or keyword_view
    bm25_view = normalize_space(
        f"{' '.join(compact_terms([intent.query, intent.query, *anchors, *query_terms, *intent.relation_terms, *slotless_templates, *slotless_answer_terms]))}"
    )
    corpus_style = build_corpus_style_view(intent.query, anchors, list(intent.relation_terms), slotless_templates, slotless_answer_terms)
    dense_safe_expansion = build_dense_safe_expansion_view(
        anchors=anchors,
        frame_terms=list(intent.relation_terms),
        slotless_frame_terms=[],
        confidence="low",
    )
    dense_safe_view = build_dense_safe_view(intent.query, dense_safe_expansion)
    dense_view = normalize_space(f"{intent.query}. {dense_safe_expansion}.")
    template_view = normalize_space(
        f"{intent.query}. "
        f"{' '.join(slotless_templates)}"
    )
    return [
        ReformatView("anchor_view", anchor_view, uses_slot=False),
        ReformatView("relation_keyword_view", keyword_view, uses_slot=False),
        ReformatView("evidence_forward_view", forward, uses_slot=True),
        ReformatView("evidence_inverse_view", inverse, uses_slot=True),
        ReformatView("slotless_evidence_view", slotless, uses_slot=False),
        ReformatView("bm25_field_view", bm25_view, uses_slot=False),
        ReformatView("dense_natural_view", dense_view, uses_slot=False),
        ReformatView("dense_safe_view", dense_safe_view, uses_slot=False),
        ReformatView("dense_safe_expansion_view", dense_safe_expansion, uses_slot=False),
        ReformatView("template_expansion_view", template_view, uses_slot=False),
        ReformatView("corpus_style_view", corpus_style, uses_slot=False),
    ]


def build_corpus_style_view(
    query: str,
    anchors: list[str],
    frame_terms: list[str],
    slotless_templates: list[str],
    answer_terms: list[str],
) -> str:
    """A leakage-free pseudo passage: document-like terms, no answer slot."""

    compact = compact_terms([query, *anchors, *frame_terms, *answer_terms])
    template_text = " ".join(compact_terms(slotless_templates)[:2])
    segments = [" ".join(compact[:28]), template_text]
    return normalize_space(". ".join(segment for segment in segments if segment))


def build_dense_safe_view(query: str, expansion: str) -> str:
    """Dense-oriented reformat: anchor-first and short enough to avoid semantic drift."""

    return normalize_space(" ".join(compact_terms([query, expansion])))


def build_dense_safe_expansion_view(
    anchors: list[str],
    frame_terms: list[str],
    slotless_frame_terms: list[str],
    confidence: str,
) -> str:
    """Build a dense expansion with an explicit drift budget.

    Dense encoders embed the whole text into one vector, so low-confidence
    relation guesses should not add semantic terms. Exact frame matches can add
    a few relation terms; generic frames back off to anchors only.
    """

    expansion_terms = compact_terms(anchors)
    relation_budget = {"high": 6, "medium": 3}.get(confidence, 0)
    if relation_budget:
        safe_terms = dense_safe_relation_terms([*frame_terms, *slotless_frame_terms])
        expansion_terms.extend(safe_terms[:relation_budget])
    return " ".join(compact_terms(expansion_terms)[:24])


def dense_safe_relation_terms(terms: list[str]) -> list[str]:
    drift_markers = {
        "answer",
        "associated",
        "fact",
        "generic",
        "history details",
        "identity",
        "overview",
        "profile",
        "query",
        "requested",
    }
    safe: list[str] = []
    for term in terms:
        cleaned = normalize_space(str(term)).strip(" .,:;")
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in drift_markers:
            continue
        if any(marker in lowered for marker in ("requested fact", "generic relation", "associated with")):
            continue
        if len(TOKEN_RE.findall(cleaned)) > 4:
            continue
        append_unique(safe, cleaned)
    return safe


def infer_lf_er_frame(intent: QueryIntent) -> dict[str, object]:
    query = intent.query
    lower = query.lower()
    slot = intent.slot

    match = re.search(r"\bwho\s+(?:sings|sang|performs|performed)\s+(.+?)\s+by\s+(.+)$", query, re.I)
    if match:
        work = clean_anchor_text(match.group(1))
        artist = clean_anchor_text(match.group(2))
        anchors = [ReformatAnchor(work, "work_title"), ReformatAnchor(artist, "artist_or_group")]
        terms = ["singer", "vocalist", "vocals", "performer", "credited", "lead vocals", "sung by"]
        return lf_frame(
            name="song_performer",
            anchors=anchors,
            terms=terms,
            forward=f"{quote_if_phrase(work)} by {artist} features vocals by {slot}.",
            inverse=f"{slot} is credited as the singer or vocalist for {quote_if_phrase(work)} by {artist}.",
            slotless_terms=[work, artist, "features vocals", "credited singer", "vocalist", "sung by"],
        )

    match = re.search(r"\bwho\s+(?:performed|performs)\s+(.+)$", query, re.I)
    if match and re.search(r"\b(?:c[-\s]?section|cesarean|caesarean|surgery|operation|procedure|transplant)\b", match.group(1), re.I):
        procedure = clean_anchor_text(match.group(1))
        anchors = [ReformatAnchor(procedure, "procedure_or_event")]
        terms = ["performed", "procedure", "surgery", "operation", "surgeon", "physician", "first"]
        return lf_frame(
            name="procedure_performer",
            anchors=anchors,
            terms=terms,
            forward=f"{procedure} was performed by {slot}.",
            inverse=f"{slot} performed {procedure}.",
            slotless_terms=[procedure, "performed", "procedure", "surgery", "operation", "surgeon", "physician"],
        )

    match = re.search(r"\bwho\s+(?:sings|sang)\s+(.+)$", query, re.I)
    if match:
        work = clean_anchor_text(match.group(1))
        anchors = [ReformatAnchor(work, "work_title")]
        terms = ["singer", "vocalist", "vocals", "performer", "credited", "lead vocals", "sung by"]
        return lf_frame(
            name="song_performer",
            anchors=anchors,
            terms=terms,
            forward=f"{quote_if_phrase(work)} features vocals by {slot}.",
            inverse=f"{slot} is credited as the singer or vocalist for {quote_if_phrase(work)}.",
            slotless_terms=[work, "features vocals", "credited singer", "vocalist", "sung by"],
        )

    match = re.search(r"\bwho\s+(?:plays|played|portrays|portrayed)\s+(.+?)\s+(?:in|on|from)\s+(.+)$", query, re.I)
    if match:
        character = clean_anchor_text(match.group(1))
        work = clean_anchor_text(match.group(2))
        anchors = [ReformatAnchor(character, "character"), ReformatAnchor(work, "work_title")]
        terms = ["cast", "actor", "played by", "portrayed", "character", "role"]
        return lf_frame(
            name="cast_character",
            anchors=anchors,
            terms=terms,
            forward=f"The character {character} in {quote_if_phrase(work)} is played by {slot}.",
            inverse=f"{slot} portrays {character} in {quote_if_phrase(work)}.",
            slotless_terms=[character, work, "cast", "actor", "played by", "character", "role"],
        )

    match = re.search(r"\bhow\s+many\s+episodes\s+are\s+in\s+(.+?)\s+season\s+(\d+)\b", query, re.I)
    if match:
        work = clean_anchor_text(match.group(1))
        season = clean_anchor_text(match.group(2))
        anchors = [ReformatAnchor(work, "work_title"), ReformatAnchor(season, "season_number")]
        terms = ["episodes", "episode count", "season", "aired", "total episodes", "number of episodes"]
        return lf_frame(
            name="episode_count",
            anchors=anchors,
            terms=terms,
            forward=f"Season {season} of {work} contains {slot} episodes.",
            inverse=f"{work} season {season} aired with {slot} episodes.",
            slotless_terms=[work, f"season {season}", "episodes", "episode count", "total episodes", "aired"],
        )

    match = re.search(r"\bhow\s+many\s+seasons\s+of\s+(.+?)\s+(?:are|were|is)\s+on\s+(.+)$", query, re.I)
    if match:
        work = clean_anchor_text(match.group(1))
        platform = clean_anchor_text(match.group(2))
        anchors = [ReformatAnchor(work, "work_title"), ReformatAnchor(platform, "platform_or_source")]
        terms = ["seasons", "available", "streaming", "on", "number of seasons", "episodes"]
        return lf_frame(
            name="season_availability_count",
            anchors=anchors,
            terms=terms,
            forward=f"{work} has {slot} seasons available on {platform}.",
            inverse=f"{slot} is the number of seasons of {work} on {platform}.",
            slotless_terms=[work, platform, "seasons", "available", "streaming", "number of seasons"],
        )

    match = re.search(r"\bwhen\s+(?:does|did|will)\s+(.+?)\s+(?:come\s+out|release|premiere|air)\b", query, re.I)
    if match:
        work = clean_anchor_text(match.group(1))
        anchors = [ReformatAnchor(work, "work_title")]
        terms = ["release date", "premiere", "came out", "announced", "scheduled", "aired"]
        return lf_frame(
            name="release_date",
            anchors=anchors,
            terms=terms,
            forward=f"{work} is scheduled to come out on {slot}.",
            inverse=f"The release date for {work} is {slot}.",
            slotless_terms=[work, "release date", "premiere", "came out", "announced", "scheduled"],
        )

    match = re.search(r"\bwhere\s+(?:are|is|were|was)\s+(?:they\s+)?(?:building|constructing)\s+(?:the\s+)?(?:new\s+)?(.+)$", query, re.I)
    if match:
        project = clean_anchor_text(match.group(1))
        anchors = [ReformatAnchor(project, "project_or_facility")]
        terms = ["building", "construction", "site", "location", "located", "new facility"]
        return lf_frame(
            name="construction_location",
            anchors=anchors,
            terms=terms,
            forward=f"{project} is being built in {slot}.",
            inverse=f"{slot} is the construction site for {project}.",
            slotless_terms=[project, "building", "construction", "site", "location", "new", "facility"],
        )

    match = re.search(r"\bwhere\s+(?:is|was)\s+the\s+setting\s+(?:for|of)\s+(.+)$", query, re.I)
    if match:
        work = clean_anchor_text(match.group(1))
        anchors = [ReformatAnchor(work, "work_title")]
        terms = ["setting", "set in", "takes place", "location", "place"]
        return lf_frame(
            name="setting_location",
            anchors=anchors,
            terms=terms,
            forward=f"{work} is set in {slot}.",
            inverse=f"{slot} is the setting of {work}.",
            slotless_terms=[work, "setting", "set in", "takes place", "location", "place"],
        )

    match = re.search(r"\bwhen\s+did\s+(.+?)\s+(?:take\s+over|annex|occupy|gain\s+control\s+of)\s+(.+)$", query, re.I)
    if match:
        actor = clean_anchor_text(match.group(1))
        subject = clean_anchor_text(match.group(2))
        anchors = [ReformatAnchor(actor, "actor"), ReformatAnchor(subject, "controlled_entity")]
        terms = ["take over", "control", "annexed", "occupied", "ceded", "sovereignty"]
        return lf_frame(
            name="takeover_date",
            anchors=anchors,
            terms=terms,
            forward=f"{actor} took control of {subject} in {slot}.",
            inverse=f"{slot} is when {actor} took control of {subject}.",
            slotless_terms=[actor, subject, "take over", "control", "annexed", "occupied", "ceded", "sovereignty"],
        )

    match = re.search(r"\bwhen\s+were\s+(.+?)\s+removed\s+from\s+(?:the\s+)?(.+)$", query, re.I)
    if match:
        removed = clean_anchor_text(match.group(1))
        source = clean_anchor_text(match.group(2))
        anchors = [ReformatAnchor(removed, "removed_item"), ReformatAnchor(source, "source_work")]
        terms = ["removed", "removed from", "books", "canon", "date", "revision"]
        return lf_frame(
            name="removal_date",
            anchors=anchors,
            terms=terms,
            forward=f"{removed} were removed from {source} in {slot}.",
            inverse=f"{slot} is when {removed} were removed from {source}.",
            slotless_terms=[removed, source, "removed", "removed from", "books", "canon", "date", "revision"],
        )

    match = re.search(r"\bwhere\s+(?:did|does|do|was|were)\s+(.+?)\s+(?:originate|originated|come\s+from|came\s+from)\b", query, re.I)
    if match:
        subject = clean_anchor_text(match.group(1))
        anchors = [ReformatAnchor(subject, "subject")]
        terms = ["origin", "originate", "source", "came from", "derived from", "location"]
        return lf_frame(
            name="origin_location",
            anchors=anchors,
            terms=terms,
            forward=f"{subject} originated from {slot}.",
            inverse=f"{slot} is the origin or source location of {subject}.",
            slotless_terms=[subject, "origin", "originate", "source", "came from", "derived from", "location"],
        )

    match = re.search(r"\bwhere\s+(?:was|is)\s+(.+?)\s+born\b", query, re.I)
    if match:
        subject = clean_anchor_text(match.group(1))
        anchors = [ReformatAnchor(subject, "subject")]
        terms = ["born", "birthplace", "place of birth", "early life", "hometown"]
        return lf_frame(
            name="birthplace",
            anchors=anchors,
            terms=terms,
            forward=f"{subject} was born in {slot}.",
            inverse=f"{slot} is listed as the birthplace of {subject}.",
            slotless_terms=[subject, "born", "birthplace", "place of birth", "early life"],
        )

    match = re.search(r"\bwhat\s+is\s+the\s+purpose\s+of\s+(.+)$", query, re.I)
    if match:
        subject = clean_anchor_text(match.group(1))
        anchors = [ReformatAnchor(subject, "subject")]
        terms = ["purpose", "function", "used for", "use", "role", "designed to"]
        return lf_frame(
            name="purpose_function",
            anchors=anchors,
            terms=terms,
            forward=f"The purpose or function of {subject} is {slot}.",
            inverse=f"{slot} describes what {subject} is used for.",
            slotless_terms=[subject, "purpose", "function", "used for", "use", "role", "designed to"],
        )

    match = re.search(r"\b(.+?)\s+belongs\s+to\s+which\s+part\s+of\s+(.+)$", query, re.I)
    if match:
        subject = clean_anchor_text(match.group(1))
        whole = clean_anchor_text(match.group(2))
        anchors = [ReformatAnchor(subject, "part"), ReformatAnchor(whole, "whole")]
        terms = ["part", "part of", "belongs to", "division", "region", "subdivision"]
        return lf_frame(
            name="part_of",
            anchors=anchors,
            terms=terms,
            forward=f"{subject} is part of {slot} within {whole}.",
            inverse=f"{slot} is the part of {whole} that contains {subject}.",
            slotless_terms=[subject, whole, "part", "part of", "belongs to", "division", "region", "subdivision"],
        )

    match = re.search(r"\bwho\s+did\s+(.+?)\s+belong\s+to\s+before\s+(.+)$", query, re.I)
    if match:
        subject = clean_anchor_text(match.group(1))
        later_owner = clean_anchor_text(match.group(2))
        anchors = [ReformatAnchor(subject, "owned_entity"), ReformatAnchor(later_owner, "later_owner")]
        terms = ["belonged", "owner", "owned by", "previous owner", "possession", "before"]
        return lf_frame(
            name="previous_owner",
            anchors=anchors,
            terms=terms,
            forward=f"{subject} belonged to {slot} before {later_owner}.",
            inverse=f"{slot} was the previous owner of {subject} before {later_owner}.",
            slotless_terms=[subject, later_owner, "belonged", "owner", "owned by", "previous owner", "before"],
        )

    match = re.search(r"\bwho\s+(?:is|was)\s+the\s+(?:leader|head|president|chair|chairman)\s+of\s+(?:the\s+)?(.+)$", query, re.I)
    if match:
        organization = clean_anchor_text(match.group(1))
        anchors = [ReformatAnchor(organization, "organization")]
        terms = ["leader", "head", "president", "chair", "party leader", "led by"]
        return lf_frame(
            name="organization_leader",
            anchors=anchors,
            terms=terms,
            forward=f"{slot} is the leader of {organization}.",
            inverse=f"{organization} is led by {slot}.",
            slotless_terms=[organization, "leader", "head", "president", "chair", "led by"],
        )

    match = re.search(r"\bhow\s+many\s+(.+?)\s+games\s+has\s+(?:the\s+)?(.+?)\s+played\s+in\b", query, re.I)
    if match:
        competition = clean_anchor_text(match.group(1))
        subject = clean_anchor_text(match.group(2))
        anchors = [ReformatAnchor(competition, "competition"), ReformatAnchor(subject, "participant")]
        terms = ["games", "played in", "appearances", "participated", "total", "career"]
        return lf_frame(
            name="games_played_count",
            anchors=anchors,
            terms=terms,
            forward=f"{subject} has played in {slot} {competition} games.",
            inverse=f"{slot} is the number of {competition} games played by {subject}.",
            slotless_terms=[competition, subject, "games", "played in", "appearances", "participated", "total"],
        )

    match = re.search(r"\bhow\s+long\s+(.+?)\s+(?:stay|stayed|serve|served)\s+in\s+office\s+(.+)$", query, re.I)
    if match:
        office_holder = clean_anchor_text(match.group(1))
        context = clean_anchor_text(match.group(2))
        anchors = [ReformatAnchor(office_holder, "office_holder"), ReformatAnchor(context, "office_context")]
        terms = ["term", "tenure", "office", "served", "minister", "length"]
        return lf_frame(
            name="tenure_length",
            anchors=anchors,
            terms=terms,
            forward=f"{office_holder} served in office in {context} for {slot}.",
            inverse=f"{slot} is the length of time {office_holder} served in office in {context}.",
            slotless_terms=[office_holder, context, "term", "tenure", "office", "served", "minister", "length"],
        )

    match = re.search(r"\bwho\s+(?:wrote|authored)\s+(.+)$", query, re.I)
    if match:
        work = clean_anchor_text(match.group(1))
        anchors = [ReformatAnchor(work, "work_title")]
        terms = ["author", "writer", "written by", "authored", "credited"]
        return lf_frame(
            name="author_work",
            anchors=anchors,
            terms=terms,
            forward=f"{work} was written by {slot}.",
            inverse=f"{slot} is credited as the author or writer of {work}.",
            slotless_terms=[work, "author", "writer", "written by", "authored", "credited"],
        )

    match = re.search(r"\bwhen\s+were\s+(.+?)\s+(?:books\s+)?written\b", query, re.I)
    if match:
        work = clean_anchor_text(match.group(1))
        anchors = [ReformatAnchor(work, "work_or_series")]
        terms = ["written", "authored", "publication", "date", "books", "writing"]
        return lf_frame(
            name="writing_date",
            anchors=anchors,
            terms=terms,
            forward=f"{work} was written in {slot}.",
            inverse=f"{slot} is when {work} was written.",
            slotless_terms=[work, "written", "authored", "publication", "books", "writing"],
        )

    match = re.search(r"\bwhere\s+(?:is|are)\s+(.+?)\s+(?:located|found|situated)\b", query, re.I)
    if match:
        subject = clean_anchor_text(match.group(1))
        anchors = [ReformatAnchor(subject, "subject")]
        terms = ["located", "location", "place", "region", "country", "situated"]
        return lf_frame(
            name="location",
            anchors=anchors,
            terms=terms,
            forward=f"{subject} is located in {slot}.",
            inverse=f"{slot} is the location of {subject}.",
            slotless_terms=[subject, "located", "location", "place", "region", "country"],
        )

    if "meaning" in lower or lower.startswith("what is "):
        anchors = infer_fallback_anchors(intent)
        anchor_text = anchor_join(anchors)
        terms = ["meaning", "definition", "refers to", "describes", "term"]
        return lf_frame(
            name="definition",
            anchors=anchors,
            terms=terms,
            forward=f"{anchor_text} refers to {slot}.",
            inverse=f"{slot} is the meaning or definition of {anchor_text}.",
            slotless_terms=[anchor_text, "meaning", "definition", "refers to", "describes"],
            confidence="medium",
        )

    anchors = infer_fallback_anchors(intent)
    anchor_text = anchor_join(anchors)
    terms = compact_terms([*intent.relation_terms, *SLOTLESS_ANSWER_TYPE_TERMS[intent.answer_type]])
    return lf_frame(
        name="generic_relation",
        anchors=anchors,
        terms=terms,
        forward=f"{anchor_text} is associated with {slot}.",
        inverse=f"{slot} is the requested fact associated with {anchor_text}.",
        slotless_terms=[anchor_text, *terms],
        confidence="low",
    )


def lf_frame(
    name: str,
    anchors: list[ReformatAnchor],
    terms: list[str],
    forward: str,
    inverse: str,
    slotless_terms: list[str],
    confidence: str = "high",
) -> dict[str, object]:
    deduped_anchors = dedupe_anchors([anchor for anchor in anchors if anchor.text])
    return {
        "name": name,
        "anchors": deduped_anchors,
        "terms": compact_terms(terms),
        "forward": normalize_space(forward),
        "inverse": normalize_space(inverse),
        "slotless_terms": compact_terms(slotless_terms),
        "confidence": confidence,
    }


def infer_fallback_anchors(intent: QueryIntent) -> list[ReformatAnchor]:
    anchors = list(intent.strong_anchors) or list(intent.anchors[:4])
    if not anchors:
        anchors = [intent.query]
    return dedupe_anchors([ReformatAnchor(anchor, "query_anchor") for anchor in anchors])


def extract_query_context_terms(query: str) -> list[str]:
    """Keep non-answer query context so reformats do not erase lexical evidence."""

    terms: list[str] = []
    for token in TOKEN_RE.findall(query):
        lowered = token.lower()
        if lowered in STOPWORDS or len(token) < 3:
            continue
        append_unique(terms, token)
    return terms[:16]


def relation_slot_templates(relation_name: str, anchors: list[str], slot: str) -> list[str]:
    anchor = quote_if_phrase(anchor_join_text(anchors))
    first = quote_if_phrase(anchors[0]) if anchors else "the query subject"
    second = quote_if_phrase(anchors[1]) if len(anchors) > 1 else ""
    if relation_name == "song_performer":
        return [
            f"{anchor} features vocals by {slot}.",
            f"{slot} is credited as the singer or vocalist for {anchor}.",
            f"{slot} performed {anchor}.",
            f"{anchor} was sung by {slot}.",
        ]
    if relation_name == "procedure_performer":
        return [
            f"{anchor} was performed by {slot}.",
            f"{slot} performed {anchor}.",
            f"procedure history for {anchor} names {slot}.",
        ]
    if relation_name == "cast_character":
        work = second or "the work"
        return [
            f"the character {first} in {work} is played by {slot}.",
            f"{slot} portrays {first} in {work}.",
            f"cast information for {work} lists {slot} as {first}.",
        ]
    if relation_name == "episode_count":
        return [
            f"{anchor} contains {slot} episodes.",
            f"{anchor} aired with {slot} episodes.",
            f"the number of episodes for {anchor} is {slot}.",
        ]
    if relation_name == "season_availability_count":
        return [
            f"{anchor} has {slot} seasons available.",
            f"the number of seasons for {anchor} is {slot}.",
            f"{anchor} streaming availability lists {slot} seasons.",
        ]
    if relation_name == "release_date":
        return [
            f"{anchor} was released on {slot}.",
            f"the release date for {anchor} is {slot}.",
            f"{anchor} premiered on {slot}.",
        ]
    if relation_name == "construction_location":
        return [
            f"{anchor} is being built in {slot}.",
            f"the construction site for {anchor} is {slot}.",
            f"{anchor} construction is located in {slot}.",
        ]
    if relation_name == "setting_location":
        return [
            f"{anchor} is set in {slot}.",
            f"the setting of {anchor} is {slot}.",
            f"{anchor} takes place in {slot}.",
        ]
    if relation_name == "takeover_date":
        return [
            f"{anchor} control changed in {slot}.",
            f"{slot} is when {anchor} was taken over.",
            f"sovereignty or control of {anchor} changed in {slot}.",
        ]
    if relation_name == "removal_date":
        return [
            f"{anchor} was removed in {slot}.",
            f"the removal date for {anchor} is {slot}.",
            f"{anchor} removal occurred in {slot}.",
        ]
    if relation_name in {"birthplace", "origin_location", "location"}:
        return [
            f"{anchor} is associated with the location {slot}.",
            f"{slot} is the place linked to {anchor}.",
            f"location evidence for {anchor} names {slot}.",
        ]
    if relation_name == "purpose_function":
        return [
            f"the purpose or function of {anchor} is {slot}.",
            f"{slot} describes what {anchor} is used for.",
            f"{anchor} is designed to {slot}.",
        ]
    if relation_name == "previous_owner":
        tail = f" before {second}" if second else ""
        return [
            f"{first} belonged to {slot}{tail}.",
            f"{slot} was the previous owner of {first}{tail}.",
            f"ownership history for {first} lists {slot}{tail}.",
        ]
    if relation_name == "part_of":
        return [
            f"{first} is part of {slot} within {second}.",
            f"{slot} contains or includes {first} in {second}.",
            f"{first} belongs to the {slot} part of {second}.",
        ]
    if relation_name == "organization_leader":
        return [
            f"{slot} is the leader of {anchor}.",
            f"{anchor} is led by {slot}.",
            f"leadership information for {anchor} names {slot}.",
        ]
    if relation_name == "games_played_count":
        return [
            f"{anchor} played in {slot} games.",
            f"the total games played by {anchor} is {slot}.",
            f"{anchor} appearances total {slot}.",
        ]
    if relation_name == "tenure_length":
        return [
            f"{anchor} served in office for {slot}.",
            f"the tenure length for {anchor} was {slot}.",
            f"{anchor} term in office lasted {slot}.",
        ]
    if relation_name == "author_work":
        return [
            f"{anchor} was written by {slot}.",
            f"{slot} is credited as the author or writer of {anchor}.",
            f"authorship information for {anchor} names {slot}.",
        ]
    if relation_name == "writing_date":
        return [
            f"{anchor} was written in {slot}.",
            f"the writing date for {anchor} is {slot}.",
            f"{slot} is when {anchor} was written.",
        ]
    if relation_name == "definition":
        return [
            f"{anchor} refers to {slot}.",
            f"{slot} is the meaning or definition of {anchor}.",
            f"definition evidence explains {anchor} as {slot}.",
        ]
    return [
        f"{anchor} is associated with {slot}.",
        f"{slot} is the requested fact associated with {anchor}.",
        f"relevant evidence links {anchor} to {slot}.",
    ]


def relation_slotless_templates(relation_name: str, anchors: list[str]) -> list[str]:
    anchor = anchor_join_text(anchors)
    first = anchors[0] if anchors else "query subject"
    second = anchors[1] if len(anchors) > 1 else ""
    if relation_name == "song_performer":
        return [
            f"{anchor} singer vocalist performer vocals lead vocals credited artist",
            f"{anchor} features vocals sung by recorded by",
            f"{anchor} song single album track vocals performer",
        ]
    if relation_name == "procedure_performer":
        return [
            f"{anchor} performed procedure surgery operation surgeon physician",
            f"{anchor} first procedure medical history",
            f"{anchor} operation performed by surgeon",
        ]
    if relation_name == "cast_character":
        return [
            f"{first} {second} cast actor played by portrayed character role",
            f"{second} cast list character {first} actor role",
            f"{first} portrayed in {second}",
        ]
    if relation_name == "episode_count":
        return [
            f"{anchor} season episodes number of episodes episode count aired",
            f"{anchor} total episodes season overview",
            f"{anchor} episode list",
        ]
    if relation_name == "season_availability_count":
        return [
            f"{anchor} seasons available streaming number of seasons",
            f"{anchor} streaming availability seasons",
            f"{anchor} episodes seasons on platform",
        ]
    if relation_name == "release_date":
        return [
            f"{anchor} release date released premiere premiered came out aired",
            f"{anchor} scheduled announced launch date",
            f"{anchor} release information",
        ]
    if relation_name == "construction_location":
        return [
            f"{anchor} building construction site located location new facility",
            f"{anchor} planned construction project site city",
            f"{anchor} new building location",
        ]
    if relation_name == "setting_location":
        return [
            f"{anchor} setting set in takes place location place",
            f"{anchor} fictional setting location",
            f"{anchor} story setting place",
        ]
    if relation_name == "takeover_date":
        return [
            f"{anchor} take over control annexed occupied ceded sovereignty date",
            f"{anchor} control changed history",
            f"{anchor} acquired possession island",
        ]
    if relation_name == "removal_date":
        return [
            f"{anchor} removed from books canon revision date",
            f"{anchor} removed deletion history",
            f"{anchor} removal date",
        ]
    if relation_name in {"birthplace", "origin_location", "location"}:
        return [
            f"{anchor} location place city country region origin source",
            f"{anchor} born birthplace originated from located in",
            f"{anchor} geography location information",
        ]
    if relation_name == "purpose_function":
        return [
            f"{anchor} purpose function use used for role designed to",
            f"{anchor} mechanism function purpose",
            f"{anchor} application use description",
        ]
    if relation_name == "previous_owner":
        return [
            f"{first} {second} belonged owner owned by previous owner before ownership history",
            f"{first} possession owner transfer history",
            f"{first} former owner before {second}",
        ]
    if relation_name == "part_of":
        return [
            f"{first} {second} part of belongs to division region subdivision",
            f"{first} located in part section of {second}",
            f"{first} administrative division region part",
        ]
    if relation_name == "organization_leader":
        return [
            f"{anchor} leader head president chair chairman led by leadership",
            f"{anchor} party leader organization head",
            f"{anchor} leadership office holder",
        ]
    if relation_name == "games_played_count":
        return [
            f"{anchor} games played in appearances participated total career",
            f"{anchor} played games participation count",
            f"{anchor} appearances games record",
        ]
    if relation_name == "tenure_length":
        return [
            f"{anchor} term tenure office served length minister",
            f"{anchor} served in office duration",
            f"{anchor} office holder tenure term",
        ]
    if relation_name == "author_work":
        return [
            f"{anchor} author writer written by authored credited",
            f"{anchor} authorship writer credits",
            f"{anchor} book work written credited author",
        ]
    if relation_name == "writing_date":
        return [
            f"{anchor} written authored publication date books writing",
            f"{anchor} writing history publication",
            f"{anchor} books written date",
        ]
    if relation_name == "definition":
        return [
            f"{anchor} meaning definition refers to describes term",
            f"{anchor} definition explanation meaning",
            f"{anchor} concept term refers to",
        ]
    return [
        f"{anchor}",
        f"{anchor} overview",
        f"{anchor} history details",
    ]


def anchor_join_text(anchors: list[str]) -> str:
    values = compact_terms(anchors)
    if not values:
        return "the query"
    return " and ".join(values[:4])


def validate_lf_er_package(query: str, views: list[ReformatView], expected_slot: str) -> FormatValidation:
    issues: list[str] = []
    if not views:
        issues.append("missing_views")
    view_names = {view.name for view in views}
    required = {
        "anchor_view",
        "relation_keyword_view",
        "evidence_forward_view",
        "evidence_inverse_view",
        "slotless_evidence_view",
        "bm25_field_view",
        "dense_natural_view",
        "dense_safe_view",
        "dense_safe_expansion_view",
        "template_expansion_view",
        "corpus_style_view",
    }
    for name in sorted(required - view_names):
        issues.append(f"missing_view:{name}")
    for view in views:
        text = normalize_space(view.text)
        if not text:
            issues.append(f"empty_view:{view.name}")
        if REFUSAL_RE.search(text):
            issues.append(f"refusal:{view.name}")
        for slot in re.findall(r"\[[A-Za-z_]+\]", text):
            if slot not in VALID_SLOTS:
                issues.append(f"invalid_slot:{view.name}:{slot}")
        if view.uses_slot and expected_slot not in text:
            issues.append(f"missing_expected_slot:{view.name}:{expected_slot}")
        if not view.uses_slot and SLOT_RE.search(text):
            issues.append(f"unexpected_slot:{view.name}")
        for span in find_query_external_specific_spans(query, text):
            issues.append(f"unblanked_specific:{view.name}:{span}")
    return FormatValidation(not issues, tuple(issues))


def sanitize_answer_blanked_query2doc(query: str, text: str) -> str:
    intent = infer_query_intent(query)
    candidate = normalize_generated_text(text)
    if not candidate or REFUSAL_RE.search(candidate):
        return deterministic_fallback(intent)
    candidate = mask_query_external_dates(query, candidate, intent)
    candidate = mask_query_external_numbers(query, candidate, intent)
    candidate = mask_query_external_capitalized_spans(query, candidate, intent)
    candidate = collapse_slots(candidate)
    candidate = normalize_space(candidate)
    validation = validate_answer_blanked_format(query, candidate)
    if validation.ok:
        return candidate
    return deterministic_fallback(intent)


def validate_answer_blanked_format(query: str, text: str) -> FormatValidation:
    intent = infer_query_intent(query)
    issues: list[str] = []
    candidate = normalize_space(text)
    if not candidate:
        issues.append("empty")
        return FormatValidation(False, tuple(issues))
    if REFUSAL_RE.search(candidate):
        issues.append("contains_refusal")
    if not SLOT_RE.search(candidate):
        issues.append("missing_slot")
    if intent.slot not in candidate:
        issues.append(f"missing_expected_slot:{intent.slot}")
    for slot in re.findall(r"\[[A-Za-z_]+\]", candidate):
        if slot not in VALID_SLOTS:
            issues.append(f"invalid_slot:{slot}")
    normalized_candidate = normalize_for_match(candidate)
    for anchor in intent.strong_anchors:
        if normalize_for_match(anchor) not in normalized_candidate:
            issues.append(f"missing_anchor:{anchor}")
    for span in find_query_external_specific_spans(query, candidate):
        issues.append(f"unblanked_specific:{span}")
    return FormatValidation(not issues, tuple(issues))


def find_query_external_specific_spans(query: str, text: str) -> list[str]:
    spans: list[str] = []
    normalized_query = normalize_for_match(query)
    scan_text = SLOT_RE.sub(" ", text)
    for match in MONTH_RE.findall(scan_text):
        if normalize_for_match(match) not in normalized_query:
            append_unique(spans, match)
    for match in NUMBER_RE.findall(scan_text):
        if normalize_for_match(match) not in normalized_query:
            append_unique(spans, match)
    for match in CAPITALIZED_SPAN_RE.findall(scan_text):
        value = clean_span(match)
        if not value or value in GENERIC_CAPITALIZED:
            continue
        if not span_supported_by_query(value, normalized_query):
            append_unique(spans, value)
    return spans


def span_supported_by_query(value: str, normalized_query: str) -> bool:
    normalized = normalize_for_match(value)
    if not normalized:
        return True
    if normalized in normalized_query:
        return True
    query_tokens = set(TOKEN_RE.findall(normalized_query))
    span_tokens = TOKEN_RE.findall(normalized)
    return bool(span_tokens) and all(token in query_tokens for token in span_tokens)


def deterministic_fallback(intent: QueryIntent) -> str:
    anchor_text = anchor_phrase(intent)
    topic_text = topic_phrase(intent)
    relation_terms = ", ".join(compact_terms(list(intent.relation_terms) + ANSWER_TYPE_TERMS[intent.answer_type])[:8])
    slot = intent.slot
    return normalize_space(
        f"{topic_text} associated with the requested answer {slot}. "
        f"Relevant passages discuss {anchor_text}, {intent.relation}, and terms such as {relation_terms}. "
        f"The answer-bearing span is intentionally blanked as {slot}."
    )


def normalize_generated_text(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in cleaned.split("\n"):
        line = LABEL_RE.sub("", line.strip(" -\t"))
        if line:
            lines.append(line)
    return normalize_space(" ".join(lines))


def mask_query_external_dates(query: str, text: str, intent: QueryIntent) -> str:
    normalized_query = normalize_for_match(query)
    replacement = intent.slot if intent.answer_type == "DATE" else "[DATE]"

    def replace(match: re.Match[str]) -> str:
        value = match.group(0)
        if normalize_for_match(value) in normalized_query:
            return value
        return replacement

    return MONTH_RE.sub(replace, text)


def mask_query_external_numbers(query: str, text: str, intent: QueryIntent) -> str:
    normalized_query = normalize_for_match(query)
    replacement = intent.slot if intent.answer_type == "NUMBER" else "[NUMBER]"

    def replace(match: re.Match[str]) -> str:
        value = match.group(0)
        if normalize_for_match(value) in normalized_query:
            return value
        return replacement

    return NUMBER_RE.sub(replace, text)


def mask_query_external_capitalized_spans(query: str, text: str, intent: QueryIntent) -> str:
    normalized_query = normalize_for_match(query)
    replacement = intent.slot if intent.answer_type in {
        "PERSON",
        "LOCATION",
        "ORGANIZATION",
        "TITLE",
        "EVENT",
        "ENTITY",
    } else "[ENTITY]"

    def replace(match: re.Match[str]) -> str:
        if match.start() > 0 and text[match.start() - 1] == "[" and match.end() < len(text) and text[match.end()] == "]":
            return match.group(0)
        value = clean_span(match.group(0))
        if not value or value in GENERIC_CAPITALIZED or value in VALID_SLOTS:
            return match.group(0)
        if normalize_for_match(value) in normalized_query:
            return match.group(0)
        return replacement

    return CAPITALIZED_SPAN_RE.sub(replace, text)


def collapse_slots(text: str) -> str:
    previous = None
    current = text
    while previous != current:
        previous = current
        current = re.sub(r"(\[[A-Z]+\])(?:\s*,?\s*(?:and\s+)?\1)+", r"\1", current)
        current = re.sub(r"\[\[([A-Z]+)\]\]", r"[\1]", current)
    return current


def anchor_phrase(intent: QueryIntent) -> str:
    anchors = compact_terms(list(intent.strong_anchors) or list(intent.anchors))
    if anchors:
        return " and ".join(anchors[:4])
    return f"the query '{intent.query}'"


def topic_phrase(intent: QueryIntent) -> str:
    anchors = compact_terms(list(intent.strong_anchors) or list(intent.anchors))
    if len(anchors) > 1:
        return f"The query anchors {' and '.join(anchors[:4])} are"
    if len(anchors) == 1:
        return f"{anchors[0]} is"
    return f"The query '{intent.query}' is"


def clean_anchor_text(value: str) -> str:
    cleaned = clean_span(value)
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    return normalize_space(cleaned)


def quote_if_phrase(value: str) -> str:
    cleaned = clean_anchor_text(value)
    if " " in cleaned and not (cleaned.startswith('"') and cleaned.endswith('"')):
        return f'"{cleaned}"'
    return cleaned


def anchor_join(anchors: list[ReformatAnchor]) -> str:
    values = compact_terms([anchor.text for anchor in anchors])
    if not values:
        return "the query"
    return " and ".join(values[:4])


def dedupe_anchors(anchors: list[ReformatAnchor]) -> list[ReformatAnchor]:
    deduped: list[ReformatAnchor] = []
    seen: set[str] = set()
    for anchor in anchors:
        text = clean_anchor_text(anchor.text)
        normalized = normalize_for_match(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(ReformatAnchor(text=text, role=anchor.role, required=anchor.required))
    return deduped


def compact_terms(terms: list[str]) -> list[str]:
    compacted: list[str] = []
    for term in terms:
        value = normalize_space(term)
        if not value:
            continue
        if value.lower() in STOPWORDS:
            continue
        append_unique(compacted, value)
    return compacted


def clean_span(value: str) -> str:
    return normalize_space(value.strip(" ,.;:!?()[]{}\"'"))


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_for_match(text: str) -> str:
    return normalize_space(re.sub(r"[^a-z0-9_]+", " ", (text or "").lower()))


def collect_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            if isinstance(item, str):
                values.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("term") or item.get("cue") or item.get("value")
                if text is not None:
                    values.append(str(text))
        return values
    return []


def normalize_relation_name(value: str) -> str:
    normalized = normalize_for_match(value).replace(" ", "_")
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized[:64] or "query_relation"


def text_supported_by_query(value: str, query: str) -> bool:
    normalized_value = normalize_for_match(value)
    normalized_query = normalize_for_match(query)
    if not normalized_value:
        return False
    if normalized_value in normalized_query:
        return True
    query_tokens = set(TOKEN_RE.findall(normalized_query))
    value_tokens = TOKEN_RE.findall(normalized_value)
    return bool(value_tokens) and all(token in query_tokens for token in value_tokens)


def looks_like_specific_answer_candidate(value: str, query: str) -> bool:
    cleaned = clean_span(value)
    if not cleaned:
        return False
    if text_supported_by_query(cleaned, query):
        return False
    if CODE_RE.search(cleaned) or MONTH_RE.search(cleaned) or NUMBER_RE.search(cleaned):
        return True
    if CAPITALIZED_SPAN_RE.fullmatch(cleaned):
        return True
    tokens = TOKEN_RE.findall(cleaned)
    if tokens and any(token[:1].isupper() for token in tokens):
        return True
    return False


def is_generic_safe_cue(value: str) -> bool:
    tokens = [
        token
        for token in TOKEN_RE.findall(normalize_for_match(value))
        if token not in STOPWORDS
    ]
    return bool(tokens) and all(token in GENERIC_SAFE_CUE_TOKENS for token in tokens)


def append_anchor(values: list[ReformatAnchor], value: ReformatAnchor) -> None:
    normalized = normalize_for_match(value.text)
    if not normalized:
        return
    if any(normalize_for_match(existing.text) == normalized for existing in values):
        return
    values.append(value)


def append_unique(values: list[str], value: str) -> None:
    normalized = normalize_for_match(value)
    if not normalized:
        return
    if any(normalize_for_match(existing) == normalized for existing in values):
        return
    values.append(value)
