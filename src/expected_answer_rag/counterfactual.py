from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from expected_answer_rag.datasets import Document, Query, RetrievalDataset
from expected_answer_rag.leakage import CAPITALIZED_RE, QUOTED_TITLE_RE, normalize_text
import spacy

_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


ORG_HINTS = {"inc", "corp", "company", "university", "college", "institute", "agency", "team", "band", "committee"}
LOC_HINTS = {"city", "state", "country", "region", "province", "county", "lake", "river", "mount", "mountain", "street", "avenue"}

FIRST_PARTS = [
    "Mira",
    "Taren",
    "Nora",
    "Levan",
    "Sorin",
    "Alto",
    "Kira",
    "Varen",
    "Milo",
    "Celia",
]
LAST_PARTS = [
    "Halden",
    "Varos",
    "Norwick",
    "Talor",
    "Merin",
    "Caspel",
    "Rovik",
    "Selden",
    "Maris",
    "Dorin",
]
LOC_PARTS = ["Varos", "Halden", "Merrow", "Calder", "Lunor", "Norwick", "Tavren", "Sorel"]
ORG_PARTS = ["Institute", "Division", "Foundation", "Collective", "Council", "Systems", "Works", "Group"]


@dataclass(frozen=True)
class CounterfactualBuildResult:
    dataset: RetrievalDataset
    alias_table: Mapping[str, Mapping[str, str]]
    validation: Mapping[str, object]


def build_entity_counterfactual_dataset(
    dataset: RetrievalDataset,
    alias_style: str = "natural",
    include_values: bool = False,
    seed: int = 13,
) -> CounterfactualBuildResult:
    rng = random.Random(seed)
    spans = collect_renamable_spans(dataset, include_values=include_values)
    alias_table = build_alias_table(spans, alias_style=alias_style, rng=rng)
    renamed_corpus = [_rename_document(doc, alias_table) for doc in dataset.corpus]
    renamed_queries = [_rename_query(query, alias_table) for query in dataset.queries]
    variant_name = f"{dataset.name}__entity_counterfactual_{alias_style}"
    if include_values:
        variant_name += "_with_values"
    metadata = {
        **dict(dataset.metadata),
        "variant": "entity_counterfactual",
        "alias_style": alias_style,
        "include_values": include_values,
        "seed": seed,
        "alias_table_size": len(alias_table),
    }
    renamed_dataset = RetrievalDataset(
        name=variant_name,
        corpus=renamed_corpus,
        queries=renamed_queries,
        qrels=dataset.qrels,
        metadata=metadata,
    )
    validation = validate_counterfactual_dataset(dataset, renamed_dataset, alias_table)
    return CounterfactualBuildResult(dataset=renamed_dataset, alias_table=alias_table, validation=validation)


def export_counterfactual_artifacts(
    result: CounterfactualBuildResult,
    output_dir: str | Path,
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "alias_table.json").write_text(json.dumps(result.alias_table, indent=2, ensure_ascii=False), encoding="utf-8")
    (root / "validation.json").write_text(json.dumps(result.validation, indent=2, ensure_ascii=False), encoding="utf-8")
    return root


def collect_renamable_spans(dataset: RetrievalDataset, include_values: bool = False) -> dict[str, str]:
    spans: dict[str, str] = {}
    for query in dataset.queries:
        _collect_from_text(query.text, spans, include_values)
        for answer in query.all_answer_strings:
            if include_values or not _looks_like_value(answer):
                spans.setdefault(answer, infer_entity_type(answer, context=query.text))
    for document in dataset.corpus:
        _collect_from_text(document.title, spans, include_values)
        _collect_from_text(document.text, spans, include_values)
    return spans


def build_alias_table(
    spans: Mapping[str, str],
    alias_style: str,
    rng: random.Random,
) -> dict[str, dict[str, str]]:
    alias_table: dict[str, dict[str, str]] = {}
    counters: dict[str, int] = {}
    for source in sorted(spans, key=lambda value: (-len(value), value.lower())):
        entity_type = spans[source]
        counters[entity_type] = counters.get(entity_type, 0) + 1
        alias_table[source] = {
            "alias": generate_alias(entity_type, counters[entity_type], alias_style=alias_style, rng=rng),
            "type": entity_type,
        }
    return alias_table


def validate_counterfactual_dataset(
    original: RetrievalDataset,
    renamed: RetrievalDataset,
    alias_table: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    original_corpus_text = "\n".join(doc.text for doc in original.corpus)
    original_query_text = "\n".join(query.text for query in original.queries)
    renamed_corpus_text = "\n".join(doc.text for doc in renamed.corpus)
    renamed_query_text = "\n".join(query.text for query in renamed.queries)
    original_text = "\n".join([original_corpus_text, original_query_text])
    renamed_text = "\n".join([renamed_corpus_text, renamed_query_text])
    residue_corpus = _residual_mentions(alias_table, renamed_corpus_text)
    residue_queries = _residual_mentions(alias_table, renamed_query_text)
    answer_preserved = 0
    answer_total = 0
    answer_missing_query_ids: list[str] = []
    support_preserved = 0
    support_total = 0
    support_missing_query_ids: list[str] = []
    qrel_doc_map = {doc.doc_id: doc for doc in renamed.corpus}
    for query in renamed.queries:
        if not query.answers:
            relevant = [qrel_doc_map[doc_id] for doc_id in renamed.qrels.get(query.query_id, {}) if doc_id in qrel_doc_map]
        else:
            answer_total += 1
            relevant = [qrel_doc_map[doc_id] for doc_id in renamed.qrels.get(query.query_id, {}) if doc_id in qrel_doc_map]
            relevant_text = normalize_text("\n".join(doc.text for doc in relevant))
            if any(normalize_text(answer) in relevant_text for answer in query.answers):
                answer_preserved += 1
            else:
                answer_missing_query_ids.append(query.query_id)
        if not query.supporting_facts:
            continue
        support_total += 1
        relevant_text = normalize_text("\n".join(doc.text for doc in relevant))
        if any(normalize_text(fact) in relevant_text for fact in query.supporting_facts if normalize_text(fact)):
            support_preserved += 1
        else:
            support_missing_query_ids.append(query.query_id)
    return {
        "replacement_coverage_estimate": _replacement_coverage_estimate(original_text, renamed_text, alias_table),
        "residual_original_mentions": sorted(set(residue_corpus + residue_queries))[:50],
        "residual_original_mentions_in_corpus": residue_corpus[:50],
        "residual_original_mentions_in_queries": residue_queries[:50],
        "residual_original_mention_count_in_corpus": len(residue_corpus),
        "residual_original_mention_count_in_queries": len(residue_queries),
        "alias_table_size": len(alias_table),
        "answer_preservation_rate": (answer_preserved / answer_total) if answer_total else None,
        "answer_preservation_query_count": answer_total,
        "answer_preservation_missing_query_ids": answer_missing_query_ids[:50],
        "support_preservation_rate": (support_preserved / support_total) if support_total else None,
        "support_preservation_query_count": support_total,
        "support_preservation_missing_query_ids": support_missing_query_ids[:50],
        "query_count": len(renamed.queries),
        "document_count": len(renamed.corpus),
    }


def infer_entity_type(text: str, context: str = "") -> str:
    lowered = text.lower()
    if _looks_like_value(text):
        if re.fullmatch(r"\d+(?:\.\d+)?", text.strip()):
            return "NUMBER"
        return "DATE"
    if any(hint in lowered for hint in ORG_HINTS):
        return "ORGANIZATION"
    if any(hint in lowered for hint in LOC_HINTS):
        return "LOCATION"
    if text.startswith('"') and text.endswith('"'):
        return "TITLE"
    if len(text.split()) >= 2 and all(part[:1].isupper() for part in text.split() if part):
        return "PERSON"
    if context and re.search(r"\b(where|born|located|city|country)\b", context.lower()):
        return "LOCATION"
    return "ENTITY"


def generate_alias(entity_type: str, index: int, alias_style: str, rng: random.Random) -> str:
    if alias_style == "coded":
        prefix = {
            "PERSON": "Person",
            "ORGANIZATION": "Organization",
            "LOCATION": "Site",
            "TITLE": "Work",
            "DATE": "Date",
            "NUMBER": "Value",
            "ENTITY": "Entity",
        }.get(entity_type, entity_type.title())
        return f"{prefix} {entity_type[:1]}-{index:03d}"
    if entity_type == "PERSON":
        return f"{FIRST_PARTS[(index - 1) % len(FIRST_PARTS)]} {LAST_PARTS[(index - 1) % len(LAST_PARTS)]}"
    if entity_type == "LOCATION":
        return f"Lake {LOC_PARTS[(index - 1) % len(LOC_PARTS)]}" if index % 2 == 1 else f"Region {LOC_PARTS[(index - 1) % len(LOC_PARTS)]}"
    if entity_type == "ORGANIZATION":
        return f"{LAST_PARTS[(index - 1) % len(LAST_PARTS)]} {ORG_PARTS[(index - 1) % len(ORG_PARTS)]}"
    if entity_type == "TITLE":
        return f"Project {LOC_PARTS[(index - 1) % len(LOC_PARTS)]}-{index}"
    if entity_type == "DATE":
        return f"Year {2000 + index}"
    if entity_type == "NUMBER":
        return f"{100 + index}"
    stem = LOC_PARTS[(index - 1) % len(LOC_PARTS)]
    return f"Entity {stem}-{index}"


def _rename_document(doc: Document, alias_table: Mapping[str, Mapping[str, str]]) -> Document:
    return Document(
        doc_id=doc.doc_id,
        title=apply_alias_table(doc.title, alias_table),
        text=apply_alias_table(doc.text, alias_table),
        metadata=doc.metadata,
        original_text=doc.text,
    )


def _rename_query(query: Query, alias_table: Mapping[str, Mapping[str, str]]) -> Query:
    return Query(
        query_id=query.query_id,
        text=apply_alias_table(query.text, alias_table),
        answers=_rename_values(query.answers, alias_table),
        answer_aliases=_rename_values(query.answer_aliases, alias_table),
        supporting_doc_ids=query.supporting_doc_ids,
        supporting_facts=_rename_values(query.supporting_facts, alias_table),
        metadata=query.metadata,
        original_text=query.text,
    )


def _rename_values(values: Iterable[str], alias_table: Mapping[str, Mapping[str, str]]) -> tuple[str, ...]:
    renamed = []
    for value in values:
        renamed.append(apply_alias_table(value, alias_table))
    return tuple(renamed)


def apply_alias_table(text: str, alias_table: Mapping[str, Mapping[str, str]]) -> str:
    updated = text
    for source, data in sorted(alias_table.items(), key=lambda item: (-len(item[0]), item[0].lower())):
        alias = data["alias"]
        flags = re.IGNORECASE if len(source) >= 4 else 0
        pattern = re.compile(rf"\b{re.escape(source)}\b", flags)
        updated = pattern.sub(alias, updated)
    return updated


def _collect_from_text(text: str, spans: dict[str, str], include_values: bool) -> None:
    nlp = get_nlp()
    doc = nlp(text)
    
    value_labels = {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}
    entity_labels = {"PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART"}
    
    for ent in doc.ents:
        if len(ent.text) < 3 and ent.label_ not in value_labels:
            continue
            
        label = ent.label_
        
        if not include_values and label in value_labels:
            continue
            
        if label not in entity_labels and label not in value_labels:
            continue
            
        mapped_type = "ENTITY"
        if label == "PERSON":
            mapped_type = "PERSON"
        elif label == "ORG":
            mapped_type = "ORGANIZATION"
        elif label in ("GPE", "LOC", "FAC"):
            mapped_type = "LOCATION"
        elif label == "WORK_OF_ART":
            mapped_type = "TITLE"
        elif label in ("DATE", "TIME"):
            mapped_type = "DATE"
        elif label in ("PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"):
            mapped_type = "NUMBER"
            
        spans.setdefault(ent.text, mapped_type)
        
    for match in QUOTED_TITLE_RE.findall(text):
        spans.setdefault(match, "TITLE")


def _looks_like_value(text: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", text.strip()) or re.search(r"\b\d{4}\b", text))


def _replacement_coverage_estimate(
    original_text: str,
    renamed_text: str,
    alias_table: Mapping[str, Mapping[str, str]],
) -> float:
    if not alias_table:
        return 1.0
    replaced = 0
    for source, data in alias_table.items():
        if normalize_text(source) not in normalize_text(original_text):
            continue
        if normalize_text(data["alias"]) in normalize_text(renamed_text):
            replaced += 1
    return replaced / max(len(alias_table), 1)


def _residual_mentions(alias_table: Mapping[str, Mapping[str, str]], text: str) -> list[str]:
    normalized_text = normalize_text(text)
    residue = []
    for source in alias_table:
        normalized_source = normalize_text(source)
        if normalized_source and normalized_source in normalized_text:
            residue.append(source)
    return residue
