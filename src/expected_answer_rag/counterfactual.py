from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

from expected_answer_rag.datasets import Document, Query, RetrievalDataset, load_local_dataset
from expected_answer_rag.leakage import QUOTED_TITLE_RE, normalize_text
import spacy

_nlp = None
ProgressCallback = Callable[[str], None]
COUNTERFACTUAL_ARTIFACT_VERSION = "v1"


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer", "attribute_ruler"])
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
    progress: ProgressCallback | None = None,
) -> CounterfactualBuildResult:
    rng = random.Random(seed)
    _progress(progress, "  - Collecting renamable spans")
    spans = collect_renamable_spans(dataset, include_values=include_values)
    _progress(progress, f"  - Collected {len(spans)} spans; building alias table")
    alias_table = build_alias_table(spans, alias_style=alias_style, rng=rng)
    compiled_aliases = compile_alias_table(alias_table)
    _progress(progress, f"  - Renaming corpus ({len(dataset.corpus)} documents)")
    renamed_corpus = [_rename_document(doc, compiled_aliases) for doc in dataset.corpus]
    _progress(progress, f"  - Renaming queries ({len(dataset.queries)} queries)")
    renamed_queries = [_rename_query(query, compiled_aliases) for query in dataset.queries]
    variant_name = f"{dataset.name}__entity_counterfactual_{alias_style}"
    if include_values:
        variant_name += "_with_values"
    metadata = {
        **dict(dataset.metadata),
        "variant": "entity_counterfactual",
        "counterfactual_regime": "entity_and_value" if include_values else "entity",
        "alias_style": alias_style,
        "include_values": include_values,
        "seed": seed,
        "alias_table_size": len(alias_table),
        "source_dataset_name": dataset.name,
        "source_document_count": len(dataset.corpus),
        "source_query_count": len(dataset.queries),
        "counterfactual_artifact_version": COUNTERFACTUAL_ARTIFACT_VERSION,
        "source_dataset_fingerprint": dataset_fingerprint(dataset),
    }
    renamed_dataset = RetrievalDataset(
        name=variant_name,
        corpus=renamed_corpus,
        queries=renamed_queries,
        qrels=dataset.qrels,
        metadata=metadata,
    )
    _progress(progress, "  - Validating counterfactual dataset")
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


def load_counterfactual_artifacts(
    output_dir: str | Path,
    max_corpus: int | None = None,
    max_queries: int | None = None,
) -> CounterfactualBuildResult:
    root = Path(output_dir)
    dataset = load_local_dataset(str(root), max_corpus=max_corpus, max_queries=max_queries)
    alias_path = root / "alias_table.json"
    validation_path = root / "validation.json"
    alias_table = json.loads(alias_path.read_text(encoding="utf-8")) if alias_path.exists() else {}
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
    return CounterfactualBuildResult(dataset=dataset, alias_table=alias_table, validation=validation)


def resolve_counterfactual_artifact_dir(
    artifact_root: str | Path,
    dataset: RetrievalDataset,
    alias_style: str = "natural",
    include_values: bool = False,
    seed: int = 13,
) -> Path:
    root = Path(artifact_root)
    regime = "entity_and_value" if include_values else "entity"
    corpus_label = f"c{len(dataset.corpus)}"
    query_label = f"q{len(dataset.queries)}"
    fingerprint = dataset_fingerprint(dataset)[:12]
    slug = _slugify(dataset.name)
    directory = (
        f"{slug}__{regime}__{alias_style}__seed{seed}"
        f"__{corpus_label}__{query_label}__{COUNTERFACTUAL_ARTIFACT_VERSION}__{fingerprint}"
    )
    return root / directory


def dataset_fingerprint(dataset: RetrievalDataset) -> str:
    digest = hashlib.sha256()
    digest.update(f"name={dataset.name}\n".encode("utf-8"))
    digest.update(f"metadata={json.dumps(dict(dataset.metadata), sort_keys=True, ensure_ascii=False)}\n".encode("utf-8"))
    for document in dataset.corpus:
        digest.update(
            json.dumps(
                {
                    "doc_id": document.doc_id,
                    "title": document.title,
                    "text": document.text,
                    "metadata": dict(document.metadata),
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    for query in dataset.queries:
        digest.update(
            json.dumps(
                {
                    "query_id": query.query_id,
                    "text": query.text,
                    "answers": list(query.answers),
                    "answer_aliases": list(query.answer_aliases),
                    "supporting_doc_ids": list(query.supporting_doc_ids),
                    "supporting_facts": list(query.supporting_facts),
                    "metadata": dict(query.metadata),
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    digest.update(json.dumps(dataset.qrels, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def collect_renamable_spans(dataset: RetrievalDataset, include_values: bool = False) -> dict[str, str]:
    spans: dict[str, str] = {}
    nlp = get_nlp()
    query_texts = [query.text for query in dataset.queries]
    for text in nlp.pipe(query_texts, batch_size=64):
        _collect_from_doc(text, spans, include_values)
    for query in dataset.queries:
        for answer in query.all_answer_strings:
            if include_values or not _looks_like_value(answer):
                spans.setdefault(answer, infer_entity_type(answer, context=query.text))
    doc_texts: list[str] = []
    for document in dataset.corpus:
        if document.title:
            doc_texts.append(document.title)
        if document.text:
            doc_texts.append(document.text)
    for text in nlp.pipe(doc_texts, batch_size=32):
        _collect_from_doc(text, spans, include_values)
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


def compile_alias_table(alias_table: Mapping[str, Mapping[str, str]]) -> list[tuple[re.Pattern[str], str]]:
    compiled: list[tuple[re.Pattern[str], str]] = []
    for source, data in sorted(alias_table.items(), key=lambda item: (-len(item[0]), item[0].lower())):
        flags = re.IGNORECASE if len(source) >= 4 else 0
        pattern = re.compile(rf"\b{re.escape(source)}\b", flags)
        compiled.append((pattern, data["alias"]))
    return compiled


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


def apply_alias_table(
    text: str,
    alias_table: Mapping[str, Mapping[str, str]] | list[tuple[re.Pattern[str], str]],
) -> str:
    updated = text
    if isinstance(alias_table, list):
        compiled_aliases = alias_table
    else:
        compiled_aliases = compile_alias_table(alias_table)
    for pattern, alias in compiled_aliases:
        updated = pattern.sub(alias, updated)
    return updated


def _collect_from_doc(doc, spans: dict[str, str], include_values: bool) -> None:
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

    for match in QUOTED_TITLE_RE.findall(doc.text):
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


def _progress(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "dataset"
