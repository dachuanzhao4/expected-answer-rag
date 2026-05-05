from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    title: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)
    original_text: str | None = None


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    answers: tuple[str, ...] = ()
    answer_aliases: tuple[str, ...] = ()
    supporting_doc_ids: tuple[str, ...] = ()
    supporting_facts: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    original_text: str | None = None

    @property
    def all_answer_strings(self) -> tuple[str, ...]:
        values = []
        seen = set()
        for value in [*self.answers, *self.answer_aliases]:
            if not value:
                continue
            key = value.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            values.append(value)
        return tuple(values)


@dataclass(frozen=True)
class RetrievalDataset:
    name: str
    corpus: List[Document]
    queries: List[Query]
    qrels: Dict[str, Dict[str, int]]
    metadata: Mapping[str, object] = field(default_factory=dict)


def load_dataset(
    name: str,
    max_corpus: Optional[int] = None,
    max_queries: Optional[int] = None,
    cache_dir: Optional[str] = None,
    query_metadata_path: Optional[str] = None,
) -> RetrievalDataset:
    if name == "toy":
        dataset = load_toy_dataset(max_queries=max_queries)
    elif _looks_like_local_dataset(name):
        dataset = load_local_dataset(name, max_corpus=max_corpus, max_queries=max_queries)
    else:
        dataset = load_beir_dataset(
            name=name,
            max_corpus=max_corpus,
            max_queries=max_queries,
            cache_dir=cache_dir,
        )
    if query_metadata_path:
        dataset = merge_query_metadata(dataset, query_metadata_path)
    return dataset


def load_toy_dataset(max_queries: Optional[int] = None) -> RetrievalDataset:
    corpus = [
        Document(
            "d1",
            "Marie Curie was born in Warsaw and later worked in Paris.",
            "Marie Curie",
        ),
        Document(
            "d2",
            "Albert Einstein was born in Ulm in the Kingdom of Wurttemberg.",
            "Albert Einstein",
        ),
        Document(
            "d3",
            "Ada Lovelace was born in London and wrote notes on the Analytical Engine.",
            "Ada Lovelace",
        ),
        Document("d4", "Boston is the capital and largest city of Massachusetts.", "Boston"),
        Document("d5", "The Eiffel Tower is a wrought-iron lattice tower in Paris.", "Eiffel Tower"),
    ]
    queries = [
        Query("q1", "Where was Marie Curie born?", ("Warsaw",), ("Warsaw, Poland",), ("d1",)),
        Query("q2", "Where was Albert Einstein born?", ("Ulm",), ("Ulm, Germany",), ("d2",)),
        Query("q3", "Where was Ada Lovelace born?", ("London",), ("London, England",), ("d3",)),
    ]
    if max_queries is not None:
        queries = queries[:max_queries]
    qrels = {
        "q1": {"d1": 1},
        "q2": {"d2": 1},
        "q3": {"d3": 1},
    }
    return RetrievalDataset(
        "toy",
        corpus,
        queries,
        qrels,
        metadata={"dataset_type": "toy", "answer_metadata": True},
    )


def load_beir_dataset(
    name: str,
    max_corpus: Optional[int] = None,
    max_queries: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> RetrievalDataset:
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required for BEIR loading. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc

    qrel_rows = _load_first_available_split(
        hf_load_dataset,
        f"BeIR/{name}-qrels",
        config=None,
        splits=["test", "validation", "dev", "train"],
        cache_dir=cache_dir,
    )
    qrel_rows_list = list(qrel_rows)
    corpus_rows = _load_first_available_split(
        hf_load_dataset,
        f"BeIR/{name}",
        config="corpus",
        splits=["corpus"],
        cache_dir=cache_dir,
    )
    query_rows = _load_first_available_split(
        hf_load_dataset,
        f"BeIR/{name}",
        config="queries",
        splits=["queries", "test", "validation", "dev"],
        cache_dir=cache_dir,
    )

    corpus = [
        Document(
            doc_id=str(row.get("_id") or row.get("id") or row.get("corpus-id")),
            title=str(row.get("title") or ""),
            text=_join_title_text(row.get("title"), row.get("text")),
            metadata={
                key: value
                for key, value in row.items()
                if key not in {"_id", "id", "corpus-id", "title", "text"}
            },
        )
        for row in _take(corpus_rows, max_corpus)
    ]
    corpus_ids = {doc.doc_id for doc in corpus}
    all_qrels_by_query = _parse_qrels(qrel_rows_list, query_ids=set(), corpus_ids=set())
    filtered_qrel_rows = [
        row
        for row in qrel_rows_list
        if not corpus_ids
        or str(row.get("corpus-id") or row.get("corpus_id") or row.get("doc_id") or row.get("pid") or row.get("docId")) in corpus_ids
    ]
    qrel_query_ids = {
        str(row.get("query-id") or row.get("query_id") or row.get("qid") or row.get("queryId"))
        for row in filtered_qrel_rows
    }
    queries = []
    for row in query_rows:
        query_id = str(row.get("_id") or row.get("id") or row.get("query-id"))
        if qrel_query_ids and query_id not in qrel_query_ids:
            continue
        queries.append(
            Query(
                query_id=query_id,
                text=str(row.get("text") or row.get("query") or row.get("question") or ""),
                answers=_tupleify(row.get("answers") or row.get("short_answers")),
                answer_aliases=_tupleify(row.get("answer_aliases") or row.get("aliases")),
                metadata={
                    key: value
                    for key, value in row.items()
                    if key not in {"_id", "id", "query-id", "text", "query", "question", "answers", "short_answers", "answer_aliases", "aliases"}
                },
            )
        )
        if max_queries is not None and len(queries) >= max_queries:
            break
    query_ids = {query.query_id for query in queries}
    qrels = _parse_qrels(filtered_qrel_rows, query_ids=query_ids, corpus_ids=corpus_ids)
    qrel_coverage = _qrel_coverage_summary(query_ids, qrels, all_qrels_by_query)

    return RetrievalDataset(
        name=name,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        metadata={
            "dataset_type": "beir",
            "answer_metadata": any(query.answers for query in queries),
            "qrel_coverage": qrel_coverage,
        },
    )


def load_local_dataset(
    path: str,
    max_corpus: Optional[int] = None,
    max_queries: Optional[int] = None,
) -> RetrievalDataset:
    root = _resolve_local_dataset_root(path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    corpus = [
        _document_from_row(row)
        for row in _take(_read_jsonl(root / "corpus.jsonl"), max_corpus)
    ]
    queries = [
        _query_from_row(row)
        for row in _take(_read_jsonl(root / "queries.jsonl"), max_queries)
    ]
    query_ids = {query.query_id for query in queries}
    corpus_ids = {doc.doc_id for doc in corpus}
    qrels = _parse_qrels(_read_jsonl(root / "qrels.jsonl"), query_ids=query_ids, corpus_ids=corpus_ids)
    name = str(manifest.get("name") or root.name)
    return RetrievalDataset(name=name, corpus=corpus, queries=queries, qrels=qrels, metadata=manifest)


def export_local_dataset(dataset: RetrievalDataset, output_dir: str | Path) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "corpus.jsonl").write_text(
        "\n".join(json.dumps(_document_to_row(doc), ensure_ascii=False) for doc in dataset.corpus) + "\n",
        encoding="utf-8",
    )
    (root / "queries.jsonl").write_text(
        "\n".join(json.dumps(_query_to_row(query), ensure_ascii=False) for query in dataset.queries) + "\n",
        encoding="utf-8",
    )
    qrel_lines = []
    for qid, rels in dataset.qrels.items():
        for doc_id, score in rels.items():
            qrel_lines.append(json.dumps({"query_id": qid, "doc_id": doc_id, "score": score}, ensure_ascii=False))
    (root / "qrels.jsonl").write_text("\n".join(qrel_lines) + ("\n" if qrel_lines else ""), encoding="utf-8")
    manifest = dict(dataset.metadata)
    manifest.setdefault("name", dataset.name)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return root


def merge_query_metadata(dataset: RetrievalDataset, metadata_path: str | Path) -> RetrievalDataset:
    rows = {str(row["query_id"]): row for row in _read_jsonl(Path(metadata_path))}
    merged_queries = []
    for query in dataset.queries:
        row = rows.get(query.query_id)
        if not row:
            merged_queries.append(query)
            continue
        merged_queries.append(
            Query(
                query_id=query.query_id,
                text=row.get("text", query.text),
                answers=_merge_tuple_values(query.answers, row.get("answers")),
                answer_aliases=_merge_tuple_values(query.answer_aliases, row.get("answer_aliases")),
                supporting_doc_ids=_merge_tuple_values(query.supporting_doc_ids, row.get("supporting_doc_ids")),
                supporting_facts=_merge_tuple_values(query.supporting_facts, row.get("supporting_facts")),
                metadata={**dict(query.metadata), **dict(row.get("metadata") or {})},
                original_text=row.get("original_text", query.original_text),
            )
        )
    metadata = dict(dataset.metadata)
    metadata["query_metadata_path"] = str(metadata_path)
    metadata["answer_metadata"] = any(query.answers for query in merged_queries)
    return RetrievalDataset(
        name=dataset.name,
        corpus=dataset.corpus,
        queries=merged_queries,
        qrels=dataset.qrels,
        metadata=metadata,
    )


def _looks_like_local_dataset(name: str) -> bool:
    if name.startswith("local:"):
        return True
    return Path(name).exists()


def _resolve_local_dataset_root(path: str) -> Path:
    if path.startswith("local:"):
        return Path(path.split(":", 1)[1]).expanduser().resolve()
    return Path(path).expanduser().resolve()


def _load_first_available_split(
    hf_load_dataset,
    dataset_name: str,
    config: Optional[str],
    splits: List[str],
    cache_dir: Optional[str],
):
    errors = []
    for split in splits:
        try:
            if config is None:
                return hf_load_dataset(dataset_name, split=split, cache_dir=cache_dir)
            return hf_load_dataset(dataset_name, config, split=split, cache_dir=cache_dir)
        except Exception as exc:  # noqa: BLE001 - keep dataset fallback robust.
            errors.append(f"{split}: {exc}")
    joined = "\n".join(errors)
    raise RuntimeError(f"Could not load any split from {dataset_name}. Tried:\n{joined}")


def _take(rows: Iterable[Mapping], limit: Optional[int]) -> Iterable[Mapping]:
    if limit is None:
        yield from rows
        return
    for idx, row in enumerate(rows):
        if idx >= limit:
            break
        yield row


def _join_title_text(title: object, text: object) -> str:
    title_text = str(title or "").strip()
    body_text = str(text or "").strip()
    if title_text and body_text:
        return f"{title_text}\n{body_text}"
    return title_text or body_text


def _parse_qrels(
    rows: Iterable[Mapping],
    query_ids: set[str],
    corpus_ids: set[str],
) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    for row in rows:
        qid = str(row.get("query-id") or row.get("query_id") or row.get("qid") or row.get("queryId"))
        did = str(row.get("corpus-id") or row.get("corpus_id") or row.get("doc_id") or row.get("pid") or row.get("docId"))
        score = int(row.get("score") or row.get("relevance") or 1)
        if query_ids and qid not in query_ids:
            continue
        if corpus_ids and did not in corpus_ids:
            continue
        if score <= 0:
            continue
        qrels.setdefault(qid, {})[did] = score
    return qrels


def _qrel_coverage_summary(
    query_ids: set[str],
    included_qrels: Mapping[str, Mapping[str, int]],
    full_qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, object]:
    rows = []
    for qid in sorted(query_ids):
        total = sum(1 for score in full_qrels.get(qid, {}).values() if score > 0)
        included = sum(1 for score in included_qrels.get(qid, {}).values() if score > 0)
        coverage = (included / total) if total else 0.0
        rows.append({"query_id": qid, "included": included, "total": total, "coverage": coverage})
    if not rows:
        return {
            "num_queries": 0,
            "queries_with_relevant": 0,
            "queries_with_included_relevant": 0,
            "mean_coverage": 0.0,
            "min_coverage": 0.0,
            "zero_coverage_queries": [],
        }
    coverages = [row["coverage"] for row in rows if row["total"] > 0]
    zero_coverage = [row["query_id"] for row in rows if row["total"] > 0 and row["included"] == 0]
    return {
        "num_queries": len(rows),
        "queries_with_relevant": sum(1 for row in rows if row["total"] > 0),
        "queries_with_included_relevant": sum(1 for row in rows if row["included"] > 0),
        "mean_coverage": (sum(coverages) / len(coverages)) if coverages else 0.0,
        "min_coverage": min(coverages) if coverages else 0.0,
        "zero_coverage_queries": zero_coverage,
    }


def _tupleify(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, (list, tuple, set)):
        values = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                values.append(text)
        return tuple(values)
    text = str(value).strip()
    return (text,) if text else ()


def _merge_tuple_values(existing: tuple[str, ...], incoming: object) -> tuple[str, ...]:
    values = []
    seen = set()
    for source in [existing, _tupleify(incoming)]:
        for item in source:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            values.append(item)
    return tuple(values)


def _read_jsonl(path: Path) -> Iterable[Mapping[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _document_from_row(row: Mapping[str, object]) -> Document:
    return Document(
        doc_id=str(row.get("doc_id") or row.get("_id") or row.get("id")),
        title=str(row.get("title") or ""),
        text=str(row.get("text") or ""),
        metadata=dict(row.get("metadata") or {}),
        original_text=str(row.get("original_text")) if row.get("original_text") is not None else None,
    )


def _query_from_row(row: Mapping[str, object]) -> Query:
    return Query(
        query_id=str(row.get("query_id") or row.get("_id") or row.get("id")),
        text=str(row.get("text") or row.get("query") or row.get("question") or ""),
        answers=_tupleify(row.get("answers")),
        answer_aliases=_tupleify(row.get("answer_aliases")),
        supporting_doc_ids=_tupleify(row.get("supporting_doc_ids")),
        supporting_facts=_tupleify(row.get("supporting_facts")),
        metadata=dict(row.get("metadata") or {}),
        original_text=str(row.get("original_text")) if row.get("original_text") is not None else None,
    )


def _document_to_row(doc: Document) -> dict[str, object]:
    row = asdict(doc)
    row["metadata"] = dict(doc.metadata)
    return row


def _query_to_row(query: Query) -> dict[str, object]:
    row = asdict(query)
    row["metadata"] = dict(query.metadata)
    return row
