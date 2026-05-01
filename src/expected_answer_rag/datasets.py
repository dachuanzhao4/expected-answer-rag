from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    title: str = ""


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    answers: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalDataset:
    name: str
    corpus: List[Document]
    queries: List[Query]
    qrels: Dict[str, Dict[str, int]]


def load_dataset(
    name: str,
    max_corpus: Optional[int] = None,
    max_queries: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> RetrievalDataset:
    if name == "toy":
        return load_toy_dataset(max_queries=max_queries)
    return load_beir_dataset(
        name=name,
        max_corpus=max_corpus,
        max_queries=max_queries,
        cache_dir=cache_dir,
    )


def load_toy_dataset(max_queries: Optional[int] = None) -> RetrievalDataset:
    corpus = [
        Document("d1", "Marie Curie was born in Warsaw and later worked in Paris.", "Marie Curie"),
        Document("d2", "Albert Einstein was born in Ulm in the Kingdom of Wurttemberg.", "Albert Einstein"),
        Document("d3", "Ada Lovelace was born in London and wrote notes on the Analytical Engine.", "Ada Lovelace"),
        Document("d4", "Boston is the capital and largest city of Massachusetts.", "Boston"),
        Document("d5", "The Eiffel Tower is a wrought-iron lattice tower in Paris.", "Eiffel Tower"),
    ]
    queries = [
        Query("q1", "Where was Marie Curie born?", ("Warsaw",)),
        Query("q2", "Where was Albert Einstein born?", ("Ulm",)),
        Query("q3", "Where was Ada Lovelace born?", ("London",)),
    ]
    if max_queries is not None:
        queries = queries[:max_queries]
    qrels = {
        "q1": {"d1": 1},
        "q2": {"d2": 1},
        "q3": {"d3": 1},
    }
    return RetrievalDataset("toy", corpus, queries, qrels)


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
    qrel_rows = _load_first_available_split(
        hf_load_dataset,
        f"BeIR/{name}-qrels",
        config=None,
        splits=["test", "validation", "dev", "train"],
        cache_dir=cache_dir,
    )

    corpus = [
        Document(
            doc_id=str(row.get("_id") or row.get("id") or row.get("corpus-id")),
            title=str(row.get("title") or ""),
            text=_join_title_text(row.get("title"), row.get("text")),
        )
        for row in _take(corpus_rows, max_corpus)
    ]
    queries = [
        Query(
            query_id=str(row.get("_id") or row.get("id") or row.get("query-id")),
            text=str(row.get("text") or row.get("query") or row.get("question") or ""),
        )
        for row in _take(query_rows, max_queries)
    ]
    query_ids = {query.query_id for query in queries}
    corpus_ids = {doc.doc_id for doc in corpus}
    qrels = _parse_qrels(qrel_rows, query_ids=query_ids, corpus_ids=corpus_ids)

    return RetrievalDataset(name=name, corpus=corpus, queries=queries, qrels=qrels)


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
        qid = str(row.get("query-id") or row.get("query_id") or row.get("qid"))
        did = str(row.get("corpus-id") or row.get("corpus_id") or row.get("doc_id") or row.get("pid"))
        score = int(row.get("score") or row.get("relevance") or 1)
        if query_ids and qid not in query_ids:
            continue
        if corpus_ids and did not in corpus_ids:
            continue
        if score <= 0:
            continue
        qrels.setdefault(qid, {})[did] = score
    return qrels
