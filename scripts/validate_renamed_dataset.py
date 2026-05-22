from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is a convenience only.
    tqdm = None


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
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
    "how",
    "in",
    "is",
    "it",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a local renamed retrieval dataset.")
    parser.add_argument("dataset_dir")
    parser.add_argument("--max-leftover-examples", type=int, default=20)
    parser.add_argument("--max-missing-anchor-examples", type=int, default=20)
    parser.add_argument("--min-query-doc-overlap", type=int, default=1)
    parser.add_argument("--leftover-chunk-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    root = Path(args.dataset_dir)
    corpus = {str(row["_id"]): row for row in read_jsonl(root / "corpus.jsonl")}
    queries = {str(row["_id"]): row for row in read_jsonl(root / "queries.jsonl")}
    qrels = list(read_jsonl(root / "qrels.jsonl"))
    mapping = json.loads((root / "mapping.json").read_text(encoding="utf-8")) if (root / "mapping.json").exists() else {}
    stats = json.loads((root / "stats.json").read_text(encoding="utf-8")) if (root / "stats.json").exists() else {}

    issues: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []

    if not corpus:
        issues.append({"type": "empty_corpus"})
    if not queries:
        issues.append({"type": "empty_queries"})
    if not qrels:
        issues.append({"type": "empty_qrels"})

    for qid, query in queries.items():
        if not str(query.get("text", "")).strip():
            issues.append({"type": "empty_query_text", "query_id": qid})

    for doc_id, doc in corpus.items():
        if not str(doc.get("text", "") or doc.get("title", "")).strip():
            issues.append({"type": "empty_doc_text", "doc_id": doc_id})

    qrels_by_query: dict[str, list[str]] = {}
    for row in qrels:
        qid = str(row.get("query-id") or row.get("query_id") or row.get("qid"))
        did = str(row.get("corpus-id") or row.get("corpus_id") or row.get("doc_id") or row.get("pid"))
        if qid not in queries:
            issues.append({"type": "qrel_missing_query", "query_id": qid, "doc_id": did})
        if did not in corpus:
            issues.append({"type": "qrel_missing_doc", "query_id": qid, "doc_id": did})
        qrels_by_query.setdefault(qid, []).append(did)

    for qid in queries:
        if qid not in qrels_by_query:
            issues.append({"type": "query_without_qrel", "query_id": qid})

    replacements = [str(row["replacement"]) for row in mapping.values()]
    replacement_counts = Counter(replacements)
    duplicate_replacements = sorted(item for item, count in replacement_counts.items() if count > 1)
    if duplicate_replacements:
        issues.append({"type": "duplicate_replacements", "values": duplicate_replacements[:20]})

    mapping_summary, mapping_issues = validate_mapping_integrity(mapping, stats)
    issues.extend(mapping_issues)

    if stats.get("replacement_granularity") != "token":
        missing_query_anchor_count, missing_query_anchor_examples = find_query_replacements_missing_from_relevant_docs(
            queries=queries,
            corpus=corpus,
            qrels_by_query=qrels_by_query,
            replacements=replacements,
            limit=args.max_missing_anchor_examples,
        )
        if missing_query_anchor_count:
            issues.append(
                {
                    "type": "query_replacement_missing_in_relevant_doc",
                    "count": missing_query_anchor_count,
                    "examples": missing_query_anchor_examples,
                }
            )

    leftover_examples = find_leftover_sources(
        corpus,
        queries,
        mapping,
        args.max_leftover_examples,
        args.leftover_chunk_size,
    )
    if leftover_examples:
        warnings.append({"type": "source_span_leftovers", "examples": leftover_examples})

    low_overlap = []
    for qid, query in queries.items():
        query_tokens = content_tokens(str(query.get("text", "")))
        best_overlap = 0
        for did in qrels_by_query.get(qid, []):
            doc = corpus.get(did)
            if not doc:
                continue
            doc_tokens = content_tokens(str(doc.get("title", "")) + " " + str(doc.get("text", "")))
            best_overlap = max(best_overlap, len(query_tokens & doc_tokens))
        if best_overlap < args.min_query_doc_overlap:
            low_overlap.append({"query_id": qid, "best_overlap": best_overlap, "query": query.get("text", "")})
    if low_overlap:
        warnings.append({"type": "low_query_relevant_doc_overlap", "count": len(low_overlap), "examples": low_overlap[:20]})

    summary = {
        "dataset_dir": str(root),
        "num_corpus": len(corpus),
        "num_queries": len(queries),
        "num_qrels": len(qrels),
        "mapping_size": len(mapping),
        "mapping_integrity": mapping_summary,
        "stats": stats,
        "num_issues": len(issues),
        "num_warnings": len(warnings),
        "issues": issues[:50],
        "warnings": warnings,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if issues:
        sys.exit(1)


def read_jsonl(path: Path) -> Iterable[Mapping]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def content_tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS}


def validate_mapping_integrity(
    mapping: dict[str, Mapping],
    stats: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    issues: list[dict[str, object]] = []
    source_norms = []
    replacement_norms = []
    source_token_counts = Counter()
    replacement_token_counts = Counter()
    source_equals_replacement = []
    bad_token_rows = []
    empty_rows = []

    for normalized, row in mapping.items():
        source = str(row.get("source", "")).strip()
        replacement = str(row.get("replacement", "")).strip()
        if not source or not replacement:
            empty_rows.append({"key": normalized, "source": source, "replacement": replacement})
            continue
        source_norm = retrieval_norm(source)
        replacement_norm = retrieval_norm(replacement)
        source_norms.append(source_norm)
        replacement_norms.append(replacement_norm)
        source_tokens = TOKEN_RE.findall(source)
        replacement_tokens = TOKEN_RE.findall(replacement)
        source_token_counts[len(source_tokens)] += 1
        replacement_token_counts[len(replacement_tokens)] += 1
        if source_norm == replacement_norm:
            source_equals_replacement.append({"source": source, "replacement": replacement})
        if stats.get("replacement_granularity") == "token" and (
            len(source_tokens) != 1 or len(replacement_tokens) != 1
        ):
            bad_token_rows.append(
                {
                    "source": source,
                    "replacement": replacement,
                    "source_token_count": len(source_tokens),
                    "replacement_token_count": len(replacement_tokens),
                }
            )

    duplicate_source_norms = sorted(value for value, count in Counter(source_norms).items() if count > 1)
    duplicate_replacement_norms = sorted(value for value, count in Counter(replacement_norms).items() if count > 1)
    source_replacement_collisions = sorted(set(source_norms) & set(replacement_norms))

    if empty_rows:
        issues.append({"type": "empty_mapping_rows", "examples": empty_rows[:20]})
    if duplicate_source_norms:
        issues.append({"type": "duplicate_normalized_sources", "values": duplicate_source_norms[:20]})
    if duplicate_replacement_norms:
        issues.append({"type": "duplicate_normalized_replacements", "values": duplicate_replacement_norms[:20]})
    if source_replacement_collisions:
        issues.append({"type": "source_replacement_collisions", "values": source_replacement_collisions[:20]})
    if source_equals_replacement:
        issues.append({"type": "source_equals_replacement", "examples": source_equals_replacement[:20]})
    if bad_token_rows:
        issues.append({"type": "token_granularity_count_mismatch", "examples": bad_token_rows[:20]})

    return (
        {
            "source_token_counts": dict(sorted(source_token_counts.items())),
            "replacement_token_counts": dict(sorted(replacement_token_counts.items())),
            "duplicate_normalized_sources": len(duplicate_source_norms),
            "duplicate_normalized_replacements": len(duplicate_replacement_norms),
            "source_replacement_collisions": len(source_replacement_collisions),
            "source_equals_replacement": len(source_equals_replacement),
            "token_granularity_count_mismatch": len(bad_token_rows),
        },
        issues,
    )


def retrieval_norm(text: str) -> str:
    return " ".join(token.lower() for token in TOKEN_RE.findall(text))


def find_query_replacements_missing_from_relevant_docs(
    queries: dict[str, Mapping],
    corpus: dict[str, Mapping],
    qrels_by_query: dict[str, list[str]],
    replacements: list[str],
    limit: int,
) -> tuple[int, list[dict[str, object]]]:
    replacement_norms = sorted(
        {(replacement, normalize(replacement)) for replacement in replacements if normalize(replacement)},
        key=lambda item: len(item[1]),
        reverse=True,
    )
    examples = []
    total_missing_queries = 0
    for qid, query in queries.items():
        query_norm = padded_norm(str(query.get("text", "")))
        query_replacements = [
            replacement
            for replacement, replacement_norm in replacement_norms
            if f" {replacement_norm} " in query_norm
        ]
        if not query_replacements:
            continue
        relevant_text = "\n".join(
            f"{corpus[did].get('title', '')}\n{corpus[did].get('text', '')}"
            for did in qrels_by_query.get(qid, [])
            if did in corpus
        )
        relevant_norm = padded_norm(relevant_text)
        missing = [
            replacement
            for replacement in query_replacements
            if f" {normalize(replacement)} " not in relevant_norm
        ]
        if not missing:
            continue
        total_missing_queries += 1
        if len(examples) < limit:
            examples.append(
                {
                    "query_id": qid,
                    "missing_replacements": missing[:10],
                    "query": query.get("text", ""),
                }
            )
    return total_missing_queries, examples


def find_leftover_sources(
    corpus: dict[str, Mapping],
    queries: dict[str, Mapping],
    mapping: dict[str, Mapping],
    limit: int,
    chunk_size: int = 256,
) -> list[dict[str, str]]:
    examples = []
    sources = sorted(
        {str(row.get("source", "")).strip() for row in mapping.values() if str(row.get("source", "")).strip()},
        key=len,
        reverse=True,
    )
    replacement_norms = [normalize(str(row.get("replacement", ""))) for row in mapping.values()]
    sources = [
        source
        for source in sources
        if is_meaningful_leftover_source(normalize(source))
        and not source_norm_is_inside_replacement(normalize(source), replacement_norms)
    ]
    if not sources or limit <= 0:
        return examples

    chunks = list(chunked(sources, max(1, chunk_size)))
    for source_chunk in progress(chunks, desc="Checking leftovers", unit="chunk"):
        pattern = span_pattern_many(source_chunk)
        for qid, query in queries.items():
            text = str(query.get("text", ""))
            match = pattern.search(text)
            if match and is_meaningful_leftover_source(normalize(match.group(0))):
                examples.append({"where": "query", "id": qid, "source": match.group(0), "text": text[:200]})
                if len(examples) >= limit:
                    return examples
        for doc_id, doc in corpus.items():
            text = f"{doc.get('title', '')}\n{doc.get('text', '')}"
            match = pattern.search(text)
            if match and is_meaningful_leftover_source(normalize(match.group(0))):
                examples.append({"where": "doc", "id": doc_id, "source": match.group(0), "text": text[:200]})
                if len(examples) >= limit:
                    return examples
    return examples


def span_pattern_many(sources: list[str]) -> re.Pattern[str]:
    alternatives = [flexible_span_piece(source) for source in sources]
    return re.compile(rf"(?<![A-Za-z0-9_])(?:{'|'.join(alternatives)})(?![A-Za-z0-9_])", flags=re.IGNORECASE)


def chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def progress(items: Iterable, **kwargs):
    if tqdm is None:
        return items
    return tqdm(items, **kwargs)


def flexible_span_piece(source: str) -> str:
    tokens = normalize(source).split()
    if not tokens:
        return re.escape(source)
    return r"[^A-Za-z0-9_]+".join(re.escape(token) for token in tokens)


def source_norm_is_inside_replacement(source_norm: str, replacement_norms: list[str]) -> bool:
    if not source_norm:
        return False
    needle = f" {source_norm} "
    return any(needle in f" {replacement_norm} " for replacement_norm in replacement_norms)


def is_meaningful_leftover_source(source_norm: str) -> bool:
    tokens = [token for token in source_norm.split() if token not in STOPWORDS]
    return any(len(token) > 1 for token in tokens)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def padded_norm(text: str) -> str:
    return f" {normalize(text)} "


if __name__ == "__main__":
    main()
