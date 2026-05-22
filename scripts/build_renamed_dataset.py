from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tqdm import tqdm

from expected_answer_rag.datasets import Document, Query, RetrievalDataset, load_dataset


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
RELATION_BOUNDARY_WORDS = {
    "album",
    "albums",
    "answer",
    "born",
    "capital",
    "city",
    "country",
    "date",
    "day",
    "episode",
    "episodes",
    "game",
    "games",
    "leader",
    "located",
    "movie",
    "name",
    "number",
    "played",
    "plays",
    "population",
    "record",
    "records",
    "sang",
    "season",
    "seasons",
    "series",
    "show",
    "sing",
    "singer",
    "sings",
    "state",
    "team",
    "year",
    "years",
}
CONNECTOR_WORDS = {"and", "of", "the", "for", "in", "on", "de", "la", "&", "at", "by", "to"}
BLOCKED_SINGLE_TOKEN_SOURCES = STOPWORDS | RELATION_BOUNDARY_WORDS | {
    "again",
    "all",
    "about",
    "above",
    "across",
    "after",
    "american",
    "any",
    "around",
    "before",
    "below",
    "between",
    "british",
    "canadian",
    "college",
    "company",
    "council",
    "congress",
    "during",
    "district",
    "east",
    "english",
    "empire",
    "every",
    "each",
    "few",
    "first",
    "government",
    "group",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "house",
    "its",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "last",
    "least",
    "many",
    "more",
    "most",
    "new",
    "next",
    "no",
    "north",
    "not",
    "office",
    "old",
    "one",
    "only",
    "our",
    "out",
    "over",
    "party",
    "place",
    "school",
    "senate",
    "she",
    "some",
    "south",
    "states",
    "than",
    "then",
    "there",
    "theme",
    "this",
    "through",
    "together",
    "under",
    "their",
    "them",
    "they",
    "two",
    "united",
    "university",
    "up",
    "us",
    "we",
    "west",
    "will",
    "world",
    "you",
    "your",
}
GENERIC_SINGLE_TOKEN_SOURCES = {
    "absolutely",
    "absorption",
    "access",
    "accord",
    "according",
    "accountant",
    "action",
    "activated",
    "active",
    "adult",
    "advertising",
    "affair",
    "afterwards",
    "age",
    "agreement",
    "air",
    "along",
    "alongside",
    "also",
    "although",
    "alternatives",
    "amendment",
    "another",
    "anything",
    "attempts",
    "authorities",
    "authority",
    "automotive",
    "background",
    "bar",
    "beginning",
    "being",
    "believing",
    "besides",
    "canal",
    "central",
    "characterizing",
    "clearing",
    "coastal",
    "commercial",
    "communication",
    "computational",
    "construction",
    "contrasting",
    "cooperation",
    "critical",
    "crossing",
    "dating",
    "depending",
    "development",
    "digital",
    "doing",
    "edition",
    "election",
    "electoral",
    "electrical",
    "elimination",
    "empirical",
    "environment",
    "environmental",
    "essential",
    "everything",
    "executive",
    "federal",
    "festival",
    "fighting",
    "filming",
    "final",
    "financial",
    "following",
    "forecasting",
    "formation",
    "foundation",
    "gathering",
    "general",
    "global",
    "going",
    "historical",
    "industrial",
    "information",
    "insisting",
}
GENERIC_TOKEN_SUFFIXES = (
    "ing",
    "tion",
    "sion",
    "ment",
    "ness",
    "ists",
    "ive",
    "ous",
    "ical",
    "ic",
    "al",
)
CAPITALIZED_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9'.-]*|[A-Z]{2,})"
    r"(?:\s+(?:of|the|and|for|in|on|de|la|&|[A-Z][A-Za-z0-9'.-]*|[A-Z]{2,})){0,5}\b"
)
QUOTED_RE = re.compile(r"[\"“”']([^\"“”']{3,80})[\"“”']")
NUMBER_RE = re.compile(r"\b\d{2,4}(?:[,.]\d+)?\b")
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
RETRIEVAL_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
ORGANIZATION_HINTS = {
    "agency",
    "association",
    "band",
    "club",
    "college",
    "commission",
    "committee",
    "company",
    "council",
    "department",
    "institute",
    "league",
    "party",
    "school",
    "team",
    "university",
}
TITLE_CONTEXT_HINTS = {
    "album",
    "book",
    "episode",
    "episodes",
    "film",
    "movie",
    "novel",
    "released",
    "season",
    "series",
    "show",
    "song",
    "soundtrack",
    "theme",
}
TITLE_SPAN_HINTS = {
    "album",
    "book",
    "episode",
    "film",
    "movie",
    "novel",
    "season",
    "song",
    "soundtrack",
}


@dataclass(frozen=True)
class Candidate:
    span: str
    kind: str
    source: str


@dataclass(frozen=True)
class TokenCorpusStats:
    total_count: int = 0
    doc_freq: int = 0
    uppercase_count: int = 0
    lowercase_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build opaque/plausible renamed local retrieval datasets.")
    parser.add_argument("--dataset", default="nq")
    parser.add_argument("--max-corpus", type=int, default=200000)
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-cache-root", default="D:/hf_cache")
    parser.add_argument("--mode", choices=["opaque", "plausible", "both"], default="both")
    parser.add_argument("--output-root", default="outputs/renamed_nq_stage2")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--min-query-replacements",
        type=int,
        default=0,
        help=(
            "Minimum replacements required to keep a query. Default 0 keeps the full query set; "
            "use 1 only for a changed-query diagnostic subset."
        ),
    )
    parser.add_argument(
        "--target-queries",
        type=int,
        default=None,
        help="Stop after this many safe renamed queries are kept. Use with a larger --max-queries source pool.",
    )
    parser.add_argument(
        "--query-rename-policy",
        choices=["safe_aligned", "all"],
        default="safe_aligned",
        help=(
            "safe_aligned only renames query anchors whose replacements also appear in the query's "
            "qrel-relevant documents after corpus renaming. all keeps the previous global replacement behavior."
        ),
    )
    parser.add_argument(
        "--query-ngram-anchors",
        choices=["off", "all"],
        default="off",
        help=(
            "Whether to add raw query n-grams as rename candidates. off is safer for BM25 because "
            "it avoids renaming generic relation phrases that happen to occur in relevant passages."
        ),
    )
    parser.add_argument(
        "--allow-entity-only-query",
        action="store_true",
        help=(
            "Compatibility flag. Entity-only queries are kept by default; use "
            "--require-named-query-replacement to restore the older conservative filter."
        ),
    )
    parser.add_argument(
        "--require-named-query-replacement",
        action="store_true",
        help="Skip queries whose aligned replacements are only generic ENTITY/NUMBER spans.",
    )
    parser.add_argument("--max-doc-entities-per-query", type=int, default=8)
    parser.add_argument("--max-number-replacements-per-query", type=int, default=3)
    parser.add_argument(
        "--max-token-doc-frequency",
        type=int,
        default=2000,
        help=(
            "For token-level renaming, skip non-number tokens that appear in more than this many "
            "documents. This keeps common words out of the synthetic entity mapping."
        ),
    )
    parser.add_argument("--max-span-tokens", type=int, default=6)
    parser.add_argument("--max-span-chars", type=int, default=80)
    parser.add_argument(
        "--replacement-token-policy",
        choices=["single", "preserve"],
        default="preserve",
        help=(
            "single replaces every span with one token. preserve keeps the replacement token count "
            "matched to the source span to avoid changing BM25 length/term structure unnecessarily."
        ),
    )
    parser.add_argument(
        "--replacement-granularity",
        choices=["token", "span"],
        default="token",
        help=(
            "token builds a corpus-wide token bijection for selected spans, which keeps BM25 closer "
            "to a vocabulary rename. span replaces whole spans and is more aggressive but can alter IDF."
        ),
    )
    parser.add_argument(
        "--allow-unaligned-query-anchors",
        action="store_true",
        help="Keep queries even when a renamed query anchor is missing from every relevant document.",
    )
    parser.add_argument(
        "--rename-workers",
        type=int,
        default=0,
        help="Parallel workers for corpus renaming. 0 uses a conservative auto setting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_local_cache(args.local_cache_root)
    dataset = load_dataset(
        args.dataset,
        max_corpus=args.max_corpus,
        max_queries=args.max_queries,
        cache_dir=args.cache_dir,
    )
    modes = ["opaque", "plausible"] if args.mode == "both" else [args.mode]
    for mode in modes:
        renamed, mapping, stats = build_renamed_dataset(dataset, mode=mode, args=args)
        out_dir = ROOT / args.output_root / mode
        write_local_dataset(renamed, out_dir, mapping=mapping, stats=stats)
        print(f"Wrote {mode} dataset to {out_dir}")


def configure_local_cache(cache_root: str | None) -> None:
    if not cache_root:
        return
    import os

    root = Path(cache_root)
    paths = {
        "HF_HOME": root,
        "HF_HUB_CACHE": root / "hub",
        "HF_DATASETS_CACHE": root / "datasets",
        "TRANSFORMERS_CACHE": root / "transformers",
        "SENTENCE_TRANSFORMERS_HOME": root / "sentence_transformers",
        "TMP": root / "tmp",
        "TEMP": root / "tmp",
    }
    for value in paths.values():
        value.mkdir(parents=True, exist_ok=True)
    for key, value in paths.items():
        os.environ.setdefault(key, str(value))


def build_renamed_dataset(
    dataset: RetrievalDataset,
    mode: str,
    args: argparse.Namespace,
) -> tuple[RetrievalDataset, dict[str, dict[str, str]], dict[str, object]]:
    random.seed(args.seed)
    doc_by_id = {doc.doc_id: doc for doc in dataset.corpus}
    query_by_id = {query.query_id: query for query in dataset.queries}
    mapping = build_mapping(dataset, doc_by_id, mode=mode, args=args)
    replacer = Replacer(mapping)

    renamed_corpus = rename_corpus(dataset.corpus, mapping, mode=mode, workers=args.rename_workers)

    renamed_queries = []
    kept_query_ids = set()
    skipped_no_qrels = 0
    skipped_no_query_change = 0
    skipped_entity_only_query = 0
    skipped_unaligned_query_anchors = 0
    queries_with_dropped_unaligned_anchors = 0
    dropped_unaligned_query_anchors = 0
    missing_query_replacements = 0
    kept_queries_with_replacements = 0
    kept_queries_without_replacements = 0
    renamed_doc_by_id = {doc.doc_id: doc for doc in renamed_corpus}
    for query in dataset.queries:
        if query.query_id not in dataset.qrels:
            skipped_no_qrels += 1
            continue
        effective_query_policy = "all" if args.replacement_granularity == "token" else args.query_rename_policy
        query_matches = query_safe_matches(
            query=query,
            qrels=dataset.qrels.get(query.query_id, {}),
            mapping=mapping,
            global_replacer=replacer,
            renamed_doc_by_id=renamed_doc_by_id,
            policy=effective_query_policy,
        )
        if args.query_rename_policy == "safe_aligned" and effective_query_policy == "safe_aligned":
            all_matches = replacer.matches(query.text)
            dropped = count_dropped_matches(all_matches, query_matches)
            if dropped:
                queries_with_dropped_unaligned_anchors += 1
                dropped_unaligned_query_anchors += dropped
        query_replacements = {match.replacement for match in query_matches}
        query_replacer = Replacer(mapping_for_matches(mapping, query_matches))
        renamed_text = query_replacer.replace(query.text)
        if len(query_replacements) < args.min_query_replacements:
            skipped_no_query_change += 1
            continue
        require_named = args.require_named_query_replacement and not args.allow_entity_only_query
        if require_named and not any(match.kind in NAMED_KINDS for match in query_matches):
            skipped_entity_only_query += 1
            continue
        missing = []
        if args.replacement_granularity != "token":
            missing = query_replacements_missing_in_relevant_docs(
                replacements=query_replacements,
                qrels=dataset.qrels.get(query.query_id, {}),
                doc_by_id=renamed_doc_by_id,
            )
        if missing and not args.allow_unaligned_query_anchors:
            skipped_unaligned_query_anchors += 1
            missing_query_replacements += len(missing)
            continue
        if query_replacements:
            kept_queries_with_replacements += 1
        else:
            kept_queries_without_replacements += 1
        renamed_queries.append(Query(query.query_id, renamed_text, query.answers))
        kept_query_ids.add(query.query_id)
        if args.target_queries is not None and len(renamed_queries) >= args.target_queries:
            break

    renamed_qrels = {
        qid: rels
        for qid, rels in dataset.qrels.items()
        if qid in kept_query_ids and any(did in doc_by_id for did in rels)
    }
    renamed = RetrievalDataset(
        name=f"{dataset.name}_renamed_{mode}",
        corpus=renamed_corpus,
        queries=renamed_queries,
        qrels=renamed_qrels,
    )
    stats = {
        "source_dataset": dataset.name,
        "mode": mode,
        "source_corpus": len(dataset.corpus),
        "source_queries": len(dataset.queries),
        "target_queries": args.target_queries,
        "source_qrels_queries": len(dataset.qrels),
        "renamed_corpus": len(renamed.corpus),
        "renamed_queries": len(renamed.queries),
        "renamed_qrels_queries": len(renamed.qrels),
        "mapping_size": len(mapping),
        "query_rename_policy": args.query_rename_policy,
        "effective_query_rename_policy": "all" if args.replacement_granularity == "token" else args.query_rename_policy,
        "replacement_token_policy": args.replacement_token_policy,
        "replacement_granularity": args.replacement_granularity,
        "query_ngram_anchors": args.query_ngram_anchors,
        "require_named_query_replacement": args.require_named_query_replacement and not args.allow_entity_only_query,
        "skipped_no_qrels": skipped_no_qrels,
        "skipped_no_query_change": skipped_no_query_change,
        "skipped_entity_only_query": skipped_entity_only_query,
        "skipped_unaligned_query_anchors": skipped_unaligned_query_anchors,
        "kept_queries_with_replacements": kept_queries_with_replacements,
        "kept_queries_without_replacements": kept_queries_without_replacements,
        "queries_with_dropped_unaligned_anchors": queries_with_dropped_unaligned_anchors,
        "dropped_unaligned_query_anchors": dropped_unaligned_query_anchors,
        "missing_query_replacements": missing_query_replacements,
        "replacement_kinds": dict(Counter(item["kind"] for item in mapping.values())),
    }
    return renamed, mapping, stats


def query_safe_matches(
    query: Query,
    qrels: dict[str, int],
    mapping: dict[str, dict[str, str]],
    global_replacer: "Replacer",
    renamed_doc_by_id: dict[str, Document],
    policy: str,
) -> list["ReplacementMatch"]:
    if policy == "all":
        return global_replacer.matches(query.text)

    query_norm = padded_norm(query.text)
    relevant_norm = padded_norm(relevant_doc_text(qrels, renamed_doc_by_id))
    safe_mapping = {
        normalized: row
        for normalized, row in mapping.items()
        if contains_phrase_in_padded_norm(query_norm, str(row["source"]))
        and contains_phrase_in_padded_norm(relevant_norm, str(row["replacement"]))
    }
    return Replacer(safe_mapping).matches(query.text)


def mapping_for_matches(
    mapping: dict[str, dict[str, str]],
    matches: list["ReplacementMatch"],
) -> dict[str, dict[str, str]]:
    allowed_sources = {match.source_norm for match in matches}
    return {normalized: row for normalized, row in mapping.items() if normalized in allowed_sources}


def count_dropped_matches(all_matches: list["ReplacementMatch"], kept_matches: list["ReplacementMatch"]) -> int:
    all_counts = Counter((match.source_norm, match.replacement) for match in all_matches)
    kept_counts = Counter((match.source_norm, match.replacement) for match in kept_matches)
    return sum(max(count - kept_counts[key], 0) for key, count in all_counts.items())


def relevant_doc_text(qrels: dict[str, int], doc_by_id: dict[str, Document]) -> str:
    return "\n".join(
        f"{doc_by_id[did].title}\n{doc_by_id[did].text}"
        for did in qrels
        if did in doc_by_id
    )


def build_mapping(
    dataset: RetrievalDataset,
    doc_by_id: dict[str, Document],
    mode: str,
    args: argparse.Namespace,
) -> dict[str, dict[str, str]]:
    raw: dict[str, Candidate] = {}
    usage: dict[str, set[str]] = defaultdict(set)
    generator = ReplacementFactory(mode=mode, seed=args.seed)
    for query in tqdm(dataset.queries, desc=f"Extracting rename spans ({mode})"):
        rels = dataset.qrels.get(query.query_id, {})
        if not rels:
            continue
        rel_docs = [doc_by_id[did] for did in rels if did in doc_by_id]
        if not rel_docs:
            continue
        rel_text = "\n".join(doc.text for doc in rel_docs[:3])
        candidates = []
        candidates.extend(extract_query_anchor_candidates(query, rel_docs, args))
        candidates.extend(extract_doc_candidates(query, rel_text, args))
        for candidate in candidates:
            normalized = normalize(candidate.span)
            if not is_valid_span(candidate.span, normalized, args.max_span_tokens, args.max_span_chars):
                continue
            existing = raw.get(normalized)
            if existing is None or kind_priority(candidate.kind) > kind_priority(existing.kind):
                raw[normalized] = candidate
            usage[normalized].add(query.query_id)

    if args.replacement_granularity == "token":
        raw, usage = token_level_candidates(raw, usage)
        raw, usage = filter_token_candidates_by_corpus_stats(
            raw,
            usage,
            corpus_token_stats(dataset.corpus, set(raw)),
            args,
        )

    mapping: dict[str, dict[str, str]] = {}
    for normalized, candidate in sorted(raw.items(), key=lambda item: (-len(item[0]), item[0])):
        replacement = generator.make(
            candidate.kind,
            token_count=replacement_token_count(candidate.span, args.replacement_token_policy),
        )
        mapping[normalized] = {
            "source": candidate.span,
            "replacement": replacement,
            "kind": candidate.kind,
            "candidate_source": candidate.source,
            "num_queries": str(len(usage[normalized])),
            "source_token_count": str(retrieval_token_count(candidate.span)),
            "replacement_token_count": str(retrieval_token_count(replacement)),
        }
    return mapping


def token_level_candidates(
    span_candidates: dict[str, Candidate],
    span_usage: dict[str, set[str]],
) -> tuple[dict[str, Candidate], dict[str, set[str]]]:
    token_candidates: dict[str, Candidate] = {}
    token_usage: dict[str, set[str]] = defaultdict(set)
    for span_norm, candidate in span_candidates.items():
        for token in candidate_tokens(candidate.span):
            token_norm = normalize(token)
            if not is_valid_token_candidate(token_norm):
                continue
            token_candidate = Candidate(
                span=token,
                kind=token_kind(token, candidate.kind),
                source=f"{candidate.source}_token",
            )
            existing = token_candidates.get(token_norm)
            if existing is None or kind_priority(token_candidate.kind) > kind_priority(existing.kind):
                token_candidates[token_norm] = token_candidate
            token_usage[token_norm].update(span_usage.get(span_norm, set()))
    return token_candidates, token_usage


def corpus_token_stats(corpus: list[Document], candidate_norms: set[str]) -> dict[str, TokenCorpusStats]:
    mutable: dict[str, list[int]] = {norm: [0, 0, 0, 0] for norm in candidate_norms}
    if not mutable:
        return {}
    for doc in tqdm(corpus, desc="Auditing token corpus stats"):
        seen_in_doc = set()
        for match in RETRIEVAL_TOKEN_RE.finditer(f"{doc.title}\n{doc.text}"):
            token = match.group(0)
            norm = normalize(token)
            counts = mutable.get(norm)
            if counts is None:
                continue
            counts[0] += 1
            if token[:1].isupper() or token.isupper():
                counts[2] += 1
            elif token[:1].islower():
                counts[3] += 1
            seen_in_doc.add(norm)
        for norm in seen_in_doc:
            mutable[norm][1] += 1
    return {
        norm: TokenCorpusStats(
            total_count=counts[0],
            doc_freq=counts[1],
            uppercase_count=counts[2],
            lowercase_count=counts[3],
        )
        for norm, counts in mutable.items()
    }


def filter_token_candidates_by_corpus_stats(
    candidates: dict[str, Candidate],
    usage: dict[str, set[str]],
    stats: dict[str, TokenCorpusStats],
    args: argparse.Namespace,
) -> tuple[dict[str, Candidate], dict[str, set[str]]]:
    kept: dict[str, Candidate] = {}
    kept_usage: dict[str, set[str]] = defaultdict(set)
    for norm, candidate in candidates.items():
        token_stats = stats.get(norm, TokenCorpusStats())
        if is_valid_token_by_corpus_stats(candidate, token_stats, args):
            kept[norm] = candidate
            kept_usage[norm].update(usage.get(norm, set()))
    return kept, kept_usage


def is_valid_token_by_corpus_stats(
    candidate: Candidate,
    stats: TokenCorpusStats,
    args: argparse.Namespace,
) -> bool:
    if candidate.kind == "NUMBER":
        return True
    source = candidate.span.strip()
    if not source:
        return False
    if stats.doc_freq > args.max_token_doc_frequency:
        return False
    if source.isupper() and len(source) >= 2:
        return True
    if stats.lowercase_count and stats.lowercase_count >= stats.uppercase_count:
        return False
    if source[:1].islower():
        return False
    return True


def candidate_tokens(span: str) -> list[str]:
    return [match.group(0) for match in RETRIEVAL_TOKEN_RE.finditer(span)]


def is_valid_token_candidate(normalized: str) -> bool:
    if not normalized or " " in normalized:
        return False
    if len(normalized) < 3:
        return False
    if normalized in STOPWORDS:
        return False
    if normalized in BLOCKED_SINGLE_TOKEN_SOURCES:
        return False
    if normalized in GENERIC_SINGLE_TOKEN_SOURCES:
        return False
    if len(normalized) > 6 and normalized.endswith(GENERIC_TOKEN_SUFFIXES):
        return False
    return True


def token_kind(token: str, parent_kind: str) -> str:
    if parent_kind == "NUMBER":
        return "NUMBER"
    return "ENTITY"


def replacement_token_count(span: str, policy: str) -> int:
    if policy == "single":
        return 1
    return max(1, retrieval_token_count(span))


def retrieval_token_count(text: str) -> int:
    return len(RETRIEVAL_TOKEN_RE.findall(text))


def extract_query_anchor_candidates(query: Query, rel_docs: list[Document], args: argparse.Namespace) -> list[Candidate]:
    candidates: list[Candidate] = []
    query_text = query.text
    rel_text = "\n".join([doc.title + "\n" + doc.text for doc in rel_docs])
    query_tokens = [token.lower() for token in TOKEN_RE.findall(query_text)]
    rel_norm = normalize(rel_text)

    for doc in rel_docs:
        for span in extract_named_spans(doc.title):
            if normalized_tokens(span) and all(token in query_tokens for token in normalized_tokens(span)):
                candidates.append(
                    Candidate(
                        span=span,
                        kind=infer_kind(span, f"{doc.title}\n{query_text}"),
                        source="title_anchor",
                    )
                )

    if args.query_ngram_anchors == "off":
        return dedupe_candidates(candidates)

    for n in range(min(7, len(query_tokens)), 1, -1):
        for start in range(0, len(query_tokens) - n + 1):
            tokens = query_tokens[start : start + n]
            if all(token in STOPWORDS for token in tokens):
                continue
            if tokens[0] in STOPWORDS or tokens[-1] in STOPWORDS:
                continue
            if tokens[0] in RELATION_BOUNDARY_WORDS or tokens[-1] in RELATION_BOUNDARY_WORDS:
                continue
            span = " ".join(tokens)
            if len(span) < 4:
                continue
            if span in rel_norm:
                candidates.append(Candidate(span=span, kind=infer_kind(span, query_text), source="query_ngram_anchor"))
    return dedupe_candidates(candidates)


def extract_doc_candidates(query: Query, rel_text: str, args: argparse.Namespace) -> list[Candidate]:
    candidates: list[Candidate] = []
    for span in extract_named_spans(rel_text):
        candidates.append(Candidate(span=span, kind=infer_kind(span, context=rel_text), source="relevant_doc_entity"))
    for span in extract_numbers(rel_text):
        if normalize(span) in normalize(query.text):
            continue
        candidates.append(Candidate(span=span, kind="NUMBER", source="relevant_doc_number"))

    entity_candidates = [item for item in candidates if item.kind != "NUMBER"]
    number_candidates = [item for item in candidates if item.kind == "NUMBER"]
    entity_candidates = [
        item
        for item in entity_candidates
        if should_keep_doc_candidate(item, query=query, rel_text=rel_text)
    ]
    entity_candidates = sorted(
        dedupe_candidates(entity_candidates),
        key=lambda item: doc_candidate_score(item.span, query.text, rel_text),
        reverse=True,
    )
    return (
        entity_candidates[: args.max_doc_entities_per_query]
        + dedupe_candidates(number_candidates)[: args.max_number_replacements_per_query]
    )


def should_keep_doc_candidate(candidate: Candidate, query: Query, rel_text: str) -> bool:
    tokens = normalized_tokens(candidate.span)
    if len(tokens) != 1:
        return True
    token = tokens[0]
    if token in GENERIC_SINGLE_TOKEN_SOURCES or token in BLOCKED_SINGLE_TOKEN_SOURCES:
        return False
    if token.isdigit():
        return True
    surface = candidate.span.strip()
    if surface.isupper() and len(surface) >= 2:
        return True
    return doc_candidate_score(candidate.span, query.text, rel_text) > 0


def doc_candidate_score(span: str, query_text: str, rel_text: str, window_chars: int = 120) -> int:
    query_terms = {
        token
        for token in normalized_tokens(query_text)
        if token not in STOPWORDS and token not in RELATION_BOUNDARY_WORDS
    }
    if not query_terms:
        return 0
    score = 0
    pattern = span_pattern(span)
    for match in pattern.finditer(rel_text):
        start = max(0, match.start() - window_chars)
        end = min(len(rel_text), match.end() + window_chars)
        window_terms = set(normalized_tokens(rel_text[start:end]))
        score = max(score, len(query_terms & window_terms))
    return score


def extract_named_spans(text: str) -> list[str]:
    spans: list[str] = []
    for segment in re.split(r"[\r\n]+", text):
        for match in QUOTED_RE.finditer(segment):
            value = clean_span(match.group(1))
            if value and is_plausible_named_span(value):
                spans.append(value)
        for match in CAPITALIZED_RE.finditer(segment):
            value = clean_span(match.group(0))
            if value and is_plausible_named_span(value):
                spans.append(value)
    return spans


def extract_numbers(text: str) -> list[str]:
    return [match.group(0) for match in NUMBER_RE.finditer(text)]


def clean_span(span: str) -> str:
    value = re.sub(r"\s+", " ", span.strip(" \t\r\n.,;:!?()[]{}")).strip()
    if ". " in value and not re.search(r"\b[A-Z]\.\s+[A-Z]", value):
        value = value.split(". ", 1)[0]
    tokens = value.split()
    while tokens and tokens[0].lower() in CONNECTOR_WORDS:
        tokens = tokens[1:]
    while tokens and tokens[-1].lower() in CONNECTOR_WORDS:
        tokens = tokens[:-1]
    return " ".join(tokens)


def has_named_signal(span: str) -> bool:
    return bool(re.search(r"\b[A-Z][a-z]+|\b[A-Z]{2,}\b|\d", span))


def is_plausible_named_span(span: str) -> bool:
    if not has_named_signal(span):
        return False
    normalized = normalize(span)
    tokens = normalized.split()
    if len(tokens) == 1:
        return is_allowed_single_token_span(span, normalized)

    surface_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'.-]*|&", span)
    meaningful = [token for token in surface_tokens if normalize(token) not in CONNECTOR_WORDS]
    if not meaningful:
        return False
    if not is_name_like_token(meaningful[0]) or not is_name_like_token(meaningful[-1]):
        return False
    for token in meaningful:
        if not is_name_like_token(token):
            return False
    return True


def is_name_like_token(token: str) -> bool:
    raw = token.strip(" \t\r\n.,;:!?()[]{}\"'")
    if not raw:
        return False
    if any(char.isdigit() for char in raw):
        return True
    compact = re.sub(r"[^A-Za-z]+", "", raw)
    if len(compact) >= 2 and compact.isupper():
        return True
    return len(compact) >= 2 and compact[0].isupper()


def is_allowed_single_token_span(span: str, normalized: str) -> bool:
    token = normalized.split()[0] if normalized.split() else ""
    if token in BLOCKED_SINGLE_TOKEN_SOURCES:
        return False
    raw = re.sub(r"[^A-Za-z0-9]+", "", span)
    if raw.isdigit():
        return True
    if len(raw) >= 2 and raw.isupper():
        return True
    return len(raw) >= 4 and bool(raw[:1].isupper())


def is_valid_span(span: str, normalized: str, max_tokens: int, max_chars: int) -> bool:
    if len(normalized) < 3:
        return False
    if len(span) > max_chars:
        return False
    tokens = normalized.split()
    if len(tokens) > max_tokens:
        return False
    if all(token in STOPWORDS for token in tokens):
        return False
    if tokens[0] in RELATION_BOUNDARY_WORDS or tokens[-1] in RELATION_BOUNDARY_WORDS:
        return False
    if len(tokens) == 1 and not is_allowed_single_token_span(span, normalized):
        return False
    return True


def infer_kind(span: str, context: str) -> str:
    lower = span.lower()
    context_lower = context.lower()
    if re.fullmatch(r"\d{2,4}(?:[,.]\d+)?", span):
        return "NUMBER"
    if any(word in lower for word in ORGANIZATION_HINTS):
        return "ORGANIZATION"
    if re.search(rf"(born in|located in|capital of|city of|country of)\s+{re.escape(lower)}", context_lower):
        return "LOCATION"
    if re.search(rf"{re.escape(lower)}\s+(city|county|province|state|country|river|mountain|island)", context_lower):
        return "LOCATION"
    if any(word in lower for word in TITLE_SPAN_HINTS):
        return "TITLE"
    if len(span.split()) >= 2 and any(word in context_lower for word in TITLE_CONTEXT_HINTS):
        return "TITLE"
    if len(span.split()) >= 2 and re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+)+\b", span):
        return "PERSON"
    if span.istitle() and len(span.split()) >= 2:
        return "TITLE"
    return "ENTITY"


def kind_priority(kind: str) -> int:
    return {
        "PERSON": 6,
        "LOCATION": 6,
        "ORGANIZATION": 5,
        "TITLE": 4,
        "NUMBER": 3,
        "ENTITY": 1,
    }.get(kind, 0)


def dedupe_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    seen = set()
    rows = []
    for candidate in candidates:
        normalized = normalize(candidate.span)
        if normalized in seen:
            continue
        seen.add(normalized)
        rows.append(candidate)
    return rows


def normalized_tokens(text: str) -> list[str]:
    return normalize(text).split()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def query_subspan_for(span: str, query: str) -> str | None:
    span_tokens = normalized_tokens(span)
    query_tokens = [token.lower() for token in TOKEN_RE.findall(query)]
    if not span_tokens:
        return None
    for start in range(0, len(query_tokens) - len(span_tokens) + 1):
        if query_tokens[start : start + len(span_tokens)] == span_tokens:
            return " ".join(query_tokens[start : start + len(span_tokens)])
    return None


NAMED_KINDS = {"PERSON", "LOCATION", "ORGANIZATION", "TITLE"}


@dataclass(frozen=True)
class ReplacementMatch:
    source_norm: str
    replacement: str
    kind: str


class Replacer:
    def __init__(self, mapping: dict[str, dict[str, str]]):
        self.items = sorted(mapping.values(), key=lambda row: len(row["source"]), reverse=True)
        self.lookup = {normalize(str(row["source"])): str(row["replacement"]) for row in self.items}
        self.kind_lookup = {normalize(str(row["source"])): str(row["kind"]) for row in self.items}
        self.pattern = build_combined_pattern([str(row["source"]) for row in self.items])

    def replace(self, text: str) -> str:
        if not text or self.pattern is None:
            return text
        return self.pattern.sub(self._replacement_for_match, text)

    def count_replacements(self, text: str) -> int:
        if not text or self.pattern is None:
            return 0
        return sum(1 for _ in self.pattern.finditer(text))

    def has_named_replacement(self, text: str) -> bool:
        return any(match.kind in NAMED_KINDS for match in self.matches(text))

    def matches(self, text: str) -> list[ReplacementMatch]:
        if not text or self.pattern is None:
            return []
        matches = []
        for match in self.pattern.finditer(text):
            source_norm = normalize(match.group(0))
            replacement = self.lookup.get(source_norm)
            if replacement is None:
                continue
            matches.append(
                ReplacementMatch(
                    source_norm=source_norm,
                    replacement=replacement,
                    kind=self.kind_lookup.get(source_norm, "ENTITY"),
                )
            )
        return matches

    def _replacement_for_match(self, match: re.Match[str]) -> str:
        return self.lookup.get(normalize(match.group(0)), match.group(0))


def replace_span(text: str, source: str, replacement: str) -> str:
    return span_pattern(source).sub(replacement, text)


def count_span(text: str, source: str) -> int:
    return len(span_pattern(source).findall(text))


def span_pattern(source: str) -> re.Pattern[str]:
    piece = flexible_span_piece(source)
    return re.compile(rf"(?<![A-Za-z0-9_]){piece}(?![A-Za-z0-9_])", flags=re.IGNORECASE)


def build_combined_pattern(sources: list[str]) -> re.Pattern[str] | None:
    pieces = []
    seen = set()
    for source in sorted(sources, key=len, reverse=True):
        normalized = normalize(source)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        pieces.append(flexible_span_piece(source))
    if not pieces:
        return None
    return re.compile(rf"(?<![A-Za-z0-9_])(?:{'|'.join(pieces)})(?![A-Za-z0-9_])", flags=re.IGNORECASE)


def flexible_span_piece(source: str) -> str:
    tokens = normalized_tokens(source)
    if not tokens:
        return re.escape(source)
    return r"[^A-Za-z0-9_]+".join(re.escape(token) for token in tokens)


def query_replacements_missing_in_relevant_docs(
    replacements: set[str],
    qrels: dict[str, int],
    doc_by_id: dict[str, Document],
) -> list[str]:
    if not replacements:
        return []
    relevant_text = "\n".join(
        f"{doc_by_id[did].title}\n{doc_by_id[did].text}"
        for did in qrels
        if did in doc_by_id
    )
    return sorted(
        replacement
        for replacement in replacements
        if not contains_normalized_phrase(relevant_text, replacement)
    )


def contains_normalized_phrase(text: str, phrase: str) -> bool:
    return contains_phrase_in_padded_norm(padded_norm(text), phrase)


def contains_phrase_in_padded_norm(normalized_text: str, phrase: str) -> bool:
    normalized_phrase = normalize(phrase)
    return bool(normalized_phrase) and f" {normalized_phrase} " in normalized_text


def padded_norm(text: str) -> str:
    return f" {normalize(text)} "


_WORKER_REPLACER: Replacer | None = None


def rename_corpus(
    corpus: list[Document],
    mapping: dict[str, dict[str, str]],
    mode: str,
    workers: int,
) -> list[Document]:
    worker_count = resolve_worker_count(workers)
    if worker_count <= 1:
        replacer = Replacer(mapping)
        return [
            Document(doc_id=doc.doc_id, title=replacer.replace(doc.title), text=replacer.replace(doc.text))
            for doc in tqdm(corpus, desc=f"Renaming corpus ({mode})")
        ]
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_rename_worker,
        initargs=(mapping,),
    ) as executor:
        return list(
            tqdm(
                executor.map(_rename_doc_worker, corpus, chunksize=128),
                total=len(corpus),
                desc=f"Renaming corpus ({mode}, workers={worker_count})",
            )
        )


def resolve_worker_count(workers: int) -> int:
    if workers > 0:
        return workers
    cpu_count = os.cpu_count() or 2
    return max(1, min(8, cpu_count - 2))


def _init_rename_worker(mapping: dict[str, dict[str, str]]) -> None:
    global _WORKER_REPLACER
    _WORKER_REPLACER = Replacer(mapping)


def _rename_doc_worker(doc: Document) -> Document:
    if _WORKER_REPLACER is None:
        raise RuntimeError("Rename worker was not initialized.")
    return Document(
        doc_id=doc.doc_id,
        title=_WORKER_REPLACER.replace(doc.title),
        text=_WORKER_REPLACER.replace(doc.text),
    )


class ReplacementFactory:
    def __init__(self, mode: str, seed: int):
        self.mode = mode
        self.random = random.Random(seed)
        self.counts: Counter[str] = Counter()
        self.used: set[str] = set()

    def make(self, kind: str, token_count: int = 1) -> str:
        self.counts[kind] += 1
        idx = self.counts[kind]
        if self.mode == "opaque":
            prefix = {
                "PERSON": "Person",
                "LOCATION": "Location",
                "ORGANIZATION": "Org",
                "TITLE": "Work",
                "NUMBER": "Number",
                "DATE": "Date",
                "ENTITY": "Entity",
            }.get(kind, "Entity")
            base = f"{prefix}_{code(idx)}"
        else:
            base = self._plausible(kind, idx)
        return self._unique(self._with_token_count(base, token_count))

    def _with_token_count(self, base: str, token_count: int) -> str:
        if token_count <= 1:
            return base
        compact = compact_token(base)
        return " ".join(f"{compact}Part{part}" for part in range(1, token_count + 1))

    def _plausible(self, kind: str, idx: int) -> str:
        if kind == "PERSON":
            first = PERSON_FIRST[(idx - 1) % len(PERSON_FIRST)]
            last = PERSON_LAST[((idx - 1) // len(PERSON_FIRST)) % len(PERSON_LAST)]
            return f"{first}{last}{idx:04d}"
        if kind == "LOCATION":
            return unique_name(LOCATION_NAMES, idx)
        if kind == "ORGANIZATION":
            return unique_name(ORG_NAMES, idx)
        if kind == "TITLE":
            return unique_name(WORK_TITLES, idx)
        if kind == "NUMBER":
            return str(900000 + idx)
        if kind == "DATE":
            return str(2100 + idx)
        return unique_name(ENTITY_NAMES, idx)

    def _unique(self, replacement: str) -> str:
        if replacement not in self.used:
            self.used.add(replacement)
            return replacement
        suffix = 2
        while f"{replacement} {suffix}" in self.used:
            suffix += 1
        unique = f"{replacement} {suffix}"
        self.used.add(unique)
        return unique


def code(idx: int) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    a = alphabet[idx % len(alphabet)]
    b = alphabet[(idx * 7) % len(alphabet)]
    return f"{a}{b}{idx:04d}"


def pick(values: list[str], idx: int) -> str:
    return values[(idx - 1) % len(values)]


def unique_name(values: list[str], idx: int) -> str:
    base = pick(values, idx)
    return f"{compact_token(base)}{idx:04d}"


def compact_token(value: str) -> str:
    return "".join(token.capitalize() for token in re.findall(r"[A-Za-z0-9]+", value))


PERSON_FIRST = ["Elena", "Tomas", "Clara", "Mira", "Jonas", "Anika", "Rafael", "Nadia", "Silas", "Leona"]
PERSON_LAST = ["Moravec", "Virek", "Densmore", "Kessari", "Halden", "Novik", "Brenner", "Sorell", "Tavik", "Marrow"]
LOCATION_NAMES = ["Brindleford", "Valemont", "Norchester", "Caldridge", "Esterwick", "Rivermere", "Lanton", "Oakhaven"]
ORG_NAMES = ["Asterline Council", "Meridian Works", "Northbridge Party", "Cobalt Union", "Harborlight Group", "Vireo Institute"]
WORK_TITLES = ["Silver Harbor", "The Last Meridian", "North of Amber", "Quiet Engines", "Riverlight", "The Glass Orchard"]
ENTITY_NAMES = ["Arden Vale", "Kelmar", "Vireo", "Lumen Gate", "Orison", "Marlow Field", "Solace Point", "Kepler Row"]


def write_local_dataset(
    dataset: RetrievalDataset,
    out_dir: Path,
    mapping: dict[str, dict[str, str]],
    stats: dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        out_dir / "corpus.jsonl",
        ({"_id": doc.doc_id, "title": doc.title, "text": corpus_body(doc)} for doc in dataset.corpus),
    )
    write_jsonl(
        out_dir / "queries.jsonl",
        ({"_id": query.query_id, "text": query.text, "answers": list(query.answers)} for query in dataset.queries),
    )
    qrel_rows = (
        {"query-id": qid, "corpus-id": did, "score": score}
        for qid, rels in dataset.qrels.items()
        for did, score in rels.items()
    )
    write_jsonl(out_dir / "qrels.jsonl", qrel_rows)
    (out_dir / "mapping.json").write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


def corpus_body(doc: Document) -> str:
    if doc.title and doc.text.startswith(f"{doc.title}\n"):
        return doc.text[len(doc.title) + 1 :]
    return doc.text


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
