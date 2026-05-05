from __future__ import annotations

import math
import re
import json
import hashlib
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Protocol, Tuple

from expected_answer_rag.datasets import Document


RankedList = List[Tuple[str, float]]


class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> RankedList:
        ...


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


@dataclass
class BM25Retriever:
    documents: List[Document]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.doc_len: Dict[str, int] = {}
        self.term_freqs: Dict[str, Counter[str]] = {}
        self.inverted: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.doc_freqs: Dict[str, int] = {}
        self.idf_by_term: Dict[str, float] = {}
        self.vocabulary_terms: List[str] = []
        total_len = 0

        for doc in self.documents:
            tokens = tokenize(doc.text)
            counts = Counter(tokens)
            self.doc_len[doc.doc_id] = len(tokens)
            self.term_freqs[doc.doc_id] = counts
            total_len += len(tokens)
            for term, freq in counts.items():
                self.inverted[term][doc.doc_id] = freq

        self.avg_doc_len = total_len / max(len(self.documents), 1)
        self.num_docs = len(self.documents)
        self.doc_freqs = {term: len(postings) for term, postings in self.inverted.items()}
        self.idf_by_term = {
            term: math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))
            for term, df in self.doc_freqs.items()
        }
        self.vocabulary_terms = sorted(self.inverted)

    def search(self, query: str, top_k: int = 10) -> RankedList:
        scores: Dict[str, float] = defaultdict(float)
        query_terms = Counter(tokenize(query))
        for term, qtf in query_terms.items():
            postings = self.inverted.get(term)
            if not postings:
                continue
            idf = self.idf_by_term.get(term)
            if idf is None:
                df = len(postings)
                idf = math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))
            for doc_id, tf in postings.items():
                doc_len = self.doc_len[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1e-9))
                scores[doc_id] += qtf * idf * (tf * (self.k1 + 1)) / denom
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]


@dataclass
class SentenceTransformerRetriever:
    documents: List[Document]
    model_name: str = "BAAI/bge-base-en-v1.5"
    batch_size: int = 64
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    embedding_cache: str | None = None
    chunk_size: int = 1024
    local_files_only: bool = False

    def __post_init__(self) -> None:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install 'sentence-transformers' and 'numpy' to use dense retrieval."
            ) from exc
        self._np = np
        local_only = self.local_files_only or _env_truthy("HF_HUB_OFFLINE") or _env_truthy("TRANSFORMERS_OFFLINE") or _env_truthy("EXPECTED_ANSWER_RAG_LOCAL_FILES_ONLY")
        self._model = SentenceTransformer(self.model_name, local_files_only=local_only)
        self._doc_ids = [doc.doc_id for doc in self.documents]
        cached = self._load_cached_embeddings()
        if cached is not None:
            self._embeddings = cached
            return
        self._embeddings = self._encode_documents_chunked()
        self._save_cached_embeddings(self._embeddings)

    def search(self, query: str, top_k: int = 10) -> RankedList:
        query_text = f"{self.query_prefix}{query}" if self.query_prefix else query
        query_embedding = self._model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self._embeddings @ query_embedding
        top_indices = self._np.argsort(-scores)[:top_k]
        return [(self._doc_ids[idx], float(scores[idx])) for idx in top_indices]

    def _encode_documents_chunked(self):
        chunks = []
        for start in range(0, len(self.documents), self.chunk_size):
            end = min(start + self.chunk_size, len(self.documents))
            cached_chunk = self._load_chunk(start, end)
            if cached_chunk is not None:
                chunks.append(cached_chunk)
                continue
            texts = [doc.text for doc in self.documents[start:end]]
            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            chunk = self._np.asarray(embeddings, dtype="float32")
            self._save_chunk(start, end, chunk)
            chunks.append(chunk)
        return self._np.vstack(chunks)

    def _cache_paths(self) -> tuple[Path, Path] | None:
        if not self.embedding_cache:
            return None
        base = Path(self.embedding_cache)
        return base.with_suffix(".npy"), base.with_suffix(".json")

    def _load_cached_embeddings(self):
        paths = self._cache_paths()
        if paths is None:
            return None
        embeddings_path, metadata_path = paths
        if not embeddings_path.exists() or not metadata_path.exists():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("model_name") != self.model_name:
            return None
        if metadata.get("doc_ids") != self._doc_ids:
            return None
        return self._np.load(embeddings_path)

    def _save_cached_embeddings(self, embeddings) -> None:
        paths = self._cache_paths()
        if paths is None:
            return
        embeddings_path, metadata_path = paths
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        self._np.save(embeddings_path, embeddings)
        metadata = {
            "model_name": self.model_name,
            "doc_ids": self._doc_ids,
            "dimension": int(embeddings.shape[1]),
            "num_documents": len(self._doc_ids),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    def _chunk_dir(self) -> Path | None:
        if not self.embedding_cache:
            return None
        return Path(f"{self.embedding_cache}_chunks")

    def _chunk_paths(self, start: int, end: int) -> tuple[Path, Path] | None:
        chunk_dir = self._chunk_dir()
        if chunk_dir is None:
            return None
        stem = f"chunk_{start:08d}_{end:08d}"
        return chunk_dir / f"{stem}.npy", chunk_dir / f"{stem}.json"

    def _load_chunk(self, start: int, end: int):
        paths = self._chunk_paths(start, end)
        if paths is None:
            return None
        embeddings_path, metadata_path = paths
        if not embeddings_path.exists() or not metadata_path.exists():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_doc_ids = self._doc_ids[start:end]
        if metadata.get("model_name") != self.model_name:
            return None
        if metadata.get("start") != start or metadata.get("end") != end:
            return None
        if metadata.get("doc_ids_sha256") != _hash_strings(expected_doc_ids):
            return None
        chunk = self._np.load(embeddings_path)
        if chunk.shape[0] != end - start:
            return None
        return chunk

    def _save_chunk(self, start: int, end: int, chunk) -> None:
        paths = self._chunk_paths(start, end)
        if paths is None:
            return
        embeddings_path, metadata_path = paths
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        self._np.save(embeddings_path, chunk)
        metadata = {
            "model_name": self.model_name,
            "start": start,
            "end": end,
            "num_documents": end - start,
            "dimension": int(chunk.shape[1]),
            "doc_ids_sha256": _hash_strings(self._doc_ids[start:end]),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")


def make_retriever(
    kind: str,
    documents: List[Document],
    embedding_model: str = "BAAI/bge-base-en-v1.5",
    embedding_batch_size: int = 64,
    query_prefix: str = "Represent this sentence for searching relevant passages: ",
    embedding_cache: str | None = None,
    embedding_chunk_size: int = 1024,
    local_files_only: bool = False,
) -> Retriever:
    if kind == "bm25":
        return BM25Retriever(documents)
    if kind == "dense":
        return SentenceTransformerRetriever(
            documents=documents,
            model_name=embedding_model,
            batch_size=embedding_batch_size,
            query_prefix=query_prefix,
            embedding_cache=embedding_cache,
            chunk_size=embedding_chunk_size,
            local_files_only=local_files_only,
        )
    raise ValueError(f"Unknown retriever kind: {kind}")


def _hash_strings(values: List[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _env_truthy(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}
