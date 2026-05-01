from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
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

    def search(self, query: str, top_k: int = 10) -> RankedList:
        scores: Dict[str, float] = defaultdict(float)
        query_terms = Counter(tokenize(query))
        for term, qtf in query_terms.items():
            postings = self.inverted.get(term)
            if not postings:
                continue
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
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64

    def __post_init__(self) -> None:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install 'sentence-transformers' and 'numpy' to use dense retrieval."
            ) from exc
        self._np = np
        self._model = SentenceTransformer(self.model_name)
        self._doc_ids = [doc.doc_id for doc in self.documents]
        texts = [doc.text for doc in self.documents]
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        self._embeddings = np.asarray(embeddings)

    def search(self, query: str, top_k: int = 10) -> RankedList:
        query_embedding = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self._embeddings @ query_embedding
        top_indices = self._np.argsort(-scores)[:top_k]
        return [(self._doc_ids[idx], float(scores[idx])) for idx in top_indices]


def make_retriever(kind: str, documents: List[Document]) -> Retriever:
    if kind == "bm25":
        return BM25Retriever(documents)
    if kind == "dense":
        return SentenceTransformerRetriever(documents)
    raise ValueError(f"Unknown retriever kind: {kind}")
