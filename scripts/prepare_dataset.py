from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from expected_answer_rag.datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and sanity-check a retrieval dataset.")
    parser.add_argument("--dataset", default="nq", help="toy, nq, hotpotqa, fiqa, scifact, ...")
    parser.add_argument("--max-corpus", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--query-metadata", default=None)
    parser.add_argument("--output", default="outputs/dataset_preview.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(
        args.dataset,
        max_corpus=args.max_corpus,
        max_queries=args.max_queries,
        cache_dir=args.cache_dir,
        query_metadata_path=args.query_metadata,
    )
    preview = {
        "dataset": dataset.name,
        "num_corpus": len(dataset.corpus),
        "num_queries": len(dataset.queries),
        "num_qrels_queries": len(dataset.qrels),
        "corpus_examples": [doc.__dict__ for doc in dataset.corpus[:3]],
        "query_examples": [query.__dict__ for query in dataset.queries[:3]],
        "qrels_examples": dict(list(dataset.qrels.items())[:3]),
    }
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(preview, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(preview, indent=2, ensure_ascii=False))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
