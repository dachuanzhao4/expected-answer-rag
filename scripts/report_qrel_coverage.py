from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from expected_answer_rag.datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report qrel coverage under corpus caps.")
    parser.add_argument("--datasets", default="nq,scifact,hotpotqa")
    parser.add_argument("--corpus-sizes", default="2000,3000")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--cache-dir", default="outputs/hf_cache")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for dataset_name in _split(args.datasets):
        for corpus_size in [int(value) for value in _split(args.corpus_sizes)]:
            dataset = load_dataset(
                dataset_name,
                max_corpus=corpus_size,
                max_queries=args.max_queries,
                cache_dir=args.cache_dir,
            )
            coverage = dict(dataset.metadata.get("qrel_coverage") or {})
            rows.append(
                {
                    "dataset": dataset_name,
                    "max_corpus": corpus_size,
                    "num_corpus": len(dataset.corpus),
                    "num_queries": len(dataset.queries),
                    "num_qrels_queries": len(dataset.qrels),
                    "queries_with_relevant": coverage.get("queries_with_relevant", 0),
                    "queries_with_included_relevant": coverage.get("queries_with_included_relevant", 0),
                    "mean_coverage": coverage.get("mean_coverage", 0.0),
                    "min_coverage": coverage.get("min_coverage", 0.0),
                    "zero_coverage_count": len(coverage.get("zero_coverage_queries", [])),
                }
            )
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


def _split(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


if __name__ == "__main__":
    main()
