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

from expected_answer_rag.metrics import per_query_metrics
from expected_answer_rag.statistics import paired_bootstrap_ci


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect metric means and bootstrap error bars from record files.")
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--methods", default="")
    parser.add_argument("--metrics", default="ndcg@10,recall@20,recall@100,mrr@10")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_methods = {item.strip() for item in args.methods.split(",") if item.strip()}
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    rows = []
    for path in args.records:
        records = _load_records(path)
        methods = requested_methods or sorted({method for record in records for method in record.get("rankings", {})})
        for method in methods:
            for metric in metrics:
                values = []
                for record in records:
                    ranking = record.get("rankings", {}).get(method)
                    if ranking is None:
                        continue
                    values.append(per_query_metrics(ranking, record.get("qrels", {}), ks=(5, 10, 20, 100)).get(metric, 0.0))
                if not values:
                    continue
                ci = paired_bootstrap_ci(values, num_samples=args.bootstrap_samples)
                rows.append(
                    {
                        "records": Path(path).name,
                        "method": method,
                        "metric": metric,
                        "n": len(values),
                        "mean": ci["mean"],
                        "ci_low": ci["low"],
                        "ci_high": ci["high"],
                    }
                )
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["records", "method", "metric", "n", "mean", "ci_low", "ci_high"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


def _load_records(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
