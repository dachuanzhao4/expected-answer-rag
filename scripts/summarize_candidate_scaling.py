#!/usr/bin/env python3
"""Summarize candidate-pool scaling runs for the FAWE draft.

The script is intentionally dependency-free. It reads completed
outputs_pi/c{size}/*_bm25_run.json files and reports the main draft placeholder:

    nDCG@10(FAWE-Q2D beta=0.25) - nDCG@10(Query2doc)

for Entity-CF and Entity+Value-CF regimes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_DATASETS = ("nq", "scifact", "hotpotqa")
DATASET_LABELS = {
    "nq": "NQ",
    "scifact": "SciFact",
    "hotpotqa": "HotpotQA",
}
REGIMES = (
    ("entity", "_cf", "Entity-CF"),
    ("entity_value", "_cf_ev", "Entity+Value-CF"),
)
BASELINE_METHOD = "query2doc_concat"
FAWE_METHOD = "fawe_query2doc_beta0p25"
METRIC = "ndcg@10"


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.replace(" ", ",").split(",") if part.strip()]


def parse_sizes(value: str) -> list[int]:
    return [int(part) for part in parse_csv(value)]


def run_path(outputs_root: Path, size: int, dataset: str, max_queries: int, suffix: str) -> Path | None:
    exact = outputs_root / f"c{size}" / f"{dataset}_{max_queries}_c{size}{suffix}_bm25_run.json"
    if exact.exists():
        return exact
    matches = sorted((outputs_root / f"c{size}").glob(f"{dataset}_*_c{size}{suffix}_bm25_run.json"))
    return matches[0] if matches else None


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_value(run: dict[str, Any] | None, method: str, metric: str) -> float | None:
    if not run:
        return None
    value = run.get("metrics", {}).get(method, {}).get(metric)
    return float(value) if value is not None else None


def fmt_delta(value: float | None) -> str:
    if value is None:
        return "TBD"
    return f"{value:+.3f}"


def fmt_latex_delta(value: float | None) -> str:
    if value is None:
        return r"\textcolor{red}{TBD}"
    return f"{value:+.3f}"


def coverage_fields(run: dict[str, Any] | None) -> dict[str, Any]:
    coverage = (run or {}).get("qrel_coverage", {}) or {}
    return {
        "num_queries": (run or {}).get("num_queries"),
        "num_corpus": (run or {}).get("num_corpus"),
        "queries_with_relevant": coverage.get("queries_with_relevant"),
        "queries_with_included_relevant": coverage.get("queries_with_included_relevant"),
        "mean_coverage": coverage.get("mean_coverage"),
        "min_coverage": coverage.get("min_coverage"),
    }


def collect_rows(outputs_root: Path, sizes: list[int], datasets: list[str], max_queries: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for regime_key, suffix, regime_label in REGIMES:
            row: dict[str, Any] = {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS.get(dataset, dataset),
                "regime": regime_key,
                "regime_label": regime_label,
                "deltas": {},
                "baseline_scores": {},
                "fawe_scores": {},
                "coverage": {},
                "paths": {},
            }
            for size in sizes:
                path = run_path(outputs_root, size, dataset, max_queries, suffix)
                run = load_json(path)
                baseline = metric_value(run, BASELINE_METHOD, METRIC)
                fawe = metric_value(run, FAWE_METHOD, METRIC)
                row["baseline_scores"][str(size)] = baseline
                row["fawe_scores"][str(size)] = fawe
                row["deltas"][str(size)] = None if baseline is None or fawe is None else fawe - baseline
                row["coverage"][str(size)] = coverage_fields(run)
                row["paths"][str(size)] = str(path) if path else None
            rows.append(row)
    return rows


def write_json(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metric": METRIC,
                "baseline_method": BASELINE_METHOD,
                "fawe_method": FAWE_METHOD,
                "rows": rows,
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")


def write_csv(rows: list[dict[str, Any]], sizes: list[int], path: Path) -> None:
    fieldnames = ["dataset", "regime"]
    for size in sizes:
        fieldnames.extend(
            [
                f"c{size}_query2doc_{METRIC}",
                f"c{size}_fawe_q2d_beta0p25_{METRIC}",
                f"c{size}_delta",
                f"c{size}_mean_qrel_coverage",
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out: dict[str, Any] = {
                "dataset": row["dataset"],
                "regime": row["regime"],
            }
            for size in sizes:
                key = str(size)
                out[f"c{size}_query2doc_{METRIC}"] = row["baseline_scores"].get(key)
                out[f"c{size}_fawe_q2d_beta0p25_{METRIC}"] = row["fawe_scores"].get(key)
                out[f"c{size}_delta"] = row["deltas"].get(key)
                out[f"c{size}_mean_qrel_coverage"] = row["coverage"].get(key, {}).get("mean_coverage")
            writer.writerow(out)


def write_coverage_csv(rows: list[dict[str, Any]], sizes: list[int], path: Path) -> None:
    fieldnames = [
        "dataset",
        "regime",
        "corpus_size",
        "num_corpus",
        "num_queries",
        "queries_with_relevant",
        "queries_with_included_relevant",
        "mean_coverage",
        "min_coverage",
        "run_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for size in sizes:
                key = str(size)
                coverage = row["coverage"].get(key, {})
                writer.writerow(
                    {
                        "dataset": row["dataset"],
                        "regime": row["regime"],
                        "corpus_size": size,
                        "num_corpus": coverage.get("num_corpus"),
                        "num_queries": coverage.get("num_queries"),
                        "queries_with_relevant": coverage.get("queries_with_relevant"),
                        "queries_with_included_relevant": coverage.get("queries_with_included_relevant"),
                        "mean_coverage": coverage.get("mean_coverage"),
                        "min_coverage": coverage.get("min_coverage"),
                        "run_path": row["paths"].get(key),
                    }
                )


def write_markdown(rows: list[dict[str, Any]], sizes: list[int], path: Path) -> None:
    headers = ["Dataset", "Regime", *[f"{size:,} Δ" for size in sizes]]
    lines = [
        "# Candidate-Pool Scaling Summary",
        "",
        f"Cells report `{METRIC}({FAWE_METHOD}) - {METRIC}({BASELINE_METHOD})`.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        cells = [row["dataset_label"], row["regime_label"]]
        cells.extend(fmt_delta(row["deltas"].get(str(size))) for size in sizes)
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Use this table to fill the draft's candidate-pool scaling placeholder.",
            "Missing cells are marked `TBD` because the corresponding run file was not found.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex(rows: list[dict[str, Any]], sizes: list[int], path: Path) -> None:
    lines = [
        "% Auto-generated by scripts/summarize_candidate_scaling.py.",
        f"% Cells report {METRIC}({FAWE_METHOD}) - {METRIC}({BASELINE_METHOD}).",
    ]
    for row in rows:
        cells = [row["dataset_label"], row["regime_label"]]
        cells.extend(fmt_latex_delta(row["deltas"].get(str(size))) for size in sizes)
        lines.append(" & ".join(cells) + r" \\")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs_pi")
    parser.add_argument("--sizes", default="1000,3000,5000")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs_pi/scaling_summary")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    sizes = parse_sizes(args.sizes)
    datasets = parse_csv(args.datasets)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(outputs_root, sizes, datasets, args.max_queries)
    write_json(rows, output_dir / "candidate_scaling_summary.json")
    write_csv(rows, sizes, output_dir / "candidate_scaling_summary.csv")
    write_coverage_csv(rows, sizes, output_dir / "candidate_scaling_coverage.csv")
    write_markdown(rows, sizes, output_dir / "candidate_scaling_table.md")
    write_latex(rows, sizes, output_dir / "candidate_scaling_latex_rows.tex")

    print(f"Wrote candidate-pool scaling summaries to {output_dir}")


if __name__ == "__main__":
    main()
