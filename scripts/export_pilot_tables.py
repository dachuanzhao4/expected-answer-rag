from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "docs"

DATASETS = ["nq", "scifact", "hotpotqa"]
DATASET_LABELS = {"nq": "NQ", "scifact": "SciFact", "hotpotqa": "HotpotQA"}
RETRIEVER_LABELS = {"bm25": "BM25", "dense": "Dense"}
METRIC = "ndcg@10"

CORE_METHODS = [
    ("query_only", "Query only"),
    ("raw_expected_answer_only", "Raw expected answer"),
    ("hyde_doc_only", "HyDE"),
    ("query2doc_concat", "Query2doc"),
    ("generative_relevance_feedback_concat", "GRF"),
    ("corpus_steered_short_concat", "Corpus-steered short"),
    ("concat_query_masked_expected", "Query + masked expected"),
    ("concat_query_answer_candidate_constrained_template", "Query + answer-constrained"),
    ("rrf_query_answer_constrained", "RRF(query, answer-constrained)"),
]

APPENDIX_METHODS = [
    ("raw_expected_answer_only", "Raw expected answer"),
    ("hyde_doc_only", "HyDE"),
    ("query2doc_concat", "Query2doc"),
    ("generative_relevance_feedback_concat", "GRF"),
    ("corpus_steered_short_concat", "Corpus-steered short"),
    ("concat_query_masked_expected", "Query + masked expected"),
    ("concat_query_answer_candidate_constrained_template", "Query + answer-constrained"),
    ("rrf_query_answer_constrained", "RRF(query, answer-constrained)"),
    ("wrong_answer_only", "Wrong answer only"),
    ("concat_query_wrong_answer", "Query + wrong answer"),
]


def main() -> None:
    runs = _load_runs()
    counts = _collect_counts(runs)
    paper_tables = build_paper_tables_md(runs, counts)
    appendix = build_appendix_md(runs, counts)

    DOCS.mkdir(parents=True, exist_ok=True)
    paper_path = DOCS / "pilot_results_tables.md"
    appendix_path = DOCS / "pilot_delta_excess_instability_appendix.md"
    paper_path.write_text(paper_tables, encoding="utf-8")
    appendix_path.write_text(appendix, encoding="utf-8")

    print(f"Wrote {paper_path}")
    print(f"Wrote {appendix_path}")


def _load_runs() -> dict[tuple[str, str, str], dict]:
    runs: dict[tuple[str, str, str], dict] = {}
    for dataset in DATASETS:
        for retriever in ["bm25", "dense"]:
            for regime in ["public", "entity_cf", "entity_value_cf"]:
                path = _run_path(dataset, retriever, regime)
                runs[(dataset, retriever, regime)] = json.loads(path.read_text(encoding="utf-8"))
    return runs


def _run_path(dataset: str, retriever: str, regime: str) -> Path:
    suffix = ""
    if regime == "entity_cf":
        suffix = "_cf"
    elif regime == "entity_value_cf":
        suffix = "_cf_ev"
    dense = "_dense" if retriever == "dense" else ""
    return OUTPUTS / f"{dataset}_100{suffix}{dense}_run.json"


def _collect_counts(runs: dict[tuple[str, str, str], dict]) -> dict[str, int]:
    return {
        dataset: int(runs[(dataset, "bm25", "public")]["num_qrels_queries"])
        for dataset in DATASETS
    }


def build_paper_tables_md(runs: dict[tuple[str, str, str], dict], counts: dict[str, int]) -> str:
    lines: list[str] = []
    lines.append("# Pilot Results Tables")
    lines.append("")
    lines.append("Metric: `nDCG@10`.")
    lines.append("These runs used `--max-queries 100` and `--max-corpus 200`.")
    lines.append("Core methods only; controls remain in the JSON outputs and [stress_test_findings.md](/Users/weiyueli/Desktop/rag/expected-answer-rag/docs/stress_test_findings.md).")
    lines.append("")
    lines.append("## Effective Query Counts")
    lines.append("")
    lines.extend(
        _markdown_table(
            ["Dataset", "Evaluable Queries"],
            [[DATASET_LABELS[dataset], str(counts[dataset])] for dataset in DATASETS],
        )
    )
    lines.append("")
    lines.append("## BM25 Public vs Entity-Counterfactual")
    lines.append("")
    lines.extend(_regime_table(runs, "bm25", "public", "entity_cf"))
    lines.append("")
    lines.append("## Dense Public vs Entity-Counterfactual")
    lines.append("")
    lines.extend(_regime_table(runs, "dense", "public", "entity_cf"))
    lines.append("")
    lines.append("## BM25 Entity+Value Counterfactual")
    lines.append("")
    lines.extend(_single_regime_table(runs, "bm25", "entity_value_cf"))
    lines.append("")
    lines.append("## Dense Entity+Value Counterfactual")
    lines.append("")
    lines.extend(_single_regime_table(runs, "dense", "entity_value_cf"))
    lines.append("")
    return "\n".join(lines)


def build_appendix_md(runs: dict[tuple[str, str, str], dict], counts: dict[str, int]) -> str:
    lines: list[str] = []
    lines.append("# Delta vs Query-Only and Excess Instability Appendix")
    lines.append("")
    lines.append("Definitions for each method `m` and dataset:")
    lines.append("")
    lines.append("`public_delta(m) = score_public(m) - score_public(query_only)`")
    lines.append("")
    lines.append("`cf_delta(m) = score_cf(m) - score_cf(query_only)`")
    lines.append("")
    lines.append("`excess_instability(m) = (score_public(m) - score_cf(m)) - (score_public(query_only) - score_cf(query_only))`")
    lines.append("")
    lines.append(f"Metric used throughout: `{METRIC}`.")
    lines.append("")
    lines.append("## Effective Query Counts")
    lines.append("")
    lines.extend(
        _markdown_table(
            ["Dataset", "Evaluable Queries"],
            [[DATASET_LABELS[dataset], str(counts[dataset])] for dataset in DATASETS],
        )
    )
    lines.append("")
    lines.append("## BM25 Entity-Counterfactual Diagnostics")
    lines.append("")
    lines.extend(_diagnostics_table(runs, "bm25"))
    lines.append("")
    lines.append("## Dense Entity-Counterfactual Diagnostics")
    lines.append("")
    lines.extend(_diagnostics_table(runs, "dense"))
    lines.append("")
    lines.append("## BM25 Entity+Value Delta vs Query-Only")
    lines.append("")
    lines.extend(_entity_value_delta_table(runs, "bm25"))
    lines.append("")
    lines.append("## Dense Entity+Value Delta vs Query-Only")
    lines.append("")
    lines.extend(_entity_value_delta_table(runs, "dense"))
    lines.append("")
    return "\n".join(lines)


def _regime_table(
    runs: dict[tuple[str, str, str], dict],
    retriever: str,
    public_regime: str,
    cf_regime: str,
) -> list[str]:
    headers = ["Method"]
    for dataset in DATASETS:
        headers.append(f"{DATASET_LABELS[dataset]} (Public)")
        headers.append(f"{DATASET_LABELS[dataset]} (CF)")
    rows = []
    for method, label in CORE_METHODS:
        row = [label]
        for dataset in DATASETS:
            row.append(_fmt_metric(_metric(runs, dataset, retriever, public_regime, method)))
            row.append(_fmt_metric(_metric(runs, dataset, retriever, cf_regime, method)))
        rows.append(row)
    return _markdown_table(headers, rows)


def _single_regime_table(
    runs: dict[tuple[str, str, str], dict],
    retriever: str,
    regime: str,
) -> list[str]:
    headers = ["Method", *[DATASET_LABELS[dataset] for dataset in DATASETS]]
    rows = []
    for method, label in CORE_METHODS:
        rows.append([label, *[_fmt_metric(_metric(runs, dataset, retriever, regime, method)) for dataset in DATASETS]])
    return _markdown_table(headers, rows)


def _diagnostics_table(runs: dict[tuple[str, str, str], dict], retriever: str) -> list[str]:
    headers = [
        "Method",
        "Avg Public Δ",
        "Avg CF Δ",
        "Avg Excess Instab.",
        "NQ Excess",
        "SciFact Excess",
        "HotpotQA Excess",
    ]
    rows = []
    for method, label in APPENDIX_METHODS:
        public_deltas = []
        cf_deltas = []
        excesses = []
        per_dataset = []
        for dataset in DATASETS:
            public_delta = _metric(runs, dataset, retriever, "public", method) - _metric(runs, dataset, retriever, "public", "query_only")
            cf_delta = _metric(runs, dataset, retriever, "entity_cf", method) - _metric(runs, dataset, retriever, "entity_cf", "query_only")
            excess = (_metric(runs, dataset, retriever, "public", method) - _metric(runs, dataset, retriever, "entity_cf", method)) - (
                _metric(runs, dataset, retriever, "public", "query_only") - _metric(runs, dataset, retriever, "entity_cf", "query_only")
            )
            public_deltas.append(public_delta)
            cf_deltas.append(cf_delta)
            excesses.append(excess)
            per_dataset.append(excess)
        rows.append(
            [
                label,
                _fmt_delta(_mean(public_deltas)),
                _fmt_delta(_mean(cf_deltas)),
                _fmt_delta(_mean(excesses)),
                *[_fmt_delta(value) for value in per_dataset],
            ]
        )
    return _markdown_table(headers, rows)


def _entity_value_delta_table(runs: dict[tuple[str, str, str], dict], retriever: str) -> list[str]:
    headers = ["Method", "NQ Δ", "SciFact Δ", "HotpotQA Δ", "Avg Δ"]
    rows = []
    for method, label in APPENDIX_METHODS:
        values = []
        row = [label]
        for dataset in DATASETS:
            delta = _metric(runs, dataset, retriever, "entity_value_cf", method) - _metric(
                runs, dataset, retriever, "entity_value_cf", "query_only"
            )
            values.append(delta)
            row.append(_fmt_delta(delta))
        row.append(_fmt_delta(_mean(values)))
        rows.append(row)
    return _markdown_table(headers, rows)


def _metric(
    runs: dict[tuple[str, str, str], dict],
    dataset: str,
    retriever: str,
    regime: str,
    method: str,
) -> float:
    return float(runs[(dataset, retriever, regime)]["metrics"][method][METRIC])


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _fmt_metric(value: float) -> str:
    return f"{value:.3f}"


def _fmt_delta(value: float) -> str:
    return f"{value:+.3f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    align = ["---"] + [":---:" for _ in headers[1:]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


if __name__ == "__main__":
    main()
