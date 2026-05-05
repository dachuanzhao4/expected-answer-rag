from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path


DATASETS = ["nq", "hotpotqa", "scifact"]
DATASET_LABELS = {"nq": "NQ", "hotpotqa": "HotpotQA", "scifact": "SciFact"}
REGIMES = [("public", ""), ("entity", "_cf"), ("entity_value", "_cf_ev")]
REGIME_LABELS = {"public": "Public", "entity": "Entity-CF", "entity_value": "Entity+Value-CF"}
CORE_METHODS = [
    ("query_only", "Query"),
    ("query2doc_concat", "Query2doc"),
    ("hyde_doc_only", "HyDE"),
    ("concat_query_raw_expected", "Anchored answer"),
    ("fawe_query2doc_beta0p25", "FAWE-Q2D"),
    ("fawe_safe_adaptive_beta", "FAWE-Adapt"),
]
BETA_RE = re.compile(r"^fawe_query2doc_beta(?P<beta>[0-9]+p[0-9]+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PI follow-up figures with available error bars.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric", default="ndcg@10")
    return parser.parse_args()


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping PI figure generation.")
        return

    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = _load_bm25_runs(input_dir)
    if runs:
        _plot_bm25_utility(plt, runs, args.metric, output_dir)
        _plot_beta_sweep(plt, runs, args.metric, output_dir)
    cross = _load_cross_runs(input_dir, expansion="query2doc")
    if cross:
        _plot_cross_regime(plt, cross, args.metric, output_dir)
    print(f"Wrote PI figures to {output_dir}")


def _load_bm25_runs(input_dir: Path) -> dict[tuple[str, str], dict]:
    runs = {}
    for ds in DATASETS:
        for regime, suffix in REGIMES:
            matches = sorted(input_dir.glob(f"{ds}_*_c*{suffix}_bm25_run.json"))
            if regime == "public":
                matches = [
                    path for path in matches
                    if "_cf_bm25_run" not in path.name and "_cf_ev_bm25_run" not in path.name
                ]
            if matches:
                runs[(ds, regime)] = json.loads(matches[-1].read_text(encoding="utf-8"))
    return runs


def _load_cross_runs(input_dir: Path, expansion: str) -> dict[tuple[str, str], dict]:
    runs = {}
    for ds in DATASETS:
        for regime, suffix in REGIMES[1:]:
            matches = sorted(input_dir.glob(f"{ds}_*_c*{suffix}_cross_{expansion}.json"))
            if matches:
                runs[(ds, regime)] = json.loads(matches[-1].read_text(encoding="utf-8"))
    return runs


def _plot_bm25_utility(plt, runs: dict[tuple[str, str], dict], metric: str, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (regime, _suffix) in zip(axes, REGIMES):
        x = list(range(len(DATASETS)))
        offsets = _offsets(len(CORE_METHODS), width=0.1)
        for offset, (method, label) in zip(offsets, CORE_METHODS):
            ys = []
            lows = []
            highs = []
            for ds in DATASETS:
                run = runs.get((ds, regime), {})
                mean, low, high = _metric_with_ci(run, method, metric)
                ys.append(mean)
                lows.append(max(0.0, mean - low))
                highs.append(max(0.0, high - mean))
            ax.errorbar(
                [value + offset for value in x],
                ys,
                yerr=[lows, highs],
                marker="o",
                capsize=3,
                linewidth=1.6,
                label=label,
            )
        ax.set_title(REGIME_LABELS[regime])
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[ds] for ds in DATASETS], rotation=20)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(metric)
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    _save(fig, output_dir / "bm25_utility_with_error_bars")


def _plot_beta_sweep(plt, runs: dict[tuple[str, str], dict], metric: str, output_dir: Path) -> None:
    beta_values = sorted({
        _parse_beta(method)
        for run in runs.values()
        for method in run.get("metrics", {})
        if BETA_RE.match(method)
    })
    beta_values = [value for value in beta_values if value is not None]
    if not beta_values:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for regime, _suffix in REGIMES:
        means = []
        errors = []
        for beta in beta_values:
            method = f"fawe_query2doc_beta{str(beta).replace('.', 'p')}"
            values = [
                _metric_with_ci(runs[(ds, regime)], method, metric)[0]
                for ds in DATASETS
                if (ds, regime) in runs
            ]
            means.append(sum(values) / len(values) if values else 0.0)
            errors.append(_stderr(values))
        ax.errorbar(beta_values, means, yerr=errors, marker="o", capsize=3, linewidth=2, label=REGIME_LABELS[regime])
    ax.set_xlabel("FAWE Query2doc beta")
    ax.set_ylabel(f"Mean {metric}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _save(fig, output_dir / "fawe_beta_sweep_with_error_bars")


def _plot_cross_regime(plt, runs: dict[tuple[str, str], dict], metric: str, output_dir: Path) -> None:
    methods = [
        ("g_pub_on_cf", "g_pub"),
        ("g_cf_on_cf", "g_cf"),
        ("q_cf_plus_g_pub_on_cf", "q_cf+g_pub"),
        ("q_cf_plus_g_cf_on_cf", "q_cf+g_cf"),
        ("fawe_q_cf_g_pub_on_cf", "FAWE g_pub"),
        ("fawe_q_cf_g_cf_on_cf", "FAWE g_cf"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.0), sharey=True)
    for ax, ((ds, regime), run) in zip(axes.ravel(), sorted(runs.items())):
        xs = list(range(len(methods)))
        ys = []
        lows = []
        highs = []
        for method, _label in methods:
            mean, low, high = _metric_with_ci(run, method, metric)
            ys.append(mean)
            lows.append(max(0.0, mean - low))
            highs.append(max(0.0, high - mean))
        ax.bar(xs, ys, color="#4c78a8", alpha=0.85)
        ax.errorbar(xs, ys, yerr=[lows, highs], fmt="none", ecolor="#1f1f1f", capsize=3, linewidth=1)
        ax.set_title(f"{DATASET_LABELS[ds]} {REGIME_LABELS[regime]}")
        ax.set_xticks(xs)
        ax.set_xticklabels([label for _method, label in methods], rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0][0].set_ylabel(metric)
    axes[1][0].set_ylabel(metric)
    fig.tight_layout()
    _save(fig, output_dir / "cross_regime_query2doc_with_error_bars")


def _metric_with_ci(run: dict, method: str, metric: str) -> tuple[float, float, float]:
    mean = float(run.get("metrics", {}).get(method, {}).get(metric, 0.0))
    ci = run.get("metric_error_bars", {}).get(method, {}).get(metric)
    if isinstance(ci, dict):
        return mean, float(ci.get("low", mean)), float(ci.get("high", mean))
    return mean, mean, mean


def _parse_beta(method: str) -> float | None:
    match = BETA_RE.match(method)
    if not match:
        return None
    return float(match.group("beta").replace("p", "."))


def _stderr(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def _offsets(count: int, width: float) -> list[float]:
    center = (count - 1) / 2
    return [(idx - center) * width for idx in range(count)]


def _save(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - plotting should not block experiment postprocessing.
        print(f"Skipping PI figure generation after plotting error: {exc}", file=sys.stderr)
