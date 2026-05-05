from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from expected_answer_rag.metrics import per_query_metrics


DEFAULT_METHODS = [
    "raw_expected_answer_only",
    "hyde_doc_only",
    "query2doc_concat",
    "concat_query_raw_expected",
    "fawe_query2doc_beta0p25",
    "fawe_safe_adaptive_beta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate expansion-content audit labels and metric associations.")
    parser.add_argument("--public-records", required=True)
    parser.add_argument("--entity-records", required=True)
    parser.add_argument("--entity-value-records", required=True)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--metric", default="ndcg@10")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    records = {
        "public": _load_records(args.public_records),
        "entity": _load_records(args.entity_records),
        "entity_value": _load_records(args.entity_value_records),
    }
    query_ids = sorted(set.intersection(*(set(rows) for rows in records.values())))
    summary = {}
    for method in methods:
        rows = []
        for qid in query_ids:
            public = records["public"][qid]
            entity = records["entity"][qid]
            entity_value = records["entity_value"][qid]
            label = _label(public, method)
            public_score = _score(public, method, args.metric)
            public_query = _score(public, "query_only", args.metric)
            entity_score = _score(entity, method, args.metric)
            entity_query = _score(entity, "query_only", args.metric)
            ev_score = _score(entity_value, method, args.metric)
            ev_query = _score(entity_value, "query_only", args.metric)
            rows.append(
                {
                    "query_id": qid,
                    "answer_bearing": _answer_bearing(label),
                    "has_exact_answer": bool(label.get("has_exact_answer_leakage")),
                    "has_alias_answer": bool(label.get("has_alias_answer_leakage")),
                    "has_candidate_injection": bool(label.get("has_candidate_injection")),
                    "has_unsupported_injection": bool(label.get("has_unsupported_injection")),
                    "public_gain": public_score - public_query,
                    "entity_excess_drop": (public_score - entity_score) - (public_query - entity_query),
                    "entity_value_excess_drop": (public_score - ev_score) - (public_query - ev_query),
                }
            )
        summary[method] = _summarize_rows(rows)
    result = {
        "metric": args.metric,
        "num_queries": len(query_ids),
        "methods": methods,
        "summary": summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.markdown_output:
        Path(args.markdown_output).write_text(_to_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {output}")


def _load_records(path: str) -> dict[str, dict]:
    rows = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["query_id"])] = row
    return rows


def _label(record: dict, method: str) -> dict:
    return dict(record.get("leakage_labels", {}).get(method, {}))


def _answer_bearing(label: dict) -> bool:
    return bool(
        label.get("has_exact_answer_leakage")
        or label.get("has_alias_answer_leakage")
        or label.get("has_candidate_injection")
    )


def _score(record: dict, method: str, metric: str) -> float:
    ranking = record.get("rankings", {}).get(method)
    if ranking is None:
        return 0.0
    return per_query_metrics(ranking, record.get("qrels", {}), ks=(5, 10, 20, 100)).get(metric, 0.0)


def _summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {}
    groups = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups["answer_bearing" if row["answer_bearing"] else "non_answer_bearing"].append(row)
    return {
        name: {
            "count": len(values),
            "fraction": len(values) / len(rows),
            "mean_public_gain": _mean(values, "public_gain"),
            "mean_entity_excess_drop": _mean(values, "entity_excess_drop"),
            "mean_entity_value_excess_drop": _mean(values, "entity_value_excess_drop"),
            "exact_answer_rate": _rate(values, "has_exact_answer"),
            "alias_answer_rate": _rate(values, "has_alias_answer"),
            "candidate_injection_rate": _rate(values, "has_candidate_injection"),
            "unsupported_injection_rate": _rate(values, "has_unsupported_injection"),
        }
        for name, values in groups.items()
    }


def _mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def _rate(rows: list[dict], key: str) -> float:
    return sum(1 for row in rows if row[key]) / len(rows) if rows else 0.0


def _to_markdown(result: dict) -> str:
    lines = ["# Expansion Content Audit", "", f"Metric: `{result['metric']}`", ""]
    for method, summary in result["summary"].items():
        lines.extend([f"## `{method}`", "", "| Group | Count | Answer-bearing frac | Public gain | Entity excess drop | Entity+Value excess drop |", "|---|---:|---:|---:|---:|---:|"])
        for group, row in summary.items():
            lines.append(
                f"| {group} | {row['count']} | {row['fraction']:.3f} | "
                f"{row['mean_public_gain']:.4f} | {row['mean_entity_excess_drop']:.4f} | "
                f"{row['mean_entity_value_excess_drop']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
