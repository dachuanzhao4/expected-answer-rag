from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from expected_answer_rag.metrics import per_query_metrics
from expected_answer_rag.statistics import paired_bootstrap_ci


METHOD_RE = re.compile(r"^fawe_(?P<family>.+)_beta(?P<beta>[0-9]+p[0-9]+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Held-out FAWE beta selection from beta-grid records.")
    parser.add_argument("--public-records", required=True)
    parser.add_argument("--entity-records", required=True)
    parser.add_argument("--entity-value-records", required=True)
    parser.add_argument("--family", default="query2doc")
    parser.add_argument("--dev-frac", type=float, default=0.3)
    parser.add_argument("--objective", choices=["public", "average", "robust"], default="public")
    parser.add_argument("--lambda-excess-drop", type=float, default=1.0)
    parser.add_argument("--metric", default="ndcg@10")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    regime_records = {
        "public": _load_records(args.public_records),
        "entity": _load_records(args.entity_records),
        "entity_value": _load_records(args.entity_value_records),
    }
    query_ids = sorted(set.intersection(*(set(records) for records in regime_records.values())))
    dev_ids, test_ids = _split_ids(query_ids, args.dev_frac)
    methods = _beta_methods(regime_records["public"], args.family)
    if not methods:
        raise SystemExit(f"No FAWE beta methods found for family={args.family}")
    scores = {
        method: {
            regime: {
                "dev": _mean_metric(records, method, dev_ids, args.metric),
                "test": _mean_metric(records, method, test_ids, args.metric),
            }
            for regime, records in regime_records.items()
        }
        for method in methods
    }
    objective_scores = {
        method: _objective(method, scores, args.objective, args.lambda_excess_drop, split="dev")
        for method in methods
    }
    selected_method = max(methods, key=lambda method: (objective_scores[method], -_method_beta(method)))
    result = {
        "family": args.family,
        "metric": args.metric,
        "objective": args.objective,
        "lambda_excess_drop": args.lambda_excess_drop,
        "dev_frac": args.dev_frac,
        "num_queries": len(query_ids),
        "num_dev": len(dev_ids),
        "num_test": len(test_ids),
        "selected_method": selected_method,
        "selected_beta": _method_beta(selected_method),
        "dev_objective_scores": objective_scores,
        "scores": scores,
        "test_error_bars": {
            regime: _metric_error_bars(records, selected_method, test_ids, args.metric)
            for regime, records in regime_records.items()
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {output}")


def _load_records(path: str) -> dict[str, dict]:
    rows = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                rows[str(record["query_id"])] = record
    return rows


def _split_ids(query_ids: list[str], dev_frac: float) -> tuple[list[str], list[str]]:
    dev = []
    test = []
    for qid in query_ids:
        bucket = int(hashlib.sha256(qid.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        (dev if bucket < dev_frac else test).append(qid)
    return dev, test


def _beta_methods(records: dict[str, dict], family: str) -> list[str]:
    if not records:
        return []
    methods = set()
    for record in records.values():
        for method in record.get("rankings", {}):
            match = METHOD_RE.match(method)
            if match and match.group("family") == family:
                methods.add(method)
    return sorted(methods, key=_method_beta)


def _method_beta(method: str) -> float:
    match = METHOD_RE.match(method)
    if not match:
        return 0.0
    return float(match.group("beta").replace("p", "."))


def _mean_metric(records: dict[str, dict], method: str, query_ids: list[str], metric: str) -> float:
    values = _metric_values(records, method, query_ids, metric)
    return sum(values) / len(values) if values else 0.0


def _metric_values(records: dict[str, dict], method: str, query_ids: list[str], metric: str) -> list[float]:
    values = []
    for qid in query_ids:
        record = records.get(qid)
        if not record:
            continue
        ranking = record.get("rankings", {}).get(method)
        if ranking is None:
            continue
        values.append(per_query_metrics(ranking, record.get("qrels", {}), ks=(5, 10, 20, 100)).get(metric, 0.0))
    return values


def _metric_error_bars(records: dict[str, dict], method: str, query_ids: list[str], metric: str) -> dict[str, float]:
    return paired_bootstrap_ci(_metric_values(records, method, query_ids, metric), num_samples=500)


def _objective(method: str, scores: dict, objective: str, lam: float, split: str) -> float:
    public = scores[method]["public"][split]
    entity = scores[method]["entity"][split]
    entity_value = scores[method]["entity_value"][split]
    if objective == "public":
        return public
    if objective == "average":
        return (public + entity + entity_value) / 3.0
    public_drop = ((public - entity) + (public - entity_value)) / 2.0
    return public - lam * public_drop


if __name__ == "__main__":
    main()
