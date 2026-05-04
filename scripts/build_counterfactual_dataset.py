from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from expected_answer_rag.counterfactual import (
    build_entity_counterfactual_dataset,
    export_counterfactual_artifacts,
    load_counterfactual_artifacts,
    resolve_counterfactual_artifact_dir,
)
from expected_answer_rag.datasets import export_local_dataset, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an entity-counterfactual local dataset export.")
    parser.add_argument("--dataset", default="toy")
    parser.add_argument("--max-corpus", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--query-metadata", default=None)
    parser.add_argument("--alias-style", choices=["natural", "coded"], default="natural")
    parser.add_argument("--include-values", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.output_dir and not args.artifact_root:
        raise SystemExit("Pass either --output-dir or --artifact-root.")
    dataset = load_dataset(
        args.dataset,
        max_corpus=args.max_corpus,
        max_queries=args.max_queries,
        query_metadata_path=args.query_metadata,
    )
    export_root = (
        Path(args.output_dir)
        if args.output_dir
        else resolve_counterfactual_artifact_dir(
            args.artifact_root,
            dataset,
            alias_style=args.alias_style,
            include_values=args.include_values,
            seed=args.seed,
        )
    )
    if export_root.exists() and not args.force_rebuild:
        result = load_counterfactual_artifacts(export_root)
        print(json.dumps(result.validation, indent=2, ensure_ascii=False))
        print(f"Reusing counterfactual dataset at {export_root}")
        return
    result = build_entity_counterfactual_dataset(
        dataset,
        alias_style=args.alias_style,
        include_values=args.include_values,
        seed=args.seed,
    )
    export_local_dataset(result.dataset, export_root)
    export_counterfactual_artifacts(result, export_root)
    print(json.dumps(result.validation, indent=2, ensure_ascii=False))
    print(f"Wrote counterfactual dataset to {export_root}")


if __name__ == "__main__":
    main()
