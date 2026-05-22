from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and run stage-2 opaque/plausible renamed NQ experiments.")
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument(
        "--target-queries",
        type=int,
        default=None,
        help="Keep at most this many safe renamed queries after building from --max-queries source queries.",
    )
    parser.add_argument("--max-corpus", type=int, default=200000)
    parser.add_argument("--mode", choices=["opaque", "plausible", "both"], default="both")
    parser.add_argument("--output-root", default="outputs/renamed_nq_stage2_token_v5_full")
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Prefix for cache/output files. Default is derived from --output-root.",
    )
    parser.add_argument(
        "--cache-tag",
        default=None,
        help="Generation cache prefix. Default follows --run-tag; set this to reuse an older cache.",
    )
    parser.add_argument(
        "--embedding-cache-tag",
        default=None,
        help="Dense embedding cache prefix. Default follows --run-tag; set this to reuse older embeddings.",
    )
    parser.add_argument("--local-cache-root", default="D:/hf_cache")
    parser.add_argument("--model", default="openai/gpt-5-mini")
    parser.add_argument("--token-param", default="none", choices=["auto", "max_tokens", "max_completion_tokens", "none"])
    parser.add_argument("--prompt-style", default="query2doc_fewshot", choices=["query2doc_fewshot", "query2doc_zero_shot"])
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--embedding-device", default=None, help="Dense embedding device, e.g. cuda or cpu.")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-chunk-size", type=int, default=512)
    parser.add_argument("--api-workers", type=int, default=1, help="Concurrent API generation workers for BM25 runs.")
    parser.add_argument("--rename-workers", type=int, default=0, help="Parallel workers for renamed corpus construction.")
    parser.add_argument("--query-rename-policy", default="safe_aligned", choices=["safe_aligned", "all"])
    parser.add_argument("--query-ngram-anchors", default="off", choices=["off", "all"])
    parser.add_argument("--replacement-granularity", default="token", choices=["token", "span"])
    parser.add_argument("--replacement-token-policy", default="preserve", choices=["single", "preserve"])
    parser.add_argument(
        "--min-query-replacements",
        type=int,
        default=0,
        help=(
            "Minimum replacements required to keep a query. Default 0 keeps the full query set; "
            "use 1 only for a changed-query diagnostic subset."
        ),
    )
    parser.add_argument("--require-named-query-replacement", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-bm25", action="store_true")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--cache-only", action="store_true", help="Use existing Query2Doc generation cache for BM25 too.")
    parser.add_argument("--include-answer-blanked", action="store_true")
    parser.add_argument("--answer-blanked-fusion-weights", default="1.0,0.8,0.6")
    parser.add_argument("--include-lf-er", action="store_true")
    parser.add_argument("--lf-er-fusion-weights", default="4.0,0.3,0.8,0.1,0.1,0.6")
    parser.add_argument("--lf-er-bm25-agreement-weights", default="4.0,0.9,0.7,0.6,0.0")
    parser.add_argument("--lf-er-dense-agreement-weights", default="4.0,1.0,0.8,0.6,0.4")
    parser.add_argument("--lf-er-inverse-query-weights", default="4.0,0.8")
    parser.add_argument("--lf-er-anchor-gated-fusion-weights", default="4.0,1.4,0.8,0.5,0.4")
    parser.add_argument("--lf-er-short-fusion-weights", default="6.0,1.2,1.0,0.8")
    parser.add_argument("--lf-er-dense-safe-fusion-weights", default="6.0,1.2,0.8,0.4")
    parser.add_argument("--include-llm-reformat", action="store_true")
    parser.add_argument("--llm-reformat-version", default="v2", choices=["v1", "v2"])
    parser.add_argument("--llm-reformat-fusion-weights", default="6.0,1.0,1.0,0.6,0.6")
    parser.add_argument("--allow-entity-only-query", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = ["opaque", "plausible"] if args.mode == "both" else [args.mode]
    if not args.skip_build:
        run(build_command(args))
    if not args.skip_bm25 and not args.cache_only and not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("Set OPENROUTER_API_KEY before running renamed BM25 generation.")
    for mode in modes:
        if not args.skip_bm25:
            run(experiment_command(args, mode=mode, retriever="bm25"))
        if not args.skip_dense:
            run(experiment_command(args, mode=mode, retriever="dense"))


def run(command: list[str]) -> None:
    print("\n$", " ".join(quote(part) for part in command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        PYTHON,
        "scripts/build_renamed_dataset.py",
        "--dataset",
        "nq",
        "--max-queries",
        str(args.max_queries),
        "--max-corpus",
        str(args.max_corpus),
        "--mode",
        args.mode,
        "--output-root",
        args.output_root,
        "--local-cache-root",
        args.local_cache_root,
        "--rename-workers",
        str(args.rename_workers),
        "--query-rename-policy",
        args.query_rename_policy,
        "--query-ngram-anchors",
        args.query_ngram_anchors,
        "--replacement-granularity",
        args.replacement_granularity,
        "--replacement-token-policy",
        args.replacement_token_policy,
        "--min-query-replacements",
        str(args.min_query_replacements),
    ]
    if args.target_queries is not None:
        command.extend(["--target-queries", str(args.target_queries)])
    if args.require_named_query_replacement:
        command.append("--require-named-query-replacement")
    if args.allow_entity_only_query:
        command.append("--allow-entity-only-query")
    return command


def experiment_command(args: argparse.Namespace, mode: str, retriever: str) -> list[str]:
    dataset = f"local:{args.output_root}/{mode}"
    tag = output_tag(args)
    cache_tag = args.cache_tag or tag
    embedding_cache_tag = args.embedding_cache_tag or tag
    cache = f"outputs/{cache_tag}_{mode}_query2doc_mask_cache.json"
    output_prefix = f"outputs/{tag}_{mode}_{retriever}_formal_query2doc_mask"
    query_limit = args.target_queries or args.max_queries
    command = [
        PYTHON,
        "scripts/run_experiment.py",
        "--dataset",
        dataset,
        "--max-queries",
        str(query_limit),
        "--max-corpus",
        str(args.max_corpus),
        "--retriever",
        retriever,
        "--generator",
        "openrouter",
        "--model",
        args.model,
        "--token-param",
        args.token_param,
        "--prompt-style",
        args.prompt_style,
        "--generation-cache",
        cache,
        "--local-cache-root",
        args.local_cache_root,
        "--output",
        f"{output_prefix}_run.json",
        "--records-output",
        f"{output_prefix}_records.jsonl",
        "--api-workers",
        str(args.api_workers),
    ]
    if args.include_answer_blanked:
        command.extend(
            [
                "--include-answer-blanked",
                "--answer-blanked-fusion-weights",
                args.answer_blanked_fusion_weights,
            ]
        )
    if args.include_lf_er:
        command.extend(
            [
                "--include-lf-er",
                "--lf-er-fusion-weights",
                args.lf_er_fusion_weights,
                "--lf-er-bm25-agreement-weights",
                args.lf_er_bm25_agreement_weights,
                "--lf-er-dense-agreement-weights",
                args.lf_er_dense_agreement_weights,
                "--lf-er-inverse-query-weights",
                args.lf_er_inverse_query_weights,
                "--lf-er-anchor-gated-fusion-weights",
                args.lf_er_anchor_gated_fusion_weights,
                "--lf-er-short-fusion-weights",
                args.lf_er_short_fusion_weights,
                "--lf-er-dense-safe-fusion-weights",
                args.lf_er_dense_safe_fusion_weights,
            ]
        )
    if args.include_llm_reformat:
        command.extend(
            [
                "--include-llm-reformat",
                "--llm-reformat-version",
                args.llm_reformat_version,
                "--llm-reformat-fusion-weights",
                args.llm_reformat_fusion_weights,
            ]
        )
    if args.cache_only and retriever == "bm25":
        command.extend(
            [
                "--cache-only",
                "--cache-namespace",
                f"openrouter:{args.model}:temp=None:prompt={args.prompt_style}",
            ]
        )
    if retriever == "dense":
        command.extend(
            [
                "--embedding-model",
                args.embedding_model,
                "--embedding-batch-size",
                str(args.embedding_batch_size),
                "--embedding-chunk-size",
                str(args.embedding_chunk_size),
                "--embedding-cache",
                f"outputs/embeddings/{embedding_cache_tag}_{mode}_200k_bge_base",
                "--cache-only",
                "--cache-namespace",
                f"openrouter:{args.model}:temp=None:prompt={args.prompt_style}",
            ]
        )
        if args.embedding_device:
            command.extend(["--embedding-device", args.embedding_device])
    return command


def quote(value: str) -> str:
    if any(char.isspace() for char in value):
        return f'"{value}"'
    return value


def output_tag(args: argparse.Namespace) -> str:
    if args.run_tag:
        return args.run_tag
    tag = Path(args.output_root).name.replace("-", "_")
    if args.include_llm_reformat and f"llm_reformat_{args.llm_reformat_version}" not in tag:
        tag = f"{tag}_llm_reformat_{args.llm_reformat_version}"
    return tag


if __name__ == "__main__":
    main()
