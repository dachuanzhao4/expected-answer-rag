from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-2 NQ Query2Doc BM25 + dense experiments.")
    parser.add_argument(
        "--local-cache-root",
        default="D:/hf_cache",
        help="Local cache root for HuggingFace, Transformers, sentence-transformers, and temp files.",
    )
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument("--max-corpus", type=int, default=200000)
    parser.add_argument("--model", default="openai/gpt-5-mini")
    parser.add_argument("--token-param", default="none", choices=["auto", "max_tokens", "max_completion_tokens", "none"])
    parser.add_argument("--prompt-style", default="query2doc_fewshot", choices=["query2doc_fewshot", "query2doc_zero_shot"])
    parser.add_argument("--generation-cache", default="outputs/nq_stage2_formal_query2doc_mask_cache.json")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--embedding-device", default=None, help="Dense embedding device, e.g. cuda or cpu.")
    parser.add_argument("--embedding-cache", default="outputs/embeddings/nq_stage2_200k_bge_base")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-chunk-size", type=int, default=512)
    parser.add_argument("--api-workers", type=int, default=1, help="Concurrent API generation workers for BM25 runs.")
    parser.add_argument("--bm25-output", default="outputs/nq_stage2_bm25_formal_query2doc_mask_run.json")
    parser.add_argument("--bm25-records", default="outputs/nq_stage2_bm25_formal_query2doc_mask_records.jsonl")
    parser.add_argument("--dense-output", default="outputs/nq_stage2_dense_bge_base_formal_query2doc_mask_run.json")
    parser.add_argument("--dense-records", default="outputs/nq_stage2_dense_bge_base_formal_query2doc_mask_records.jsonl")
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
    parser.add_argument("--skip-bm25", action="store_true", help="Run only dense, assuming generation cache already exists.")
    parser.add_argument("--skip-dense", action="store_true", help="Run only BM25 and generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_local_cache(args.local_cache_root)
    if not args.skip_bm25 and not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("Set OPENROUTER_API_KEY before running BM25 generation.")

    if not args.skip_bm25:
        run(_bm25_command(args))
    if not args.skip_dense:
        run(_dense_command(args))


def run(command: list[str]) -> None:
    print("\n$", " ".join(_quote(part) for part in command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def configure_local_cache(cache_root: str | None) -> None:
    if not cache_root:
        return
    root = Path(cache_root)
    paths = {
        "HF_HOME": root,
        "HF_HUB_CACHE": root / "hub",
        "HF_DATASETS_CACHE": root / "datasets",
        "TRANSFORMERS_CACHE": root / "transformers",
        "SENTENCE_TRANSFORMERS_HOME": root / "sentence_transformers",
        "TMP": root / "tmp",
        "TEMP": root / "tmp",
    }
    for value in paths.values():
        value.mkdir(parents=True, exist_ok=True)
    for key, value in paths.items():
        os.environ[key] = str(value)


def _bm25_command(args: argparse.Namespace) -> list[str]:
    command = [
        PYTHON,
        "scripts/run_experiment.py",
        "--dataset",
        "nq",
        "--max-queries",
        str(args.max_queries),
        "--max-corpus",
        str(args.max_corpus),
        "--local-cache-root",
        args.local_cache_root,
        "--retriever",
        "bm25",
        "--generator",
        "openrouter",
        "--model",
        args.model,
        "--token-param",
        args.token_param,
        "--prompt-style",
        args.prompt_style,
        "--generation-cache",
        args.generation_cache,
        "--output",
        args.bm25_output,
        "--records-output",
        args.bm25_records,
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
    return command


def _dense_command(args: argparse.Namespace) -> list[str]:
    command = [
        PYTHON,
        "scripts/run_experiment.py",
        "--dataset",
        "nq",
        "--max-queries",
        str(args.max_queries),
        "--max-corpus",
        str(args.max_corpus),
        "--local-cache-root",
        args.local_cache_root,
        "--retriever",
        "dense",
        "--embedding-model",
        args.embedding_model,
        "--embedding-batch-size",
        str(args.embedding_batch_size),
        "--embedding-chunk-size",
        str(args.embedding_chunk_size),
        "--embedding-cache",
        args.embedding_cache,
        "--generator",
        "openrouter",
        "--model",
        args.model,
        "--token-param",
        args.token_param,
        "--prompt-style",
        args.prompt_style,
        "--generation-cache",
        args.generation_cache,
        "--cache-only",
        "--cache-namespace",
        f"openrouter:{args.model}:temp=None:prompt={args.prompt_style}",
        "--output",
        args.dense_output,
        "--records-output",
        args.dense_records,
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
    if args.embedding_device:
        command.extend(["--embedding-device", args.embedding_device])
    return command


def _quote(value: str) -> str:
    if any(char.isspace() for char in value):
        return f'"{value}"'
    return value


if __name__ == "__main__":
    main()
