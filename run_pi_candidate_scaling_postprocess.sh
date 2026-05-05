#!/usr/bin/env bash
set -euo pipefail

# Postprocesses completed candidate-pool scaling runs and emits draft-ready
# Markdown/CSV/LaTeX summaries. By default it summarizes 1k/3k/5k, matching the
# current low-compute placeholder plan.

CANDIDATE_CORPUS_SIZES="${CANDIDATE_CORPUS_SIZES:-1000 3000 5000}"
MAX_QUERIES="${MAX_QUERIES:-100}"
DATASETS="${DATASETS:-nq scifact hotpotqa}"
RUN_POSTPROCESS_PER_SIZE="${RUN_POSTPROCESS_PER_SIZE:-1}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs_pi}"
SUMMARY_DIR="${SUMMARY_DIR:-${OUTPUTS_ROOT}/scaling_summary}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ "$RUN_POSTPROCESS_PER_SIZE" = "1" ]; then
  for size in $CANDIDATE_CORPUS_SIZES; do
    out_dir="${OUTPUTS_ROOT}/c${size}"
    if [ ! -d "$out_dir" ]; then
      echo "Skipping postprocess for missing $out_dir"
      continue
    fi

    echo "=== Postprocessing candidate-pool size ${size} ==="
    MAX_CORPUS="$size" \
    MAX_QUERIES="$MAX_QUERIES" \
    DATASETS="$DATASETS" \
    OUT_DIR="$out_dir" \
      bash run_pi_postprocess.sh
  done
fi

sizes_csv="${CANDIDATE_CORPUS_SIZES// /,}"
mkdir -p "$SUMMARY_DIR"

conda run -n rag python scripts/summarize_candidate_scaling.py \
  --outputs-root "$OUTPUTS_ROOT" \
  --sizes "$sizes_csv" \
  --max-queries "$MAX_QUERIES" \
  --datasets "${DATASETS// /,}" \
  --output-dir "$SUMMARY_DIR"
