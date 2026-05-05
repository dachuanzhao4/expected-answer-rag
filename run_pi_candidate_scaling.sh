#!/usr/bin/env bash
set -euo pipefail

# Runs the BM25 candidate-pool scaling package used to fill the draft's
# 1k/3k/5k/10k/... placeholder table. Existing 3k runs are expected in
# outputs_pi/c3000, so the default sizes skip 3000. If 3000 is included
# explicitly, run_pi_larger_corpus_3000.sh will read/write outputs_pi/c3000
# and skip completed run files.

CANDIDATE_CORPUS_SIZES="${CANDIDATE_CORPUS_SIZES:-1000 5000}"
MAX_QUERIES="${MAX_QUERIES:-100}"
GENERATION_WORKERS="${GENERATION_WORKERS:-30}"
DATASETS="${DATASETS:-nq scifact hotpotqa}"
MODEL="${MODEL:-openai/gpt-4o-mini}"
HF_CACHE="${HF_CACHE:-outputs/hf_cache}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

for size in $CANDIDATE_CORPUS_SIZES; do
  echo "=== Candidate-pool scaling: max_corpus=${size}, max_queries=${MAX_QUERIES} ==="
  echo "Run directory: outputs_pi/c${size}"
  MAX_CORPUS="$size" \
  MAX_QUERIES="$MAX_QUERIES" \
  GENERATION_WORKERS="$GENERATION_WORKERS" \
  DATASETS="$DATASETS" \
  MODEL="$MODEL" \
  HF_CACHE="$HF_CACHE" \
  OUT_DIR="outputs_pi/c${size}" \
  ARTIFACT_ROOT="outputs_pi/c${size}/counterfactual_artifacts" \
    bash run_pi_larger_corpus_3000.sh
done
