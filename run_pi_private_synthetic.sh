#!/usr/bin/env bash
set -euo pipefail

MAX_QUERIES="${MAX_QUERIES:-100}"
MAX_CORPUS="${MAX_CORPUS:-2000}"
GENERATION_WORKERS="${GENERATION_WORKERS:-30}"
OUT_DIR="${OUT_DIR:-outputs_pi/private_synthetic}"
DATASET_DIR="${DATASET_DIR:-${OUT_DIR}/dataset}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${OUT_DIR}/counterfactual_artifacts}"
MODEL="${MODEL:-openai/gpt-4o-mini}"

mkdir -p "$OUT_DIR"

conda run -n rag python scripts/build_private_synthetic_dataset.py \
  --output "$DATASET_DIR" \
  --num-records 120

for cf in none entity entity_and_value; do
  suffix=""
  if [ "$cf" = "entity" ]; then
    suffix="_cf"
  elif [ "$cf" = "entity_and_value" ]; then
    suffix="_cf_ev"
  fi

  run="${OUT_DIR}/private_synthetic_${MAX_QUERIES}${suffix}_bm25_run.json"
  records="${OUT_DIR}/private_synthetic_${MAX_QUERIES}${suffix}_bm25_records.jsonl"
  cache="${OUT_DIR}/private_synthetic_${MAX_QUERIES}${suffix}_cache.json"
  log="${OUT_DIR}/private_synthetic_${MAX_QUERIES}${suffix}_bm25.log"

  if [ -f "$run" ]; then
    echo "Skipping existing $run"
    continue
  fi

  cmd=(
    conda run -n rag python scripts/run_experiment.py
    --dataset "$DATASET_DIR"
    --max-queries "$MAX_QUERIES"
    --max-corpus "$MAX_CORPUS"
    --retriever bm25
    --generator openrouter
    --model "$MODEL"
    --token-param none
    --generation-cache "$cache"
    --generation-workers "$GENERATION_WORKERS"
    --method-profile main
    --include-fawe-beta-grid
    --fawe-betas 0.05,0.10,0.25,0.50,0.75,1.00
    --top-k 100
    --metric-ks 5,10,20,100
    --counterfactual-artifact-root "$ARTIFACT_ROOT"
    --output "$run"
    --records-output "$records"
  )

  if [ "$cf" != "none" ]; then
    cmd+=(--counterfactual "$cf")
  fi

  PYTHONUNBUFFERED=1 "${cmd[@]}" | tee "$log"
done
