#!/usr/bin/env bash
set -euo pipefail

MAX_QUERIES="${MAX_QUERIES:-100}"
MAX_CORPUS="${MAX_CORPUS:-3000}"
GENERATION_WORKERS="${GENERATION_WORKERS:-30}"
OUT_DIR="${OUT_DIR:-outputs_pi/c${MAX_CORPUS}}"
HF_CACHE="${HF_CACHE:-outputs/hf_cache}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${OUT_DIR}/counterfactual_artifacts}"
MODEL="${MODEL:-openai/gpt-4o-mini}"
DATASETS="${DATASETS:-nq scifact hotpotqa}"

mkdir -p "$OUT_DIR"

for ds in $DATASETS; do
  for cf in none entity entity_and_value; do
    suffix=""
    if [ "$cf" = "entity" ]; then
      suffix="_cf"
    elif [ "$cf" = "entity_and_value" ]; then
      suffix="_cf_ev"
    fi

    cache="outputs/${ds}_100${suffix}_cache.json"
    run="${OUT_DIR}/${ds}_${MAX_QUERIES}_c${MAX_CORPUS}${suffix}_bm25_run.json"
    records="${OUT_DIR}/${ds}_${MAX_QUERIES}_c${MAX_CORPUS}${suffix}_bm25_records.jsonl"
    log="${OUT_DIR}/${ds}_${MAX_QUERIES}_c${MAX_CORPUS}${suffix}_bm25.log"

    if [ -f "$run" ]; then
      echo "Skipping existing $run"
      continue
    fi

    cmd=(
      conda run -n rag python scripts/run_experiment.py
      --dataset "$ds"
      --max-queries "$MAX_QUERIES"
      --max-corpus "$MAX_CORPUS"
      --cache-dir "$HF_CACHE"
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
      --audit-sample-size 10
      --counterfactual-artifact-root "$ARTIFACT_ROOT"
      --output "$run"
      --records-output "$records"
    )

    if [ "$cf" != "none" ]; then
      cmd+=(--counterfactual "$cf")
    fi

    PYTHONUNBUFFERED=1 "${cmd[@]}" | tee "$log"
  done
done
