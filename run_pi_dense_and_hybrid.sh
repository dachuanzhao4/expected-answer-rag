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

mkdir -p "$OUT_DIR/embeddings"

run_one() {
  local ds="$1"
  local cf="$2"
  local retriever="$3"
  local tag="$4"
  local embedding_model="$5"
  local query_prefix="$6"

  local suffix=""
  if [ "$cf" = "entity" ]; then
    suffix="_cf"
  elif [ "$cf" = "entity_and_value" ]; then
    suffix="_cf_ev"
  fi

  local cache="outputs/${ds}_100${suffix}_cache.json"
  local run="${OUT_DIR}/${ds}_${MAX_QUERIES}_c${MAX_CORPUS}${suffix}_${tag}_run.json"
  local records="${OUT_DIR}/${ds}_${MAX_QUERIES}_c${MAX_CORPUS}${suffix}_${tag}_records.jsonl"
  local log="${OUT_DIR}/${ds}_${MAX_QUERIES}_c${MAX_CORPUS}${suffix}_${tag}.log"
  local embedding_cache="${OUT_DIR}/embeddings/${ds}_${MAX_QUERIES}_c${MAX_CORPUS}${suffix}_${tag}"

  if [ -f "$run" ]; then
    echo "Skipping existing $run"
    return
  fi

  local cmd=(
    conda run -n rag python scripts/run_experiment.py
    --dataset "$ds"
    --max-queries "$MAX_QUERIES"
    --max-corpus "$MAX_CORPUS"
    --cache-dir "$HF_CACHE"
    --retriever "$retriever"
    --embedding-model "$embedding_model"
    --embedding-cache "$embedding_cache"
    --query-prefix "$query_prefix"
    --generator openrouter
    --model "$MODEL"
    --token-param none
    --generation-cache "$cache"
    --generation-workers "$GENERATION_WORKERS"
    --cache-only
    --method-profile main
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
}

for ds in $DATASETS; do
  for cf in none entity entity_and_value; do
    run_one "$ds" "$cf" dense bge BAAI/bge-base-en-v1.5 "Represent this sentence for searching relevant passages: "
    run_one "$ds" "$cf" dense e5 intfloat/e5-base-v2 "query: "
    run_one "$ds" "$cf" dense contriever facebook/contriever-msmarco ""
    run_one "$ds" "$cf" hybrid hybrid_e5 intfloat/e5-base-v2 "query: "
  done
done
