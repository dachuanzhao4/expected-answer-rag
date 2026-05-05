#!/usr/bin/env bash
set -euo pipefail

MAX_QUERIES="${MAX_QUERIES:-100}"
MAX_CORPUS="${MAX_CORPUS:-3000}"
OUT_DIR="${OUT_DIR:-outputs_pi/c${MAX_CORPUS}}"
HF_CACHE="${HF_CACHE:-outputs/hf_cache}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${OUT_DIR}/counterfactual_artifacts}"
BETA="${BETA:-0.25}"
DATASETS="${DATASETS:-nq scifact hotpotqa}"

mkdir -p "$OUT_DIR"

for ds in $DATASETS; do
  public_records="${OUT_DIR}/${ds}_${MAX_QUERIES}_c${MAX_CORPUS}_bm25_records.jsonl"
  if [ ! -f "$public_records" ]; then
    echo "Missing public records: $public_records"
    echo "Run run_pi_larger_corpus_3000.sh first."
    exit 1
  fi

  for cf in entity entity_and_value; do
    suffix="_cf"
    if [ "$cf" = "entity_and_value" ]; then
      suffix="_cf_ev"
    fi

    cf_records="${OUT_DIR}/${ds}_${MAX_QUERIES}_c${MAX_CORPUS}${suffix}_bm25_records.jsonl"
    if [ ! -f "$cf_records" ]; then
      echo "Missing counterfactual records: $cf_records"
      echo "Run run_pi_larger_corpus_3000.sh first."
      exit 1
    fi

    artifact_dir="$(find "$ARTIFACT_ROOT" -mindepth 1 -maxdepth 1 -type d -name "${ds}__${cf}__natural__seed13__c${MAX_CORPUS}__q*__v1__*" | sort | head -n 1)"
    if [ -z "$artifact_dir" ]; then
      echo "Missing counterfactual artifact for dataset=$ds regime=$cf under $ARTIFACT_ROOT"
      echo "Run run_pi_larger_corpus_3000.sh first."
      exit 1
    fi

    for expansion in expected query2doc hyde; do
      output="${OUT_DIR}/${ds}_${MAX_QUERIES}_c${MAX_CORPUS}${suffix}_cross_${expansion}.json"
      log="${OUT_DIR}/${ds}_${MAX_QUERIES}_c${MAX_CORPUS}${suffix}_cross_${expansion}.log"
      if [ -f "$output" ]; then
        echo "Skipping existing $output"
        continue
      fi

      PYTHONUNBUFFERED=1 conda run -n rag python scripts/run_cross_regime_experiment.py \
        --public-dataset "$ds" \
        --public-max-queries "$MAX_QUERIES" \
        --public-max-corpus "$MAX_CORPUS" \
        --cache-dir "$HF_CACHE" \
        --cf-dataset "$artifact_dir" \
        --public-records "$public_records" \
        --cf-records "$cf_records" \
        --retriever bm25 \
        --expansion "$expansion" \
        --beta "$BETA" \
        --top-k 100 \
        --output "$output" | tee "$log"
    done
  done
done
