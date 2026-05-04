#!/bin/bash
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not available on PATH" >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate rag

N="${N:-100}"
MAX_CORPUS="${MAX_CORPUS:-200}"
FULL_CORPUS="${FULL_CORPUS:-0}"
GENERATION_WORKERS="${GENERATION_WORKERS:-4}"
MODEL="${MODEL:-openai/gpt-5-mini}"
RUN_TAG="${RUN_TAG:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${OUTPUT_DIR}/hf_cache}"
EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR:-${OUTPUT_DIR}/embeddings}"

if [ "$FULL_CORPUS" = "1" ] && [ -z "$RUN_TAG" ]; then
    RUN_TAG="full"
fi

datasets=("nq" "scifact" "hotpotqa")
retrievers=("bm25" "dense")
cfs=("" "entity" "entity_and_value")

mkdir -p "$OUTPUT_DIR" "$HF_CACHE_DIR" "$EMBEDDING_CACHE_DIR"

for ds in "${datasets[@]}"; do
    for ret in "${retrievers[@]}"; do
        for cf in "${cfs[@]}"; do
            out_prefix="${ds}_${N}${RUN_TAG:+_${RUN_TAG}}"
            if [ "$cf" == "entity" ]; then
                out_prefix="${ds}_${N}${RUN_TAG:+_${RUN_TAG}}_cf"
            elif [ "$cf" == "entity_and_value" ]; then
                out_prefix="${ds}_${N}${RUN_TAG:+_${RUN_TAG}}_cf_ev"
            fi
            
            cache_file="${OUTPUT_DIR}/${out_prefix}_cache.json"
            records_file="${OUTPUT_DIR}/${out_prefix}_records.jsonl"
            
            if [ "$ret" == "bm25" ]; then
                run_file="${OUTPUT_DIR}/${out_prefix}_run.json"
                log_file="${OUTPUT_DIR}/${out_prefix}.log"
            else
                run_file="${OUTPUT_DIR}/${out_prefix}_dense_run.json"
                log_file="${OUTPUT_DIR}/${out_prefix}_dense.log"
                embedding_cache="${EMBEDDING_CACHE_DIR}/${out_prefix}_bge"
            fi
            
            echo "Running $ds $ret ${cf:-public} (N=$N, model=$MODEL, workers=$GENERATION_WORKERS, full_corpus=$FULL_CORPUS)..."
            cmd=(
                python scripts/run_experiment.py
                --dataset "$ds"
                --max-queries "$N"
                --cache-dir "$HF_CACHE_DIR"
                --generator openrouter
                --model "$MODEL"
                --token-param none
                --generation-cache "$cache_file"
                --generation-workers "$GENERATION_WORKERS"
                --retriever "$ret"
                --output "$run_file"
                --records-output "$records_file"
            )
            if [ "$FULL_CORPUS" != "1" ] && [ -n "$MAX_CORPUS" ]; then
                cmd+=(--max-corpus "$MAX_CORPUS")
            fi
            if [ "$cf" == "entity" ] || [ "$cf" == "entity_and_value" ]; then
                cmd+=(--counterfactual "$cf")
            fi
            if [ "$ret" == "dense" ]; then
                cmd+=(--embedding-cache "$embedding_cache" --local-files-only)
            fi
            PYTHONUNBUFFERED=1 "${cmd[@]}" | tee "$log_file"
        done
    done
done
