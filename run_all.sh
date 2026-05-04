#!/bin/bash
set -euo pipefail

N=100
MAX_CORPUS=200
MODEL="openai/gpt-5-mini"
OUTPUT_DIR="outputs"
HF_CACHE_DIR="${OUTPUT_DIR}/hf_cache"
EMBEDDING_CACHE_DIR="${OUTPUT_DIR}/embeddings"

datasets=("nq" "scifact" "hotpotqa")
retrievers=("bm25" "dense")
cfs=("" "entity" "entity_and_value")

mkdir -p "$OUTPUT_DIR" "$HF_CACHE_DIR" "$EMBEDDING_CACHE_DIR"

for ds in "${datasets[@]}"; do
    for ret in "${retrievers[@]}"; do
        for cf in "${cfs[@]}"; do
            cf_flag=""
            out_prefix="${ds}_${N}"
            if [ "$cf" == "entity" ]; then
                cf_flag="--counterfactual entity"
                out_prefix="${ds}_${N}_cf"
            elif [ "$cf" == "entity_and_value" ]; then
                cf_flag="--counterfactual entity_and_value"
                out_prefix="${ds}_${N}_cf_ev"
            fi
            
            cache_file="${OUTPUT_DIR}/${out_prefix}_cache.json"
            records_file="${OUTPUT_DIR}/${out_prefix}_records.jsonl"
            
            if [ "$ret" == "bm25" ]; then
                run_file="${OUTPUT_DIR}/${out_prefix}_run.json"
                log_file="${OUTPUT_DIR}/${out_prefix}.log"
                ret_flags=""
            else
                run_file="${OUTPUT_DIR}/${out_prefix}_dense_run.json"
                log_file="${OUTPUT_DIR}/${out_prefix}_dense.log"
                embedding_cache="${EMBEDDING_CACHE_DIR}/${out_prefix}_bge"
                ret_flags="--embedding-cache ${embedding_cache} --local-files-only"
            fi
            
            echo "Running $ds $ret ${cf:-public} (N=$N, model=$MODEL)..."
            conda run -n rag python scripts/run_experiment.py \
                --dataset "$ds" \
                --max-queries "$N" \
                --max-corpus "$MAX_CORPUS" \
                --cache-dir "$HF_CACHE_DIR" \
                $cf_flag \
                --generator openrouter \
                --model "$MODEL" \
                --token-param none \
                --generation-cache "$cache_file" \
                --retriever "$ret" \
                $ret_flags \
                --output "$run_file" \
                --records-output "$records_file" \
                > "$log_file" 2>&1
        done
    done
done
