#!/bin/bash
# rm -f outputs/*_cf*.log outputs/*_cf_run.json outputs/*_cf_ev_run.json outputs/*_cf_cache.json outputs/*_cf_ev_cache.json outputs/*_cf_records.jsonl outputs/*dense_cf*.log outputs/*dense_cf_run.json

N=100
MAX_CORPUS=200

datasets=("nq" "scifact" "hotpotqa")
retrievers=("bm25" "dense")
cfs=("" "entity" "entity_and_value")

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
            
            cache_file="outputs/${out_prefix}_cache.json"
            
            if [ "$ret" == "bm25" ]; then
                run_file="outputs/${out_prefix}_run.json"
                log_file="outputs/${out_prefix}.log"
            else
                run_file="outputs/${out_prefix}_dense_run.json"
                log_file="outputs/${out_prefix}_dense.log"
            fi
            
            echo "Running $ds $ret $cf (N=$N)..."
            conda run -n rag python scripts/run_experiment.py --dataset $ds --max-queries $N --max-corpus $MAX_CORPUS --cache-dir outputs/hf_cache $cf_flag --generator openrouter --model openai/gpt-4o-mini --generation-cache $cache_file --retriever $ret --output $run_file > $log_file 2>&1
        done
    done
done
