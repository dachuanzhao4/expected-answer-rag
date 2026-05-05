#!/bin/bash
set -euo pipefail

cd /Users/weiyueli/Desktop/rag/expected-answer-rag

for ds in nq scifact hotpotqa; do
  for cf in none entity entity_and_value; do
    suffix=""
    if [ "$cf" = "entity" ]; then
      suffix="_cf"
    elif [ "$cf" = "entity_and_value" ]; then
      suffix="_cf_ev"
    fi

    cache="outputs/${ds}_100${suffix}_cache.json"
    run="outputs/${ds}_100${suffix}_bm25_rm3_fawe_run.json"
    records="outputs/${ds}_100${suffix}_bm25_rm3_fawe_records.jsonl"
    log="outputs/${ds}_100${suffix}_bm25_rm3_fawe.log"

    if [ -f "$run" ]; then
      echo "Skipping $run"
      continue
    fi

    cmd=(
      conda run -n rag python scripts/run_experiment.py
      --dataset "$ds"
      --max-queries 100
      --max-corpus 2000
      --cache-dir outputs/hf_cache
      --retriever bm25
      --generator openrouter
      --model openai/gpt-4o-mini
      --token-param none
      --generation-cache "$cache"
      --generation-workers 30
      --method-profile main
      --include-rm3-baseline
      --include-fawe-controls
      --include-fawe-beta-grid
      --fawe-betas 0.05,0.10,0.25,0.50,0.75,1.00
      --audit-sample-size 10
      --output "$run"
      --records-output "$records"
    )

    if [ "$cf" != "none" ]; then
      cmd+=(--counterfactual "$cf" --counterfactual-artifact-root outputs/counterfactual_artifacts)
    fi

    PYTHONUNBUFFERED=1 "${cmd[@]}" | tee "$log"
  done
done
