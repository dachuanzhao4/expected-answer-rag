#!/bin/bash
conda run -n rag python scripts/run_experiment.py --dataset nq --max-queries 10 --max-corpus 200 --cache-dir outputs/hf_cache --counterfactual entity_and_value --generator openrouter --model openai/gpt-4o-mini --generation-cache outputs/nq_10_cf_ev_cache.json --retriever bm25 --output outputs/nq_10_cf_ev_run.json > outputs/nq_10_cf_ev.log 2>&1

conda run -n rag python scripts/run_experiment.py --dataset scifact --max-queries 10 --max-corpus 200 --cache-dir outputs/hf_cache --counterfactual entity_and_value --generator openrouter --model openai/gpt-4o-mini --generation-cache outputs/scifact_10_cf_ev_cache.json --retriever bm25 --output outputs/scifact_10_cf_ev_run.json > outputs/scifact_10_cf_ev.log 2>&1

conda run -n rag python scripts/run_experiment.py --dataset hotpotqa --max-queries 10 --max-corpus 200 --cache-dir outputs/hf_cache --counterfactual entity_and_value --generator openrouter --model openai/gpt-4o-mini --generation-cache outputs/hotpotqa_10_cf_ev_cache.json --retriever bm25 --output outputs/hotpotqa_10_cf_ev_run.json > outputs/hotpotqa_10_cf_ev.log 2>&1
