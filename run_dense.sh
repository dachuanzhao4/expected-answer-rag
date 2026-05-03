#!/bin/bash
conda run -n rag python scripts/run_experiment.py --dataset nq --max-queries 10 --max-corpus 200 --cache-dir outputs/hf_cache --generator openrouter --model openai/gpt-4o-mini --generation-cache outputs/nq_10_cache.json --retriever dense --embedding-model BAAI/bge-base-en-v1.5 --output outputs/nq_10_dense_run.json > outputs/nq_10_dense.log 2>&1

conda run -n rag python scripts/run_experiment.py --dataset nq --max-queries 10 --max-corpus 200 --cache-dir outputs/hf_cache --counterfactual entity --generator openrouter --model openai/gpt-4o-mini --generation-cache outputs/nq_10_cf_cache.json --retriever dense --embedding-model BAAI/bge-base-en-v1.5 --output outputs/nq_10_dense_cf_run.json > outputs/nq_10_dense_cf.log 2>&1

conda run -n rag python scripts/run_experiment.py --dataset scifact --max-queries 10 --max-corpus 200 --cache-dir outputs/hf_cache --generator openrouter --model openai/gpt-4o-mini --generation-cache outputs/scifact_10_cache.json --retriever dense --embedding-model BAAI/bge-base-en-v1.5 --output outputs/scifact_10_dense_run.json > outputs/scifact_10_dense.log 2>&1

conda run -n rag python scripts/run_experiment.py --dataset scifact --max-queries 10 --max-corpus 200 --cache-dir outputs/hf_cache --counterfactual entity --generator openrouter --model openai/gpt-4o-mini --generation-cache outputs/scifact_10_cf_cache.json --retriever dense --embedding-model BAAI/bge-base-en-v1.5 --output outputs/scifact_10_dense_cf_run.json > outputs/scifact_10_dense_cf.log 2>&1

conda run -n rag python scripts/run_experiment.py --dataset hotpotqa --max-queries 10 --max-corpus 200 --cache-dir outputs/hf_cache --generator openrouter --model openai/gpt-4o-mini --generation-cache outputs/hotpotqa_10_cache.json --retriever dense --embedding-model BAAI/bge-base-en-v1.5 --output outputs/hotpotqa_10_dense_run.json > outputs/hotpotqa_10_dense.log 2>&1

conda run -n rag python scripts/run_experiment.py --dataset hotpotqa --max-queries 10 --max-corpus 200 --cache-dir outputs/hf_cache --counterfactual entity --generator openrouter --model openai/gpt-4o-mini --generation-cache outputs/hotpotqa_10_cf_cache.json --retriever dense --embedding-model BAAI/bge-base-en-v1.5 --output outputs/hotpotqa_10_dense_cf_run.json > outputs/hotpotqa_10_dense_cf.log 2>&1
