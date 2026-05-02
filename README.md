# Expected Answer RAG

Minimal experiment scaffold for comparing standard query retrieval, HyDE-style
hypothetical documents, concise expected answers, answer masking, and dual-route
late-fusion retrieval.

## Setup

```powershell
& 'C:\Users\pc\AppData\Local\Programs\Python\Python313\python.exe' -m pip install -r requirements.txt
```

## Dry run

This uses tiny synthetic data and the built-in BM25 retriever, so it does not
need network access or model downloads.

```powershell
& 'C:\Users\pc\AppData\Local\Programs\Python\Python313\python.exe' scripts\run_experiment.py --dataset toy --max-queries 5
```

## BEIR run

First download/check the dataset:

```powershell
& 'C:\Users\pc\AppData\Local\Programs\Python\Python313\python.exe' scripts\prepare_dataset.py --dataset nq --max-queries 10 --max-corpus 1000
```

Then run the retrieval baselines:

```powershell
& 'C:\Users\pc\AppData\Local\Programs\Python\Python313\python.exe' scripts\run_experiment.py --dataset nq --max-queries 200 --max-corpus 50000
```

Useful datasets for the first pass:

- `nq`
- `hotpotqa`
- `fiqa`
- `scifact`

The first pass reports retrieval metrics only. Generation quality evaluation can
be added after the retrieval story is clear.

## Full LLM run with OpenRouter

Set an OpenRouter API key, then run with cached generation:

```powershell
$env:OPENROUTER_API_KEY="your_api_key"
& 'C:\Users\pc\AppData\Local\Programs\Python\Python313\python.exe' scripts\run_experiment.py --dataset nq --max-queries 200 --max-corpus 50000 --generator openrouter --model openai/gpt-5-mini --token-param max_completion_tokens --generation-cache outputs\nq_generation_cache.json --output outputs\nq_run.json --records-output outputs\nq_records.jsonl
```

Before a full run, smoke-test the provider response shape:

```powershell
& 'C:\Users\pc\AppData\Local\Programs\Python\Python313\python.exe' scripts\test_openrouter_call.py --model openai/gpt-5-mini --token-param none
```

If a provider returned empty generations during an earlier run, refresh the cache:

```powershell
& 'C:\Users\pc\AppData\Local\Programs\Python\Python313\python.exe' scripts\run_experiment.py --dataset nq --max-queries 20 --max-corpus 5000 --generator openrouter --model openai/gpt-5-mini --token-param max_completion_tokens --generation-cache outputs\nq_20_generation_cache.json --clear-generation-cache --output outputs\nq_20_run.json --records-output outputs\nq_20_records.jsonl
```

The generator uses OpenRouter's OpenAI-compatible Chat Completions format:
`base_url=https://openrouter.ai/api/v1`, model `openai/gpt-5-mini`, API key
from `OPENROUTER_API_KEY`, and `max_completion_tokens` for GPT-5 models. It
does not pass `temperature` unless you explicitly set `--temperature`.

## Dense Retriever Run

Reuse an existing generation cache and run dense retrieval with BGE:

```powershell
& 'C:\Users\pc\AppData\Local\Programs\Python\Python313\python.exe' scripts\run_experiment.py --dataset nq --max-queries 50 --max-corpus 5000 --retriever dense --embedding-model BAAI/bge-base-en-v1.5 --embedding-batch-size 16 --embedding-chunk-size 512 --embedding-cache outputs\embeddings\nq_50_5k_bge_base --generator openrouter --model openai/gpt-5-mini --token-param none --generation-cache outputs\nq_50_query_aware_cache.json --cache-only --cache-namespace openrouter:openai/gpt-5-mini:temp=None --output outputs\nq_50_dense_bge_base_5k_weighted_run.json --records-output outputs\nq_50_dense_bge_base_5k_weighted_records.jsonl
```

Dense embedding caches are checkpointed by chunk. During a long run, completed
chunks are written under:

```text
outputs/embeddings/<cache_name>_chunks/
```

If the run is interrupted, rerun the same command and existing chunks will be
reused. When all chunks are complete, the final combined cache is written as:

```text
outputs/embeddings/<cache_name>.npy
outputs/embeddings/<cache_name>.json
```

Longer 50k-corpus dense run:

```powershell
& 'C:\Users\pc\AppData\Local\Programs\Python\Python313\python.exe' scripts\run_experiment.py --dataset nq --max-queries 50 --max-corpus 50000 --retriever dense --embedding-model BAAI/bge-base-en-v1.5 --embedding-batch-size 16 --embedding-chunk-size 512 --embedding-cache outputs\embeddings\nq_50_50k_bge_base --generator openrouter --model openai/gpt-5-mini --token-param none --generation-cache outputs\nq_50_query_aware_cache.json --cache-only --cache-namespace openrouter:openai/gpt-5-mini:temp=None --output outputs\nq_50_dense_bge_base_50k_weighted_run.json --records-output outputs\nq_50_dense_bge_base_50k_weighted_records.jsonl
```

Outputs:

- `outputs/*_run.json`: aggregate metrics, method ranking, generation summary,
  and leakage-bucket metrics.
- `outputs/*_records.jsonl`: per-query generations, rankings, and qrels.
- `outputs/*_generation_cache.json`: reusable LLM generations so interrupted
  runs can resume without paying for completed queries again.
