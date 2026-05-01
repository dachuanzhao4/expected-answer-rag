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

## Full LLM run

Set an OpenAI API key, then run with cached generation:

```powershell
$env:OPENAI_API_KEY="your_api_key"
& 'C:\Users\pc\AppData\Local\Programs\Python\Python313\python.exe' scripts\run_experiment.py --dataset nq --max-queries 200 --max-corpus 50000 --generator openai --model gpt-4.1-mini --generation-cache outputs\nq_generation_cache.json --output outputs\nq_run.json --records-output outputs\nq_records.jsonl
```

Outputs:

- `outputs/*_run.json`: aggregate metrics, method ranking, generation summary,
  and leakage-bucket metrics.
- `outputs/*_records.jsonl`: per-query generations, rankings, and qrels.
- `outputs/*_generation_cache.json`: reusable LLM generations so interrupted
  runs can resume without paying for completed queries again.
