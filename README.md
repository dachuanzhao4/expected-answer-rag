# Expected Answer RAG

Minimal experiment scaffold for comparing standard query retrieval, HyDE-style
hypothetical documents, concise expected answers, answer masking, and dual-route
late-fusion retrieval.

## Plan 1: Private-Like Leakage Bias Study

### Research Hypothesis

HyDE-style retrieval can work very well on public Wikipedia-like QA benchmarks
because the LLM may already have useful answer priors from pretraining. In that
setting, generating a hypothetical answer passage can accidentally inject the
right answer and improve retrieval.

The same behavior can be harmful in private-domain RAG. For internal enterprise
corpora, the LLM often does not know the ground-truth answer. A hypothetical
answer passage may then introduce unsupported names, teams, dates, project codes,
or policies, steering retrieval toward the model's prior instead of the private
corpus evidence.

This project studies that gap:

```text
Public QA:   HyDE may benefit from memorized or broadly learned answer priors.
Private RAG: HyDE may inject wrong answer candidates and retrieval bias.
```

The goal is not simply to beat HyDE on public benchmarks. The goal is to measure
and reduce **prior-induced retrieval bias** when LLM priors are unreliable.

Planned mitigation methods:

- `query-aware masking`: generate answer-like text, then replace only the
  answer-bearing span with a typed slot such as `[PERSON]`, `[LOCATION]`, or
  `[NUMBER]`.
- `answer-agnostic templates`: generate retrieval expressions with known query
  anchors and typed unknown slots, without generating a concrete answer.

The key experimental direction is to compare these methods on both normal public
QA data and private-like data where LLM answer priors should not help.

### Private-Like Benchmark Plan

To simulate private enterprise data in a reproducible way, build a renamed QA
benchmark from public QA corpora:

```text
Original:
Marie Curie was born in Warsaw.
Where was Marie Curie born?

Private-like renamed:
Employee ZQ-17 was born in Site LM-42.
Where was Employee ZQ-17 born?
```

The corpus and query are renamed consistently, so the answer is still present in
the documents. But the LLM should no longer know the answer from pretraining.

Expected behavior:

- `HyDE` may hallucinate or inject wrong answer candidates.
- `masked HyDE / masked expected answer` should reduce wrong answer steering.
- `answer-agnostic templates` should avoid concrete answer injection entirely.

Important metrics:

- Retrieval quality: `Recall@k`, `MRR@10`, `nDCG@10`
- Answer leakage rate: whether generated retrieval text contains the gold answer
- Wrong answer injection rate: whether generated text introduces unsupported
  entities or values
- Robustness on query-only failure cases

### Answer-Agnostic Template JSON

The template method should generate retrieval plans that preserve known query
anchors but represent the unknown answer only as a typed slot. A reasonable JSON
format is:

```json
{
  "known_anchors": [
    {
      "text": "Love Will Keep Us Alive",
      "type": "TITLE",
      "role": "song"
    },
    {
      "text": "Eagles",
      "type": "ORGANIZATION",
      "role": "artist_or_band"
    }
  ],
  "unknown_answer": {
    "slot": "[PERSON]",
    "type": "PERSON",
    "role": "singer_or_lead_vocalist"
  },
  "relation_intent": "performer / lead vocalist",
  "leakage_free_queries": [
    "\"Love Will Keep Us Alive\" Eagles lead vocals",
    "\"Love Will Keep Us Alive\" features lead vocals by [PERSON]",
    "Eagles \"Love Will Keep Us Alive\" vocalist",
    "\"Love Will Keep Us Alive\" singer Eagles"
  ],
  "evidence_templates": [
    "\"Love Will Keep Us Alive\" features lead vocals by [PERSON].",
    "[PERSON] sings \"Love Will Keep Us Alive\" by the Eagles."
  ],
  "constraints": {
    "do_not_generate_concrete_answer": true,
    "only_use_entities_from_question": true,
    "preserve_known_anchors": true
  }
}
```

For a private-style renamed query:

```text
Where was Employee ZQ-17 born?
```

the plan should look like:

```json
{
  "known_anchors": [
    {
      "text": "Employee ZQ-17",
      "type": "PERSON",
      "role": "subject"
    }
  ],
  "unknown_answer": {
    "slot": "[LOCATION]",
    "type": "LOCATION",
    "role": "birthplace"
  },
  "relation_intent": "birthplace",
  "leakage_free_queries": [
    "Employee ZQ-17 birthplace",
    "Employee ZQ-17 born in [LOCATION]",
    "place of birth Employee ZQ-17"
  ],
  "evidence_templates": [
    "Employee ZQ-17 was born in [LOCATION].",
    "[LOCATION] is the birthplace of Employee ZQ-17."
  ],
  "constraints": {
    "do_not_generate_concrete_answer": true,
    "only_use_entities_from_question": true,
    "preserve_known_anchors": true
  }
}
```

Retrieval can then use:

```text
retrieve(original query)
retrieve(each leakage_free_query)
retrieve(each evidence_template)
merge with concat, RRF, or weighted RRF
```

The important constraint is that the template method should never introduce a
new concrete answer candidate. It may introduce relation words such as
`birthplace`, `lead vocals`, or `episodes`, but not a new person, location,
date, project name, or value absent from the question.

---

## Plan 2: Expected Answer / HyDE Baseline Experiments

This was the original experiment plan. It compares standard query retrieval,
HyDE-style hypothetical passages, concise expected answers, query-aware masking,
and fusion strategies. These experiments are still useful as baselines and
motivation, but the stronger research direction is Plan 1.

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
