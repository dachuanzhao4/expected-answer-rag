# Experiment Notes

## Project Goal

This project studies whether retrieval can be improved by generating a concise
expected answer before retrieval. The current method compares standard query
retrieval, HyDE-style hypothetical documents, raw expected answers, query-aware
masked expected answers, and two fusion strategies.

The main hypothesis is that concise expected answers may reduce uncontrolled
semantic drift compared with long HyDE documents, while query-aware masking can
reduce answer leakage by replacing only the answer-bearing span with a typed
slot.

## Current Pipeline

For each query:

1. Generate a concise expected answer.
2. Generate a HyDE-style hypothetical document.
3. Generate a query-aware masked expected answer.
4. Retrieve with BM25 using multiple query variants.
5. Evaluate retrieval against BEIR qrels.

The query-aware mask prompt receives both the original query and the expected
answer. It preserves entities already present in the query and masks only the
span that directly answers the query.

Example:

```text
Question: how many episodes are in chicago fire season 4
Expected answer: Chicago Fire Season 4 has 23 episodes.
Masked answer: Chicago Fire Season 4 has [NUMBER] episodes.
```

## Generator Setting

- Provider: OpenRouter
- Model: `openai/gpt-5-mini`
- API format: OpenAI-compatible Chat Completions
- Token parameter: `none`
- Temperature: not passed
- Generation cache: enabled
- Retriever: BM25

The smoke test confirmed that `openai/gpt-5-mini` returns normal
`choices[0].message.content` when no token parameter is passed.

## Baselines

| Method | Description |
|---|---|
| `query_only` | Retrieve with the original query. |
| `hyde_doc_only` | Retrieve with generated HyDE document only. |
| `raw_expected_answer_only` | Retrieve with raw expected answer only. |
| `masked_expected_answer_only` | Retrieve with query-aware masked expected answer only. |
| `concat_query_raw_expected` | Retrieve with query and raw expected answer concatenated. |
| `concat_query_masked_expected` | Retrieve with query and masked expected answer concatenated. |
| `dual_query_raw_expected_rrf` | Retrieve query and raw expected answer separately, then RRF. |
| `dual_query_masked_expected_rrf` | Retrieve query and masked expected answer separately, then RRF. |

## NQ-5 Smoke Test

Output file:

```text
outputs/nq_5_fixed_run.json
```

Setting:

```text
dataset: nq
max_queries: 5
max_corpus: 1000
retriever: BM25
generator: openrouter
model: openai/gpt-5-mini
token_param: none
```

Results:

| Method | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| `concat_query_raw_expected` | 1.00 | 1.00 | 1.00 | 0.90 | 0.9141 |
| `concat_query_masked_expected` | 1.00 | 1.00 | 1.00 | 0.90 | 0.9141 |
| `dual_query_raw_expected_rrf` | 1.00 | 1.00 | 1.00 | 0.90 | 0.9141 |
| `dual_query_masked_expected_rrf` | 1.00 | 1.00 | 1.00 | 0.90 | 0.9141 |
| `raw_expected_answer_only` | 0.90 | 1.00 | 1.00 | 0.90 | 0.9022 |
| `masked_expected_answer_only` | 0.90 | 1.00 | 1.00 | 0.90 | 0.9022 |
| `query_only` | 1.00 | 1.00 | 1.00 | 0.80 | 0.8774 |
| `hyde_doc_only` | 0.90 | 1.00 | 1.00 | 0.80 | 0.8597 |

Generation summary:

| Statistic | Value |
|---|---:|
| Avg expected answer tokens | 15.4 |
| Avg HyDE document tokens | 138.0 |
| Avg expected capitalized spans | 2.8 |
| Avg HyDE capitalized spans | 13.8 |
| Avg mask slots | 1.4 |

Observation:

The NQ-5 smoke test showed that query-aware masking fixed the earlier
over-masking problem. Masked expected answers preserved the retrieval anchors and
performed similarly to raw expected answers on this small sample.

## NQ-50 Initial Run

Output file:

```text
outputs/nq_50_query_aware_run.json
```

Setting:

```text
dataset: nq
max_queries: 50
max_corpus: 50000
retriever: BM25
generator: openrouter
model: openai/gpt-5-mini
token_param: none
```

Results:

| Method | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 | Delta vs Query |
|---|---:|---:|---:|---:|---:|---:|
| `hyde_doc_only` | 0.92 | 0.95 | 0.95 | 0.7578 | 0.7963 | +0.3030 |
| `raw_expected_answer_only` | 0.81 | 0.86 | 0.89 | 0.6973 | 0.7252 | +0.2320 |
| `concat_query_raw_expected` | 0.83 | 0.87 | 0.90 | 0.6862 | 0.7167 | +0.2235 |
| `dual_query_raw_expected_rrf` | 0.72 | 0.87 | 0.91 | 0.5806 | 0.6309 | +0.1377 |
| `concat_query_masked_expected` | 0.69 | 0.77 | 0.85 | 0.5134 | 0.5575 | +0.0642 |
| `masked_expected_answer_only` | 0.64 | 0.74 | 0.78 | 0.5302 | 0.5542 | +0.0609 |
| `dual_query_masked_expected_rrf` | 0.66 | 0.79 | 0.83 | 0.4894 | 0.5418 | +0.0485 |
| `query_only` | 0.53 | 0.67 | 0.82 | 0.4579 | 0.4933 | +0.0000 |

Generation summary:

| Statistic | Value |
|---|---:|
| Avg expected answer tokens | 18.24 |
| Avg HyDE document tokens | 157.08 |
| Avg expected capitalized spans | 2.78 |
| Avg HyDE capitalized spans | 14.08 |
| Avg mask slots | 1.22 |

Observation:

All generated-query methods outperformed the original query baseline on this
BM25 BEIR-NQ setting. HyDE was strongest, likely because long hypothetical
documents create more lexical overlap with Wikipedia-style corpus passages.

The raw expected answer remained strong despite being much shorter than HyDE.
This supports the value of expected-answer retrieval, but the current result
does not support the claim that concise expected answers always outperform HyDE.

Query-aware masked expected answers still outperformed query-only retrieval, but
they were weaker than raw expected answers. This suggests a retrieval/leakage
tradeoff: masking removes answer-bearing terms that help retrieval.

The current RRF fusion strategy underperformed concatenation. Weighted RRF or a
reranker should be tested before drawing conclusions about dual-route retrieval.

## NQ-50 Dense BGE Run

Output file:

```text
outputs/nq_50_dense_bge_base_5k_run.json
```

Setting:

```text
dataset: nq
max_queries: 50
max_corpus: 5000
retriever: dense
embedding_model: BAAI/bge-base-en-v1.5
query_prefix: "Represent this sentence for searching relevant passages: "
generator: cache-only reuse of OpenRouter generations
```

Model choice:

`BAAI/bge-base-en-v1.5` was selected as a common recent open-source dense
retrieval baseline. The BGE v1.5 model card reports strong MTEB retrieval
performance and recommends the query instruction used above for retrieval.

Results:

| Method | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 | Delta vs Query |
|---|---:|---:|---:|---:|---:|---:|
| `concat_query_raw_expected` | 0.98 | 0.98 | 1.00 | 0.8833 | 0.8983 | +0.0840 |
| `raw_expected_answer_only` | 0.98 | 1.00 | 1.00 | 0.8352 | 0.8689 | +0.0546 |
| `dual_query_raw_expected_rrf` | 0.97 | 0.98 | 1.00 | 0.8423 | 0.8684 | +0.0541 |
| `hyde_doc_only` | 0.94 | 1.00 | 1.00 | 0.8107 | 0.8491 | +0.0348 |
| `concat_query_masked_expected` | 0.96 | 0.98 | 0.98 | 0.8123 | 0.8392 | +0.0249 |
| `query_only` | 0.95 | 0.97 | 0.98 | 0.7752 | 0.8143 | +0.0000 |
| `dual_query_masked_expected_rrf` | 0.95 | 0.97 | 0.98 | 0.7695 | 0.8089 | -0.0054 |
| `masked_expected_answer_only` | 0.84 | 0.90 | 0.90 | 0.7212 | 0.7592 | -0.0551 |

Observation:

Dense retrieval changes the story compared with BM25. On this smaller 5k-corpus
NQ-50 dense run, raw expected-answer retrieval outperformed HyDE, and
concatenating the original query with the raw expected answer was best overall.
This supports the hypothesis that concise expected answers can be more useful
than long HyDE documents under semantic embedding retrieval.

The masked expected answer still loses performance relative to raw expected
answers, which is expected because answer-bearing tokens are useful retrieval
signals. Query-aware masking is therefore better framed as a leakage-control
tradeoff rather than a pure retrieval improvement.

Attempted full 50k-corpus dense run:

```text
dataset: nq
max_queries: 50
max_corpus: 50000
embedding_model: BAAI/bge-base-en-v1.5
embedding_batch_size: 64
```

The model downloaded and loaded, but the process exited during corpus encoding
without a Python traceback. This looks like a local memory or torch runtime
issue. The successful 5k run used `embedding_batch_size=16`.

Update:

Dense retrieval now supports chunk-level embedding checkpoints. Completed chunks
are written immediately under `<embedding_cache>_chunks/`, and rerunning the same
command will skip valid existing chunks. The previous interrupted 50k attempt did
not produce reusable chunks because chunk checkpointing had not been implemented
yet.

## Current Interpretation

Supported by current experiments:

- Generated answer/document retrieval substantially improves over query-only
  BM25 retrieval on BEIR-NQ.
- Raw expected answer retrieval is a strong retrieval signal.
- Query-aware masking avoids the severe over-masking failure seen in the first
  prompt version.
- Expected answers are much shorter and contain fewer extra named entities than
  HyDE documents.

Not yet supported:

- Expected answers outperform HyDE on BM25 BEIR-NQ.
- Masked expected answers outperform raw expected answers.
- Current unweighted RRF is better than concatenation.
- Leakage reduction improves retrieval metrics; BEIR-NQ qrels do not include
  gold answer strings in the loaded format, so leakage buckets remain unknown.

## Next Experiments

1. Run dense retrieval.
   Initial BGE dense results support this. Next try larger corpus sizes with
   smaller batch sizes or add embedding caching/chunking.

2. Add weighted RRF.
   Current RRF gives masked answer retrieval too much influence. Test weights
   such as query 1.0 and masked expected answer 0.2, 0.5, 1.0.

3. Use a dataset with gold answer strings.
   Needed for answer leakage analysis. Candidates include NQ-Open or TriviaQA.

4. Evaluate with larger NQ samples.
   Run 200 queries and possibly 100k corpus if cost and runtime are acceptable.

5. Run HotpotQA.
   Multi-hop questions may expose different tradeoffs between long HyDE
   documents and concise expected answers.

## Commands

NQ-50:

```bash
'/c/Users/pc/AppData/Local/Programs/Python/Python313/python.exe' scripts/run_experiment.py \
  --dataset nq \
  --max-queries 50 \
  --max-corpus 50000 \
  --generator openrouter \
  --model openai/gpt-5-mini \
  --token-param none \
  --generation-cache outputs/nq_50_query_aware_cache.json \
  --output outputs/nq_50_query_aware_run.json \
  --records-output outputs/nq_50_query_aware_records.jsonl
```

NQ-200:

```bash
'/c/Users/pc/AppData/Local/Programs/Python/Python313/python.exe' scripts/run_experiment.py \
  --dataset nq \
  --max-queries 200 \
  --max-corpus 100000 \
  --generator openrouter \
  --model openai/gpt-5-mini \
  --token-param none \
  --generation-cache outputs/nq_200_query_aware_cache.json \
  --output outputs/nq_200_query_aware_run.json \
  --records-output outputs/nq_200_query_aware_records.jsonl
```

HotpotQA-50:

```bash
'/c/Users/pc/AppData/Local/Programs/Python/Python313/python.exe' scripts/run_experiment.py \
  --dataset hotpotqa \
  --max-queries 50 \
  --max-corpus 50000 \
  --generator openrouter \
  --model openai/gpt-5-mini \
  --token-param none \
  --generation-cache outputs/hotpotqa_50_query_aware_cache.json \
  --output outputs/hotpotqa_50_query_aware_run.json \
  --records-output outputs/hotpotqa_50_query_aware_records.jsonl
```
