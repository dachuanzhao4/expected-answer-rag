# Expected Answer RAG

This repo studies leakage-aware query reformulation for retrieval and private-like
RAG. The main question is when LLM-generated expansions genuinely improve
evidence retrieval and when they succeed by injecting answer-bearing content
from the model's parametric priors.

The current project compares public BEIR-style evaluation against
entity-counterfactual and entity+value-counterfactual settings, where entities
and optionally dates/values are consistently renamed across queries, corpus
documents, answers, and supporting metadata. The goal is to measure retrieval
quality, leakage, and instability under a setting where public answer priors
should not help.

## Current Scope

Implemented method families include:

- `query_only`
- `raw_expected_answer_only`
- `hyde_doc_only`
- `query2doc_concat`
- `generative_relevance_feedback_concat`
- `corpus_steered_expansion_concat`
- `corpus_steered_short_concat`
- `masked_expected_answer_only`
- `concat_query_masked_expected`
- `answer_candidate_constrained_template_only`
- `concat_query_answer_candidate_constrained_template`
- `rrf_query_answer_constrained`
- `safe_rrf_v0`
- `safe_rrf_v1`
- `cf_prompt_query_expansion_rrf`
- weighted RRF variants
- ablation and control methods such as wrong-answer, random masking, entity-only masking, and generic masking

Retrievers:

- `bm25`
- `dense` with `BAAI/bge-base-en-v1.5`

Counterfactual regimes:

- `public`
- `entity`
- `entity_and_value`

## Repo Layout

- [scripts/run_experiment.py](/Users/weiyueli/Desktop/rag/expected-answer-rag/scripts/run_experiment.py): main experiment runner
- [scripts/prepare_dataset.py](/Users/weiyueli/Desktop/rag/expected-answer-rag/scripts/prepare_dataset.py): dataset download and preview
- [scripts/build_counterfactual_dataset.py](/Users/weiyueli/Desktop/rag/expected-answer-rag/scripts/build_counterfactual_dataset.py): export renamed local datasets
- [scripts/export_pilot_tables.py](/Users/weiyueli/Desktop/rag/expected-answer-rag/scripts/export_pilot_tables.py): build compact markdown tables from finished JSON runs
- [run_all.sh](/Users/weiyueli/Desktop/rag/expected-answer-rag/run_all.sh): pilot matrix runner across datasets, retrievers, and regimes
- [src/expected_answer_rag/](/Users/weiyueli/Desktop/rag/expected-answer-rag/src/expected_answer_rag): datasets, generators, counterfactual builder, leakage scoring, retrieval, metrics, and stats
- [docs/stress_test_findings.md](/Users/weiyueli/Desktop/rag/expected-answer-rag/docs/stress_test_findings.md): current integrated pilot findings

## Environment

The current workflow assumes a conda environment named `rag`.

```bash
conda create -n rag python=3.11 -y
conda run -n rag python -m pip install -r requirements.txt
```

If you use OpenRouter-backed generation, set:

```bash
export OPENROUTER_API_KEY=...
```

## Quick Start

### 1. Dry run on toy data

```bash
conda run -n rag python scripts/run_experiment.py \
  --dataset toy \
  --max-queries 5 \
  --generator heuristic \
  --output outputs/toy_run.json \
  --records-output outputs/toy_records.jsonl
```

### 2. Download and preview a dataset

```bash
conda run -n rag python scripts/prepare_dataset.py \
  --dataset nq \
  --max-queries 20 \
  --max-corpus 1000 \
  --cache-dir outputs/hf_cache \
  --output outputs/nq_preview.json
```

### 3. Run a public BM25 experiment

```bash
conda run -n rag python scripts/run_experiment.py \
  --dataset nq \
  --max-queries 100 \
  --max-corpus 200 \
  --cache-dir outputs/hf_cache \
  --generator openrouter \
  --model openai/gpt-5-mini \
  --token-param none \
  --generation-workers 4 \
  --generation-cache outputs/nq_100_cache.json \
  --retriever bm25 \
  --output outputs/nq_100_run.json \
  --records-output outputs/nq_100_records.jsonl
```

### 4. Run a dense experiment from the same generation cache

```bash
conda run -n rag python scripts/run_experiment.py \
  --dataset nq \
  --max-queries 100 \
  --max-corpus 200 \
  --cache-dir outputs/hf_cache \
  --generator openrouter \
  --model openai/gpt-5-mini \
  --token-param none \
  --generation-workers 4 \
  --generation-cache outputs/nq_100_cache.json \
  --retriever dense \
  --embedding-cache outputs/embeddings/nq_100_bge \
  --local-files-only \
  --output outputs/nq_100_dense_run.json \
  --records-output outputs/nq_100_records.jsonl
```

### 5. Run an entity-counterfactual experiment

```bash
conda run -n rag python scripts/run_experiment.py \
  --dataset nq \
  --max-queries 100 \
  --max-corpus 200 \
  --cache-dir outputs/hf_cache \
  --counterfactual entity \
  --counterfactual-alias-style natural \
  --generator openrouter \
  --model openai/gpt-5-mini \
  --token-param none \
  --generation-workers 4 \
  --generation-cache outputs/nq_100_cf_cache.json \
  --retriever bm25 \
  --output outputs/nq_100_cf_run.json \
  --records-output outputs/nq_100_cf_records.jsonl
```

### 6. Run the pilot matrix

```bash
bash run_all.sh
```

This runs:

- datasets: `nq`, `scifact`, `hotpotqa`
- retrievers: `bm25`, `dense`
- regimes: `public`, `entity`, `entity_and_value`

### 7. Run the matrix with all corpus

```bash
FULL_CORPUS=1 GENERATION_WORKERS=4 bash run_all.sh
```

This omits `--max-corpus`, appends `_full` to output filenames, and uses multithreaded generation precompute.

### 8. Run the matrix with a custom corpus cap and worker count

```bash
MAX_CORPUS=1000 GENERATION_WORKERS=6 RUN_TAG=cap1000 bash run_all.sh
```

Useful environment variables for [run_all.sh](/Users/weiyueli/Desktop/rag/expected-answer-rag/run_all.sh):

- `FULL_CORPUS=1` omits `--max-corpus`
- `MAX_CORPUS=...` overrides the default cap when `FULL_CORPUS` is not set
- `GENERATION_WORKERS=...` controls threaded generation precompute
- `RUN_TAG=...` adds a suffix to output filenames to avoid overwriting prior runs

## New Methods

The runner now emits these additional methods in every experiment JSON:

- `safe_rrf_v0`
  Fixed weighted RRF over `query_only`, `generative_relevance_feedback_concat`, `query2doc_concat`, and `concat_query_answer_candidate_constrained_template` with weights `1.0 / 0.8 / 0.55 / 0.55`.
- `safe_rrf_v1`
  The same route set, but with per-query gating from route support, unsupported candidate count, anchor coverage, route agreement, and answer-form penalty.
- `cf_prompt_query_expansion_rrf`
  Counterfactual-prompted multi-query expansion RRF that obfuscates public entity triggers during generation, then de-obfuscates known anchors before retrieval.

No extra flag is required. If you run [scripts/run_experiment.py](/Users/weiyueli/Desktop/rag/expected-answer-rag/scripts/run_experiment.py) or [run_all.sh](/Users/weiyueli/Desktop/rag/expected-answer-rag/run_all.sh), these methods are included automatically.
Generation can be parallelized across queries with `--generation-workers`.

### Focused commands for the new methods

Run one public BM25 job:

```bash
conda run -n rag python scripts/run_experiment.py \
  --dataset nq \
  --max-queries 100 \
  --max-corpus 200 \
  --cache-dir outputs/hf_cache \
  --generator openrouter \
  --model openai/gpt-5-mini \
  --token-param none \
  --generation-cache outputs/nq_100_cache.json \
  --retriever bm25 \
  --output outputs/nq_100_run.json \
  --records-output outputs/nq_100_records.jsonl
```

Run one entity-counterfactual BM25 job:

```bash
conda run -n rag python scripts/run_experiment.py \
  --dataset nq \
  --max-queries 100 \
  --max-corpus 200 \
  --cache-dir outputs/hf_cache \
  --counterfactual entity \
  --counterfactual-alias-style natural \
  --generator openrouter \
  --model openai/gpt-5-mini \
  --token-param none \
  --generation-cache outputs/nq_100_cf_cache.json \
  --retriever bm25 \
  --output outputs/nq_100_cf_run.json \
  --records-output outputs/nq_100_cf_records.jsonl
```

Run the full matrix:

```bash
bash run_all.sh
```

Print only the new method metrics from a finished run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
path = Path("outputs/nq_100_run.json")
data = json.loads(path.read_text())
for name in ["safe_rrf_v0", "safe_rrf_v1", "cf_prompt_query_expansion_rrf"]:
    print(name, json.dumps(data["metrics"].get(name, {}), ensure_ascii=False))
PY
```

## Counterfactual Dataset Export

To export a renamed local dataset:

```bash
conda run -n rag python scripts/build_counterfactual_dataset.py \
  --dataset nq \
  --max-queries 100 \
  --max-corpus 200 \
  --alias-style natural \
  --output-dir outputs/nq_counterfactual_local
```

This writes:

- `corpus.jsonl`
- `queries.jsonl`
- `qrels.jsonl`
- `manifest.json`
- `alias_table.json`
- `validation.json`

## Outputs

Each experiment writes:

- a summary JSON via `--output`
- a per-query JSONL dump via `--records-output`
- a generation cache via `--generation-cache`

The summary JSON includes:

- aggregate metrics
- method ranking
- leakage summaries
- primary comparison stats
- integrity summary
- sample generations

The records JSONL includes:

- exact retrieval strings per method
- retrieval specs for concat / RRF methods
- per-query leakage labels
- wrong-answer verification
- full rankings and qrels

Dense embedding caches are stored under:

```text
outputs/embeddings/
```

## Current Caveat

The current pilot configuration often uses `--max-corpus 200` to align public
and counterfactual distractor pools. That makes runs cheap, but it also reduces
the number of evaluable queries after qrel filtering. The current pilot
findings should therefore be treated as stress-test results, not final paper
results.

The latest integrated write-up is in
[docs/stress_test_findings.md](/Users/weiyueli/Desktop/rag/expected-answer-rag/docs/stress_test_findings.md).

## Regenerating Pilot Tables

If you update the JSON outputs and want refreshed compact markdown tables:

```bash
python3 scripts/export_pilot_tables.py
```

## Main Dependencies

- `datasets`
- `rank-bm25`
- `sentence-transformers`
- `torch`
- `openai`
- `spacy`

If the counterfactual builder uses spaCy NER, make sure the English model is
installed in the active environment.
