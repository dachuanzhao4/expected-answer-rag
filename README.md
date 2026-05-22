# Leakage-Free Reformat for Private RAG

> Encoding note: this file is UTF-8. If Chinese text looks garbled in old Windows PowerShell, open it in VS Code, GitHub, or any UTF-8 Markdown viewer.

这个目录是从原始实验仓库中整理出来的 final workspace。它只保留当前 paper 叙事真正需要的脚本、结果、数据元信息和 README；早期 toy/smoke、expected-answer、失败 format、LF-ER v1/v3/v4/v41/v42、系统维护脚本都没有放进来。

## TL;DR

我们想研究一个问题：

**Query2Doc/HyDE 在公开 QA 数据集上有效，是否是因为它真的学到了“检索格式”，还是因为 LLM 已经知道公开答案并把答案先验写进了 pseudo-doc？如果换成 private corpus，LLM 不知道答案，生成式 query expansion 会不会反而伤害检索？**

当前结果支持一个初步叙事：

- Public NQ 上，Query2Doc 很强。
- Private-like renamed NQ 上，LLM 对 renamed entity 没有可靠先验。
- 在真正被 rename 的 private-effective subset 中，dense 检索里 Query2Doc 低于 query-only。
- `llm_reformat_v2` 不生成答案，只输出结构化 retrieval intent，在 private-effective dense 上超过 query-only 和 Query2Doc。
- 但 BM25 上 Query2Doc 仍然更强，说明 sparse 场景还需要更好的 answer-free lexical reformat。

## Paper Narrative

### Claim 1: Public gains mix format and answer prior

Query2Doc/HyDE 类方法把 query 改写成 pseudo document。这个 pseudo document 有两个可能贡献：

- **format contribution**: 生成文本更像 corpus 中的 passage，包含关系词、上下文词、文档式句子。
- **hallucination/prior contribution**: LLM 可能已经知道公开数据里的答案，并把答案或近似答案写进生成文本。

在普通 NQ 上，这两个贡献混在一起，很难直接分开。

### Claim 2: Private-like rename isolates unreliable prior

为了模拟 private corpus，我们把 corpus 和 query 中的关键实体同步 rename。这样：

- 检索任务仍然成立，因为 query 和 corpus 两边一起改。
- BM25/dense 仍然可以通过 renamed anchors 找文档。
- LLM 不能靠公开知识直接知道 `Entity_2J0504` 或 `KeplerRow0504` 对应什么。

如果 Query2Doc 的提升主要依赖 public answer prior，那么在这种 private-like setting 下，生成 pseudo-doc 就可能引入错误先验或拒答文本。

### Claim 3: Reformat should preserve format without answer leakage

最终方法不是让 LLM 继续写 pseudo-doc，而是让它只做 **answer-free reformat**：

- 保留原 query 中已经知道的 anchors。
- 判断 answer type。
- 判断 relation class。
- 生成短、受控、无答案的 retrieval views。

这就是 `llm_reformat_v2`。

## Folder Layout

```text
rag_final/
  README.md
  requirements.txt
  scripts/
    run_experiment.py
    run_stage2_nq.py
    run_stage2_renamed_nq.py
    build_renamed_dataset.py
    validate_renamed_dataset.py
    probe_openrouter_concurrency.py
    test_answer_blanked_format.py
    summarize_final_results.py
  src/expected_answer_rag/
    runtime source package
  data_metadata/renamed_nq_stage2_token_v5_full/
    opaque/      stats, mapping, queries, qrels
    plausible/  stats, mapping, queries, qrels
  results/
    final_summary.json
    public_nq/
    renamed_private_like/
      caches/
      final_v2/
```

Full `corpus.jsonl` files are not copied here because each is over 100 MB. They should be rebuilt locally with `scripts/build_renamed_dataset.py`.

## Step-by-Step Logic

### Step 1: Run normal NQ

Purpose:

Show the ordinary Query2Doc effect on public NQ.

Methods compared:

- `query_only`
- `query2doc_expanded_query`
- `masked_query2doc_expanded_query`

Small example:

```text
Query:
who sings Love Will Keep Us Alive by the Eagles

Query2Doc may generate:
Love Will Keep Us Alive is a song by the Eagles, with lead vocals by Timothy B. Schmit...

Masked Query2Doc becomes:
Love Will Keep Us Alive is a song by the Eagles, with lead vocals by [PERSON]...
```

Why this matters:

If Query2Doc improves and masked Query2Doc also improves but less, that suggests both format and answer prior matter.

Implementation:

- Runner: `scripts/run_stage2_nq.py`
- Core experiment: `scripts/run_experiment.py`
- Generator prompts: `src/expected_answer_rag/generators.py`
- Retrieval: `src/expected_answer_rag/retrieval.py`

### Step 2: Build private-like renamed NQ

Purpose:

Create a benchmark where the answer remains retrievable from corpus, but LLM public prior becomes unreliable.

Small example:

```text
Original query:
how many episodes are in chicago fire season 4

Renamed query, opaque:
how many episodes are in Entity_2J0504 fire season 4

Renamed query, plausible:
how many episodes are in KeplerRow0504 fire season 4
```

The same replacement is applied inside corpus passages, so this is not random corruption. Retrieval should still be possible.

Current final dataset settings:

- source dataset: NQ
- source corpus: 200,000 passages
- source queries: 500
- replacement granularity: `token`
- replacement token policy: `preserve`
- query rename policy: `safe_aligned`
- kept queries: 500
- queries with replacements: 284
- unchanged control queries: 216

Why 284 and 216:

- 284 queries contain safe corpus-aligned replacement tokens, so they are the **private-effective subset**.
- 216 queries are unchanged controls because no safe single-token replacement was available. They are still kept; we do not drop half the benchmark.

Implementation:

- Builder: `scripts/build_renamed_dataset.py`
- Validator: `scripts/validate_renamed_dataset.py`
- Metadata: `data_metadata/renamed_nq_stage2_token_v5_full/`

### Step 3: Run private-like Query2Doc and masked Query2Doc

Purpose:

Test whether Query2Doc still helps when the LLM cannot know renamed entities.

Expected behavior:

- If Query2Doc relies heavily on public answer prior, it may generate wrong text or refusal-like text.
- Masking answer-bearing spans reduces leakage but cannot fully repair bad pseudo-doc structure.

Small example:

```text
Renamed query:
who sings love will keep us alive by the Entity_LG0842

Bad Query2Doc risk:
The model may say it cannot identify Entity_LG0842, or may guess a public-world answer.

Masked Query2Doc:
Can blank answer spans, but if the pseudo-doc is already about the wrong world, masking alone is not enough.
```

Implementation:

- Same runner: `scripts/run_stage2_renamed_nq.py`
- Same generation cache also stores Query2Doc and masked Query2Doc.

### Step 4: Run LLM reformat v2

Purpose:

Keep useful retrieval format while preventing answer hallucination.

Instead of asking LLM to write a document, ask it for structured intent only.

Example input:

```text
who sings love will keep us alive by the Entity_LG0842
```

LLM output schema:

```json
{
  "answer_type": "PERSON",
  "anchors": [
    {"text": "love will keep us alive", "role": "subject", "importance": "primary"},
    {"text": "Entity_LG0842", "role": "context", "importance": "support"}
  ],
  "query_focus_terms": ["sings", "love will keep us alive"],
  "relation_class": "performer",
  "relation_confidence": "high",
  "retrieval_policy": "anchor_plus_one_cue"
}
```

Rendered retrieval views:

```text
anchor view:
who sings love will keep us alive by the Entity_LG0842 love will keep us alive Entity_LG0842

dense view:
who sings love will keep us alive by the Entity_LG0842 performer

bm25 view:
who sings love will keep us alive by the Entity_LG0842 love will keep us alive Entity_LG0842 singer performer vocals
```

Important constraints:

- Anchors must be exact query substrings.
- The LLM is not allowed to output a concrete answer.
- Query-external entities are rejected or pruned.
- Refusal text such as "I do not know" must not enter retrieval.
- Dense view is intentionally short to avoid embedding drift.

Implementation:

- Prompt: `src/expected_answer_rag/generators.py`
- Schema/rendering/validation: `src/expected_answer_rag/answer_blanked.py`
- Experiment integration: `scripts/run_experiment.py`

## Current Results

Primary metric: `ndcg@10`.

Machine-readable summary:

```bash
python scripts/summarize_final_results.py --output results/final_summary.json
```

Full 500-query runs:

| setting | query-only | query2doc | masked query2doc | llm reformat v2 |
| --- | ---: | ---: | ---: | ---: |
| public NQ BM25 | 0.3818 | 0.6049 | 0.5424 | - |
| public NQ dense | 0.6868 | 0.7451 | 0.7178 | - |
| opaque private BM25 | 0.3818 | 0.4910 | 0.4583 | 0.3824 |
| opaque private dense | 0.5239 | 0.5403 | 0.5356 | 0.5252 |
| plausible private BM25 | 0.3818 | 0.4944 | 0.4593 | 0.3808 |
| plausible private dense | 0.5273 | 0.5436 | 0.5368 | 0.5276 |

Private-effective subset, 284 renamed queries:

| setting | n | query-only | query2doc | masked query2doc | llm reformat v2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| opaque BM25 | 284 | 0.4031 | 0.4654 | 0.4471 | 0.4074 |
| opaque dense | 284 | 0.4519 | 0.4346 | 0.4316 | 0.4585 |
| plausible BM25 | 284 | 0.4031 | 0.4661 | 0.4515 | 0.4038 |
| plausible dense | 284 | 0.4567 | 0.4394 | 0.4397 | 0.4632 |

Interpretation:

- Public NQ supports the standard Query2Doc/HyDE effect.
- Private-effective dense shows the key failure: Query2Doc becomes worse than query-only.
- LLM reformat v2 recovers signal in private-effective dense.
- BM25 still benefits from long Query2Doc lexical expansion, so our current reformat is not yet sparse-optimal.

## How To Reproduce

Install dependencies:

```bash
cd rag_final
python -m pip install -r requirements.txt
```

Set OpenRouter API key.

Git Bash:

```bash
export OPENROUTER_API_KEY="your_key_here"
```

PowerShell:

```powershell
$env:OPENROUTER_API_KEY = "your_key_here"
```

### 1. Public NQ

```bash
python scripts/run_stage2_nq.py \
  --max-queries 500 \
  --max-corpus 200000 \
  --model openai/gpt-5-mini \
  --token-param none \
  --api-workers 16 \
  --embedding-device cuda \
  --embedding-batch-size 8 \
  --embedding-chunk-size 256
```

If GPU is unstable:

```bash
python scripts/run_stage2_nq.py \
  --max-queries 500 \
  --max-corpus 200000 \
  --model openai/gpt-5-mini \
  --token-param none \
  --api-workers 16 \
  --embedding-device cuda \
  --embedding-batch-size 4 \
  --embedding-chunk-size 128
```

### 2. Rebuild private-like renamed NQ

```bash
python scripts/build_renamed_dataset.py \
  --dataset nq \
  --max-queries 500 \
  --max-corpus 200000 \
  --mode both \
  --output-root outputs/renamed_nq_stage2_token_v5_full \
  --query-rename-policy safe_aligned \
  --replacement-granularity token \
  --replacement-token-policy preserve \
  --query-ngram-anchors off \
  --allow-entity-only-query
```

Validate:

```bash
python scripts/validate_renamed_dataset.py outputs/renamed_nq_stage2_token_v5_full/opaque
python scripts/validate_renamed_dataset.py outputs/renamed_nq_stage2_token_v5_full/plausible
```

### 3. Private-like final v2

```bash
python scripts/run_stage2_renamed_nq.py \
  --skip-build \
  --mode both \
  --max-queries 500 \
  --max-corpus 200000 \
  --include-llm-reformat \
  --llm-reformat-version v2 \
  --model openai/gpt-5-mini \
  --token-param none \
  --cache-tag renamed_nq_stage2_token_v5_full \
  --embedding-cache-tag renamed_nq_stage2_token_v5_full_v42_dense_safe \
  --api-workers 16 \
  --embedding-device cuda \
  --embedding-batch-size 8 \
  --embedding-chunk-size 256
```

### 4. Summarize

```bash
python scripts/summarize_final_results.py --output results/final_summary.json
```

## What Is Included

Included:

- public NQ 500-query BM25/dense results
- final renamed NQ token-v5 metadata
- final `llm_reformat_v2` BM25/dense records
- Query2Doc/masked Query2Doc caches needed to inspect generated text
- OpenRouter concurrency probe
- final format validation test
- final summary script

Not included:

- early expected-answer experiments
- toy/smoke runs
- partial files
- pagefile/GPU/system maintenance scripts
- old LF-ER heuristic branches
- failed or overfit format attempts
- dense embedding caches
- full `corpus.jsonl`

## Current Problems

1. **Evidence scale is still small.**

Current main run is 500 queries and 200k passages. This is enough for method debugging and stage-1 evidence, but not enough for final paper-level claims.

2. **Private-like data is synthetic.**

Renamed NQ is useful because it controls public prior, but it is still not a real enterprise/private RAG benchmark.

3. **BM25 is not solved.**

Query2Doc remains strong for BM25 because long pseudo-docs add many lexical bridge terms. The current reformat is deliberately conservative and dense-friendly.

4. **The method currently uses an LLM call.**

This is fine for studying leakage-free reformat, but cost/latency should be discussed. A smaller local classifier or distilled reformatter may be a future direction.

5. **The unchanged 216 queries dilute the private effect.**

They are kept as controls, but headline private claims should report both full 500 and private-effective 284.

## Next Steps

1. **Scale the experiment.**

Run more queries and larger corpora after the 500-query pipeline is stable. Report confidence intervals or bootstrap significance.

2. **Add stronger private benchmarks.**

Use at least one non-Wikipedia or post-training/private-like corpus where LLM prior is genuinely unreliable.

3. **Improve sparse/BM25 reformat.**

Design answer-free lexical expansion that adds relation synonyms without becoming a long hallucinated pseudo-doc.

4. **Add ablations.**

Recommended ablations:

- anchors only
- anchors + answer type
- anchors + relation class
- anchors + relation class + controlled cue
- dense short view vs BM25 lexical view
- opaque vs plausible rename

5. **Diagnose failure modes.**

For each failed query, classify whether the issue is:

- missing anchor
- wrong relation class
- too generic cue
- dense embedding drift
- BM25 lexical mismatch
- renamed token not sufficiently represented in relevant docs

6. **Consider training-free first, distillation later.**

The current method is prompt-based and training-free, which is good for positioning. If results scale, a distilled local reformatter can be proposed as an efficiency variant rather than the core contribution.
