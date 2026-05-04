# Initial Stress Test Findings & Project Status

This document summarizes the current pilot stress tests, following earlier N=5 dry-run validation, and tracks experimental progress against the `updated_project_experimental_design.md` memo.

## 1. Key Findings from Stress Tests

### Summary of Performance (BM25 nDCG@10, `--max-queries 100`, `--max-corpus 200`)

| Method | NQ | SciFact | HotpotQA | NQ (CF) | SciFact (CF) | HotpotQA (CF) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `query_only` | 0.699 | 1.000 | 0.897 | 0.639 | 0.944 | 0.646 |
| `raw_expected_answer_only` | 0.889 | 1.000 | 0.842 | 0.562 | 0.889 | 0.320 |
| `hyde_doc_only` | 0.918 | 0.959 | 1.000 | 0.630 | 0.959 | 0.539 |
| `query2doc_concat` | 0.910 | 0.924 | 0.975 | 0.739 | 0.932 | 0.619 |
| `generative_relevance_feedback_concat` | 0.903 | 0.959 | 0.975 | 0.407 | 0.865 | 0.408 |
| `corpus_steered_expansion_concat` | 0.651 | 0.836 | 0.697 | 0.612 | 0.780 | 0.523 |
| `corpus_steered_short_concat` | 0.668 | 0.918 | 0.826 | 0.674 | 0.780 | 0.470 |
| `masked_expected_answer_only` | 0.687 | 1.000 | 0.651 | 0.490 | 0.881 | 0.226 |
| `concat_query_masked_expected` | 0.779 | 1.000 | 0.942 | 0.641 | 0.944 | 0.535 |
| `random_span_masking` | 0.889 | 1.000 | 0.779 | 0.588 | 0.896 | 0.278 |
| `entity_only_masking` | 0.702 | 1.000 | 0.635 | 0.617 | 0.889 | 0.197 |
| `generic_mask_slot` | 0.585 | 0.959 | 0.167 | 0.374 | 0.785 | 0.182 |
| `wrong_answer_only` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `concat_query_wrong_answer` | 0.698 | 1.000 | 0.897 | 0.636 | 0.944 | 0.646 |
| `answer_candidate_constrained_template_only` | 0.718 | 1.000 | 0.811 | 0.739 | 0.937 | 0.533 |
| `concat_query_answer_candidate_constrained_template` | 0.757 | 1.000 | 0.883 | 0.693 | 0.959 | 0.606 |
| `rrf_query_answer_constrained` | 0.747 | 1.000 | 0.898 | 0.669 | 1.000 | 0.639 |

*Note: these runs targeted `--max-queries 100`, but because `--max-corpus 200` truncates the corpus before qrel filtering, the actual evaluable query counts are much smaller: NQ=`14`, SciFact=`9`, HotpotQA=`15`. These are pilot results, not final experiments.*

### A. Public-Original Evaluation
- **First caveat:** these should not be described as true N=100 results. The realized evaluable counts are `14/9/15` queries because the `max-corpus=200` truncation removes many relevant documents before query selection.
- **NQ:** Public NQ no longer says "raw expected answers are best." `hyde_doc_only` (0.918), `query2doc_concat` (0.910), and `generative_relevance_feedback_concat` (0.903) all outperform `query_only` (0.699), while `raw_expected_answer_only` is still strong (0.889). The updated pattern is that multiple expansion families help on public NQ, not only answer-like generation.
- **SciFact:** Public SciFact remains heavily saturated. Several methods hit 1.000, so this regime is still not informative enough by itself to separate leakage from genuine reformulation.
- **HotpotQA:** Public HotpotQA strongly favors pseudo-document style expansion. `hyde_doc_only` reaches 1.000, `query2doc_concat` and `generative_relevance_feedback_concat` reach 0.975, while `raw_expected_answer_only` drops below the baseline (0.842 vs 0.897).

### B. Entity-Counterfactual Private-Like Evaluation
- **NQ remains the clearest leakage case:** `raw_expected_answer_only` drops from 0.889 in Public to 0.562 in CF, moving from well above baseline to below baseline (0.639). That is a much stronger leakage-sensitive pattern than in the earlier draft.
- **Query-preserving reformulation now looks stronger than GRF:** on NQ CF, `query2doc_concat` and `answer_candidate_constrained_template_only` both reach 0.739, while `generative_relevance_feedback_concat` collapses to 0.407. This reverses the earlier GRF-friendly interpretation.
- **SciFact CF is more favorable to constrained fusion:** `rrf_query_answer_constrained` reaches 1.000 and `concat_query_answer_candidate_constrained_template` reaches 0.959, while `raw_expected_answer_only` drops to 0.889 from a saturated public 1.000.
- **HotpotQA CF is harsh:** most expansions fall below the baseline. The best of the listed methods, `rrf_query_answer_constrained`, still lands just under `query_only` (0.639 vs 0.646), so HotpotQA CF remains the most difficult regime.

### C. Dense Retriever Stress Testing (nDCG@10, `--max-queries 100`, `--max-corpus 200`)

| Method | NQ | SciFact | HotpotQA | NQ (CF) | SciFact (CF) | HotpotQA (CF) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `query_only` | 0.907 | 1.000 | 0.975 | 0.705 | 0.799 | 0.553 |
| `raw_expected_answer_only` | 0.965 | 1.000 | 0.835 | 0.565 | 0.707 | 0.434 |
| `hyde_doc_only` | 0.923 | 1.000 | 1.000 | 0.645 | 0.944 | 0.569 |
| `query2doc_concat` | 0.938 | 1.000 | 1.000 | 0.746 | 0.944 | 0.622 |
| `generative_relevance_feedback_concat` | 0.941 | 0.937 | 1.000 | 0.704 | 0.776 | 0.604 |
| `corpus_steered_expansion_concat` | 0.820 | 0.836 | 0.810 | 0.670 | 0.688 | 0.501 |
| `corpus_steered_short_concat` | 0.907 | 0.918 | 0.951 | 0.710 | 0.710 | 0.535 |
| `masked_expected_answer_only` | 0.909 | 0.959 | 0.697 | 0.469 | 0.726 | 0.313 |
| `concat_query_masked_expected` | 0.909 | 1.000 | 1.000 | 0.662 | 0.776 | 0.501 |
| `random_span_masking` | 0.968 | 1.000 | 0.810 | 0.560 | 0.701 | 0.410 |
| `entity_only_masking` | 0.729 | 1.000 | 0.666 | 0.524 | 0.726 | 0.328 |
| `generic_mask_slot` | 0.587 | 1.000 | 0.203 | 0.333 | 0.744 | 0.288 |
| `wrong_answer_only` | 0.000 | 0.000 | 0.033 | 0.000 | 0.000 | 0.000 |
| `concat_query_wrong_answer` | 0.819 | 0.959 | 0.975 | 0.696 | 0.785 | 0.541 |
| `answer_candidate_constrained_template_only` | 0.781 | 0.921 | 1.000 | 0.651 | 0.739 | 0.574 |
| `concat_query_answer_candidate_constrained_template` | 0.845 | 0.959 | 1.000 | 0.671 | 0.753 | 0.597 |
| `rrf_query_answer_constrained` | 0.860 | 0.944 | 0.975 | 0.676 | 0.739 | 0.546 |
| `rrf_query_corpus_steered_short` | 0.922 | 1.000 | 0.975 | 0.714 | 0.803 | 0.559 |

**Key Takeaways for Dense:**
- **NQ dense still shows leakage sensitivity, but less simply than before:** `raw_expected_answer_only` is very strong in Public (0.965) and clearly worse in CF (0.565), while `query2doc_concat` remains slightly above the CF baseline (0.746 vs 0.705). This supports the NQ leakage story while favoring query-preserving methods over answer-only ones.
- **SciFact dense no longer supports a blanket "dense collapse" narrative:** the baseline drops from 1.000 to 0.799 under CF, but `hyde_doc_only` and `query2doc_concat` both recover to 0.944. The revised interpretation is that dense retrieval is counterfactually sensitive, but some pseudo-document expansions remain highly effective on SciFact.
- **HotpotQA dense CF is weak in absolute terms but not uniformly hostile to expansion:** `query2doc_concat` reaches 0.622 and `concat_query_answer_candidate_constrained_template` reaches 0.597, both above the 0.553 baseline. The earlier "everything collapses" framing is too strong.

### D. Entity and Value Counterfactual Ablation (BM25 nDCG@10, N=100)

This ablation goes beyond just renaming entities (like PERSON and LOCATION) and also rewrites numbers, dates, and values.

| Method | NQ (CF E+V) | SciFact (CF E+V) | HotpotQA (CF E+V) |
|---|:---:|:---:|:---:|
| `query_only` | 0.639 | 0.944 | 0.672 |
| `raw_expected_answer_only` | 0.521 | 0.588 | 0.390 |
| `hyde_doc_only` | 0.630 | 0.959 | 0.424 |
| `query2doc_concat` | 0.774 | 0.937 | 0.711 |
| `generative_relevance_feedback_concat` | 0.677 | 0.807 | 0.534 |
| `corpus_steered_expansion_concat` | 0.593 | 0.780 | 0.553 |
| `corpus_steered_short_concat` | 0.649 | 0.780 | 0.516 |
| `masked_expected_answer_only` | 0.493 | 0.556 | 0.213 |
| `concat_query_masked_expected` | 0.629 | 0.937 | 0.640 |
| `random_span_masking` | 0.543 | 0.556 | 0.369 |
| `entity_only_masking` | 0.516 | 0.556 | 0.299 |
| `generic_mask_slot` | 0.338 | 0.507 | 0.252 |
| `concat_query_wrong_answer` | 0.636 | 0.944 | 0.671 |
| `answer_candidate_constrained_template_only` | 0.653 | 0.944 | 0.495 |
| `concat_query_answer_candidate_constrained_template` | 0.703 | 1.000 | 0.644 |
| `rrf_query_answer_constrained` | 0.678 | 1.000 | 0.651 |

**Key Takeaways for Entity+Value Counterfactual:**
- **NQ E+V strengthens the leakage story:** `raw_expected_answer_only` falls to 0.521, well below the 0.639 baseline, while `query2doc_concat` rises to 0.774. On this rerun, value scrambling does not merely neutralize raw-answer gains; it makes answer-only reformulation actively harmful.
- **SciFact E+V separates raw answers from structured reformulation:** `raw_expected_answer_only` drops to 0.588, but `hyde_doc_only` stays at 0.959 and answer-constrained fusion reaches 1.000. The revised pattern is that value scrambling hurts answer-shaped generation much more than pseudo-document or constrained query-preserving methods.
- **HotpotQA remains fragile, but query-preserving methods still help:** `query2doc_concat` reaches 0.711 above the 0.672 baseline, while raw-answer and mask-only methods remain poor.

### E. Control Re-Check
- **The wrong-answer control is now much cleaner:** `wrong_answer_only` is effectively zero across the board, while `concat_query_wrong_answer` stays close to `query_only`. This suggests the earlier "wrong-answer injection is a no-op" concern was mostly a query-dominance effect in concatenation, not a bug in the control itself.
- **Short corpus-steered expansion is an improvement, but not a breakthrough:** `corpus_steered_short_concat` is consistently better than the longer corpus-steered concat on several settings, especially HotpotQA Public (0.826 vs 0.697) and NQ CF (0.674 vs 0.612), but it is still not a leading method overall.

---

## 2. Experimental Checklist

Based on the **Checklist Before Running Experiments** and the **Minimum Viable Paper Package** defined in the memo, here is the current status:

### ✅ Completed & Validated
- [x] Verify repo method implementations and output fields.
- [x] Freeze dataset versions and qrel formats.
- [x] Add answer/evidence metadata to every query record.
- [x] Freeze generation prompts and schemas (using OpenRouter).
- [x] Implement leakage scorer (scoring via `score_generation_methods` works perfectly).
- [x] Implement and validate the entity-counterfactual benchmark builder (`--counterfactual entity` works).
- [x] Run small dry-run validation (N=5 public and counterfactual tests complete and validate the thesis).
- [x] Run the current pilot matrix across BM25/dense and public/entity/entity+value regimes for NQ, SciFact, and HotpotQA.
- [x] Add integrity outputs for per-query retrieval strings, leakage labels, wrong-answer verification, and counterfactual validation summaries.

### ⏳ Pending (To Be Run)
The following experiments are required for the full conference submission:

**1. Harder Retrieval Pools / Larger Effective Sample Sizes**
- [ ] Move beyond `max-corpus=200`, since the current setting leaves only `14/9/15` evaluable queries for NQ/SciFact/HotpotQA.
  - *Note: this is now the biggest methodological limitation in the pilot. Either the counterfactual corpus generation must be optimized, or a larger fixed hard-negative pool must be built so the effective query count and distractor difficulty both increase.*

**2. Missing Dataset Regimes**
- [x] HotpotQA Entity-Counterfactual test.
- [x] SciFact Entity-Counterfactual test.
- [x] Entity-and-value counterfactual test for all three datasets.

**3. Missing Retrievers**
- [x] Dense Retriever Entity-Counterfactual tests.
- [ ] BM25 + RM3 (Optional traditional pseudo-relevance feedback baseline).

**4. Ablations & Controls**
- [x] Wrong-answer control variants now verify correctly in the new outputs.
- [ ] Export and review qualitative examples showing successes/failures from the new `records.jsonl` dumps.
- [ ] Human spot checks on a stratified sample of renamed documents.

---

## 3. Compact Paper-Style Tables

Metric: `nDCG@10`.
These runs used `--max-queries 100` and `--max-corpus 200`.
The tables below focus on the core methods; the broader control set remains in the JSON outputs and in the earlier sections of this note.

### Effective Query Counts

| Dataset | Evaluable Queries |
| --- | :---: |
| NQ | 14 |
| SciFact | 9 |
| HotpotQA | 15 |

### BM25 Public vs Entity-Counterfactual

| Method | NQ (Public) | NQ (CF) | SciFact (Public) | SciFact (CF) | HotpotQA (Public) | HotpotQA (CF) |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| Query only | 0.699 | 0.639 | 1.000 | 0.944 | 0.897 | 0.646 |
| Raw expected answer | 0.889 | 0.562 | 1.000 | 0.889 | 0.842 | 0.320 |
| HyDE | 0.918 | 0.630 | 0.959 | 0.959 | 1.000 | 0.539 |
| Query2doc | 0.910 | 0.739 | 0.924 | 0.932 | 0.975 | 0.619 |
| GRF | 0.903 | 0.407 | 0.959 | 0.865 | 0.975 | 0.408 |
| Corpus-steered short | 0.668 | 0.674 | 0.918 | 0.780 | 0.826 | 0.470 |
| Query + masked expected | 0.779 | 0.641 | 1.000 | 0.944 | 0.942 | 0.535 |
| Query + answer-constrained | 0.757 | 0.693 | 1.000 | 0.959 | 0.883 | 0.606 |
| RRF(query, answer-constrained) | 0.747 | 0.669 | 1.000 | 1.000 | 0.898 | 0.639 |

### Dense Public vs Entity-Counterfactual

| Method | NQ (Public) | NQ (CF) | SciFact (Public) | SciFact (CF) | HotpotQA (Public) | HotpotQA (CF) |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| Query only | 0.907 | 0.705 | 1.000 | 0.799 | 0.975 | 0.553 |
| Raw expected answer | 0.965 | 0.565 | 1.000 | 0.707 | 0.835 | 0.434 |
| HyDE | 0.923 | 0.645 | 1.000 | 0.944 | 1.000 | 0.569 |
| Query2doc | 0.938 | 0.746 | 1.000 | 0.944 | 1.000 | 0.622 |
| GRF | 0.941 | 0.704 | 0.937 | 0.776 | 1.000 | 0.604 |
| Corpus-steered short | 0.907 | 0.710 | 0.918 | 0.710 | 0.951 | 0.535 |
| Query + masked expected | 0.909 | 0.662 | 1.000 | 0.776 | 1.000 | 0.501 |
| Query + answer-constrained | 0.845 | 0.671 | 0.959 | 0.753 | 1.000 | 0.597 |
| RRF(query, answer-constrained) | 0.860 | 0.676 | 0.944 | 0.739 | 0.975 | 0.546 |

### BM25 Entity+Value Counterfactual

| Method | NQ | SciFact | HotpotQA |
| --- | :---: | :---: | :---: |
| Query only | 0.639 | 0.944 | 0.672 |
| Raw expected answer | 0.521 | 0.588 | 0.390 |
| HyDE | 0.630 | 0.959 | 0.424 |
| Query2doc | 0.774 | 0.937 | 0.711 |
| GRF | 0.677 | 0.807 | 0.534 |
| Corpus-steered short | 0.649 | 0.780 | 0.516 |
| Query + masked expected | 0.629 | 0.937 | 0.640 |
| Query + answer-constrained | 0.703 | 1.000 | 0.644 |
| RRF(query, answer-constrained) | 0.678 | 1.000 | 0.651 |

### Dense Entity+Value Counterfactual

| Method | NQ | SciFact | HotpotQA |
| --- | :---: | :---: | :---: |
| Query only | 0.751 | 0.791 | 0.558 |
| Raw expected answer | 0.677 | 0.522 | 0.377 |
| HyDE | 0.681 | 0.876 | 0.515 |
| Query2doc | 0.732 | 0.862 | 0.649 |
| GRF | 0.710 | 0.807 | 0.590 |
| Corpus-steered short | 0.714 | 0.720 | 0.468 |
| Query + masked expected | 0.670 | 0.821 | 0.504 |
| Query + answer-constrained | 0.705 | 0.799 | 0.524 |
| RRF(query, answer-constrained) | 0.716 | 0.828 | 0.568 |

---

## 4. Delta vs Query-Only and Excess Instability Appendix

Definitions for each method `m` and dataset:

`public_delta(m) = score_public(m) - score_public(query_only)`

`cf_delta(m) = score_cf(m) - score_cf(query_only)`

`excess_instability(m) = (score_public(m) - score_cf(m)) - (score_public(query_only) - score_cf(query_only))`

Metric used throughout: `nDCG@10`.

### Effective Query Counts

| Dataset | Evaluable Queries |
| --- | :---: |
| NQ | 14 |
| SciFact | 9 |
| HotpotQA | 15 |

### BM25 Entity-Counterfactual Diagnostics

| Method | Avg Public Δ | Avg CF Δ | Avg Excess Instab. | NQ Excess | SciFact Excess | HotpotQA Excess |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| Raw expected answer | +0.045 | -0.153 | +0.198 | +0.268 | +0.056 | +0.272 |
| HyDE | +0.094 | -0.034 | +0.128 | +0.228 | -0.056 | +0.211 |
| Query2doc | +0.071 | +0.020 | +0.051 | +0.111 | -0.063 | +0.106 |
| GRF | +0.081 | -0.183 | +0.264 | +0.436 | +0.038 | +0.317 |
| Corpus-steered short | -0.061 | -0.102 | +0.041 | -0.066 | +0.082 | +0.106 |
| Query + masked expected | +0.042 | -0.037 | +0.078 | +0.079 | +0.000 | +0.157 |
| Query + answer-constrained | +0.015 | +0.009 | +0.005 | +0.004 | -0.015 | +0.026 |
| RRF(query, answer-constrained) | +0.016 | +0.026 | -0.010 | +0.017 | -0.056 | +0.009 |
| Wrong answer only | -0.865 | -0.743 | -0.122 | -0.060 | -0.056 | -0.250 |
| Query + wrong answer | -0.000 | -0.001 | +0.001 | +0.002 | +0.000 | +0.001 |

### Dense Entity-Counterfactual Diagnostics

| Method | Avg Public Δ | Avg CF Δ | Avg Excess Instab. | NQ Excess | SciFact Excess | HotpotQA Excess |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| Raw expected answer | -0.028 | -0.117 | +0.089 | +0.198 | +0.092 | -0.022 |
| HyDE | +0.014 | +0.034 | -0.020 | +0.076 | -0.145 | +0.008 |
| Query2doc | +0.018 | +0.085 | -0.067 | -0.010 | -0.145 | -0.045 |
| GRF | -0.002 | +0.009 | -0.011 | +0.035 | -0.040 | -0.027 |
| Corpus-steered short | -0.036 | -0.034 | -0.002 | -0.005 | +0.007 | -0.007 |
| Query + masked expected | +0.009 | -0.039 | +0.048 | +0.046 | +0.023 | +0.076 |
| Query + answer-constrained | -0.026 | -0.012 | -0.014 | -0.028 | +0.005 | -0.020 |
| RRF(query, answer-constrained) | -0.034 | -0.032 | -0.002 | -0.018 | +0.005 | +0.006 |
| Wrong answer only | -0.950 | -0.686 | -0.264 | -0.202 | -0.201 | -0.390 |
| Query + wrong answer | -0.043 | -0.012 | -0.031 | -0.079 | -0.026 | +0.012 |

### BM25 Entity+Value Delta vs Query-Only

| Method | NQ Δ | SciFact Δ | HotpotQA Δ | Avg Δ |
| --- | :---: | :---: | :---: | :---: |
| Raw expected answer | -0.118 | -0.357 | -0.282 | -0.252 |
| HyDE | -0.010 | +0.015 | -0.248 | -0.081 |
| Query2doc | +0.134 | -0.008 | +0.039 | +0.055 |
| GRF | +0.038 | -0.138 | -0.138 | -0.079 |
| Corpus-steered short | +0.009 | -0.164 | -0.156 | -0.103 |
| Query + masked expected | -0.011 | -0.008 | -0.032 | -0.017 |
| Query + answer-constrained | +0.064 | +0.056 | -0.028 | +0.030 |
| RRF(query, answer-constrained) | +0.039 | +0.056 | -0.021 | +0.024 |
| Wrong answer only | -0.639 | -0.944 | -0.672 | -0.752 |
| Query + wrong answer | -0.003 | +0.000 | -0.001 | -0.001 |

### Dense Entity+Value Delta vs Query-Only

| Method | NQ Δ | SciFact Δ | HotpotQA Δ | Avg Δ |
| --- | :---: | :---: | :---: | :---: |
| Raw expected answer | -0.074 | -0.269 | -0.181 | -0.175 |
| HyDE | -0.070 | +0.085 | -0.043 | -0.009 |
| Query2doc | -0.019 | +0.071 | +0.091 | +0.048 |
| GRF | -0.041 | +0.015 | +0.032 | +0.002 |
| Corpus-steered short | -0.037 | -0.071 | -0.090 | -0.066 |
| Query + masked expected | -0.081 | +0.029 | -0.054 | -0.035 |
| Query + answer-constrained | -0.046 | +0.008 | -0.033 | -0.024 |
| RRF(query, answer-constrained) | -0.035 | +0.036 | +0.011 | +0.004 |
| Wrong answer only | -0.735 | -0.791 | -0.538 | -0.688 |
| Query + wrong answer | -0.050 | +0.015 | +0.014 | -0.007 |
