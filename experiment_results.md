# Experiment Results

## Run Configuration

- Command: `CUDA_VISIBLE_DEVICES=1 GENERATION_WORKERS=30 bash run_all.sh`
- Repo: `/Users/weiyueli/Desktop/rag/expected-answer-rag`
- Model: `openai/gpt-4o-mini`
- Query cap: `N=100`
- Corpus cap: `MAX_CORPUS=2000`
- Corpus mode: `FULL_CORPUS=0`
- Retrievers: `bm25`, `dense`
- Regimes: `public`, `entity-counterfactual`, `entity+value-counterfactual`
- Method profile: `main`
- FAWE betas: `0.25,0.5`
- Audit sample size: `10`

Observed run sizes from the output JSONs:
- `nq`: 68 evaluated queries, 2000 corpus docs
- `scifact`: 100 evaluated queries, 2000 corpus docs
- `hotpotqa`: 85 evaluated queries, 2000 corpus docs

## Per-Run Summary

| Dataset | Retriever | Regime | q | corpus | query_only | Best method | Best nDCG@10 | Delta vs query_only | Runner-up | Third |
|---|---|---:|---:|---:|---:|---|---:|---:|---|---|
| hotpotqa | bm25 | entity | 85 | 2000 | 0.506 | `fawe_answer_constrained_beta0p5` | 0.526 | +0.020 | `fawe_safe_adaptive_beta` | `fawe_raw_expected_beta0p25` |
| hotpotqa | bm25 | entity+value | 85 | 2000 | 0.510 | `fawe_masked_expected_beta0p25` | 0.512 | +0.002 | `fawe_safe_adaptive_beta` | `fawe_answer_constrained_beta0p5` |
| hotpotqa | bm25 | public | 85 | 2000 | 0.802 | `fawe_query2doc_beta0p25` | 0.906 | +0.104 | `concat_query_raw_expected` | `query_plus_shuffled_expected` |
| hotpotqa | dense | entity | 85 | 2000 | 0.237 | `concat_query_random_span_masking` | 0.290 | +0.053 | `concat_query_answer_candidate_constrained_template` | `concat_query_generic_mask_slot` |
| hotpotqa | dense | entity+value | 85 | 2000 | 0.289 | `raw_expected_then_query` | 0.324 | +0.035 | `concat_query_random_span_masking` | `concat_query_generic_mask_slot` |
| hotpotqa | dense | public | 85 | 2000 | 0.913 | `raw_expected_then_query` | 0.948 | +0.035 | `concat_query_random_span_masking` | `query_plus_shuffled_expected` |
| nq | bm25 | entity | 68 | 2000 | 0.430 | `fawe_query2doc_beta0p25` | 0.532 | +0.102 | `concat_query_raw_expected` | `query_plus_shuffled_expected` |
| nq | bm25 | entity+value | 68 | 2000 | 0.433 | `fawe_query2doc_beta0p25` | 0.506 | +0.073 | `concat_query_raw_expected` | `query_plus_shuffled_expected` |
| nq | bm25 | public | 68 | 2000 | 0.545 | `query2doc_concat` | 0.780 | +0.235 | `fawe_query2doc_beta0p25` | `raw_expected_answer_only` |
| nq | dense | entity | 68 | 2000 | 0.534 | `concat_query_raw_expected` | 0.557 | +0.023 | `concat_query_random_span_masking` | `concat_query_entity_only_masking` |
| nq | dense | entity+value | 68 | 2000 | 0.475 | `concat_query_generic_mask_slot` | 0.527 | +0.051 | `concat_query_random_span_masking` | `raw_expected_then_query` |
| nq | dense | public | 68 | 2000 | 0.827 | `query_plus_shuffled_expected` | 0.863 | +0.037 | `concat_query_random_span_masking` | `generative_relevance_feedback_concat` |
| scifact | bm25 | entity | 100 | 2000 | 0.682 | `fawe_raw_expected_beta0p25` | 0.707 | +0.025 | `fawe_query2doc_beta0p25` | `concat_query_raw_expected` |
| scifact | bm25 | entity+value | 100 | 2000 | 0.714 | `fawe_safe_adaptive_beta` | 0.723 | +0.009 | `fawe_answer_constrained_beta0p5` | `concat_query_raw_expected` |
| scifact | bm25 | public | 100 | 2000 | 0.787 | `fawe_query2doc_beta0p25` | 0.831 | +0.044 | `query2doc_concat` | `safe_rrf_v0` |
| scifact | dense | entity | 100 | 2000 | 0.532 | `hyde_doc_only` | 0.626 | +0.094 | `query2doc_concat` | `concat_query_generic_mask_slot` |
| scifact | dense | entity+value | 100 | 2000 | 0.529 | `hyde_doc_only` | 0.657 | +0.129 | `query2doc_concat` | `generative_relevance_feedback_concat` |
| scifact | dense | public | 100 | 2000 | 0.818 | `hyde_doc_only` | 0.841 | +0.024 | `query2doc_concat` | `fawe_query2doc_beta0p25` |

## Cross-Run Averages For Key Methods

| Method | Avg nDCG@10 across all 18 runs |
|---|---:|
| `query_only` | 0.5867 |
| `raw_expected_answer_only` | 0.5590 |
| `concat_query_raw_expected` | 0.6196 |
| `query2doc_concat` | 0.6038 |
| `fawe_query2doc_beta0p25` | 0.6201 |
| `safe_rrf_v1` | 0.6022 |
| `fawe_safe_adaptive_beta` | 0.5936 |

## Average Best Methods By Retriever/Regime

| Retriever | Regime | Top avg method | Avg nDCG@10 | Next two |
|---|---|---|---:|---|
| bm25 | public | `fawe_query2doc_beta0p25` | 0.8385 | `query2doc_concat` (0.829), `concat_query_raw_expected` (0.811) |
| bm25 | entity | `fawe_query2doc_beta0p25` | 0.5727 | `concat_query_raw_expected` (0.570), `query_plus_shuffled_expected` (0.570) |
| bm25 | entity+value | `fawe_raw_expected_beta0p25` | 0.5667 | `fawe_answer_constrained_beta0p5` (0.567), `safe_rrf_v0` (0.565) |
| dense | public | `fawe_raw_expected_beta0p25` | 0.8663 | `concat_query_random_span_masking` (0.864), `query2doc_concat` (0.864) |
| dense | entity | `concat_query_generic_mask_slot` | 0.4693 | `concat_query_random_span_masking` (0.467), `concat_query_raw_expected` (0.460) |
| dense | entity+value | `concat_query_generic_mask_slot` | 0.4688 | `raw_expected_then_query` (0.464), `concat_query_random_span_masking` (0.460) |

## Excess Counterfactual Instability

Excess instability = method drop from public to counterfactual minus the `query_only` drop. Positive means the method is more brittle than the baseline under counterfactual shift.

| Retriever | Method | Avg excess instability |
|---|---|---:|
| bm25 | `raw_expected_answer_only` | +0.1457 |
| bm25 | `concat_query_raw_expected` | +0.0784 |
| bm25 | `query2doc_concat` | +0.1514 |
| bm25 | `fawe_query2doc_beta0p25` | +0.1064 |
| bm25 | `safe_rrf_v1` | +0.0322 |
| bm25 | `fawe_safe_adaptive_beta` | -0.0024 |
| dense | `raw_expected_answer_only` | -0.0023 |
| dense | `concat_query_raw_expected` | -0.0152 |
| dense | `query2doc_concat` | -0.0078 |
| dense | `fawe_query2doc_beta0p25` | -0.0018 |
| dense | `safe_rrf_v1` | -0.0010 |
| dense | `fawe_safe_adaptive_beta` | -0.0122 |

## Counterfactual Rewrite Validation Snapshot

| Dataset | Regime | Replacement coverage | Residual corpus mentions | Residual query mentions |
|---|---|---:|---:|---:|
| nq | entity | 0.9724 | 661 | 48 |
| nq | entity+value | 0.7677 | 1053 | 141 |
| scifact | entity | 0.9943 | 1262 | 185 |
| scifact | entity+value | 0.7891 | 2608 | 321 |
| hotpotqa | entity | 0.9644 | 1057 | 129 |
| hotpotqa | entity+value | 0.9396 | 1608 | 233 |

## Main Findings

1. **BM25 public runs still show the largest upside for expansion methods.** Averaged over the three public BM25 runs, `fawe_query2doc_beta0p25` is strongest at **0.8385 nDCG@10**, slightly ahead of `query2doc_concat` at **0.8292** and clearly above `query_only` at **0.7112**.
2. **The counterfactual story remains strong for BM25: standalone answer-like retrieval is much less robust than anchored methods.** Across BM25 runs, `raw_expected_answer_only` has the largest excess counterfactual instability among the tracked answer-centric methods (**+0.1457**), while anchored variants such as `concat_query_raw_expected` (**+0.0784**), `safe_rrf_v1` (**+0.0322**), and especially `fawe_safe_adaptive_beta` (**-0.0024**) are much more stable.
3. **`fawe_query2doc_beta0p25` is the strongest single BM25 method overall.** It is the top method in **4 of the 18 runs**, including all three public BM25 datasets except NQ, where plain `query2doc_concat` is marginally higher (**0.7801 vs 0.7783**). It also leads the average BM25 entity-counterfactual bucket (**0.5727**).
4. **Dense public runs are high-ceiling and crowded.** Public dense `query_only` is already very strong (**0.8524 average nDCG@10**), so most gains are modest. The best average dense public methods are clustered tightly: `fawe_raw_expected_beta0p25` (**0.8663**), `concat_query_random_span_masking` (**0.8643**), `query2doc_concat` (**0.8643**), and `hyde_doc_only` (**0.8640**).
5. **Dense counterfactual runs are much harder for everyone, and the relative gaps are smaller.** Dense `query_only` drops from **0.8524** on public to roughly **0.4340 / 0.4310** on entity / entity+value. Because the baseline itself collapses so much, the excess instability values for dense methods are near zero. In other words, the dense counterfactual effect currently looks more like a broad alias/distribution-shift problem than a method-specific leakage effect.
6. **Natural Questions is still the clearest BM25 leakage case.** On public NQ BM25, `raw_expected_answer_only` reaches **0.7633** against `query_only` at **0.5450**. Under counterfactual renaming it falls to **0.4889** (entity) and **0.4855** (entity+value), while anchored/query-preserving methods such as `fawe_query2doc_beta0p25` stay substantially stronger (**0.5319** and **0.5058**).
7. **SciFact remains the main exception where dense HyDE/query2doc stay strong even under counterfactual evaluation.** `hyde_doc_only` is the top method on both SciFact dense counterfactual runs (**0.6259** entity, **0.6574** entity+value), suggesting dataset-specific behavior that is worth discussing explicitly rather than folding into one universal claim.
8. **HotpotQA BM25 counterfactual runs are comparatively stable.** `query_only` is already strong (**0.5064** on entity and **0.5100** on entity+value), and the best methods only improve it slightly. That makes HotpotQA less about spectacular gains and more about whether a method preserves quality under renaming.

## Interpretation For PI Review

- The current run supports the core paper claim on the **BM25 side**: public gains can be inflated by answer-like or unanchored generations, while anchored/fielded methods remain stronger under counterfactual shift.
- The strongest candidate method family is still **FAWE**, especially `fawe_query2doc_beta0p25` for overall BM25 accuracy and `fawe_safe_adaptive_beta` / `fawe_answer_constrained_beta0p5` for robustness-oriented BM25 counterfactual settings.
- The dense results need more careful framing. They do **not** cleanly isolate leakage in the same way BM25 does, because dense retrieval itself is heavily affected by counterfactual renaming and alias shift. This should likely be presented as a limitation or as a separate phenomenon rather than merged into the main BM25-style claim.
- The current run is materially stronger than the old tiny-corpus pilot, but it is still a **2000-document capped** study rather than full corpus. That should be stated explicitly in any note to your PI.

## Recommended Next Actions

1. Use these results to position **FAWE query2doc** as the main accuracy method and **FAWE safe/adaptive** as the main robustness method.
2. In the writeup, separate **BM25 leakage findings** from **dense alias-shift findings** instead of forcing one narrative across both retrievers.
3. Flag SciFact dense as an exception case for targeted qualitative/error analysis.
4. If compute allows, repeat the strongest configurations with a larger corpus or full corpus before making top-conference-strength claims.
