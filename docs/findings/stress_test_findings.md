# Stress Test Findings and Current Status

This note aggregates the corrected `cap1000` pilot produced with:

- `N=20`
- `MAX_CORPUS=1000`
- `RUN_TAG=cap1000`
- `METHOD_PROFILE=main`

It covers `NQ`, `SciFact`, and `HotpotQA` under:

- `BM25`
- dense retrieval with `BAAI/bge-base-en-v1.5`
- `public`
- `entity-counterfactual`
- `entity+value-counterfactual`

## 1. Important Correction

The earlier dense outputs were invalid because dense jobs initially reused `BM25` checkpoint records through a shared `records.jsonl` path. That resume bug has now been fixed by:

- using retriever-specific checkpoint files
- requiring checkpoint-context compatibility before resume
- rerunning the dense matrix

All dense results in this memo come from the corrected reruns and should be treated as the current version for PI review.

## 2. Scope and Caveat

This is still a pilot:

- only `20` queries per dataset
- corpus cap `1000`, not full corpus
- useful for method triage, story revision, and audit findings, not final paper claims

The new result is more informative than the earlier all-method stress test because it uses the reduced main-profile matrix plus the new FAWE and query-dominance controls.

## 3. Headline Takeaways

- `fawe_query2doc_beta0p25` is now the strongest overall averaged method across all six regimes, with all-regime average `nDCG@10 = 0.730`.
- The strongest simple anchored baseline remains `concat_query_raw_expected` at all-regime average `0.726`.
- `safe_rrf_v1` still has a good robustness shape, but it is not the best absolute method. Its all-regime average is `0.704`, slightly above `query_only` at `0.694`.
- `fawe_safe_adaptive_beta` is competitive but not dominant. Its all-regime average is `0.705`, essentially tied with `safe_rrf_v1`.
- `raw_expected_answer_only` remains the clearest leakage probe. Its regime averages are:
  - `BM25 public / entity / entity+value = 0.765 / 0.429 / 0.345`
  - `dense public / entity / entity+value = 0.818 / 0.442 / 0.363`
- The updated query-dominance controls strengthen the revised paper story:
  - `query_repeated` stays almost identical to `query_only`
  - `query_plus_shuffled_expected` remains very strong
  - `query_plus_neutral_filler` is disastrous for `BM25` but only mildly harmful for dense retrieval
- The dense reruns no longer mirror `BM25`. Dense public performance favors strong anchored concatenation and pseudo-document routes, while dense counterfactual winners are more mixed and less dominated by `query2doc`.
- The duplicate-method audit still shows that several oracle/post-hoc variants are exact duplicates of the raw route in both retrieval text and top-10 output. That remains an implementation or design cleanup issue before final-scale experiments.

## 4. Overall Average Table

Metric throughout: `nDCG@10`, averaged over all `18` dataset x retriever x regime slices.

| Method | All-Regime Average | Interpretation |
| --- | ---: | --- |
| `fawe_query2doc_beta0p25` | `0.730` | Strongest overall method in this pilot. |
| `raw_expected_then_query` | `0.727` | Effectively tied with the strongest anchored expansions. |
| `concat_query_random_span_masking` | `0.727` | Still a very strong control, not a paper-facing main method. |
| `concat_query_raw_expected` | `0.726` | Strongest simple answer-anchored method. |
| `fawe_raw_expected_beta0p25` | `0.724` | Strong FAWE variant, especially outside dense public. |
| `query_plus_shuffled_expected` | `0.724` | Strong evidence that query anchoring plus lexical enrichment matters more than answer correctness alone. |
| `query2doc_concat` | `0.710` | Still the strongest external prior-work baseline. |
| `fawe_safe_adaptive_beta` | `0.705` | Competitive adaptive FAWE, but not better than simple FAWE-query2doc. |
| `safe_rrf_v1` | `0.704` | Best SAFE-family method overall, but not best absolute. |
| `safe_rrf_v0` | `0.701` | Slightly weaker overall than `v1`. |
| `query_only` | `0.694` | Still a very strong anchor. |
| `concat_query_answer_candidate_constrained_template` | `0.685` | Safer, but well below the strongest anchored methods. |
| `cf_prompt_query_expansion_rrf` | `0.656` | Mid-tier; improved but not a lead method. |
| `raw_expected_answer_only` | `0.527` | Clear standalone leakage/failure probe. |

## 5. Regime Leaders

Average over `NQ`, `SciFact`, and `HotpotQA`.

| Regime | Best Average Method | Avg `nDCG@10` |
| --- | --- | ---: |
| `BM25 public` | `fawe_query2doc_beta0p25` | `0.899` |
| `BM25 entity-CF` | `concat_query_raw_expected` / `query_plus_shuffled_expected` / `raw_expected_then_query` / `concat_query_entity_only_masking` | `0.698` |
| `BM25 entity+value-CF` | `fawe_query2doc_beta0p25` | `0.711` |
| `dense public` | `concat_query_random_span_masking` | `0.931` |
| `dense entity-CF` | `fawe_raw_expected_beta0p25` | `0.606` |
| `dense entity+value-CF` | `query_plus_shuffled_expected` | `0.614` |

Interpretation:

- `BM25` still rewards anchored lexical expansion strongly.
- FAWE is most convincing in the `query2doc` route, especially on `BM25 public` and `BM25 entity+value`.
- Dense retrieval does not favor the same method family in every regime.
- No SAFE-family method wins a dense regime on absolute score.

## 6. Compact Regime Table

Average `nDCG@10` over the three datasets.

| Method | BM25 Public | BM25 Entity | BM25 E+V | Dense Public | Dense Entity | Dense E+V |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_only` | `0.781` | `0.663` | `0.666` | `0.898` | `0.572` | `0.586` |
| `raw_expected_answer_only` | `0.765` | `0.429` | `0.345` | `0.818` | `0.442` | `0.363` |
| `concat_query_raw_expected` | `0.888` | `0.698` | `0.656` | `0.920` | `0.588` | `0.605` |
| `hyde_doc_only` | `0.859` | `0.538` | `0.480` | `0.885` | `0.535` | `0.570` |
| `query2doc_concat` | `0.883` | `0.651` | `0.690` | `0.903` | `0.549` | `0.585` |
| `generative_relevance_feedback_concat` | `0.813` | `0.454` | `0.450` | `0.877` | `0.535` | `0.569` |
| `corpus_steered_short_concat` | `0.729` | `0.601` | `0.600` | `0.863` | `0.527` | `0.551` |
| `concat_query_masked_expected` | `0.834` | `0.683` | `0.636` | `0.879` | `0.547` | `0.577` |
| `concat_query_answer_candidate_constrained_template` | `0.812` | `0.668` | `0.663` | `0.832` | `0.554` | `0.582` |
| `safe_rrf_v0` | `0.832` | `0.674` | `0.661` | `0.892` | `0.573` | `0.573` |
| `safe_rrf_v1` | `0.818` | `0.676` | `0.689` | `0.891` | `0.566` | `0.582` |
| `cf_prompt_query_expansion_rrf` | `0.749` | `0.601` | `0.612` | `0.866` | `0.540` | `0.568` |
| `fawe_raw_expected_beta0p25` | `0.855` | `0.686` | `0.678` | `0.915` | `0.606` | `0.607` |
| `fawe_masked_expected_beta0p25` | `0.811` | `0.668` | `0.673` | `0.890` | `0.573` | `0.600` |
| `fawe_answer_constrained_beta0p5` | `0.815` | `0.668` | `0.685` | `0.886` | `0.556` | `0.603` |
| `fawe_query2doc_beta0p25` | `0.899` | `0.675` | `0.711` | `0.915` | `0.579` | `0.599` |
| `fawe_safe_adaptive_beta` | `0.810` | `0.661` | `0.690` | `0.890` | `0.574` | `0.603` |

## 7. Updated Interpretation

### 7.1 FAWE is promising, but only in some forms

The clearest positive result from the new implementation round is that **FAWE is worth keeping**, but not every FAWE route is equally good.

- `fawe_query2doc_beta0p25` is the strongest overall method and the strongest `BM25` method under public and entity+value settings.
- `fawe_raw_expected_beta0p25` is the strongest dense entity-counterfactual average method.
- `fawe_safe_adaptive_beta` is respectable, but it does **not** beat simpler FAWE variants or the strongest anchored baselines.

This suggests that **fielded anchor-weighted fusion is viable**, but the winning signal seems to come from lightly weighting already-strong expansion families rather than from the current adaptive policy itself.

### 7.2 The paper story still centers on query anchoring

The revised claim from the PI memo still holds and is now stronger:

- standalone answer-style retrieval is unstable
- preserving the original query rescues generated expansions
- exact answer correctness is not the only driver of gains

`query_plus_shuffled_expected` is especially important here. Its all-regime average is `0.724`, almost identical to `concat_query_raw_expected` at `0.726`. That is strong evidence that a large part of the benefit is coming from **query anchoring plus lexical/semantic enrichment**, not only from the model predicting the right answer string.

### 7.3 SAFE remains robust-looking, not best-looking

`safe_rrf_v1` still has a good paper shape because it avoids the collapse of the raw standalone route and remains competitive in harder counterfactual settings. But it is not the top-scoring method overall, and it is not the strongest dense method.

Recommended interpretation:

> SAFE remains a useful conservative leakage-aware method family, but the current pilot favors simpler anchored and fielded combinations for peak retrieval score.

### 7.4 Constrained templates remain safe but not strong enough yet

`concat_query_answer_candidate_constrained_template` averages:

- `0.812 / 0.668 / 0.663` on `BM25 public / entity / entity+value`
- `0.832 / 0.554 / 0.582` on dense

So the constrained route still looks sensible as a leakage-aware baseline, but it is well behind the strongest anchored raw-answer and FAWE variants.

## 8. Query-Dominance Audit

Average delta vs `query_only`:

| Control | BM25 Public | BM25 Entity | BM25 E+V | Dense Public | Dense Entity | Dense E+V |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `concat_query_wrong_answer` | `-0.009` | `-0.003` | `-0.005` | `-0.034` | `-0.045` | `-0.013` |
| `query_repeated` | `0.000` | `0.000` | `0.000` | `-0.020` | `-0.009` | `-0.001` |
| `query_repeated_length_matched` | `-0.020` | `-0.020` | `-0.017` | `-0.040` | `-0.030` | `-0.001` |
| `query_plus_shuffled_expected` | `+0.107` | `+0.035` | `-0.010` | `+0.013` | `+0.006` | `+0.028` |
| `query_plus_neutral_filler` | `-0.409` | `-0.256` | `-0.256` | `-0.017` | `-0.018` | `-0.015` |
| `neutral_filler_plus_query` | `-0.409` | `-0.256` | `-0.256` | `-0.042` | `-0.037` | `-0.039` |

Interpretation:

- `query_repeated` is effectively a null control.
- `query_plus_shuffled_expected` remains surprisingly strong, again pointing to query anchoring plus token enrichment rather than precise answer correctness.
- `query_plus_neutral_filler` is catastrophic for `BM25` but only mildly harmful for dense retrieval. Dense is much less sensitive to appended filler than `BM25`.
- `concat_query_wrong_answer` stays close to `query_only`, which supports the query-dominance explanation rather than a broken wrong-answer control.

## 9. Concatenation Rescue Audit

Average `score(query + generated) - score(generated_only)`:

| Pair | BM25 Public | BM25 Entity | BM25 E+V | Dense Public | Dense Entity | Dense E+V |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw expected -> query + raw expected | `0.124` | `0.269` | `0.311` | `0.101` | `0.147` | `0.241` |
| masked expected -> query + masked expected | `0.219` | `0.302` | `0.318` | `0.183` | `0.168` | `0.246` |
| constrained -> query + constrained | `0.058` | `0.107` | `0.098` | `0.031` | `0.073` | `0.022` |
| random span -> query + random span | `0.128` | `0.274` | `0.322` | `0.126` | `0.165` | `0.264` |
| entity-only -> query + entity-only | `0.266` | `0.305` | `0.281` | `0.297` | `0.190` | `0.262` |
| generic slot -> query + generic slot | `0.362` | `0.396` | `0.376` | `0.420` | `0.309` | `0.343` |

This is one of the clearest results in the whole pilot. The original query consistently rescues weak generated text, often by very large margins.

## 10. Dense Position Audit

Average dense ordering effect:

| Comparison | Dense Public | Dense Entity | Dense E+V |
| --- | ---: | ---: | ---: |
| `concat_query_raw_expected - raw_expected_then_query` | `+0.001` | `-0.002` | `-0.007` |
| `query_plus_neutral_filler - neutral_filler_plus_query` | `+0.025` | `+0.019` | `+0.024` |

Interpretation:

- For dense retrieval, swapping `query` and `raw_expected` changes little on average.
- The filler-first vs query-first difference is small but non-zero; dense still appears relatively insensitive to appended filler order.
- The earlier suspicion that dense might simply be ignoring appended text is now too strong. Appended text matters, but much less destructively than in `BM25`.

## 11. Duplicate Audit

Public `BM25` average duplicate rates:

| Pair | Identical Retrieval Text Rate | Identical Top-10 Rate |
| --- | ---: | ---: |
| `gold_answer_only` vs `raw_expected_answer_only` | `1.0` | `1.0` |
| `oracle_answer_masked` vs `raw_expected_answer_only` | `1.0` | `1.0` |
| `post_hoc_gold_removed_expected_answer` vs `raw_expected_answer_only` | `1.0` | `1.0` |
| `concat_query_oracle_answer_masked` vs `concat_query_raw_expected` | `1.0` | `1.0` |
| `concat_query_post_hoc_gold_removed_expected` vs `concat_query_raw_expected` | `1.0` | `1.0` |
| `concat_query_wrong_answer` vs `wrong_answer_injection` | `1.0` | `1.0` |

These are still exact aliases in practice. They should be consolidated, renamed as aliases, or fixed before the next full-scale run.

## 12. Current Recommendation

If the next round needs a small set of methods for paper-facing analysis, the priority order should now be:

1. `query_only`
2. `query2doc_concat`
3. `concat_query_raw_expected`
4. `safe_rrf_v1`
5. `fawe_query2doc_beta0p25`
6. `fawe_raw_expected_beta0p25`
7. `fawe_safe_adaptive_beta`
8. `concat_query_answer_candidate_constrained_template`
9. `cf_prompt_query_expansion_rrf`
10. `raw_expected_answer_only`
11. `query_plus_shuffled_expected`
12. `concat_query_wrong_answer`
13. `query_repeated`
14. `query_plus_neutral_filler`

Paper-facing interpretation:

- **Evaluation story:** still strong and probably stronger than the pure-method story.
- **Mechanistic story:** now even clearer. LLM-assisted retrieval gains are a mixture of answer priors, lexical enrichment, and query anchoring.
- **Method story:** FAWE is promising, especially in the `query2doc` route, but the adaptive FAWE and SAFE families are not yet dominant enough to support a “new method wins everything” claim.

The most defensible current thesis remains:

> Public benchmark gains from LLM query expansion conflate answer-prior leakage with the retrieval benefits of query anchoring and lexical enrichment. Counterfactual evaluation reveals that standalone answer-like routes are unstable, while anchored and lightly fielded expansions are much more robust.
