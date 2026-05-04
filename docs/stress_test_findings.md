# Stress Test Findings and Current Status

This document aggregates the latest `run_all.sh` matrix produced with:

- `N=20`
- `MAX_CORPUS=1000`
- `RUN_TAG=cap1000`

It covers `NQ`, `SciFact`, and `HotpotQA`, each under:

- `BM25`
- dense retrieval
- `public`
- `entity-counterfactual`
- `entity+value-counterfactual`

All summary files come from the latest `*_20_cap1000*_run.json` outputs in [outputs](/Users/weiyueli/Desktop/rag/expected-answer-rag/outputs). The corresponding `records.jsonl` files contain `20` per-query records for each `BM25` regime, so this note reflects a genuine `N=20` pilot rather than the earlier `max-corpus=200` truncated setting.

## 1. Scope and Caveat

This updated pilot is stronger than the earlier `max-corpus=200` stress test, but it is still not a final experiment:

- it uses only `20` queries per dataset
- it still uses a corpus cap of `1000`, not the full benchmark corpus
- the results are useful for pattern-finding, implementation validation, and method triage, not for final claims

The main change relative to the earlier note is that some older conclusions no longer hold cleanly. In particular, query-preserving concatenation methods are now much stronger than before, including some oracle and masking controls. The current evidence still supports a leakage story, but it is now more specifically a story about the difference between:

- answer-like text used alone
- answer-like text appended to the original query

## 2. Headline Takeaways

- `query2doc_concat` is the strongest and most consistent non-fusion baseline in the latest pilot. Its average `nDCG@10` is `0.883 / 0.651 / 0.690` on `BM25 public / entity-CF / entity+value-CF`, and `0.903 / 0.549 / 0.585` on dense.
- `safe_rrf_v1` is the strongest new SAFE-family method under the harder `BM25 entity+value-counterfactual` regime. Its average there is `0.689`, essentially tied with `query2doc_concat` at `0.690`, and it is better than `safe_rrf_v0` at `0.661`.
- `safe_rrf_v0` is the stronger public-facing SAFE-family method. It is clearly better than `safe_rrf_v1` in `BM25 public` (`0.832` vs `0.818`) and slightly better in dense `entity` (`0.573` vs `0.566`).
- `cf_prompt_query_expansion_rrf` improved enough to become a credible mid-tier method, but it still is not the leading method in any averaged regime. Its best shape is now `BM25 entity+value` (`0.612`) and dense public (`0.866`), not the earlier `NQ-only` story.
- `raw_expected_answer_only` still behaves like the clearest leakage probe. It is competitive in public settings, then collapses sharply under both counterfactual regimes, especially on `BM25`.
- The strongest overall average methods in this pilot are not the new SAFE methods. They are still query-preserving concatenation variants such as `concat_query_raw_expected`, `concat_query_random_span_masking`, and `query2doc_concat`.

## 3. Compact Result Tables

Metric throughout: `nDCG@10`.

### BM25 Public

| Method | NQ | SciFact | HotpotQA |
| --- | :---: | :---: | :---: |
| Query only | 0.653 | 0.893 | 0.798 |
| Raw expected answer | 0.850 | 0.812 | 0.632 |
| HyDE | 0.838 | 0.820 | 0.919 |
| Query2doc | 0.889 | 0.859 | 0.900 |
| GRF | 0.876 | 0.751 | 0.812 |
| Corpus-steered short | 0.634 | 0.811 | 0.741 |
| Query + masked expected | 0.768 | 0.899 | 0.835 |
| Query + answer-constrained | 0.735 | 0.882 | 0.820 |
| RRF(query, answer-constrained) | 0.724 | 0.882 | 0.806 |
| SAFE-RRF v0 | 0.786 | 0.866 | 0.845 |
| SAFE-RRF v1 | 0.732 | 0.884 | 0.838 |
| CF-prompt QE RRF | 0.692 | 0.830 | 0.726 |

### BM25 Entity-Counterfactual

| Method | NQ | SciFact | HotpotQA |
| --- | :---: | :---: | :---: |
| Query only | 0.606 | 0.789 | 0.593 |
| Raw expected answer | 0.465 | 0.475 | 0.347 |
| HyDE | 0.530 | 0.679 | 0.404 |
| Query2doc | 0.615 | 0.760 | 0.579 |
| GRF | 0.447 | 0.636 | 0.278 |
| Corpus-steered short | 0.570 | 0.720 | 0.512 |
| Query + masked expected | 0.605 | 0.807 | 0.638 |
| Query + answer-constrained | 0.622 | 0.802 | 0.581 |
| RRF(query, answer-constrained) | 0.609 | 0.804 | 0.566 |
| SAFE-RRF v0 | 0.635 | 0.785 | 0.601 |
| SAFE-RRF v1 | 0.645 | 0.788 | 0.595 |
| CF-prompt QE RRF | 0.635 | 0.668 | 0.501 |

### BM25 Entity+Value Counterfactual

| Method | NQ | SciFact | HotpotQA |
| --- | :---: | :---: | :---: |
| Query only | 0.584 | 0.811 | 0.604 |
| Raw expected answer | 0.400 | 0.453 | 0.182 |
| HyDE | 0.503 | 0.674 | 0.264 |
| Query2doc | 0.746 | 0.767 | 0.556 |
| GRF | 0.391 | 0.679 | 0.282 |
| Corpus-steered short | 0.571 | 0.685 | 0.546 |
| Query + masked expected | 0.531 | 0.812 | 0.566 |
| Query + answer-constrained | 0.628 | 0.755 | 0.606 |
| RRF(query, answer-constrained) | 0.653 | 0.788 | 0.535 |
| SAFE-RRF v0 | 0.651 | 0.750 | 0.581 |
| SAFE-RRF v1 | 0.643 | 0.799 | 0.625 |
| CF-prompt QE RRF | 0.613 | 0.705 | 0.519 |

### Dense Public

| Method | NQ | SciFact | HotpotQA |
| --- | :---: | :---: | :---: |
| Query only | 0.882 | 0.928 | 0.885 |
| Raw expected answer | 0.881 | 0.895 | 0.680 |
| HyDE | 0.865 | 0.857 | 0.932 |
| Query2doc | 0.926 | 0.851 | 0.932 |
| GRF | 0.908 | 0.829 | 0.895 |
| Corpus-steered short | 0.881 | 0.888 | 0.820 |
| Query + masked expected | 0.828 | 0.938 | 0.872 |
| Query + answer-constrained | 0.824 | 0.904 | 0.768 |
| RRF(query, answer-constrained) | 0.838 | 0.863 | 0.845 |
| SAFE-RRF v0 | 0.937 | 0.897 | 0.841 |
| SAFE-RRF v1 | 0.896 | 0.900 | 0.878 |
| CF-prompt QE RRF | 0.845 | 0.907 | 0.845 |

### Dense Entity-Counterfactual

| Method | NQ | SciFact | HotpotQA |
| --- | :---: | :---: | :---: |
| Query only | 0.664 | 0.688 | 0.363 |
| Raw expected answer | 0.536 | 0.423 | 0.366 |
| HyDE | 0.610 | 0.673 | 0.322 |
| Query2doc | 0.602 | 0.711 | 0.333 |
| GRF | 0.627 | 0.641 | 0.336 |
| Corpus-steered short | 0.672 | 0.532 | 0.378 |
| Query + masked expected | 0.603 | 0.715 | 0.322 |
| Query + answer-constrained | 0.626 | 0.712 | 0.323 |
| RRF(query, answer-constrained) | 0.610 | 0.714 | 0.273 |
| SAFE-RRF v0 | 0.664 | 0.708 | 0.348 |
| SAFE-RRF v1 | 0.657 | 0.684 | 0.356 |
| CF-prompt QE RRF | 0.677 | 0.593 | 0.349 |

### Dense Entity+Value Counterfactual

| Method | NQ | SciFact | HotpotQA |
| --- | :---: | :---: | :---: |
| Query only | 0.628 | 0.720 | 0.411 |
| Raw expected answer | 0.521 | 0.375 | 0.194 |
| HyDE | 0.567 | 0.762 | 0.382 |
| Query2doc | 0.612 | 0.745 | 0.400 |
| GRF | 0.609 | 0.688 | 0.409 |
| Corpus-steered short | 0.611 | 0.641 | 0.399 |
| Query + masked expected | 0.613 | 0.699 | 0.420 |
| Query + answer-constrained | 0.610 | 0.703 | 0.435 |
| RRF(query, answer-constrained) | 0.610 | 0.741 | 0.427 |
| SAFE-RRF v0 | 0.595 | 0.718 | 0.405 |
| SAFE-RRF v1 | 0.602 | 0.735 | 0.410 |
| CF-prompt QE RRF | 0.622 | 0.657 | 0.424 |

## 4. Main Interpretation

### 4.1 Overall Ranking Shift

The new `max-corpus=1000` pilot gives a more complicated picture than the earlier small-corpus pilot:

- `query2doc_concat` is the cleanest strong baseline overall.
- `safe_rrf_v1` is the best new SAFE-family method under the hardest `BM25 entity+value` shift.
- `safe_rrf_v0` is still the better public-setting SAFE fusion.
- `cf_prompt_query_expansion_rrf` improved into the middle tier, but not the top tier.
- the strongest absolute averages still come from query-preserving concatenation methods, not from raw standalone generations.

The key scientific point is now narrower and stronger: the risky behavior is concentrated in standalone answer-like routes, while query-anchored concatenation often remains competitive even when the appended text is noisy or partially contaminated.

### 4.2 Answer-Generation Family

This family includes:

- `raw_expected_answer_only`
- `concat_query_raw_expected`
- `dual_query_raw_expected_rrf`
- `weighted_dual_query_raw_expected_rrf_w0p25`
- `weighted_dual_query_raw_expected_rrf_w0p5`
- `weighted_dual_query_raw_expected_rrf_w0p75`

Findings:

- `raw_expected_answer_only` remains the clearest leakage probe. It drops from `0.765` to `0.429` to `0.345` on `BM25 public / entity-CF / entity+value-CF`.
- `concat_query_raw_expected` is still extremely strong. Its averages are `0.888 / 0.698 / 0.656` on `BM25` and `0.920 / 0.588 / 0.605` on dense.
- `dual_query_raw_expected_rrf` is consistently weaker than direct concatenation.
- The weighted raw-answer RRF variants help somewhat, but none of them beat `concat_query_raw_expected`.
- Among weighted raw-answer RRF variants, `w0p25` is the strongest on `BM25 entity` and `BM25 entity+value`, while `w0p75` is the strongest on dense `entity`.

Interpretation:

- answer text by itself is still risky
- answer text appended to the original query is now a very strong lexical expansion baseline
- this weakens a simple “all answer-like expansion is bad” story and strengthens a more precise “unanchored answer generation is bad” story

### 4.3 Oracle and Post-Hoc Diagnostics

This family includes:

- `gold_answer_only`
- `oracle_answer_masked`
- `concat_query_oracle_answer_masked`
- `post_hoc_gold_removed_expected_answer`
- `concat_query_post_hoc_gold_removed_expected`

Findings:

- In this pilot, `gold_answer_only`, `oracle_answer_masked`, and `post_hoc_gold_removed_expected_answer` are numerically identical to `raw_expected_answer_only`.
- `concat_query_oracle_answer_masked` and `concat_query_post_hoc_gold_removed_expected` are numerically identical to `concat_query_raw_expected`.

Interpretation:

- the current corpora and prompts are producing answer strings whose retrieval behavior is functionally equivalent across these oracle/post-hoc variants
- these diagnostics remain useful, but they are not currently separating distinct mechanisms in this pilot

### 4.4 HyDE, Query2doc, and GRF

This family includes:

- `hyde_doc_only`
- `query2doc_concat`
- `generative_relevance_feedback_concat`

Findings:

- `query2doc_concat` is the strongest overall non-fusion baseline and the cleanest baseline to beat.
- `hyde_doc_only` is strong in public settings but unstable under counterfactual shift, especially on `BM25`.
- `generative_relevance_feedback_concat` has the sharpest `BM25` instability among the three. Its `BM25` averages fall from `0.813` to `0.454` to `0.450`.
- On dense retrieval, `HyDE` and `GRF` recover somewhat and become closer to one another.

Interpretation:

- `query2doc_concat` should remain a mandatory paper baseline
- `HyDE` and `GRF` are useful but currently not stable enough to anchor the main method story

### 4.5 Corpus-Steered Methods

This family includes:

- `corpus_steered_expansion_concat`
- `corpus_steered_short_concat`
- `rrf_query_corpus_steered_short`

Findings:

- `corpus_steered_expansion_concat` is weak almost everywhere. The long expansion version is clearly the wrong form.
- `corpus_steered_short_concat` is materially better than the long version and is one of the more stable middle-tier methods.
- `rrf_query_corpus_steered_short` is usually better than `corpus_steered_short_concat` on `BM25`, but not clearly better on dense.

Interpretation:

- short corpus steering is worth keeping
- long appended corpus text is still too noisy
- if this line of work continues, the short/RRF version is the right base

### 4.6 Masked-Answer Family

This family includes:

- `masked_expected_answer_only`
- `concat_query_masked_expected`
- `dual_query_masked_expected_rrf`
- `rrf_query_masked_expected`
- `weighted_dual_query_masked_expected_rrf_w0p25`
- `weighted_dual_query_masked_expected_rrf_w0p5`
- `weighted_dual_query_masked_expected_rrf_w0p75`

Findings:

- `masked_expected_answer_only` is weak by itself.
- `concat_query_masked_expected` is strong and consistently better than all masked-answer RRF variants.
- `dual_query_masked_expected_rrf` and `rrf_query_masked_expected` are numerically identical in this matrix.
- The weighted masked-answer RRF variants are all mid-tier and do not beat plain concatenation.
- `w0p75` is the best masked-answer weighted RRF on `BM25 entity`, while `w0p5` is slightly better on `BM25 entity+value`.

Interpretation:

- masking alone does not save the standalone route
- the simple query-plus-masked-answer concatenation remains the best version of this family

### 4.7 Constrained Template Family

This family includes:

- `answer_candidate_constrained_template_only`
- `concat_query_answer_candidate_constrained_template`
- `dual_query_answer_candidate_constrained_template_rrf`
- `rrf_query_answer_constrained`
- `weighted_rrf_query_answer_constrained_w0p25`
- `weighted_rrf_query_answer_constrained_w0p5`
- `weighted_rrf_query_answer_constrained_w0p75`

Findings:

- `answer_candidate_constrained_template_only` is respectable but clearly below its concatenated version.
- `concat_query_answer_candidate_constrained_template` is the strongest member of the family on `BM25`.
- `dual_query_answer_candidate_constrained_template_rrf` and `rrf_query_answer_constrained` are numerically identical in this matrix.
- The weighted answer-constrained RRF variants do not beat the simple concatenated version.
- Among weighted answer-constrained RRF variants, `w0p75` is the strongest in `BM25 public` and dense `entity+value`, while `w0p25` and `w0p5` are slightly better in some denser public/CF slices.

Interpretation:

- answer-constrained prompting remains one of the safest paper-facing baselines
- but the best practical form is still direct query concatenation, not weighted RRF

### 4.8 SAFE-RRF and CF-Prompt Methods

This family includes:

- `safe_rrf_v0`
- `safe_rrf_v1`
- `cf_prompt_query_expansion_rrf`

Findings:

- `safe_rrf_v0` is the stronger public SAFE method.
- `safe_rrf_v1` is the stronger counterfactual SAFE method, especially on `BM25 entity+value` where it reaches `0.689` on average.
- `cf_prompt_query_expansion_rrf` is now a legitimate middle-tier method:
  - `BM25 public / entity / entity+value = 0.749 / 0.601 / 0.612`
  - `dense public / entity / entity+value = 0.866 / 0.540 / 0.568`
- `cf_prompt_query_expansion_rrf` still trails `query2doc_concat`, `concat_query_raw_expected`, and most strong concatenation baselines.

Interpretation:

- the fusion half of the SAFE-QE idea is currently stronger than the prompting half
- `safe_rrf_v1` has the best paper shape if the goal is a leakage-aware method
- `cf_prompt_query_expansion_rrf` is no longer failing, but it still is not the lead method

### 4.9 Masking Controls and Neutral Filler Controls

This family includes:

- `random_span_masking`
- `concat_query_random_span_masking`
- `entity_only_masking`
- `concat_query_entity_only_masking`
- `generic_mask_slot`
- `concat_query_generic_mask_slot`
- `length_matched_neutral_filler`

Findings:

- `random_span_masking` alone is weak, but `concat_query_random_span_masking` is one of the strongest methods in the entire matrix.
- `entity_only_masking` and `generic_mask_slot` are poor standalone methods, but their concatenated variants are much stronger.
- `concat_query_entity_only_masking` is notably strong on `BM25 entity`, with an average of `0.698`.
- `concat_query_generic_mask_slot` is a stable middle-tier method across all six averaged settings.
- `length_matched_neutral_filler` is a revealing control:
  - very poor on `BM25`
  - surprisingly strong on dense, where it averages `0.881 / 0.554 / 0.571`

Interpretation:

- many of these gains look less like precise semantic reformulation and more like “query-preserving extra lexical context helps”
- the neutral-filler result suggests dense retrieval is much less sensitive than `BM25` to noisy appended text

### 4.10 Wrong-Answer Controls

This family includes:

- `wrong_answer_only`
- `concat_query_wrong_answer`
- `rrf_query_wrong_answer`
- `wrong_answer_injection`

Findings:

- `wrong_answer_only` is effectively zero everywhere, except a tiny dense public average of `0.028`.
- `concat_query_wrong_answer` and `wrong_answer_injection` are numerically identical in this matrix.
- `concat_query_wrong_answer` is close to `query_only` in every averaged regime.
- `rrf_query_wrong_answer` is worse than both `query_only` and `concat_query_wrong_answer`.

Interpretation:

- the wrong-answer control is still basically clean
- the reason `concat_query_wrong_answer` remains strong is query dominance, not a broken control

## 5. Full All-Method Average Table

Each number below is the average `nDCG@10` over `NQ`, `SciFact`, and `HotpotQA`.

| Method | Family | BM25 Public | BM25 Entity | BM25 E+V | Dense Public | Dense Entity | Dense E+V |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: |
| `query_only` | Core | 0.781 | 0.663 | 0.666 | 0.898 | 0.572 | 0.586 |
| `hyde_doc_only` | Pseudo-doc | 0.859 | 0.538 | 0.480 | 0.885 | 0.535 | 0.570 |
| `query2doc_concat` | Pseudo-doc | 0.883 | 0.651 | 0.690 | 0.903 | 0.549 | 0.585 |
| `generative_relevance_feedback_concat` | Pseudo-doc | 0.813 | 0.454 | 0.450 | 0.877 | 0.535 | 0.569 |
| `corpus_steered_expansion_concat` | Corpus-steered | 0.661 | 0.520 | 0.496 | 0.775 | 0.518 | 0.529 |
| `corpus_steered_short_concat` | Corpus-steered | 0.729 | 0.601 | 0.600 | 0.863 | 0.527 | 0.551 |
| `raw_expected_answer_only` | Answer-generation | 0.765 | 0.429 | 0.345 | 0.818 | 0.442 | 0.363 |
| `concat_query_raw_expected` | Answer-generation | 0.888 | 0.698 | 0.656 | 0.920 | 0.588 | 0.605 |
| `dual_query_raw_expected_rrf` | Answer-generation | 0.786 | 0.637 | 0.621 | 0.887 | 0.575 | 0.555 |
| `masked_expected_answer_only` | Masking | 0.615 | 0.381 | 0.319 | 0.696 | 0.379 | 0.331 |
| `concat_query_masked_expected` | Masking | 0.834 | 0.683 | 0.636 | 0.879 | 0.547 | 0.577 |
| `dual_query_masked_expected_rrf` | Masking | 0.730 | 0.635 | 0.591 | 0.804 | 0.525 | 0.558 |
| `rrf_query_masked_expected` | Masking | 0.730 | 0.635 | 0.591 | 0.804 | 0.525 | 0.558 |
| `answer_candidate_constrained_template_only` | Constrained | 0.754 | 0.561 | 0.565 | 0.801 | 0.481 | 0.561 |
| `concat_query_answer_candidate_constrained_template` | Constrained | 0.812 | 0.668 | 0.663 | 0.832 | 0.554 | 0.582 |
| `dual_query_answer_candidate_constrained_template_rrf` | Constrained | 0.804 | 0.660 | 0.659 | 0.848 | 0.532 | 0.593 |
| `rrf_query_answer_constrained` | Constrained | 0.804 | 0.660 | 0.659 | 0.848 | 0.532 | 0.593 |
| `gold_answer_only` | Oracle/diagnostic | 0.765 | 0.429 | 0.345 | 0.818 | 0.442 | 0.363 |
| `oracle_answer_masked` | Oracle/diagnostic | 0.765 | 0.429 | 0.345 | 0.818 | 0.442 | 0.363 |
| `concat_query_oracle_answer_masked` | Oracle/diagnostic | 0.888 | 0.698 | 0.656 | 0.920 | 0.588 | 0.605 |
| `post_hoc_gold_removed_expected_answer` | Oracle/diagnostic | 0.765 | 0.429 | 0.345 | 0.818 | 0.442 | 0.363 |
| `concat_query_post_hoc_gold_removed_expected` | Oracle/diagnostic | 0.888 | 0.698 | 0.656 | 0.920 | 0.588 | 0.605 |
| `random_span_masking` | Masking control | 0.758 | 0.422 | 0.338 | 0.806 | 0.410 | 0.350 |
| `concat_query_random_span_masking` | Masking control | 0.886 | 0.696 | 0.660 | 0.931 | 0.575 | 0.613 |
| `entity_only_masking` | Masking control | 0.566 | 0.393 | 0.352 | 0.579 | 0.367 | 0.330 |
| `concat_query_entity_only_masking` | Masking control | 0.832 | 0.698 | 0.632 | 0.877 | 0.557 | 0.592 |
| `generic_mask_slot` | Masking control | 0.444 | 0.222 | 0.221 | 0.467 | 0.264 | 0.258 |
| `concat_query_generic_mask_slot` | Masking control | 0.806 | 0.617 | 0.597 | 0.887 | 0.573 | 0.600 |
| `length_matched_neutral_filler` | Neutral filler | 0.372 | 0.406 | 0.410 | 0.881 | 0.554 | 0.571 |
| `wrong_answer_only` | Wrong-answer control | 0.000 | 0.000 | 0.000 | 0.028 | 0.000 | 0.000 |
| `concat_query_wrong_answer` | Wrong-answer control | 0.772 | 0.659 | 0.662 | 0.864 | 0.526 | 0.574 |
| `rrf_query_wrong_answer` | Wrong-answer control | 0.695 | 0.614 | 0.606 | 0.794 | 0.467 | 0.523 |
| `wrong_answer_injection` | Wrong-answer control | 0.772 | 0.659 | 0.662 | 0.864 | 0.526 | 0.574 |
| `rrf_query_corpus_steered_short` | Corpus-steered | 0.784 | 0.664 | 0.661 | 0.887 | 0.542 | 0.577 |
| `safe_rrf_v0` | SAFE-QE fusion | 0.832 | 0.674 | 0.661 | 0.892 | 0.573 | 0.573 |
| `safe_rrf_v1` | SAFE-QE fusion | 0.818 | 0.676 | 0.689 | 0.891 | 0.566 | 0.582 |
| `cf_prompt_query_expansion_rrf` | SAFE-QE prompting | 0.749 | 0.601 | 0.612 | 0.866 | 0.540 | 0.568 |
| `weighted_dual_query_raw_expected_rrf_w0p25` | Weighted RRF | 0.783 | 0.650 | 0.635 | 0.884 | 0.572 | 0.586 |
| `weighted_dual_query_masked_expected_rrf_w0p25` | Weighted RRF | 0.749 | 0.636 | 0.603 | 0.829 | 0.561 | 0.582 |
| `weighted_rrf_query_answer_constrained_w0p25` | Weighted RRF | 0.794 | 0.660 | 0.647 | 0.884 | 0.557 | 0.584 |
| `weighted_dual_query_raw_expected_rrf_w0p5` | Weighted RRF | 0.768 | 0.642 | 0.633 | 0.888 | 0.580 | 0.566 |
| `weighted_dual_query_masked_expected_rrf_w0p5` | Weighted RRF | 0.723 | 0.644 | 0.610 | 0.825 | 0.553 | 0.563 |
| `weighted_rrf_query_answer_constrained_w0p5` | Weighted RRF | 0.796 | 0.658 | 0.651 | 0.876 | 0.555 | 0.588 |
| `weighted_dual_query_raw_expected_rrf_w0p75` | Weighted RRF | 0.778 | 0.645 | 0.623 | 0.897 | 0.591 | 0.557 |
| `weighted_dual_query_masked_expected_rrf_w0p75` | Weighted RRF | 0.719 | 0.646 | 0.600 | 0.814 | 0.552 | 0.569 |
| `weighted_rrf_query_answer_constrained_w0p75` | Weighted RRF | 0.807 | 0.650 | 0.658 | 0.844 | 0.537 | 0.591 |

## 6. Current Recommendation

If the next round has to prioritize only a few method families, the current ordering should be:

1. `query2doc_concat` as the strongest baseline to beat.
2. `safe_rrf_v1` as the strongest leakage-aware method candidate.
3. `safe_rrf_v0` as the stronger public-setting SAFE ablation.
4. `concat_query_answer_candidate_constrained_template` as the safest simple query-preserving baseline.
5. `cf_prompt_query_expansion_rrf` as a promising but still mid-tier prompting method.

The main paper-facing claim supported by this pilot is no longer “masked expected answers are simply safer.” The stronger claim is:

- standalone answer-like generations are still the clearest failure mode
- query-preserving concatenation is unexpectedly robust
- leakage-aware fusion has real promise
- counterfactual prompting is improving, but it is not yet the dominant method
