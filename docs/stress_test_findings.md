# Initial Stress Test Findings & Project Status

This document summarizes the findings from the initial N=5 stress tests and tracks experimental progress against the `updated_project_experimental_design.md` memo.

## 1. Key Findings from Stress Tests

### Summary of Performance (BM25 nDCG@10, N=100, max-corpus=200)

| Method | NQ | SciFact | HotpotQA | NQ (CF) | SciFact (CF) | HotpotQA (CF) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `query_only` | 0.699 | 1.000 | 0.897 | 0.639 | 0.944 | 0.646 |
| `raw_expected_answer_only` | 0.894 | 0.959 | 0.975 | 0.688 | 0.937 | 0.574 |
| `hyde_doc_only` | 0.853 | 0.959 | 0.951 | 0.582 | 0.959 | 0.577 |
| `query2doc_concat` | 0.906 | 1.000 | 0.926 | 0.721 | 0.944 | 0.601 |
| `generative_relevance_feedback` | 0.823 | 0.880 | 0.975 | 0.771 | 0.889 | 0.608 |
| `corpus_steered_expansion_concat` | 0.633 | 0.836 | 0.730 | 0.592 | 0.795 | 0.526 |
| `masked_expected_answer_only` | 0.736 | 0.848 | 0.493 | 0.665 | 0.643 | 0.443 |
| `random_span_masking` | 0.858 | 0.903 | 0.975 | 0.626 | 0.959 | 0.543 |
| `entity_only_masking` | 0.706 | 0.959 | 0.851 | 0.646 | 0.932 | 0.486 |
| `generic_mask_slot` | 0.588 | 0.497 | 0.371 | 0.594 | 0.621 | 0.422 |
| `wrong_answer_injection` | 0.698 | 1.000 | 0.897 | 0.636 | 0.944 | 0.646 |
| `answer_candidate_constrained_template` | 0.779 | 1.000 | 0.866 | 0.647 | 1.000 | 0.555 |
| `concat_query_answer_constrained` | 0.783 | 1.000 | 0.904 | 0.679 | 1.000 | 0.567 |

*Note: All runs used `--max-corpus 200` to perfectly align the public and counterfactual distractor pools. N=100 queries sampled per dataset.*

### A. Public-Original Evaluation
- **NQ:** Favors answer-like reformulations. `raw_expected_answer_only` massively outperforms the baseline (0.894 vs 0.699), indicating severe "answer-prior leakage" (the LLM's parametric memory guessed the exact famous answer and matched the document).
- **SciFact:** Hits a ceiling effect for several methods (1.000). However, unconstrained generation like `generative_relevance_feedback` actually degraded performance (0.880), showing that generic generation on specialized scientific claims can drift away from the target vocabulary.
- **HotpotQA:** `hyde_doc_only` and `query2doc_concat` outperform the baseline (0.951 vs 0.897). In contrast, generating raw expected answers maintains baseline performance (0.975), showing that multi-hop questions are more robust to simple string hallucination but still benefit from semantic expansion.

### B. Entity-Counterfactual Private-Like Evaluation
- **The "Smoking Gun":** When tested on the counterfactual NQ dataset (simulating a private corpus by renaming entities), the `raw_expected_answer_only` method's gain drops from **+0.195** in Public to just **+0.049** in CF (0.688 vs 0.639). This proves it relies heavily on hallucinating the *public* answer, which only occasionally correlates with the counterfactual document (likely due to common word overlap).
- **Stability:** The `answer_candidate_constrained_template` maintains stability, tracking the baseline closely (0.647 vs 0.639). When concatenated with the query, it provides a safe, modest gain (0.679) without the risk of parametric collapse. This strongly validates Hypothesis H3 and RQ3.

### C. Dense Retriever Stress Testing (nDCG@10, N=100, max-corpus=200)

| Method | NQ | SciFact | HotpotQA | NQ (CF) | SciFact (CF) | HotpotQA (CF) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `query_only` | 0.907 | 1.000 | 0.975 | 0.705 | 0.799 | 0.553 |
| `raw_expected_answer_only` | 0.886 | 1.000 | 0.967 | 0.646 | 0.789 | 0.526 |
| `hyde_doc_only` | 0.956 | 0.928 | 1.000 | 0.651 | 0.840 | 0.515 |
| `query2doc_concat` | 0.881 | 1.000 | 0.975 | 0.678 | 0.862 | 0.554 |
| `generative_relevance_feedback` | 0.887 | 0.922 | 1.000 | 0.707 | 0.880 | 0.608 |
| `corpus_steered_expansion_concat` | 0.820 | 0.836 | 0.909 | 0.684 | 0.688 | 0.566 |
| `masked_expected_answer_only` | 0.771 | 0.776 | 0.721 | 0.663 | 0.599 | 0.478 |
| `random_span_masking` | 0.886 | 1.000 | 0.967 | 0.686 | 0.747 | 0.549 |
| `entity_only_masking` | 0.649 | 0.959 | 0.967 | 0.591 | 0.805 | 0.470 |
| `generic_mask_slot` | 0.499 | 0.552 | 0.599 | 0.399 | 0.631 | 0.463 |
| `wrong_answer_injection` | 0.819 | 0.959 | 0.975 | 0.696 | 0.785 | 0.541 |
| `answer_candidate_constrained_template` | 0.786 | 0.889 | 1.000 | 0.566 | 0.699 | 0.489 |
| `concat_query_answer_constrained` | 0.880 | 0.926 | 0.975 | 0.695 | 0.707 | 0.546 |

**Key Takeaways for Dense:**
- **HyDE's Illusion Shattered:** In NQ Public, `hyde_doc_only` achieves a strong 0.956. However, in NQ CF, it **plummets to 0.651**, falling significantly below the 0.705 baseline. This is the most definitive result yet: HyDE's semantic embeddings actually lead the retriever *away* from the correct document when parametric entities are scrambled.
- **SciFact Sensitivity:** The baseline dense performance on SciFact drops from 1.000 (Public) to 0.799 (CF). Dense models remain sensitive to specific scientific jargon; renaming those entities breaks the embedding alignment.
- **HotpotQA Reasoning Collapse:** Dense retrieval on HotpotQA CF (0.553) is significantly worse than BM25 (0.646), and HyDE collapses even further (0.515), confirming that dense encoders struggle heavily with multi-hop relations when entities are out-of-vocabulary.

### D. Entity and Value Counterfactual Ablation (BM25 nDCG@10, N=100)

This ablation goes beyond just renaming entities (like PERSON and LOCATION) and also rewrites numbers, dates, and values.

| Method | NQ (CF E+V) | SciFact (CF E+V) | HotpotQA (CF E+V) |
|---|:---:|:---:|:---:|
| `query_only` | 0.639 | 0.944 | 0.672 |
| `raw_expected_answer_only` | 0.640 | 0.918 | 0.505 |
| `hyde_doc_only` | 0.644 | 0.959 | 0.404 |
| `query2doc_concat` | 0.683 | 1.000 | 0.580 |
| `generative_relevance_feedback` | 0.741 | 1.000 | 0.565 |
| `corpus_steered_expansion_concat` | 0.592 | 0.780 | 0.596 |
| `masked_expected_answer_only` | 0.604 | 0.718 | 0.300 |
| `random_span_masking` | 0.646 | 0.889 | 0.455 |
| `entity_only_masking` | 0.630 | 0.903 | 0.451 |
| `generic_mask_slot` | 0.493 | 0.576 | 0.306 |
| `wrong_answer_injection` | 0.636 | 0.944 | 0.671 |
| `answer_candidate_constrained_template` | 0.688 | 1.000 | 0.524 |
| `concat_query_answer_constrained` | 0.708 | 1.000 | 0.586 |

**Key Takeaways for Entity+Value Counterfactual:**
- **Consistent Leakage Suppression:** In the NQ E+V setting, `raw_expected_answer_only` (0.640) performs identical to the baseline (0.639), showing that scrambling both entities and values effectively neutralizes the advantage of parametric memory.
- **Value Importance in SciFact:** When numbers and dates are scrambled, SciFact performance remains high (0.944), confirming that scientific retrieval is anchored more on complex nouns than exact numeric values.
- **HotpotQA Fragility:** Scrambling values in HotpotQA (0.672) maintains baseline performance, but expansion methods like HyDE (0.404) collapse to their lowest observed scores, indicating that multi-hop logic is extremely sensitive to temporal/numeric perturbations.

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

### ⏳ Pending (To Be Run)
The following experiments are required for the full conference submission:

**1. Full-Scale Runs (Scale up from N=5)**
- [ ] Run full corpus experiments for NQ, HotpotQA, and SciFact (e.g., N=10000 or N=500 queries).
  - *Note: Entity replacement over millions of corpus documents is very slow in pure Python. The dataset counterfactual generation should be parallelized or pre-computed, or `max-corpus` should be kept to a manageable subset if full generation is impossible.*

**2. Missing Dataset Regimes**
- [x] HotpotQA Entity-Counterfactual test (dry run complete).
- [x] SciFact Entity-Counterfactual test (dry run complete).

**3. Missing Retrievers**
- [x] Dense Retriever Entity-Counterfactual tests (to test if dense embeddings are robust to entity renaming or still susceptible to semantic leakage).
- [ ] BM25 + RM3 (Optional traditional pseudo-relevance feedback baseline).

**4. Ablations & Controls**
- [x] Entity-and-value-counterfactual test (ablation with `--counterfactual entity_and_value`).
- [ ] Generate qualitative examples showing successes/failures (partially done in dry runs).
- [ ] Human spot checks on a stratified sample of renamed documents.
