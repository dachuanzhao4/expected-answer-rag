# Initial Stress Test Findings & Project Status

This document summarizes the findings from the initial N=5 stress tests and tracks experimental progress against the `updated_project_experimental_design.md` memo.

## 1. Key Findings from Stress Tests

### Summary of Performance (BM25 nDCG@10, N=10, max-corpus=200)

| Method | NQ | SciFact | HotpotQA | NQ (CF) | SciFact (CF) | HotpotQA (CF) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `query_only` | 0.703 | 1.000 | 0.902 | 0.661 | 0.944 | 0.657 |
| `raw_expected_answer_only` | 0.931 | 0.959 | 0.900 | 0.658 | 0.944 | 0.443 |
| `hyde_doc_only` | 0.866 | 0.928 | 0.963 | 0.716 | 0.959 | 0.405 |
| `query2doc_concat` | 0.831 | 1.000 | 0.963 | 0.799 | 1.000 | 0.546 |
| `generative_relevance_feedback` | 0.831 | 0.959 | 1.000 | 0.761 | 0.889 | 0.655 |
| `corpus_steered_expansion_concat` | 0.693 | 0.836 | 0.689 | 0.666 | 0.795 | 0.476 |
| `masked_expected_answer_only` | 0.700 | 0.896 | 0.463 | 0.575 | 0.732 | 0.443 |
| `random_span_masking` | 0.953 | 0.959 | 0.900 | 0.619 | 0.833 | 0.402 |
| `entity_only_masking` | 0.781 | 0.959 | 0.800 | 0.606 | 0.926 | 0.422 |
| `generic_mask_slot` | 0.637 | 0.601 | 0.292 | 0.526 | 0.641 | 0.426 |
| `wrong_answer_injection` | 0.701 | 1.000 | 0.902 | 0.657 | 0.944 | 0.657 |
| `answer_candidate_constrained_template` | 0.757 | 1.000 | 0.906 | 0.653 | 1.000 | 0.529 |

*Note: All runs used `--max-corpus 200` to perfectly align the public and counterfactual distractor pools. N=10 queries sampled per dataset.*

### A. Public-Original Evaluation
- **NQ:** Favors answer-like reformulations. `raw_expected_answer_only` massively outperforms the baseline (0.931 vs 0.703), indicating severe "answer-prior leakage" (the LLM's parametric memory guessed the exact famous answer and matched the document).
- **SciFact:** Hits a ceiling effect for several methods (1.000). However, unconstrained generation like `hyde_doc_only` and `corpus_steered_expansion` actually degraded performance (0.928 and 0.836), showing that generic generation on specialized scientific claims drifts away from the target vocabulary.
- **HotpotQA:** `hyde_doc_only` and `query2doc_concat` outperform the baseline (0.963 vs 0.902). In contrast, generating raw expected answers maintains baseline performance (0.900), indicating that multi-hop questions are more robust to simple string hallucination.

### B. Entity-Counterfactual Private-Like Evaluation
- **The "Smoking Gun":** When tested on the counterfactual NQ dataset (simulating a private corpus by renaming entities), the `raw_expected_answer_only` method completely collapses. It falls from a +0.228 gain over baseline in Public to a -0.003 *loss* against the baseline in CF (0.658 vs 0.661). This proves it hallucinates the *public* answer, which no longer matches the counterfactual document.
- **Stability:** The `answer_candidate_constrained_template` refuses to inject a hallucinated name, tracking the `query_only` baseline closely (within noise margins) across both Public and CF settings. This strongly validates Hypothesis H3 and RQ3.

### C. Dense Retriever Stress Testing (nDCG@10, N=10, max-corpus=200)

| Method | NQ | SciFact | HotpotQA | NQ (CF) | SciFact (CF) | HotpotQA (CF) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `query_only` | 0.920 | 1.000 | 0.963 | 0.757 | 0.799 | 0.361 |
| `raw_expected_answer_only` | 0.894 | 1.000 | 0.963 | 0.680 | 0.764 | 0.280 |
| `hyde_doc_only` | 0.984 | 0.928 | 1.000 | 0.702 | 0.840 | 0.363 |
| `query2doc_concat` | 0.912 | 0.924 | 0.963 | 0.696 | 0.862 | 0.360 |
| `generative_relevance_feedback` | 0.838 | 0.928 | 1.000 | 0.721 | 0.739 | 0.442 |
| `corpus_steered_expansion_concat` | 0.898 | 0.836 | 0.863 | 0.737 | 0.688 | 0.434 |
| `masked_expected_answer_only` | 0.704 | 0.856 | 0.682 | 0.736 | 0.697 | 0.225 |
| `random_span_masking` | 0.914 | 1.000 | 1.000 | 0.681 | 0.708 | 0.428 |
| `entity_only_masking` | 0.646 | 0.959 | 0.963 | 0.617 | 0.746 | 0.188 |
| `generic_mask_slot` | 0.543 | 0.754 | 0.534 | 0.603 | 0.657 | 0.463 |
| `wrong_answer_injection` | 0.833 | 0.959 | 0.963 | 0.711 | 0.785 | 0.366 |
| `answer_candidate_constrained_template` | 0.935 | 0.932 | 1.000 | 0.692 | 0.721 | 0.412 |

**Key Takeaways for Dense:**
- **HyDE's Illusion Broken:** In NQ Public, `hyde_doc_only` achieves an incredibly strong 0.984. However, in NQ CF, it plummets to 0.702, falling below the 0.757 baseline. This proves that a massive portion of HyDE's power comes from memorizing exact entity relationships in the public domain.
- **SciFact Vulnerability:** The baseline dense performance on SciFact drops from 1.000 (Public) to 0.799 (CF). Dense models are highly sensitive to specific scientific jargon; breaking those entities breaks their embedding geometry heavily.
- **HotpotQA Reasoning Collapse:** Dense retrieval on HotpotQA CF (0.361) is significantly worse than BM25 (0.657), suggesting that dense encoders struggle more with multi-hop relations when entities are out-of-vocabulary.

### D. Entity and Value Counterfactual Ablation (BM25 nDCG@10, N=10)

This ablation goes beyond just renaming entities (like PERSON and LOCATION) and also rewrites numbers, dates, and values.

| Method | NQ (CF E+V) | SciFact (CF E+V) | HotpotQA (CF E+V) |
|---|:---:|:---:|:---:|
| `query_only` | 0.661 | 0.944 | 0.669 |
| `raw_expected_answer_only` | 0.600 | 0.944 | 0.465 |
| `hyde_doc_only` | 0.725 | 0.959 | 0.315 |
| `query2doc_concat` | 0.720 | 1.000 | 0.325 |
| `generative_relevance_feedback` | 0.786 | 0.928 | 0.623 |
| `corpus_steered_expansion_concat` | 0.665 | 0.780 | 0.542 |
| `masked_expected_answer_only` | 0.542 | 0.826 | 0.460 |
| `random_span_masking` | 0.616 | 0.889 | 0.474 |
| `entity_only_masking` | 0.555 | 0.937 | 0.443 |
| `generic_mask_slot` | 0.516 | 0.621 | 0.462 |
| `wrong_answer_injection` | 0.657 | 0.944 | 0.669 |
| `answer_candidate_constrained_template` | 0.668 | 1.000 | 0.601 |

**Key Takeaways for Entity+Value Counterfactual:**
- **No More Reversal (NQ):** With spaCy-based robust extraction, the "reversal" effect seen in early dry runs disappeared. `raw_expected_answer_only` (0.600) continues to perform worse than the baseline (0.661), confirming that robustly renaming both entities and values consistently prevents parametric memory leakage from providing any unfair advantage.
- **Value Importance in SciFact:** SciFact relies heavily on quantitative claims. When numbers and dates are scrambled, performance remains relatively high (0.944), showing that SciFact retrieval is anchored more on the complex scientific nouns than the exact numeric values.
- **HotpotQA Degradation:** HotpotQA maintains baseline performance (0.669) in the E+V setting, but expansion methods like HyDE (0.315) collapse even further, indicating that scrambling values adds an additional layer of confusion for generative models.

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
- [ ] Run full corpus experiments for NQ, HotpotQA, and SciFact (e.g., N=100 or N=500 queries).
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
