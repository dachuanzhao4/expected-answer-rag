# Comprehensive Experiment Results

## Outputs Inventory

Result-bearing artifacts currently present in `outputs/`:

- `27` run summaries: `*_run.json`
- `27` per-query record files: `*_records.jsonl`
- `27` logs: `*.log`
- `9` generation caches: `*_cache.json`
- cached counterfactual corpora: `outputs/counterfactual_artifacts/`
- cached embeddings: `outputs/embeddings/`
- Hugging Face cache: `outputs/hf_cache/`

The substantive findings below are derived from the `27` run-summary JSON files. The logs, records, and caches are support artifacts for the same experiment set rather than separate result conditions.

## Experiment Families

There are three completed result families, each spanning `3` datasets (`nq`, `scifact`, `hotpotqa`) and `3` regimes (`public`, `entity`, `entity+value`):

| Family | Files | Retriever | Notes |
|---|---:|---|---|
| Original BM25 matrix | `9` | `bm25` | Main comparison set used in the first consolidated report |
| Dense matrix | `9` | `dense` | Same dataset/regime grid with dense retrieval |
| BM25 RM3 + FAWE follow-up | `9` | `bm25` | Same BM25 grid rerun with `bm25_rm3_query_only`, FAWE controls, and FAWE beta sweep |

Observed evaluated query counts per dataset:

- `nq`: `68`
- `hotpotqa`: `85`
- `scifact`: `100`

All runs use `N=100` and `MAX_CORPUS=2000`.

## Best Method In Every Run

| Family | Dataset | Regime | `query_only` nDCG@10 | Best method | Best nDCG@10 | Delta |
|---|---|---|---:|---|---:|---:|
| BM25 main | `hotpotqa` | public | 0.8019 | `fawe_query2doc_beta0p25` | 0.9059 | +0.1040 |
| BM25 main | `hotpotqa` | entity | 0.5064 | `fawe_answer_constrained_beta0p5` | 0.5260 | +0.0196 |
| BM25 main | `hotpotqa` | entity+value | 0.5100 | `fawe_masked_expected_beta0p25` | 0.5117 | +0.0016 |
| BM25 main | `nq` | public | 0.5450 | `query2doc_concat` | 0.7801 | +0.2352 |
| BM25 main | `nq` | entity | 0.4298 | `fawe_query2doc_beta0p25` | 0.5319 | +0.1022 |
| BM25 main | `nq` | entity+value | 0.4332 | `fawe_query2doc_beta0p25` | 0.5058 | +0.0726 |
| BM25 main | `scifact` | public | 0.7868 | `fawe_query2doc_beta0p25` | 0.8312 | +0.0444 |
| BM25 main | `scifact` | entity | 0.6818 | `fawe_raw_expected_beta0p25` | 0.7069 | +0.0251 |
| BM25 main | `scifact` | entity+value | 0.7144 | `fawe_safe_adaptive_beta` | 0.7232 | +0.0088 |
| Dense | `hotpotqa` | public | 0.9129 | `raw_expected_then_query` | 0.9482 | +0.0352 |
| Dense | `hotpotqa` | entity | 0.2366 | `concat_query_random_span_masking` | 0.2897 | +0.0531 |
| Dense | `hotpotqa` | entity+value | 0.2890 | `raw_expected_then_query` | 0.3241 | +0.0351 |
| Dense | `nq` | public | 0.8266 | `query_plus_shuffled_expected` | 0.8635 | +0.0368 |
| Dense | `nq` | entity | 0.5335 | `concat_query_raw_expected` | 0.5569 | +0.0234 |
| Dense | `nq` | entity+value | 0.4754 | `concat_query_generic_mask_slot` | 0.5269 | +0.0515 |
| Dense | `scifact` | public | 0.8176 | `hyde_doc_only` | 0.8414 | +0.0238 |
| Dense | `scifact` | entity | 0.5318 | `hyde_doc_only` | 0.6259 | +0.0940 |
| Dense | `scifact` | entity+value | 0.5287 | `hyde_doc_only` | 0.6574 | +0.1287 |
| BM25 follow-up | `hotpotqa` | public | 0.8019 | `fawe_query2doc_beta0p25` | 0.9059 | +0.1040 |
| BM25 follow-up | `hotpotqa` | entity | 0.5064 | `fawe_answer_constrained_beta0p5` | 0.5260 | +0.0196 |
| BM25 follow-up | `hotpotqa` | entity+value | 0.5100 | `fawe_raw_expected_beta0p05` | 0.5136 | +0.0035 |
| BM25 follow-up | `nq` | public | 0.5450 | `query2doc_concat` | 0.7801 | +0.2352 |
| BM25 follow-up | `nq` | entity | 0.4298 | `fawe_query2doc_beta0p25` | 0.5319 | +0.1022 |
| BM25 follow-up | `nq` | entity+value | 0.4332 | `fawe_query2doc_beta0p25` | 0.5058 | +0.0726 |
| BM25 follow-up | `scifact` | public | 0.7868 | `fawe_query2doc_beta0p25` | 0.8312 | +0.0444 |
| BM25 follow-up | `scifact` | entity | 0.6818 | `fawe_raw_expected_beta0p5` | 0.7105 | +0.0288 |
| BM25 follow-up | `scifact` | entity+value | 0.7144 | `fawe_query2doc_beta0p05` | 0.7250 | +0.0107 |

Winner counts across all `27` run files:

- `fawe_query2doc_beta0p25`: `8`
- `hyde_doc_only`: `3`
- `fawe_answer_constrained_beta0p5`: `2`
- `raw_expected_then_query`: `2`
- `query2doc_concat`: `2`
- `concat_query_random_span_masking`: `1`
- `concat_query_raw_expected`: `1`
- `concat_query_generic_mask_slot`: `1`
- `query_plus_shuffled_expected`: `1`
- `fawe_masked_expected_beta0p25`: `1`
- `fawe_raw_expected_beta0p25`: `1`
- `fawe_raw_expected_beta0p5`: `1`
- `fawe_raw_expected_beta0p05`: `1`
- `fawe_query2doc_beta0p05`: `1`
- `fawe_safe_adaptive_beta`: `1`

Because the BM25 follow-up family reruns the same `9` dataset/regime settings with extra methods, these counts should be read as file-level wins, not unique-condition wins.

## Cross-Family Averages On Shared Methods

These methods are present in all three families, so they are the cleanest apples-to-apples comparison.

| Family | `query_only` | `raw_expected_answer_only` | `query2doc_concat` | `concat_query_raw_expected` | `safe_rrf_v1` | `fawe_query2doc_beta0p25` | `fawe_safe_adaptive_beta` |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 main | 0.6010 | 0.5768 | 0.6180 | 0.6483 | 0.6239 | 0.6573 | 0.6136 |
| Dense | 0.5725 | 0.5412 | 0.5896 | 0.5909 | 0.5806 | 0.5829 | 0.5736 |
| BM25 follow-up | 0.6010 | 0.5768 | 0.6180 | 0.6483 | 0.6239 | 0.6573 | 0.6136 |

The shared-method averages for BM25 main and BM25 follow-up are identical, which is what we want: the rerun preserved the original BM25 results and only added extra baselines and ablations.

## Regime-Level Averages On Shared Methods

### BM25 main / BM25 follow-up

| Regime | `query_only` | `query2doc_concat` | `concat_query_raw_expected` | `fawe_query2doc_beta0p25` | `fawe_safe_adaptive_beta` | `hyde_doc_only` |
|---|---:|---:|---:|---:|---:|---:|
| public | 0.7112 | 0.8292 | 0.8107 | 0.8385 | 0.7222 | 0.8090 |
| entity | 0.5393 | 0.5214 | 0.5700 | 0.5727 | 0.5584 | 0.4465 |
| entity+value | 0.5525 | 0.5035 | 0.5641 | 0.5608 | 0.5602 | 0.4332 |

### Dense

| Regime | `query_only` | `query2doc_concat` | `concat_query_raw_expected` | `fawe_query2doc_beta0p25` | `fawe_safe_adaptive_beta` | `hyde_doc_only` |
|---|---:|---:|---:|---:|---:|---:|
| public | 0.8524 | 0.8643 | 0.8607 | 0.8616 | 0.8454 | 0.8640 |
| entity | 0.4340 | 0.4567 | 0.4604 | 0.4413 | 0.4359 | 0.4436 |
| entity+value | 0.4310 | 0.4478 | 0.4516 | 0.4457 | 0.4396 | 0.4334 |

## Main Findings Across The Full Outputs Folder

1. **The BM25 claim is still the strongest headline.** On the BM25 runs, FAWE and anchored concatenation methods clearly outperform `query_only` in public retrieval while remaining much more stable than answer-only methods under counterfactual rewrites. `fawe_query2doc_beta0p25` is the strongest overall BM25 method on the shared metrics, averaging **0.6573** across the `9` BM25 conditions.

2. **The dense story is materially different.** Dense retrieval is very strong on public subsets, but counterfactual degradation is much harsher and more uniform. Dense public averages are high (`query_only` **0.8524**, `query2doc_concat` **0.8643**, `hyde_doc_only` **0.8640**), yet dense counterfactual averages collapse to the mid-`0.43` to `0.46` range. That makes the dense results harder to frame as clean “leakage isolation”; they look more like broad representation or alias sensitivity.

3. **Natural Questions remains the clearest leakage-sensitive dataset.** On BM25 public NQ, answer-like expansions are very strong:
   - `query_only`: **0.5450**
   - `raw_expected_answer_only`: **0.7633**
   - `query2doc_concat`: **0.7801**
   - `fawe_query2doc_beta0p25`: **0.7783**

   But once entities are rewritten, the answer-only advantage contracts substantially:
   - entity `raw_expected_answer_only`: **0.4889**
   - entity+value `raw_expected_answer_only`: **0.4855**
   - entity `fawe_query2doc_beta0p25`: **0.5319**
   - entity+value `fawe_query2doc_beta0p25`: **0.5058**

4. **HotpotQA and SciFact split the story.**
   - On **HotpotQA BM25**, public gains are large, but entity and entity+value counterfactual performance is compressed; the practical question is mostly which method avoids hurting too much.
   - On **SciFact BM25**, the public baseline is already strong and several FAWE variants yield smaller but consistent robustness gains.
   - On **SciFact dense**, `hyde_doc_only` is the best dense method in all three regimes, including both counterfactual settings.

5. **The BM25 follow-up reruns mostly confirmed the original BM25 results.** On the `9` BM25 dataset/regime settings, adding RM3, FAWE controls, and the beta sweep changed the winner in only `3` cases:
   - `hotpotqa` entity+value: `fawe_masked_expected_beta0p25` -> `fawe_raw_expected_beta0p05`
   - `scifact` entity: `fawe_raw_expected_beta0p25` -> `fawe_raw_expected_beta0p5`
   - `scifact` entity+value: `fawe_safe_adaptive_beta` -> `fawe_query2doc_beta0p05`

   Everything else stayed unchanged, which increases confidence that the first BM25 matrix was already pointing in the right direction.

6. **RM3 is not a serious competitor in this setup.** In the BM25 follow-up family, `bm25_rm3_query_only` averages **0.5981**, slightly below `query_only` at **0.6010**. It helps on public and entity NQ, but hurts on most of the remaining settings. It is worth keeping as a classical baseline, but not as a main story.

7. **The FAWE beta sweep supports the PI’s robustness intuition.** Public BM25 still favors `fawe_query2doc_beta0p25`, but smaller weights are better under stronger counterfactual shift:
   - public best average: `fawe_query2doc_beta0p25` at **0.8385**
   - entity best average: `fawe_query2doc_beta0p10` at **0.5754**
   - entity+value best average: `fawe_query2doc_beta0p05` at **0.5722**

8. **The FAWE controls support a lexical-enrichment-under-anchoring mechanism.** In the BM25 follow-up runs:
   - `fawe_query_repeated_beta0p25` is exactly baseline on average: **0.6010**
   - `fawe_wrong_answer_beta0p25` is also baseline-like: **0.6009**
   - `fawe_random_terms_from_corpus_beta0p25` and `fawe_idf_matched_random_terms_beta0p25` are near baseline: **0.6018** and **0.6022**
   - `fawe_neutral_filler_beta0p25` is clearly harmful: average delta vs `query_only` is **-0.0989**
   - `fawe_shuffled_expected_beta0p25` exactly matches `fawe_raw_expected_beta0p25` on average: **0.6337**

   The `shuffled_expected == raw_expected` result is especially important. It suggests that much of the FAWE gain does not depend on coherent answer generation; it is consistent with the claim that anchored lexical support is doing much of the work.

## Public-To-Counterfactual Stability Examples

Selected drops from public to counterfactual BM25:

- `nq` BM25 `raw_expected_answer_only`: **0.7633 -> 0.4889 / 0.4855**
- `nq` BM25 `fawe_query2doc_beta0p25`: **0.7783 -> 0.5319 / 0.5058**
- `nq` BM25 `fawe_safe_adaptive_beta`: **0.5516 -> 0.4604 / 0.4464**
- `hotpotqa` BM25 `query_only`: **0.8019 -> 0.5064 / 0.5100**
- `scifact` BM25 `fawe_safe_adaptive_beta`: **0.8045 -> 0.6981 / 0.7232**

Selected drops from public to counterfactual dense:

- `hotpotqa` dense `query_only`: **0.9129 -> 0.2366 / 0.2890**
- `hotpotqa` dense `concat_query_raw_expected`: **0.9346 -> 0.2773 / 0.2997**
- `nq` dense `query_only`: **0.8266 -> 0.5335 / 0.4754**
- `scifact` dense `hyde_doc_only`: **0.8414 -> 0.6259 / 0.6574**

These examples capture the main interpretation: BM25 counterfactual degradation is meaningful but still leaves room to separate brittle answer-like methods from more robust anchored methods, whereas dense degradation often overwhelms that distinction.

## Recommended PI-Facing Summary

The complete `outputs/` folder now tells a consistent story. The original BM25 matrix, the dense matrix, and the BM25 follow-up reruns all point in the same direction: the cleanest and strongest result is still the BM25 counterfactual analysis. FAWE remains the best overall BM25 method family, especially `fawe_query2doc_beta0p25` for public retrieval and smaller beta values for counterfactual robustness. RM3 does not explain away the gains. The new FAWE controls also strengthen the mechanism claim: gains are consistent with anchored lexical enrichment rather than coherent answer generation alone. Dense retrieval is strong on public data but much less clean under counterfactual rewriting, so it should be framed as a weaker and more ambiguous supporting result rather than the main headline.
