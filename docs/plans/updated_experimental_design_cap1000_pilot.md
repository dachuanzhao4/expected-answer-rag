# Updated Experimental Design and Paper Positioning Memo

**Project:** Leakage-aware LLM query expansion for private-like RAG  
**Update basis:** User-provided `N=20`, `max-corpus=1000`, `cap1000` pilot across NQ, SciFact, and HotpotQA under BM25 and dense retrieval.  
**Status:** Pilot interpretation and revised plan only. No experiments were run as part of this review.

---

## 1. Executive Summary

The latest pilot changes the paper story in an important way.

The earlier framing was:

> Raw answer-like generations leak public answers; leakage-aware reformulations should be safer and may outperform raw expansion under private-like counterfactuals.

The revised framing should be:

> LLM-generated retrieval expansions fail most severely when they are used as standalone retrieval objects. Query anchoring -- preserving the original user query and appending generated text -- often makes even noisy or partially contaminated expansions surprisingly robust. Counterfactual evaluation reveals that the key confound is not simply "answer text is bad," but the interaction among answer-bearing generation, query anchoring, corpus shift, and retriever type.

This is a stronger and more nuanced top-conference story. It avoids overclaiming that all answer-like expansion is unsafe, while preserving the core scientific insight that public benchmark gains can be inflated or distorted by LLM priors.

### Current headline observations from the `cap1000` pilot

Across the six averaged regimes (`BM25 public`, `BM25 entity-CF`, `BM25 entity+value-CF`, `dense public`, `dense entity-CF`, `dense entity+value-CF`):

| Method | Average nDCG@10 | Interpretation |
|---|---:|---|
| `concat_query_random_span_masking` | 0.727 | Strongest all-regime average, but should remain a control because it is not a principled method. |
| `concat_query_raw_expected` | 0.726 | Strongest practical project-owned expansion in this pilot. |
| `query2doc_concat` | 0.710 | Strongest external prior-work baseline. This is the baseline to beat. |
| `safe_rrf_v1` | 0.704 | Best SAFE-family method overall and strongest under `BM25 entity+value-CF`. |
| `safe_rrf_v0` | 0.701 | Stronger public-facing SAFE variant. |
| `query_only` | 0.694 | Still a very strong anchor and essential control. |

The strongest paper-facing finding is no longer that a particular masking method wins. It is that **query anchoring rescues generated expansion**, while **standalone answer-like and pseudo-document routes are counterfactually unstable**.

---

## 2. Important Correction to Method Taxonomy

The project must separate external baselines from project-owned methods. Otherwise reviewers may misunderstand which methods are prior work and which are original diagnostics or interventions.

### 2.1 External prior-work baselines

These are baselines that should be described as prior methods or standard retrieval/fusion tools:

| Method in repo | Paper role | Notes |
|---|---|---|
| `query_only` | Core no-expansion baseline | Not a prior LLM-QE method, but mandatory. |
| `BM25` | Sparse retriever baseline | Standard lexical retriever. |
| Dense `query_only` | Dense retriever baseline | Use at least one fixed zero-shot dense encoder. |
| `BM25 + RM3` | Traditional PRF baseline | Still pending; include if feasible. |
| `hyde_doc_only` | Prior LLM pseudo-document baseline | HyDE-style hypothetical document retrieval. |
| `query2doc_concat` | Prior LLM query-expansion baseline | Currently the strongest external baseline. |
| `generative_relevance_feedback_concat` | Prior LLM feedback baseline | Important because it represents generated feedback models. |
| `corpus_steered_*` | Prior-inspired CSQE baseline | Needs faithful implementation; current short version is better than long version. |
| RRF | Fusion primitive | RRF itself is prior work; SAFE weighting/gating is project-owned. |

### 2.2 Project-owned diagnostic/proposed methods

These should be clearly labeled as our own design space, not prior baselines:

| Family | Methods | Paper role |
|---|---|---|
| Answer-prediction expansion | `raw_expected_answer_only`, `concat_query_raw_expected`, raw-answer RRF variants | Main diagnostic and potential simple method. |
| Masked expected answer | `masked_expected_answer_only`, `concat_query_masked_expected`, weighted/RRF variants | Leakage-control family. |
| Answer-constrained templates | `answer_candidate_constrained_template_only`, concatenated/RRF variants | Main leakage-aware project-owned reformulation. |
| SAFE fusion | `safe_rrf_v0`, `safe_rrf_v1` | Main project-owned adaptive/fusion method. |
| CF-prompt QE | `cf_prompt_query_expansion_rrf` | Project-owned counterfactual prompting method; currently mid-tier. |
| Controls | wrong-answer, neutral filler, random masking, entity-only masking, generic slot, oracle/post-hoc variants | Diagnostic controls, mostly appendix. |

This distinction should be explicit in the paper's method section and result tables.

---

## 3. Revised Paper Thesis

### 3.1 One-sentence thesis

Public-benchmark gains from LLM query expansion conflate answer-prior leakage with query anchoring; entity/value-counterfactual evaluation shows that unanchored generated answers and pseudo-documents are unstable, while query-preserving and leakage-aware expansions provide a more reliable retrieval interface for private-like RAG.

### 3.2 Better title candidates

1. **Anchors, Not Just Answers: Counterfactual Evaluation of LLM Query Expansion for Retrieval**
2. **When Generated Answers Hurt Retrieval: Entity-Counterfactual Stress Tests for LLM Query Expansion**
3. **Query Anchoring Mitigates Answer Leakage in LLM-Based Retrieval Reformulation**
4. **Leakage, Anchoring, and Fusion in LLM-Based Query Expansion for Private-Like RAG**
5. **Beyond HyDE: Counterfactual Evidence for Prior-Induced Bias in LLM Query Expansion**

### 3.3 Recommended main claim

The paper should not claim:

> All answer-like expansion is unsafe.

The current pilot supports a narrower and better claim:

> Standalone answer-like generations are highly vulnerable to prior-induced failure under counterfactual private-like shifts, but query-anchored generated expansions are much more robust. Therefore, leakage-aware evaluation must distinguish generated-text-only retrieval from query-preserving integration.

### 3.4 Contribution framing

The top-conference contribution should be framed as a combined **evaluation + analysis + method** paper:

1. **Evaluation protocol:** entity- and entity+value-counterfactual retrieval evaluation for private-like RAG.
2. **Mechanistic analysis:** decomposition of LLM query expansion into unanchored generation, query-anchored concatenation, fusion, and corpus-steered expansion.
3. **Diagnostic finding:** raw standalone answer/pseudo-doc routes show strong counterfactual instability; concatenation often rescues performance.
4. **Method family:** SAFE-QE and answer-candidate-constrained expansions as leakage-aware routes on the robustness/accuracy Pareto frontier.
5. **Controls:** wrong-answer, neutral-filler, random-mask, oracle-mask, and post-hoc removal controls showing how much improvement comes from anchoring rather than answer correctness.

---

## 4. Updated Interpretation of the Latest Pilot

### 4.1 The result is not simply "answer leakage bad"

`raw_expected_answer_only` is still the cleanest leakage probe:

| Regime | `query_only` | `raw_expected_answer_only` | Delta |
|---|---:|---:|---:|
| BM25 public | 0.781 | 0.765 | -0.016 |
| BM25 entity-CF | 0.663 | 0.429 | -0.234 |
| BM25 entity+value-CF | 0.666 | 0.345 | -0.321 |
| Dense public | 0.898 | 0.818 | -0.080 |
| Dense entity-CF | 0.572 | 0.442 | -0.130 |
| Dense entity+value-CF | 0.586 | 0.363 | -0.223 |

This supports the leakage-failure thesis for standalone answer-style retrieval.

However, `concat_query_raw_expected` is extremely strong:

| Regime | `query_only` | `concat_query_raw_expected` | Delta |
|---|---:|---:|---:|
| BM25 public | 0.781 | 0.888 | +0.107 |
| BM25 entity-CF | 0.663 | 0.698 | +0.035 |
| BM25 entity+value-CF | 0.666 | 0.656 | -0.010 |
| Dense public | 0.898 | 0.920 | +0.022 |
| Dense entity-CF | 0.572 | 0.588 | +0.016 |
| Dense entity+value-CF | 0.586 | 0.605 | +0.019 |

This implies that query anchoring is doing a lot of the robustness work. The paper should explicitly analyze this.

### 4.2 Query2doc is the main external baseline to beat

`query2doc_concat` is the strongest external prior-work baseline:

| Regime | `query2doc_concat` |
|---|---:|
| BM25 public | 0.883 |
| BM25 entity-CF | 0.651 |
| BM25 entity+value-CF | 0.690 |
| Dense public | 0.903 |
| Dense entity-CF | 0.549 |
| Dense entity+value-CF | 0.585 |

The project-owned `concat_query_raw_expected` beats `query2doc_concat` in five of six averaged settings, but loses in `BM25 entity+value-CF`:

| Regime | `concat_query_raw_expected` | `query2doc_concat` | Difference |
|---|---:|---:|---:|
| BM25 public | 0.888 | 0.883 | +0.005 |
| BM25 entity-CF | 0.698 | 0.651 | +0.047 |
| BM25 entity+value-CF | 0.656 | 0.690 | -0.034 |
| Dense public | 0.920 | 0.903 | +0.017 |
| Dense entity-CF | 0.588 | 0.549 | +0.039 |
| Dense entity+value-CF | 0.605 | 0.585 | +0.020 |

This suggests that a concise answer-shaped expansion appended to the original query may be a competitive practical method, but its mechanism needs to be explained carefully.

### 4.3 SAFE-RRF is not the best absolute method yet

`safe_rrf_v1` is valuable because it is robust, not because it dominates all regimes.

| Regime | `safe_rrf_v1` | Best observed average in regime | Gap |
|---|---:|---:|---:|
| BM25 public | 0.818 | 0.888 | -0.070 |
| BM25 entity-CF | 0.676 | 0.698 | -0.022 |
| BM25 entity+value-CF | 0.689 | 0.690 | -0.001 |
| Dense public | 0.891 | 0.931 | -0.040 |
| Dense entity-CF | 0.566 | 0.591 | -0.025 |
| Dense entity+value-CF | 0.582 | 0.613 | -0.031 |

Recommended interpretation:

> SAFE-RRF is the best current leakage-aware fusion candidate and is nearly tied with the best prior baseline in the hardest BM25 entity+value setting, but it does not yet beat query-anchored concatenation overall.

### 4.4 The random-mask and neutral-filler controls are now central diagnostics

`concat_query_random_span_masking` is the strongest all-regime average. This should not be promoted as the main method. Instead, it should be used to show that:

1. preserving the original query is critical;
2. adding semantically adjacent generated text can help even when the answer span is corrupted;
3. the gains may be due to lexical/semantic enrichment and query anchoring, not necessarily correct answer prediction.

The dense `length_matched_neutral_filler` result is suspicious or at least highly diagnostic. It is close to dense `query_only`, which may mean the dense encoder is mostly preserving the original query, ignoring appended filler, truncating appended text, or that the method is accidentally using the original query. This must be audited before drawing conclusions from dense concatenation controls.

---

## 5. New Core Concepts and Metrics

The paper should add the following metrics.

### 5.1 Excess instability over query-only

For method `m`, retriever `r`, and counterfactual regime `c`:

```text
excess_instability(m, r, c)
= [score_public(m, r) - score_cf_c(m, r)]
  - [score_public(query_only, r) - score_cf_c(query_only, r)]
```

This controls for the fact that counterfactual renaming itself can make retrieval harder.

From the pilot:

| Method | BM25 excess entity | BM25 excess E+V | Dense excess entity | Dense excess E+V |
|---|---:|---:|---:|---:|
| `raw_expected_answer_only` | +0.218 | +0.305 | +0.050 | +0.143 |
| `hyde_doc_only` | +0.203 | +0.264 | +0.024 | +0.003 |
| `query2doc_concat` | +0.114 | +0.078 | +0.028 | +0.006 |
| `concat_query_raw_expected` | +0.072 | +0.117 | +0.006 | +0.003 |
| `concat_query_answer_constrained` | +0.026 | +0.034 | -0.048 | -0.062 |
| `safe_rrf_v1` | +0.024 | +0.014 | -0.001 | -0.003 |
| `cf_prompt_query_expansion_rrf` | +0.030 | +0.022 | ~0.000 | -0.014 |

This is where SAFE-RRF and constrained templates look strongest: not peak score, but low excess instability.

### 5.2 Concatenation rescue delta

For generated text `g`:

```text
concat_rescue(g) = score(concat(q, g)) - score(g_only)
```

This directly quantifies how much the original query rescues a generated expansion.

Pilot examples:

| Pair | BM25 public | BM25 entity | BM25 E+V | Dense public | Dense entity | Dense E+V |
|---|---:|---:|---:|---:|---:|---:|
| raw expected -> query + raw expected | +0.123 | +0.269 | +0.311 | +0.102 | +0.146 | +0.242 |
| masked expected -> query + masked expected | +0.219 | +0.302 | +0.317 | +0.183 | +0.168 | +0.246 |
| answer-constrained -> query + constrained | +0.058 | +0.107 | +0.098 | +0.031 | +0.073 | +0.021 |
| random span -> query + random span | +0.128 | +0.274 | +0.322 | +0.125 | +0.165 | +0.263 |
| entity-only -> query + entity-only | +0.266 | +0.305 | +0.280 | +0.298 | +0.190 | +0.262 |
| generic slot -> query + generic slot | +0.362 | +0.395 | +0.376 | +0.420 | +0.309 | +0.342 |

This should become one of the main analysis figures.

### 5.3 Query dominance controls

Because concatenation is so strong, the paper must include controls that test whether the appended text matters or whether the original query dominates.

Add the following controls:

1. `query_repeated`: retrieve with `q + q`.
2. `query_repeated_length_matched`: repeat/truncate the query to match expansion length.
3. `query_plus_shuffled_expansion`: append the same tokens from the expansion in random order.
4. `query_plus_wrong_answer`: already implemented; keep.
5. `query_plus_neutral_filler`: ensure this is actually query + filler; if it is filler-only, rename it.
6. `fielded_query_expansion`: score query and expansion separately instead of raw concatenation.

### 5.4 Robustness-adjusted score

Report both absolute score and a robustness-adjusted score:

```text
robust_score(m) = average_counterfactual_score(m) - lambda * max(0, average_excess_instability(m))
```

Use `lambda = 0.5` for analysis only, not as the sole headline metric.

The goal is not to hide lower absolute performance; it is to show which methods lie on the accuracy/robustness Pareto frontier.

---

## 6. Revised Method Set for the Next Full Run

The current all-method matrix is too large for a main paper. Keep the full matrix in an appendix, but prioritize a smaller set.

### 6.1 Main-table methods

| Category | Method | Why keep |
|---|---|---|
| Core | `query_only` | Mandatory anchor. |
| Prior baseline | `hyde_doc_only` | High-risk pseudo-doc baseline. |
| Prior baseline | `query2doc_concat` | Strongest external baseline. |
| Prior baseline | `generative_relevance_feedback_concat` | Important prior LLM feedback method. |
| Prior baseline | `corpus_steered_short_concat` or faithful CSQE | Corpus-grounded LLM expansion baseline. |
| Traditional baseline | `BM25 + RM3` | Needed to compare against non-LLM PRF. |
| Project method | `concat_query_raw_expected` | Strong project-owned answer-prediction expansion. |
| Project method | `concat_query_masked_expected` | Leakage-reduced answer-prediction expansion. |
| Project method | `concat_query_answer_candidate_constrained_template` | Safer constrained expansion. |
| Project method | `safe_rrf_v1` | Main leakage-aware fusion method. |
| Project ablation | `safe_rrf_v0` | Less conservative SAFE variant. |
| Project ablation | `cf_prompt_query_expansion_rrf` | Counterfactual prompting method. |
| Diagnostic control | `raw_expected_answer_only` | Leakage probe. |
| Diagnostic control | `concat_query_random_span_masking` | Strong control showing query anchoring. |
| Diagnostic control | `concat_query_wrong_answer` | Query-dominance and wrong-answer control. |
| Diagnostic control | `query_repeated` / `query_plus_neutral_filler` | Needed to interpret concatenation gains. |

### 6.2 Appendix-only methods

Move these out of main tables unless they become important after full-scale runs:

- standalone `masked_expected_answer_only`
- standalone `generic_mask_slot`
- standalone `entity_only_masking`
- long `corpus_steered_expansion_concat`
- duplicate RRF aliases if numerically identical
- all weighted RRF variants except the best pre-registered one per family
- oracle/post-hoc variants unless their implementation is fixed and they produce distinct text

---

## 7. Recommended New Method Variant: Fielded Anchor-Weighted Expansion

The latest pilot suggests that raw concatenation works better than many RRF variants. This means the next method to try should not be another RRF variant; it should be a **fielded or weighted concatenation substitute** that preserves the benefit of query anchoring while preventing the expansion from dominating.

### 7.1 Method name

**FAWE: Fielded Anchor-Weighted Expansion**

### 7.2 Sparse retrieval version

Instead of retrieving with one concatenated string:

```text
q + expansion
```

score documents as:

```text
score(d) = BM25(q, d) + beta * BM25(expansion, d)
```

Tune or sweep:

```text
beta in {0.1, 0.25, 0.5, 0.75, 1.0}
```

This is different from RRF because it fuses scores before rank truncation and avoids raw concatenation length-normalization artifacts.

Recommended variants:

- `fawe_raw_expected_beta025`
- `fawe_masked_expected_beta025`
- `fawe_answer_constrained_beta05`
- `fawe_query2doc_beta025`
- `fawe_safe_adaptive_beta`

### 7.3 Dense retrieval version

Compute embeddings separately:

```text
e = normalize( normalize(E(q)) + beta * normalize(E(expansion)) )
```

Sweep:

```text
beta in {0.1, 0.25, 0.5, 0.75, 1.0}
```

This tests whether dense concatenation is actually using the appended expansion or mostly encoding the original query.

### 7.4 Why this should be tried

The pilot indicates:

- raw concatenation beats RRF variants;
- query anchoring matters more than standalone generated quality;
- wrong-answer concatenation remains close to query-only;
- SAFE-RRF is robust but not peak-performing.

FAWE directly targets this pattern and may beat both `query2doc_concat` and `concat_query_raw_expected` by preserving the useful expansion signal without allowing generated answer priors to dominate.

---

## 8. Required Sanity Checks Before More Scale-Up

The following checks are now mandatory because several exact numerical identities and dense-control results are suspicious.

### 8.1 Retrieval text audit

For 20 random examples per dataset/regime/method, dump:

```json
{
  "query_id": "...",
  "original_query": "...",
  "counterfactual_query": "...",
  "method": "...",
  "generated_text": "...",
  "final_retrieval_text": "...",
  "retriever_input_hash": "...",
  "top10_docids": [...],
  "relevant_docids": [...]
}
```

Check especially:

- `gold_answer_only`
- `oracle_answer_masked`
- `post_hoc_gold_removed_expected_answer`
- `concat_query_oracle_answer_masked`
- `concat_query_post_hoc_gold_removed_expected`
- `length_matched_neutral_filler`
- dense concatenation methods

### 8.2 Exact-identity audit

Several methods are numerically identical in the pilot. Some identities may be expected aliases, but others are red flags.

Audit these pairs:

| Pair | Status |
|---|---|
| `gold_answer_only` vs `raw_expected_answer_only` | Suspicious unless generated answer exactly equals gold answer for all examples. |
| `oracle_answer_masked` vs `raw_expected_answer_only` | Suspicious. Masking should change retrieval text. |
| `post_hoc_gold_removed_expected_answer` vs `raw_expected_answer_only` | Suspicious. Post-hoc removal should change retrieval text when answer appears. |
| `concat_query_oracle_answer_masked` vs `concat_query_raw_expected` | Suspicious. |
| `concat_query_post_hoc_gold_removed_expected` vs `concat_query_raw_expected` | Suspicious. |
| `dual_query_*_rrf` vs `rrf_query_*` | May be duplicate implementation; if so, consolidate. |
| `concat_query_wrong_answer` vs `wrong_answer_injection` | May be duplicate by design; if so, rename one. |

### 8.3 Dense encoder audit

The dense `length_matched_neutral_filler` result is close to dense `query_only`. Verify:

1. whether this method includes the original query;
2. whether appended text is truncated due to max input length;
3. whether the dense encoder pools mostly from the query prefix;
4. whether all dense methods are actually receiving the intended retrieval text;
5. whether generated text after the query has lower influence than generated text before the query.

Add a position ablation:

- `query_then_expansion`
- `expansion_then_query`
- `query_only`
- `expansion_only`
- `query_plus_neutral_filler`
- `neutral_filler_plus_query`

---

## 9. Revised Experimental Design

### 9.1 Dataset/regime matrix

Keep the three-dataset structure:

1. **Natural Questions**: main answer-prior leakage test.
2. **HotpotQA**: multi-hop and bridge-entity stress test.
3. **SciFact**: scientific evidence and entailment-oriented stress test.

For each dataset, run:

1. public/original benchmark;
2. entity-counterfactual benchmark;
3. entity+value-counterfactual benchmark;
4. optional naturalistic-alias counterfactual benchmark;
5. optional coded-alias stress counterfactual benchmark.

### 9.2 Corpus size plan

The `max-corpus=1000` pilot is useful but not submission-level.

Recommended scale-up path:

1. **Debug run:** `N=20`, `max-corpus=1000`, all methods, multiple seeds.
2. **Pilot run:** `N=100`, `max-corpus=5000` or `10000`, reduced method set.
3. **Main run:** `N=500+` or all available queries, largest feasible corpus.
4. **Final run:** full corpus for SciFact; full or large hard-negative corpus for NQ/HotpotQA.

If full corpus counterfactual rewriting is too slow, use a fixed qrels-preserving candidate pool:

- include all relevant documents for sampled queries;
- include top-k BM25 distractors from public query-only retrieval;
- include random distractors stratified by document length and entity density;
- use the same document pool for public and counterfactual conditions;
- report pool construction transparently.

### 9.3 Retriever plan

Required:

- BM25
- one dense retriever
- BM25 + RM3, if feasible

Recommended additions:

- one stronger dense retriever or reranker for robustness checks;
- a cross-encoder reranking analysis over top-100 to distinguish first-stage retrieval failure from rank-order failure.

### 9.4 Metrics

Main retrieval metrics:

- `nDCG@10`
- `Recall@10`
- `Recall@20`
- `MRR@10` for QA-style datasets

Leakage/anchoring metrics:

- exact answer leakage rate in generated text;
- alias-answer leakage rate under counterfactual mapping;
- unsupported entity injection rate;
- unsupported value/date injection rate;
- excess instability over query-only;
- concatenation rescue delta;
- query-dominance score;
- expansion-only vs query+expansion gap;
- public-to-counterfactual delta by answer type.

Efficiency metrics:

- generated token count;
- retrieval input length;
- generation cost;
- indexing/retrieval latency if materially affected.

Efficiency matters because a concise expected-answer expansion may be competitive with longer pseudo-document methods at lower generation cost.

### 9.5 Statistical analysis

Use paired tests because all methods evaluate the same queries.

Required:

1. paired bootstrap confidence intervals for `nDCG@10`, `Recall@20`, and excess instability;
2. paired randomization or permutation tests for primary method comparisons;
3. per-query win/tie/loss counts;
4. effect sizes, not only p-values;
5. multiple-comparison handling for main claims, e.g. Holm correction over pre-registered primary comparisons.

Primary comparisons should be pre-registered before the main run:

1. `concat_query_raw_expected` vs `query2doc_concat`;
2. `safe_rrf_v1` vs `query2doc_concat`;
3. `safe_rrf_v1` vs `concat_query_raw_expected`;
4. `concat_query_answer_constrained` vs `concat_query_raw_expected`;
5. `raw_expected_answer_only` vs `query_only` under counterfactual regimes;
6. `generated_only` vs `query+generated` within each expansion family.

---

## 10. Revised Main Tables and Figures

### Table 1: Method taxonomy

Separate:

- external prior-work baselines;
- project-owned proposed methods;
- project-owned diagnostics/controls.

### Table 2: Public benchmark performance

Show only main-table methods.

### Table 3: Entity-counterfactual performance

Show only main-table methods.

### Table 4: Entity+value-counterfactual performance

Show only main-table methods.

### Table 5: Excess instability over query-only

This is now a central table.

### Table 6: Concatenation rescue analysis

Compare:

- generated-only;
- query+generated;
- query+wrong answer;
- query+neutral filler;
- query+random mask;
- query+constrained template.

### Table 7: Accuracy/robustness Pareto frontier

Show methods on axes:

- average counterfactual nDCG@10;
- excess instability;
- generated token cost.

### Figure 1: Public vs counterfactual scatterplot

Each point is a method. Highlight:

- external baselines;
- project methods;
- controls.

### Figure 2: Concatenation rescue bar chart

Show `score(query+generated) - score(generated-only)` by family.

### Figure 3: Leakage rate vs excess instability

Show whether generated answer/entity/value leakage predicts counterfactual instability.

### Figure 4: Case studies

Use examples showing:

1. standalone raw answer succeeds publicly but fails counterfactually;
2. query+raw answer survives because query anchors the relevant document;
3. query+wrong answer behaves like query-only;
4. SAFE-RRF downweights a risky generated route;
5. query2doc beats constrained methods in some entity+value cases.

---

## 11. Updated Claim Boundaries

### 11.1 Safe claims if full-scale confirms the pilot

1. Standalone answer-like LLM expansions are counterfactually unstable.
2. Query-preserving concatenation is much more robust than generated-text-only retrieval.
3. Public-only evaluation can exaggerate or obscure the failure modes of LLM query expansion.
4. Query anchoring is a key variable that prior evaluations do not sufficiently isolate.
5. SAFE-RRF and answer-constrained templates reduce excess instability, even when they do not maximize public benchmark scores.
6. Query2doc remains a strong prior baseline; project-owned methods should be compared against it directly.

### 11.2 Claims not currently supported

1. SAFE-QE beats the best baseline overall.
2. Counterfactual prompting is the best method.
3. Masking alone is enough to solve leakage.
4. All answer-like expansion is harmful.
5. Public benchmark gains are mostly due to leakage.
6. Dense retrieval collapse necessarily means OOV entity failure, without dense encoder audits.

### 11.3 Revised abstract-level claim

A suitable abstract claim would be:

> We show that LLM-generated query expansions exhibit sharply different behavior depending on whether generated text is used alone or anchored to the original query. On entity- and value-counterfactual private-like benchmarks, standalone answer-like and pseudo-document expansions are unstable, while query-preserving expansion often remains competitive. We introduce leakage- and anchoring-sensitive diagnostics, plus SAFE-QE, a conservative fusion family that trades peak public performance for lower counterfactual instability.

---

## 12. Updated Top-Conference Framing

### 12.1 Best framing

This should be framed as:

> a counterfactual evaluation and method paper about leakage, anchoring, and robustness in LLM-assisted retrieval.

Not as:

> a paper that simply proposes a new query expansion method and wins all metrics.

The method results are currently not strong enough for a pure "new method beats all baselines" paper. The evaluation/analysis contribution is much stronger.

### 12.2 Why this can still be top-conference-worthy

The project has a credible top-conference shape because it addresses a real hidden confound:

- LLM query expansion methods are widely used;
- public QA benchmarks are likely to trigger parametric priors;
- private-domain RAG cannot assume those priors are correct;
- current evaluation often does not distinguish answer leakage from useful query reformulation;
- your counterfactual protocol directly tests this distinction.

The surprising new result -- that query anchoring often rescues noisy expansions -- makes the paper more interesting, not less. It means the final paper can contribute a more general theory:

> LLM query expansion works through at least three mechanisms: answer-prior injection, semantic/lexical enrichment, and query anchoring. These mechanisms are conflated in public-only evaluation.

### 12.3 Best venue fit

The most natural venues are:

- **SIGIR / WWW / CIKM** if framed as an information retrieval evaluation and query expansion paper;
- **ACL / EMNLP / NAACL** if framed as LLM/RAG evaluation and retrieval augmentation;
- **NeurIPS Datasets & Benchmarks / ICLR** only if the counterfactual benchmark and robustness methodology become rigorous, scalable, and broadly reusable.

For a top general AI conference, the paper needs:

1. a formal problem definition;
2. a reusable benchmark construction protocol;
3. a clear metric suite;
4. a reduced, well-justified method set;
5. full-scale experiments;
6. statistically supported claims;
7. evidence that the benchmark reveals failures not visible in public-only evaluation.

---

## 13. Immediate Action Plan

### Step 1: Audit before scaling

Do not scale until the exact-identity and dense-control issues are resolved.

Required outputs:

- method input hashes;
- example retrieval strings;
- duplicate-method audit;
- dense truncation/position audit;
- counterfactual qrel preservation checks.

### Step 2: Reduce the method matrix

Freeze a main method list of about 15 methods and move the rest to appendix.

### Step 3: Add query-dominance controls

Implement:

- `query_repeated`
- `query_repeated_length_matched`
- `query_plus_neutral_filler`
- `query_plus_shuffled_expansion`
- `fielded_anchor_weighted_expansion`

### Step 4: Implement FAWE

This is the most promising next method because the pilot suggests weighted anchoring may outperform both raw concatenation and RRF.

### Step 5: Run multi-seed pilot

Before large scale, run:

- `N=100`
- `max-corpus=5000` or `10000`
- 3 random seeds
- reduced method list

### Step 6: Main run

Run full or largest feasible corpus with:

- `N=500+` or all queries;
- public/entity/entity+value regimes;
- BM25 and dense;
- BM25+RM3 if feasible;
- bootstrap CIs and paired tests.

---

## 14. Recommended Paper Outline

1. **Introduction**
   - LLM query expansion is powerful but evaluated mostly on public benchmarks.
   - Public benchmarks conflate answer priors, enrichment, and anchoring.
   - Private-like RAG needs a different evaluation lens.

2. **Related Work**
   - LLM query expansion: HyDE, Query2doc, GRF, CSQE.
   - Fusion and PRF: RRF, BM25+RM3.
   - Contamination and counterfactual evaluation.
   - RAG/private-domain robustness.

3. **Problem Formulation**
   - Define generated expansion, answer leakage, unsupported injection, query anchoring, counterfactual instability.

4. **Entity/Value-Counterfactual Benchmark**
   - Construction, alias tables, qrel preservation, validation checks.

5. **Methods**
   - External baselines.
   - Project-owned expansion families.
   - SAFE-QE.
   - FAWE if implemented.

6. **Experimental Setup**
   - Datasets, retrievers, metrics, prompts, statistical tests.

7. **Results**
   - Public performance.
   - Counterfactual performance.
   - Excess instability.
   - Concatenation rescue.
   - Leakage buckets.

8. **Analysis**
   - Query anchoring vs answer injection.
   - Retriever-specific behavior.
   - Per-dataset behavior.
   - Case studies.

9. **Limitations**
   - Artificiality of counterfactual aliases.
   - Corpus cap or full-corpus constraints.
   - Dependence on NER/alias quality.
   - Generated text and prompting variance.

10. **Conclusion**
   - Public-only gains do not isolate reformulation quality.
   - Query anchoring is a critical design and evaluation variable.
   - Counterfactual private-like evaluation is necessary for reliable RAG retrieval methods.

---

## 15. References to Keep in the Literature Review

- Gao et al. **Precise Zero-Shot Dense Retrieval without Relevance Labels**. ACL 2023. HyDE. https://aclanthology.org/2023.acl-long.99/
- Wang et al. **Query2doc: Query Expansion with Large Language Models**. EMNLP 2023. https://aclanthology.org/2023.emnlp-main.585/
- Mackie et al. **Generative Relevance Feedback with Large Language Models**. SIGIR 2023. https://arxiv.org/abs/2304.13157
- Lei et al. **Corpus-Steered Query Expansion with Large Language Models**. EACL 2024. https://aclanthology.org/2024.eacl-short.34/
- Yoon et al. **Hypothetical Documents or Knowledge Leakage? Rethinking LLM-based Query Expansion**. Findings of ACL 2025. https://aclanthology.org/2025.findings-acl.980/
- Thakur et al. **BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models**. NeurIPS Datasets and Benchmarks 2021. https://openreview.net/forum?id=wCu6T5xFjeJ
- Cormack et al. **Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods**. SIGIR 2009. https://dl.acm.org/doi/10.1145/1571941.1572114
- Xu et al. **Benchmark Data Contamination of Large Language Models: A Survey**. 2024. https://arxiv.org/abs/2406.04244
- Longpre et al. / related contamination and counterfactual robustness work should be added if the paper expands the contamination section.

---

## 16. Final Recommendation

Pivot the paper from:

> leakage-aware query expansion method beats baselines

Toward:

> counterfactual evaluation reveals that LLM query expansion gains depend critically on query anchoring; standalone generated answers and pseudo-documents are unstable, while query-preserving and leakage-aware integration methods form a better design space for private-like RAG.

This framing matches the latest data, respects prior work, and gives reviewers a more precise scientific contribution.
