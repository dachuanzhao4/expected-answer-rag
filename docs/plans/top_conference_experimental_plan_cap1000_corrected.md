# Updated Experimental Design Plan After Corrected `cap1000` Pilot

**Project:** Leakage-aware LLM query expansion for retrieval and private-like RAG  
**Status:** Corrected `N=20`, `MAX_CORPUS=1000`, `METHOD_PROFILE=main` pilot reviewed  
**Retrievers in pilot:** BM25 and `BAAI/bge-base-en-v1.5` dense retrieval  
**Datasets in pilot:** Natural Questions, SciFact, HotpotQA  
**Regimes in pilot:** public, entity-counterfactual, entity+value-counterfactual  

This memo supersedes the prior cap1000 pilot plan. The corrected dense rerun changes the method story: the paper should now center on **query anchoring, fielded expansion weighting, and counterfactual leakage evaluation**, not on masked expected answers as the main intervention.

---

## 1. Executive Summary

The latest corrected pilot supports a more mature and top-conference-ready framing:

> Public benchmark gains from LLM query expansion conflate at least three mechanisms: answer-prior leakage, lexical/semantic enrichment, and query anchoring. Entity/value-counterfactual evaluation separates these mechanisms. Standalone answer-like generation is highly unstable, while query-preserving and fielded expansion methods are substantially more robust. We propose **FAWE**, a fielded anchor-weighted expansion method, and use leakage-aware counterfactual evaluation to characterize when and why LLM expansions help retrieval.

The strongest current method result is:

- `fawe_query2doc_beta0p25` has the best all-regime average in the corrected pilot: **0.730 nDCG@10**.
- It improves over the strongest external prior-work baseline, `query2doc_concat`, which averages **0.710**.
- It also improves over `query_only`, which averages **0.694**.
- However, the difference is still from a small `N=20` pilot and must not be claimed as conclusive until full-scale runs and paired significance tests are complete.

The strongest current scientific result is:

- `raw_expected_answer_only` averages only **0.527**, despite being competitive in public settings, because it collapses under counterfactual evaluation.
- Query-anchored methods such as `concat_query_raw_expected`, `query_plus_shuffled_expected`, and FAWE variants remain strong.
- This shows that the failure mode is not simply “generated text is bad” or “answers are bad.” The failure mode is **unanchored generated answer-like retrieval**, especially when public parametric priors no longer match the corpus.

---

## 2. Updated Paper Positioning

### Recommended thesis

> LLM-generated retrieval expansions often help because they enrich and anchor the query, but public-only evaluation can mistake answer-prior leakage for genuine reformulation quality. Entity/value-counterfactual retrieval evaluation exposes this confound. Fielded anchor-weighted expansion preserves the original query while limiting expansion dominance, yielding stronger and more robust retrieval than naive pseudo-document concatenation in pilot experiments.

### Recommended title candidates

1. **Anchored or Leaked? Counterfactual Evaluation of LLM Query Expansion for Retrieval**
2. **Fielded Anchor-Weighted Expansion for Leakage-Aware Retrieval**
3. **When Query Expansion Knows Too Much: Counterfactual Evaluation of LLM-Assisted Retrieval**
4. **Disentangling Answer Priors, Query Anchoring, and Lexical Enrichment in LLM Query Expansion**
5. **FAWE: Fielded Anchor-Weighted Expansion for Robust LLM-Assisted Retrieval**

### Current best venue fit

- **SIGIR / WWW / KDD:** strongest if the paper emphasizes retrieval methodology, query expansion, fusion, and rigorous IR evaluation.
- **ACL / EMNLP / NAACL:** strongest if the paper emphasizes LLM/RAG evaluation, benchmark contamination, counterfactual private-like evaluation, and error analysis.
- **ICLR:** possible if the counterfactual evaluation and mechanism analysis become broad enough, but the current project is more naturally an IR/NLP retrieval paper.

---

## 3. Important Scope Boundary: Prior-Work Baselines vs Project Methods

This distinction must be explicit in the paper. Only a small subset of methods are external prior-work baselines; most methods in the matrix are project-designed methods, diagnostics, or controls.

### External prior-work baselines

| Method | Role in paper |
|---|---|
| `query_only` | No-expansion baseline. |
| BM25 | Sparse retrieval baseline. |
| BM25 + RM3 | Classical pseudo-relevance feedback baseline; still pending. |
| Dense `query_only` | Dense no-expansion baseline. |
| `hyde_doc_only` | HyDE-style hypothetical-document baseline. |
| `query2doc_concat` | Main external LLM query-expansion baseline to beat. |
| `generative_relevance_feedback_concat` | External-style generative relevance-feedback baseline. |
| `corpus_steered_short_concat` / CSQE-style variants | Corpus-steered expansion baseline. |
| RRF | Prior rank-fusion primitive, not a new contribution by itself. |

### Project-designed methods and diagnostics

| Method family | Role in paper |
|---|---|
| `raw_expected_answer_only` | Diagnostic leakage probe. |
| `concat_query_raw_expected` | Strong project-designed answer-anchored expansion baseline. |
| `masked_expected` variants | Leakage-control ablations; most should move to appendix. |
| `answer_candidate_constrained_template` variants | Safer answer-candidate-constrained project method. |
| `safe_rrf_v0`, `safe_rrf_v1` | Leakage-aware fusion methods; robustness-focused. |
| `cf_prompt_query_expansion_rrf` | Counterfactual-prompted query expansion; promising but not lead method yet. |
| `fawe_*` | Main proposed method family. |
| `query_plus_shuffled_expected` | Key query-anchoring / lexical-enrichment control. |
| `query_repeated`, `query_plus_neutral_filler`, `concat_query_wrong_answer` | Query-dominance and negative controls. |
| Oracle/post-hoc variants | Currently exact aliases; fix or remove. |

This should appear as a method taxonomy table in the submission. It avoids two reviewer problems:

1. Accidentally presenting external baselines as new.
2. Understating that several strong methods are project-designed methods, not prior baselines.

---

## 4. Corrected Pilot Results: What Changed

### 4.1 Main ranking

The corrected pilot changes the lead method from SAFE-RRF to FAWE.

| Method | All-regime avg. nDCG@10 | Avg. gain vs `query_only` | Excess instability vs `query_only` |
|---|---:|---:|---:|
| `fawe_query2doc_beta0p25` | **0.730** | **+0.035** | +0.048 |
| `concat_query_raw_expected` | 0.726 | +0.032 | +0.050 |
| `fawe_raw_expected_beta0p25` | 0.724 | +0.030 | +0.023 |
| `query2doc_concat` | 0.710 | +0.016 | +0.056 |
| `fawe_safe_adaptive_beta` | 0.705 | +0.010 | **+0.000** |
| `safe_rrf_v1` | 0.704 | +0.009 | +0.008 |
| `query_only` | 0.694 | 0.000 | 0.000 |
| `concat_query_answer_candidate_constrained_template` | 0.685 | -0.009 | **-0.013** |
| `raw_expected_answer_only` | 0.527 | -0.167 | +0.179 |

Interpretation:

- `fawe_query2doc_beta0p25` is the best **accuracy** method in the corrected pilot.
- `fawe_safe_adaptive_beta`, `safe_rrf_v1`, and `concat_query_answer_candidate_constrained_template` are better **robustness/Pareto** methods.
- `raw_expected_answer_only` remains the clearest leakage/failure probe.
- The paper should not claim one method dominates all criteria. It should show an **accuracy-robustness tradeoff**.

### 4.2 Main empirical mechanism

The strongest mechanism result is the concatenation rescue audit:

| Pair | BM25 public | BM25 entity | BM25 E+V | Dense public | Dense entity | Dense E+V |
|---|---:|---:|---:|---:|---:|---:|
| raw expected -> query + raw expected | +0.124 | +0.269 | +0.311 | +0.101 | +0.147 | +0.241 |
| masked expected -> query + masked expected | +0.219 | +0.302 | +0.318 | +0.183 | +0.168 | +0.246 |
| constrained -> query + constrained | +0.058 | +0.107 | +0.098 | +0.031 | +0.073 | +0.022 |
| random span -> query + random span | +0.128 | +0.274 | +0.322 | +0.126 | +0.165 | +0.264 |
| entity-only -> query + entity-only | +0.266 | +0.305 | +0.281 | +0.297 | +0.190 | +0.262 |
| generic slot -> query + generic slot | +0.362 | +0.396 | +0.376 | +0.420 | +0.309 | +0.343 |

This should be a central figure/table. It shows that the original query is not just extra context; it is a strong stabilizing anchor.

### 4.3 Query-dominance controls

The current query-dominance controls are very important:

- `query_repeated` is essentially a null control.
- `query_plus_shuffled_expected` is extremely strong, almost tied with `concat_query_raw_expected`.
- `concat_query_wrong_answer` stays close to `query_only`.
- `query_plus_neutral_filler` is catastrophic for BM25 but only mildly harmful for dense retrieval.

This supports a refined claim:

> Correct answer prediction is not the only source of retrieval gains. Much of the benefit comes from query anchoring plus lexical/semantic enrichment. However, unanchored answer-like text remains highly vulnerable to counterfactual shift.

---

## 5. Revised Contributions

The paper should claim four contributions.

### Contribution 1: Counterfactual leakage evaluation for retrieval

A reproducible evaluation protocol that constructs entity-counterfactual and entity+value-counterfactual retrieval settings while preserving:

- document evidence structure,
- query-document relevance labels,
- answer-bearing evidence in the corpus,
- entity/value type consistency.

This evaluation is not just a robustness perturbation. It targets the mismatch between public parametric priors and private-style corpora.

### Contribution 2: Mechanistic decomposition of LLM query expansion gains

The paper should explicitly separate:

1. **answer-prior leakage**: generated text contains public-answer information;
2. **lexical/semantic enrichment**: generated text adds useful related terms;
3. **query anchoring**: original query terms prevent retrieval drift;
4. **integration effects**: generated text used alone, concatenated, fused, or fielded.

This is now the strongest scientific contribution.

### Contribution 3: FAWE, a fielded anchor-weighted expansion method

FAWE scores the original query and generated expansion as separate fields, rather than blindly concatenating them.

For sparse retrieval:

```text
score_FAWE(d | q, g) = score_BM25(q, d) + beta * score_BM25(g, d)
```

For dense retrieval:

```text
score_FAWE(d | q, g) = cos(E(q), E(d)) + beta * cos(E(g), E(d))
```

or equivalently, when implemented by vector fusion:

```text
z = normalize(normalize(E(q)) + beta * normalize(E(g)))
score(d) = cos(z, E(d))
```

The current best pilot configuration is:

```text
fawe_query2doc_beta0p25
```

Recommended initial beta grid:

```text
beta in {0.05, 0.10, 0.25, 0.50, 0.75, 1.00}
```

Tune beta on a held-out development split, not on the test split.

### Contribution 4: Leakage-aware robustness analysis

Report not only retrieval quality but also leakage sensitivity:

```text
excess_instability(method)
= instability(method) - instability(query_only)
```

where:

```text
instability(method)
= average over counterfactual regimes of public_score(method) - counterfactual_score(method)
```

This controls for the fact that counterfactual renaming makes retrieval harder even for `query_only`.

---

## 6. Revised Research Questions

### RQ1. Which LLM expansion methods collapse when public entities and values are counterfactually renamed?

Expected answer from current pilot:

- standalone answer-like and standalone pseudo-document routes are most vulnerable;
- raw expected-answer-only is the cleanest diagnostic probe;
- HyDE and GRF are unstable under BM25 counterfactual settings.

### RQ2. How much of the benefit comes from answer correctness versus lexical enrichment and query anchoring?

Expected answer from current pilot:

- `query_plus_shuffled_expected` being near `concat_query_raw_expected` suggests answer correctness is not the only source of gains;
- `query_repeated` being null rules out simple query repetition;
- neutral filler being harmful for BM25 shows that not all appended text helps.

### RQ3. Does fielded weighting improve over naive concatenation?

Expected answer from current pilot:

- `fawe_query2doc_beta0p25` improves over `query2doc_concat` in the averaged pilot;
- `fawe_raw_expected_beta0p25` improves robustness relative to naive raw expected-answer concatenation;
- larger runs are needed to test significance.

### RQ4. Can leakage-aware adaptive weighting reduce instability without losing too much accuracy?

Expected answer from current pilot:

- `fawe_safe_adaptive_beta` has nearly query-only-level excess instability and modest gain over query-only;
- `safe_rrf_v1` is also stable-looking but not best absolute;
- adaptive weighting is promising as a Pareto/robustness method, not yet as the peak-accuracy method.

### RQ5. Do sparse and dense retrievers exhibit different sensitivity to appended text and counterfactual aliases?

Expected answer from current pilot:

- BM25 is highly sensitive to neutral filler and lexical swamping;
- dense retrieval is less damaged by filler but not immune to appended text or counterfactual renaming;
- final analysis should separate sparse and dense conclusions.

---

## 7. Revised Hypotheses

### H1. Standalone answer-like expansions are counterfactually unstable.

Supported by pilot. `raw_expected_answer_only` collapses sharply under entity and entity+value counterfactual settings.

### H2. Query anchoring is a major mechanism behind robust LLM query expansion.

Supported by pilot. Query-preserving concatenation rescues weak generated text by large margins.

### H3. Correct answer prediction is not necessary for many query-anchored gains.

Supported by pilot. `query_plus_shuffled_expected` is close to `concat_query_raw_expected`, implying lexical/semantic enrichment and anchoring explain a substantial portion of the gain.

### H4. Fielded anchor-weighted expansion can outperform naive pseudo-document concatenation.

Promising but not final. `fawe_query2doc_beta0p25` is currently stronger than `query2doc_concat` across the averaged pilot.

### H5. Leakage-aware adaptive methods are more stable but may trade off peak accuracy.

Supported by pilot. `fawe_safe_adaptive_beta` and `safe_rrf_v1` have good robustness shape but do not dominate absolute performance.

---

## 8. Main Method: FAWE

### 8.1 Motivation

Naive concatenation mixes the original query and generated text into a single retrieval string. This creates two problems:

1. The generated expansion can dominate the original query.
2. BM25 length normalization and term-frequency effects can behave unpredictably when long or noisy expansions are appended.

The corrected pilot shows that query anchoring is critical. FAWE makes anchoring explicit by scoring the original query and generated expansion separately.

### 8.2 FAWE variants to keep

| Variant | Status | Reason |
|---|---|---|
| `fawe_query2doc_beta0p25` | Primary method | Best all-regime pilot average and cleanest comparison to external Query2doc. |
| `fawe_raw_expected_beta0p25` | Secondary project method | Strong dense counterfactual performance; tests fielded answer-style expansion. |
| `fawe_masked_expected_beta0p25` | Ablation | Tests leakage-reduced answer-style expansion. |
| `fawe_answer_constrained_beta0p5` | Ablation / safe method | Tests answer-candidate-constrained expansion. |
| `fawe_safe_adaptive_beta` | Robustness method | Best robustness story but not best peak score. |

### 8.3 FAWE controls to add before scale-up

These are necessary because `query_plus_shuffled_expected` is strong.

| Control | Purpose |
|---|---|
| `fawe_shuffled_expected_beta0p25` | Tests whether FAWE gains require coherent generated text. |
| `fawe_wrong_answer_beta0p25` | Tests whether wrong concrete answer candidates hurt under fielded weighting. |
| `fawe_neutral_filler_beta0p25` | Tests whether FAWE is robust to irrelevant expansion text. |
| `fawe_query_repeated_beta0p25` | Null control for fielded scoring. |
| `fawe_random_terms_from_corpus_beta0p25` | Tests whether random corpus vocabulary helps. |
| `fawe_idf_matched_random_terms_beta0p25` | Stronger lexical confound control. |

### 8.4 Beta tuning protocol

Do not tune beta on the same queries used for final evaluation.

Recommended protocol:

1. Split each dataset into `dev` and `test` query sets.
2. Tune beta on `dev` using the public + counterfactual regimes.
3. Freeze beta per expansion family.
4. Evaluate once on held-out `test`.
5. Report both fixed-beta and tuned-beta variants.

Recommended beta grid:

```text
{0.05, 0.10, 0.25, 0.50, 0.75, 1.00}
```

Recommended tuned families:

```text
fawe_query2doc
fawe_raw_expected
fawe_masked_expected
fawe_answer_constrained
fawe_safe_adaptive
```

---

## 9. Methods to Include in the Main Paper

The main tables should be compact. Too many project-designed variants will distract reviewers.

### 9.1 Main methods table

| Category | Method |
|---|---|
| No expansion | `query_only` |
| External baseline | `query2doc_concat` |
| External baseline | `hyde_doc_only` |
| External baseline | `generative_relevance_feedback_concat` |
| External baseline | `corpus_steered_short_concat` |
| Classical PRF | `BM25 + RM3` |
| Leakage probe | `raw_expected_answer_only` |
| Strong project baseline | `concat_query_raw_expected` |
| Safe project baseline | `concat_query_answer_candidate_constrained_template` |
| Main proposed method | `fawe_query2doc_beta0p25` or tuned `fawe_query2doc` |
| Proposed answer-style FAWE | `fawe_raw_expected_beta0p25` |
| Robust proposed method | `fawe_safe_adaptive_beta` |
| Robust fusion | `safe_rrf_v1` |
| Counterfactual prompting | `cf_prompt_query_expansion_rrf` |
| Query anchoring control | `query_plus_shuffled_expected` |
| Negative control | `concat_query_wrong_answer` |
| Null control | `query_repeated` |
| Filler control | `query_plus_neutral_filler` |

### 9.2 Appendix-only methods

Move these to appendix unless they become central in full-scale results:

- `masked_expected_answer_only`
- `random_span_masking`
- `entity_only_masking`
- `generic_mask_slot`
- all standalone masking controls
- most weighted RRF variants
- duplicate oracle/post-hoc aliases

### 9.3 Methods to fix or remove

The duplicate audit shows exact aliases. Do not include these as separate methods unless fixed:

- `gold_answer_only` vs `raw_expected_answer_only`
- `oracle_answer_masked` vs `raw_expected_answer_only`
- `post_hoc_gold_removed_expected_answer` vs `raw_expected_answer_only`
- `concat_query_oracle_answer_masked` vs `concat_query_raw_expected`
- `concat_query_post_hoc_gold_removed_expected` vs `concat_query_raw_expected`
- `concat_query_wrong_answer` vs `wrong_answer_injection`

Recommended action:

- If these are design-equivalent aliases, consolidate and state that they were removed.
- If they were intended to be distinct, fix the transformation and verify retrieval-string hashes before large runs.

---

## 10. Benchmark Construction Plan

### 10.1 Regimes

Keep three main regimes:

1. **Public:** original benchmark.
2. **Entity-counterfactual:** rename named entities with type-preserving aliases.
3. **Entity+value-counterfactual:** rename entities and sensitive values such as dates, years, quantities, and numeric answer-bearing values.

### 10.2 Alias regimes

Add two alias styles:

| Alias style | Use |
|---|---|
| Naturalistic aliases | Main condition. Reduces artificial dense-retrieval artifacts. |
| Coded aliases | Stress-test condition. Preserves stronger privacy-like scrambling. |

Naturalistic aliases are important because coded aliases can introduce distribution shift that reviewers may mistake for the main effect.

### 10.3 Validation checks

Before final runs, every counterfactual benchmark must pass:

- one-to-one alias mapping;
- consistency across queries, corpus, qrels, and answer metadata;
- no unmapped original answer strings in counterfactual query text;
- answer-bearing evidence still appears in at least one relevant document;
- qrels are unchanged except for document IDs if indexing changes;
- entity type is preserved;
- value type is preserved;
- random sample human audit.

Recommended human audit:

```text
30 examples per dataset x counterfactual regime
= 3 datasets x 2 regimes x 30 = 180 audited examples
```

Audit labels:

- alias consistency;
- grammatical plausibility;
- answer evidence preserved;
- query remains answerable;
- no original public answer leak.

---

## 11. Dataset Plan

### 11.1 Required datasets

Keep the three current datasets:

| Dataset | Role |
|---|---|
| Natural Questions | Entity-centric public-prior leakage benchmark. |
| HotpotQA | Multi-hop / bridge-entity leakage benchmark. |
| SciFact | Scientific evidence retrieval and evidence-entailment benchmark. |

### 11.2 Optional additional datasets

Add only if compute and implementation budget allow:

| Dataset | Reason |
|---|---|
| FiQA | Finance-oriented domain shift; useful for private-domain motivation. |
| TREC-COVID | Specialized scientific/medical retrieval; strong corpus-grounding test. |
| FEVER | Fact verification; direct comparison to leakage findings in prior LLM-QE leakage work. |
| A truly private or semi-private corpus | Strongest external-validity addition, if publishable. |

### 11.3 Query count targets

Pilot results are not enough. Recommended scale:

| Stage | Queries per dataset | Corpus |
|---|---:|---|
| Debug | 20 | cap 1000 |
| Development | 200-500 | large fixed pool or full corpus |
| Main test | 500-2000, depending on dataset | full corpus if feasible |
| Final robustness | all available eval queries where feasible | full corpus |

For a top-tier submission, use at least:

```text
>= 500 queries per dataset
full corpus, or a documented fixed hard-negative pool of >= 50k documents
```

If full counterfactual corpus rewriting is too slow, precompute the counterfactual corpus once per dataset and store versioned artifacts.

---

## 12. Retrieval Backends

### 12.1 Required

1. **BM25**
2. **BM25 + RM3**
3. **Dense retrieval with `BAAI/bge-base-en-v1.5`**

### 12.2 Recommended if resources allow

4. **One dense retriever with a different training lineage**, e.g. Contriever or E5.
5. **One reranking setting**, e.g. BM25 top-100 plus cross-encoder reranker, only if the paper expands toward end-to-end RAG.

### 12.3 Sparse and dense reporting

Do not average sparse and dense as the only headline. Report:

- BM25 public/entity/entity+value;
- dense public/entity/entity+value;
- all-regime aggregate only as a compact summary;
- per-dataset tables in appendix.

The pilot shows sparse and dense behave differently; reviewers will want both separated.

---

## 13. Metrics

### 13.1 Retrieval metrics

Primary:

- `nDCG@10`
- `Recall@20`

Secondary:

- `MRR@10` for QA-style datasets;
- `Recall@100` for RAG pipeline utility;
- `nDCG@5` if top-rank sensitivity matters.

### 13.2 Leakage and mechanism metrics

Add these as first-class metrics:

#### Exact answer leakage rate

```text
percentage of generations containing exact gold answer string or alias
```

#### Alias answer leakage rate

```text
percentage containing counterfactual answer alias or original public answer string
```

#### Unsupported entity/value injection rate

```text
percentage of generated entities/values absent from query and absent from top-k query-only retrieved documents
```

#### Public-answer carryover rate

```text
percentage of counterfactual generations that still mention the original public answer
```

#### Query anchor retention

```text
fraction of important query anchors preserved in generated expansion
```

#### Concatenation rescue

```text
score(query + generation) - score(generation_only)
```

#### Fielded rescue

```text
score_FAWE(query, generation) - score(generation_only)
```

#### Excess counterfactual instability

```text
[public - counterfactual]_method - [public - counterfactual]_query_only
```

#### Accuracy-robustness Pareto score

Optional scalar summary:

```text
pareto_score = average_nDCG - lambda * max(0, excess_instability)
```

Use this only as an analysis aid, not the main metric.

---

## 14. Statistical Testing

Use paired tests because all methods are evaluated on the same queries.

Required:

1. Paired bootstrap confidence intervals for `nDCG@10` and `Recall@20`.
2. Paired randomization or permutation tests for key comparisons.
3. Per-query win/tie/loss counts.
4. Multiple-comparison control for main method comparisons.

Primary comparisons:

```text
fawe_query2doc vs query2doc_concat
fawe_query2doc vs query_only
fawe_query2doc vs concat_query_raw_expected
fawe_raw_expected vs concat_query_raw_expected
fawe_safe_adaptive_beta vs safe_rrf_v1
raw_expected_answer_only vs concat_query_raw_expected
query_plus_shuffled_expected vs concat_query_raw_expected
```

Report significance separately for:

- BM25 public;
- BM25 entity-CF;
- BM25 entity+value-CF;
- dense public;
- dense entity-CF;
- dense entity+value-CF.

Avoid claiming superiority from the all-regime average alone.

---

## 15. Main Tables and Figures

### Table 1: Main method taxonomy

Columns:

- method;
- prior-work baseline or project method;
- generated text type;
- integration mode;
- answer candidates allowed;
- leakage risk.

### Table 2: Main retrieval performance

Rows:

- `query_only`
- `query2doc_concat`
- `hyde_doc_only`
- `generative_relevance_feedback_concat`
- `corpus_steered_short_concat`
- `raw_expected_answer_only`
- `concat_query_raw_expected`
- `fawe_query2doc`
- `fawe_raw_expected`
- `fawe_safe_adaptive`
- `safe_rrf_v1`
- `concat_query_answer_candidate_constrained_template`
- controls

Columns:

- BM25 public;
- BM25 entity;
- BM25 entity+value;
- dense public;
- dense entity;
- dense entity+value.

### Table 3: Excess instability and Pareto analysis

Rows: main methods.  
Columns:

- average public score;
- average counterfactual score;
- raw drop;
- query-only-adjusted excess instability;
- all-regime average.

### Table 4: Mechanism controls

Rows:

- `query_only`
- `query_repeated`
- `query_repeated_length_matched`
- `query_plus_neutral_filler`
- `neutral_filler_plus_query`
- `concat_query_wrong_answer`
- `query_plus_shuffled_expected`
- `concat_query_raw_expected`

Purpose: show what kind of appended text helps or hurts.

### Table 5: FAWE ablations

Rows:

- `query2doc_concat`
- `fawe_query2doc_beta0p05`
- `fawe_query2doc_beta0p10`
- `fawe_query2doc_beta0p25`
- `fawe_query2doc_beta0p50`
- `fawe_query2doc_beta0p75`
- `fawe_query2doc_beta1p00`
- tuned beta

Repeat for raw expected and constrained expansions in appendix.

### Figure 1: Method map

A conceptual diagram showing:

```text
original query q
  -> generated expansion g(q)
  -> integration mode: alone / concat / RRF / FAWE
  -> retrieval
  -> public vs counterfactual evaluation
```

### Figure 2: Concatenation rescue plot

Bar chart of:

```text
score(query + g) - score(g only)
```

for each generated text family.

### Figure 3: Accuracy vs excess instability Pareto plot

X-axis:

```text
excess counterfactual instability
```

Y-axis:

```text
average nDCG@10
```

This will make the FAWE vs SAFE tradeoff visually clear.

### Figure 4: Leakage rate vs public-to-counterfactual drop

Each point is a method or query bucket.

### Figure 5: Qualitative examples

Include examples of:

1. raw expected answer succeeds publicly via public answer prior;
2. raw expected answer fails under counterfactual renaming;
3. query anchoring rescues noisy generation;
4. FAWE improves over naive concatenation by limiting expansion dominance;
5. dense retrieval differs from BM25 under filler or entity alias shift.

---

## 16. Implementation and Audit Checklist Before Full Runs

### 16.1 Critical fixes

- [ ] Ensure retriever-specific checkpoint paths for every run.
- [ ] Store checkpoint context with dataset, retriever, regime, method, corpus cap, model, prompt version, and generation hash.
- [ ] Refuse resume if context mismatches.
- [ ] Store final retrieval text and retrieval-text hash for every method/query.
- [ ] Store top-k document IDs and scores for every method/query.
- [ ] Fix or consolidate duplicate oracle/post-hoc aliases.
- [ ] Confirm `wrong_answer_injection` is intentionally equivalent to `concat_query_wrong_answer`, or remove one.

### 16.2 Generation integrity checks

For every generated method:

- [ ] generation prompt version stored;
- [ ] generation model stored;
- [ ] raw LLM response stored;
- [ ] parsed generation stored;
- [ ] final retrieval string stored;
- [ ] exact-answer leakage score stored;
- [ ] unsupported entity/value score stored;
- [ ] query anchor coverage stored.

### 16.3 Counterfactual data checks

- [ ] Alias table versioned.
- [ ] Naturalistic and coded alias regimes separated.
- [ ] Entity and value replacement logs stored.
- [ ] qrel preservation validated.
- [ ] answer evidence preservation validated.
- [ ] human spot check completed.

### 16.4 Dense retrieval checks

- [ ] Confirm query/expansion input order.
- [ ] Confirm max token length and truncation behavior.
- [ ] Confirm vector normalization.
- [ ] Confirm FAWE dense scoring implementation.
- [ ] Confirm no BM25 records are reused by dense jobs.
- [ ] Store embedding model name and revision.

---

## 17. Full-Scale Experimental Plan

### Phase 0: Method cleanup

Goal: remove avoidable reviewer objections.

Tasks:

1. Fix duplicate oracle/post-hoc methods or remove them.
2. Add FAWE controls:
   - `fawe_shuffled_expected_beta0p25`
   - `fawe_wrong_answer_beta0p25`
   - `fawe_neutral_filler_beta0p25`
   - `fawe_query_repeated_beta0p25`
   - `fawe_idf_matched_random_terms_beta0p25`
3. Add beta grid for FAWE.
4. Add BM25+RM3.
5. Add bootstrap/permutation scripts.

### Phase 1: Development-scale runs

Recommended:

```text
N = 200-500 queries per dataset
corpus = full corpus if possible, otherwise >= 50k fixed pool
retrievers = BM25, BM25+RM3, BGE dense
regimes = public, entity-CF, entity+value-CF
```

Use this phase for:

- beta tuning;
- method pruning;
- qualitative example discovery;
- alias regime debugging;
- runtime estimation.

### Phase 2: Final-scale runs

Recommended minimum:

```text
N >= 500 queries per dataset
full corpus if feasible
```

If full corpus is not feasible:

- construct fixed hard-negative pools;
- document construction method;
- ensure relevant documents are always included;
- include at least 50k negatives where possible;
- run a smaller full-corpus sanity subset.

Final method set should be frozen before this phase.

### Phase 3: Analysis

Required analyses:

1. Main performance with confidence intervals.
2. Excess instability over `query_only`.
3. Leakage-rate analysis.
4. Query anchoring / concatenation rescue analysis.
5. FAWE beta sensitivity.
6. Sparse vs dense comparison.
7. Dataset-specific breakdown.
8. Qualitative examples.
9. Failure cases.

---

## 18. Updated Claim Boundaries

### Safe claims if full-scale results match the pilot

1. Standalone answer-like retrieval expansions are highly unstable under entity/value-counterfactual evaluation.
2. Query-preserving expansion is substantially more robust than generated-text-only retrieval.
3. Query anchoring and lexical/semantic enrichment explain a large part of LLM expansion gains.
4. FAWE improves over naive Query2doc-style concatenation in the tested settings.
5. Leakage-aware methods improve stability, but often trade off peak accuracy.

### Claims requiring stronger evidence

1. FAWE universally beats Query2doc.
2. FAWE is the best method on every dataset/retriever/regime.
3. Public benchmark gains are mostly caused by memorization.
4. Counterfactual renaming perfectly simulates private-domain RAG.
5. Dense retrievers are generally robust to appended noise.

### Claims to avoid

1. “This is the first paper to study leakage in LLM query expansion.”
2. “Answer-like expansion is bad.”
3. “Masked expected answers solve leakage.”
4. “SAFE-RRF is the dominant method.”
5. “Counterfactual evaluation proves private-domain performance.”

---

## 19. Recommended Abstract Shape

> LLM-generated query expansions can substantially improve retrieval, but public benchmark evaluations make it difficult to tell whether gains come from useful reformulation or from answer-bearing parametric priors. We introduce an entity/value-counterfactual retrieval evaluation protocol that preserves corpus evidence and relevance labels while suppressing public answer priors. Across Natural Questions, HotpotQA, and SciFact with sparse and dense retrievers, we find that standalone answer-like expansions are highly unstable under counterfactual shift, while query-preserving expansions remain much more robust. Mechanism controls show that query anchoring and lexical enrichment explain a large portion of the gains, independent of exact answer correctness. We propose Fielded Anchor-Weighted Expansion (FAWE), which scores the original query and generated expansion as separate fields to preserve query anchoring while limiting expansion dominance. FAWE improves over naive pseudo-document concatenation in pilot experiments and provides an accuracy-robustness tradeoff for leakage-aware retrieval.

---

## 20. Updated Recommendation to PI

The project is now best framed as an **evaluation + method** paper, not as a pure method-wins paper.

### Primary story

> Counterfactual private-like evaluation reveals that LLM query expansion gains arise from a mixture of answer-prior leakage, query anchoring, and lexical enrichment.

### Primary method

> FAWE: fielded anchor-weighted expansion.

### Main external baseline to beat

> Query2doc-style pseudo-document concatenation.

### Main diagnostic probe

> Raw expected-answer-only retrieval.

### Main robustness baseline

> SAFE-RRF v1 and FAWE-safe-adaptive.

### Main control insight

> Query plus shuffled expected answer remains strong, showing that exact answer correctness is not the only driver of gains.

### Final recommendation

Proceed with FAWE as the lead proposed method, but write the paper around the broader scientific finding. The top-conference version should argue that **how generated text is integrated into retrieval matters as much as what text is generated**, and that public-only evaluation cannot distinguish answer leakage from robust query anchoring without counterfactual controls.

---

## References to Use in the Paper

- Gao et al. 2023. [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://aclanthology.org/2023.acl-long.99/). HyDE.
- Wang et al. 2023. [Query2doc: Query Expansion with Large Language Models](https://aclanthology.org/2023.emnlp-main.585/).
- Lei et al. 2024. [Corpus-Steered Query Expansion with Large Language Models](https://aclanthology.org/2024.eacl-short.34/).
- Yoon et al. 2025. [Hypothetical Documents or Knowledge Leakage? Rethinking LLM-based Query Expansion](https://aclanthology.org/2025.findings-acl.980/).
- Thakur et al. 2021. [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663).
- Yan et al. 2022. [On the Robustness of Reading Comprehension Models to Entity Renaming](https://aclanthology.org/2022.naacl-main.37.pdf).
- Cormack et al. 2009. [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf).
- BAAI. [`bge-base-en-v1.5` model card](https://huggingface.co/BAAI/bge-base-en-v1.5).
