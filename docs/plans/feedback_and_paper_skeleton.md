# Updated Feedback, Next-Step Experimental Plan, and Paper Skeleton

**Project:** Leakage-aware LLM query expansion for retrieval/RAG  
**Current evidence package:** N=100 requested, N_eval = 68 NQ / 85 HotpotQA / 100 SciFact, max-corpus=2000, BM25 + BM25 follow-up + corrected dense matrix  
**Status:** promising for an initial paper draft; not yet sufficient for a top-conference final submission

---

## 1. Executive Verdict

Yes, these results look promising enough to start an initial draft of the paper. The project now has a coherent empirical pattern, a plausible new method family, and a good evaluation contribution. The strongest paper is not a simple method paper claiming that one new expansion dominates all baselines. The stronger top-conference framing is:

> LLM query expansion gains are not a single phenomenon. Public benchmark improvements conflate at least three mechanisms: answer-prior leakage, lexical/semantic enrichment, and query anchoring. Entity/value-counterfactual evaluation separates these mechanisms. Standalone answer-like generation is brittle, while query-preserving and fielded expansion methods are substantially more robust. We propose FAWE, a fielded anchor-weighted expansion strategy that improves how generated expansions are integrated with the original query.

The current evidence is strongest for **BM25**. The dense results are still useful, but they should be framed as an auxiliary stress test showing that counterfactual entity/value rewriting can induce broad representation sensitivity in dense retrievers, not as the cleanest evidence of answer leakage.

The project is now draft-ready because it has:

1. A clear phenomenon: standalone answer-like retrieval collapses under counterfactual rewriting.
2. A strong integration insight: preserving the original query rescues weak or noisy generated expansions.
3. A proposed method with evidence: FAWE, especially `fawe_query2doc_beta0p25`, is the best BM25 method in the current shared-method comparison.
4. Strong controls: wrong-answer, query-repeat, neutral-filler, random-term, and shuffled-expected controls help separate answer correctness from anchored lexical enrichment.
5. A publishable nuance: coherent answer generation is not always necessary; fielded lexical enrichment under a query anchor can explain much of the gain.

The project is not yet submission-ready because:

1. The corpus is still capped at 2000 documents.
2. The evaluated query counts are below the intended N=100 for NQ and HotpotQA.
3. Several oracle/post-hoc methods are exact aliases and should be removed or fixed.
4. The dense story is ambiguous and needs better alias-naturalness controls.
5. Statistical confidence intervals and paired tests are not yet reported.
6. FAWE beta selection must be made methodologically clean before final runs.

---

## 2. How I Would Interpret the Current Results

### 2.1 The strongest result is the BM25 counterfactual story

The BM25 shared-method averages across the 9 BM25 dataset/regime conditions are:

| Method | BM25 average nDCG@10 | Delta vs query-only |
|---|---:|---:|
| `query_only` | 0.6010 | 0.0000 |
| `query2doc_concat` | 0.6180 | +0.0170 |
| `concat_query_raw_expected` | 0.6483 | +0.0473 |
| `safe_rrf_v1` | 0.6239 | +0.0229 |
| `fawe_query2doc_beta0p25` | **0.6573** | **+0.0563** |
| `fawe_safe_adaptive_beta` | 0.6136 | +0.0126 |
| `raw_expected_answer_only` | 0.5768 | -0.0242 |

This is a good pattern. The prior-work baseline `query2doc_concat` beats `query_only`, but the project-owned integration method `fawe_query2doc_beta0p25` improves further. The raw answer route alone is not reliable, which supports the leakage/failure-mode story.

### 2.2 Do not double-count the BM25 follow-up family

The BM25 follow-up reruns preserve the same shared-method averages as BM25 main. That is good for reproducibility, but for paper claims you should not average BM25 main + BM25 follow-up + dense as if they were 27 independent experimental conditions. The BM25 follow-up is an ablation family, not an independent retriever family.

For the main paper, I recommend reporting two primary blocks:

1. **BM25 main/follow-up:** one unified sparse-retrieval result set.
2. **Dense:** one auxiliary dense-retrieval result set.

When computing all-condition averages over shared methods, use the 18 unique dataset x regime x retriever conditions, not the 27 output files.

### 2.3 Unique sparse+dense shared-method average

If we average only the unique BM25 and dense shared-method averages, the picture is:

| Method | Unique BM25+dense average | Delta vs query-only |
|---|---:|---:|
| `query_only` | 0.5868 | 0.0000 |
| `raw_expected_answer_only` | 0.5590 | -0.0278 |
| `query2doc_concat` | 0.6038 | +0.0171 |
| `concat_query_raw_expected` | 0.6196 | +0.0328 |
| `safe_rrf_v1` | 0.6023 | +0.0155 |
| `fawe_query2doc_beta0p25` | **0.6201** | **+0.0333** |
| `fawe_safe_adaptive_beta` | 0.5936 | +0.0069 |

This still supports FAWE, but it also shows that `fawe_query2doc_beta0p25` and `concat_query_raw_expected` are effectively tied overall at current scale. The cleanest method claim should therefore be:

> FAWE is the strongest BM25 method and remains competitive in the dense setting; it improves over naive Query2doc concatenation on BM25 and provides a principled fielded alternative to blind query-expansion concatenation.

Do not claim yet that FAWE universally beats all anchored expansion methods.

---

## 3. What the Current Results Support

### Claim 1: Standalone answer-like expansion is brittle

Supported. `raw_expected_answer_only` remains the cleanest leakage/failure probe. It is strong in public NQ but much weaker under counterfactual rewriting. The NQ BM25 example is especially compelling:

| Method | Public | Entity-CF | Entity+Value-CF |
|---|---:|---:|---:|
| `query_only` | 0.5450 | 0.4298 | 0.4332 |
| `raw_expected_answer_only` | 0.7633 | 0.4889 | 0.4855 |
| `fawe_query2doc_beta0p25` | 0.7783 | 0.5319 | 0.5058 |

The public gain from `raw_expected_answer_only` is large, but the counterfactual advantage contracts substantially. This supports the claim that answer-only retrieval is highly sensitive to public entity/value priors.

### Claim 2: Query anchoring is a major mechanism

Strongly supported. The main story is no longer simply “answer expansion leaks.” The stronger story is:

> Generated text used alone is brittle; generated text anchored to the original query is often robust.

This is supported by the behavior of `concat_query_raw_expected`, `query_plus_shuffled_expected`, and FAWE variants. The shuffled-expected result is especially important: if `fawe_shuffled_expected_beta0p25` matches `fawe_raw_expected_beta0p25`, then coherent sentence structure and exact answer correctness are not the only sources of improvement. The gains are consistent with anchored lexical enrichment.

### Claim 3: FAWE is now a credible lead method

Supported for BM25. The strongest current FAWE method is `fawe_query2doc_beta0p25`:

| Regime | `query_only` | `query2doc_concat` | `fawe_query2doc_beta0p25` | FAWE gain over Query2doc |
|---|---:|---:|---:|---:|
| BM25 public | 0.7112 | 0.8292 | **0.8385** | +0.0093 |
| BM25 entity | 0.5393 | 0.5214 | **0.5727** | +0.0513 |
| BM25 entity+value | 0.5525 | 0.5035 | **0.5608** | +0.0573 |

This is a very good pattern for a top-conference paper because the method does not merely improve public benchmarks. It especially helps in the counterfactual regimes where naive Query2doc can underperform `query_only`.

### Claim 4: Dense retrieval is a supporting result, not the headline

Supported. Dense retrieval shows much higher public scores but much harsher counterfactual drops. For example:

| Dense method | Public avg | Entity-CF avg | Entity+Value-CF avg |
|---|---:|---:|---:|
| `query_only` | 0.8524 | 0.4340 | 0.4310 |
| `query2doc_concat` | 0.8643 | 0.4567 | 0.4478 |
| `concat_query_raw_expected` | 0.8607 | 0.4604 | 0.4516 |
| `fawe_query2doc_beta0p25` | 0.8616 | 0.4413 | 0.4457 |
| `hyde_doc_only` | 0.8640 | 0.4436 | 0.4334 |

The dense counterfactual scores are compressed. The most plausible interpretation is that BGE-style dense retrieval is sensitive to alias distribution shift and counterfactual entity/value rewriting. This is still interesting, but it should not be the main proof of answer leakage.

### Claim 5: BM25+RM3 does not explain away the gains

Supported. In the BM25 follow-up, `bm25_rm3_query_only` averages 0.5981, slightly below `query_only` at 0.6010. RM3 remains useful as a classical pseudo-relevance-feedback baseline, but it is not a serious competitor in this setup.

---

## 4. What the Current Results Do Not Yet Support

### Unsupported claim A: “Our method wins everywhere”

Do not claim this. FAWE is strong, especially on BM25, but dense winners are mixed. `hyde_doc_only` wins dense SciFact in all regimes, while different anchored controls win other dense regimes.

### Unsupported claim B: “Answer-constrained templates are the best mitigation”

Not supported. Constrained templates are safe and interpretable, but not the strongest method. They should be framed as a leakage-aware baseline or component, not the lead method.

### Unsupported claim C: “Dense counterfactual degradation proves leakage”

Too strong. Dense counterfactual degradation could reflect broad embedding-space alias sensitivity rather than answer leakage alone. Treat dense as a representation-shift stress test.

### Unsupported claim D: “Coherent generated answers are necessary”

The shuffled-expected controls argue against this. Shuffled expected text matching raw expected text means lexical content under a query anchor may matter more than coherent answer discourse.

### Unsupported claim E: “N=100 was fully realized for every dataset”

The actual evaluated counts are NQ=68, HotpotQA=85, SciFact=100. The paper must report N_eval exactly and explain the filtering.

---

## 5. Recommended Top-Conference Framing

### 5.1 Best title direction

The paper should emphasize mechanism and evaluation, not just a new expansion trick.

Good title candidates:

1. **Anchors, Answers, and Aliases: Disentangling LLM Query Expansion under Counterfactual Retrieval**
2. **When Query Expansion Knows Too Much: Entity-Counterfactual Evaluation for LLM-Augmented Retrieval**
3. **Fielded Anchors for Leakage-Aware LLM Query Expansion**
4. **Beyond Generated Answers: Query Anchoring and Counterfactual Evaluation for LLM-Based Retrieval**
5. **Are LLM Query Expansions Reformulating or Remembering? A Counterfactual Retrieval Study**

My top choice is title 1 or 2. Title 1 fits the most nuanced version of the paper.

### 5.2 Updated thesis

> Public benchmark gains from LLM query expansion conflate answer-prior leakage, lexical enrichment, and query anchoring. We introduce entity/value-counterfactual retrieval evaluation to separate these mechanisms and show that standalone answer-like expansions are unstable, while anchored and fielded expansion methods are more robust. We propose FAWE, a fielded anchor-weighted expansion method that preserves the original query as a first-class retrieval signal while lightly exploiting generated expansions.

### 5.3 Contributions

Use four contributions:

1. **Mechanistic diagnosis.** Show that LLM query expansion gains arise from at least three mechanisms: answer-prior leakage, lexical enrichment, and query anchoring.
2. **Evaluation protocol.** Introduce entity/value-counterfactual retrieval evaluation for public-to-private-like RAG stress testing, preserving qrels and evidence while suppressing public answer priors.
3. **Method.** Propose FAWE, a fielded anchor-weighted expansion method that integrates generated expansions without allowing them to dominate the original query.
4. **Empirical study.** Compare external prior-work baselines and project-designed diagnostics across NQ, HotpotQA, and SciFact using BM25, BM25+RM3, and dense retrieval, with leakage, anchoring, and control analyses.

### 5.4 Novelty boundary

Do not claim:

- first LLM query expansion method
- first pseudo-document generation method
- first contamination/leakage analysis in LLM query expansion
- first entity-renaming robustness study
- universal superiority over HyDE or Query2doc

You can claim:

- first, or among the first, systematic entity/value-counterfactual evaluation of LLM-generated query expansion for retrieval/RAG-style QA across public and private-like regimes
- a mechanistic decomposition of generated-expansion gains into answer priors, lexical enrichment, and query anchoring
- FAWE as a simple, reproducible fielded integration method that improves over naive pseudo-document concatenation in BM25 counterfactual retrieval

Phrase “first” carefully. A safer sentence is:

> To our knowledge, prior work has not systematically evaluated LLM-generated retrieval expansions under an entity/value-counterfactual protocol that preserves corpus evidence and qrels while suppressing public answer priors.

---

## 6. Method Ownership and Baseline Taxonomy

This is important for the paper. Only a few methods are external prior-work baselines. Most are your own project-designed methods or controls. The method table should make that explicit.

| Method | Category | Paper role |
|---|---|---|
| `query_only` | Standard baseline | No-expansion anchor |
| `bm25_rm3_query_only` | Classical IR baseline | Pseudo-relevance feedback control |
| `hyde_doc_only` | External prior-work baseline | Hypothetical document generation baseline |
| `query2doc_concat` | External prior-work baseline | Strongest LLM pseudo-document baseline |
| `generative_relevance_feedback_concat` | External/prior-inspired baseline | Generated relevance feedback baseline |
| `corpus_steered_short_concat` | External/prior-inspired baseline | Corpus-grounded expansion baseline |
| `raw_expected_answer_only` | Project diagnostic | Answer-prior leakage probe |
| `concat_query_raw_expected` | Project method/diagnostic | Strong answer-anchored expansion |
| `concat_query_answer_candidate_constrained_template` | Project method | Answer-candidate-constrained expansion |
| `safe_rrf_v1` | Project method | Conservative leakage-aware fusion |
| `cf_prompt_query_expansion_rrf` | Project method | Counterfactual prompting route |
| `fawe_query2doc_beta*` | Project method | Lead method family |
| `fawe_raw_expected_beta*` | Project method/diagnostic | Fielded answer expansion |
| `fawe_safe_adaptive_beta` | Project method | Adaptive fielded expansion |
| `query_plus_shuffled_expected` | Project control | Tests coherence vs lexical enrichment |
| `concat_query_wrong_answer` | Project control | Tests query dominance under wrong answer |
| `query_repeated` | Project control | Tests length/repetition artifact |
| `query_plus_neutral_filler` | Project control | Tests generic appended text |
| random/idf-matched terms | Project control | Tests arbitrary lexical enrichment |

Recommended wording:

> We use HyDE, Query2doc, RM3, and CSQE-style expansion as external baselines. We introduce expected-answer diagnostics, answer-candidate-constrained templates, SAFE-RRF, counterfactual prompting, and FAWE as project-designed interventions and controls.

---

## 7. Revised Experimental Plan

### 7.1 Primary research questions

**RQ1. Answer-prior leakage.**  
Do standalone LLM-generated answer-like expansions perform well on public benchmarks but lose utility under entity/value-counterfactual rewriting?

**RQ2. Query anchoring.**  
How much of the gain from generated expansion comes from preserving the original query rather than using the generated text alone?

**RQ3. Fielded integration.**  
Does FAWE improve over naive concatenation and prior-work Query2doc, especially under counterfactual regimes?

**RQ4. Mechanism controls.**  
Are improvements driven by answer correctness, lexical enrichment, query length, arbitrary added tokens, or coherent generated discourse?

**RQ5. Retriever dependence.**  
Do sparse and dense retrievers respond differently to counterfactual entity/value rewriting and generated expansions?

### 7.2 Main hypotheses after the latest results

**H1.** Standalone answer-like expansion will show strong public gains on entity-centric public QA, especially NQ, but will be unstable under counterfactual rewriting.

**H2.** Query-preserving methods will be substantially more stable than generated-only methods.

**H3.** FAWE will improve over naive Query2doc concatenation in BM25, particularly under counterfactual rewriting.

**H4.** FAWE beta should decrease as counterfactual shift increases; smaller beta values should be more robust under entity+value rewriting.

**H5.** Dense retrieval will show broader alias sensitivity, making it less diagnostic of answer leakage than BM25.

**H6.** Shuffled or non-coherent but query-related expansions will remain competitive with coherent expansions when the original query is preserved, indicating that lexical enrichment is a major mechanism.

### 7.3 Dataset plan

Minimum paper set:

| Dataset | Role | Main analysis |
|---|---|---|
| Natural Questions | Entity-centric public QA | Best leakage-prior testbed |
| HotpotQA | Multi-hop QA | Tests bridge/entity sensitivity and evidence structure |
| SciFact | Scientific evidence retrieval | Domain-specialized retrieval and claim/evidence setting |

Optional extension if resources allow:

| Dataset | Why add it |
|---|---|
| FiQA | Non-Wikipedia finance domain, tests domain shift |
| TREC-COVID | Scientific/biomedical retrieval, if qrels and licenses are clean |
| One internal/private dataset | Strongest real-world validation if publishable |

For a top-conference submission, three datasets can be enough if the analysis is deep, but adding one non-Wikipedia dataset would reduce the risk that reviewers see the work as a Wikipedia-only artifact.

### 7.4 Query-count and corpus-size plan

Current evaluated counts are NQ=68, HotpotQA=85, SciFact=100. Before final experiments:

1. Report intended N and evaluated N separately.
2. Explain why NQ and HotpotQA have fewer evaluated queries.
3. Audit whether filtering is due to missing answer metadata, missing qrels, max-corpus truncation, or counterfactual-construction failures.
4. For final runs, target at least:
   - NQ: >=500 evaluated queries
   - HotpotQA: >=500 evaluated queries
   - SciFact: all available queries if feasible, or >=300 if constrained
5. Use full corpus if feasible. If full corpus is impossible, use a fixed hard-negative pool with at least 50k documents and guarantee that all relevant documents for evaluated queries are included.

Critical audit:

```text
qrel_coverage(query) = number of relevant documents included in retrieval corpus
```

Exclude or separately report queries where qrel coverage is zero. Otherwise retrieval scores become uninterpretable.

### 7.5 Counterfactual regimes

Keep three regimes:

1. `public`: original benchmark.
2. `entity-CF`: named entities rewritten consistently.
3. `entity+value-CF`: named entities plus sensitive values, dates, and numbers rewritten.

Add two alias styles:

1. **Naturalistic aliases** as the main condition.
   - PERSON -> plausible person names
   - ORG -> plausible organization names
   - LOCATION -> plausible location names
   - WORK/TITLE -> plausible titles
2. **Coded aliases** as stress-test ablation.
   - PERSON -> `Person ZQ-17`
   - ORG -> `Unit RK-9`
   - LOCATION -> `Region PX-3`

Reason: Dense retrieval may be harmed by unnatural alias strings. Naturalistic aliases reduce the risk that dense degradation is just out-of-distribution tokenization.

### 7.6 Method set for next-scale runs

Use a smaller paper-facing method matrix.

#### External baselines

1. `query_only`
2. `bm25_rm3_query_only`
3. `hyde_doc_only`
4. `query2doc_concat`
5. `generative_relevance_feedback_concat`
6. `corpus_steered_short_concat`

#### Project methods

7. `raw_expected_answer_only` as diagnostic leakage probe
8. `concat_query_raw_expected`
9. `concat_query_answer_candidate_constrained_template`
10. `safe_rrf_v1`
11. `cf_prompt_query_expansion_rrf`
12. `fawe_query2doc_beta0p25` as fixed-beta lead method
13. `fawe_query2doc_beta0p05` and/or `fawe_query2doc_beta0p10` as robustness ablations
14. `fawe_raw_expected_beta0p25`
15. `fawe_safe_adaptive_beta`

#### Controls

16. `query_repeated`
17. `query_plus_neutral_filler`
18. `concat_query_wrong_answer`
19. `query_plus_shuffled_expected`
20. `fawe_shuffled_expected_beta0p25`
21. `fawe_wrong_answer_beta0p25`
22. `fawe_neutral_filler_beta0p25`
23. `fawe_idf_matched_random_terms_beta0p25`
24. `fawe_random_terms_from_corpus_beta0p25`

Move the following to appendix or remove unless needed:

- standalone masked expected answer
- generic mask slot only
- entity-only masking only
- many weighted RRF variants
- duplicate oracle/post-hoc aliases

### 7.7 FAWE beta protocol

The beta sweep is important, but it can become an overfitting risk. Use a clean protocol:

**Option A: Fixed beta main method.**

- Main method: `fawe_query2doc_beta0p25`
- Motivation: best public BM25 and strongest current overall BM25 method
- Report beta sweep as ablation

**Option B: Dev-tuned beta.**

- Split each dataset into dev/test by query ID.
- Tune beta on public-dev only, then evaluate public-test and counterfactual-test.
- This avoids tuning directly on the counterfactual test condition.

**Option C: Unsupervised adaptive beta.**

- Estimate expansion risk from unsupported entities/values, query-anchor coverage, and first-pass support.
- Choose smaller beta for higher-risk expansions.
- Tune thresholds on dev, freeze for test.

Recommended for paper:

1. Main: fixed `beta=0.25` FAWE-Query2doc.
2. Robustness ablation: beta sweep `{0.05, 0.10, 0.25, 0.50, 1.00}`.
3. Optional: adaptive beta as secondary method, not the lead method unless it improves.

### 7.8 Metrics

Primary retrieval metrics:

- `nDCG@10` as the main metric
- `Recall@20`
- `Recall@100`
- `MRR@10` for QA-style datasets

Counterfactual and mechanism metrics:

1. **Counterfactual utility**

```text
CF_utility(method, regime) = score(method, regime) - score(query_only, regime)
```

2. **Raw drop**

```text
drop(method, regime) = score(method, public) - score(method, regime)
```

3. **Excess drop over query-only**

```text
excess_drop(method, regime)
= [score(method, public) - score(method, regime)]
  - [score(query_only, public) - score(query_only, regime)]
```

4. **Anchoring rescue**

```text
anchor_rescue(g) = score(query + g) - score(g only)
```

5. **FAWE advantage**

```text
fawe_advantage(g) = score(FAWE(query, g)) - score(concat(query, g))
```

6. **Leakage rate**

```text
exact_answer_leakage = fraction of expansions containing gold answer or alias
unsupported_entity_rate = generated entities absent from query and first-pass corpus evidence
unsupported_value_rate = generated dates/numbers/values absent from query and first-pass corpus evidence
```

7. **Expansion-support metrics**

```text
query_anchor_coverage = fraction of key query anchors preserved in expansion
first_pass_support = fraction of expansion entities/terms appearing in query_only top-k docs
```

### 7.9 Statistical analysis

For each dataset/regime/retriever:

1. Use paired bootstrap confidence intervals over queries for nDCG@10.
2. Use paired randomization or permutation tests for primary comparisons.
3. Correct the main comparison family with Holm-Bonferroni or Benjamini-Hochberg.
4. Report per-query win/tie/loss counts for:
   - FAWE-Query2doc vs Query2doc concat
   - FAWE-Query2doc vs query-only
   - concat raw expected vs raw expected only
   - raw expected only vs query-only
5. Add effect sizes, not only p-values.

Primary comparisons:

1. `fawe_query2doc_beta0p25` vs `query2doc_concat`
2. `fawe_query2doc_beta0p25` vs `query_only`
3. `concat_query_raw_expected` vs `raw_expected_answer_only`
4. `raw_expected_answer_only` vs `query_only` public vs counterfactual
5. `fawe_query2doc_beta0p25` vs `fawe_query2doc_beta0p05/0p10` under counterfactual regimes

### 7.10 Dense retrieval analysis plan

Dense should be analyzed separately. Recommended language:

> Dense retrieval exhibits strong public performance but broad counterfactual sensitivity, suggesting that alias rewriting perturbs representation geometry beyond answer leakage alone.

Dense-specific analyses:

1. Compare naturalistic vs coded aliases.
2. Measure embedding similarity between original and counterfactual queries.
3. Measure embedding similarity between original and counterfactual relevant documents.
4. Report whether dense failures are due to relevant documents falling far in embedding space or expansions retrieving semantically adjacent but wrong alias clusters.
5. Include one additional dense retriever if resources allow, e.g. E5 or Contriever, to avoid overfitting the conclusion to BGE.

---

## 8. Key Derived Analyses to Add to the Paper

### 8.1 BM25 counterfactual utility table

This table is more interpretable than raw scores alone.

| Method | Public utility | Entity-CF utility | Entity+Value-CF utility |
|---|---:|---:|---:|
| `query2doc_concat` | +0.1180 | -0.0179 | -0.0490 |
| `concat_query_raw_expected` | +0.0995 | +0.0307 | +0.0116 |
| `fawe_query2doc_beta0p25` | **+0.1273** | **+0.0334** | +0.0083 |
| `fawe_safe_adaptive_beta` | +0.0110 | +0.0191 | +0.0077 |
| `hyde_doc_only` | +0.0978 | -0.0928 | -0.1193 |

This table makes the main BM25 story very clear:

- HyDE and naive Query2doc look good publicly but lose counterfactual utility.
- FAWE-Query2doc preserves positive utility in counterfactual regimes.
- Concatenated raw expected answers are also strong, but they are less clean as a paper-facing method because they can contain answer leakage.

### 8.2 Excess-drop table

Interpretation: positive means more unstable than `query_only`; near zero or negative means stability comparable to or better than query-only.

| Method | BM25 entity excess drop | BM25 entity+value excess drop |
|---|---:|---:|
| `query2doc_concat` | +0.1359 | +0.1670 |
| `concat_query_raw_expected` | +0.0688 | +0.0879 |
| `fawe_query2doc_beta0p25` | +0.0939 | +0.1190 |
| `fawe_safe_adaptive_beta` | -0.0081 | +0.0033 |
| `hyde_doc_only` | +0.1906 | +0.2171 |

This reveals a useful nuance:

- FAWE-Query2doc is best for BM25 effectiveness.
- FAWE-safe/adaptive is best for stability.
- These are two different Pareto points.

This is an important top-conference framing: **accuracy-robustness tradeoff**, not one universal best method.

### 8.3 Dense excess-drop table

| Method | Dense entity excess drop | Dense entity+value excess drop |
|---|---:|---:|
| `query2doc_concat` | -0.0108 | -0.0049 |
| `concat_query_raw_expected` | -0.0181 | -0.0123 |
| `fawe_query2doc_beta0p25` | +0.0019 | -0.0055 |
| `hyde_doc_only` | +0.0020 | +0.0092 |

Dense excess drops are small because `query_only` itself collapses under counterfactual shift. This supports the decision to treat dense as a secondary stress test rather than the main leakage evidence.

---

## 9. Immediate Implementation and Cleanup Checklist

### 9.1 Must fix before next full run

- [ ] Remove or fix exact duplicate oracle/post-hoc methods.
- [ ] Store final retrieval text hash for every method/query.
- [ ] Store method config hash, prompt version, LLM model ID, retriever ID, corpus ID, counterfactual alias table ID, and beta value.
- [ ] Report N_requested and N_evaluated separately.
- [ ] Add qrel coverage audits for every capped corpus.
- [ ] Verify that every evaluated query has at least one relevant document in the retrieval corpus.
- [ ] Verify that counterfactual replacements are consistent across corpus, query, answer metadata, and qrels.
- [ ] Verify that no original public entity strings remain in counterfactual queries/corpora except where intentionally preserved.
- [ ] Add naturalistic alias regime.
- [ ] Add paired bootstrap confidence intervals.

### 9.2 Should fix before paper submission

- [ ] Add full-corpus or large hard-negative runs.
- [ ] Add at least one additional dense retriever or justify why BGE is sufficient.
- [ ] Add leakage scorer outputs to all result tables.
- [ ] Add query-type stratification: person, organization, location, date, number, work/title, multi-hop bridge.
- [ ] Add qualitative examples for all major mechanisms.
- [ ] Add cost/runtime table.
- [ ] Release counterfactual construction code and alias tables if licensing permits.

### 9.3 Duplicate-method cleanup

Current exact alias pairs should not appear as separate main methods:

```text
gold_answer_only == raw_expected_answer_only
oracle_answer_masked == raw_expected_answer_only
post_hoc_gold_removed_expected_answer == raw_expected_answer_only
concat_query_oracle_answer_masked == concat_query_raw_expected
concat_query_post_hoc_gold_removed_expected == concat_query_raw_expected
concat_query_wrong_answer == wrong_answer_injection
```

Either:

1. fix them so they actually implement distinct interventions, or
2. remove them and state that earlier variants collapsed to identical retrieval strings and were consolidated.

Do not include exact aliases in the final main tables.

---

## 10. Recommended Final Main Tables and Figures

### Main paper tables

**Table 1: Method taxonomy**  
External baselines vs project-designed methods vs controls.

**Table 2: BM25 main results**  
Rows: key methods. Columns: NQ/SciFact/HotpotQA x public/entity/entity+value or averaged by regime.

**Table 3: Dense supporting results**  
Same rows, fewer columns, framed as representation-shift stress test.

**Table 4: BM25 counterfactual utility and excess drop**  
This may be the most important table in the paper.

**Table 5: FAWE beta sweep**  
Show public vs counterfactual tradeoff.

**Table 6: Mechanism controls**  
Wrong answer, repeated query, neutral filler, shuffled expected, random terms, IDF-matched random terms.

**Table 7: Leakage and support metrics**  
Exact answer leakage, unsupported entity/value injection, query anchor coverage, first-pass support.

### Main paper figures

**Figure 1: Concept diagram**  
Public query -> LLM expansion -> answer-prior leakage; counterfactual query -> prior mismatch; FAWE preserves query anchor.

**Figure 2: Public-to-counterfactual drop plot**  
Grouped bars for query-only, HyDE, Query2doc, concat raw, FAWE-Query2doc, FAWE-safe.

**Figure 3: Accuracy-stability Pareto plot**  
x-axis: BM25 public nDCG@10. y-axis: negative excess drop or counterfactual utility. Show FAWE-Query2doc and FAWE-safe as different Pareto points.

**Figure 4: Beta sweep curve**  
FAWE-Query2doc beta vs nDCG@10 for public, entity-CF, and entity+value-CF.

**Figure 5: Concatenation rescue plot**  
Generated-only vs query+generated for raw expected, masked expected, constrained templates, random span, generic slot.

**Figure 6: Qualitative examples**  
Three examples:

1. Public success via answer-prior leakage.
2. Counterfactual failure of standalone generated answer or HyDE.
3. FAWE or anchored expansion recovery.

---

## 11. Proposed Paper Skeleton Draft

# Anchors, Answers, and Aliases: Disentangling LLM Query Expansion under Counterfactual Retrieval

## Abstract

Large language models are increasingly used to generate query expansions for retrieval-augmented generation, including hypothetical documents, pseudo-documents, and answer-like reformulations. However, public benchmark gains can arise from multiple mechanisms: genuine reformulation, lexical enrichment, query anchoring, and answer-prior leakage from the model's parametric knowledge. We propose an entity/value-counterfactual retrieval evaluation protocol that preserves corpus evidence and relevance labels while rewriting public entities and sensitive values to suppress memorized answer priors. Across Natural Questions, HotpotQA, and SciFact, we find that standalone answer-like expansions are brittle under counterfactual rewriting, while query-preserving integration is substantially more robust. We introduce Fielded Anchor-Weighted Expansion (FAWE), which scores the original query and generated expansion as separate fields, preserving the query as a first-class retrieval signal. In BM25 retrieval, FAWE improves over naive Query2doc-style concatenation and maintains positive counterfactual utility where several generated-only or pseudo-document baselines degrade. Dense retrieval shows broader alias sensitivity, highlighting a distinct representation-shift failure mode. Our results suggest that LLM query expansion should be evaluated not only by public benchmark gains, but by whether gains survive private-like counterfactual shifts.

## 1. Introduction

### 1.1 Motivation

Retrieval-augmented generation systems often rely on high-quality retrieval reformulations. Recent methods ask an LLM to generate a hypothetical document, pseudo-document, or expected answer and then use that generated text for retrieval. These methods can improve public benchmark retrieval, but the source of the improvement is ambiguous.

Generated expansions may help because they clarify the information need. They may also help because the LLM already knows the public benchmark answer and injects answer-bearing content into the retrieval query. This distinction matters for private-domain RAG. In private corpora, parametric priors may be wrong, stale, or irrelevant. A reformulation method that works by injecting public answer priors may fail when entities and facts are private.

### 1.2 Running example

Use a concrete NQ-style example.

Public query:

```text
Who wrote [famous work]?
```

LLM-generated expected answer:

```text
[famous author] wrote [famous work].
```

In the public benchmark, the generated answer can retrieve the relevant document by lexical overlap with the gold answer. In a counterfactual private-like corpus, the work and author names are consistently renamed. The LLM may still generate the public author, which is now wrong for the renamed corpus. The same expansion can misdirect retrieval.

### 1.3 Key observation

The project reveals a more subtle phenomenon than “answer expansion is bad.” Standalone answer-like text is brittle, but query-preserving expansion can remain strong. In other words, the original query acts as an anchor. The gain from LLM expansion is a mixture of:

1. answer-prior leakage,
2. lexical/semantic enrichment,
3. query anchoring,
4. retriever-specific scoring effects.

### 1.4 Contributions

List four contributions:

1. We formulate answer-prior leakage and query anchoring as distinct mechanisms in LLM-generated retrieval expansion.
2. We introduce an entity/value-counterfactual private-like evaluation protocol for retrieval expansion.
3. We propose FAWE, a fielded anchor-weighted method that integrates generated expansions while preserving the original query as a dominant signal.
4. We conduct a controlled empirical study across three datasets, sparse and dense retrieval, external baselines, project-designed interventions, and mechanism controls.

## 2. Related Work

### 2.1 LLM-generated query expansion

Discuss HyDE, Query2doc, generative relevance feedback, and corpus-steered query expansion. Emphasize that these methods establish generated text as a strong retrieval ingredient, but generally do not separate answer-prior leakage from reformulation and anchoring effects.

### 2.2 Benchmark contamination and knowledge leakage

Discuss LLM benchmark contamination and the specific LLM query-expansion leakage paper. Position this paper as extending leakage analysis to retrieval/RAG-style QA with entity/value-counterfactual corpus transformations.

### 2.3 Counterfactual robustness and entity renaming

Discuss prior entity-renaming robustness work in reading comprehension. Clarify that the novelty here is not entity renaming alone, but using entity/value counterfactuals to evaluate retrieval expansion under preserved qrels and evidence.

### 2.4 Classical pseudo-relevance feedback and fusion

Discuss RM3, pseudo-relevance feedback, and RRF. Explain that FAWE is different from generic fusion because it explicitly treats the original query and generated expansion as separate fields with controlled expansion weight.

## 3. Problem Setup

Define:

```text
q: original query
C: retrieval corpus
R(q): relevant documents/qrels
g(q): LLM-generated expansion
s(q, d): retriever score for document d
```

Standard expansion methods construct either:

```text
g(q) only
q || g(q)
RRF(rank(q), rank(g(q)))
```

The problem is that `g(q)` may introduce answer-bearing content not present in `q`.

### 3.1 Mechanisms

Define three mechanisms:

1. **Answer-prior leakage:** `g(q)` contains the gold answer, answer alias, bridge entity, date, number, or evidence entailed by the gold document.
2. **Lexical enrichment:** `g(q)` adds terms related to the information need, regardless of answer correctness.
3. **Query anchoring:** the original query remains present and prevents the generated text from fully dominating retrieval.

### 3.2 Evaluation desiderata

A strong evaluation should:

1. preserve evidence and qrels,
2. reduce usefulness of public memorized answer priors,
3. maintain query type and retrieval task structure,
4. distinguish generated-only from query-anchored integration,
5. include controls for wrong answers, random terms, and filler text.

## 4. Entity/Value-Counterfactual Retrieval Evaluation

### 4.1 Construction

For each dataset, construct two counterfactual variants:

1. Entity-CF: rewrite named entities consistently.
2. Entity+Value-CF: rewrite named entities plus dates, numbers, and sensitive values.

The alias mapping is one-to-one and consistent across corpus, query text, and answer/evidence metadata. Qrels are preserved because document identities and relevance labels are unchanged.

### 4.2 Validation checks

Automatic checks:

- alias consistency,
- no original public names remain,
- qrel coverage is preserved,
- relevant documents still contain answer/evidence spans after rewriting,
- query type is preserved,
- answer metadata is rewritten consistently.

Human spot checks:

- 50 examples per dataset/regime,
- verify grammatical plausibility,
- verify no obvious original-public answer leakage,
- verify relevant evidence remains sufficient.

### 4.3 Alias naturalness

Main experiments should use naturalistic aliases. Coded aliases should be an ablation because they may create tokenization and embedding artifacts, especially for dense retrievers.

## 5. Methods

### 5.1 External baselines

**BM25 query-only.** Standard sparse lexical retrieval.

**BM25+RM3.** Classical pseudo-relevance-feedback baseline.

**Dense query-only.** BGE-base-en-v1.5 or a comparable zero-shot dense retriever.

**HyDE.** Generate a hypothetical document and retrieve with it.

**Query2doc.** Generate a pseudo-document and concatenate it to the query.

**Generative relevance feedback.** Generate relevance feedback text and concatenate it to the query.

**Corpus-steered short expansion.** Use first-pass corpus evidence to steer short expansion.

### 5.2 Project-designed diagnostics

**Raw expected answer only.** Generate a concise expected answer and retrieve using only that answer. This is a diagnostic leakage probe, not a proposed production method.

**Query + raw expected answer.** Concatenate the original query with the expected answer. This tests query anchoring.

**Query + constrained template.** Generate answer-candidate-constrained templates that do not introduce new concrete answer candidates.

**SAFE-RRF.** Fuse routes with leakage-aware weights. Present as a conservative robustness-oriented method.

**CF-prompt QE.** Obfuscate public entity triggers before generating answer-free retrieval queries.

### 5.3 FAWE: Fielded Anchor-Weighted Expansion

FAWE scores the original query and generated expansion as separate fields rather than blindly concatenating them.

For BM25:

```text
score_FAWE(d | q, g) = BM25(q, d) + beta * BM25(g, d)
```

For dense retrieval:

```text
score_FAWE(d | q, g) = cos(E(q), E(d)) + beta * cos(E(g), E(d))
```

where `beta` controls how much the generated expansion can influence the ranking.

FAWE is motivated by the pilot finding that generated text is helpful when anchored, but harmful when allowed to dominate. A small beta preserves lexical enrichment while limiting prior-induced drift.

### 5.4 FAWE variants

Main:

```text
fawe_query2doc_beta0p25
```

Ablations:

```text
fawe_query2doc_beta0p05
fawe_query2doc_beta0p10
fawe_query2doc_beta0p50
fawe_raw_expected_beta0p25
fawe_safe_adaptive_beta
```

Controls:

```text
fawe_shuffled_expected_beta0p25
fawe_wrong_answer_beta0p25
fawe_neutral_filler_beta0p25
fawe_random_terms_from_corpus_beta0p25
fawe_idf_matched_random_terms_beta0p25
fawe_query_repeated_beta0p25
```

## 6. Experimental Setup

### 6.1 Datasets

Natural Questions, HotpotQA, and SciFact.

Report:

- source split,
- corpus size,
- number of queries requested,
- number of queries evaluated,
- filtering criteria,
- qrel coverage,
- answer/evidence metadata availability.

### 6.2 Retrieval backends

Sparse:

- BM25
- BM25+RM3

Dense:

- BAAI/bge-base-en-v1.5
- optional additional retriever: E5 or Contriever

### 6.3 Generation setup

Report:

- LLM provider and model version,
- decoding temperature,
- prompt templates,
- cache policy,
- JSON schemas,
- prompt version hashes,
- whether generations are shared across public and counterfactual regimes or generated separately.

Important design decision:

For leakage analysis, generate expansions from the query in each regime. For counterfactual regimes, the LLM sees counterfactual aliases. This tests whether the method can operate when public triggers are suppressed.

### 6.4 Metrics

Primary: nDCG@10.  
Secondary: Recall@20, Recall@100, MRR@10.  
Mechanism: leakage rate, unsupported entity/value rate, query-anchor coverage, first-pass support, anchoring rescue, excess drop.

### 6.5 Statistical testing

Use paired bootstrap confidence intervals and paired randomization tests.

## 7. Results

### 7.1 RQ1: Standalone answer-like expansion is brittle

Show raw expected answer results across public and counterfactual regimes. Emphasize NQ as the cleanest case.

Expected statement:

> Raw expected answers can perform well in public NQ but lose much of their advantage after entity/value rewriting, indicating that answer-like standalone retrieval is sensitive to public answer priors.

### 7.2 RQ2: Query anchoring rescues generated text

Show generated-only vs query+generated comparisons.

Expected statement:

> Across expansion families, appending generated text to the original query is substantially stronger than using generated text alone. This shows that query anchoring is a primary mechanism behind robust expansion.

### 7.3 RQ3: FAWE improves fielded integration

Main table: FAWE-Query2doc vs Query2doc concat vs query-only.

Expected statement:

> In BM25 retrieval, FAWE-Query2doc improves over naive Query2doc concatenation, especially under counterfactual regimes where naive concatenation can underperform query-only.

### 7.4 RQ4: Controls reveal lexical enrichment under anchoring

Show shuffled expected, wrong answer, neutral filler, random terms, IDF-matched random terms.

Expected statement:

> Shuffled expected text remains competitive with coherent expected text, while wrong answers and random/neutral fillers are baseline-like or harmful. This suggests that gains come from query-related lexical enrichment rather than answer correctness alone or arbitrary added length.

### 7.5 RQ5: Sparse and dense retrievers differ

Show dense table separately.

Expected statement:

> Dense retrieval has high public performance but broad counterfactual sensitivity. Unlike BM25, the dense setting compresses differences among methods under counterfactual rewriting, suggesting that alias rewriting induces representation shift in addition to leakage suppression.

## 8. Discussion

### 8.1 What public benchmark gains mean

Public gains from generated expansion do not necessarily mean the method learned a better retrieval reformulation. Gains can reflect answer priors, lexical enrichment, and anchoring. Therefore, public-only retrieval evaluation is incomplete for private-domain RAG.

### 8.2 Why FAWE helps

FAWE makes the original query non-negotiable. The generated expansion can add useful terms, but the beta weight prevents it from dominating retrieval. This explains why FAWE improves over naive Query2doc in BM25 counterfactual regimes.

### 8.3 Why dense is different

Dense retrievers encode aliases and relation context into a shared embedding space. Counterfactual rewriting may perturb the embedding space even when the retrieval task is structurally preserved. Dense results should therefore be interpreted as a mixture of leakage suppression and representation robustness.

### 8.4 Practical implications for private RAG

For private RAG, avoid standalone generated answers as retrieval queries. Prefer query-preserving integration, fielded weighting, corpus support checks, and conservative beta values. Treat public benchmark wins from LLM expansion as insufficient evidence of private-domain robustness.

## 9. Limitations

1. Current results use capped corpora and partial N_eval.
2. Entity/value counterfactuals are an approximation to private-domain RAG, not a true private corpus.
3. Alias naturalness may affect dense retrievers.
4. FAWE beta may require tuning or robust default selection.
5. LLM-generated expansions depend on model version and prompt design.
6. Current datasets are mostly QA/fact-verification; additional domains would strengthen generality.
7. Counterfactual rewriting may alter surface statistics even when evidence structure is preserved.

## 10. Conclusion

LLM-generated query expansion should not be evaluated solely by public benchmark retrieval gains. The same generated text can help because it reformulates the query, because it injects answer priors, or because it provides query-related lexical enrichment under the protection of the original query. Entity/value-counterfactual evaluation exposes these mechanisms. FAWE offers a simple and effective way to preserve query anchoring while using generated expansions, improving BM25 robustness over naive pseudo-document concatenation in current results.

---

## 12. Recommended Claim Boundaries for the Initial Draft

### Safe claims now

- Standalone answer-like retrieval is brittle under entity/value counterfactual rewriting.
- Query-preserving integration is much more robust than generated-only retrieval.
- FAWE-Query2doc is the strongest BM25 method in the current capped-corpus results.
- BM25 provides the cleanest evidence for separating leakage, enrichment, and anchoring.
- Dense retrieval shows broad counterfactual alias sensitivity and should be analyzed separately.
- Shuffled-expected controls suggest that coherent answer generation is not necessary for all gains.

### Claims requiring final-scale evidence

- FAWE significantly outperforms Query2doc across datasets.
- FAWE is robust under full-corpus retrieval.
- Counterfactual gains generalize beyond NQ, HotpotQA, and SciFact.
- Dense retrieval behaves similarly across embedding models.
- The majority of public Query2doc gains are due to leakage rather than useful reformulation.

### Claims to avoid

- FAWE wins everywhere.
- Answer-constrained templates are the best method.
- Dense counterfactual drops prove answer leakage.
- Entity/value counterfactual evaluation perfectly simulates private RAG.
- Prior work has not studied leakage in LLM query expansion at all.

---

## 13. Recommended Next-Step Work Plan

### Stage 1: Clean and freeze

- Consolidate duplicate methods.
- Freeze method names and taxonomy.
- Add retrieval-text hashes.
- Add qrel-coverage reports.
- Add N_requested/N_eval reports.
- Freeze prompt/model versions.
- Create one canonical result aggregation script.

### Stage 2: Strengthen counterfactual construction

- Add naturalistic aliases.
- Keep coded aliases as ablation.
- Add human spot checks.
- Add automatic leakage checks.
- Verify entity/value consistency across query, corpus, and metadata.

### Stage 3: Scale BM25 first

- Run BM25 full corpus or large fixed hard-negative pool.
- Use the reduced method matrix.
- Add bootstrap CIs and significance testing.
- Prioritize FAWE-Query2doc, Query2doc concat, concat raw expected, raw expected only, HyDE, RM3, and controls.

### Stage 4: Dense follow-up

- Rerun BGE with naturalistic aliases.
- Add one more dense retriever if feasible.
- Add embedding similarity diagnostics.
- Frame dense as a representation-shift analysis.

### Stage 5: Paper assembly

- Draft introduction and method sections now.
- Fill results with current pilot numbers as “preliminary” internally.
- Replace with final-scale results when available.
- Prepare qualitative examples early, because they will shape the story.

---

## 14. Reviewer Risk Register

| Risk | Likely reviewer criticism | Mitigation |
|---|---|---|
| Corpus cap | Results may be artifact of max-corpus=2000 | Run full corpus or >=50k hard negatives with qrel coverage audit |
| Small/uneven N_eval | NQ=68 and HotpotQA=85 are small | Scale to >=500 or all available; report filtering |
| Dense ambiguity | Dense degradation may be alias OOD, not leakage | Use naturalistic aliases and extra dense retriever |
| FAWE simplicity | FAWE is just weighted scoring | Emphasize mechanism, controls, and robustness; compare to concat and RRF |
| Shuffled control | Coherence not needed, weakens LLM story | Embrace this as a mechanistic finding: lexical enrichment under anchoring matters |
| Prior leakage paper | Not first to study leakage | Position as QA/private-like counterfactual retrieval evaluation and integration-mode analysis |
| Entity renaming prior work | Entity renaming is not new | Novelty is retrieval expansion + preserved qrels/evidence + LLM-generated routes |
| Exact aliases | Duplicate methods inflate matrix | Remove/fix duplicates before final paper |
| Beta tuning | Overfit beta to test conditions | Use fixed beta or dev-tuned beta protocol |

---

## 15. References to Anchor the Draft

- Gao et al. **Precise Zero-Shot Dense Retrieval without Relevance Labels**. ACL 2023. HyDE. https://aclanthology.org/2023.acl-long.99/
- Wang et al. **Query2doc: Query Expansion with Large Language Models**. EMNLP 2023. https://aclanthology.org/2023.emnlp-main.585/
- Yoon et al. **Hypothetical Documents or Knowledge Leakage? Rethinking LLM-based Query Expansion**. Findings of ACL 2025. https://aclanthology.org/2025.findings-acl.980/
- Lei et al. **Corpus-Steered Query Expansion with Large Language Models**. EACL 2024. https://aclanthology.org/2024.eacl-short.34/
- Thakur et al. **BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models**. NeurIPS Datasets and Benchmarks 2021. https://arxiv.org/abs/2104.08663
- Cormack et al. **Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods**. SIGIR 2009. https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf
- Kwiatkowski et al. **Natural Questions: A Benchmark for Question Answering Research**. TACL 2019. https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00276/43518/Natural-Questions-A-Benchmark-for-Question
- Yang et al. **HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering**. EMNLP 2018. https://aclanthology.org/D18-1259/
- Wadden et al. **Fact or Fiction: Verifying Scientific Claims**. EMNLP 2020. https://aclanthology.org/2020.emnlp-main.609/
- Yan et al. **On the Robustness of Reading Comprehension Models to Entity Renaming**. NAACL 2022. https://aclanthology.org/2022.naacl-main.37/
- Xu et al. **Benchmark Data Contamination of Large Language Models: A Survey**. arXiv 2024. https://arxiv.org/abs/2406.04244
- BAAI. **bge-base-en-v1.5 model card**. Hugging Face. https://huggingface.co/BAAI/bge-base-en-v1.5

