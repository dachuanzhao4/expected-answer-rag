# Pilot Results Review and Revised Experimental Plan

**Project:** Leakage-aware query reformulation for private-like RAG  
**Input reviewed:** N=100 pilot results across Natural Questions, SciFact, and HotpotQA with BM25, dense retrieval, entity-counterfactual, and entity+value-counterfactual regimes  
**Action taken here:** No experiments were run. I only computed secondary diagnostics from the numbers supplied and revised the research plan accordingly.

---

## 1. Executive Assessment

The pilot is encouraging. It supports the central thesis that raw LLM-generated expansions can look strong on public benchmarks and then lose much of their advantage under entity-counterfactual private-like evaluation. The clearest evidence is on Natural Questions, especially for BM25 and HyDE/dense settings.

However, the current write-up should be made more conservative before it is shown to reviewers or collaborators. The present evidence is a strong pilot, not yet a proof. The two main reasons are:

1. **Small and easy retrieval setting:** N=100 and `max-corpus=200` create high-ceiling conditions. Several public scores are near 1.0, and some method differences may be unstable.
2. **Counterfactual artifact risk:** Entity and value renaming can change retrieval difficulty in ways unrelated to leakage, especially for dense encoders if aliases are code-like, unnatural, or out-of-distribution.

The right conclusion is:

> The pilot provides strong preliminary evidence that public benchmark gains from raw answer-shaped and pseudo-document expansions are sensitive to answer-prior leakage, especially on NQ. The next stage should quantify this with full or hard-negative corpora, per-query leakage labels, confidence intervals, and alias-naturalness controls.

Do **not** write that the pilot "proves" leakage, that the "illusion is shattered," or that dense retrievers generally fail under entity renaming. Those are top-conference-dangerous claims until the full-scale, statistically controlled results exist.

---

## 2. Literature Boundary Update

The novelty boundary remains the same as in the prior memo, but the pilot strengthens the paper's positioning.

The closest direct prior work is still **Yoon et al. 2025, "Hypothetical Documents or Knowledge Leakage? Rethinking LLM-based Query Expansion"**, which studies whether LLM-generated hypothetical documents contain information entailed by gold evidence and whether such leakage inflates query-expansion gains in fact verification. This project must be positioned as extending that line to QA-style and private-like RAG settings with entity-counterfactual evaluation, answer-candidate-constrained reformulation, and retriever-specific instability analysis.

Mandatory related baselines remain:

- **HyDE:** generates hypothetical documents and encodes them for dense retrieval.
- **Query2doc:** expands queries with LLM-generated pseudo-documents for sparse and dense retrieval.
- **Corpus-Steered Query Expansion:** uses corpus-originated evidence from first-pass retrieval to reduce reliance on pure LLM priors.
- **Entity-renaming robustness work:** establishes that entity renaming is a known robustness intervention, so the novelty here is not renaming itself but its use as a leakage-sensitive retrieval/RAG evaluation protocol.

References are listed at the end of this memo.

---

## 3. Important Correction to the Status Memo

The status memo currently says:

> "This document summarizes the findings from the initial N=5 stress tests..."

but the tables report:

> N=100 queries sampled per dataset.

Revise the wording to:

> This document summarizes N=100 pilot stress tests following earlier N=5 dry-run validation.

This small inconsistency will matter in a paper-facing workflow because reviewers and collaborators will read it as a reproducibility red flag.

---

## 4. Secondary Diagnostics from the Supplied Results

### 4.1 Key diagnostic: delta versus query-only

The most important number is not the raw score. It is the method's gain relative to `query_only` in each regime.

For BM25 on NQ:

| Method | Public score | Public delta vs query | CF score | CF delta vs query | Interpretation |
|---|---:|---:|---:|---:|---|
| `query_only` | 0.699 | 0.000 | 0.639 | 0.000 | Baseline drops by 0.060 under renaming. |
| `raw_expected_answer_only` | 0.894 | +0.195 | 0.688 | +0.049 | Large public gain mostly disappears under CF. |
| `hyde_doc_only` | 0.853 | +0.154 | 0.582 | -0.057 | Public gain reverses under CF. |
| `query2doc_concat` | 0.906 | +0.207 | 0.721 | +0.082 | Still helps under CF, though less. |
| `generative_relevance_feedback` | 0.823 | +0.124 | 0.771 | +0.132 | Surprisingly stable and should be elevated as a key baseline. |
| `answer_candidate_constrained_template` | 0.779 | +0.080 | 0.647 | +0.008 | Safe but modest; template-only is not strong enough as a primary method. |
| `concat_query_answer_constrained` | 0.783 | +0.084 | 0.679 | +0.040 | Safer than raw expansion, but not as strong as Query2doc/GRF in this pilot. |

This is strong evidence for a leakage-sensitive phenomenon on NQ, but it also shows that the best mitigation story may not be "templates win." A more defensible story is:

> Raw answer-like expansions are highly leakage-sensitive; query-preserving and corpus-grounded expansion routes are more stable; answer-candidate-constrained templates offer a stricter safety-control point but need improved integration, especially with dense retrieval.

### 4.2 Public-to-CF excess instability

Define:

```text
excess_instability(method) = (score_public(method) - score_CF(method))
                           - (score_public(query_only) - score_CF(query_only))
```

This subtracts the baseline difficulty introduced by renaming. It is a better paper metric than raw public-to-CF drop.

Average BM25 excess instability across the three pilot datasets:

| Method | Avg excess instability | Pilot interpretation |
|---|---:|---|
| `hyde_doc_only` | +0.093 | Most unstable among selected methods. |
| `raw_expected_answer_only` | +0.087 | Strong leakage-sensitivity signal. |
| `query2doc_concat` | +0.066 | Still somewhat unstable but often useful. |
| `answer_candidate_constrained_template` | +0.025 | More stable but lower absolute utility. |
| `concat_query_answer_constrained` | +0.025 | Similar stability with better utility than template-only. |
| `generative_relevance_feedback` | +0.014 | Surprisingly stable; important baseline. |
| `masked_expected_answer_only` | -0.014 | Stable mainly because it is often weak. |
| `corpus_steered_expansion_concat` | -0.027 | Stable but underperforming; likely integration issue. |

This supports the paper's core metric choice. The full paper should report excess instability with paired confidence intervals.

### 4.3 Entity+value counterfactual results sharpen the NQ story

For BM25 NQ entity+value counterfactual:

| Method | NQ E+V score | Delta vs query-only |
|---|---:|---:|
| `query_only` | 0.639 | 0.000 |
| `raw_expected_answer_only` | 0.640 | +0.001 |
| `hyde_doc_only` | 0.644 | +0.005 |
| `query2doc_concat` | 0.683 | +0.044 |
| `generative_relevance_feedback` | 0.741 | +0.102 |
| `answer_candidate_constrained_template` | 0.688 | +0.049 |
| `concat_query_answer_constrained` | 0.708 | +0.069 |

This is one of the strongest pilot findings: scrambling both entities and values nearly eliminates the raw expected-answer advantage on NQ, while query-preserving or feedback-style methods still retain some value.

The paper should emphasize this as a controlled diagnostic, but still avoid saying it "proves" memorization. It supports the interpretation that raw expected-answer gains are heavily dependent on public answer priors in this pilot setting.

### 4.4 Dense results are mixed, not simply anti-HyDE

Dense NQ is consistent with leakage sensitivity:

| Method | NQ public | Public delta | NQ CF | CF delta | Interpretation |
|---|---:|---:|---:|---:|---|
| `query_only` | 0.907 | 0.000 | 0.705 | 0.000 | Baseline itself is very strong in the small public pool. |
| `hyde_doc_only` | 0.956 | +0.049 | 0.651 | -0.054 | HyDE flips from helpful to harmful on CF NQ. |
| `raw_expected_answer_only` | 0.886 | -0.021 | 0.646 | -0.059 | Raw answer-only is not strong for dense even in public NQ. |
| `generative_relevance_feedback` | 0.887 | -0.020 | 0.707 | +0.002 | Stable, but not a public gain. |
| `concat_query_answer_constrained` | 0.880 | -0.027 | 0.695 | -0.010 | Query-preserving template is stable but not strong. |

The dense story should therefore be revised:

- Good: HyDE shows a clear NQ public-to-CF sensitivity pattern.
- Caution: dense `query_only` is already very high in public NQ with `max-corpus=200`, leaving little room for expansion methods.
- Caution: dense performance may be hurt by unnatural synthetic aliases, not only by absence of world knowledge.
- Required next control: compare coded aliases against naturalistic aliases.

---

## 5. Method-Level Feedback

### 5.1 `raw_expected_answer_only`

Keep it, but frame it as a **diagnostic leakage probe**, not a proposed method.

Current evidence:

- Strong public BM25 NQ gain: +0.195 over query-only.
- Much smaller NQ CF gain: +0.049.
- No gain under NQ entity+value CF: +0.001.

This is excellent for the paper's diagnosis section.

### 5.2 `hyde_doc_only`

Keep as a mandatory baseline and as a high-risk pseudo-document expansion condition.

Current evidence:

- BM25 NQ: +0.154 public, -0.057 CF.
- Dense NQ: +0.049 public, -0.054 CF.
- BM25 HotpotQA entity+value CF: -0.268 below query-only.

But revise the language. Do not say "HyDE's illusion shattered." Say:

> HyDE-style pseudo-documents are highly sensitive to entity-counterfactual evaluation in this pilot, especially on NQ and HotpotQA.

### 5.3 `query2doc_concat`

Do not treat Query2doc as simply leakage-prone or weak. In this pilot it is often one of the better query-preserving methods.

Current evidence:

- BM25 NQ public: +0.207.
- BM25 NQ CF: +0.082.
- BM25 NQ E+V: +0.044.
- Dense average is relatively stable.

This should remain a central baseline, and the paper should compare whether its retained CF gains come from useful reformulation rather than concrete answer injection.

### 5.4 `generative_relevance_feedback`

Elevate this baseline. It is more important than the original plan suggested.

Current evidence:

- BM25 NQ CF: +0.132 over query-only.
- BM25 NQ E+V: +0.102.
- Dense HotpotQA CF: +0.055.

This may become the strongest practical baseline because it preserves or recovers useful retrieval terms without relying as heavily on public answer strings. The full paper should include it in main tables, not just appendix.

### 5.5 `corpus_steered_expansion_concat`

Do not discard it yet, but the current implementation likely needs diagnosis.

Current evidence:

- It underperforms query-only almost everywhere.
- This conflicts with the motivation of corpus-steered expansion as a corpus-grounded way to reduce hallucination.

Likely causes to check:

1. First-pass retrieval may be too weak, so the selected corpus text is noisy.
2. Expansion text may be too long and may swamp the original query in BM25.
3. Concatenation may overweight feedback terms.
4. The implementation may not match the CSQE design closely enough.

Revised plan:

- Add `corpus_steered_key_sentences_only`.
- Add `concat_query_corpus_steered_short` with strict length budget.
- Add `RRF(query, corpus_steered)` rather than only concatenation.
- Log first-pass recall@k and whether selected sentences come from relevant documents.
- Treat CSQE as a grounding baseline, not as a failed method, until implementation diagnostics are complete.

### 5.6 `masked_expected_answer_only`

Move this out of the main method set. It is too destructive as a standalone retrieval query.

Current evidence:

- BM25 HotpotQA public: -0.404 below query-only.
- BM25 HotpotQA E+V: -0.372.
- Dense NQ public: -0.136.

The paper should still include masking, but mainly as:

- `concat_query_masked_expected`
- `RRF(query, masked_expected)`
- `oracle_gold_removed_expected_answer`
- leakage-rate ablation

Do not make `masked_expected_answer_only` a core mitigation method.

### 5.7 `generic_mask_slot`

Keep only as a negative control. It consistently underperforms and mainly shows that removing concrete content without preserving relation structure is not sufficient.

### 5.8 `random_span_masking` and `entity_only_masking`

Keep as ablations, but not as central baselines. Their behavior suggests that partial masking can leave substantial leakage intact or remove useful anchors.

The key paper question should be:

> Which masking strategy removes new answer candidates while preserving query anchors and relation intent?

not:

> Does any mask improve retrieval?

### 5.9 `wrong_answer_injection`

This result needs an integrity check.

In several BM25 settings, `wrong_answer_injection` is nearly identical to `query_only`. That could be a valid finding if wrong answers are ignored by BM25 when concatenated with a strong query, but it could also indicate that the control is effectively a no-op.

Before scaling, verify:

- The injected wrong answer is actually present in the retrieval query string.
- The wrong answer is not accidentally filtered by preprocessing.
- The cache key differs from query-only.
- There are variants for `wrong_answer_only`, `concat_query_wrong_answer`, and `RRF(query, wrong_answer)`.
- Wrong answers are type-compatible hard negatives, not random unrelated strings.

### 5.10 `answer_candidate_constrained_template`

The method is promising but not yet a clear winner.

Current evidence:

- Strong on SciFact BM25, including CF and E+V ceilings.
- Modestly helpful on BM25 NQ, especially when concatenated.
- Weak on dense NQ/HotpotQA when used alone.

Revised framing:

> Answer-candidate-constrained reformulation is a leakage-control intervention. Its value should be judged by candidate injection rate, excess instability, and private-like robustness, not only by public nDCG peak score.

Revised implementation:

- Use it mainly in query-preserving modes: concatenation, RRF, and weighted RRF.
- Add relation-preserving templates with dataset-specific schema fields.
- Avoid template-only dense retrieval as a main headline condition.

---

## 6. Revised Research Questions

### RQ1. Leakage prevalence

How often do LLM-generated expansions introduce exact answers, aliases, renamed answers, public-original answers in counterfactual settings, bridge entities, dates, values, or evidence-entailed statements not present in the original query?

### RQ2. Leakage-performance coupling

Are retrieval gains concentrated among leakage-positive expansions?

### RQ3. Counterfactual instability

Do raw answer-shaped expansions and pseudo-document expansions show larger excess instability than query-only under entity-counterfactual and entity+value-counterfactual evaluation?

### RQ4. Mitigation through query-preserving reformulation

Can query-preserving, answer-candidate-constrained reformulations reduce candidate injection while retaining useful retrieval terms?

### RQ5. Corpus grounding

Do feedback-style or corpus-steered expansions reduce unsupported candidate injection relative to pure LLM-prior expansion?

### RQ6. Retriever sensitivity

Are BM25, BM25+RM3, dense retrieval, hybrid retrieval, concatenation, and late fusion sensitive to different leakage modes?

### RQ7. Alias artifact control

How much of the public-to-counterfactual drop is caused by leakage removal versus unnatural alias artifacts, especially in dense retrieval?

---

## 7. Revised Hypotheses

### H1. Raw answer-shaped expansion is leakage-sensitive.

On public entity-centric QA, `raw_expected_answer_only` will often improve BM25 retrieval by inserting answer-bearing or answer-adjacent tokens. Under entity+value counterfactual evaluation, this advantage should shrink sharply.

Pilot status: **supported on NQ BM25**.

### H2. HyDE is sensitive to answer-prior mismatch, but not uniformly bad.

HyDE-style pseudo-documents will be strong on some public settings but can become unstable when generated content follows public priors that no longer match the counterfactual corpus.

Pilot status: **supported on NQ and HotpotQA; needs full-scale validation**.

### H3. Query-preserving expansion is safer than expansion-only retrieval.

Methods that retain the original query, such as Query2doc concat, GRF, constrained-template concat, and RRF variants, will be more robust than expansion-only methods.

Pilot status: **supported; strengthen this as a central design principle**.

### H4. Mask-only retrieval is not a viable main method.

Masking can reduce leakage, but using the masked text alone often destroys too much retrieval signal.

Pilot status: **supported**.

### H5. Candidate-constrained templates reduce leakage but need fusion.

Answer-candidate-constrained templates will have low candidate injection rates and lower excess instability, but their absolute retrieval utility will depend on concatenation, RRF, or weighted fusion.

Pilot status: **partially supported**.

### H6. Corpus-grounded methods may be robust but implementation-sensitive.

GRF and corpus-steered variants should be evaluated as serious baselines. If CSQE underperforms, the paper must distinguish method failure from implementation/integration failure.

Pilot status: **GRF looks strong; CSQE needs debugging**.

### H7. Dense counterfactual degradation requires alias controls.

Dense retrieval drops under counterfactual renaming may reflect both leakage removal and alias distribution shift. Naturalistic alias controls are required.

Pilot status: **required before making strong dense claims**.

---

## 8. Revised Experimental Design

### 8.1 Evaluation regimes

Keep three regimes, but rename them precisely:

1. **Public-original:** original benchmark.
2. **Entity-counterfactual:** consistent entity renaming across queries, corpus, answers, supporting facts, and qrels.
3. **Entity+value-counterfactual:** consistent entity plus date/value/number rewriting, used as an ablation rather than the main benchmark.

Use **entity-counterfactual** as the main technical term and **private-like** as the motivation.

### 8.2 Add alias naturalness controls

The current alias examples such as `Employee ZQ-17` and `Region PX-3` are useful stress-test aliases but may be hostile to dense encoders.

Add two alias modes:

1. **Naturalistic aliases:** plausible but fictitious names, organizations, places, titles, and project names.
   - PERSON: `Mara Vellin`, `Tomas Edrin`
   - ORGANIZATION: `Veyron Institute`, `Northlake Council`
   - LOCATION: `Eldmere`, `Rovinia Province`
   - WORK/TITLE: `The Lantern Archive`, `Project Aster`

2. **Coded aliases:** ID-like labels.
   - PERSON: `Person ZQ-17`
   - ORG: `Unit TQ-5`
   - LOCATION: `Site LM-42`

The main paper should use naturalistic aliases. Coded aliases should be an ablation showing an extreme private-enterprise setting.

### 8.3 Corpus scale plan

Do not scale only by increasing query count while keeping `max-corpus=200`. The next stage should prioritize harder retrieval pools.

Recommended scale ladder:

| Stage | Query count | Corpus setting | Purpose |
|---|---:|---|---|
| Pilot sanity | 100 | `max-corpus=200` | Already completed; thesis signal. |
| Hard-pool validation | 300-500 | 2k-10k docs/query dataset pool with hard negatives | Check whether signal survives harder distractors. |
| Main run | 500-1000 per dataset if feasible | Full corpus or large fixed pool | Paper tables. |
| Robustness run | 1000+ where feasible | Full corpus | Confidence intervals and stratification. |

For top-conference strength, full corpus is ideal for at least the primary dataset. If full counterfactual corpus generation is too slow, use a **fixed hard-negative pool** rather than a tiny random pool:

- all relevant documents;
- top-k BM25 hard negatives from original query;
- top-k dense hard negatives from original query;
- type-matched random distractors;
- the same pool transformed consistently for public and counterfactual regimes.

Always report corpus pool size and construction method.

### 8.4 Dataset roles

Keep the three datasets, but assign clearer roles.

#### Natural Questions

Primary public-prior leakage benchmark.

Use for:

- exact answer leakage;
- alias leakage;
- public-original answer appearing in CF generation;
- entity+value counterfactual stress test.

#### HotpotQA

Primary multi-hop and bridge-entity leakage benchmark.

Use for:

- final-answer leakage;
- bridge-entity leakage;
- support-document recall;
- any-support vs all-support retrieval metrics.

Add metrics:

- Recall@20 for retrieving at least one supporting document.
- Recall@50 for retrieving all supporting documents.
- Bridge entity injection rate.

#### SciFact

Evidence-entailment and domain-specific benchmark.

Do not rely on exact answer leakage because SciFact is claim verification rather than answer-string QA.

Use for:

- evidence-entailment leakage;
- scientific entity/value perturbation sensitivity;
- claim vocabulary drift;
- comparison with leakage findings in fact verification.

Because SciFact shows ceiling effects in this pilot, full-corpus evaluation and harder negative pools are especially important.

### 8.5 Main method set

Reduce the main table to methods that answer the scientific question. Move some stress controls to appendix.

#### Main methods

1. `query_only`
2. `BM25 + RM3`
3. `dense_query_only`
4. `hybrid_RRF(BM25, dense)` if feasible
5. `raw_expected_answer_only` as leakage probe
6. `hyde_doc_only`
7. `query2doc_concat`
8. `generative_relevance_feedback`
9. `corpus_steered_short_concat`
10. `RRF(query, corpus_steered)`
11. `concat_query_masked_expected`
12. `RRF(query, masked_expected)`
13. `concat_query_answer_constrained`
14. `RRF(query, answer_constrained)`
15. `weighted_RRF(query, answer_constrained)`

#### Appendix / control methods

1. `masked_expected_answer_only`
2. `generic_mask_slot`
3. `random_span_masking`
4. `entity_only_masking`
5. `wrong_answer_only`
6. `concat_query_wrong_answer`
7. `RRF(query, wrong_answer)`
8. `oracle_gold_removed_expected_answer`
9. `post_hoc_public_answer_removed_generation`
10. `length_matched_neutral_filler`

### 8.6 Generation integrity checks

Before scaling, add checks that can be run over cached generations:

1. **Regime-specific cache keys:** public and CF generations must not accidentally share the same generated text unless intentionally transformed.
2. **Prompt provenance:** record prompt version, model name, decoding parameters, and timestamp.
3. **Expansion string logging:** dump the exact retrieval string for every query/method.
4. **Wrong-answer verification:** confirm wrong answers are present after preprocessing.
5. **Alias replacement verification:** confirm original entities are absent from CF corpus/query text except in diagnostic metadata.
6. **Answer preservation verification:** confirm the renamed answer still appears in at least one relevant document when the task requires answer-string evidence.
7. **Support preservation verification:** for HotpotQA and SciFact, confirm supporting facts/rationales are still recoverable after replacement.

### 8.7 Leakage metrics

Report leakage metrics before retrieval metrics. The strongest paper will show mechanism, not only score changes.

#### QA-style leakage metrics

- Exact gold answer present in expansion.
- Gold answer alias present.
- Public-original answer present in CF expansion.
- Counterfactual answer alias present in CF expansion.
- New concrete answer candidate introduced.
- Unsupported entity introduced.
- Unsupported date/value introduced.

#### HotpotQA-specific leakage metrics

- Final answer present.
- Bridge entity present.
- Supporting-page title present.
- Public-original bridge entity present in CF expansion.

#### SciFact-specific leakage metrics

- Evidence-entailed sentence in expansion.
- Claim label implied by expansion.
- Scientific entity overlap with gold evidence.
- Unsupported biomedical/scientific term injection.

### 8.8 Retrieval metrics

Main metrics:

- nDCG@10
- Recall@10
- Recall@20
- Recall@50 for multi-hop support retrieval
- MRR@10 for QA-style datasets

Counterfactual metrics:

- Public-to-CF drop.
- Excess instability over query-only.
- CF delta versus query-only.
- Leakage-positive versus leakage-negative performance gap.
- Per-query win/tie/loss versus query-only.

### 8.9 Statistical analysis

For every main metric:

1. Use paired bootstrap confidence intervals over queries.
2. Use paired permutation/randomization tests for primary comparisons.
3. Apply multiple-comparison control for large method tables or pre-register a small set of primary contrasts.
4. Report win/tie/loss counts per query.
5. Report effect sizes, not only p-values.

Primary contrasts:

1. `raw_expected_answer_only` vs `query_only` in public and CF.
2. `hyde_doc_only` vs `query_only` in public and CF.
3. `query2doc_concat` vs `concat_query_answer_constrained`.
4. `generative_relevance_feedback` vs `concat_query_answer_constrained`.
5. `RRF(query, answer_constrained)` vs `query_only`.
6. Excess instability of raw/HyDE versus constrained/fusion methods.

---

## 9. Revised Paper Claims

### Claims now supported by the pilot, with cautious wording

1. In NQ BM25 pilots, raw expected-answer expansion has a large public gain that shrinks sharply under entity-counterfactual and entity+value-counterfactual evaluation.
2. HyDE-style expansion is sensitive to counterfactual renaming in the pilot, especially on NQ and HotpotQA.
3. Mask-only retrieval is too destructive to be a main mitigation method.
4. Query-preserving methods are more promising than expansion-only methods.
5. Entity+value counterfactual evaluation is a useful ablation for separating answer/value priors from relation-level reformulation.

### Claims that remain unsafe

1. Public benchmark gains are mostly contamination-driven.
2. HyDE generally fails in private RAG.
3. Dense retrievers generally struggle with counterfactual entities.
4. Answer-candidate-constrained templates are the best method.
5. SciFact retrieval is anchored more on nouns than values.
6. HotpotQA multi-hop logic collapses under temporal/numeric perturbation.

### Stronger final target claim

A reviewer-safe final claim would be:

> Across public and entity-counterfactual retrieval settings, answer-bearing LLM expansions show higher excess instability than query-preserving and leakage-constrained reformulations. This suggests that public benchmark gains from LLM query expansion can conflate useful reformulation with answer-prior injection. Entity-counterfactual evaluation and answer-candidate-constrained reformulation provide a practical protocol for measuring and reducing this failure mode in private-like RAG.

---

## 10. Revised Next-Step Checklist

### Priority 0: Fix status wording and result interpretation

- [ ] Replace "N=5 stress tests" with "N=100 pilot stress tests following N=5 dry runs."
- [ ] Replace "proves" with "supports" or "is consistent with."
- [ ] Replace "HyDE's Illusion Shattered" with "HyDE is counterfactually sensitive in the pilot."
- [ ] Add `max-corpus=200` caveat to every pilot table caption.

### Priority 1: Sanity checks before scale-up

- [ ] Verify public and CF generation caches are separate.
- [ ] Verify exact retrieval strings for each method.
- [ ] Verify `wrong_answer_injection` is not a no-op.
- [ ] Verify answer preservation in relevant CF documents.
- [ ] Verify no original public entities remain in CF query/corpus text.
- [ ] Verify dense alias artifacts using naturalistic aliases.
- [ ] Add per-query leakage labels to result dumps.

### Priority 2: Method cleanup

- [ ] Promote GRF to main baseline.
- [ ] Keep Query2doc as a central baseline.
- [ ] Move mask-only and generic mask to appendix controls.
- [ ] Add `concat_query_masked_expected` and `RRF(query, masked_expected)` if not already in the current result set.
- [ ] Add `RRF(query, answer_constrained)` and weighted RRF.
- [ ] Add shorter and RRF versions of corpus-steered expansion.
- [ ] Add BM25+RM3.

### Priority 3: Scale and rigor

- [ ] Run N=300-500 hard-pool validation before full corpus.
- [ ] Run full corpus or large fixed hard-negative pool for main results.
- [ ] Add paired bootstrap CIs.
- [ ] Add paired permutation tests for primary contrasts.
- [ ] Add leakage-positive vs leakage-negative stratification.
- [ ] Add qualitative examples.
- [ ] Add human spot checks for renamed examples.

### Priority 4: Paper table plan

Main tables should be revised to the following:

1. **Table 1:** Leakage rates by method and dataset.
2. **Table 2:** Public-original retrieval performance.
3. **Table 3:** Entity-counterfactual retrieval performance.
4. **Table 4:** Entity+value counterfactual ablation.
5. **Table 5:** Excess instability over query-only.
6. **Table 6:** Leakage-positive vs leakage-negative retrieval gains.
7. **Table 7:** Method ablations: masking, constrained templates, RRF, wrong-answer controls.

Main figures:

1. Leakage rate versus retrieval gain.
2. Excess instability by method.
3. Per-query public-to-CF drop histograms.
4. Qualitative examples of public leak success, CF raw failure, and constrained/fusion recovery.

---

## 11. Updated Minimum Viable Paper Package

The new minimum package for a strong submission is:

1. Three datasets: NQ, HotpotQA, SciFact.
2. Public, entity-counterfactual, and entity+value-counterfactual regimes.
3. At least one realistic alias mode and one coded-alias ablation.
4. BM25, BM25+RM3, one dense retriever, and ideally one hybrid RRF retriever.
5. Main baselines: query-only, HyDE, Query2doc, GRF, CSQE-style corpus-steered expansion.
6. Main interventions: query-preserving masked expansion and answer-candidate-constrained fusion.
7. Leakage metrics before retrieval metrics.
8. Excess instability over query-only as a headline metric.
9. Paired confidence intervals and primary statistical tests.
10. Qualitative examples and human spot checks.

If full-corpus generation is infeasible, the paper can still be viable with large fixed hard-negative pools, but the pool construction must be rigorous and fully reported.

---

## 12. Recommended Revised Abstract Sketch

LLM-generated query expansions can improve retrieval by clarifying underspecified information needs, but they can also inject answer-bearing content from the model's parametric priors. We study this confound in QA-style and private-like RAG settings. We introduce entity-counterfactual retrieval benchmarks that consistently rename entities, and optionally values, across queries, corpora, and relevance metadata while preserving evidence structure. Across public and counterfactual settings, we measure exact-answer, alias, bridge-entity, and evidence-entailment leakage in generated expansions, and evaluate their relationship to retrieval gains under sparse and dense retrieval. Pilot results show that raw answer-shaped and HyDE-style expansions can have large public gains that shrink or reverse under counterfactual evaluation, while query-preserving and feedback-style methods are more stable. We further evaluate answer-candidate-constrained reformulations and fusion strategies as leakage-aware alternatives for private-like RAG.

---

## 13. References

- Gao et al. 2023. **Precise Zero-Shot Dense Retrieval without Relevance Labels.** ACL. https://aclanthology.org/2023.acl-long.99/
- Wang et al. 2023. **Query2doc: Query Expansion with Large Language Models.** EMNLP. https://aclanthology.org/2023.emnlp-main.585/
- Lei et al. 2024. **Corpus-Steered Query Expansion with Large Language Models.** EACL. https://aclanthology.org/2024.eacl-short.34/
- Yoon et al. 2025. **Hypothetical Documents or Knowledge Leakage? Rethinking LLM-based Query Expansion.** ACL Findings. https://aclanthology.org/2025.findings-acl.980/
- Yan et al. 2022. **On the Robustness of Reading Comprehension Models to Entity Renaming.** NAACL. https://aclanthology.org/2022.naacl-main.37/
- Thakur et al. 2021. **BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.** NeurIPS Datasets and Benchmarks. https://arxiv.org/abs/2104.08663
- Kwiatkowski et al. 2019. **Natural Questions: A Benchmark for Question Answering Research.** TACL. https://aclanthology.org/Q19-1026/
- Yang et al. 2018. **HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.** EMNLP. https://aclanthology.org/D18-1259/
- Wadden et al. 2020. **Fact or Fiction: Verifying Scientific Claims.** EMNLP. https://aclanthology.org/2020.emnlp-main.609/
