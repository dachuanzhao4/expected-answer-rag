# Leakage-Aware Query Reformulation for Private-Like RAG

**Updated project and experimental design memo**  
**Role assumption:** senior AI scientist review for a top AI/NLP/IR conference submission  
**Status:** design review only; no experiments should be run from this memo

---

## 0. Executive Assessment

The core instinct in the original memo is strong: the paper should not be framed as a generic *expected-answer retrieval* method. That would collide with a broad and growing line of LLM-based query expansion, pseudo-document generation, generative relevance feedback, query rewriting, and retrieval-generation-loop work.

The most important update from the literature search is that the paper cannot claim that retrieval-query-expansion leakage is an untouched problem. **Yoon et al. (ACL Findings 2025), “Hypothetical Documents or Knowledge Leakage? Rethinking LLM-based Query Expansion,” directly studies whether LLM-generated hypothetical documents contain information entailed by gold evidence and whether this knowledge leakage inflates query-expansion gains in fact-verification benchmarks.** This is highly relevant and must be treated as a central prior work, not a peripheral citation.

The revised top-conference positioning should therefore be:

> LLM-generated retrieval expansions entangle two mechanisms: genuine reformulation of the information need and answer-prior injection from the generator. Existing leakage analyses focus mainly on generated pseudo-documents and fact verification. We extend the problem to QA-style and private-domain RAG by introducing entity-counterfactual private-like retrieval benchmarks, answer-candidate leakage metrics, and leakage-aware reformulations that avoid introducing new concrete answer candidates.

The best paper shape is now:

1. **Diagnosis:** quantify how much raw LLM expansion gain is correlated with answer-bearing or evidence-entailed content.
2. **Counterfactual evaluation:** compare public benchmarks with consistently renamed, private-like versions that preserve corpus evidence and qrels while suppressing public entity priors.
3. **Mitigation:** compare query-aware masking and answer-agnostic template reformulation against HyDE, Query2doc-style pseudo-documents, generative relevance feedback, and corpus-steered expansion.
4. **Practical implication:** show when LLM query expansion is helpful, when it is prior-driven, and which reformulations are safer for private or enterprise RAG.

---

## 1. Validated Claim Matrix

| Original or implied claim | Validation status | Evidence / correction | Revised paper-safe wording |
|---|---:|---|---|
| A generic “expected answer retrieval” paper is too close to existing work. | **Supported** | HyDE, Query2doc, Query Expansion by Prompting LLMs, and Generative Relevance Feedback already use LLM-generated text or terms to improve retrieval. | “Expected-answer retrieval should be an ablation inside a leakage-aware reformulation study.” |
| LLM-generated pseudo-documents improve retrieval and are a strong baseline. | **Supported** | HyDE generates hypothetical documents and encodes them for zero-shot dense retrieval; Query2doc expands queries with LLM-generated pseudo-documents and reports improvements for sparse and dense retrieval. | “HyDE and Query2doc are mandatory baselines.” |
| HyDE’s encoder bottleneck can filter away incorrect generated details. | **Supported but incomplete** | The HyDE paper explicitly motivates the dense encoder as a bottleneck that filters generated false details, but later leakage work challenges whether gains are always due to useful hypothetical-document semantics. | “HyDE’s bottleneck may reduce some hallucinated details, but leakage-sensitive evaluation is still required.” |
| Query2doc improves BM25 by 3–15% on ad-hoc IR datasets. | **Supported** | Query2doc reports BM25 gains on MS MARCO and TREC DL. | Keep, with citation. |
| Public QA benchmark gains may be inflated by LLM priors. | **Plausible but empirical** | LLM benchmark contamination is a known evaluation risk, and Yoon et al. show leakage-related gains for fact verification, but the exact magnitude for NQ/HotpotQA must be measured. | “We test whether public QA retrieval gains correlate with answer-bearing generation.” |
| Prior work has not studied leakage in LLM-based query expansion. | **Incorrect / outdated** | Yoon et al. 2025 directly studies knowledge leakage in LLM-based query expansion. | “Prior work has begun studying leakage, but has not fully addressed private-like QA/RAG, entity-counterfactual evaluation, or answer-agnostic reformulation.” |
| A renamed private-like benchmark can suppress memorized public priors while preserving corpus evidence. | **Methodologically plausible; not yet proven** | Entity renaming has been used for robustness analysis in reading comprehension and document-level relation extraction, but the retrieval/RAG setting needs careful validation. | “We propose and validate an entity-counterfactual private-like retrieval protocol.” |
| BEIR is an appropriate heterogeneous retrieval evaluation suite. | **Supported** | BEIR is a heterogeneous zero-shot IR benchmark spanning multiple tasks and domains. | Keep. |
| Natural Questions is public, real-query, Wikipedia-grounded QA. | **Supported** | NQ consists of real anonymized Google search queries paired with Wikipedia-page annotations. | Keep, but ensure answer metadata is available if using BEIR-derived NQ. |
| HotpotQA is useful for multi-hop retrieval. | **Supported** | HotpotQA contains 113K Wikipedia-based question-answer pairs with supporting facts and multi-hop reasoning requirements. | Keep. |
| SciFact is a specialized evidence-focused dataset. | **Supported** | SciFact contains expert-written scientific claims and evidence-containing abstracts with support/refute labels and rationales. | Keep. |
| Answer-agnostic templates are not covered by existing query expansion work. | **Partially supported** | There is related work on query rewriting, step-back prompting, and corpus-steered expansion. The novelty is not generic rewriting; it is forbidding new concrete answer candidates and measuring leakage. | “We introduce answer-candidate-constrained reformulation for leakage-aware retrieval.” |
| Repo-specific method availability and current results. | **Not externally validated in this review** | The repo was not inspected here. | Treat as implementation assumptions to verify before submission. |

---

## 2. Revised Paper Thesis

### One-sentence thesis

LLM-generated retrieval expansions can improve public QA retrieval by injecting answer-bearing priors as well as by clarifying the information need; entity-counterfactual private-like evaluation reveals this confound, and answer-candidate-constrained reformulations reduce leakage while preserving useful retrieval signal.

### Stronger working title candidates

1. **From Answer Priors to Evidence Retrieval: Leakage-Aware Query Expansion for Private-Like RAG**
2. **When Query Expansion Answers the Question: Measuring Leakage in LLM-Assisted Retrieval**
3. **Entity-Counterfactual Evaluation of LLM Query Expansion for Retrieval-Augmented Generation**
4. **Answer-Candidate-Constrained Reformulation for Leakage-Aware Retrieval**
5. **When HyDE Knows the Answer: Public Priors and Private-Like Retrieval Failure in LLM Query Expansion**

### Central claim to test, not assume

LLM-generated expansions can help retrieval through at least three mechanisms:

1. **Useful reformulation:** adding relation terms, context, or missing descriptors.
2. **Corpus alignment:** adding terms likely to appear in relevant evidence.
3. **Answer-prior injection:** adding the answer, answer aliases, or evidence-entailed facts not present in the query.

The paper should test whether public benchmark gains are partly driven by the third mechanism and whether leakage-aware methods improve robustness when answer priors are disrupted.

---

## 3. Revised Contributions

The submission should claim the following only if experiments support them:

1. **Problem extension:** a retrieval-specific formulation of *answer-prior leakage* in LLM-generated query expansions for QA-style and private-domain RAG, extending recent leakage analyses beyond fact verification.
2. **Entity-counterfactual private-like evaluation:** a reproducible benchmark transformation that consistently renames entities and optionally sensitive values across corpus, queries, answers, and relevance metadata, preserving qrels while suppressing public entity priors.
3. **Leakage measurement:** metrics for exact answer leakage, alias leakage, evidence-entailment leakage, unsupported entity/value injection, and public-to-renamed instability.
4. **Mitigation study:** a comparative evaluation of raw answer-shaped expansions, HyDE, Query2doc-style pseudo-documents, generative relevance feedback, corpus-steered expansion, query-aware masking, and answer-candidate-constrained templates.
5. **Retriever-regime analysis:** evidence about which retrieval settings are most sensitive to answer-prior leakage: BM25, BM25+RM3, dense retrieval, concatenation, RRF, and weighted RRF.

The submission should **not** claim:

- first LLM-generated query expansion method;
- first pseudo-document retrieval method;
- first work on leakage in LLM query expansion;
- universal superiority over HyDE, Query2doc, or generative relevance feedback;
- proof that public benchmark gains are mostly contamination-driven unless the evidence is very strong.

---

## 4. Updated Literature Review

### 4.1 LLM-generated query expansion and pseudo-documents

**HyDE.** Gao et al. propose Hypothetical Document Embeddings: generate a hypothetical document for the query, encode it with an unsupervised dense encoder, and retrieve real documents by vector similarity. HyDE is foundational because it directly raises the question of whether generated hypothetical content helps by capturing relevance structure or by injecting facts from the generator’s parametric memory.  
Reference: Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels*, ACL 2023. https://arxiv.org/abs/2212.10496

**Query2doc.** Wang et al. generate pseudo-documents with LLM prompting and concatenate them to queries, improving sparse and dense retrieval. Query2doc is close to `concat_query_raw_expected` and HyDE-style use of generated text, so it must be a primary baseline.  
Reference: Wang et al., *Query2doc: Query Expansion with Large Language Models*, EMNLP 2023. https://aclanthology.org/2023.emnlp-main.585/

**Query Expansion by Prompting LLMs.** Jagerman et al. study zero-shot, few-shot, and chain-of-thought prompting for LLM-based query expansion, explicitly relying on knowledge inherent in the model and evaluating on MS MARCO and BEIR. This reinforces why “LLM-generated expansion helps retrieval” is not novel by itself.  
Reference: Jagerman et al., *Query Expansion by Prompting Large Language Models*, 2023. https://arxiv.org/abs/2305.03653

**Generative Relevance Feedback.** Mackie et al. propose building probabilistic feedback models from LLM-generated text rather than top retrieved documents, reporting improvements over traditional PRF/RM3-style baselines. This is important because the proposed paper must compare against more than HyDE/Query2doc.  
Reference: Mackie et al., *Generative Relevance Feedback with Large Language Models*, 2023. https://arxiv.org/abs/2304.13157

**Corpus-Steered Query Expansion.** Lei et al. argue that LLM-only expansions can be misaligned with the target corpus due to hallucinations and outdated information, and propose incorporating corpus-originated texts from initially retrieved documents. This is a highly relevant mitigation baseline because it shifts expansions away from pure parametric prior.  
Reference: Lei et al., *Corpus-Steered Query Expansion with Large Language Models*, EACL 2024. https://aclanthology.org/2024.eacl-short.34/

**GOLFer.** Liu and Zhang propose filtering and combining smaller-LM generated documents for query expansion, addressing hallucination and practical deployment constraints. It is relevant as a recent hallucination-aware query expansion method.  
Reference: Liu and Zhang, *GOLFer: Smaller LMs-Generated Documents Hallucination Filter & Combiner for Query Expansion in Information Retrieval*, ACL Findings 2025. https://aclanthology.org/2025.findings-acl.8/

### 4.2 LLM query rewriting and retrieval-generation loops

**Rewrite-Retrieve-Read.** Ma et al. frame RAG as a pipeline in which the query itself is rewritten before retrieval, showing consistent improvements on QA tasks. This is related to answer-agnostic templates but not leakage-specific.  
Reference: Ma et al., *Query Rewriting in Retrieval-Augmented Large Language Models*, EMNLP 2023. https://aclanthology.org/2023.emnlp-main.322/

**IRCoT.** Trivedi et al. interleave chain-of-thought reasoning and retrieval for multi-step QA, where generated reasoning steps guide what to retrieve next. It is relevant because intermediate generated text can become a retrieval query, but the leakage risk should be analyzed separately.  
Reference: Trivedi et al., *Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions*, ACL 2023. https://aclanthology.org/2023.acl-long.557/

**Iter-RetGen.** Shao et al. use model outputs to retrieve more relevant knowledge iteratively, showing that generated outputs can act as retrieval signals. This is related but more focused on end-to-end RAG generation than leakage-controlled first-pass retrieval.  
Reference: Shao et al., *Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy*, EMNLP Findings 2023. https://aclanthology.org/2023.findings-emnlp.620/

**Step-Back Prompting.** Zheng et al. generate higher-level abstractions before reasoning. It is not a retrieval method per se, but it is related to answer-agnostic reformulation because both encourage abstracting away from the specific answer candidate.  
Reference: Zheng et al., *Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models*, ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/592da1445a51e54a3987958b5831948f-Abstract-Conference.html

### 4.3 Leakage, contamination, and evaluation validity

**Direct prior work: Hypothetical Documents or Knowledge Leakage?** Yoon et al. directly analyze whether generated documents contain information entailed by gold evidence and whether such leakage contributes to LLM-based query expansion gains in fact verification. This paper is the closest prior work and changes the novelty boundary of this project.  
Reference: Yoon et al., *Hypothetical Documents or Knowledge Leakage? Rethinking LLM-based Query Expansion*, ACL Findings 2025. https://aclanthology.org/2025.findings-acl.980/

**Benchmark contamination surveys and methods.** Xu et al. survey benchmark data contamination risks in LLM evaluation; Oren et al. propose a black-box statistical test for test-set contamination; LiveBench proposes frequently updated, automatically scored tasks to reduce contamination risk. These motivate contamination-resistant evaluation but are not retrieval-specific baselines.  
References:  
- Xu et al., *Benchmark Data Contamination of Large Language Models: A Survey*, 2024. https://arxiv.org/abs/2406.04244  
- Oren et al., *Proving Test Set Contamination in Black-Box Language Models*, ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/46e624c244cff669223d488defd4e835-Abstract-Conference.html  
- White et al., *LiveBench: A Challenging, Contamination-Free LLM Benchmark*, 2024. https://arxiv.org/abs/2406.19314

### 4.4 Entity renaming and counterfactual robustness

Entity renaming is not novel in itself. It has been used to test whether NLP models rely too heavily on entity names.

**Reading comprehension entity renaming.** Prior work studies whether machine reading comprehension models fail when entities are renamed and introduces scalable pipelines for replacing entity names.  
Reference: *On the Robustness of Reading Comprehension Models to Entity Renaming*, ACL ARR 2021 submission. https://openreview.net/forum?id=lXczoncSyt0

**Document-level relation extraction entity variation.** Meng et al. propose entity-renamed DocRE benchmarks and show robustness failures for both standard models and in-context learned LLMs. This supports the feasibility of entity-renamed benchmarks while warning that renaming can introduce artifacts.  
Reference: Meng et al., *On the Robustness of Document-Level Relation Extraction Models to Entity Name Variations*, ACL Findings 2024. https://aclanthology.org/2024.findings-acl.969/

### 4.5 Retrieval benchmarks

**BEIR.** BEIR is a heterogeneous zero-shot retrieval benchmark covering diverse tasks and domains. It is appropriate for robust retrieval evaluation, but each BEIR subset must be checked for answer metadata availability if leakage metrics require gold answer strings.  
Reference: Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*, NeurIPS Datasets and Benchmarks 2021. https://arxiv.org/abs/2104.08663

**Natural Questions.** NQ uses real anonymized Google queries and Wikipedia-page annotations with long and short answers. It is a good public-prior stress test but requires careful answer metadata alignment.  
Reference: Kwiatkowski et al., *Natural Questions: A Benchmark for Question Answering Research*, TACL 2019. https://aclanthology.org/Q19-1026/

**HotpotQA.** HotpotQA contains multi-hop Wikipedia-based QA with supporting facts, making it a useful test for bridging-entity leakage and multi-hop reformulation.  
Reference: Yang et al., *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*, EMNLP 2018. https://aclanthology.org/D18-1259/

**SciFact.** SciFact focuses on scientific claim verification with evidence-containing abstracts, labels, and rationales. It is useful for comparison with Yoon et al. and for domain-specific evidence retrieval, but leakage metrics should be entailment/evidence-based rather than answer-string-based.  
Reference: Wadden et al., *Fact or Fiction: Verifying Scientific Claims*, EMNLP 2020. https://aclanthology.org/2020.emnlp-main.609/

---

## 5. Refined Research Questions

**RQ1: Leakage prevalence.** How often do LLM-generated expansions contain exact answers, answer aliases, new answer candidates, or evidence-entailed facts not present in the original query?

**RQ2: Leakage-performance coupling.** Are retrieval gains concentrated among queries whose expansions contain answer-bearing or evidence-entailed content?

**RQ3: Public-to-private instability.** Do raw answer-shaped, HyDE, Query2doc, and GRF-style expansions degrade more under entity-counterfactual private-like renaming than query-only or leakage-aware methods?

**RQ4: Mitigation.** Can query-aware masking and answer-candidate-constrained templates reduce leakage while preserving retrieval quality?

**RQ5: Retriever sensitivity.** Are sparse, dense, concatenation, late-fusion, and weighted-fusion retrieval regimes sensitive to different forms of leakage?

**RQ6: Corpus grounding.** Does corpus-steered expansion reduce unsupported candidate injection relative to pure LLM-prior expansion?

---

## 6. Revised Hypotheses

**H1.** On public QA benchmarks, LLM-generated answer-shaped and pseudo-document expansions will often contain exact answers, aliases, or evidence-entailed facts not present in the query.

**H2.** Retrieval gains from raw expansions will be larger for leakage-positive expansions than for leakage-negative expansions.

**H3.** Under entity-counterfactual private-like renaming, raw expansions will show a larger performance drop than query-only and leakage-aware reformulations, after controlling for the baseline difficulty introduced by renaming.

**H4.** Query-aware masking will reduce exact answer and alias leakage while preserving part of the raw-expansion retrieval gain.

**H5.** Answer-candidate-constrained templates will have the lowest concrete-candidate injection rate and the best public-to-private stability, even if they do not always achieve the highest public benchmark score.

**H6.** Sparse retrieval will be especially sensitive to exact lexical answer leakage, while dense retrieval will remain sensitive to semantic/evidence-entailment leakage.

**H7.** Corpus-steered expansion will reduce unsupported candidate injection compared with pure parametric generation, but it may inherit errors from weak first-pass retrieval.

---

## 7. Experimental Design

### 7.1 Evaluation regimes

For every dataset, evaluate three regimes where feasible:

1. **Public-original:** the original benchmark text, qrels, and answer metadata.
2. **Entity-counterfactual private-like:** named entities are consistently replaced across corpus, queries, answers, supporting facts, and metadata.
3. **Entity-and-value-counterfactual private-like:** named entities plus sensitive dates, values, and numeric identifiers are consistently replaced where doing so preserves task semantics.

The second regime should be the main “private-like” condition. The third should be an ablation because numeric/date transformations are more likely to introduce semantic artifacts.

### 7.2 Why “entity-counterfactual” is better wording than only “private-like”

“Private-like” is a motivation, not a guarantee. The actual intervention is counterfactual renaming. The paper should use both terms carefully:

- **Entity-counterfactual benchmark** = the precise technical construct.
- **Private-like RAG stress test** = the intended real-world analogue.

This avoids reviewer criticism that synthetic renaming is not truly private data.

---

## 8. Benchmark Construction Protocol

### 8.1 Core requirements

The renaming pipeline must preserve:

1. document IDs and qrels;
2. query-document relevance structure;
3. entity coreference consistency;
4. coarse entity type;
5. answer evidence in relevant documents;
6. answer metadata after alias mapping;
7. supporting-fact metadata for HotpotQA and SciFact where applicable.

It should disrupt:

1. exact public entity names;
2. famous entity-answer associations;
3. answer lookup from public parametric memory alone;
4. memorized benchmark snippets when entity names are the main anchors.

### 8.2 Alias styles

Use two alias styles to separate prior suppression from unnatural-token artifacts.

| Alias style | Example | Purpose | Risk |
|---|---|---|---|
| Natural aliases | `Mira Halden`, `Norwick Institute`, `Lake Varos` | Reduces public priors while preserving language naturalness | Some names may carry unintended associations |
| Coded typed aliases | `Person P-041`, `Organization O-219`, `Site L-087` | Strong prior suppression and easy validation | Can harm dense retrieval due to unnatural text |

The main paper should report natural-alias results and include coded-alias results as a stricter robustness condition.

### 8.3 Type system

Recommended alias types:

- `PERSON`
- `LOCATION`
- `ORGANIZATION`
- `WORK_OF_ART` / `TITLE`
- `EVENT`
- `PRODUCT`
- `DATE`
- `TIME`
- `NUMBER`
- `QUANTITY`
- `MISC_ENTITY`

For dates and numbers, define explicit transformation rules and avoid changing values when the transformation would alter comparison logic or entailment labels.

### 8.4 Renaming steps

1. Load corpus, queries, qrels, answer metadata, and supporting facts.
2. Detect named entities using a high-recall NER system plus dataset-specific answer spans.
3. Resolve aliases and coreference within each dataset where feasible.
4. Build a deterministic alias table with one-to-one mappings.
5. Replace mentions consistently across corpus, queries, gold answers, supporting facts, and evaluation metadata.
6. Preserve qrels by document ID.
7. Validate automatic constraints.
8. Run human spot checks on a stratified sample before any large experiment.

### 8.5 Required validation checks

| Validation check | Pass criterion |
|---|---|
| Replacement coverage | At least 99% of targeted original entity mentions replaced in corpus/query text, measured after normalization. |
| Alias consistency | Same source entity maps to one alias across corpus, query, and answer metadata. |
| Type preservation | Alias type matches original entity type in at least 95% of sampled cases. |
| Qrel preservation | Same relevant document IDs remain valid after renaming. |
| Answer preservation | Remapped answer appears in at least one relevant document for QA datasets. |
| Supporting-fact preservation | HotpotQA/SciFact supporting sentence or abstract remains retrievable and semantically valid. |
| Prior suppression | No-context LLM answer accuracy drops substantially from public-original to entity-counterfactual conditions. |
| Query-only sanity | Query-only retrieval should not collapse under renaming; otherwise the benchmark artifact is too severe. |

### 8.6 Important control: delta over query-only

Renaming can make retrieval harder independently of leakage. Therefore report:

```text
method_instability = (score_public_method - score_private_method)
                   - (score_public_query_only - score_private_query_only)
```

This isolates instability beyond the baseline difficulty introduced by renaming.

---

## 9. Datasets

### 9.1 Minimum viable top-conference dataset set

1. **Natural Questions or NQ-open with retrieval qrels and answer metadata**  
   Use for public entity-centric QA and answer-string leakage.

2. **HotpotQA**  
   Use for multi-hop retrieval, bridge-entity leakage, and supporting-fact analysis.

3. **SciFact**  
   Use for specialized domain evidence retrieval and comparison to fact-verification leakage work.

### 9.2 Optional additions

- **FEVER**: useful for direct comparability to fact-verification leakage studies, but avoid overloading the paper.
- **FiQA**: useful for finance-oriented retrieval if private-domain motivation is important.
- **A semi-private or time-sliced enterprise-like corpus**: very valuable if publishable, but not required for the first submission.

### 9.3 Dataset-specific leakage definitions

| Dataset | Primary leakage signal | Notes |
|---|---|---|
| NQ | exact/alias answer string in expansion; unsupported candidate entities | Need reliable short-answer metadata. |
| HotpotQA | final answer leakage; bridge entity leakage; supporting-fact entailment | Multi-hop leakage may occur via bridge entity even if final answer is masked. |
| SciFact | evidence-entailed sentence leakage; claim-label leakage; unsupported biomedical/scientific entity injection | Exact answer strings are less meaningful. |

---

## 10. Methods

### 10.1 Core baselines

1. **`query_only`**  
   Retrieve with the original query only. This is the most important baseline.

2. **`BM25 + RM3`**  
   Traditional pseudo-relevance feedback baseline. Include if implementation time permits.

3. **`dense_query_only`**  
   Use a fixed zero-shot dense retriever. Suggested options: Contriever, E5, or BGE. Use one as mandatory and a second as optional robustness.

4. **`hyde`**  
   Generate a hypothetical supporting passage and retrieve using the generated document or its embedding.

5. **`query2doc`**  
   Generate a pseudo-document and concatenate it with the query.

6. **`generative_relevance_feedback`**  
   Generate expansion terms/facts/documents and use them as feedback models or query expansions.

7. **`corpus_steered_expansion`**  
   Retrieve initial candidates, select pivotal corpus-originated sentences or terms, and combine them with LLM expansion. This is a key hallucination/prior-reduction baseline.

### 10.2 Repo-native answer-shaped methods

8. **`raw_expected_answer_only`**  
   Generate a concise expected answer and retrieve using only it. Treat as a diagnostic probe, not as a deployable method.

9. **`concat_query_raw_expected`**  
   Concatenate original query and raw expected answer.

10. **`dual_query_raw_expected_rrf`**  
    Retrieve separately using query and raw expected answer, then fuse with RRF.

11. **`weighted_dual_query_raw_expected_rrf`**  
    Same as above, but down-weight the expected-answer route.

### 10.3 Leakage-aware methods

12. **`masked_expected_answer_only`**  
    Generate a concise answer-shaped expansion, then mask answer-bearing spans.

13. **`concat_query_masked_expected`**  
    Concatenate query and masked expected answer.

14. **`dual_query_masked_expected_rrf`**  
    Retrieve query and masked answer-shaped expansion separately, then fuse.

15. **`weighted_dual_query_masked_expected_rrf`**  
    Down-weight masked route relative to the original query.

16. **`answer_candidate_constrained_template_only`**  
    Generate a retrieval reformulation that preserves only query-known anchors and relation intent, using typed unknown slots without proposing new concrete entities, dates, or values.

17. **`concat_query_answer_candidate_constrained_template`**

18. **`dual_query_answer_candidate_constrained_template_rrf`**

### 10.4 Controls

| Control | Purpose |
|---|---|
| `gold_answer_only` | Upper bound for answer-token leakage effect; report only as diagnostic, not as a method. |
| `oracle_answer_masked` | Shows what happens if true gold answer spans are removed perfectly. |
| `post_hoc_gold_removed_expected_answer` | Removes gold answer strings and aliases from raw LLM output after generation. |
| `random_span_masking` | Tests whether masking helps because of leakage removal or just length/noise changes. |
| `entity_only_masking` | Tests whether masking all new entities is better than masking only answer spans. |
| `generic_mask_slot` | Compares typed slots with generic `[MASK]` or `[UNKNOWN]`. |
| `length_matched_neutral_filler` | Controls for expansion length effects. |
| `wrong_answer_injection` | Stress test: intentionally inject a plausible wrong answer to measure retriever susceptibility. Use carefully and label as diagnostic. |

### 10.5 Method-set reduction for main tables

The full Cartesian product is too large for a clean main paper. Use this compact main set:

1. `query_only`
2. `BM25+RM3` or `dense_query_only`, depending on table
3. `HyDE`
4. `Query2doc`
5. `Generative Relevance Feedback`
6. `Corpus-Steered Expansion`
7. `concat_query_raw_expected`
8. `concat_query_masked_expected`
9. `concat_query_answer_candidate_constrained_template`
10. best fixed RRF variant chosen on development data or reported only in ablation

Put expansion-only and weighted-RRF variants in ablations unless they are central to the story.

---

## 11. Prompt Specifications

### 11.1 Raw expected answer prompt

```text
Given the user question, write a single-sentence answer that you expect to be correct.
Do not mention uncertainty.
Question: {query}
Expected answer:
```

This prompt is intentionally leakage-prone and should be treated as a diagnostic condition.

### 11.2 Masked expected answer prompt

Two-stage procedure:

1. Generate the expected answer.
2. Mask answer-bearing spans while preserving query-known anchors and relational context.

```text
Question: {query}
Candidate answer sentence: {raw_expected_answer}

Rewrite the candidate answer sentence for retrieval.
Rules:
- Replace the unknown answer span with a typed slot such as [PERSON], [LOCATION], [DATE], [NUMBER], or [WORK].
- Preserve entities and descriptors that already appear in the question.
- Do not introduce new concrete entities, dates, numbers, titles, or locations.
- Output only the rewritten retrieval text.
```

### 11.3 Answer-candidate-constrained template prompt

```text
You are writing a retrieval query, not answering the question.

Question: {query}

Create a retrieval-oriented reformulation that helps find evidence documents.
Rules:
1. Preserve concrete entities, dates, titles, and values only if they appear in the original question.
2. Do not introduce any new concrete person, organization, location, title, date, number, or answer candidate.
3. Represent the unknown answer with a typed slot such as [PERSON], [ORGANIZATION], [LOCATION], [DATE], [NUMBER], [WORK], or [EVENT].
4. Include the relation or evidence need in natural language.
5. Return JSON with keys:
   - known_anchors
   - unknown_slot_type
   - relation_intent
   - retrieval_text
```

Example output:

```json
{
  "known_anchors": ["<entities from query only>"],
  "unknown_slot_type": "PERSON",
  "relation_intent": "identify the person who held the specified role during the specified period",
  "retrieval_text": "evidence about [PERSON] holding the specified role for <query anchor> during <query time>"
}
```

### 11.4 Candidate-introduction validator

After every generation, run a validator:

1. extract named entities, dates, numbers, and titles from the generated expansion;
2. subtract entities already present in the original query;
3. flag remaining concrete candidates as introduced candidates;
4. compare introduced candidates to gold answers, answer aliases, relevant-document entities, and non-relevant-document entities.

This validator is central to the paper.

---

## 12. Retrieval Settings

### 12.1 Sparse retrieval

Required:

- BM25

Recommended:

- BM25 + RM3 or a comparable PRF baseline

Rationale: sparse retrieval should be most sensitive to exact answer-token and alias leakage.

### 12.2 Dense retrieval

Required:

- one fixed zero-shot dense retriever, such as Contriever, E5, or BGE

Recommended:

- a second dense retriever from a different training family if runtime permits

Rationale: dense retrieval tests whether leakage persists through semantic similarity rather than exact lexical overlap.

### 12.3 Integration modes

Use the following as ablations:

1. expansion only;
2. query + expansion concatenation;
3. RRF of separate query and expansion retrieval runs;
4. weighted RRF.

For main results, use one fixed integration mode per method family or a development-selected integration setting to avoid an unreadable table.

---

## 13. Metrics

### 13.1 Retrieval metrics

Primary:

- `nDCG@10`
- `Recall@10`
- `Recall@20`

Dataset-specific:

- `MRR@10` for QA-style retrieval;
- evidence recall for HotpotQA/SciFact supporting facts;
- `Recall@100` if comparing with BEIR-style retrieval reports.

### 13.2 Leakage metrics

| Metric | Definition |
|---|---|
| Exact answer leakage rate | Fraction of expansions containing normalized gold answer string. |
| Alias answer leakage rate | Fraction containing known answer aliases after alias expansion. |
| Entity-counterfactual answer leakage | Fraction containing original public answer or remapped private-like answer. Distinguish both. |
| Evidence-entailment leakage | Fraction of generated sentences entailed by gold evidence but not by the query alone. |
| Unsupported candidate injection rate | Fraction of expansions introducing concrete entities/values not in query and not supported by relevant evidence. |
| Wrong-prior injection rate | Fraction introducing a concrete answer candidate that conflicts with gold metadata. |
| Bridge-entity leakage | For multi-hop datasets, fraction introducing a bridge entity not present in the query. |
| Label leakage | For fact verification, fraction directly stating support/refute label or evidence conclusion. |
| Leakage-positive gain | Retrieval gain conditioned on leakage-positive expansions. |
| Leakage-negative gain | Retrieval gain conditioned on leakage-negative expansions. |
| Public-to-private instability | Raw performance drop from public-original to entity-counterfactual. |
| Excess instability over query-only | Method instability minus query-only instability. |

### 13.3 Prior-strength metrics

Before retrieval, prompt the generation LLM to answer each query with no context. Record:

1. no-context answer correctness;
2. confidence if available or self-rated;
3. overlap between no-context answer and generated expansion;
4. whether no-context answer correctness predicts retrieval gains.

This directly tests whether the generator’s parametric prior is driving retrieval behavior.

### 13.4 Renaming quality metrics

- entity replacement coverage;
- alias consistency;
- type preservation;
- no original-name residue;
- answer preservation in relevant documents;
- no-context answer accuracy drop;
- query-only retrieval degradation;
- human validation pass rate.

---

## 14. Statistical Analysis

### 14.1 Primary comparisons

Pre-register these comparisons:

1. raw expected answer vs masked expected answer;
2. raw expected answer vs answer-candidate-constrained template;
3. HyDE vs answer-candidate-constrained template;
4. Query2doc vs answer-candidate-constrained template;
5. raw expected answer vs corpus-steered expansion;
6. public-to-private excess instability for raw methods vs leakage-aware methods.

### 14.2 Tests

Use paired tests because the same queries are evaluated across methods.

Recommended:

- paired bootstrap confidence intervals for `nDCG@10`, `Recall@20`, and `MRR@10`;
- paired permutation tests for primary method comparisons;
- Holm-Bonferroni correction or a small pre-registered comparison set;
- per-query win/tie/loss counts;
- mixed-effects regression for leakage effects:

```text
retrieval_gain ~ leakage_positive + method + retriever + dataset
                 + renamed_condition + no_context_answer_correct
                 + (1 | query)
```

### 14.3 Effect-size reporting

Do not rely only on p-values. Report:

- absolute metric differences;
- relative differences;
- excess instability over query-only;
- leakage-rate reductions;
- gain per leakage bucket;
- confidence intervals.

---

## 15. Recommended Main Tables and Figures

### Main tables

**Table 1. Literature-positioning matrix**  
Rows: HyDE, Query2doc, GRF, CSQE, Yoon et al., this work.  
Columns: generated expansion, leakage metric, private-like evaluation, answer-candidate constraint, QA datasets, sparse/dense/fusion analysis.

**Table 2. Renaming validation**  
Coverage, consistency, answer preservation, prior suppression, query-only degradation.

**Table 3. Public-original retrieval performance**  
Main methods across datasets and retrievers.

**Table 4. Entity-counterfactual retrieval performance**  
Same methods under private-like renaming.

**Table 5. Excess public-to-private instability**  
Method drop minus query-only drop.

**Table 6. Leakage bucket analysis**  
Performance and gains for exact-answer leakage, alias leakage, evidence-entailment leakage, unsupported injection, and leakage-negative cases.

**Table 7. Mitigation ablations**  
Typed mask, generic mask, oracle mask, random mask, entity-only mask, post-hoc gold removal, answer-candidate-constrained template.

### Main figures

1. **Leakage rate vs retrieval gain scatterplot** by method and retriever.
2. **Public-to-private excess instability bar plot** with confidence intervals.
3. **Per-query drop histogram** for raw vs masked vs template.
4. **Qualitative examples** showing:
   - public success via answer leakage;
   - private-like failure from wrong prior;
   - masked/template recovery;
   - case where masking hurts due to losing a legitimate bridge entity.

---

## 16. Implementation Plan

This section assumes the repo claims in the original memo are correct, but they should be verified before implementation.

### 16.1 What likely already exists

- BM25 retrieval;
- one dense retriever path;
- HyDE generation;
- expected-answer generation;
- query-aware masking;
- concatenation;
- RRF and weighted RRF;
- per-query record dumps.

### 16.2 What must be added

1. **Answer and evidence metadata loaders**
   - NQ short answers;
   - HotpotQA answers and supporting facts;
   - SciFact evidence abstracts, labels, and rationales.

2. **Generation cache and prompt versioning**
   - every prompt, model, temperature, and output schema must be reproducible;
   - generation outputs must be cached before retrieval.

3. **Leakage scorer**
   - exact match;
   - alias match;
   - entity/value/date extraction;
   - candidate-introduction validation;
   - evidence-entailment scoring for SciFact and HotpotQA.

4. **Entity-counterfactual benchmark builder**
   - deterministic alias tables;
   - corpus/query/answer/supporting-fact rewriting;
   - validation report.

5. **Answer-candidate-constrained template generator**
   - JSON schema;
   - strict validator;
   - automatic rejection/regeneration for schema violations.

6. **Corpus-steered expansion baseline**
   - initial retrieval;
   - pivotal sentence/term selection;
   - expansion construction.

7. **Controls**
   - oracle mask;
   - random mask;
   - generic mask;
   - post-hoc gold removal;
   - length-matched neutral filler;
   - wrong-answer injection diagnostic.

8. **Statistical analysis scripts**
   - paired bootstrap;
   - permutation tests;
   - win/tie/loss;
   - regression by leakage bucket.

### 16.3 Recommended implementation order

1. Verify repo method outputs and per-query dumps.
2. Add answer/evidence metadata loading.
3. Add generation cache with raw outputs for every method.
4. Build leakage scorer on public-original data.
5. Implement answer-candidate-constrained template generation and validator.
6. Build entity-counterfactual benchmark pipeline.
7. Run small dry-run validation only on a tiny sample; do not report as experiments.
8. Freeze prompts, metrics, and primary comparisons.
9. Run BM25 public/private-like experiments.
10. Run dense public/private-like experiments.
11. Add ablations and statistical tests.
12. Conduct qualitative error analysis.

---

## 17. Top-Conference Risk Assessment

### Risk 1: Direct prior work weakens novelty

**Issue:** Yoon et al. 2025 already studies knowledge leakage in LLM-based query expansion.

**Mitigation:** Reframe the paper as extending leakage analysis from fact-verification pseudo-documents to QA-style and private-like RAG, with entity-counterfactual evaluation and answer-candidate-constrained mitigation.

### Risk 2: Reviewers say “renaming is artificial”

**Issue:** Synthetic aliasing can distort language and retrieval difficulty.

**Mitigation:** Use natural aliases and coded aliases; report query-only degradation; report excess instability over query-only; include human spot checks; optionally add a real or semi-private corpus.

### Risk 3: Renaming hurts dense retrieval for reasons unrelated to leakage

**Issue:** Dense retrievers may handle `Person P-041` poorly.

**Mitigation:** Natural-alias condition should be primary. Coded aliases are a stress test. Excess instability over query-only is mandatory.

### Risk 4: Masking hurts retrieval

**Issue:** Removing answer-bearing spans may remove useful evidence terms.

**Mitigation:** The claim should be leakage reduction and stability, not universal score improvement. Include leakage-negative and bridge-entity analyses.

### Risk 5: Answer-agnostic templates underperform

**Issue:** Strict no-new-candidate prompts may be too weak for lexical retrieval.

**Mitigation:** Use them as the strict safety baseline; evaluate concat and RRF modes; include corpus-steered templates; emphasize private-like stability and low unsupported-injection rates.

### Risk 6: Leakage detection is brittle

**Issue:** Answers can be paraphrased or represented by aliases.

**Mitigation:** Combine exact matching, alias tables, entity extraction, and entailment-based scoring. Report precision/recall of leakage labels on a human-audited sample.

### Risk 7: Factorial experiment becomes unreadable

**Issue:** Many methods, retrievers, datasets, and integration modes.

**Mitigation:** Predefine a compact main method set and push integration/controls to ablations.

---

## 18. Claim Boundaries for the Paper

### Safe claims

1. LLM-generated retrieval expansions can introduce concrete answer candidates not present in the original query.
2. Public benchmark improvements from generated expansions can conflate reformulation with answer-prior injection.
3. Leakage-aware reformulations can reduce direct answer/candidate injection.
4. Entity-counterfactual evaluation can reveal public-to-private-like instability not visible in public-only evaluation.
5. Different retrievers and integration modes exhibit different sensitivity to leakage.

### Claims requiring strong evidence

1. Raw expansions mostly work because of leakage.
2. Masked expected answers are better than HyDE.
3. Answer-candidate-constrained templates are the best retrieval method.
4. Public benchmark results are broadly contaminated.
5. Entity-counterfactual benchmarks fully simulate private enterprise RAG.

### Claims to avoid

1. “First to study leakage in LLM query expansion.”
2. “First generated answer retrieval method.”
3. “Private-like renaming proves real private-domain performance.”
4. “Dense retrieval solves leakage because the encoder filters hallucinations.”

---

## 19. Minimum Viable Paper Package

For a credible strong-conference submission, the minimum package should be:

1. three datasets across at least two task types: NQ/HotpotQA plus SciFact;
2. public-original and entity-counterfactual private-like versions for each dataset;
3. BM25 and one dense retriever;
4. HyDE, Query2doc, GRF or LLM-prompted expansion, raw expected answer, masked expected answer, answer-candidate-constrained template, and corpus-steered expansion;
5. exact/alias answer leakage and evidence-entailment leakage metrics;
6. public-to-private excess instability analysis;
7. at least three controls: oracle mask, random mask, post-hoc gold removal or entity-only mask;
8. human-audited leakage and renaming validation on a stratified sample;
9. paired statistical testing;
10. qualitative examples showing both successes and failures.

If the project cannot include entity-counterfactual evaluation plus leakage-bucket analysis, it is better positioned as a workshop paper or internal technical report rather than a top-conference main submission.

---

## 20. Recommended Project Direction

Pivot fully to the leakage-aware, entity-counterfactual RAG framing.

However, update the novelty claim:

- **Old novelty:** “We identify answer leakage in LLM query expansion.”
- **Revised novelty:** “We extend leakage analysis to QA-style/private-like RAG using entity-counterfactual retrieval evaluation and answer-candidate-constrained reformulation, showing when generated expansions improve evidence retrieval versus inject public answer priors.”

The currently implemented baseline and ablation scaffold remains valuable, but it should not be the headline paper.

The paper’s strongest identity is:

> a contamination-aware retrieval study that separates reformulation gains from answer-prior gains and proposes stricter, leakage-aware retrieval reformulations for private-like RAG.

---

## 21. Checklist Before Running Experiments

- [ ] Verify repo method implementations and output fields.
- [ ] Freeze dataset versions and qrel formats.
- [ ] Add answer/evidence metadata to every query record.
- [ ] Define alias-generation policy and random seed.
- [ ] Validate entity renaming on a small human-audited sample.
- [ ] Freeze generation prompts and schemas.
- [ ] Implement leakage scorer before inspecting performance.
- [ ] Pre-register primary comparisons and metrics.
- [ ] Decide main method set and ablation-only method set.
- [ ] Confirm compute budget for BM25, dense retrieval, and generation.
- [ ] Do not tune prompts on test results.

---

## 22. References

- Gao et al. 2023. *Precise Zero-Shot Dense Retrieval without Relevance Labels*. https://arxiv.org/abs/2212.10496
- Wang et al. 2023. *Query2doc: Query Expansion with Large Language Models*. https://aclanthology.org/2023.emnlp-main.585/
- Yu et al. 2023. *Generate rather than Retrieve: Large Language Models are Strong Context Generators*. https://openreview.net/forum?id=fB0hRu9GZUS
- Dai et al. 2023. *Promptagator: Few-shot Dense Retrieval From 8 Examples*. https://openreview.net/forum?id=gmL46YMpu2J
- Jagerman et al. 2023. *Query Expansion by Prompting Large Language Models*. https://arxiv.org/abs/2305.03653
- Mackie et al. 2023. *Generative Relevance Feedback with Large Language Models*. https://arxiv.org/abs/2304.13157
- Lei et al. 2024. *Corpus-Steered Query Expansion with Large Language Models*. https://aclanthology.org/2024.eacl-short.34/
- Liu and Zhang 2025. *GOLFer: Smaller LMs-Generated Documents Hallucination Filter & Combiner for Query Expansion in Information Retrieval*. https://aclanthology.org/2025.findings-acl.8/
- Ma et al. 2023. *Query Rewriting in Retrieval-Augmented Large Language Models*. https://aclanthology.org/2023.emnlp-main.322/
- Trivedi et al. 2023. *Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions*. https://aclanthology.org/2023.acl-long.557/
- Shao et al. 2023. *Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy*. https://aclanthology.org/2023.findings-emnlp.620/
- Zheng et al. 2024. *Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models*. https://proceedings.iclr.cc/paper_files/paper/2024/hash/592da1445a51e54a3987958b5831948f-Abstract-Conference.html
- Yoon et al. 2025. *Hypothetical Documents or Knowledge Leakage? Rethinking LLM-based Query Expansion*. https://aclanthology.org/2025.findings-acl.980/
- Xu et al. 2024. *Benchmark Data Contamination of Large Language Models: A Survey*. https://arxiv.org/abs/2406.04244
- Oren et al. 2024. *Proving Test Set Contamination in Black-Box Language Models*. https://proceedings.iclr.cc/paper_files/paper/2024/hash/46e624c244cff669223d488defd4e835-Abstract-Conference.html
- White et al. 2024. *LiveBench: A Challenging, Contamination-Free LLM Benchmark*. https://arxiv.org/abs/2406.19314
- Thakur et al. 2021. *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*. https://arxiv.org/abs/2104.08663
- Kwiatkowski et al. 2019. *Natural Questions: A Benchmark for Question Answering Research*. https://aclanthology.org/Q19-1026/
- Yang et al. 2018. *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*. https://aclanthology.org/D18-1259/
- Wadden et al. 2020. *Fact or Fiction: Verifying Scientific Claims*. https://aclanthology.org/2020.emnlp-main.609/
- Izacard et al. 2021. *Unsupervised Dense Information Retrieval with Contrastive Learning*. https://arxiv.org/abs/2112.09118
- Wang et al. 2022. *Text Embeddings by Weakly-Supervised Contrastive Pre-training*. https://arxiv.org/abs/2212.03533
- Meng et al. 2024. *On the Robustness of Document-Level Relation Extraction Models to Entity Name Variations*. https://aclanthology.org/2024.findings-acl.969/
- *On the Robustness of Reading Comprehension Models to Entity Renaming*. https://openreview.net/forum?id=lXczoncSyt0
- Cormack et al. 2009. *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*. https://doi.org/10.1145/1571941.1572114
- Lavrenko and Croft 2001. *Relevance-Based Language Models*. https://ciir.cs.umass.edu/pubfiles/ir-225.pdf
