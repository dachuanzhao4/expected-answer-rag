# Paper Positioning Memo

## Working Thesis

The strongest framing is not "we propose two new retrieval methods." That is too weak because `expected answer` substantially overlaps with prior LLM-based query expansion and generate-then-retrieve work.

The stronger framing is:

> We study answer leakage in LLM-based query expansion for retrieval, and propose leakage-aware answer masking as a query reformulation strategy that preserves contextual retrieval cues while suppressing direct answer injection.

Under this framing:

- `expected answer` is an important ablation and comparison point.
- `masked expected answer` is the main method.
- `answer leakage` is the main scientific problem.

## Core Claim

Primary claim:

> LLM-generated answer-form query expansions improve retrieval partly by injecting answer-bearing tokens. Query-aware masking removes these tokens while preserving answer-shape context and query anchors, yielding a cleaner and often more robust retrieval signal.

Stronger claim to use only if supported by results:

> Across sparse and dense retrieval settings, masked expected answers recover a substantial fraction of the gains of raw expected-answer expansion while materially reducing leakage-sensitive performance inflation.

Claims to avoid unless the evidence is overwhelming:

- "masked expected answer consistently outperforms all prior LLM expansion methods"
- "expected answer is a fundamentally new retrieval paradigm"
- "first answer-based retrieval method"

## Novelty Statement

The novelty should be stated conservatively and precisely.

### Main novelty

1. Problem formulation: explicit study of answer leakage in answer-form LLM query expansion for retrieval.
2. Method: query-aware answer masking that masks only the answer-bearing span in an LLM-generated expected answer while preserving query anchors and surrounding answer context.
3. Evaluation protocol: leakage-sensitive analysis that separates cases where generated expansions contain gold answer strings from cases where they do not.
4. Retrieval formulation comparison: direct comparison of answer-only retrieval, concatenation, and dual-route fusion under both raw and masked expansions.

### What is not novel

The paper should not claim novelty for the general idea of using LLM-generated text to improve retrieval. That space is already populated by:

- HyDE
- Query2doc
- GRF
- CSQE
- MuGI and related LLM query expansion work

### What is plausibly novel

What is potentially novel is the combination of:

- answer-shaped expansions rather than generic pseudo-documents
- answer leakage as the confound of interest
- masking as the intervention
- leakage-sensitive evaluation as the validation protocol

## Threat Model

The paper should define answer leakage explicitly.

### Threat

LLM-generated query expansions may contain the gold answer or highly entailed answer-bearing content. This can artificially improve retrieval by lexical or semantic answer injection rather than by genuinely improving the reformulation of the information need.

### Attack surface

- Exact answer string appears in the generated expected answer.
- A paraphrased but equivalent answer appears.
- A generated passage contains evidence entailed by the ground-truth evidence.
- The answer-bearing content is fed into retrieval through concatenation or through independent retrieval followed by fusion.

### Failure mode

Observed retrieval gains are attributed to better query understanding, while the actual cause is answer injection from the LLM.

### Adversary

There is no malicious external adversary. The effective adversary is the interaction between:

- benchmark design
- parametric knowledge stored in the LLM
- answer-bearing expansions used as retrieval inputs

### Assumption

The LLM may already know the answer from pretraining and may expose that answer directly in the generated expansion.

### Goal of the proposed method

Preserve useful contextual structure around the answer while preventing the retriever from exploiting the answer token itself.

### Regimes to distinguish

1. Benign reformulation: expansion adds useful context without the answer.
2. Leakage-prone reformulation: expansion contains the answer string or entailed evidence.
3. Over-masked reformulation: masking removes too much useful signal and hurts retrieval.

## Method Framing

Let `q` be a query and `g(q)` be an LLM-generated concise expected answer.

### Expected Answer Expansion

- Generate `e = g(q)`.
- Retrieve using:
  - `e`
  - `q + e`
  - `RRF(retrieve(q), retrieve(e))`

### Masked Expected Answer Expansion

- Generate `e = g(q)`.
- Apply a query-aware masking operator `M(q, e)` that replaces only answer-bearing span(s) with typed slots.
- Let `m = M(q, e)`.
- Retrieve using:
  - `m`
  - `q + m`
  - `RRF(retrieve(q), retrieve(m))`

### Hypothesis

`m` should preserve contextual utility from answer-form rewriting while being less leakage-sensitive than `e`.

## Baseline Matrix

The paper should include the following baseline groups.

| Group | Method | Description | Priority |
|---|---|---|---|
| Sparse base | BM25 | Raw query only | Required |
| Dense base | Contriever or E5/BGE | Raw query only | Required |
| Classical QE | BM25 + RM3 | Standard pseudo relevance feedback | Required |
| LLM QE | HyDE | Hypothetical document expansion | Required |
| LLM QE | Query2doc | Pseudo-document concatenated to query | Required |
| LLM QE | GRF | LLM-generated relevance feedback | Recommended |
| LLM QE | CSQE | Corpus-steered expansion | Recommended if feasible |
| Ablation | Expected answer only | Retrieve with `e` | Required |
| Ablation | Query + expected answer | Retrieve with `q + e` | Required |
| Ablation | RRF(query, expected) | Late fusion | Required |
| Method | Masked expected answer only | Retrieve with `m` | Required |
| Method | Query + masked expected answer | Retrieve with `q + m` | Required |
| Method | RRF(query, masked) | Late fusion | Required |
| Control | Oracle masked answer | Mask using gold answer span rather than LLM answer | Required |
| Control | Random span masking | Mask same token count at random | Required |
| Control | Entity-only masking | Mask named entities only | Required |
| Control | Slotless masking | Use generic `[MASK]` rather than typed slots | Required |
| Control | Length-matched filler | Query-only plus neutral added text | Recommended |
| Analysis control | Gold answer removed post hoc | Strip exact gold aliases from raw expected answer | Required |

## Minimal Main-Paper Table

If the main paper needs a smaller comparison table, include:

1. BM25
2. Dense retriever
3. BM25 + RM3
4. HyDE
5. Query2doc
6. Expected answer
7. Query + expected answer
8. RRF(query, expected)
9. Masked expected answer
10. Query + masked expected answer
11. RRF(query, masked)
12. Oracle masked answer

Move the remaining controls to the appendix if needed.

## Recommended Experimental Axes

The evaluation should not stop at average retrieval metrics.

### Retriever axis

- BM25
- one zero-shot dense retriever
- one stronger dense retriever

### Dataset axis

- entity-centric QA
- multi-hop QA
- fact verification
- at least one domain-shifted setting if possible

### Leakage axis

- raw expansion contains exact gold answer
- raw expansion does not contain exact gold answer
- raw expansion semantically entails the answer

### Masking axis

- typed slots
- generic slot
- oracle masking
- entity-only masking

### Integration axis

- expansion only
- concatenation
- RRF fusion

## Metrics

### Main retrieval metrics

- `nDCG@10`
- `Recall@10` or `Recall@20`
- `MRR@10` for QA-style datasets

### Leakage-sensitive metrics

- percentage of raw expansions containing the exact gold answer
- percentage of masked expansions still containing the exact gold answer
- gains restricted to answer-containing cases
- gains restricted to answer-absent cases
- delta from raw expected answer to masked expected answer within leakage buckets

### Additional analysis metrics

If feasible:

- semantic entailment of the answer in the generated expansion
- entity overlap between expansion and relevant document
- masking granularity statistics

## Recommended Result Structure

### Table 1

Overall retrieval performance across datasets and retrievers.

### Table 2

Performance by leakage bucket.

### Table 3

Masking ablations:

- typed slots
- generic slot
- oracle mask
- random mask
- entity-only mask

### Table 4

Integration comparison:

- expansion only
- concatenation
- RRF fusion

### Figure 1

Worked example showing:

- query
- expected answer
- masked expected answer
- retrieved documents

### Figure 2

Performance difference between raw and masked expansion as a function of leakage rate.

## Reviewer-Safe Contribution Paragraph

This paragraph is close to publication-ready and can be reused later:

> This paper revisits LLM-based query expansion through the lens of answer leakage. We show that concise answer-form expansions can improve retrieval, but part of this gain may arise from direct injection of answer-bearing tokens. To address this, we propose query-aware answer masking, which preserves the contextual structure of an expected answer while masking only the answer span with typed placeholders. This yields a leakage-aware expansion signal that is easy to apply to both sparse and dense retrievers. Across multiple datasets and retrieval settings, we compare raw answer expansions, masked answer expansions, and prior LLM-based expansion baselines, and perform a dedicated leakage-sensitive analysis to disentangle reformulation gains from leakage-driven gains.

## What To Downplay

Avoid claiming:

- first answer-based query expansion
- first use of masking for retrieval
- novel retrieval architecture
- fundamentally new retrieval paradigm

These are fragile claims and are likely to trigger reviewer pushback.

## What To Emphasize

Emphasize:

- leakage-aware evaluation
- query-aware masking intervention
- contextual preservation under masking
- analysis across retrievers and datasets
- disentangling reformulation gains from answer injection

## Candidate Titles

Safer title directions:

- `Leakage-Aware Query Expansion with Masked Expected Answers for Retrieval`
- `Do LLM Query Expansions Help Retrieval or Leak Answers?`
- `Masked Expected Answers for Leakage-Aware Retrieval Augmentation`

Less safe title directions:

- `Expected Answer Retrieval`
- `A Novel Retrieval Paradigm with Expected Answers`

## Bottom Line

The paper-worthy version of the project is:

- `expected answer` is an ablation and baseline
- `masked expected answer` is the method
- `answer leakage` is the central problem
- `leakage-aware evaluation` is the main scientific contribution

## Related Work To Cite

- [Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE), ACL 2023](https://aclanthology.org/2023.acl-long.99/)
- [Query2doc: Query Expansion with Large Language Models, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.585/)
- [Generate-then-Retrieve: Intent-Aware FAQ Retrieval in Product Search, ACL 2023](https://aclanthology.org/2023.acl-industry.73/)
- [Expand, Rerank, and Retrieve: Query Reranking for Open-Domain Question Answering, Findings ACL 2023](https://aclanthology.org/2023.findings-acl.768/)
- [Corpus-Steered Query Expansion with Large Language Models, EACL 2024](https://aclanthology.org/2024.eacl-short.34/)
- [Exploring the Best Practices of Query Expansion with Large Language Models, Findings EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.103/)
- [When do Generative Query and Document Expansions Fail? A Comprehensive Study Across Methods, Retrievers, and Datasets, Findings EACL 2024](https://aclanthology.org/2024.findings-eacl.134/)
- [Hypothetical Documents or Knowledge Leakage? Rethinking LLM-based Query Expansion, Findings ACL 2025](https://aclanthology.org/2025.findings-acl.980/)
- [Generative Relevance Feedback with Large Language Models, arXiv 2023](https://arxiv.org/pdf/2304.13157)
