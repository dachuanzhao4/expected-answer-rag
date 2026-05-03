# Paper Positioning Memo

## Executive Summary

The stronger top-conference paper is not a generic "expected answer retrieval"
paper. That framing is too close to existing LLM query expansion, pseudo-doc,
and generate-then-retrieve work.

The stronger paper is:

> LLM-generated retrieval expansions can be confounded by answer leakage from
> model priors. This leakage is especially problematic when moving from public
> QA benchmarks to private-domain RAG, where the model's answer prior is often
> wrong. We propose leakage-aware retrieval reformulation and a contamination-
> resistant evaluation protocol to measure and mitigate this failure mode.

In that framing:

- `expected answer` is an ablation, not the headline contribution
- `masked expected answer` is one intervention
- `answer-agnostic templates` are a stronger intervention
- the renamed "private-like" benchmark is a core methodological contribution

## Recommended Paper Thesis

### One-sentence thesis

Public-benchmark gains from LLM-generated retrieval expansions may be partially
inflated by answer leakage from parametric knowledge, and this effect breaks
down in private-style RAG; leakage-aware reformulations recover useful context
while reducing prior-induced retrieval bias.

### Working title candidates

1. `Answer Leakage in LLM-Based Query Expansion for Retrieval`
2. `Private-Like RAG Reveals Prior-Induced Retrieval Bias in LLM Query Expansion`
3. `Leakage-Aware Query Reformulation for Retrieval-Augmented Generation`
4. `When HyDE Knows the Answer: Measuring and Mitigating Prior-Induced Retrieval Bias`

### Central claim

LLM-generated answer-form or pseudo-document expansions can improve retrieval
for two different reasons:

1. they genuinely clarify the information need
2. they inject answer-bearing content that the retriever can exploit

The paper should argue that current public QA-style evaluation often conflates
these mechanisms, and that a private-like renamed benchmark exposes the gap.

## Why This Is the Stronger Paper

### The current baseline package is useful but not enough

The current repository implements a solid experimental scaffold for:

- `query_only`
- `HyDE`
- `expected_answer`
- `masked_expected_answer`
- concatenation
- reciprocal-rank-fusion variants

This is useful engineering and gives a baseline section. It is not, by itself,
an especially strong top-tier research claim because:

1. `expected answer` substantially overlaps with prior LLM-based expansion
   methods such as HyDE, Query2doc, and related pseudo-document generation.
2. Current results do not support a clean "our method wins" story.
3. The most interesting observation is not performance alone, but the
   possibility that some gains are leakage-driven.

## Repo Baseline Methods

This section briefly defines the baseline methods already implemented in the
repository so that this memo can be read as a standalone document.

### Retrieval backends

The repo currently supports two retrieval backends:

- `BM25`: standard sparse lexical retrieval based on term overlap and inverse
  document frequency
- `dense retrieval`: embedding-based retrieval using a sentence-transformer
  encoder and vector similarity

The same query reformulation methods can be run on top of either backend.

### Query reformulation baselines currently in the repo

#### `query_only`

Retrieve using only the original user query, with no LLM-generated expansion.
This is the main no-expansion baseline.

#### `hyde_doc_only`

Generate a longer hypothetical document in the style of a relevant passage, then
retrieve using that generated document alone. This is the repo's HyDE-style
baseline.

#### `raw_expected_answer_only`

Generate a concise one-sentence expected answer to the query, then retrieve
using only that expected answer text. This tests whether short answer-shaped
generation is itself a strong retrieval signal.

#### `masked_expected_answer_only`

Generate a concise expected answer, then mask only the span that directly
answers the question while preserving surrounding context and anchors already in
the query. Retrieve using only this masked text.

#### `concat_query_raw_expected`

Concatenate the original query with the raw expected answer and retrieve using
the combined text. This tests whether the expected answer helps when added to,
rather than substituted for, the original query.

#### `concat_query_masked_expected`

Concatenate the original query with the masked expected answer and retrieve with
the combined text. This is the leakage-aware concatenation variant.

#### `dual_query_raw_expected_rrf`

Retrieve separately with the original query and the raw expected answer, then
merge the two ranked lists using reciprocal-rank fusion (RRF). This treats the
query and answer routes as separate evidence sources.

#### `dual_query_masked_expected_rrf`

Retrieve separately with the original query and the masked expected answer, then
merge the two rankings with RRF. This is the leakage-aware late-fusion variant.

#### `weighted_dual_query_raw_expected_rrf`

Same as `dual_query_raw_expected_rrf`, except the expected-answer route can be
down-weighted relative to the original query. This is useful when the generated
answer is informative but overly dominant.

#### `weighted_dual_query_masked_expected_rrf`

Same as `dual_query_masked_expected_rrf`, except the masked-answer route can be
down-weighted relative to the original query.

### How these baselines relate conceptually

The implemented methods correspond to three main design choices:

1. `What text is generated?`
   - long hypothetical document (`hyde_doc_only`)
   - short answer-shaped text (`raw_expected_answer_only`)
   - leakage-aware masked answer (`masked_expected_answer_only`)
2. `How is generated text used?`
   - alone
   - concatenated to the original query
   - retrieved separately and fused
3. `How much concrete answer information is preserved?`
   - all of it (`raw_expected_answer_*`)
   - some of it removed (`masked_expected_answer_*`)

This structure is why the current repo is already a strong baseline scaffold:
it already covers the main axes of generation type, integration strategy, and
leakage sensitivity, even though the private-like benchmark and answer-agnostic
template methods still need to be added.

### The leakage-focused framing creates a cleaner scientific contribution

This framing has a stronger paper shape because it offers:

1. A sharper problem statement: `prior-induced retrieval bias`
2. A clearer threat model: `public memorization-like priors help; private-domain priors hurt`
3. A new evaluation protocol: `public vs private-like renamed benchmark`
4. A more defensible intervention family:
   - query-aware masking
   - answer-agnostic templates
   - leakage-sensitive controls

This moves the work from "yet another query expansion method" toward
"identifying and correcting a hidden confound in LLM-assisted retrieval."

## Scope of the Proposed Paper

### Problem statement

Given a user query `q`, an LLM is asked to generate an expansion `g(q)`, such as
a hypothetical document, an expected answer, or an answer-shaped template. This
expansion is then used for retrieval.

The problem is that `g(q)` may contain:

- the exact gold answer
- a paraphrase of the answer
- unsupported but plausible entities, dates, or values
- entailed evidence not present in the original query

This can inflate retrieval on public benchmarks if the model already "knows" the
answer, while causing harmful misdirection in private corpora where that prior
is unreliable.

### Threat model

There is no malicious external attacker. The effective adversary is the
interaction between:

- benchmark design
- LLM pretraining priors
- answer-bearing expansions used as retrieval inputs

### Research questions

1. How much of the observed gain from LLM-generated retrieval expansions is
   associated with answer-bearing content?
2. Does this gain transfer from public benchmarks to private-like renamed
   corpora where the model should not know the answer?
3. Can masking or answer-agnostic reformulation preserve useful context while
   reducing leakage-sensitive inflation?
4. Which retrieval regimes are most sensitive to leakage:
   sparse, dense, concatenation, or late fusion?

### Main hypotheses

H1. On public QA benchmarks, raw expected-answer and HyDE-style expansions will
often improve retrieval in part because they contain answer-bearing tokens.

H2. On private-like renamed benchmarks, raw expansions will lose more utility
than leakage-aware expansions because parametric priors no longer match the
corpus.

H3. Query-aware masking will recover part of the raw expansion gain while
reducing exact-answer leakage.

H4. Answer-agnostic templates will be the most leakage-resistant method, even
if they do not always maximize absolute retrieval on public QA.

## Claimed Contributions

The paper can plausibly claim the following contributions if the experiments
support them:

1. A problem formulation for `answer leakage` or `prior-induced retrieval bias`
   in LLM-based query expansion for retrieval.
2. A reproducible `private-like renamed benchmark` protocol that suppresses
   parametric answer priors without removing answer evidence from the corpus.
3. A comparative study of raw expansions, query-aware masking, and
   answer-agnostic templates under sparse and dense retrieval.
4. A leakage-sensitive evaluation protocol that separates retrieval gains with
   and without answer-bearing generation.

The paper should not claim:

- first use of generated text for retrieval
- first answer-based expansion method
- universal superiority over HyDE or Query2doc

## Initial Literature Review

This section is intentionally selective and oriented around the paper's core
positioning question: what is already known about LLM-generated retrieval
expansion, and what remains under-studied about contamination or leakage.

### 1. LLM-generated text as retrieval augmentation

#### HyDE

Gao et al. propose HyDE, which generates a hypothetical document and encodes it
for dense retrieval rather than retrieving directly from the raw query. The key
idea is that the dense encoder bottleneck can filter away some hallucinated
details while preserving relevance structure. This is a foundational baseline
for this project because it directly motivates the question of whether generated
content helps because it clarifies intent or because it injects answer-bearing
content.

Reference:
- Gao et al., `Precise Zero-Shot Dense Retrieval without Relevance Labels`,
  2022, [arXiv:2212.10496](https://arxiv.org/abs/2212.10496)

#### Query2doc

Wang et al. show that pseudo-documents generated by LLMs can be concatenated to
queries and improve both sparse and dense retrieval. This is close to the
current repository's `query + expected answer` and `query + HyDE` style of use.
Query2doc demonstrates that retrieval benefits can come from synthetic text
expansion, but it does not center the issue of answer leakage as a confound.

Reference:
- Wang et al., `Query2doc: Query Expansion with Large Language Models`, EMNLP
  2023, [arXiv:2303.07678](https://arxiv.org/abs/2303.07678)

#### Generate-then-read / generated context

Yu et al. show that generated contextual documents can sometimes substitute for
retrieved evidence in knowledge-intensive tasks. This is important background
because it reinforces the idea that LLMs can carry substantial task-relevant
knowledge, which is exactly why contamination-sensitive evaluation matters.

Reference:
- Yu et al., `Generate rather than Retrieve: Large Language Models are Strong
  Context Generators`, ICLR 2023,
  [arXiv:2209.10063](https://arxiv.org/abs/2209.10063)

#### Promptagator and related synthetic generation for retrieval

Promptagator uses LLM-based synthetic query generation to train task-specific
retrievers from a handful of examples. It is not the same problem as online
query expansion, but it is relevant because it shows that generated artifacts
can materially affect retrieval quality and that synthetic generation is already
well established as a retrieval ingredient.

Reference:
- Dai et al., `Promptagator: Few-shot Dense Retrieval From 8 Examples`, ICLR
  2023, [arXiv:2209.11755](https://arxiv.org/abs/2209.11755)

### 2. Benchmark contamination and evaluation leakage

The broader LLM evaluation community has already identified contamination as a
serious methodological risk. That literature is not retrieval-specific, but it
supports the paper's central premise: when models have prior exposure to test
content, measured gains can overstate real generalization.

Two useful references here are:

- Xu et al., `Benchmark Data Contamination of Large Language Models: A Survey`,
  2024, [arXiv:2406.04244](https://arxiv.org/abs/2406.04244)
- White et al., `LiveBench: A Challenging, Contamination-Free LLM Benchmark`,
  2024, [arXiv:2406.19314](https://arxiv.org/abs/2406.19314)

These papers are not direct baselines for retrieval. Their value is conceptual:
they establish that contamination-aware evaluation is now recognized as a
serious issue for LLM assessment, which strengthens the motivation for a
retrieval-specific analogue.

### 3. Retrieval evaluation benchmarks and domains

#### BEIR

BEIR is the right starting point for heterogeneous retrieval evaluation because
it covers multiple tasks and domains and highlights zero-shot robustness rather
than narrow in-domain tuning.

Reference:
- Thakur et al., `BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of
  Information Retrieval Models`, 2021,
  [arXiv:2104.08663](https://arxiv.org/abs/2104.08663)

#### Natural Questions

Natural Questions is useful for entity-centric, Wikipedia-grounded QA. It is
also exactly the kind of public benchmark where pretraining priors may be
strong, making it a good public-side test bed.

Reference:
- Kwiatkowski et al., `Natural Questions: A Benchmark for Question Answering
  Research`, TACL 2019, [ACL Anthology](https://aclanthology.org/Q19-1026/)

#### HotpotQA

HotpotQA introduces multi-hop retrieval needs and can test whether long
generated expansions help primarily by introducing bridging entities or whether
they overfit to public answer priors.

Reference:
- Yang et al., `HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question
  Answering`, EMNLP 2018,
  [ACL Anthology](https://aclanthology.org/D18-1259/)

#### SciFact

SciFact gives a domain-shifted, evidence-focused setting that is less likely to
reward generic world-knowledge priors in the same way as Wikipedia QA. It is a
good proxy for specialized knowledge settings and should be included if runtime
permits.

Reference:
- Wadden et al., `Fact or Fiction: Verifying Scientific Claims`, EMNLP 2020,
  [ACL Anthology](https://aclanthology.org/2020.emnlp-main.609/)

### 4. What appears to be missing in the literature

The gap this project should target is not "nobody has used LLMs for retrieval
expansion." That is false.

The more defensible gap is:

1. Prior work usually evaluates generated expansions on public benchmarks where
   answer-bearing priors may be helpful.
2. Prior work generally treats improved retrieval as evidence of better
   reformulation, but does not cleanly separate reformulation from answer
   injection.
3. Prior work has not, to our knowledge, systematically evaluated the same
   methods on a renamed private-like benchmark designed to suppress answer
   memorization while preserving corpus evidence.
4. Prior work has not centered answer-agnostic reformulation as a retrieval
   intervention specifically designed to avoid introducing new concrete answer
   candidates.

That is the gap the paper should claim.

## Detailed Experimental Design

## Experimental Goal

Measure how much LLM-assisted retrieval gains depend on answer-bearing priors,
and evaluate whether leakage-aware reformulations are more robust when those
priors are removed.

## Experimental Overview

For each dataset, we evaluate two regimes:

1. `public`: the original benchmark
2. `private-like`: a consistently renamed version of the benchmark designed to
   suppress memorized answer priors

For each query, we compare retrieval methods that vary in whether they are
allowed to introduce a concrete answer candidate.

### Methods

#### Core baselines

1. `query_only`
2. `BM25 + RM3` if feasible
3. `dense query_only` with one zero-shot dense retriever
4. `HyDE`
5. `Query2doc` or closest feasible pseudo-doc concatenation baseline

#### Repo-native methods

6. `raw_expected_answer_only`
7. `concat_query_raw_expected`
8. `RRF(query, raw_expected_answer)`
9. `masked_expected_answer_only`
10. `concat_query_masked_expected`
11. `RRF(query, masked_expected_answer)`

#### Additional methods recommended for the paper

12. `answer_agnostic_template_only`
13. `concat_query_answer_agnostic_template`
14. `RRF(query, answer_agnostic_template)`

#### Controls

15. `oracle_masked_answer`
16. `random_span_masking`
17. `entity_only_masking`
18. `generic_mask_slot`
19. `post_hoc_gold_removed_expected_answer`
20. `length_matched_neutral_filler`

## Private-Like Benchmark Construction

### Core idea

Construct a reproducible renamed benchmark from a public corpus so that:

- the evidence structure is preserved
- relevance labels are preserved
- answer strings remain in the corpus
- parametric priors from pretraining become much less useful

### Renaming requirements

The renaming pipeline must preserve:

1. one-to-one mapping between original and renamed entities
2. consistency across corpus, queries, and qrels
3. coarse semantic type
4. local grammatical plausibility

It should break:

1. exact memorized public names
2. obvious world-knowledge associations
3. answer lookup through public prior alone

### Type system

Use a typed alias inventory such as:

- PERSON -> `Employee ZQ-17`, `Researcher AV-04`
- LOCATION -> `Site LM-42`, `Region PX-3`
- ORGANIZATION -> `Division RK-9`, `Unit TQ-5`
- DATE/TIME -> structured synthetic dates or coded periods
- TITLE/WORK -> `Project Kappa-12`, `Record MV-8`
- NUMBER -> optionally left unchanged or mapped depending on the experiment

### Renaming procedure

1. Detect named entities and answer spans in corpus and queries.
2. Build a per-dataset alias table by entity type.
3. Replace mentions consistently across corpus and query text.
4. Preserve answer-bearing evidence in relevant documents.
5. Validate with spot checks plus automatic constraints:
   - alias consistency
   - no original names remain
   - answer still appears in at least one relevant document
   - query type is preserved

### Number handling

Use two benchmark variants:

1. `rename-entities-only`
2. `rename-entities-and-sensitive-values`

This separates the effect of named-entity priors from numeric/date priors.

## Query Generation / Reformulation Conditions

### Raw expected answer

Prompt the model to produce a concise expected answer in one sentence.

Risk:
- may contain exact gold answer
- may contain unsupported but plausible alternatives

### Query-aware masking

Generate a concise expected answer, then mask only the answer-bearing span while
preserving anchors that already appear in the query.

Goal:
- preserve answer-shape context
- suppress direct answer token leakage

### Answer-agnostic templates

Generate a structured reformulation that:

- preserves known query anchors
- includes relation intent
- uses typed slots for the unknown answer
- forbids introduction of new concrete answer candidates

This is the most distinctive new method proposed in this memo.

### HyDE

Generate a longer hypothetical supporting passage.

Role in the paper:
- strong baseline
- likely high-risk for prior-induced answer injection

## Retrieval Settings

### Sparse retrieval

Required:

- BM25

Recommended:

- BM25 + RM3

Sparse retrieval is important because answer leakage can help through direct
lexical overlap.

### Dense retrieval

Required:

- one zero-shot dense retriever such as `bge-base-en-v1.5`, `e5-base-v2`, or
  `Contriever`

Recommended:

- one stronger dense retriever if infrastructure allows

Dense retrieval is important because it tests whether leakage effects persist
without exact lexical matching.

### Integration modes

For each method family, compare:

1. expansion only
2. concatenation
3. reciprocal-rank fusion
4. weighted reciprocal-rank fusion

## Datasets

### Phase 1: Minimum viable paper set

1. `Natural Questions`
2. `HotpotQA`
3. `SciFact`

Rationale:

- NQ gives public entity-centric QA
- HotpotQA gives multi-hop retrieval
- SciFact gives specialized domain shift

### Phase 2: Optional additions

- `FiQA` for finance-oriented retrieval
- one in-house or semi-private dataset if available and publishable

## Metrics

### Main retrieval metrics

- `nDCG@10`
- `Recall@10`
- `Recall@20`
- `MRR@10` for QA-style datasets

### Leakage-sensitive metrics

1. exact answer leakage rate
2. masked answer residual leakage rate
3. unsupported entity injection rate
4. unsupported value/date injection rate
5. gains on answer-containing expansions only
6. gains on answer-absent expansions only
7. delta from public to private-like performance

### Robustness metrics

1. relative drop from public to private-like
2. performance on query-only failure cases
3. sensitivity by answer type:
   - person
   - location
   - organization
   - date
   - number

### Optional semantic metrics

If feasible:

- NLI-style entailment between generated expansion and gold evidence
- entity overlap between generated expansion and relevant documents
- lexical overlap of expansion with gold answer aliases

## Statistical Analysis

Use paired evaluation because methods are compared on the same queries.

Recommended:

1. paired bootstrap confidence intervals for `nDCG@10` and `MRR@10`
2. randomization or paired permutation tests for primary comparisons
3. report per-query win/tie/loss counts for:
   - raw expected vs masked expected
   - masked expected vs answer-agnostic template
   - HyDE vs answer-agnostic template

The most important comparisons are not just average scores; they are:

1. performance drop from public to private-like
2. performance stratified by leakage bucket
3. whether leakage-aware methods are more stable under renaming

## Recommended Main Tables

### Table 1

Overall performance on public benchmarks.

### Table 2

Overall performance on private-like renamed benchmarks.

### Table 3

Public-to-private delta by method.

### Table 4

Leakage-bucket analysis:

- exact-answer present
- exact-answer absent
- unsupported entity/value injected

### Table 5

Masking and template ablations:

- typed mask
- generic mask
- oracle mask
- random mask
- entity-only mask
- answer-agnostic template

## Recommended Main Figures

1. Histogram of public-to-private performance drop per method
2. Leakage rate vs retrieval gain scatterplot
3. Example queries showing:
   - public success via answer leakage
   - private-like failure of raw expansion
   - recovery via masked/template reformulation

## Concrete Implementation Plan for This Repo

### What already exists

The current repository already supports:

- BM25 retrieval
- one dense retriever path
- expected-answer generation
- HyDE generation
- query-aware masking
- concatenation
- RRF and weighted RRF
- per-query record dumps

### What must be added

1. A renamed benchmark construction pipeline
   - likely a new script under `scripts/`
   - probably a new module under `src/expected_answer_rag/`

2. Answer-bearing gold metadata in loaded queries
   - current BEIR loading does not populate query answers
   - this blocks accurate leakage accounting on current public runs

3. An answer-agnostic template generator
   - prompt
   - JSON schema
   - retrieval query extraction

4. New controls
   - oracle masking
   - random masking
   - entity-only masking
   - generic mask
   - post-hoc gold removal

5. Stronger leakage analysis
   - exact match
   - alias match
   - unsupported entity injection
   - private-like stability metrics

6. Statistical testing and paper-ready aggregation scripts

### Suggested implementation order

1. Add answer-bearing metadata support for datasets with available gold answers
2. Add answer-agnostic template generation
3. Add private-like renaming pipeline
4. Add control variants
5. Run BM25 public vs private-like
6. Run dense public vs private-like
7. Perform significance testing and error analysis

## Risks and How to Manage Them

### Risk 1: Masking simply hurts retrieval

That is acceptable if the paper's claim is stability and contamination control,
not universal metric improvement.

### Risk 2: Private-like renaming makes the benchmark too artificial

Mitigation:

- preserve entity types
- preserve evidence structure
- include both renamed and original conditions
- include human spot checks

### Risk 3: Reviewers argue this is just contamination, not retrieval

Mitigation:

- frame it as a retrieval-specific confound caused by generated reformulations
- show retriever-dependent behavior
- show practical implications for enterprise/private RAG

### Risk 4: Answer-agnostic templates underperform badly

Mitigation:

- present them as a strict leakage-control baseline
- compare them on private-like stability, not only public benchmark peak score

### Risk 5: Current repo results are too small

Mitigation:

- explicitly treat current results as pilot evidence
- run at least one larger public benchmark sweep before paper drafting

## Recommended Claim Boundaries

### Safe claims

1. Generated retrieval expansions can contain answer-bearing information that is
   not present in the original query.
2. Public-benchmark retrieval gains do not automatically imply genuine query
   reformulation gains.
3. Leakage-aware reformulations can reduce direct answer injection.
4. Public-to-private-like evaluation reveals different behavior than public-only
   evaluation.

### Claims that require strong evidence

1. Masked expected answers are universally better than HyDE
2. Answer-agnostic templates are best on all datasets
3. Public benchmark gains are mostly contamination-driven

## Minimum Viable Paper Package

To make this submission-ready at a strong venue, the minimum convincing package
is:

1. Three datasets across at least two task types
2. Public and private-like conditions for each dataset
3. BM25 plus one dense retriever
4. HyDE, Query2doc-style concat, raw expected answer, masked expected answer,
   answer-agnostic template
5. Leakage-sensitive analysis with exact-answer and unsupported-injection rates
6. At least two ablation controls
7. Qualitative examples and significance testing

If the project falls short of this package, it is more appropriate for a
workshop, short paper, or strong internal technical report than a top
conference.

## Recommendation to PI

The repo should pivot fully to the leakage-focused, private-like evaluation
framing described in this memo.

The currently implemented baseline package should remain in the project, but only as:

- baseline infrastructure
- ablation support
- pilot evidence motivating the leakage question

The paper should be sold as a contamination-aware retrieval study for
LLM-generated expansions, with private-like renamed evaluation and leakage-aware
reformulation as the core novelty.
