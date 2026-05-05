# Additional Experiment Results

This report summarizes the PI-requested follow-up experiments available under `outputs_pi/c3000`. Unless otherwise stated, all scores are `nDCG@10`. Error bars are paired bootstrap 95% confidence intervals over queries, reported as approximate half-widths.

## Output Inventory

Completed artifacts:

- Larger-corpus BM25 replication at `max_corpus=3000` for NQ, HotpotQA, and SciFact across Public, Entity-CF, and Entity+Value-CF.
- Cross-regime leakage-isolation runs for `expected`, `query2doc`, and `hyde` expansions across Entity-CF and Entity+Value-CF corpora.
- Held-out FAWE beta selection for `fawe_query2doc_beta*`.
- Expansion-content audit summaries.
- Postprocessed metric error bars for the new runs and legacy outputs.
- Figures with error bars in `outputs_pi/c3000/figures`.

Incomplete artifacts:

- The dense/hybrid follow-up is not complete. I found `outputs_pi/c3000/nq_100_c3000_bge.log` and BGE embedding cache files, but no dense, E5, Contriever, or hybrid `*_run.json` outputs. The log stops during generation precomputation for NQ BGE, so this family should not be used for claims yet.

## Qrel Coverage

Increasing the corpus cap from 2k to 3k increases the evaluable query count for NQ and HotpotQA while keeping zero missing-coverage queries. SciFact already had 100 evaluable queries at 2k, but 3k improves mean qrel coverage.

| Dataset | Corpus cap | Corpus docs | Eval queries | Mean qrel coverage | Min qrel coverage | Zero-coverage queries |
|---|---:|---:|---:|---:|---:|---:|
| NQ | 2000 | 2000 | 68 | 1.000 | 1.000 | 0 |
| NQ | 3000 | 3000 | 92 | 1.000 | 1.000 | 0 |
| HotpotQA | 2000 | 2000 | 85 | 0.500 | 0.500 | 0 |
| HotpotQA | 3000 | 3000 | 100 | 0.505 | 0.500 | 0 |
| SciFact | 2000 | 2000 | 100 | 0.949 | 0.200 | 0 |
| SciFact | 3000 | 3000 | 100 | 0.988 | 0.500 | 0 |

## Larger-Corpus BM25 Replication

The 3k replication preserves the main counterfactual finding: FAWE-Q2D is better than naive Query2doc in all six counterfactual conditions. Public performance is mixed, with Query2doc slightly ahead on NQ, FAWE-Q2D ahead on HotpotQA and SciFact, and anchored expected-answer concatenation strongest on public HotpotQA.

| Dataset | Regime | Query | AnswerOnly | HyDE | Query2doc | AnchoredAnswer | FAWE-Q2D | FAWE-Adapt | Best method |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| NQ | Public | 0.4887 | 0.7314 | 0.7285 | 0.7526 | 0.6994 | 0.7367 | 0.5068 | Query2doc 0.7526 |
| NQ | Entity-CF | 0.3459 | 0.4122 | 0.3509 | 0.4057 | 0.4185 | 0.4412 | 0.3708 | GRF 0.4772 |
| NQ | Entity+Value-CF | 0.3455 | 0.3621 | 0.3331 | 0.3845 | 0.4039 | 0.4275 | 0.3812 | FAWE-Q2D beta 0.10 0.4393 |
| HotpotQA | Public | 0.7931 | 0.8193 | 0.8540 | 0.8245 | 0.8814 | 0.8478 | 0.8020 | AnchoredAnswer 0.8814 |
| HotpotQA | Entity-CF | 0.4167 | 0.2919 | 0.1721 | 0.3373 | 0.4388 | 0.3997 | 0.4331 | FAWE-answer beta 0.75 0.4548 |
| HotpotQA | Entity+Value-CF | 0.4038 | 0.2467 | 0.1085 | 0.2801 | 0.3932 | 0.3828 | 0.4275 | FAWE-answer beta 0.50 0.4421 |
| SciFact | Public | 0.7513 | 0.7313 | 0.7420 | 0.7744 | 0.7593 | 0.7817 | 0.7461 | FAWE-Q2D 0.7817 |
| SciFact | Entity-CF | 0.6399 | 0.6364 | 0.5419 | 0.6302 | 0.6685 | 0.6528 | 0.6507 | FAWE-answer beta 0.75 0.6754 |
| SciFact | Entity+Value-CF | 0.6435 | 0.6143 | 0.5750 | 0.6483 | 0.6814 | 0.6560 | 0.6527 | AnchoredAnswer 0.6814 |

Average across the three datasets:

| Regime | Query | AnswerOnly | HyDE | Query2doc | AnchoredAnswer | FAWE-Q2D | FAWE-Adapt |
|---|---:|---:|---:|---:|---:|---:|---:|
| Public | 0.6777 | 0.7606 | 0.7749 | 0.7838 | 0.7800 | 0.7887 | 0.6850 |
| Entity-CF | 0.4675 | 0.4468 | 0.3549 | 0.4577 | 0.5086 | 0.4979 | 0.4849 |
| Entity+Value-CF | 0.4643 | 0.4077 | 0.3389 | 0.4376 | 0.4929 | 0.4888 | 0.4871 |

FAWE-Q2D versus Query2doc with bootstrap error bars:

| Dataset | Regime | Query2doc | FAWE-Q2D | Delta |
|---|---|---:|---:|---:|
| NQ | Public | 0.7526 ± 0.0609 | 0.7367 ± 0.0657 | -0.0159 |
| NQ | Entity-CF | 0.4057 ± 0.0736 | 0.4412 ± 0.0764 | +0.0355 |
| NQ | Entity+Value-CF | 0.3845 ± 0.0798 | 0.4275 ± 0.0825 | +0.0431 |
| HotpotQA | Public | 0.8245 ± 0.0619 | 0.8478 ± 0.0618 | +0.0233 |
| HotpotQA | Entity-CF | 0.3373 ± 0.0837 | 0.3997 ± 0.0865 | +0.0624 |
| HotpotQA | Entity+Value-CF | 0.2801 ± 0.0748 | 0.3828 ± 0.0820 | +0.1027 |
| SciFact | Public | 0.7744 ± 0.0634 | 0.7817 ± 0.0603 | +0.0073 |
| SciFact | Entity-CF | 0.6302 ± 0.0883 | 0.6528 ± 0.0842 | +0.0226 |
| SciFact | Entity+Value-CF | 0.6483 ± 0.0790 | 0.6560 ± 0.0803 | +0.0077 |

Compared with the 2k BM25 follow-up, absolute scores generally decrease at 3k because the candidate pool is harder, but the FAWE-Q2D utility pattern does not flip. The counterfactual FAWE-Q2D gains over Query2doc remain positive in every dataset/regime.

| Dataset | Regime | 2k Query2doc | 3k Query2doc | 2k FAWE-Q2D | 3k FAWE-Q2D | 2k Delta | 3k Delta |
|---|---|---:|---:|---:|---:|---:|---:|
| NQ | Public | 0.7801 | 0.7526 | 0.7783 | 0.7367 | -0.0018 | -0.0159 |
| NQ | Entity-CF | 0.4856 | 0.4057 | 0.5319 | 0.4412 | +0.0463 | +0.0355 |
| NQ | Entity+Value-CF | 0.4596 | 0.3845 | 0.5058 | 0.4275 | +0.0462 | +0.0431 |
| HotpotQA | Public | 0.8793 | 0.8245 | 0.9059 | 0.8478 | +0.0266 | +0.0233 |
| HotpotQA | Entity-CF | 0.3951 | 0.3373 | 0.4803 | 0.3997 | +0.0852 | +0.0624 |
| HotpotQA | Entity+Value-CF | 0.3669 | 0.2801 | 0.4771 | 0.3828 | +0.1103 | +0.1027 |
| SciFact | Public | 0.8281 | 0.7744 | 0.8312 | 0.7817 | +0.0031 | +0.0073 |
| SciFact | Entity-CF | 0.6835 | 0.6302 | 0.7057 | 0.6528 | +0.0222 | +0.0226 |
| SciFact | Entity+Value-CF | 0.6840 | 0.6483 | 0.6995 | 0.6560 | +0.0154 | +0.0077 |

FAWE-Q2D recall:

| Dataset | Regime | Recall@20 | Recall@100 |
|---|---|---:|---:|
| NQ | Public | 0.9728 | 1.0000 |
| NQ | Entity-CF | 0.7464 | 0.8768 |
| NQ | Entity+Value-CF | 0.7138 | 0.8551 |
| HotpotQA | Public | 0.9600 | 0.9800 |
| HotpotQA | Entity-CF | 0.5850 | 0.6750 |
| HotpotQA | Entity+Value-CF | 0.5650 | 0.6750 |
| SciFact | Public | 0.9300 | 0.9700 |
| SciFact | Entity-CF | 0.8292 | 0.8817 |
| SciFact | Entity+Value-CF | 0.8367 | 0.9042 |

## Cross-Regime Leakage Isolation

This experiment evaluates public generations `g_pub` and counterfactual generations `g_cf` against public and counterfactual corpora. The first clear result is that public generated-only retrieval loses substantially on counterfactual corpora. The average public-to-CF drop is 0.235 to 0.350 depending on expansion type and regime.

| Expansion | Regime | g_pub on public | g_pub on CF | Drop | q+g_pub on public | q+g_pub on CF | Drop |
|---|---|---:|---:|---:|---:|---:|---:|
| Expected answer | Entity-CF | 0.7606 | 0.4596 | +0.3011 | 0.7800 | 0.5458 | +0.2342 |
| Expected answer | Entity+Value-CF | 0.7606 | 0.4105 | +0.3501 | 0.7800 | 0.5066 | +0.2735 |
| Query2doc | Entity-CF | 0.7764 | 0.5254 | +0.2510 | 0.7838 | 0.5514 | +0.2324 |
| Query2doc | Entity+Value-CF | 0.7764 | 0.4889 | +0.2875 | 0.7838 | 0.5192 | +0.2646 |
| HyDE | Entity-CF | 0.7749 | 0.5395 | +0.2354 | 0.7920 | 0.6027 | +0.1893 |
| HyDE | Entity+Value-CF | 0.7749 | 0.5115 | +0.2633 | 0.7920 | 0.5730 | +0.2190 |

The second result is that anchoring and FAWE rescue counterfactual performance more reliably than regenerating under the counterfactual query. For Query2doc, `g_cf` is usually worse than `g_pub` on the CF corpus, while `q_cf + g_pub` and FAWE with `g_pub` are stronger. This means the current experiment supports public-prior brittleness and query anchoring, but it does not support a simple claim that counterfactual generation alone recovers performance.

| Dataset | Regime | g_pub→CF | g_cf→CF | q_cf+g_pub | q_cf+g_cf | FAWE g_pub | FAWE g_cf | Anchor rescue | CF-gen gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NQ | Entity-CF | 0.5637 | 0.3864 | 0.5764 | 0.4057 | 0.5706 | 0.4412 | +0.0127 | -0.1773 |
| NQ | Entity+Value-CF | 0.5019 | 0.3652 | 0.5236 | 0.3845 | 0.5508 | 0.4275 | +0.0217 | -0.1367 |
| HotpotQA | Entity-CF | 0.3769 | 0.2670 | 0.4307 | 0.3373 | 0.5347 | 0.3997 | +0.0538 | -0.1099 |
| HotpotQA | Entity+Value-CF | 0.3351 | 0.2187 | 0.3857 | 0.2801 | 0.4965 | 0.3828 | +0.0506 | -0.1164 |
| SciFact | Entity-CF | 0.6357 | 0.6210 | 0.6471 | 0.6302 | 0.6632 | 0.6528 | +0.0114 | -0.0147 |
| SciFact | Entity+Value-CF | 0.6298 | 0.6485 | 0.6484 | 0.6483 | 0.6635 | 0.6560 | +0.0186 | +0.0187 |

Average cross-regime results:

| Expansion | Regime | g_pub→CF | g_cf→CF | q_cf+g_pub | q_cf+g_cf | FAWE g_pub | FAWE g_cf | Anchor rescue | CF-gen gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Expected answer | Entity-CF | 0.4596 | 0.4468 | 0.5458 | 0.5086 | 0.5153 | 0.4991 | +0.0863 | -0.0128 |
| Expected answer | Entity+Value-CF | 0.4105 | 0.4077 | 0.5066 | 0.4929 | 0.4978 | 0.4861 | +0.0961 | -0.0028 |
| Query2doc | Entity-CF | 0.5254 | 0.4248 | 0.5514 | 0.4577 | 0.5895 | 0.4979 | +0.0260 | -0.1006 |
| Query2doc | Entity+Value-CF | 0.4889 | 0.4108 | 0.5192 | 0.4376 | 0.5703 | 0.4888 | +0.0303 | -0.0781 |
| HyDE | Entity-CF | 0.5395 | 0.3549 | 0.6027 | 0.4358 | 0.6353 | 0.5073 | +0.0632 | -0.1846 |
| HyDE | Entity+Value-CF | 0.5115 | 0.3389 | 0.5730 | 0.4191 | 0.6182 | 0.5021 | +0.0614 | -0.1727 |

Interpretation: generated-only public expansions clearly lose under CF corpora, but the strongest CF condition is often not `g_cf`; it is query-anchored or FAWE retrieval using `g_pub`. This suggests the public expansion still contributes useful non-entity relation and lexical structure, while CF generation may degrade expansion quality.

## Held-Out Beta Selection

Held-out beta selection supports the PI’s robustness concern. Public-only selection can choose a larger beta, but average/robust objectives select smaller betas for HotpotQA and SciFact. NQ consistently selects 0.10.

| Dataset | Objective | Selected beta | Dev queries | Test queries | Public test | Entity-CF test | Entity+Value-CF test |
|---|---|---:|---:|---:|---:|---:|---:|
| NQ | Public | 0.10 | 24 | 68 | 0.6906 | 0.4717 | 0.4505 |
| NQ | Average | 0.10 | 24 | 68 | 0.6906 | 0.4717 | 0.4505 |
| NQ | Robust | 0.10 | 24 | 68 | 0.6906 | 0.4717 | 0.4505 |
| HotpotQA | Public | 0.50 | 29 | 71 | 0.8482 | 0.3672 | 0.3202 |
| HotpotQA | Average | 0.05 | 29 | 71 | 0.8548 | 0.4309 | 0.4160 |
| HotpotQA | Robust | 0.05 | 29 | 71 | 0.8548 | 0.4309 | 0.4160 |
| SciFact | Public | 0.25 | 28 | 72 | 0.7729 | 0.6513 | 0.6502 |
| SciFact | Average | 0.05 | 28 | 72 | 0.7652 | 0.6588 | 0.6445 |
| SciFact | Robust | 0.05 | 28 | 72 | 0.7652 | 0.6588 | 0.6445 |

The full 3k beta sweep, averaged across datasets, shows the same pattern: public performance peaks around beta 0.50, while both counterfactual regimes peak at beta 0.10.

| Regime | beta=0.05 | beta=0.10 | beta=0.25 | beta=0.50 | beta=0.75 | beta=1.00 | Best |
|---|---:|---:|---:|---:|---:|---:|---|
| Public | 0.7665 | 0.7818 | 0.7887 | 0.7894 | 0.7847 | 0.7838 | 0.50 |
| Entity-CF | 0.5082 | 0.5196 | 0.4979 | 0.4819 | 0.4705 | 0.4577 | 0.10 |
| Entity+Value-CF | 0.4982 | 0.5075 | 0.4888 | 0.4622 | 0.4446 | 0.4376 | 0.10 |

## Expansion Content Audit

The automatic audit labels most Query2doc, HyDE, and FAWE-Q2D expansions as answer-bearing because they contain candidate or unsupported injected entities. Exact-answer and alias-answer rates are zero in this audit because these datasets mostly lack explicit answer-alias metadata in the BEIR-loaded records; the candidate and unsupported-injection columns are more informative.

Average across datasets:

| Method | Answer-bearing | Candidate injection | Unsupported injection | Public gain | Entity excess drop | E+V excess drop |
|---|---:|---:|---:|---:|---:|---:|
| AnswerOnly | 0.79 | 0.79 | 0.65 | +0.0829 | +0.1036 | +0.1395 |
| HyDE | 1.00 | 1.00 | 1.00 | +0.0972 | +0.2097 | +0.2226 |
| Query2doc | 1.00 | 1.00 | 1.00 | +0.1061 | +0.1159 | +0.1328 |
| AnchoredAnswer | 0.79 | 0.79 | 0.65 | +0.1023 | +0.0613 | +0.0738 |
| FAWE-Q2D | 1.00 | 1.00 | 1.00 | +0.1110 | +0.0806 | +0.0865 |
| FAWE-Adapt | 1.00 | 1.00 | 1.00 | +0.0073 | -0.0101 | -0.0155 |

Dataset-level audit highlights:

| Dataset | Method | Answer-bearing | Public gain | Entity excess drop | E+V excess drop |
|---|---|---:|---:|---:|---:|
| NQ | HyDE | 1.00 | +0.2398 | +0.2349 | +0.2523 |
| NQ | Query2doc | 1.00 | +0.2639 | +0.2042 | +0.2250 |
| NQ | FAWE-Q2D | 1.00 | +0.2480 | +0.1528 | +0.1660 |
| HotpotQA | HyDE | 1.00 | +0.0610 | +0.3056 | +0.3563 |
| HotpotQA | Query2doc | 1.00 | +0.0314 | +0.1107 | +0.1551 |
| HotpotQA | FAWE-Q2D | 1.00 | +0.0547 | +0.0716 | +0.0757 |
| SciFact | Query2doc | 1.00 | +0.0231 | +0.0328 | +0.0183 |
| SciFact | AnchoredAnswer | 0.79 | +0.0080 | -0.0206 | -0.0299 |
| SciFact | FAWE-Q2D | 1.00 | +0.0304 | +0.0175 | +0.0178 |

The audit supports the paper mechanism: answer-bearing/candidate-injecting expansions can produce public gains and larger counterfactual excess drops. AnchoredAnswer and FAWE-Q2D reduce the excess drop relative to naive generated-only or pseudo-document methods. FAWE-Adapt is the most stable but gives little public gain, so it is better framed as a conservative robustness control than the main effectiveness method.

## Cost and Runtime

Recorded wall-clock runtime for the BM25 3k runs:

| Dataset | Public | Entity-CF | Entity+Value-CF |
|---|---:|---:|---:|
| NQ | 1.1m | 38.6m | 55.2m |
| HotpotQA | 11.2m | 42.1m | 51.4m |
| SciFact | 1.8m | 90.1m | 160.8m |

The counterfactual runs are much slower because they include artifact construction and full-corpus counterfactual rewriting. Since the artifacts are cached under `outputs_pi/c3000/counterfactual_artifacts`, reruns should be faster.

## Figures Generated

The postprocess step generated error-bar figures:

- `outputs_pi/c3000/figures/bm25_utility_with_error_bars.pdf`
- `outputs_pi/c3000/figures/fawe_beta_sweep_with_error_bars.pdf`
- `outputs_pi/c3000/figures/cross_regime_query2doc_with_error_bars.pdf`

PNG versions are available in the same directory.

## Bottom-Line Findings

The strongest completed result is the larger-corpus replication. At 3k corpus size, FAWE-Q2D remains better than Query2doc in every counterfactual BM25 condition. The gain is largest on HotpotQA Entity+Value-CF (+0.1027) and remains positive on SciFact, where the task is less answer-string driven.

The cross-regime experiment clarifies the mechanism. Public generated expansions lose substantially on counterfactual corpora, confirming leakage or public-prior brittleness. However, counterfactual generation does not reliably recover performance. The more reliable recovery comes from preserving the counterfactual query as an anchor and using generated text as auxiliary evidence, especially through FAWE.

Held-out beta selection strengthens the robustness story. Public-only tuning can choose larger beta values, but robust objectives prefer smaller beta values, and the averaged beta sweep peaks at beta 0.10 in both counterfactual regimes. This directly supports presenting FAWE as a controlled fielded integration method rather than unrestricted pseudo-document concatenation.

The expansion audit connects content to behavior. Candidate-injecting expansions are common and show larger excess drops, while anchored and FAWE variants reduce the damage. FAWE-Adapt is very stable but sacrifices effectiveness, so the paper should position FAWE-Q2D as the main method and FAWE-Adapt as a conservative deployment variant.

The dense/generalized retriever claim remains unresolved because the dense/hybrid follow-up did not finish. The paper can keep the existing dense results as secondary evidence, but claims about BGE versus E5/Contriever/hybrid should wait until those missing runs complete.
