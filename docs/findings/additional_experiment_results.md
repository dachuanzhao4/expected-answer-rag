# Additional Experiment Results

This report summarizes the PI-requested follow-up experiments under `outputs_pi/c3000`. Unless otherwise stated, scores are `nDCG@10`. Bootstrap error bars are paired 95% confidence intervals over queries, shown as approximate half-widths where included.

## Output Inventory

The new result package is now complete for the requested retrieval matrix:

| Family | Completed outputs |
|---|---:|
| BM25 larger-corpus replication | 3 datasets x 3 regimes |
| Dense BGE | 3 datasets x 3 regimes |
| Dense E5 | 3 datasets x 3 regimes |
| Dense Contriever | 3 datasets x 3 regimes |
| Hybrid BM25+E5 | 3 datasets x 3 regimes |
| Cross-regime leakage isolation | 3 datasets x 2 CF regimes x 3 expansion types |
| Held-out beta selection | 3 datasets x 3 objectives |
| Expansion content audit | 3 datasets |

The run directory contains 45 `*_run.json` files for the retriever matrix. Generated figures with error bars are in `outputs_pi/c3000/figures`.

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

## BM25 Larger-Corpus Replication

The 3k BM25 replication preserves the main counterfactual result: FAWE-Q2D is better than naive Query2doc in every counterfactual condition. Public performance is mixed, but the qualitative counterfactual ranking does not flip.

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

Average across datasets:

| Regime | Query | AnswerOnly | HyDE | Query2doc | AnchoredAnswer | FAWE-Q2D | FAWE-Adapt |
|---|---:|---:|---:|---:|---:|---:|---:|
| Public | 0.6777 | 0.7606 | 0.7749 | 0.7838 | 0.7800 | 0.7887 | 0.6850 |
| Entity-CF | 0.4675 | 0.4468 | 0.3549 | 0.4577 | 0.5086 | 0.4979 | 0.4849 |
| Entity+Value-CF | 0.4643 | 0.4077 | 0.3389 | 0.4376 | 0.4929 | 0.4888 | 0.4871 |

FAWE-Q2D versus Query2doc:

| Dataset | Regime | Query2doc | FAWE-Q2D | Delta |
|---|---|---:|---:|---:|
| NQ | Public | 0.7526 +/- 0.0609 | 0.7367 +/- 0.0657 | -0.0159 |
| NQ | Entity-CF | 0.4057 +/- 0.0736 | 0.4412 +/- 0.0764 | +0.0355 |
| NQ | Entity+Value-CF | 0.3845 +/- 0.0798 | 0.4275 +/- 0.0825 | +0.0431 |
| HotpotQA | Public | 0.8245 +/- 0.0619 | 0.8478 +/- 0.0618 | +0.0233 |
| HotpotQA | Entity-CF | 0.3373 +/- 0.0837 | 0.3997 +/- 0.0865 | +0.0624 |
| HotpotQA | Entity+Value-CF | 0.2801 +/- 0.0748 | 0.3828 +/- 0.0820 | +0.1027 |
| SciFact | Public | 0.7744 +/- 0.0634 | 0.7817 +/- 0.0603 | +0.0073 |
| SciFact | Entity-CF | 0.6302 +/- 0.0883 | 0.6528 +/- 0.0842 | +0.0226 |
| SciFact | Entity+Value-CF | 0.6483 +/- 0.0790 | 0.6560 +/- 0.0803 | +0.0077 |

Compared with the 2k BM25 follow-up, absolute scores generally decrease at 3k because retrieval is harder, but the counterfactual FAWE-Q2D gains remain positive.

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

FAWE-Q2D recall remains high in public retrieval and drops most on HotpotQA counterfactual retrieval:

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

## Dense and Hybrid Retriever Matrix

The dense/hybrid matrix supports a broad alias-sensitivity claim, with an important caveat: FAWE-Q2D is not uniformly the strongest dense integration strategy. All dense models lose much more query-only performance under counterfactual rewriting than BM25. Hybrid BM25+E5 is more robust than pure dense retrieval, but still drops more than BM25 on average.

Average query-only score by retriever:

| Retriever | Public | Entity-CF | Entity+Value-CF | Public to Entity drop | Public to E+V drop |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.6777 | 0.4675 | 0.4643 | +0.2102 | +0.2134 |
| BGE | 0.8258 | 0.3798 | 0.3724 | +0.4460 | +0.4534 |
| E5 | 0.8078 | 0.3829 | 0.3850 | +0.4249 | +0.4228 |
| Contriever | 0.7118 | 0.3329 | 0.3187 | +0.3789 | +0.3931 |
| Hybrid BM25+E5 | 0.7728 | 0.4955 | 0.4862 | +0.2773 | +0.2866 |

Average `nDCG@10` by retriever, regime, and core method:

| Retriever | Regime | Query | HyDE | Query2doc | AnchoredAnswer | FAWE-Q2D | FAWE-Adapt |
|---|---|---:|---:|---:|---:|---:|---:|
| BM25 | Public | 0.6777 | 0.7749 | 0.7838 | 0.7800 | 0.7887 | 0.6850 |
| BM25 | Entity-CF | 0.4675 | 0.3549 | 0.4577 | 0.5086 | 0.4979 | 0.4849 |
| BM25 | Entity+Value-CF | 0.4643 | 0.3389 | 0.4376 | 0.4929 | 0.4888 | 0.4871 |
| BGE | Public | 0.8258 | 0.8310 | 0.8335 | 0.8397 | 0.8318 | 0.8183 |
| BGE | Entity-CF | 0.3798 | 0.3647 | 0.3949 | 0.4011 | 0.3873 | 0.3880 |
| BGE | Entity+Value-CF | 0.3724 | 0.3730 | 0.3901 | 0.3994 | 0.3871 | 0.3754 |
| E5 | Public | 0.8078 | 0.7976 | 0.7760 | 0.8334 | 0.8177 | 0.8016 |
| E5 | Entity-CF | 0.3829 | 0.3275 | 0.3583 | 0.4054 | 0.3825 | 0.3836 |
| E5 | Entity+Value-CF | 0.3850 | 0.3152 | 0.3349 | 0.4002 | 0.3846 | 0.3783 |
| Contriever | Public | 0.7118 | 0.7994 | 0.7831 | 0.7816 | 0.7548 | 0.7195 |
| Contriever | Entity-CF | 0.3329 | 0.3503 | 0.3704 | 0.3578 | 0.3537 | 0.3392 |
| Contriever | Entity+Value-CF | 0.3187 | 0.3538 | 0.3571 | 0.3415 | 0.3406 | 0.3287 |
| Hybrid BM25+E5 | Public | 0.7728 | 0.8085 | 0.8125 | 0.8294 | 0.8061 | 0.7791 |
| Hybrid BM25+E5 | Entity-CF | 0.4955 | 0.4294 | 0.4666 | 0.5270 | 0.4995 | 0.4902 |
| Hybrid BM25+E5 | Entity+Value-CF | 0.4862 | 0.4144 | 0.4655 | 0.4992 | 0.4899 | 0.4784 |

FAWE-Q2D versus Query2doc by retriever:

| Retriever | Public delta | Entity-CF delta | Entity+Value-CF delta |
|---|---:|---:|---:|
| BM25 | +0.0049 | +0.0402 | +0.0512 |
| BGE | -0.0017 | -0.0076 | -0.0030 |
| E5 | +0.0417 | +0.0242 | +0.0497 |
| Contriever | -0.0283 | -0.0167 | -0.0166 |
| Hybrid BM25+E5 | -0.0064 | +0.0329 | +0.0243 |

Best core method by retriever and regime:

| Retriever | Regime | Best core method | Score | Query-only score |
|---|---|---|---:|---:|
| BM25 | Public | FAWE-Q2D | 0.7887 | 0.6777 |
| BM25 | Entity-CF | AnchoredAnswer | 0.5086 | 0.4675 |
| BM25 | Entity+Value-CF | AnchoredAnswer | 0.4929 | 0.4643 |
| BGE | Public | AnchoredAnswer | 0.8397 | 0.8258 |
| BGE | Entity-CF | AnchoredAnswer | 0.4011 | 0.3798 |
| BGE | Entity+Value-CF | AnchoredAnswer | 0.3994 | 0.3724 |
| E5 | Public | AnchoredAnswer | 0.8334 | 0.8078 |
| E5 | Entity-CF | AnchoredAnswer | 0.4054 | 0.3829 |
| E5 | Entity+Value-CF | AnchoredAnswer | 0.4002 | 0.3850 |
| Contriever | Public | HyDE | 0.7994 | 0.7118 |
| Contriever | Entity-CF | Query2doc | 0.3704 | 0.3329 |
| Contriever | Entity+Value-CF | Query2doc | 0.3571 | 0.3187 |
| Hybrid BM25+E5 | Public | AnchoredAnswer | 0.8294 | 0.7728 |
| Hybrid BM25+E5 | Entity-CF | AnchoredAnswer | 0.5270 | 0.4955 |
| Hybrid BM25+E5 | Entity+Value-CF | AnchoredAnswer | 0.4992 | 0.4862 |

Per-dataset query-only scores show that HotpotQA is the clearest dense alias-sensitivity case:

| Retriever | Dataset | Public | Entity-CF | Entity+Value-CF |
|---|---|---:|---:|---:|
| BM25 | NQ | 0.4887 | 0.3459 | 0.3455 |
| BM25 | HotpotQA | 0.7931 | 0.4167 | 0.4038 |
| BM25 | SciFact | 0.7513 | 0.6399 | 0.6435 |
| BGE | NQ | 0.7987 | 0.4268 | 0.4453 |
| BGE | HotpotQA | 0.8905 | 0.2014 | 0.2174 |
| BGE | SciFact | 0.7883 | 0.5113 | 0.4545 |
| E5 | NQ | 0.7852 | 0.4618 | 0.4591 |
| E5 | HotpotQA | 0.8834 | 0.1906 | 0.2479 |
| E5 | SciFact | 0.7548 | 0.4964 | 0.4480 |
| Contriever | NQ | 0.6174 | 0.3273 | 0.3366 |
| Contriever | HotpotQA | 0.8075 | 0.1729 | 0.1921 |
| Contriever | SciFact | 0.7105 | 0.4985 | 0.4274 |
| Hybrid BM25+E5 | NQ | 0.6460 | 0.4555 | 0.4654 |
| Hybrid BM25+E5 | HotpotQA | 0.8798 | 0.3800 | 0.3736 |
| Hybrid BM25+E5 | SciFact | 0.7928 | 0.6510 | 0.6197 |

Average query-only `Recall@100` follows the same pattern: hybrid retrieval recovers some counterfactual coverage, but pure dense models lose substantial recall.

| Retriever | Public | Entity-CF | Entity+Value-CF |
|---|---:|---:|---:|
| BM25 | 0.9633 | 0.7980 | 0.7958 |
| BGE | 0.9900 | 0.7668 | 0.7494 |
| E5 | 0.9867 | 0.7617 | 0.7607 |
| Contriever | 0.9750 | 0.7664 | 0.7472 |
| Hybrid BM25+E5 | 0.9867 | 0.8332 | 0.8464 |

Interpretation: the dense result is not BGE-specific. BGE, E5, and Contriever all collapse sharply under entity/value rewriting, especially on HotpotQA. However, the proposed FAWE-Q2D integration is mainly a BM25 and E5/hybrid improvement; it is not consistently beneficial for BGE or Contriever. The paper should therefore claim broad dense alias sensitivity, but avoid claiming FAWE is universally best for dense retrieval.

## Cross-Regime Leakage Isolation

This experiment compares public generations `g_pub` and counterfactual generations `g_cf`. Public generated-only retrieval loses substantially on counterfactual corpora. The average public-to-CF drop ranges from 0.235 to 0.350, depending on expansion type and regime.

| Expansion | Regime | g_pub public | g_pub CF | Drop | q+g_pub public | q+g_pub CF | Drop |
|---|---|---:|---:|---:|---:|---:|---:|
| Expected answer | Entity-CF | 0.7606 | 0.4596 | +0.3011 | 0.7800 | 0.5458 | +0.2342 |
| Expected answer | Entity+Value-CF | 0.7606 | 0.4105 | +0.3501 | 0.7800 | 0.5066 | +0.2735 |
| Query2doc | Entity-CF | 0.7764 | 0.5254 | +0.2510 | 0.7838 | 0.5514 | +0.2324 |
| Query2doc | Entity+Value-CF | 0.7764 | 0.4889 | +0.2875 | 0.7838 | 0.5192 | +0.2646 |
| HyDE | Entity-CF | 0.7749 | 0.5395 | +0.2354 | 0.7920 | 0.6027 | +0.1893 |
| HyDE | Entity+Value-CF | 0.7749 | 0.5115 | +0.2633 | 0.7920 | 0.5730 | +0.2190 |

For Query2doc, `g_cf` is usually worse than `g_pub` on the CF corpus, while anchoring and FAWE with `g_pub` are stronger. This supports public-prior brittleness and query anchoring, but not a simple claim that counterfactual generation alone recovers performance.

| Dataset | Regime | g_pub CF | g_cf CF | q_cf+g_pub | q_cf+g_cf | FAWE g_pub | FAWE g_cf | Anchor rescue | CF-gen gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NQ | Entity-CF | 0.5637 | 0.3864 | 0.5764 | 0.4057 | 0.5706 | 0.4412 | +0.0127 | -0.1773 |
| NQ | Entity+Value-CF | 0.5019 | 0.3652 | 0.5236 | 0.3845 | 0.5508 | 0.4275 | +0.0217 | -0.1367 |
| HotpotQA | Entity-CF | 0.3769 | 0.2670 | 0.4307 | 0.3373 | 0.5347 | 0.3997 | +0.0538 | -0.1099 |
| HotpotQA | Entity+Value-CF | 0.3351 | 0.2187 | 0.3857 | 0.2801 | 0.4965 | 0.3828 | +0.0506 | -0.1164 |
| SciFact | Entity-CF | 0.6357 | 0.6210 | 0.6471 | 0.6302 | 0.6632 | 0.6528 | +0.0114 | -0.0147 |
| SciFact | Entity+Value-CF | 0.6298 | 0.6485 | 0.6484 | 0.6483 | 0.6635 | 0.6560 | +0.0186 | +0.0187 |

Average cross-regime results:

| Expansion | Regime | g_pub CF | g_cf CF | q_cf+g_pub | q_cf+g_cf | FAWE g_pub | FAWE g_cf | Anchor rescue | CF-gen gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Expected answer | Entity-CF | 0.4596 | 0.4468 | 0.5458 | 0.5086 | 0.5153 | 0.4991 | +0.0863 | -0.0128 |
| Expected answer | Entity+Value-CF | 0.4105 | 0.4077 | 0.5066 | 0.4929 | 0.4978 | 0.4861 | +0.0961 | -0.0028 |
| Query2doc | Entity-CF | 0.5254 | 0.4248 | 0.5514 | 0.4577 | 0.5895 | 0.4979 | +0.0260 | -0.1006 |
| Query2doc | Entity+Value-CF | 0.4889 | 0.4108 | 0.5192 | 0.4376 | 0.5703 | 0.4888 | +0.0303 | -0.0781 |
| HyDE | Entity-CF | 0.5395 | 0.3549 | 0.6027 | 0.4358 | 0.6353 | 0.5073 | +0.0632 | -0.1846 |
| HyDE | Entity+Value-CF | 0.5115 | 0.3389 | 0.5730 | 0.4191 | 0.6182 | 0.5021 | +0.0614 | -0.1727 |

## Held-Out Beta Selection

Held-out beta selection supports the robustness concern. Public-only selection can choose larger beta values, while average/robust objectives select smaller beta values for HotpotQA and SciFact. NQ consistently selects 0.10.

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

The full 3k BM25 beta sweep, averaged across datasets, peaks at beta 0.50 in public retrieval and beta 0.10 in both counterfactual regimes.

| Regime | beta=0.05 | beta=0.10 | beta=0.25 | beta=0.50 | beta=0.75 | beta=1.00 | Best |
|---|---:|---:|---:|---:|---:|---:|---|
| Public | 0.7665 | 0.7818 | 0.7887 | 0.7894 | 0.7847 | 0.7838 | 0.50 |
| Entity-CF | 0.5082 | 0.5196 | 0.4979 | 0.4819 | 0.4705 | 0.4577 | 0.10 |
| Entity+Value-CF | 0.4982 | 0.5075 | 0.4888 | 0.4622 | 0.4446 | 0.4376 | 0.10 |

## Expansion Content Audit

The automatic audit labels most Query2doc, HyDE, and FAWE-Q2D expansions as answer-bearing because they contain candidate or unsupported injected entities. Exact-answer and alias-answer rates are zero in this audit because the BEIR-loaded records generally lack explicit answer-alias metadata; candidate and unsupported-injection rates are the more informative labels.

Average across datasets:

| Method | Answer-bearing | Candidate injection | Unsupported injection | Public gain | Entity excess drop | E+V excess drop |
|---|---:|---:|---:|---:|---:|---:|
| AnswerOnly | 0.79 | 0.79 | 0.65 | +0.0829 | +0.1036 | +0.1395 |
| HyDE | 1.00 | 1.00 | 1.00 | +0.0972 | +0.2097 | +0.2226 |
| Query2doc | 1.00 | 1.00 | 1.00 | +0.1061 | +0.1159 | +0.1328 |
| AnchoredAnswer | 0.79 | 0.79 | 0.65 | +0.1023 | +0.0613 | +0.0738 |
| FAWE-Q2D | 1.00 | 1.00 | 1.00 | +0.1110 | +0.0806 | +0.0865 |
| FAWE-Adapt | 1.00 | 1.00 | 1.00 | +0.0073 | -0.0101 | -0.0155 |

The audit supports the mechanism: answer-bearing/candidate-injecting expansions can produce public gains and larger counterfactual excess drops. AnchoredAnswer and FAWE-Q2D reduce excess drop relative to naive generated-only or pseudo-document methods. FAWE-Adapt is the most stable but gives little public gain, so it is better framed as a conservative deployment variant.

## Runtime

Average recorded runtime in minutes:

| Retriever | Public | Entity-CF | Entity+Value-CF |
|---|---:|---:|---:|
| BM25 | 4.7 | 56.9 | 89.1 |
| BGE | 1.1 | 1.3 | 1.3 |
| E5 | 1.2 | 1.3 | 1.4 |
| Contriever | 1.2 | 1.3 | 1.4 |
| Hybrid BM25+E5 | 1.4 | 1.6 | 1.7 |

The BM25 counterfactual runs include artifact construction and full-corpus counterfactual rewriting, which explains the much larger wall-clock time. The dense reruns reused counterfactual artifacts and generation caches, so their runtimes mostly reflect embedding/search work and missing retriever-specific generation cache fills.

## Figures Generated

The postprocess step generated:

- `outputs_pi/c3000/figures/bm25_utility_with_error_bars.pdf`
- `outputs_pi/c3000/figures/fawe_beta_sweep_with_error_bars.pdf`
- `outputs_pi/c3000/figures/cross_regime_query2doc_with_error_bars.pdf`

PNG versions are available in the same directory.

## Bottom-Line Findings

The strongest completed result remains the BM25 larger-corpus replication. At 3k corpus size, FAWE-Q2D is better than Query2doc in every counterfactual BM25 condition, with the largest gain on HotpotQA Entity+Value-CF (+0.1027). This supports the paper claim that FAWE-Q2D improves counterfactual utility over naive Query2doc concatenation for sparse retrieval.

The cross-regime experiment shows that public generations lose substantially on counterfactual corpora, confirming public-prior brittleness. Counterfactual generation alone does not reliably recover performance; anchoring and FAWE are the more reliable recovery mechanisms.

The dense retriever extension supports a broader alias-sensitivity claim. BGE, E5, and Contriever all show large public-to-counterfactual query-only drops, especially on HotpotQA. Hybrid BM25+E5 mitigates but does not eliminate the drop.

The dense extension also constrains the FAWE claim. FAWE-Q2D helps BM25, E5, and hybrid retrieval on counterfactual averages, but not BGE or Contriever. The paper should present FAWE as a strong sparse and partly hybrid/E5 method, not as a universal dense-retrieval solution.

Held-out beta selection strengthens the robustness story: public-oriented tuning tolerates larger expansion weights, while robust objectives prefer small beta values. This supports the design principle that generated text should be an auxiliary field, not a replacement for the original query.
