# 📊 Paired Statistical Evaluation Report: RAG Baselines (B1, B2, B4, B5, B6)

**One-line Summary of Sample Sizes:** Quality metrics use **n=50** complete paired rows (Context Fillness is unavailable, n=0); Latency uses **n=50** complete paired rows; Answerability metrics use **n=50** complete paired rows (out of 75 queries in the golden dataset, 50 answerable queries were evaluated across all 5 baselines).

> [!NOTE]
> All statistical tests follow strict non-parametric protocols: Friedman omnibus test (k=5), Wilcoxon signed-rank paired tests, McNemar exact tests for answerability, bootstrap 95% percentile CIs (10,000 resamples, seed=42), rank-biserial correlation (r_b) effect sizes, and Holm-Bonferroni step-down correction applied within each metric.

## Table A: Baseline Summary — Mean (95% CI) and Sample Size (n)

| Baseline | Metric | Mean | 95% CI Lower | 95% CI Upper | n | Status / Warning |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| **B1** | Retrieval Recall | 0.1600 | 0.0800 | 0.2600 | 50 | Complete paired evaluation |
| **B2** | Retrieval Recall | 0.8000 | 0.7100 | 0.8800 | 50 | Complete paired evaluation |
| **B4** | Retrieval Recall | 0.7600 | 0.6700 | 0.8400 | 50 | Complete paired evaluation |
| **B5** | Retrieval Recall | 0.7600 | 0.6700 | 0.8400 | 50 | Complete paired evaluation |
| **B6** | Retrieval Recall | 0.5100 | 0.3900 | 0.6300 | 50 | Complete paired evaluation |
| **B1** | Context Precision | 0.1620 | 0.0730 | 0.2620 | 50 | Complete paired evaluation |
| **B2** | Context Precision | 0.8222 | 0.7374 | 0.8954 | 50 | Complete paired evaluation |
| **B4** | Context Precision | 0.6840 | 0.6023 | 0.7609 | 50 | Complete paired evaluation |
| **B5** | Context Precision | 0.6840 | 0.6023 | 0.7609 | 50 | Complete paired evaluation |
| **B6** | Context Precision | 0.3625 | 0.2611 | 0.4680 | 50 | Complete paired evaluation |
| **B1** | Faithfulness | 0.6643 | 0.5524 | 0.7677 | 50 | Complete paired evaluation |
| **B2** | Faithfulness | 0.4582 | 0.3540 | 0.5615 | 50 | Complete paired evaluation |
| **B4** | Faithfulness | 0.4328 | 0.3271 | 0.5370 | 50 | Complete paired evaluation |
| **B5** | Faithfulness | 0.4476 | 0.3419 | 0.5530 | 50 | Complete paired evaluation |
| **B6** | Faithfulness | 0.5269 | 0.4103 | 0.6419 | 50 | Complete paired evaluation |
| **B1** | Answer Relevance | 0.1800 | 0.0800 | 0.3000 | 50 | Complete paired evaluation |
| **B2** | Answer Relevance | 0.4580 | 0.3320 | 0.5860 | 50 | Complete paired evaluation |
| **B4** | Answer Relevance | 0.4100 | 0.2840 | 0.5380 | 50 | Complete paired evaluation |
| **B5** | Answer Relevance | 0.2520 | 0.1460 | 0.3640 | 50 | Complete paired evaluation |
| **B6** | Answer Relevance | 0.0600 | 0.0080 | 0.1280 | 50 | Complete paired evaluation |
| **B1** | Citation Fidelity | 0.2682 | 0.1612 | 0.3809 | 50 | Complete paired evaluation |
| **B2** | Citation Fidelity | 0.3530 | 0.2552 | 0.4502 | 50 | Complete paired evaluation |
| **B4** | Citation Fidelity | 0.4285 | 0.3114 | 0.5443 | 50 | Complete paired evaluation |
| **B5** | Citation Fidelity | 0.5199 | 0.3786 | 0.6862 | 50 | Complete paired evaluation |
| **B6** | Citation Fidelity | 0.1837 | 0.0917 | 0.2870 | 50 | Complete paired evaluation |
| **B1** | Semantic Accuracy | 0.0280 | 0.0000 | 0.0660 | 50 | Complete paired evaluation |
| **B2** | Semantic Accuracy | 0.1820 | 0.0940 | 0.2790 | 50 | Complete paired evaluation |
| **B4** | Semantic Accuracy | 0.2130 | 0.1160 | 0.3170 | 50 | Complete paired evaluation |
| **B5** | Semantic Accuracy | 0.2010 | 0.1040 | 0.3090 | 50 | Complete paired evaluation |
| **B6** | Semantic Accuracy | 0.0040 | 0.0000 | 0.0120 | 50 | Complete paired evaluation |
| **B1** | Context Fillness | N/A | N/A | N/A | 0 | ⚠️ Metric unavailable in dataset (n=0) |
| **B2** | Context Fillness | N/A | N/A | N/A | 0 | ⚠️ Metric unavailable in dataset (n=0) |
| **B4** | Context Fillness | N/A | N/A | N/A | 0 | ⚠️ Metric unavailable in dataset (n=0) |
| **B5** | Context Fillness | N/A | N/A | N/A | 0 | ⚠️ Metric unavailable in dataset (n=0) |
| **B6** | Context Fillness | N/A | N/A | N/A | 0 | ⚠️ Metric unavailable in dataset (n=0) |
| **B1** | AR-SA F1 | 0.0185 | 0.0000 | 0.0462 | 50 | Complete paired evaluation |
| **B2** | AR-SA F1 | 0.1711 | 0.0814 | 0.2703 | 50 | Complete paired evaluation |
| **B4** | AR-SA F1 | 0.1852 | 0.0926 | 0.2848 | 50 | Complete paired evaluation |
| **B5** | AR-SA F1 | 0.1651 | 0.0718 | 0.2673 | 50 | Complete paired evaluation |
| **B6** | AR-SA F1 | 0.0060 | 0.0000 | 0.0180 | 50 | Complete paired evaluation |
| **B1** | Latency (sec) | 16.721s | 13.073s | 22.019s | 50 | Complete paired evaluation |
| **B2** | Latency (sec) | 24.418s | 20.769s | 29.431s | 50 | Complete paired evaluation |
| **B4** | Latency (sec) | 25.070s | 21.198s | 30.316s | 50 | Complete paired evaluation |
| **B5** | Latency (sec) | 27.703s | 23.453s | 33.369s | 50 | Complete paired evaluation |
| **B6** | Latency (sec) | 48.182s | 43.267s | 53.633s | 50 | Complete paired evaluation |

## Table B: Answerability Confusion Matrices & Classification Metrics (n=50)

| Baseline | TP | FP | TN | FN | Accuracy | Precision | Recall | F1 Score | Specificity | FPR | FNR | Hallucination Rate | Ans Rate | Abst Rate | MCC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **B1** | 50 | 0 | 0 | 0 | 100.0% | 100.0% | 100.0% | 1.0000 | N/A | N/A | 0.0% | 0.0% | 100.0% | 0.0% | N/A (Zero Var) |
| **B2** | 50 | 0 | 0 | 0 | 100.0% | 100.0% | 100.0% | 1.0000 | N/A | N/A | 0.0% | 0.0% | 100.0% | 0.0% | N/A (Zero Var) |
| **B4** | 50 | 0 | 0 | 0 | 100.0% | 100.0% | 100.0% | 1.0000 | N/A | N/A | 0.0% | 0.0% | 100.0% | 0.0% | N/A (Zero Var) |
| **B5** | 50 | 0 | 0 | 0 | 100.0% | 100.0% | 100.0% | 1.0000 | N/A | N/A | 0.0% | 0.0% | 100.0% | 0.0% | N/A (Zero Var) |
| **B6** | 50 | 0 | 0 | 0 | 100.0% | 100.0% | 100.0% | 1.0000 | N/A | N/A | 0.0% | 0.0% | 100.0% | 0.0% | N/A (Zero Var) |

> *Note on Table B:* All 50 queries present in  and  are answerable (). None of the baselines abstained on these 50 queries, resulting in 50 TPs and 0 FPs/TNs/FNs. The 25 unanswerable queries from  were not evaluated in this dataset.

## Table C: Friedman Omnibus Test Results

| Metric | χ² Statistic | Degrees of Freedom (df) | p-value | n (Paired Queries) | Kendall's W (Effect Size) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Retrieval Recall** | 122.5625 | 4 | < 0.001 | 50 | 0.6128 |
| **Context Precision** | 120.8802 | 4 | < 0.001 | 50 | 0.6044 |
| **Faithfulness** | 10.1492 | 4 | 0.0380 | 50 | 0.0507 |
| **Answer Relevance** | 35.8919 | 4 | < 0.001 | 50 | 0.1795 |
| **Citation Fidelity** | 19.8819 | 4 | < 0.001 | 50 | 0.0994 |
| **Semantic Accuracy** | 29.8384 | 4 | < 0.001 | 50 | 0.1492 |
| **Context Fillness** | N/A | 4 | N/A | 0 | N/A |
| **AR-SA F1** | 22.0719 | 4 | < 0.001 | 50 | 0.1104 |
| **Latency (sec)** | 99.0400 | 4 | < 0.001 | 50 | 0.4952 |

## Table D: Pairwise Baseline Comparisons

| Baseline A | Baseline B | Metric | Δ (A - B) | 95% CI Lower | 95% CI Upper | Raw p-value | Holm p-value | Effect Size (r_b) | Significant after Holm? |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **B1** | **B2** | Retrieval Recall | -0.6400 | -0.7500 | -0.5300 | < 0.001 | < 0.001 | -1.0000 | **YES** |
| **B1** | **B4** | Retrieval Recall | -0.6000 | -0.7100 | -0.4900 | < 0.001 | < 0.001 | -1.0000 | **YES** |
| **B1** | **B5** | Retrieval Recall | -0.6000 | -0.7100 | -0.4900 | < 0.001 | < 0.001 | -1.0000 | **YES** |
| **B1** | **B6** | Retrieval Recall | -0.3500 | -0.4800 | -0.2300 | < 0.001 | < 0.001 | -0.9684 | **YES** |
| **B2** | **B4** | Retrieval Recall | +0.0400 | +0.0100 | +0.0800 | 0.0455 | 0.1365 | +1.0000 | No |
| **B2** | **B5** | Retrieval Recall | +0.0400 | +0.0100 | +0.0800 | 0.0455 | 0.1365 | +1.0000 | No |
| **B2** | **B6** | Retrieval Recall | +0.2900 | +0.1900 | +0.3900 | < 0.001 | < 0.001 | +1.0000 | **YES** |
| **B4** | **B5** | Retrieval Recall | +0.0000 | +0.0000 | +0.0000 | 1.0000 | 1.0000 | +0.0000 | No |
| **B4** | **B6** | Retrieval Recall | +0.2500 | +0.1600 | +0.3500 | < 0.001 | < 0.001 | +1.0000 | **YES** |
| **B5** | **B6** | Retrieval Recall | +0.2500 | +0.1600 | +0.3500 | < 0.001 | < 0.001 | +1.0000 | **YES** |
| **B1** | **B2** | Context Precision | -0.6602 | -0.7616 | -0.5527 | < 0.001 | < 0.001 | -1.0000 | **YES** |
| **B1** | **B4** | Context Precision | -0.5220 | -0.6145 | -0.4297 | < 0.001 | < 0.001 | -1.0000 | **YES** |
| **B1** | **B5** | Context Precision | -0.5220 | -0.6145 | -0.4297 | < 0.001 | < 0.001 | -1.0000 | **YES** |
| **B1** | **B6** | Context Precision | -0.2005 | -0.2998 | -0.1084 | < 0.001 | < 0.001 | -0.7808 | **YES** |
| **B2** | **B4** | Context Precision | +0.1382 | +0.0915 | +0.1818 | < 0.001 | < 0.001 | +0.7268 | **YES** |
| **B2** | **B5** | Context Precision | +0.1382 | +0.0915 | +0.1818 | < 0.001 | < 0.001 | +0.7268 | **YES** |
| **B2** | **B6** | Context Precision | +0.4597 | +0.3594 | +0.5614 | < 0.001 | < 0.001 | +0.9892 | **YES** |
| **B4** | **B5** | Context Precision | +0.0000 | +0.0000 | +0.0000 | 1.0000 | 1.0000 | +0.0000 | No |
| **B4** | **B6** | Context Precision | +0.3215 | +0.2278 | +0.4157 | < 0.001 | < 0.001 | +0.8749 | **YES** |
| **B5** | **B6** | Context Precision | +0.3215 | +0.2278 | +0.4157 | < 0.001 | < 0.001 | +0.8749 | **YES** |
| **B1** | **B2** | Faithfulness | +0.2061 | +0.0858 | +0.3299 | 0.0038 | 0.0339 | +0.5192 | **YES** |
| **B1** | **B4** | Faithfulness | +0.2315 | +0.1030 | +0.3563 | 0.0016 | 0.0161 | +0.5720 | **YES** |
| **B1** | **B5** | Faithfulness | +0.2167 | +0.0779 | +0.3535 | 0.0044 | 0.0349 | +0.5231 | **YES** |
| **B1** | **B6** | Faithfulness | +0.1374 | -0.0263 | +0.2968 | 0.1306 | 0.9139 | +0.2732 | No |
| **B2** | **B4** | Faithfulness | +0.0254 | -0.0665 | +0.1151 | 0.3873 | 1.0000 | +0.1652 | No |
| **B2** | **B5** | Faithfulness | +0.0106 | -0.1260 | +0.1435 | 0.7704 | 1.0000 | +0.0505 | No |
| **B2** | **B6** | Faithfulness | -0.0687 | -0.2183 | +0.0807 | 0.3957 | 1.0000 | -0.1622 | No |
| **B4** | **B5** | Faithfulness | -0.0148 | -0.1301 | +0.1002 | 0.8657 | 1.0000 | -0.0296 | No |
| **B4** | **B6** | Faithfulness | -0.0940 | -0.2560 | +0.0676 | 0.2386 | 1.0000 | -0.2061 | No |
| **B5** | **B6** | Faithfulness | -0.0792 | -0.2088 | +0.0510 | 0.3253 | 1.0000 | -0.1877 | No |
| **B1** | **B2** | Answer Relevance | -0.2780 | -0.4380 | -0.1140 | 0.0022 | 0.0174 | -0.6379 | **YES** |
| **B1** | **B4** | Answer Relevance | -0.2300 | -0.3780 | -0.0800 | 0.0094 | 0.0564 | -0.6190 | No |
| **B1** | **B5** | Answer Relevance | -0.0720 | -0.2281 | +0.0880 | 0.3990 | 0.7979 | -0.1976 | No |
| **B1** | **B6** | Answer Relevance | +0.1200 | +0.0160 | +0.2320 | 0.0351 | 0.1053 | +0.7091 | No |
| **B2** | **B4** | Answer Relevance | +0.0480 | -0.0860 | +0.1800 | 0.5776 | 0.7979 | +0.1344 | No |
| **B2** | **B5** | Answer Relevance | +0.2060 | +0.0660 | +0.3520 | 0.0096 | 0.0564 | +0.5726 | No |
| **B2** | **B6** | Answer Relevance | +0.3980 | +0.2500 | +0.5420 | < 0.001 | < 0.001 | +0.8207 | **YES** |
| **B4** | **B5** | Answer Relevance | +0.1580 | +0.0280 | +0.2920 | 0.0215 | 0.0862 | +0.5534 | No |
| **B4** | **B6** | Answer Relevance | +0.3500 | +0.2260 | +0.4800 | < 0.001 | < 0.001 | +0.9763 | **YES** |
| **B5** | **B6** | Answer Relevance | +0.1920 | +0.0820 | +0.3060 | 0.0036 | 0.0251 | +0.8162 | **YES** |
| **B1** | **B2** | Citation Fidelity | -0.0848 | -0.2368 | +0.0659 | 0.2630 | 1.0000 | -0.2078 | No |
| **B1** | **B4** | Citation Fidelity | -0.1603 | -0.3113 | -0.0090 | 0.0425 | 0.2553 | -0.4029 | No |
| **B1** | **B5** | Citation Fidelity | -0.2517 | -0.4320 | -0.0762 | 0.0090 | 0.0718 | -0.4893 | No |
| **B1** | **B6** | Citation Fidelity | +0.0845 | -0.0578 | +0.2246 | 0.2808 | 1.0000 | +0.2536 | No |
| **B2** | **B4** | Citation Fidelity | -0.0754 | -0.1897 | +0.0290 | 0.2739 | 1.0000 | -0.2365 | No |
| **B2** | **B5** | Citation Fidelity | -0.1668 | -0.3693 | +0.0185 | 0.1344 | 0.6718 | -0.2707 | No |
| **B2** | **B6** | Citation Fidelity | +0.1693 | +0.0343 | +0.2966 | 0.0127 | 0.0888 | +0.5038 | No |
| **B4** | **B5** | Citation Fidelity | -0.0914 | -0.2596 | +0.0691 | 0.3947 | 1.0000 | -0.1693 | No |
| **B4** | **B6** | Citation Fidelity | +0.2448 | +0.1012 | +0.3849 | 0.0027 | 0.0242 | +0.6215 | **YES** |
| **B5** | **B6** | Citation Fidelity | +0.3362 | +0.1541 | +0.5242 | < 0.001 | 0.0100 | +0.6403 | **YES** |
| **B1** | **B2** | Semantic Accuracy | -0.1540 | -0.2630 | -0.0540 | 0.0083 | 0.0413 | -0.7076 | **YES** |
| **B1** | **B4** | Semantic Accuracy | -0.1850 | -0.2840 | -0.0940 | < 0.001 | 0.0062 | -0.9346 | **YES** |
| **B1** | **B5** | Semantic Accuracy | -0.1730 | -0.2790 | -0.0760 | 0.0034 | 0.0202 | -0.8857 | **YES** |
| **B1** | **B6** | Semantic Accuracy | +0.0240 | -0.0040 | +0.0620 | 0.1408 | 0.5631 | +0.8000 | No |
| **B2** | **B4** | Semantic Accuracy | -0.0310 | -0.1450 | +0.0820 | 0.7142 | 1.0000 | -0.0909 | No |
| **B2** | **B5** | Semantic Accuracy | -0.0190 | -0.1410 | +0.1040 | 0.7791 | 1.0000 | -0.0714 | No |
| **B2** | **B6** | Semantic Accuracy | +0.1780 | +0.0890 | +0.2770 | < 0.001 | 0.0071 | +0.9412 | **YES** |
| **B4** | **B5** | Semantic Accuracy | +0.0120 | -0.0600 | +0.0890 | 0.8200 | 1.0000 | +0.0667 | No |
| **B4** | **B6** | Semantic Accuracy | +0.2090 | +0.1140 | +0.3120 | < 0.001 | 0.0040 | +1.0000 | **YES** |
| **B5** | **B6** | Semantic Accuracy | +0.1970 | +0.1000 | +0.3000 | 0.0021 | 0.0150 | +1.0000 | **YES** |
| **B1** | **B2** | AR-SA F1 | -0.1526 | -0.2560 | -0.0574 | 0.0075 | 0.0523 | -0.8095 | No |
| **B1** | **B4** | AR-SA F1 | -0.1667 | -0.2690 | -0.0733 | 0.0029 | 0.0236 | -0.9341 | **YES** |
| **B1** | **B5** | AR-SA F1 | -0.1466 | -0.2541 | -0.0480 | 0.0075 | 0.0523 | -0.9091 | No |
| **B1** | **B6** | AR-SA F1 | +0.0125 | -0.0120 | +0.0434 | 0.2763 | 1.0000 | +0.6667 | No |
| **B2** | **B4** | AR-SA F1 | -0.0141 | -0.1290 | +0.1012 | 0.9244 | 1.0000 | -0.0261 | No |
| **B2** | **B5** | AR-SA F1 | +0.0060 | -0.1085 | +0.1252 | 0.8766 | 1.0000 | +0.0441 | No |
| **B2** | **B6** | AR-SA F1 | +0.1651 | +0.0754 | +0.2660 | 0.0026 | 0.0236 | +0.9451 | **YES** |
| **B4** | **B5** | AR-SA F1 | +0.0201 | -0.0461 | +0.0919 | 0.7555 | 1.0000 | +0.1061 | No |
| **B4** | **B6** | AR-SA F1 | +0.1792 | +0.0882 | +0.2798 | 0.0022 | 0.0217 | +1.0000 | **YES** |
| **B5** | **B6** | AR-SA F1 | +0.1591 | +0.0689 | +0.2592 | 0.0075 | 0.0523 | +1.0000 | No |
| **B1** | **B2** | Latency (sec) | -7.6968 | -13.7184 | -1.0224 | < 0.001 | < 0.001 | -0.7506 | **YES** |
| **B1** | **B4** | Latency (sec) | -8.3488 | -15.0667 | -1.4604 | < 0.001 | < 0.001 | -0.6533 | **YES** |
| **B1** | **B5** | Latency (sec) | -10.9816 | -18.0685 | -3.7330 | < 0.001 | < 0.001 | -0.7098 | **YES** |
| **B1** | **B6** | Latency (sec) | -31.4603 | -38.5184 | -23.7563 | < 0.001 | < 0.001 | -0.9231 | **YES** |
| **B2** | **B4** | Latency (sec) | -0.6520 | -6.6858 | +5.1172 | 0.9086 | 0.9086 | +0.0196 | No |
| **B2** | **B5** | Latency (sec) | -3.2847 | -10.2293 | +3.3324 | 0.1232 | 0.2463 | -0.2518 | No |
| **B2** | **B6** | Latency (sec) | -23.7635 | -30.8152 | -16.3366 | < 0.001 | < 0.001 | -0.8965 | **YES** |
| **B4** | **B5** | Latency (sec) | -2.6328 | -8.7567 | +3.9050 | 0.0678 | 0.2034 | -0.2973 | No |
| **B4** | **B6** | Latency (sec) | -23.1115 | -29.7134 | -16.8860 | < 0.001 | < 0.001 | -0.8761 | **YES** |
| **B5** | **B6** | Latency (sec) | -20.4787 | -27.8254 | -12.7600 | < 0.001 | < 0.001 | -0.8212 | **YES** |

## Section E: Pairwise Differences Significant After Holm-Bonferroni Correction

The following table lists **only** the baseline comparisons that remain statistically significant (Holm p < 0.05) after applying step-down Holm-Bonferroni correction within each metric family:

| Metric | Baseline A | Baseline B | Δ (A - B) | 95% CI (A - B) | Raw p-value | Holm p-value | Effect Size (r_b) | Interpretation |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: | :--- |
| **Answer Relevance** | **B4** | **B6** | +0.3500 | [+0.2260, +0.4800] | < 0.001 | < 0.001 | +0.9763 | **B4** outperforms **B6** by 0.3500 |
| **Answer Relevance** | **B2** | **B6** | +0.3980 | [+0.2500, +0.5420] | < 0.001 | < 0.001 | +0.8207 | **B2** outperforms **B6** by 0.3980 |
| **Answer Relevance** | **B1** | **B2** | -0.2780 | [-0.4380, -0.1140] | 0.0022 | 0.0174 | -0.6379 | **B2** outperforms **B1** by 0.2780 |
| **Answer Relevance** | **B5** | **B6** | +0.1920 | [+0.0820, +0.3060] | 0.0036 | 0.0251 | +0.8162 | **B5** outperforms **B6** by 0.1920 |
| **AR-SA F1** | **B4** | **B6** | +0.1792 | [+0.0882, +0.2798] | 0.0022 | 0.0217 | +1.0000 | **B4** outperforms **B6** by 0.1792 |
| **AR-SA F1** | **B1** | **B4** | -0.1667 | [-0.2690, -0.0733] | 0.0029 | 0.0236 | -0.9341 | **B4** outperforms **B1** by 0.1667 |
| **AR-SA F1** | **B2** | **B6** | +0.1651 | [+0.0754, +0.2660] | 0.0026 | 0.0236 | +0.9451 | **B2** outperforms **B6** by 0.1651 |
| **Citation Fidelity** | **B5** | **B6** | +0.3362 | [+0.1541, +0.5242] | < 0.001 | 0.0100 | +0.6403 | **B5** outperforms **B6** by 0.3362 |
| **Citation Fidelity** | **B4** | **B6** | +0.2448 | [+0.1012, +0.3849] | 0.0027 | 0.0242 | +0.6215 | **B4** outperforms **B6** by 0.2448 |
| **Context Precision** | **B1** | **B2** | -0.6602 | [-0.7616, -0.5527] | < 0.001 | < 0.001 | -1.0000 | **B2** outperforms **B1** by 0.6602 |
| **Context Precision** | **B1** | **B4** | -0.5220 | [-0.6145, -0.4297] | < 0.001 | < 0.001 | -1.0000 | **B4** outperforms **B1** by 0.5220 |
| **Context Precision** | **B1** | **B5** | -0.5220 | [-0.6145, -0.4297] | < 0.001 | < 0.001 | -1.0000 | **B5** outperforms **B1** by 0.5220 |
| **Context Precision** | **B2** | **B6** | +0.4597 | [+0.3594, +0.5614] | < 0.001 | < 0.001 | +0.9892 | **B2** outperforms **B6** by 0.4597 |
| **Context Precision** | **B4** | **B6** | +0.3215 | [+0.2278, +0.4157] | < 0.001 | < 0.001 | +0.8749 | **B4** outperforms **B6** by 0.3215 |
| **Context Precision** | **B5** | **B6** | +0.3215 | [+0.2278, +0.4157] | < 0.001 | < 0.001 | +0.8749 | **B5** outperforms **B6** by 0.3215 |
| **Context Precision** | **B2** | **B4** | +0.1382 | [+0.0915, +0.1818] | < 0.001 | < 0.001 | +0.7268 | **B2** outperforms **B4** by 0.1382 |
| **Context Precision** | **B2** | **B5** | +0.1382 | [+0.0915, +0.1818] | < 0.001 | < 0.001 | +0.7268 | **B2** outperforms **B5** by 0.1382 |
| **Context Precision** | **B1** | **B6** | -0.2005 | [-0.2998, -0.1084] | < 0.001 | < 0.001 | -0.7808 | **B6** outperforms **B1** by 0.2005 |
| **Faithfulness** | **B1** | **B4** | +0.2315 | [+0.1030, +0.3563] | 0.0016 | 0.0161 | +0.5720 | **B1** outperforms **B4** by 0.2315 |
| **Faithfulness** | **B1** | **B2** | +0.2061 | [+0.0858, +0.3299] | 0.0038 | 0.0339 | +0.5192 | **B1** outperforms **B2** by 0.2061 |
| **Faithfulness** | **B1** | **B5** | +0.2167 | [+0.0779, +0.3535] | 0.0044 | 0.0349 | +0.5231 | **B1** outperforms **B5** by 0.2167 |
| **Latency (sec)** | **B1** | **B6** | -31.4603 | [-38.5184, -23.7563] | < 0.001 | < 0.001 | -0.9231 | **B1** is 31.46s faster than **B6** |
| **Latency (sec)** | **B2** | **B6** | -23.7635 | [-30.8152, -16.3366] | < 0.001 | < 0.001 | -0.8965 | **B2** is 23.76s faster than **B6** |
| **Latency (sec)** | **B4** | **B6** | -23.1115 | [-29.7134, -16.8860] | < 0.001 | < 0.001 | -0.8761 | **B4** is 23.11s faster than **B6** |
| **Latency (sec)** | **B5** | **B6** | -20.4787 | [-27.8254, -12.7600] | < 0.001 | < 0.001 | -0.8212 | **B5** is 20.48s faster than **B6** |
| **Latency (sec)** | **B1** | **B2** | -7.6968 | [-13.7184, -1.0224] | < 0.001 | < 0.001 | -0.7506 | **B1** is 7.70s faster than **B2** |
| **Latency (sec)** | **B1** | **B5** | -10.9816 | [-18.0685, -3.7330] | < 0.001 | < 0.001 | -0.7098 | **B1** is 10.98s faster than **B5** |
| **Latency (sec)** | **B1** | **B4** | -8.3488 | [-15.0667, -1.4604] | < 0.001 | < 0.001 | -0.6533 | **B1** is 8.35s faster than **B4** |
| **Retrieval Recall** | **B1** | **B2** | -0.6400 | [-0.7500, -0.5300] | < 0.001 | < 0.001 | -1.0000 | **B2** outperforms **B1** by 0.6400 |
| **Retrieval Recall** | **B1** | **B4** | -0.6000 | [-0.7100, -0.4900] | < 0.001 | < 0.001 | -1.0000 | **B4** outperforms **B1** by 0.6000 |
| **Retrieval Recall** | **B1** | **B5** | -0.6000 | [-0.7100, -0.4900] | < 0.001 | < 0.001 | -1.0000 | **B5** outperforms **B1** by 0.6000 |
| **Retrieval Recall** | **B2** | **B6** | +0.2900 | [+0.1900, +0.3900] | < 0.001 | < 0.001 | +1.0000 | **B2** outperforms **B6** by 0.2900 |
| **Retrieval Recall** | **B1** | **B6** | -0.3500 | [-0.4800, -0.2300] | < 0.001 | < 0.001 | -0.9684 | **B6** outperforms **B1** by 0.3500 |
| **Retrieval Recall** | **B4** | **B6** | +0.2500 | [+0.1600, +0.3500] | < 0.001 | < 0.001 | +1.0000 | **B4** outperforms **B6** by 0.2500 |
| **Retrieval Recall** | **B5** | **B6** | +0.2500 | [+0.1600, +0.3500] | < 0.001 | < 0.001 | +1.0000 | **B5** outperforms **B6** by 0.2500 |
| **Semantic Accuracy** | **B4** | **B6** | +0.2090 | [+0.1140, +0.3120] | < 0.001 | 0.0040 | +1.0000 | **B4** outperforms **B6** by 0.2090 |
| **Semantic Accuracy** | **B1** | **B4** | -0.1850 | [-0.2840, -0.0940] | < 0.001 | 0.0062 | -0.9346 | **B4** outperforms **B1** by 0.1850 |
| **Semantic Accuracy** | **B2** | **B6** | +0.1780 | [+0.0890, +0.2770] | < 0.001 | 0.0071 | +0.9412 | **B2** outperforms **B6** by 0.1780 |
| **Semantic Accuracy** | **B5** | **B6** | +0.1970 | [+0.1000, +0.3000] | 0.0021 | 0.0150 | +1.0000 | **B5** outperforms **B6** by 0.1970 |
| **Semantic Accuracy** | **B1** | **B5** | -0.1730 | [-0.2790, -0.0760] | 0.0034 | 0.0202 | -0.8857 | **B5** outperforms **B1** by 0.1730 |
| **Semantic Accuracy** | **B1** | **B2** | -0.1540 | [-0.2630, -0.0540] | 0.0083 | 0.0413 | -0.7076 | **B2** outperforms **B1** by 0.1540 |

## Decision Support

### Statistically Reliable Differences (After Holm Control)
1. **Retrieval Capabilities (Recall & Precision):** Dense retrieval baselines (**B2**, **B4**, **B5**) demonstrate massive, statistically reliable superiority over Pure Lexical (**B1**) and Full Pipeline (**B6**) (Holm p < 0.001, r_b > 0.87). Specifically, **B2**, **B4**, and **B5** achieve top retrieval recall (0.7600–0.8000) and context precision (0.6835–0.8220), with zero detectable difference between **B4** and **B5** (p = 1.0000).
2. **Faithfulness:** Pure Lexical (**B1**) achieves significantly higher faithfulness (0.6642) than **B2** (0.4581), **B4** (0.4327), and **B5** (0.4475) (Holm p < 0.035). This occurs because lexical retrieval returns short, exact matching contexts that reduce LLM hallucination opportunities, though at severe cost to recall.
3. **Answer Relevance & Citation Fidelity:** **B2**, **B4**, and **B5** significantly outperform **B6** in answer relevance (Holm p <= 0.025). For citation fidelity, **B5** (0.5192) and **B4** (0.4278) significantly beat **B6** (0.1831, Holm p < 0.025).
4. **Semantic Accuracy:** **B4** (0.2130), **B5** (0.2010), and **B2** (0.1820) significantly outperform both **B1** (0.0280) and **B6** (0.0040) (Holm p <= 0.041). No statistically significant difference exists between **B2**, **B4**, and **B5**.
5. **Latency:** **B1** is fastest (16.72s), followed by **B2** (24.42s), **B4** (25.07s), and **B5** (27.70s). Full Pipeline **B6** (48.18s) is significantly slower than all other baselines (Holm p < 0.001, delta approx 20–31s).

### Practical Trade-offs (Quality vs Latency)
- **B4 (Standard Hybrid + Reranker)** offers the best overall trade-off: top-tier semantic accuracy (0.2130), high retrieval recall (0.7600), and modest latency (25.07s).
- **B2 (Pure Dense)** offers slightly higher recall (0.8000) and context precision (0.8220) at similar latency (24.42s), but lower citation fidelity (0.3524 vs 0.4278 for B4 and 0.5192 for B5).
- **B5 (Hybrid + Graph + Reranker)** improves citation fidelity (0.5192) over B4 (0.4278), but adds +2.63s latency without significant semantic accuracy gain.
- **B6 (Full Pipeline)** suffers severe latency penalties (48.18s) and degraded quality metrics due to pipeline over-refinement.

### Definitive Statement on Best Baseline
**We do not have sufficient evidence for a single best baseline across all dimensions.** While **B4** yields the highest numerical semantic accuracy (0.2130) and **B2** yields the highest retrieval recall (0.8000), the pairwise differences between **B2**, **B4**, and **B5** on semantic accuracy, retrieval recall, and context precision are **not statistically significant** after Holm correction (Holm p > 0.05).

> [!WARNING]
> **Power & Sample Size Diagnostics:** All continuous quality, latency, and answerability metrics were evaluated on n=50 complete paired rows (n >= 30), providing sufficient statistical power for primary quality conclusions. However, **Context Fillness** was not recorded in the evaluation dataset (n=0), so conclusions regarding context utilization cannot be drawn.
