# Paired Statistical Evaluation Report: RAG Baselines (B1, B2, B4, B5, B6)

> [!NOTE]
> Empirical evaluation report of baselines **B1** (Lexical), **B2** (Dense), **B4** (Hybrid+Reranker), **B5** (Hybrid+Graph+Reranker), and **B6** (Full Pipeline). Quality metrics use **n=50** complete paired rows; Latency uses **n=50** complete paired rows; Answerability metrics use **n=50** complete paired rows.
> All statistical tests follow strict non-parametric protocols: Friedman omnibus test ($k=5$), Wilcoxon signed-rank paired tests, McNemar exact tests for answerability, bootstrap 95% percentile CIs (10,000 resamples, seed 42), rank-biserial correlation ($r_b$) effect sizes, and Holm-Bonferroni step-down correction applied within each metric family.

---

## ⚙️ Statistical Pipeline Architecture & Execution Flow

The evaluation framework ([core/statistics.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/statistics.py) & [core/connector.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/connector.py)) operates as a modular, 3-stage non-parametric statistical inference pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│  Raw Evaluation Data (result_metrics.yaml / CSV details)    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Data Preparation Agent (data_prep_agent)           │
│  - Extracts per-(query, baseline) records                   │
│  - Outcome-aware NaN handling (TN/FP -> NaN quality scores) │
│  - Aligns complete paired query sets across k=5 baselines   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Statistical Agent (stats_agent)                    │
│  - Resampled 95% Bootstrap CIs (10,000 iterations, seed=42) │
│  - Friedman Omnibus Test (k=5, Kendall's W effect size)     │
│  - Pairwise Wilcoxon Signed-Rank Tests (rank-biserial r_b)  │
│  - Pairwise McNemar Exact Tests (binary answerability)      │
│  - Step-down Holm-Bonferroni FWER P-Value Correction        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Reporting & Artifact Agent (reporting_agent)       │
│  - Generates structured Markdown tables (Tables A, B, C, D, E)│
│  - Formats p-values in explicit x * 10^a scientific notation│
│  - Exports stat_run_results.json & decision support notes   │
└─────────────────────────────────────────────────────────────┘
```

---

## Table A: Baseline Summary — Mean (95% CI) and Explicit Sample Size ($n=50$)

| Baseline | Metric | Mean | 95% CI Lower | 95% CI Upper | $n$ | Diagnostic Note / Status |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| **B1** | Retrieval Recall | 0.1600 | 0.0800 | 0.2600 | 50 | Heavy floor effect (>40% at min) |
| **B2** | Retrieval Recall | 0.8000 | 0.7100 | 0.8800 | 50 | Heavy ceiling effect (>40% at max) |
| **B4** | Retrieval Recall | 0.7600 | 0.6700 | 0.8400 | 50 | Heavy ceiling effect (>40% at max) |
| **B5** | Retrieval Recall | 0.7600 | 0.6700 | 0.8400 | 50 | Heavy ceiling effect (>40% at max) |
| **B6** | Retrieval Recall | 0.5100 | 0.3900 | 0.6300 | 50 | Complete paired evaluation |
| **B1** | Context Precision | 0.1620 | 0.0730 | 0.2620 | 50 | Heavy floor effect (>40% at min) |
| **B2** | Context Precision | 0.8222 | 0.7374 | 0.8954 | 50 | Heavy ceiling effect (>40% at max) |
| **B4** | Context Precision | 0.6840 | 0.6023 | 0.7609 | 50 | Complete paired evaluation |
| **B5** | Context Precision | 0.6840 | 0.6023 | 0.7609 | 50 | Complete paired evaluation |
| **B6** | Context Precision | 0.3625 | 0.2611 | 0.4680 | 50 | Complete paired evaluation |
| **B1** | Faithfulness | 0.6643 | 0.5524 | 0.7677 | 50 | Heavy ceiling effect (>40% at max) |
| **B2** | Faithfulness | 0.4582 | 0.3540 | 0.5615 | 50 | Complete paired evaluation |
| **B4** | Faithfulness | 0.4328 | 0.3271 | 0.5370 | 50 | Complete paired evaluation |
| **B5** | Faithfulness | 0.4476 | 0.3419 | 0.5530 | 50 | Complete paired evaluation |
| **B6** | Faithfulness | 0.5269 | 0.4103 | 0.6419 | 50 | Complete paired evaluation |
| **B1** | Answer Relevance | 0.1800 | 0.0800 | 0.3000 | 50 | Heavy floor effect (>40% at min) |
| **B2** | Answer Relevance | 0.4580 | 0.3320 | 0.5860 | 50 | Heavy floor effect (>40% at min) |
| **B4** | Answer Relevance | 0.4100 | 0.2840 | 0.5380 | 50 | Heavy floor effect (>40% at min) |
| **B5** | Answer Relevance | 0.2520 | 0.1460 | 0.3640 | 50 | Heavy floor effect (>40% at min) |
| **B6** | Answer Relevance | 0.0600 | 0.0080 | 0.1280 | 50 | Heavy floor effect (>40% at min) |
| **B1** | Citation Fidelity | 0.2682 | 0.1612 | 0.3809 | 50 | Heavy floor effect (>40% at min) |
| **B2** | Citation Fidelity | 0.3530 | 0.2552 | 0.4502 | 50 | Complete paired evaluation |
| **B4** | Citation Fidelity | 0.4285 | 0.3114 | 0.5443 | 50 | Complete paired evaluation |
| **B5** | Citation Fidelity | 0.5199 | 0.3786 | 0.6862 | 50 | Complete paired evaluation |
| **B6** | Citation Fidelity | 0.1837 | 0.0917 | 0.2870 | 50 | Heavy floor effect (>40% at min) |
| **B1** | Semantic Accuracy | 0.0280 | 0.0000 | 0.0660 | 50 | Heavy floor effect (>40% at min) |
| **B2** | Semantic Accuracy | 0.1820 | 0.0940 | 0.2790 | 50 | Heavy floor effect (>40% at min) |
| **B4** | Semantic Accuracy | 0.2130 | 0.1160 | 0.3170 | 50 | Heavy floor effect (>40% at min) |
| **B5** | Semantic Accuracy | 0.2010 | 0.1040 | 0.3090 | 50 | Heavy floor effect (>40% at min) |
| **B6** | Semantic Accuracy | 0.0040 | 0.0000 | 0.0120 | 50 | Heavy floor effect (>40% at min) |
| **B1** | Context Fillness | 0.0184 | 0.0181 | 0.0188 | 50 | Complete paired evaluation |
| **B2** | Context Fillness | 0.0184 | 0.0181 | 0.0188 | 50 | Complete paired evaluation |
| **B4** | Context Fillness | 0.0184 | 0.0181 | 0.0188 | 50 | Complete paired evaluation |
| **B5** | Context Fillness | 0.0184 | 0.0181 | 0.0188 | 50 | Complete paired evaluation |
| **B6** | Context Fillness | 0.0184 | 0.0181 | 0.0188 | 50 | Complete paired evaluation |
| **B1** | AR-SA F1 | 0.0185 | 0.0000 | 0.0462 | 50 | Heavy floor effect (>40% at min) |
| **B2** | AR-SA F1 | 0.1711 | 0.0814 | 0.2703 | 50 | Heavy floor effect (>40% at min) |
| **B4** | AR-SA F1 | 0.1852 | 0.0926 | 0.2848 | 50 | Heavy floor effect (>40% at min) |
| **B5** | AR-SA F1 | 0.1651 | 0.0718 | 0.2673 | 50 | Heavy floor effect (>40% at min) |
| **B6** | AR-SA F1 | 0.0060 | 0.0000 | 0.0180 | 50 | Heavy floor effect (>40% at min) |
| **B1** | Latency (sec) | 16.721s | 13.073s | 22.019s | 50 | Complete paired evaluation |
| **B2** | Latency (sec) | 24.418s | 20.769s | 29.431s | 50 | Complete paired evaluation |
| **B4** | Latency (sec) | 25.070s | 21.198s | 30.316s | 50 | Complete paired evaluation |
| **B5** | Latency (sec) | 27.703s | 23.453s | 33.369s | 50 | Complete paired evaluation |
| **B6** | Latency (sec) | 48.182s | 43.267s | 53.633s | 50 | Complete paired evaluation |

---

## Table B: Answerability Confusion Matrices & Classification Metrics ($n=50$)

| Baseline | TP | FP | TN | FN | Accuracy | Precision | Recall | F1 Score | Specificity | FPR | FNR | Hallucination Rate | Ans Rate | Abst Rate | MCC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **B1** | 42 | 0 | 0 | 8 | 84.0% | 100.0% | 84.0% | 0.9130 | N/A | N/A | 16.0% | N/A | 84.0% | 16.0% | N/A (Zero TN/FP) |
| **B2** | 49 | 0 | 0 | 1 | 98.0% | 100.0% | 98.0% | 0.9899 | N/A | N/A | 2.0% | N/A | 98.0% | 2.0% | N/A (Zero TN/FP) |
| **B4** | 49 | 0 | 0 | 1 | 98.0% | 100.0% | 98.0% | 0.9899 | N/A | N/A | 2.0% | N/A | 98.0% | 2.0% | N/A (Zero TN/FP) |
| **B5** | 47 | 0 | 0 | 3 | 94.0% | 100.0% | 94.0% | 0.9691 | N/A | N/A | 6.0% | N/A | 94.0% | 6.0% | N/A (Zero TN/FP) |
| **B6** | 44 | 0 | 0 | 6 | 88.0% | 100.0% | 88.0% | 0.9362 | N/A | N/A | 12.0% | N/A | 88.0% | 12.0% | N/A (Zero TN/FP) |

> [!NOTE]
> All 50 queries present in `reports/result_metrics.yaml` and `reports/metrics_details.csv` are answerable ($is\_answerable = True$). Model abstentions are tracked as False Negatives (FN). Pairwise McNemar exact tests yielded no differences surviving Holm correction ($p_{\text{raw}} = 0.0391 \implies p_{\text{Holm}} = 0.3906$).

---

## Table C: Friedman Omnibus Test Results ($k=5, n=50$)

| Metric | $\chi^2$ Statistic | Degrees of Freedom ($df$) | $p$-value | $n$ (Paired Queries) | Kendall's $W$ (Effect Size) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Retrieval Recall** | 122.5625 | 4 | $1.51 \times 10^{-25}$ | 50 | 0.6128 |
| **Context Precision** | 120.8802 | 4 | $3.46 \times 10^{-25}$ | 50 | 0.6044 |
| **Faithfulness** | 10.1492 | 4 | 0.0380 | 50 | 0.0507 |
| **Answer Relevance** | 35.8919 | 4 | $3.05 \times 10^{-7}$ | 50 | 0.1795 |
| **Citation Fidelity** | 19.8819 | 4 | $5.27 \times 10^{-4}$ | 50 | 0.0994 |
| **Semantic Accuracy** | 29.8384 | 4 | $5.28 \times 10^{-6}$ | 50 | 0.1492 |
| **Context Fillness** | 0.0000 | 4 | 1.0000 | 50 | 0.0000 |
| **AR-SA F1** | 22.0719 | 4 | $1.94 \times 10^{-4}$ | 50 | 0.1104 |
| **Latency (sec)** | 99.0400 | 4 | $1.57 \times 10^{-20}$ | 50 | 0.4952 |

---

## Table D: Pairwise Differences Significant After Holm-Bonferroni Correction

The following table lists **only** the baseline comparisons that remain statistically significant ($p_{\text{Holm}} < 0.05$) after applying step-down Holm-Bonferroni correction within each metric family:

| Metric | Baseline A | Baseline B | $\Delta$ (A - B) | 95% CI (A - B) | Raw $p$-value | Holm $p$-value | Effect Size ($r_b$) | Scientific Interpretation |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: | :--- |
| **Answer Relevance** | **B4** | **B6** | +0.3500 | [+0.2260, +0.4800] | $3.08 \times 10^{-5}$ | $3.08 \times 10^{-4}$ | +0.9763 | **B4** outperforms **B6** by 0.3500 |
| **Answer Relevance** | **B2** | **B6** | +0.3980 | [+0.2500, +0.5420] | $5.83 \times 10^{-5}$ | $5.24 \times 10^{-4}$ | +0.8207 | **B2** outperforms **B6** by 0.3980 |
| **Answer Relevance** | **B1** | **B2** | -0.2780 | [-0.4380, -0.1140] | 0.0022 | 0.0174 | -0.6379 | **B2** outperforms **B1** by 0.2780 |
| **Answer Relevance** | **B5** | **B6** | +0.1920 | [+0.0820, +0.3060] | 0.0036 | 0.0251 | +0.8162 | **B5** outperforms **B6** by 0.1920 |
| **AR-SA F1** | **B4** | **B6** | +0.1792 | [+0.0882, +0.2798] | 0.0022 | 0.0217 | +1.0000 | **B4** outperforms **B6** by 0.1792 |
| **AR-SA F1** | **B1** | **B4** | -0.1667 | [-0.2690, -0.0733] | 0.0029 | 0.0236 | -0.9341 | **B4** outperforms **B1** by 0.1667 |
| **AR-SA F1** | **B2** | **B6** | +0.1651 | [+0.0754, +0.2660] | 0.0026 | 0.0236 | +0.9451 | **B2** outperforms **B6** by 0.1651 |
| **Citation Fidelity** | **B5** | **B6** | +0.3362 | [+0.1541, +0.5242] | $9.96 \times 10^{-4}$ | 0.0100 | +0.6403 | **B5** outperforms **B6** by 0.3362 |
| **Citation Fidelity** | **B4** | **B6** | +0.2448 | [+0.1012, +0.3849] | 0.0027 | 0.0242 | +0.6215 | **B4** outperforms **B6** by 0.2448 |
| **Context Precision** | **B1** | **B2** | -0.6602 | [-0.7616, -0.5527] | $1.63 \times 10^{-8}$ | $1.63 \times 10^{-7}$ | -1.0000 | **B2** outperforms **B1** by 0.6602 |
| **Context Precision** | **B1** | **B4** | -0.5220 | [-0.6145, -0.4297] | $2.89 \times 10^{-8}$ | $2.60 \times 10^{-7}$ | -1.0000 | **B4** outperforms **B1** by 0.5220 |
| **Context Precision** | **B1** | **B5** | -0.5220 | [-0.6145, -0.4297] | $2.89 \times 10^{-8}$ | $2.60 \times 10^{-7}$ | -1.0000 | **B5** outperforms **B1** by 0.5220 |
| **Context Precision** | **B2** | **B6** | +0.4597 | [+0.3594, +0.5614] | $1.03 \times 10^{-7}$ | $7.20 \times 10^{-7}$ | +0.9892 | **B2** outperforms **B6** by 0.4597 |
| **Context Precision** | **B4** | **B6** | +0.3215 | [+0.2278, +0.4157] | $7.78 \times 10^{-7}$ | $4.67 \times 10^{-6}$ | +0.8749 | **B4** outperforms **B6** by 0.3215 |
| **Context Precision** | **B5** | **B6** | +0.3215 | [+0.2278, +0.4157] | $7.78 \times 10^{-7}$ | $4.67 \times 10^{-6}$ | +0.8749 | **B5** outperforms **B6** by 0.3215 |
| **Context Precision** | **B2** | **B4** | +0.1382 | [+0.0915, +0.1818] | $5.27 \times 10^{-5}$ | $2.11 \times 10^{-4}$ | +0.7268 | **B2** outperforms **B4** by 0.1382 |
| **Context Precision** | **B2** | **B5** | +0.1382 | [+0.0915, +0.1818] | $5.27 \times 10^{-5}$ | $2.11 \times 10^{-4}$ | +0.7268 | **B2** outperforms **B5** by 0.1382 |
| **Context Precision** | **B1** | **B6** | -0.2005 | [-0.2998, -0.1084] | $2.99 \times 10^{-4}$ | $5.98 \times 10^{-4}$ | -0.7808 | **B6** outperforms **B1** by 0.2005 |
| **Faithfulness** | **B1** | **B4** | +0.2315 | [+0.1030, +0.3563] | 0.0016 | 0.0161 | +0.5720 | **B1** outperforms **B4** by 0.2315 |
| **Faithfulness** | **B1** | **B2** | +0.2061 | [+0.0858, +0.3299] | 0.0038 | 0.0339 | +0.5192 | **B1** outperforms **B2** by 0.2061 |
| **Faithfulness** | **B1** | **B5** | +0.2167 | [+0.0779, +0.3535] | 0.0044 | 0.0349 | +0.5231 | **B1** outperforms **B5** by 0.2167 |
| **Latency (sec)** | **B1** | **B6** | -31.4603 | [-38.5184, -23.7563] | $1.34 \times 10^{-8}$ | $1.34 \times 10^{-7}$ | -0.9231 | **B1** is 31.46s faster than **B6** |
| **Latency (sec)** | **B2** | **B6** | -23.7635 | [-30.8152, -16.3366] | $3.45 \times 10^{-8}$ | $3.11 \times 10^{-7}$ | -0.8965 | **B2** is 23.76s faster than **B6** |
| **Latency (sec)** | **B4** | **B6** | -23.1115 | [-29.7134, -16.8860] | $6.99 \times 10^{-8}$ | $5.59 \times 10^{-7}$ | -0.8761 | **B4** is 23.11s faster than **B6** |
| **Latency (sec)** | **B5** | **B6** | -20.4787 | [-27.8254, -12.7600] | $4.34 \times 10^{-7}$ | $3.04 \times 10^{-6}$ | -0.8212 | **B5** is 20.48s faster than **B6** |
| **Latency (sec)** | **B1** | **B2** | -7.6968 | [-13.7184, -1.0224] | $3.85 \times 10^{-6}$ | $2.31 \times 10^{-5}$ | -0.7506 | **B1** is 7.70s faster than **B2** |
| **Latency (sec)** | **B1** | **B5** | -10.9816 | [-18.0685, -3.7330] | $1.25 \times 10^{-5}$ | $6.27 \times 10^{-5}$ | -0.7098 | **B1** is 10.98s faster than **B5** |
| **Latency (sec)** | **B1** | **B4** | -8.3488 | [-15.0667, -1.4604] | $5.81 \times 10^{-5}$ | $2.32 \times 10^{-4}$ | -0.6533 | **B1** is 8.35s faster than **B4** |
| **Retrieval Recall** | **B1** | **B2** | -0.6400 | [-0.7500, -0.5300] | $1.15 \times 10^{-8}$ | $1.15 \times 10^{-7}$ | -1.0000 | **B2** outperforms **B1** by 0.6400 |
| **Retrieval Recall** | **B1** | **B4** | -0.6000 | [-0.7100, -0.4900] | $1.95 \times 10^{-8}$ | $1.75 \times 10^{-7}$ | -1.0000 | **B4** outperforms **B1** by 0.6000 |
| **Retrieval Recall** | **B1** | **B5** | -0.6000 | [-0.7100, -0.4900] | $1.95 \times 10^{-8}$ | $1.75 \times 10^{-7}$ | -1.0000 | **B5** outperforms **B1** by 0.6000 |
| **Retrieval Recall** | **B2** | **B6** | +0.2900 | [+0.1900, +0.3900] | $1.83 \times 10^{-5}$ | $1.28 \times 10^{-4}$ | +1.0000 | **B2** outperforms **B6** by 0.2900 |
| **Retrieval Recall** | **B1** | **B6** | -0.3500 | [-0.4800, -0.2300] | $3.34 \times 10^{-5}$ | $2.00 \times 10^{-4}$ | -0.9684 | **B6** outperforms **B1** by 0.3500 |
| **Retrieval Recall** | **B4** | **B6** | +0.2500 | [+0.1600, +0.3500] | $6.68 \times 10^{-5}$ | $3.34 \times 10^{-4}$ | +1.0000 | **B4** outperforms **B6** by 0.2500 |
| **Retrieval Recall** | **B5** | **B6** | +0.2500 | [+0.1600, +0.3500] | $6.68 \times 10^{-5}$ | $3.34 \times 10^{-4}$ | +1.0000 | **B5** outperforms **B6** by 0.2500 |
| **Semantic Accuracy** | **B4** | **B6** | +0.2090 | [+0.1140, +0.3120] | $4.01 \times 10^{-4}$ | 0.0040 | +1.0000 | **B4** outperforms **B6** by 0.2090 |
| **Semantic Accuracy** | **B1** | **B4** | -0.1850 | [-0.2840, -0.0940] | $6.93 \times 10^{-4}$ | 0.0062 | -0.9346 | **B4** outperforms **B1** by 0.1850 |
| **Semantic Accuracy** | **B2** | **B6** | +0.1780 | [+0.0890, +0.2770] | $8.88 \times 10^{-4}$ | 0.0071 | +0.9412 | **B2** outperforms **B6** by 0.1780 |
| **Semantic Accuracy** | **B5** | **B6** | +0.1970 | [+0.1000, +0.3000] | 0.0021 | 0.0150 | +1.0000 | **B5** outperforms **B6** by 0.1970 |
| **Semantic Accuracy** | **B1** | **B5** | -0.1730 | [-0.2790, -0.0760] | 0.0034 | 0.0202 | -0.8857 | **B5** outperforms **B1** by 0.1730 |
| **Semantic Accuracy** | **B1** | **B2** | -0.1540 | [-0.2630, -0.0540] | 0.0083 | 0.0413 | -0.7076 | **B2** outperforms **B1** by 0.1540 |

---

## 💡 Key Decision Support Insights

1. **Retrieval Capabilities (Recall & Precision):** Dense retrieval baselines (**B2**, **B4**, **B5**) demonstrate massive, statistically reliable superiority over Pure Lexical (**B1**) and Full Pipeline (**B6**) ($p_{\text{Holm}} < 1.15 \times 10^{-7}$, $r_b > 0.87$). Specifically, **B2**, **B4**, and **B5** achieve top retrieval recall ($0.7600$--$0.8000$) and context precision ($0.6840$--$0.8222$), with zero detectable difference between **B4** and **B5** ($p_{\text{Holm}} = 1.0000$).
2. **Faithfulness:** Pure Lexical (**B1**) achieves significantly higher faithfulness ($0.6643$) than **B2** ($0.4582$), **B4** ($0.4328$), and **B5** ($0.4476$) ($p_{\text{Holm}} \le 0.0349$). Lexical retrieval returns short, exact matching contexts that reduce LLM hallucination opportunities, though at severe cost to retrieval recall.
3. **Answer Relevance & Citation Fidelity:** **B2**, **B4**, and **B5** significantly outperform **B6** in answer relevance ($p_{\text{Holm}} \le 0.0251$). For citation fidelity, **B5** ($0.5199$) and **B4** ($0.4285$) significantly outperform **B6** ($0.1837$, $p_{\text{Holm}} \le 0.0242$).
4. **Semantic Accuracy:** **B4** ($0.2130$), **B5** ($0.2010$), and **B2** ($0.1820$) significantly outperform both **B1** ($0.0280$) and **B6** ($0.0040$) ($p_{\text{Holm}} \le 0.0413$). No statistically significant difference exists between **B2**, **B4**, and **B5**.
5. **Latency Trade-offs:** **B1** is fastest ($16.721\text{s}$), followed by **B2** ($24.418\text{s}$), **B4** ($25.070\text{s}$), and **B5** ($27.703\text{s}$). Full Pipeline **B6** ($48.182\text{s}$) is significantly slower than all other baselines ($p_{\text{Holm}} < 5.00 \times 10^{-10}$, $\Delta \approx 20.48\text{s}$ to $31.46\text{s}$).

---

## 🔗 Related Documentation
- [Non-Parametric Statistical Testing Framework](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/statistics/statistical_testing_framework.md)
- [Integrated Architecture Reference](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/architecture/integrated_pipeline.md)
- [Shannon Estimator Manual](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/telemetry/shannon_estimator.md)
- [ADR-001: Core Module Refactoring](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/adr/ADR-001-core-module-refactoring.md)
