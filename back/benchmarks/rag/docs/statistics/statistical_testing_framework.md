# Statistical Testing Architecture for Science Graph RAG Benchmarks

> [!NOTE]
> Non-parametric statistical inference framework specifications implemented in `core/statistics.py`. Details hypothesis testing hierarchies, omnibus ANOVA, pairwise rank tests, effect size estimators, Holm-Bonferroni FWER control, paired query bootstrap CIs, and outcome-aware missing data handling.

---

## 1. Overview & Statistical Requirements

Evaluating Retrieval-Augmented Generation (RAG) pipelines over scientific knowledge graphs requires rigorous statistical inference. Standard parametric tests (e.g., paired $t$-tests) assume normal error distributions, which are frequently violated in LLM-as-a-judge quality scores (Likert-scale or bounded $[0, 1]$ metrics with extreme skewness and zero-inflation) and execution latencies (heavy-tailed distributions).

The framework in [core/statistics.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/statistics.py) provides:
- Multi-baseline omnibus evaluation and post-hoc pairwise hypothesis testing.
- Effect size quantification using standardized non-parametric metrics.
- Multiple-comparison correction using step-down Family-Wise Error Rate (FWER) control.
- Resampling-based uncertainty estimation via paired query bootstrapping.
- Outcome-aware filtering and missing data (`NaN`) propagation.
- Boundary condition protocols for high-tie regimes, ceiling/floor effects, and sample size sensitivity.

---

## 2. Non-Parametric Paired Hypothesis Hierarchy

The evaluation pipeline follows a two-tier non-parametric hypothesis testing hierarchy:

```
                  ┌─────────────────────────────────────────┐
                  │ Evaluate k Baselines on Shared Queries  │
                  └────────────────────┬────────────────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         │   Metric Classification   │
                         └─────────────┬─────────────┘
                                       │
             ┌─────────────────────────┼─────────────────────────┐
             ▼                         ▼                         ▼
   Continuous / Quality      Continuous / Quality         Binary Classification
        (k >= 3)                   (k = 2)                   (e.g., Correctness)
             │                         │                         │
             ▼                         ▼                         ▼
   Friedman Omnibus Test       Wilcoxon Signed-Rank         McNemar Exact Test
  (Kendall's W Effect)         (Rank-Biserial r_b)          (Discordant Pairs)
             │                         │                         │
             └─────────────┬───────────┴─────────────────────────┘
                           │
                           ▼
            Family-Wise Holm-Bonferroni
               P-Value Correction
```

### 2.1 Friedman Omnibus Test ($k \ge 3$)
When evaluating $k \ge 3$ baselines across $n$ paired queries, the Friedman test detects whether at least one baseline systematically differs in performance without assuming normality.

- **Hypotheses**:
  - $H_0$: The median performance across all $k$ baselines is identical.
  - $H_1$: At least one baseline's performance median differs from another.
- **Matrix Setup**: For metric $M$, scores are aligned in block matrix $X \in \mathbb{R}^{n \times k}$, where $X_{i, j}$ is the score of query $i$ under baseline $j$.
- **Ranking**: Each query row $i$ is ranked independently across the $k$ baselines, producing rank matrix $R \in \mathbb{R}^{n \times k}$ with average ranks assigned to ties:
  $$R_{i, j} = \text{rank}(X_{i, j}) \quad \text{for } j = 1, \dots, k$$
- **Test Statistic**:
  $$\chi^2_F = \frac{12}{n k (k + 1)} \sum_{j=1}^k R_{\cdot, j}^2 - 3 n (k + 1)$$
  where $R_{\cdot, j} = \sum_{i=1}^n R_{i, j}$ is the rank sum for baseline $j$.

### 2.2 Post-Hoc Pairwise Testing: Wilcoxon Signed-Rank Test
For pairwise baseline comparisons $(A, B)$ on continuous quality or latency metrics, the Wilcoxon signed-rank test evaluates paired differences $d_i = X_{i, A} - X_{i, B}$.

- **Hypotheses**:
  - $H_0$: The distribution of paired differences $d_i$ is symmetric around zero.
  - $H_1$: The median of paired differences $d_i$ deviates significantly from zero.
- **Zero-Difference Exclusion**: Pairs with $d_i = 0$ are excluded ($n_r$ denotes non-zero pairs).
- **Ranking**: Absolute non-zero differences $|d_i|$ are ranked $r_i = \text{rank}(|d_i|)$.
- **Test Statistic**:
  $$W_+ = \sum_{i: d_i > 0} r_i, \quad W_- = \sum_{i: d_i < 0} r_i, \quad T = \min(W_+, W_-)$$

### 2.3 Binary Classification Comparison: McNemar Exact Test
For paired binary response metrics (e.g., `answerability_correct` $\in \{0, 1\}$), the McNemar test evaluates marginal homogeneity across paired binary outcomes focusing on discordant pairs ($b$: $A$ pass & $B$ fail, $c$: $A$ fail & $B$ pass):

- **Exact Binomial Test** ($b + c < 25$):
  $$p = 2 \sum_{k=0}^{\min(b, c)} \binom{b+c}{k} (0.5)^{b+c}$$
- **Asymptotic $\chi^2$ Test with Continuity Correction** ($b + c \ge 25$):
  $$\chi^2_{\text{McNemar}} = \frac{(|b - c| - 1)^2}{b + c}$$

---

## 3. Effect Size Estimation

### 3.1 Kendall's $W$ (Friedman Omnibus Effect Size)

$$W = \frac{\chi^2_F}{n (k - 1)}$$

- **Interpretation Thresholds**: $W < 0.1$ (Negligible), $0.1 \le W < 0.3$ (Small), $0.3 \le W < 0.5$ (Moderate), $W \ge 0.5$ (Large).

### 3.2 Rank-Biserial Correlation $r_b$ (Wilcoxon Signed-Rank Effect Size)

$$r_b = \frac{W_+ - W_-}{W_+ + W_-} = \frac{2 W_+}{W_+ + W_-} - 1 \in [-1.0, +1.0]$$

- **Interpretation Thresholds**: $|r_b| < 0.1$ (Negligible), $0.1 \le |r_b| < 0.3$ (Small), $0.3 \le |r_b| < 0.5$ (Medium), $|r_b| \ge 0.5$ (Large).

---

## 4. Multiple-Comparison Correction Strategy

Hypothesis testing across multiple metrics and baseline pairs increases Type I errors (false positives). The framework enforces step-down Family-Wise Error Rate (FWER) control.

### 4.1 Metric Family Partitioning
Rather than pooling all $p$-values into a single global correction vector (which overly penalizes independent evaluation dimensions), tests are partitioned into distinct **Metric Families**:
1. **Quality Family**: Faithfulness, Answer Relevance, Context Precision, Retrieval Recall, Semantic Accuracy.
2. **Performance / Operational Family**: End-to-end Latency, Retrieval Latency, Generation Latency.
3. **Safety / Answerability Family**: Classification accuracy, MCC, Hallucination Rate, Abstention Accuracy.

Corrections are applied **within each metric family** across all pairwise baseline combinations $\binom{k}{2}$.

### 4.2 Step-Down Holm-Bonferroni Algorithm

Given $m$ hypotheses in a family with unadjusted $p$-values $p_1, p_2, \dots, p_m$:
1. Sort $p$-values in ascending order: $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$.
2. For rank $i = 1, \dots, m$, compare $p_{(i)}$ against the adjusted threshold:
   $$\text{Threshold}_i = \frac{\alpha}{m - i + 1}$$
3. Reject null hypotheses $H_{(1)}, \dots, H_{(k-1)}$ where $p_{(i)} \le \text{Threshold}_i$.

---

## 5. Paired Difference Bootstrap Confidence Intervals

### 5.1 Query-Level Paired Resampling
To preserve inter-system query-level correlation, bootstrapping **resamples intact queries** ($B = 10,000$ iterations, seed 42), not individual system outputs independently.

### 5.2 Dynamic Alpha Synchronization
To prevent contradictions where 95% CIs exclude $0.0$ on pairwise comparisons that fail Holm-Bonferroni significance testing ($p_{\text{corrected}} \ge \alpha$), the alpha level for bootstrap percentile bounds is dynamically updated to $\alpha_{\text{adjusted}}$ on a per-comparison basis:

$$\text{Percentile Bounds} = \left[ \frac{\alpha_{\text{adjusted}}}{2} \times 100, \, \left(1 - \frac{\alpha_{\text{adjusted}}}{2}\right) \times 100 \right]$$

This mathematically guarantees that non-significant comparisons ($p_{\text{corrected}} \ge \alpha$) have confidence intervals spanning $0.0$.

---

## 6. Outcome-Aware NaN Handling & Filtering Policy

> [!IMPORTANT]
> Filling missing scores on abstained or unanswerable queries with zero or mean values distorts statistical inference.

### 6.1 Quality Metric Abstention Rules
- **True Negatives (TN)** (Correct abstention on unanswerable query): Quality scores (faithfulness, relevance) are **conceptually undefined**. Quality values MUST be set to `NaN`.
- **False Positives (FP)** (Hallucinated answer on unanswerable query): Excluded from quality aggregation via outcome filtering.
- **False Negatives (FN)** (Abstained on answerable query): Quality judge skipped. Set to `NaN`.

$$\text{Quality Filter Rule: } \text{Query } i \text{ included in } M_{\text{quality}} \iff \text{Outcome}_i \in \{\text{TP}, \text{FP}\}$$

### 6.2 Complete Paired Case Analysis vs Full Query Set
- **Quality Metrics**: Tested using **Complete Paired Case Analysis** ($i \in \text{TP}_A \cap \text{TP}_B$). When all dataset queries are answerable ($is\_answerable = True$), $n = 50$ complete pairs are preserved.
- **Safety / Classification Metrics**: Evaluated over the **Full Query Set** ($n = 50$).

---

## 7. Core Statistics API Reference ([core/statistics.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/statistics.py))

| Function | Primary Purpose | Input Parameters | Output Return Schema |
| :--- | :--- | :--- | :--- |
| `prepare_per_query_records()` | Extract per-query metrics with outcome NaNs | Pipeline JSON dict | `(records, baselines)` |
| `paired_metric_vectors()` | Extract aligned paired vectors for $(A, B)$ | Records, baselines, metric | `(vec_a, vec_b, query_ids)` |
| `friedman_omnibus_test()` | Omnibus test across $k \ge 3$ systems | Records, baselines, metric | `{"statistic", "p_value", "n"}` |
| `wilcoxon_signed_rank_test()` | Pairwise non-parametric continuous test | `vec_a`, `vec_b` | `{"statistic", "p_value", "effect_size", "n"}` |
| `mcnemar_answerability_test()` | Pairwise binary test on correctness | Records, $b_A$, $b_B$ | `{"p_value", "b", "c", "n"}` |
| `rank_biserial_effect_size()` | Wilcoxon effect size $r_b$ | Differences array | Float in $[-1.0, +1.0]$ |
| `bootstrap_paired_difference_ci()`| Resampling paired difference 95% CI | `vec_a`, `vec_b`, config | `{"delta", "ci_lower", "ci_upper", "n"}` |
| `holm_correction()` | FWER step-down p-value adjustment | `p_values`, alpha | `list[bool]` (significance flags) |

---

## 🔗 Related Documentation
- [Empirical Baseline Evaluation Report](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/statistics/baseline_evaluation_report.md)
- [Integrated Architecture Reference](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/architecture/integrated_pipeline.md)
- [Shannon Estimator Manual](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/telemetry/shannon_estimator.md)
- [ADR-001: Core Module Refactoring](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/adr/ADR-001-core-module-refactoring.md)
