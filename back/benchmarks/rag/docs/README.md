# Science Graph RAG Benchmark Documentation

Welcome to the technical documentation suite for the **Science Graph Retrieval-Augmented Generation (RAG) Benchmark Framework**. This repository contains detailed architectural references, mathematical formulations for logit-level telemetry and information-theoretic diagnostics, rigorous non-parametric statistical evaluation methodology, and architectural decision records (ADRs).

---

## 📂 Documentation Sitemap & Taxonomy

The documentation is organized into four core domain subdirectories:

```
docs/
├── README.md                              # Master index & sitemap (this file)
├── architecture/                          # Core System Architecture & Pipeline Orchestration
│   ├── integrated_pipeline.md            # End-to-end multi-stage pipeline architecture
│   └── pipeline_orchestration.md         # Process isolation, CLI execution, and run lifecycles
├── telemetry/                             # Logit Telemetry & Shannon Entropy Diagnostics
│   ├── logit_telemetry.md                # Token logprob extraction, alignment, MSP, CLR
│   └── shannon_estimator.md              # Information-theoretic entropy math & diagnostics manual
├── statistics/                            # Statistical Methodology & Empirical Benchmark Results
│   ├── statistical_testing_framework.md  # Non-parametric testing (Friedman, Wilcoxon, McNemar)
│   └── baseline_evaluation_report.md     # Paired empirical evaluation report (B1–B6, n=50)
└── adr/                                   # Architectural Decision Records
    └── ADR-001-core-module-refactoring.md # Decoupling core modules, state drift & matrix fixes
```

---

## 🧭 Navigation Guide

| Domain | Document | Key Topics & Scope |
| :--- | :--- | :--- |
| **System Architecture** | [integrated_pipeline.md](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/architecture/integrated_pipeline.md) | Multi-stage retrieval (FTS5, Dense, Graph), MinMax score scaling, ID normalization (`normalize_id`), end-to-end data contracts. |
| **Orchestration & Workflow** | [pipeline_orchestration.md](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/architecture/pipeline_orchestration.md) | Process-isolated stage execution (`run_pipeline.py`), VRAM management, CLI flags, sequential vs. pipelined modes, run directories. |
| **Logit Telemetry** | [logit_telemetry.md](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/telemetry/logit_telemetry.md) | Maximum Softmax Probability ($MSP$), top-1 vs top-2 logit margin ($\Delta z_{1,2}$), citation onset mapping, Contextual Log-Likelihood Ratio ($CLR$). |
| **Shannon Diagnostics** | [shannon_estimator.md](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/telemetry/shannon_estimator.md) | Candidate rank entropy ($H_{\text{rank}}$), lexical unigram entropy ($H_{\text{lexical}}$), graph topology entropy ($H_{\text{graph}}$), generation entropy ($H_{\text{gen}}$), entropy reduction ($\Delta H_{\text{gen}}$). |
| **Statistical Framework** | [statistical_testing_framework.md](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/statistics/statistical_testing_framework.md) | Non-parametric hypothesis testing hierarchy (Friedman omnibus, Wilcoxon signed-rank, McNemar), Kendall's $W$, rank-biserial $r_b$, Holm-Bonferroni FWER control, bootstrap CIs. |
| **Empirical Results** | [baseline_evaluation_report.md](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/statistics/baseline_evaluation_report.md) | Full paired statistical benchmark evaluation ($n=50$) across baselines **B1** (Lexical), **B2** (Dense), **B4** (Hybrid+Rerank), **B5** (Hybrid+Graph), **B6** (Full Pipeline). |
| **Architecture Decisions** | [ADR-001-core-module-refactoring.md](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/adr/ADR-001-core-module-refactoring.md) | Record of refactoring root monolithic scripts into modular subpackages (`core/traces.py`, `core/visualization.py`, `core/connector.py`), graph baseline isolation, sample size scope preservation ($n=50$). |

---

## ⚛️ Key Scientific Principles & Metrics Summary

### 1. Multi-Stage Retrieval & Score Scaling
- **Heterogeneous Score Normalization**:
  $$S_{\text{norm}} = \frac{S - S_{\min}}{S_{\max} - S_{\min}}$$
  Guarantees uniform cross-component fusion across BM25 float scores ($[0, \infty)$), dense vector cosine similarities ($[-1, 1]$), and graph structural weights.
- **Ground-Truth ID Normalization**: `normalize_id()` strips path prefixes, extensions, separators, and chunk markers (`_chunk.*`) to align retrieved chunks with canonical document ground truth, eliminating false recall inflation.

### 2. Logit Telemetry & Information-Theoretic Diagnostics
- **Maximum Softmax Probability ($MSP$)**: $MSP = \max_i p_i$, where $p_i = \frac{\exp(z_i - \max z)}{\sum \exp(z_j - \max z)}$.
- **Logit Margin ($\Delta z_{1,2}$)**: Gap between top-1 and top-2 unnormalized logits ($z_1 - z_2$).
- **Contextual Log-Likelihood Ratio ($CLR$)**: Quantitative log-likelihood boost provided by retrieved context relative to a zero-context baseline prompt ($B_0$):
  $$CLR = LL_{\text{rag}} - LL_{\text{base}} = \sum_{t=1}^N \log P_{RAG}(w_t \mid w_{<t}) - \sum_{t=1}^N \log P_{B0}(w_t \mid w_{<t})$$
- **Shannon Entropy Diagnostics**: Measured strictly in **bits** ($\log_2$):
  $$H = -\sum_{i} P_i \log_2 P_i$$
  Calculated across candidate rankings ($H_{\text{rank}}$), lexical text ($H_{\text{lexical}}$), graph relation/degree distributions ($H_{\text{graph}}$), predictive output generation ($H_{\text{gen}}$), and citation onset spans ($H_{\text{citation}}$).

### 3. Non-Parametric Statistical Inference
- **Friedman Omnibus Test**: Omnibus non-parametric ANOVA ($k \ge 3$ systems) with Kendall's $W$ effect size.
- **Wilcoxon Signed-Rank Test**: Pairwise rank test for continuous quality and latency metrics, evaluated alongside rank-biserial correlation effect size $r_b \in [-1.0, +1.0]$.
- **McNemar Exact Test**: Paired binary test for answerability correctness over discordant cases.
- **FWER Control**: Step-down Holm-Bonferroni correction applied independently within **Quality**, **Performance**, and **Safety** metric families.
- **Paired Bootstrap CIs**: Non-parametric percentile confidence intervals (10,000 resamples, seed 42) over intact query pairs.

---

## 🛠️ Codebase Mapping

| Subsystem | Primary Source Files | Related Documentation |
| :--- | :--- | :--- |
| **Pipeline Orchestrator** | `run_pipeline.py`, `core/subprocess_runner.py`, `core/pipelined.py` | [pipeline_orchestration.md](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/architecture/pipeline_orchestration.md) |
| **Retrieval & Normalization** | `core/retrieval.py`, `core/metrics.py` | [integrated_pipeline.md](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/architecture/integrated_pipeline.md) |
| **Logit Telemetry & Generation** | `core/generation.py`, `core/shannon_estimator.py` | [logit_telemetry.md](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/telemetry/logit_telemetry.md) |
| **Shannon Entropy Diagnostics** | `core/shannon_estimator.py`, `core/analytics.py` | [shannon_estimator.md](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/telemetry/shannon_estimator.md) |
| **Statistical Engine** | `core/statistics.py`, `core/connector.py` | [statistical_testing_framework.md](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/statistics/statistical_testing_framework.md) |
| **Pydantic Data Models** | `core/models.py` | [integrated_pipeline.md](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/architecture/integrated_pipeline.md) |
| **Reporting & Exporters** | `core/reporting.py`, `parse_metrics.py` | [pipeline_orchestration.md](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/architecture/pipeline_orchestration.md) |
