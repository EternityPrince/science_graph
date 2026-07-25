# Shannon Estimator: Calculation, Aggregation, and Diagnostics Manual

> [!NOTE]
> Comprehensive technical manual and mathematical reference for **Shannon Entropy Diagnostics** in the **Science Graph RAG Benchmark Suite**. All entropy values are measured and reported in **bits** ($\log_2$).

---

## 1. Architectural Overview & Diagnostics Lifecycle

The Shannon Estimator tracks uncertainty across all processing stages of Retrieval-Augmented Generation (RAG): candidate score uncertainty, lexical context diversity, graph topological entropy, token-level predictive generation uncertainty, citation span entropy, and entropy reduction relative to zero-shot baselines.

```
                    ┌─────────────────────────────────────────┐
                    │    1. RETRIEVAL & TRIMMING STAGE        │
                    │         (core/retrieval.py)             │
                    │  - Capture pre-rerank & post-rerank     │
                    │  - Capture pre-trim & post-trim text    │
                    │  - Capture graph relations / Cypher text│
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │          2. GENERATION STAGE            │
                    │   (core/generation.py & pipelined.py)   │
                    │  - Stream token logprobs / logits       │
                    │  - compute_generation_entropy (H_gen)   │
                    │  - align_tokens_info & find_spans       │
                    │  - compute_citation_entropy (H_cit)     │
                    │  - Cache B0 entropy & compute ΔH_gen    │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │      3. AGGREGATION & ANALYTICS         │
                    │          (core/analytics.py)            │
                    │  - Extract/backfill shannon_diagnostics │
                    │  - Statistical means, min, max, stdev   │
                    │  - Flag has_shannon in summary stats    │
                    └────────────────────┬────────────────────┘
                                         │
                      ┌───────────────────┴───────────────────┐
                      ▼                                       ▼
        ┌──────────────────────────┐             ┌──────────────────────────┐
        │   4. REPORT GENERATION   │             │ 5. ACADEMIC VISUALIZATION│
        │    (core/reporting.py)   │             │(generate_visualizations) │
        │ - Rich Console Tables    │             │ - H_gen vs Hallucinations│
        │ - Markdown Summary Tables│             │ - H_rank Collapse        │
        │                          │             │ - Trade-off Bubble Plots │
        │                          │             │ - Metric Heatmaps        │
        └──────────────────────────┘             └──────────────────────────┘
```

---

## 2. Metric Catalog

| Metric Name | Symbol | Code Function / Source | Scope & Mathematical Meaning |
| :--- | :--- | :--- | :--- |
| **Rank Entropy (Pre-Rerank)** | $H_{\text{rank,pre}}$ | `compute_rank_entropy(pre_scores)` | Candidate relevance uncertainty before reranking |
| **Rank Entropy (Post-Rerank)** | $H_{\text{rank,post}}$ | `compute_rank_entropy(post_scores)` | Candidate relevance uncertainty after cross-encoder reranking |
| **Lexical Entropy (Pre-Trim)** | $H_{\text{lex,pre}}$ | `compute_lexical_entropy(context_text)` | Unigram vocabulary entropy of raw retrieved context |
| **Lexical Entropy (Post-Trim)** | $H_{\text{lex,post}}$ | `compute_lexical_entropy(trimmed_text)` | Unigram vocabulary entropy of token-trimmed context |
| **Graph Relation Entropy** | $H_{\text{graph,rel}}$ | `compute_graph_entropy` $\rightarrow$ `rel_entropy` | Diversity of edge relation types in retrieved subgraph |
| **Graph Degree Entropy** | $H_{\text{graph,deg}}$ | `compute_graph_entropy` $\rightarrow$ `degree_entropy` | Diversity of node degree mass distribution |
| **Generation Entropy** | $H_{\text{gen}}$ | `compute_generation_entropy` | Mean per-token predictive entropy over emitted response |
| **Citation Entropy** | $H_{\text{citation}}$ | `compute_citation_entropy` | Mean entropy of tokens positioned within citation markers |
| **Citation Token Count** | $n_{\text{citation}}$ | `compute_citation_entropy` | Count of generated tokens belonging to citation spans |
| **Entropy Reduction** | $\Delta H_{\text{gen}}$ | `compute_entropy_reduction` | Uncertainty reduction vs Zero-Shot baseline ($H_{\text{gen}}^{B0} - H_{\text{gen}}^{\text{RAG}}$) |

---

## 3. Mathematical Formulas & Calculation Logic ([core/shannon_estimator.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/shannon_estimator.py))

All entropy metrics are computed in bits ($\log_2$) in `core/shannon_estimator.py`.

### 3.1 Candidate Rank Entropy ($H_{\text{rank}}$)

Given candidate retrieval scores $s_1, s_2, \ldots, s_N$:

1. **Boundary Condition**: If $N \le 1$, return $0.0$ bits.
2. **Softmax Normalization** (Default, `method="softmax"`):
   Temperature $\tau > 0$ (if $\tau \le 0$, clamped to $10^{-6}$). Shift by $s_{\max} = \max(s)$ for numerical stability:
   $$P(c_i) = \frac{\exp\left(\frac{s_i - s_{\max}}{\tau}\right)}{\sum_{j=1}^{N} \exp\left(\frac{s_j - s_{\max}}{\tau}\right)}$$
3. **MinMax Normalization** (`method="minmax"`):
   $$\tilde{s}_i = \begin{cases} \frac{1}{N} & \text{if } \max(s) = \min(s) \\ \frac{s_i - \min(s)}{\max(s) - \min(s)} & \text{otherwise} \end{cases}, \qquad P(c_i) = \frac{\tilde{s}_i}{\sum_{j=1}^{N} \tilde{s}_j}$$
4. **Shannon Entropy Calculation**:
   $$H_{\text{rank}} = -\sum_{i: P(c_i) > 0} P(c_i) \log_2 P(c_i)$$

---

### 3.2 Lexical Unigram Entropy ($H_{\text{lexical}}$)

1. Tokenize text into words via regex matching:
   $$\text{tokens} = \text{re.findall}(r"\w+", \text{text.lower()})$$
2. For empirical word counts $c(w)$ and total token count $T = \sum_w c(w)$:
   $$P(w) = \frac{c(w)}{T}$$
3. Shannon Entropy:
   $$H_{\text{lexical}} = -\sum_{w: P(w) > 0} P(w) \log_2 P(w)$$

---

### 3.3 Graph Topology Entropy ($H_{\text{graph,rel}}$, $H_{\text{graph,deg}}$)

1. **Relation Type Entropy ($H_{\text{graph,rel}}$)**:
   For $|E|$ total edges and relation type counts $c(r)$:
   $$P(r) = \frac{c(r)}{|E|}, \qquad H_{\text{graph,rel}} = -\sum_{r} P(r) \log_2 P(r)$$
2. **Degree Entropy ($H_{\text{graph,deg}}$)**:
   Calculates undirected incidence degree $d(v)$ for each node $v \in V$. Total degree mass $D = \sum_v d(v)$:
   $$P(v) = \frac{d(v)}{D}, \qquad H_{\text{graph,deg}} = -\sum_{v} P(v) \log_2 P(v)$$

---

### 3.4 Generation Entropy ($H_{\text{gen}}$) & Resolution Hierarchy

For a sequence of $M$ generated token metadata dictionaries:

$$H_{\text{gen}} = \frac{1}{M} \sum_{k=1}^{M} h_k$$

Where per-token entropy $h_k$ is extracted by `_extract_single_token_entropy(t)` according to the resolution hierarchy:
1. **Explicit Entropy Field**: Returns `t["entropy"]` in bits if present.
2. **Probability Distribution (`probs` / `top_probs`)**: Calculates $-\sum p \log_2 p$.
3. **Logprob Distribution (`top_logprobs` / `logprobs`)**: Exponentiates $p_i = \exp(lp_i)$, renormalizes $\sum p_i = 1$, calculates $-\sum p_i \log_2 p_i$.
4. **Single Token Surprisal (`logprob`)**: Converts log probability to surprisal in bits $-\log_2 P(t_k)$.
5. **Fallback**: Returns $0.0$.

---

### 3.5 Citation Entropy ($H_{\text{citation}}$) & Token Alignment

1. **Citation Span Extraction (`find_citation_spans`)**:
   Uses regex matching to locate $[start, end)$ character offsets for citation markers including `[sciq_paper_X]`, `[Block X]`, `[1]`, `[paper_1]`, DOIs (`10.xxxx/...`), arXiv IDs (`arXiv:XXXX`), and author-year citations (`[Smith et al., 2020]`). Overlapping spans are merged.
2. **Citation Entropy Calculation**:
   Selects tokens whose character intervals overlap any citation span, computing:
   $$H_{\text{citation}} = \text{compute\_generation\_entropy}(\text{citation\_tokens}), \qquad n_{\text{citation}} = |\text{citation\_tokens}|$$

---

### 3.6 Entropy Reduction ($\Delta H_{\text{gen}}$)

$$\Delta H_{\text{gen}} = H_{\text{gen}}^{B0} - H_{\text{gen}}^{\text{RAG}}$$

- **Positive $\Delta H_{\text{gen}} > 0$**: RAG context successfully **reduced** predictive uncertainty relative to zero-shot generation ($B_0$).
- **Negative $\Delta H_{\text{gen}} < 0$**: Context increased predictive uncertainty (e.g. conflicting or noisy context).

---

## 4. Worked Numerical Examples

### Example A — Uniform Rank Entropy ($H = 2.0$ bits)
Four candidates with equal scores $[1.0, 1.0, 1.0, 1.0]$ under Softmax ($\tau=1.0$):
$$P(c_i) = \frac{1}{4} \quad \forall i, \qquad H_{\text{rank}} = -4 \times \left(\frac{1}{4} \log_2 \frac{1}{4}\right) = 2.0 \text{ bits}$$

### Example B — Cross-Encoder Reranking Entropy Collapse
Pre-rerank scores $[1, 1, 1, 1]$ ($H_{\text{pre}} = 2.0$) $\rightarrow$ Post-rerank scores $[10.0, 0.1, 0.1, 0.1]$ under Softmax ($\tau=1.0$):
$$P_0 \approx 0.99985, \quad P_{1,2,3} \approx 5.016 \times 10^{-5} \implies H_{\text{post}} \approx 0.0026 \text{ bits}$$
*Large positive $\Delta H_{\text{rank}} = H_{\text{pre}} - H_{\text{post}} = 1.9974$ confirms cross-encoder uncertainty collapse onto the top candidate.*

### Example C — Lexical Unigram Entropy ($H = 1.0$ bit)
Input string `"alpha beta alpha beta"`:
Counts: `alpha`: 2, `beta`: 2 ($T=4$). Probabilities: $P(\text{alpha}) = 0.5, P(\text{beta}) = 0.5$.
$$H_{\text{lexical}} = - (0.5 \log_2 0.5 + 0.5 \log_2 0.5) = 1.0 \text{ bit}$$

---

## 🔗 Related Documentation
- [Integrated Architecture Reference](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/architecture/integrated_pipeline.md)
- [Logit Telemetry Specifications](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/telemetry/logit_telemetry.md)
- [Non-Parametric Statistical Testing Framework](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/statistics/statistical_testing_framework.md)
- [Empirical Baseline Evaluation Report](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/statistics/baseline_evaluation_report.md)
