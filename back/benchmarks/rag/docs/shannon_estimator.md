# Shannon Estimator: Calculation, Aggregation, and Pipeline Interaction Manual

This document provides a comprehensive technical manual and mathematical reference for **Shannon Entropy Diagnostics** in the **Science Graph RAG Benchmark Suite**. All entropy values are measured and reported in **bits** ($\log_2$).

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
       └──────────────────────────┘             │ - Metric Heatmaps        │
                                                └──────────────────────────┘
```

---

## 2. Metric Catalog

| Metric | Symbol | Function / Source | Scope / Meaning |
|---|---|---|---|
| **Rank Entropy (Pre-Rerank)** | $H_{\text{rank,pre}}$ | `compute_rank_entropy(pre_scores)` | Candidate relevance uncertainty before reranking |
| **Rank Entropy (Post-Rerank)** | $H_{\text{rank,post}}$ | `compute_rank_entropy(post_scores)` | Candidate relevance uncertainty after reranking |
| **Lexical Entropy (Pre-Trim)** | $H_{\text{lex,pre}}$ | `compute_lexical_entropy(context_text)` | Unigram vocabulary entropy of raw retrieved context |
| **Lexical Entropy (Post-Trim)** | $H_{\text{lex,post}}$ | `compute_lexical_entropy(trimmed_text)` | Unigram vocabulary entropy of token-trimmed context |
| **Graph Relation Entropy** | $H_{\text{graph,rel}}$ | `compute_graph_entropy` $\rightarrow$ `relation_type_entropy` | Diversity of edge relation types in retrieved subgraph |
| **Graph Degree Entropy** | $H_{\text{graph,deg}}$ | `compute_graph_entropy` $\rightarrow$ `degree_entropy` | Diversity of node degree mass distribution |
| **Generation Entropy** | $H_{\text{gen}}$ | `compute_generation_entropy` | Mean per-token predictive entropy over generated response |
| **Citation Entropy** | $H_{\text{citation}}$ | `compute_citation_entropy` | Mean entropy of tokens positioned within citation markers |
| **Citation Token Count** | $n_{\text{citation}}$ | `compute_citation_entropy` | Count of generated tokens belonging to citation spans |
| **Entropy Reduction** | $\Delta H_{\text{gen}}$ | `compute_entropy_reduction` | Uncertainty reduction vs Zero-Shot B0 ($H_{\text{gen}}^{B0} - H_{\text{gen}}^{RAG}$) |

---

## 3. Mathematical Formulas & Calculation Logic (`core/shannon_estimator.py`)

All calculations are implemented in `core/shannon_estimator.py`.

### 3.1 Candidate Rank Entropy ($H_{\text{rank}}$)

Given a sequence of candidate retrieval scores $s_1, s_2, \ldots, s_N$:

1. **Empty / Single Item Boundary**:
   If $N \le 1$, return $0.0$ bits.

2. **Softmax Normalization** (Default, `method="softmax"`):
   Temperature $\tau > 0$ (if $\tau \le 0$, clamped to $10^{-6}$). To prevent numerical overflow, scores are shifted by $s_{\max} = \max(s)$:
   $$P(c_i) = \frac{\exp\left(\frac{s_i - s_{\max}}{\tau}\right)}{\sum_{j=1}^{N} \exp\left(\frac{s_j - s_{\max}}{\tau}\right)}$$

3. **MinMax Normalization** (`method="minmax"`):
   $$\tilde{s}_i = \begin{cases} \frac{1}{N} & \text{if } \max(s) = \min(s) \\ \frac{s_i - \min(s)}{\max(s) - \min(s)} & \text{otherwise} \end{cases}, \qquad P(c_i) = \frac{\tilde{s}_i}{\sum_{j=1}^{N} \tilde{s}_j}$$

4. **Sum / L1 / Linear Normalization** (`method` in `"sum"`, `"l1"`, `"linear"`):
   $$P(c_i) = \frac{s_i}{\sum_{j=1}^{N} s_j} \quad \left(\text{returns } 0.0 \text{ if } \sum s_i \le 0\right)$$

5. **Shannon Entropy Calculation**:
   $$H_{\text{rank}} = -\sum_{i: P(c_i) > 0} P(c_i) \log_2 P(c_i)$$

---

### 3.2 Lexical Unigram Entropy ($H_{\text{lexical}}$)

1. Lowercase and tokenize text via regex word boundary matching:
   $$\text{tokens} = \text{re.findall}(r"\w+", \text{text.lower()})$$
2. For empirical word counts $c(w)$ and total token count $T = \sum_w c(w)$:
   $$P(w) = \frac{c(w)}{T}$$
3. Shannon Entropy:
   $$H_{\text{lexical}} = -\sum_{w: P(w) > 0} P(w) \log_2 P(w)$$
   *Returns $0.0$ for empty, whitespace-only, or non-word strings.*

---

### 3.3 Graph Topology Entropy ($H_{\text{graph,rel}}$, $H_{\text{graph,deg}}$)

Accepts edge dictionaries with flexible key schemas (`type`/`relation`/`label`/`predicate` for relation type; `source`/`head`/`subject`/`src`/`from` and `target`/`tail`/`object`/`dst`/`to` for endpoints).

1. **Relation Type Entropy ($H_{\text{graph,rel}}$)**:
   For $|E|$ total edges and relation type counts $c(r)$:
   $$P(r) = \frac{c(r)}{|E|}, \qquad H_{\text{graph,rel}} = -\sum_{r} P(r) \log_2 P(r)$$

2. **Degree Entropy ($H_{\text{graph,deg}}$)**:
   Calculates undirected incidence degree $d(v)$ for each node $v \in V$. Total degree mass $D = \sum_v d(v)$:
   $$P(v) = \frac{d(v)}{D}, \qquad H_{\text{graph,deg}} = -\sum_{v} P(v) \log_2 P(v)$$

3. **Cypher-Style Text Parsing (`parse_graph_relations_from_text`)**:
   If raw relation dicts are missing, formatted Cypher context lines (e.g. `('Paper Title':Paper)-[CITES]->(work:doi:10.1:ExternalWork)`) are parsed into relation dicts using `_GRAPH_LINE_RE`.

---

### 3.4 Generation Entropy ($H_{\text{gen}}$) & Logit Resolution Hierarchy

For a sequence of $M$ generated token metadata dictionaries:
$$H_{\text{gen}} = \frac{1}{M} \sum_{k=1}^{M} h_k$$

Where per-token entropy $h_k$ is extracted by `_extract_single_token_entropy(t)` according to the following resolution hierarchy:
1. **Explicit Entropy Field**: Returns `t["entropy"]` in bits if present.
2. **Probability Distribution (`probs` or `top_probs`)**: Calculates $-\sum p \log_2 p$.
3. **Logprob Distribution (`top_logprobs` or `logprobs`)**: Exponentiates $p_i = \exp(lp_i)$, renormalizes $\sum p_i = 1$, calculates $-\sum p_i \log_2 p_i$.
4. **Single Token Surprisal (`logprob`)**: Converts log probability to surprisal in bits $-\log_2 P(t_k)$.
5. **Single Token Probability (`prob`)**: Returns $-\log_2 P(t_k)$.
6. **Fallback**: Returns $0.0$.

---

### 3.5 Citation Entropy ($H_{\text{citation}}$) & Token Alignment

1. **Citation Span Extraction (`find_citation_spans`)**:
   Uses regex matching to locate $[start, end)$ character offsets for citation markers including:
   - `[sciq_paper_X]`, `[Block X]`
   - `[1]`, `[1, 2]`, `[1-3]`
   - `[paper_1]`, `[doc_2]`, `[ref_1]`, `[id_1]`, `[source_1]`, `[Source_1]`, `[Источник: 1]`
   - DOIs (`10.xxxx/...`) and arXiv IDs (`arXiv:XXXX.XXXXX`)
   - Author-year patterns (`[Smith et al., 2020]`, `(Jones, 2019)`)
   Overlapping or contiguous spans are merged into sorted non-overlapping intervals.

2. **Token Bound Realignment (`align_tokens_info`)**:
   When raw model outputs undergo reasoning tag stripping or text cleanup (`full_text` $\neq$ `clean_text`), character offsets `[char_start, char_end)` are shifted by subtracting string offsets or re-aligned using token sequence search.

3. **Citation Entropy Calculation**:
   Tokens whose character intervals overlap any citation span are selected. Returns:
   $$H_{\text{citation}} = \text{compute\_generation\_entropy}(\text{citation\_tokens}), \qquad n_{\text{citation}} = |\text{citation\_tokens}|$$

---

### 3.6 Entropy Reduction ($\Delta H_{\text{gen}}$)

$$\Delta H_{\text{gen}} = H_{\text{gen}}^{B0} - H_{\text{gen}}^{RAG}$$

- **Positive $\Delta H_{\text{gen}}$**: RAG context successfully **reduced** output uncertainty relative to zero-shot generation.
- **Negative $\Delta H_{\text{gen}}$**: Context increased uncertainty (e.g. noisy, conflicting, or confusing context).
- **Query $B0$ Cache (`_ensure_b0_entropy`)**: Baseline $B0$ generation entropy is computed per query and persisted in `.cache/b0_entropy.json` to allow offline or pipelined $\Delta H_{\text{gen}}$ computation.

---

## 4. Data Models & Schemas

### 4.1 Pydantic Model Schema (`core/models.py`)

`BaselineOutput` encapsulates baseline outputs and entropy diagnostics:

```python
class BaselineOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str = "success"
    latency_sec: Optional[float] = 0.0
    retrieved_papers: List[str] = Field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    
    # Stage inputs for Shannon diagnostics
    pre_rerank_scores: Optional[List[float]] = None
    context_text: Optional[str] = None
    context_graph: Optional[str] = None
    graph_relations: Optional[List[Dict[str, Any]]] = None
    trimmed_text: Optional[str] = ""
    trimmed_graph: Optional[str] = ""
    enrichment_block: Optional[str] = ""
    generated_answer: Optional[str] = None
    
    # Diagnostic outputs
    metrics: Optional[StageMetrics] = None
    eval_metrics: Optional[Dict[str, Any]] = None
    shannon_diagnostics: Optional[Dict[str, Any]] = None
```

---

### 4.2 `shannon_diagnostics` Dictionary Specification

The `shannon_diagnostics` dictionary on each baseline output contains 10 standardized fields:

```json
{
  "h_rank_pre_rerank": 2.3219,
  "h_rank_post_rerank": 0.7912,
  "h_lexical_pre_trim": 5.8123,
  "h_lexical_post_trim": 5.4312,
  "h_graph_relation_type": 1.0000,
  "h_graph_degree": 1.9061,
  "h_gen": 1.2543,
  "h_citation": 0.3294,
  "n_citation_tokens": 11,
  "delta_h_gen": 0.4512
}
```

---

## 5. Pipeline Interactions & Execution Control

### 5.1 Retrieval Boundary Capture (`core/retrieval.py`)
During Stage 5 (Graph & Trimming stage), `run_staged_retrieval` saves pre- and post-processing stage inputs into the candidate context file (`retrieved_contexts.yaml`).

### 5.2 Live & Pipelined Generation (`core/generation.py` & `core/pipelined.py`)
- Reads component flag `shannon_enabled = components_settings.get("shannon_estimator_enabled", True)`.
- If enabled, invokes `_generate_with_logits_safe` to capture token logprobs.
- Computes $H_{\text{gen}}$ via `compute_generation_entropy` and $H_{\text{citation}}$ via `compute_citation_entropy`.
- Assembles retrieval fields via `assemble_retrieval_shannon_fields` and constructs `shannon_diagnostics`.
- Stores `shannon_diagnostics` under `metrics["shannon_diagnostics"]` and directly on `BaselineOutput`.

### 5.3 Configuration Enablement (`core/config.py`)
In `core/config.py`, `shannon_estimator_enabled` is registered as a RAG component setting (defaulting to `True` for all standard baselines `B0`–`B6`).

```yaml
rag_components:
  shannon_estimator_enabled: true
```

---

## 6. Aggregation, Analytics & Offline Backfill (`core/analytics.py`)

`core/analytics.py` processes evaluated benchmark outputs (`evaluation_results.yaml`):

1. **Offline Backfill**: If legacy result files lack explicit `shannon_diagnostics`, `analyze_metrics` inspects saved stage inputs (`pre_rerank_scores`, `context_text`, `graph_relations`, etc.) and calls `assemble_retrieval_shannon_fields` to backfill diagnostic fields on the fly.
2. **Statistical Summaries**: Computes statistical metrics (mean, min, max, stdev) for all Shannon diagnostic fields per baseline and computes the global `has_shannon` boolean flag.

---

## 7. Reporting & Console Presentation (`core/reporting.py`)

`core/reporting.py` renders Shannon diagnostics in both terminal and markdown formats:

### 7.1 Rich Terminal Table (`table_shannon`)
Renders an interactive Rich table displaying baseline comparative entropy progression:
`Baseline` | `H_rank (pre/post)` | `H_lexical (pre/post)` | `H_graph (rel/deg)` | `H_gen` | `H_citation` | `Cit Tokens` | `ΔH_gen`.

### 7.2 Markdown Report Section
Appends a dedicated section `## ⚛️ Shannon Estimator Diagnostics (Entropy in Bits)` containing structured markdown comparison tables for publication and automated reporting.

---

## 8. Academic Visualization Suite (`generate_scientific_visualizations.py`)

`generate_scientific_visualizations.py` produces high-resolution (300 DPI) publication-ready plots illustrating entropy phenomena:

1. **Figure 1A (`01a_generation_entropy_paradox.png`)**:
   *Panel 1: Generation Entropy vs. Hallucination Rate (The Overconfidence Paradox)*. Demonstrates how low $H_{\text{gen}}$ can correlate with overconfident hallucinations.
2. **Figure 1B (`01b_rank_compression_and_citations.png`)**:
   *Panel 1: Cross-Encoder Entropy Collapse ($H_{\text{rank}}$ pre $\rightarrow$ post)* showing score distribution sharpening.
   *Panel 2: Citation Tokens vs Citation Entropy ($H_{\text{citation}}$)* highlighting prompt-forced citation inflation.
3. **Figure 3 (`03_entropy_quality_tradeoff.png`)**:
   *Information Entropy ($H_{\text{gen}}$) vs. Semantic Accuracy Trade-off*. Scatter plot with bubble size proportional to Context Fillness.
4. **Figure 6 (`06_entropy_metrics_correlation_heatmap.png`)**:
   *Shannon Entropy & RAG Performance Metrics Correlation Heatmap*. Pearson correlation matrix ($r$) matching entropy metrics against Semantic Accuracy, Faithfulness, and Hallucination Rate.

---

## 9. Worked Numerical Examples

### Example A — Uniform Rank Entropy ($H = 2.0$ bits)
Four candidates with equal scores $[1.0, 1.0, 1.0, 1.0]$ under Softmax ($\tau=1.0$):
$$P(c_i) = \frac{1}{4} \quad \forall i, \qquad H_{\text{rank}} = -4 \times \left(\frac{1}{4} \log_2 \frac{1}{4}\right) = 2.0 \text{ bits}$$

### Example B — Reranking Entropy Collapse
Pre-rerank scores $[1, 1, 1, 1]$ ($H_{\text{pre}} = 2.0$) $\rightarrow$ Post-rerank scores $[10.0, 0.1, 0.1, 0.1]$ under Softmax ($\tau=1.0$):
$$P_0 \approx 0.99985, \quad P_{1,2,3} \approx 5.016 \times 10^{-5} \implies H_{\text{post}} \approx 0.0026 \text{ bits}$$
*Large positive $\Delta H_{\text{rank}}$ confirms reranker uncertainty collapse onto the top candidate.*

### Example C — Lexical Unigram Entropy ($H = 1.0$ bit)
Input string `"alpha beta alpha beta"`:
Counts: `alpha`: 2, `beta`: 2 ($T=4$). Probabilities: $P(\text{alpha}) = 0.5, P(\text{beta}) = 0.5$.
$$H_{\text{lexical}} = - (0.5 \log_2 0.5 + 0.5 \log_2 0.5) = 1.0 \text{ bit}$$

### Example D — Graph Relation and Degree Entropy
Given relations:
`[{"source": "A", "target": "B", "type": "cites"}, {"source": "B", "target": "C", "type": "cites"}, {"source": "C", "target": "A", "type": "supports"}, {"source": "A", "target": "D", "type": "supports"}]`
- **Relation Types**: 2 `cites`, 2 `supports` $\implies P = 0.5 \implies H_{\text{graph,rel}} = 1.0$ bit.
- **Node Degrees**: $d(A)=3, d(B)=2, d(C)=2, d(D)=1 \implies D_{total}=8$.
  $$H_{\text{graph,deg}} = -\left(\frac{3}{8}\log_2\frac{3}{8} + \frac{2}{8}\log_2\frac{2}{8} + \frac{2}{8}\log_2\frac{2}{8} + \frac{1}{8}\log_2\frac{1}{8}\right) \approx 1.9061 \text{ bits}$$

### Example E — Generation and Citation Entropy
Given generated tokens: `Results` ($h=0.5$), `in` ($h=0.2$), `[Block 3]` ($h=0.1$), `show` ($h=0.3$), `gravity` ($h=0.4$).
- **Generation Entropy**: $H_{\text{gen}} = \frac{0.5 + 0.2 + 0.1 + 0.3 + 0.4}{5} = 0.3000$ bits.
- **Citation Span**: `[Block 3]` (token 3). $H_{\text{citation}} = 0.1000$ bits, $n_{\text{citation}} = 1$.

---

## 10. Test Suite Verification

The Shannon Estimator functionality is verified by automated unit and integration tests:

- `tests/test_shannon_estimator.py`: Unit tests for closed-form rank entropy (softmax, minmax, sum), lexical unigram entropy, graph relation and star degree calculations, regex citation span extraction, token alignment, and assembled retrieval fields.
- `tests/test_shannon_diagnostics_wiring.py`: End-to-end integration tests verifying that `shannon_diagnostics` are correctly constructed, saved to YAML reports, and parsed during evaluation.
- `tests/test_occ_rag_logits_e2e.py`: Mock model logit generation test asserting proper token entropy calculation during LLM generation calls.
