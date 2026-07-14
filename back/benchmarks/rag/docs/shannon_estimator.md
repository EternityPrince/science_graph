# Shannon Estimator: Information-Theoretic Diagnostics for RAG Baselines

The **Shannon Estimator** is a component-level diagnostic layer designed to measure information uncertainty, vocabulary richness, graph structural complexity, predictive token entropy, citation uncertainty, and predictive entropy reduction ($\Delta H_{\text{gen}}$) across individual components of the RAG pipeline.

Unlike end-to-end evaluation metrics (such as recall, precision, or LLM-as-a-judge scores), the Shannon Estimator acts as a **component estimator**, measuring how uncertainty changes at each stage of processing (retrieval, reranking, context assembly/trimming, graph expansion, and generation).

---

## 1. Core Entropy Diagnostics & Mathematical Formulas

All Shannon entropy values are expressed in **bits** ($\log_2$):

1. **Retrieval / Rank Entropy ($H_{\text{rank}}$)**:
   - **Formula**:
     $$H_{\text{rank}} = -\sum_{i=1}^K P(c_i) \log_2 P(c_i)$$
     where candidate probabilities $P(c_i)$ are obtained via softmax over relevance scores $s_i$:
     $$P(c_i) = \frac{\exp(s_i / \tau)}{\sum_{j=1}^K \exp(s_j / \tau)}$$
   - **Diagnostic Insight**: Sharp probability distributions yield low entropy (high confidence in top candidates); flat distributions yield higher entropy approaching $\log_2(K)$ bits. Measuring $H_{\text{rank\_pre}}$ (before reranker) vs $H_{\text{rank\_post}}$ (after reranker) quantifies how much candidate uncertainty the Cross-Encoder eliminates.

2. **Lexical Context Entropy ($H_{\text{lexical}}$)**:
   - **Formula**:
     $$H_{\text{lexical}} = -\sum_{t \in V} P(t) \log_2 P(t)$$
     where $P(t) = \frac{\text{count}(t)}{N}$ is the unigram token frequency distribution.
   - **Diagnostic Insight**: Tracks vocabulary diversity and information density before vs after context trimming ($H_{\text{lexical\_pre}}$ vs $H_{\text{lexical\_post}}$).

3. **Graph Topology Entropy ($H_{\text{graph}}$)**:
   - **Formula**:
     $$H_{\text{rel}} = -\sum_{r \in R} P(r) \log_2 P(r), \quad H_{\text{deg}} = -\sum_{v \in V} P(v) \log_2 P(v)$$
   - **Diagnostic Insight**: Measures relation diversity ($H_{\text{rel}}$) and node connection degree distribution ($H_{\text{deg}}$) in Graph-RAG subgraphs (B5/B6).

4. **Generation / Predictive Entropy ($H_{\text{gen}}$)**:
   - **Formula**:
     $$H_t = -\sum_{v \in V} P_t(v) \log_2 P_t(v), \quad H_{\text{gen}} = \frac{1}{T}\sum_{t=1}^T H_t$$
     where $P_t(v) = \text{softmax}(\text{logits}_t)_v$ from local `OCC-RAG-1.7B` via `mlx`.
   - **Diagnostic Insight**: Quantifies average next-token uncertainty across the model's generated response.

5. **Citation-Specific Token Entropy ($H_{\text{citation}}$)**:
   - **Formula**: Average Shannon entropy $H_t$ restricted to tokens $t \in S_{\text{citation}}$ located inside citation marker spans.
   - **Diagnostic Insight**: Isolates model uncertainty specifically during paper/chunk citation generation, serving as a diagnostic for citation fidelity.

6. **Predictive Entropy Reduction ($\Delta H_{\text{gen}}$)**:
   - **Formula**:
     $$\Delta H_{\text{gen}} = H_{\text{gen}}^{\text{B0}} - H_{\text{gen}}^{\text{RAG}}$$
   - **Diagnostic Insight**: Measures how many bits of predictive uncertainty are eliminated when the model is supplied with retrieved context ($C$) versus zero-shot generation ($B0$).

---

## 2. How Citation Token Entropy is Calculated

Citation fidelity is often the most fragile metric in RAG evaluation. To compute $H_{\text{citation}}$ without adding runtime latency or secondary LLM calls, the estimator employs a deterministic **character-to-token span alignment algorithm**.

### Step-by-Step Character-to-Token Alignment:

1. **Token Streaming & Offset Recording**:
   During answer generation, `MlxLLMEngine.generate_response_with_logits` streams generated tokens. For each token $t_i$, it computes token entropy $H_{t_i}$ via `mx.softmax(logprobs)` and records character offsets $[char\_start_i, char\_end_i)$ relative to the accumulated text string.

   *Example token sequence*:
   - Token 0 (`"According"`): range `[0, 9)`, entropy `1.20`
   - Token 1 (`" to"`): range `[9, 12)`, entropy `0.45`
   - Token 2 (`" ["`): range `[12, 14)`, entropy `1.80`
   - Token 3 (`"sciq_paper_1"`): range `[14, 26)`, entropy `2.10`
   - Token 4 (`"]"`): range `[26, 27)`, entropy `0.30`

2. **Regex Citation Span Extraction**:
   Once the full answer string is constructed, `find_citation_spans(generated_text)` runs a suite of regular expression patterns to identify citation boundary ranges $[start, end)$:
   - `\[sciq_paper_\w+\]`
   - `\[Block[\s_]?\w+\]`
   - Numbered citations: `\[\d+\]`, `\[\d+,\s*\d+\]`, `\[\d+-\d+\]`
   - Generic paper markers: `\[paper_\w+\]`, `\[doc_\w+\]`
   - DOIs: `10.\d{4,9}/[-._;()/:A-Z0-9]+`
   - Author-year patterns: `\([A-Z][a-z]+(?:\s+et\s+al\.)?,\s+\d{4}\)`

   *Extracted span*: `[12, 27)` matching `"[sciq_paper_1]"`.

3. **Overlap Alignment**:
   For each token $t_i$, the estimator checks whether its character range $[char\_start_i, char\_end_i)$ intersects any detected citation span $[start, end)$:
   $$\text{Overlap}(t_i) = \Big(char\_end_i > start\Big) \;\land\; \Big(char\_start_i < end\Big)$$
   Tokens matching this criterion (Token 2 `" ["`, Token 3 `"sciq_paper_1"`, and Token 4 `"]"`) are collected into the citation subset $S_{\text{citation}}$.

4. **Raw Uncertainty Pre-Repair Calculation**:
   The average entropy $H_{\text{citation}}$ is computed over $S_{\text{citation}}$ **before citation repair (B5)** so that true model output uncertainty is preserved.

---

## 3. Configuration & Enabling/Disabling

The Shannon Estimator is controlled by the configuration option `shannon_estimator_enabled: bool` (default `True`).

In `config.yaml`:
```yaml
rag_components:
  shannon_estimator_enabled: true
```

Or programmatically:
```python
config.data["rag_components"]["shannon_estimator_enabled"] = True
```

When set to `False`:
- Standard text generation is used (`generate_response`).
- Logit streaming and per-token entropy calls are bypassed for maximum performance.

---

## 4. Diagnostics Interpretation across Baselines (B4, B5, B6)

- **B4 (Hybrid + Reranker)**:
  Compare $H_{\text{rank\_pre}}$ vs $H_{\text{rank\_post}}$. A significant entropy reduction after reranking indicates that the Cross-Encoder successfully concentrated candidate probability onto relevant top chunks.

- **B5 (Hybrid + Graph + Trimming)**:
  Examine $H_{\text{graph\_relation}}$ and $H_{\text{graph\_degree}}$ to evaluate subgraph structural variety. Compare $H_{\text{lexical\_pre}}$ vs $H_{\text{lexical\_post}}$ to ensure context trimming removes redundant vocabulary without dropping essential information.

- **B6 (Full Pipeline)**:
  Track $H_{\text{citation}}$ alongside $\Delta H_{\text{gen}}$. Lower $H_{\text{citation}}$ indicates the model generates paper citations with high confidence, correlating with improved citation fidelity scores.
