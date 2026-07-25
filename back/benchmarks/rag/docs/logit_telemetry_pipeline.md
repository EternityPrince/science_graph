# Logit Telemetry & Shannon Diagnostics Pipeline

> [!NOTE]
> Architectural overview and mathematical specifications for logit/logprob extraction, sequence character-to-token alignment, uncertainty quantification, normalization routines, and data schemas in the RAG benchmark evaluation pipeline.

---

## 1. Pipeline Architecture

The logit telemetry pipeline provides end-to-end extraction, alignment, and mathematical calculation of confidence metrics and entropy diagnostics across RAG evaluation stages.

```
                    ┌──────────────────────────────────────────────────┐
                    │               LLM Generation Pass                │
                    │         _generate_with_logits_safe()             │
                    └────────────────────────┬─────────────────────────┘
                                             │ (raw_text, tokens_info)
                                             ▼
                    ┌──────────────────────────────────────────────────┐
                    │           Token Span & Char Alignment            │
                    │             build_token_char_spans()             │
                    └────────────────────────┬─────────────────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
         ┌───────────────────────────┐               ┌───────────────────────────┐
         │ Citation Onset Extraction │               │ Primary Sequence Telemetry│
         │ map_char_offset_to_idx()  │               │ compute_sequence_telemetry│
         │ compute_citation_onset_h  │               │ MSP, Margin, 1st-Token    │
         └─────────────┬─────────────┘               └─────────────┬─────────────┘
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             │
                                             ▼
                    ┌──────────────────────────────────────────────────┐
                    │          Secondary Base Scoring Pass             │
                    │           score_text_logprobs_base()             │
                    └────────────────────────┬─────────────────────────┘
                                             │
                                             ▼
                    ┌──────────────────────────────────────────────────┐
                    │            Log-Likelihood Ratio (CLR)            │
                    │        compute_clr(ll_rag, ll_base)              │
                    └──────────────────────────────────────────────────┘
```

### 1.1 Logit & Logprob Extraction (`_generate_with_logits_safe`)
During generation, LLM engines return generated text along with per-token metadata (`tokens_info`).
- Implemented in [_generate_with_logits_safe](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/generation.py#L58-L102) in [generation.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/generation.py).
- Safely handles mock engines, standard callable `generate_response_with_logits`, and fallback generation modes.
- Normalizes token dictionaries to ensure standardized keys: `token`, `logprob`, `top_logprobs`, `char_start`, and `char_end`.

### 1.2 Character-to-Token Index Mapping (`build_token_char_spans`, `map_char_offset_to_token_idx`)
Text-based pattern matches (such as citation brackets `[` or `Doc`) yield string character offsets that must be mapped to discrete token indices $t_c$ in `tokens_info`:
- Implemented in [build_token_char_spans](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/shannon_estimator.py#L446-L466) and [map_char_offset_to_token_idx](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/shannon_estimator.py#L469-L485) in [shannon_estimator.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/shannon_estimator.py).
- `build_token_char_spans`: Computes exact character start and end boundary tuples `(start_char, end_char)` for every token step.
- `map_char_offset_to_token_idx`: Efficiently locates the token index $t_c$ containing a given character offset $c$, supporting exact character alignment even when responses are trimmed or formatted.

### 1.3 Secondary Pass for Contextual Log-Likelihood Ratio (CLR) Ablation (`score_text_logprobs_base`)
To compute the Contextual Log-Likelihood Ratio ($CLR$), the response generated under the RAG prompt must be evaluated under a context-free baseline prompt ($B_0$):
- Implemented in [score_text_logprobs_base](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/generation.py#L105-L145) in [generation.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/generation.py).
- Evaluates the generated answer string under prompt `Question: {query}\nAnswer based on your general knowledge.` to compute $LL_{base} = \sum_{i} \log P_{B0}(w_i \mid w_{<i})$.
- $CLR$ is then derived as $CLR = LL_{rag} - LL_{base}$.

---

## 2. Metric Definitions & Mathematical Formulas

### 2.1 Maximum Softmax Probability (MSP)
- **Mathematical Formula**:
  $$MSP = \max_{i} p_i = \max_{i} \left( \frac{\exp(z_i - \max_k z_k)}{\sum_j \exp(z_j - \max_k z_k)} \right)$$
- **Intuitive Explanation**: Quantifies the peak probability assigned to the highest-scoring candidate token by numerically stable softmax normalization over raw logits $z$. The sequence average (`avg_msp`) summarizes model confidence across all generated tokens.
- **Range**: Strictly bounded within $(0.0, 1.0]$.
- **Code Pointer**: Implemented in [compute_msp](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/shannon_estimator.py#L533-L568) and [compute_softmax](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/shannon_estimator.py#L487-L530) in [shannon_estimator.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/shannon_estimator.py).

### 2.2 Logit Margin ($\Delta z_{1,2}$)
- **Mathematical Formula**:
  $$\Delta z_{1,2} = z_1 - z_2$$
  where $z_1, z_2$ are the top-1 and top-2 unnormalized logits (or logprobs) sorted in descending order ($z_1 \ge z_2$).
- **Intuitive Explanation**: Represents the raw absolute gap between the model's top choice and runner-up alternative. A large margin signifies clear, unambiguous selection, whereas a margin near $0.0$ indicates strong competition and uncertainty between top candidate tokens.
- **Range**: $\Delta z_{1,2} \ge 0.0$.
- **Code Pointer**: Implemented in [compute_logit_margin](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/shannon_estimator.py#L571-L601) in [shannon_estimator.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/shannon_estimator.py).

### 2.3 First-Token Decision Metrics
- **Mathematical Formulas**:
  - `first_token_margin`: $\Delta z_{1,2}^{(0)} = z_1^{(0)} - z_2^{(0)}$
  - `first_token_msp`: $MSP^{(0)} = \max_i p_i^{(0)}$
- **Intuitive Explanation**: Measures decision confidence specifically at sequence index $t=0$ (the very first generated token). This single-token checkpoint reveals whether the model makes an immediate, confident decision to answer, refuse, or cite, before emitting downstream sequence text.
- **Code Pointer**: Implemented in [compute_first_token_metrics](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/shannon_estimator.py#L616-L636) in [shannon_estimator.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/shannon_estimator.py).

### 2.4 Citation Onset Entropy ($H_{citation}$)
- **Mathematical Formula**:
  $$H_{citation} = -\sum_{i=1}^{|V|} p_i(t_c) \log_2 p_i(t_c)$$
  where $t_c$ is the token index where citation onset patterns (`\[|Doc`) match generated text character spans via `map_char_offset_to_token_idx`.
- **Intuitive Explanation**: Calculates the vocabulary Shannon entropy (in bits) precisely at the onset token index $t_c$ of a citation marker. High citation onset entropy indicates hesitation or confusion regarding which document or source block to attribute information to.
- **Code Pointer**: Implemented in [compute_citation_onset_entropy](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/shannon_estimator.py#L674-L712) in [shannon_estimator.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/shannon_estimator.py).

### 2.5 Contextual Log-Likelihood Ratio ($CLR$)
- **Mathematical Formula**:
  $$CLR = LL_{rag} - LL_{base}$$
  where:
  $$LL_{rag} = \sum_{t=1}^{N} \log P(w_t \mid w_{<t}, \text{Context}_{RAG})$$
  $$LL_{base} = \sum_{t=1}^{N} \log P(w_t \mid w_{<t}, \text{Prompt}_{B0})$$
- **Intuitive Explanation**: Quantifies the shift in log-likelihood for generating the exact response token sequence $W = (w_1, \dots, w_N)$ when provided with RAG context versus a zero-context prompt. Positive $CLR > 0$ demonstrates that retrieved context boosted model likelihood for the emitted response.
- **Code Pointer**: Implemented in [compute_log_likelihood](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/shannon_estimator.py#L715-L729) and [compute_clr](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/shannon_estimator.py#L732-L735) in [shannon_estimator.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/shannon_estimator.py).

---

## 3. Normalization Logic

### 3.1 ID Normalization (`normalize_id`)
To calculate accurate retrieval precision and recall metrics across heterogeneous databases, document and chunk identifiers are normalized according to five strict rules:

```
  Raw Input: "docs/paper_101.pdf#chunk_1"
      │
      ├─ 1. Lowercase: "docs/paper_101.pdf#chunk_1"
      ├─ 2. Strip Path Prefixes: "paper_101.pdf#chunk_1"
      ├─ 3. Strip Extensions & paper_ Prefix: "101#chunk_1"
      ├─ 4. Standardize Separators: "101_chunk_1"
      └─ 5. Truncate Chunk Markers: "101"
```

1. **Lowercase Enforcement**: Converts all string characters to lowercase.
2. **Path Prefix Stripping**: Removes standard directory prefixes (`docs/`, `data/`, `papers/`, `corpus/`, `dataset/`, `files/`, `./`, `../`, etc.), preserving DOIs starting with `10.`.
3. **Paper Prefix & File Extension Stripping**: Removes optional `paper_` prefixes and standard extensions (`.pdf`, `.txt`, `.md`, `.json`, `.html`).
4. **Separator Standardization**: Replaces `-`, `#`, `:`, whitespace, and `/` characters with standard underscores `_`.
5. **Chunk-to-Document Truncation**: Truncates chunk-level suffixes (`_chunk_1`, `_chunk3`, etc.) using regex `r'_chunk.*$'` to map chunk identifiers back to document-level entities.

- **Code Pointer**: Implemented in [normalize_id](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/metrics.py#L27-L61) in [metrics.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/metrics.py).

### 3.2 Cross-Component Score Normalization (`normalize_component_scores`)
Different retrieval components produce scores on radically different scales: BM25 produces unbounded positive floats, Dense Vector search produces cosine similarities $[-1.0, 1.0]$ or dot products, and Graph retrieval produces structural graph weights.

```python
def normalize_component_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    score_range = max_score - min_score
    return [(s - min_score) / score_range for s in scores]
```

- **Min-Max Scaling**: Rescales raw component scores into $[0.0, 1.0]$.
- **Single-Document / Constant Score Fallback**: If $\max(S) == \min(S)$ (e.g. single document returned or identical candidate scores), returns `1.0` for all scores to prevent division by zero or NaN values.
- **Code Pointer**: Implemented in [normalize_component_scores](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/retrieval.py#L21-L35) in [retrieval.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/retrieval.py).

---

## 4. Code Pointers & Verification Suite

| Component / Task | Source File | Key Functions / Classes |
| :--- | :--- | :--- |
| ID & Text Normalization | [metrics.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/metrics.py) | [normalize_id](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/metrics.py#L27-L61), `detect_abstention` |
| Shannon Estimators & Logit Math | [shannon_estimator.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/shannon_estimator.py) | `compute_softmax`, `compute_msp`, `compute_logit_margin`, `compute_first_token_metrics`, `compute_sequence_telemetry`, `compute_citation_onset_entropy`, `build_token_char_spans`, `map_char_offset_to_token_idx`, `compute_clr` |
| LLM Logit Generation & CLR Pass | [generation.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/generation.py) | `_generate_with_logits_safe`, `score_text_logprobs_base`, `run_query_on_baseline` |
| Retrieval & Score Normalization | [retrieval.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/retrieval.py) | `normalize_component_scores`, `run_staged_retrieval` |
| Pydantic Data Models | [models.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/models.py) | `ShannonDiagnostics`, `BaselineOutput`, `TestCaseOutput`, `ReportOutput` |
| Report Generation & Formatting | [reporting.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/reporting.py) | `print_rich_tables`, `generate_markdown_report`, `NEW_CSV_FIELDS`, `NEW_SUMMARY_HEADERS` |
| CLI Metric & Trace Aggregator | [parse_metrics.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/parse_metrics.py) | `MetricsParser`, `_collect_metrics_rows`, `_build_joined_data` |
| Verification Test Checkpoints | [test_verification_checkpoints.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/tests/test_verification_checkpoints.py) | Verification tests for Checkpoints 1–6 |

---

## 5. Data Schema Specifications

### 5.1 `ShannonDiagnostics` Pydantic Model Schema
Defined in [models.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/models.py#L39-L63):

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "ShannonDiagnostics",
  "type": "object",
  "properties": {
    "h_rank_pre_rerank": { "type": ["number", "null"], "default": 0.0 },
    "h_rank_post_rerank": { "type": ["number", "null"], "default": 0.0 },
    "h_lexical_pre_trim": { "type": ["number", "null"], "default": 0.0 },
    "h_lexical_post_trim": { "type": ["number", "null"], "default": 0.0 },
    "h_graph_relation_type": { "type": ["number", "null"], "default": 0.0 },
    "h_graph_degree": { "type": ["number", "null"], "default": 0.0 },
    "h_gen": { "type": ["number", "null"], "default": 0.0 },
    "h_citation": { "type": ["number", "null"], "default": 0.0 },
    "n_citation_tokens": { "type": ["integer", "null"], "default": 0 },
    "delta_h_gen": { "type": ["number", "null"], "default": 0.0 },
    "msp": { "type": ["number", "null"] },
    "avg_msp": { "type": ["number", "null"] },
    "logit_margin": { "type": ["number", "null"] },
    "avg_logit_margin": { "type": ["number", "null"] },
    "first_token_margin": { "type": ["number", "null"] },
    "first_token_msp": { "type": ["number", "null"] },
    "citation_entropy": { "type": ["number", "null"] },
    "ll_rag": { "type": ["number", "null"] },
    "ll_base": { "type": ["number", "null"] },
    "clr": { "type": ["number", "null"] }
  }
}
```

### 5.2 `BaselineOutput` Pydantic Model Schema (Telemetry Segment)
Defined in [models.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/models.py#L65-L97):

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "BaselineOutput",
  "type": "object",
  "properties": {
    "status": { "type": "string", "default": "success" },
    "latency_sec": { "type": ["number", "null"], "default": 0.0 },
    "retrieved_papers": { "type": "array", "items": { "type": "string" } },
    "retrieved_chunks": { "type": "array", "items": { "$ref": "#/$defs/RetrievedChunk" } },
    "generated_answer": { "type": ["string", "null"] },
    "shannon_diagnostics": { "$ref": "#/$defs/ShannonDiagnostics" },
    "msp": { "type": ["number", "null"] },
    "avg_msp": { "type": ["number", "null"] },
    "logit_margin": { "type": ["number", "null"] },
    "avg_logit_margin": { "type": ["number", "null"] },
    "first_token_margin": { "type": ["number", "null"] },
    "first_token_msp": { "type": ["number", "null"] },
    "citation_entropy": { "type": ["number", "null"] },
    "ll_rag": { "type": ["number", "null"] },
    "ll_base": { "type": ["number", "null"] },
    "clr": { "type": ["number", "null"] }
  }
}
```

### 5.3 Per-Query Joined CSV / Dataframe Schema (`per_query_joined.csv`)
Emitted by [parse_metrics.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/parse_metrics.py#L521-L543):

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `query_id` | string | Unique test case identifier |
| `baseline` | string | Baseline configuration name (e.g. `B0`, `B6`, `CUSTOM`) |
| `category` | string | Benchmark question domain category |
| `is_answerable` | boolean | True if question can be answered from corpus |
| `predicted_abstained` | boolean | True if model refused to answer |
| `answerability_outcome`| string | Confusion matrix outcome (`TP`, `FP`, `TN`, `FN`) |
| `retrieval_recall` | float | Paper retrieval recall |
| `context_precision` | float | Context mean average precision |
| `faithfulness` | float | LLM-judge context faithfulness score |
| `answer_relevance` | float | LLM-judge answer relevance score |
| `semantic_accuracy` | float | Cosine similarity to golden answer embedding |
| `latency_sec` | float | End-to-end execution latency in seconds |
| `msp` / `avg_msp` | float | Maximum softmax probability across output sequence |
| `logit_margin` / `avg_logit_margin` | float | Top-1 vs top-2 logit margin gap ($\Delta z_{1,2}$) |
| `first_token_margin` | float | Top-1 vs top-2 logit margin gap at token step 0 |
| `first_token_msp` | float | Maximum softmax probability at token step 0 |
| `citation_entropy` | float | Average Shannon entropy at citation onset markers |
| `ll_rag` | float | Total sequence log-likelihood under RAG context |
| `ll_base` | float | Total sequence log-likelihood under base $B_0$ prompt |
| `clr` | float | Contextual Log-Likelihood Ratio ($LL_{rag} - LL_{base}$) |
