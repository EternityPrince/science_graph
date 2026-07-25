# Logit Telemetry & Confidence Diagnostics

> [!NOTE]
> Technical manual for token logprob extraction, character-to-token span alignment, uncertainty quantification, normalization routines, and data schemas in the RAG evaluation pipeline.

---

## 1. Architectural Architecture & Logit Pipeline Flow

The logit telemetry pipeline provides end-to-end extraction, alignment, and mathematical calculation of confidence metrics across generation stages.

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

### 1.1 Logit & Logprob Extraction ([core/generation.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/generation.py))
During generation, LLM engines return generated text along with per-token metadata (`tokens_info`):
- Implemented in [_generate_with_logits_safe](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/generation.py#L58-L102).
- Safely handles mock engines, standard callable `generate_response_with_logits`, and fallback generation modes.
- Standardizes token dictionaries to ensure uniform keys: `token`, `logprob`, `top_logprobs`, `char_start`, and `char_end`.

### 1.2 Character-to-Token Index Mapping ([core/shannon_estimator.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/shannon_estimator.py))
Text-based pattern matches (such as citation brackets `[` or `Doc`) yield string character offsets that must be mapped to discrete token indices $t_c$ in `tokens_info`:
- Implemented in [build_token_char_spans](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/shannon_estimator.py#L446-L466) and [map_char_offset_to_token_idx](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/shannon_estimator.py#L469-L485).
- `build_token_char_spans`: Computes exact character start and end boundary tuples `(start_char, end_char)` for every token step.
- `map_char_offset_to_token_idx`: Efficiently locates the token index $t_c$ containing a given character offset $c$, supporting exact character alignment even when responses are trimmed or formatted.

### 1.3 Secondary Pass for Contextual Log-Likelihood Ratio (CLR) Ablation ([core/generation.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/generation.py))
To compute the Contextual Log-Likelihood Ratio ($CLR$), the response generated under the RAG prompt must be evaluated under a context-free baseline prompt ($B_0$):
- Implemented in [score_text_logprobs_base](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/generation.py#L105-L145).
- Evaluates the generated answer string under prompt `Question: {query}\nAnswer based on your general knowledge.` to compute:
  $$LL_{\text{base}} = \sum_{t=1}^N \log P_{B0}(w_t \mid w_{<t})$$
- $CLR$ is then derived as:
  $$CLR = LL_{\text{rag}} - LL_{\text{base}}$$

---

## 2. Metric Definitions & Mathematical Formulas

### 2.1 Maximum Softmax Probability ($MSP$)

- **Mathematical Formula**:
  $$MSP = \max_{i} p_i = \max_{i} \left( \frac{\exp(z_i - \max_k z_k)}{\sum_j \exp(z_j - \max_k z_k)} \right)$$
- **Intuitive Explanation**: Quantifies the peak probability assigned to the highest-scoring candidate token by numerically stable softmax normalization over raw logits $z$. The sequence average (`avg_msp`) summarizes overall model confidence across generated output sequences.
- **Bounded Range**: $MSP \in (0.0, 1.0]$.
- **Code Pointer**: Implemented in `compute_msp` and `compute_softmax` in [shannon_estimator.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/shannon_estimator.py).

### 2.2 Logit Margin ($\Delta z_{1,2}$)

- **Mathematical Formula**:
  $$\Delta z_{1,2} = z_1 - z_2$$
  where $z_1 \ge z_2$ are the top-1 and top-2 unnormalized logits (or logprobs) sorted in descending order.
- **Intuitive Explanation**: Represents the raw absolute gap between the model's top choice and runner-up alternative. A large margin signifies clear, unambiguous selection, whereas a margin near $0.0$ indicates strong competition and uncertainty between top candidate tokens.
- **Bounded Range**: $\Delta z_{1,2} \ge 0.0$.
- **Code Pointer**: Implemented in `compute_logit_margin` in [shannon_estimator.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/shannon_estimator.py).

### 2.3 First-Token Decision Metrics

- **Mathematical Formulas**:
  - `first_token_margin`: $\Delta z_{1,2}^{(0)} = z_1^{(0)} - z_2^{(0)}$
  - `first_token_msp`: $MSP^{(0)} = \max_i p_i^{(0)}$
- **Intuitive Explanation**: Measures decision confidence specifically at sequence index $t=0$ (the very first generated token). This single-token checkpoint reveals whether the model makes an immediate, confident decision to answer, refuse, or cite, before emitting downstream sequence text.
- **Code Pointer**: Implemented in `compute_first_token_metrics` in [shannon_estimator.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/shannon_estimator.py).

### 2.4 Citation Onset Entropy ($H_{\text{citation}}$)

- **Mathematical Formula**:
  $$H_{\text{citation}} = -\sum_{i=1}^{|V|} p_i(t_c) \log_2 p_i(t_c)$$
  where $t_c$ is the token index where citation onset patterns (`\[|Doc`) match generated text character spans via `map_char_offset_to_token_idx`.
- **Intuitive Explanation**: Calculates vocabulary Shannon entropy (in bits) precisely at the onset token index $t_c$ of a citation marker. High citation onset entropy indicates hesitation or confusion regarding which document or source block to attribute information to.
- **Code Pointer**: Implemented in `compute_citation_onset_entropy` in [shannon_estimator.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/shannon_estimator.py).

### 2.5 Contextual Log-Likelihood Ratio ($CLR$)

- **Mathematical Formula**:
  $$CLR = LL_{\text{rag}} - LL_{\text{base}}$$
  where:
  $$LL_{\text{rag}} = \sum_{t=1}^{N} \log P(w_t \mid w_{<t}, \text{Context}_{\text{RAG}})$$
  $$LL_{\text{base}} = \sum_{t=1}^{N} \log P(w_t \mid w_{<t}, \text{Prompt}_{B0})$$
- **Intuitive Explanation**: Quantifies the shift in log-likelihood for generating token sequence $W = (w_1, \dots, w_N)$ when provided with RAG context versus a zero-context prompt. Positive $CLR > 0$ demonstrates that retrieved context boosted model likelihood for the emitted response.
- **Code Pointer**: Implemented in `compute_log_likelihood` and `compute_clr` in [shannon_estimator.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/shannon_estimator.py).

---

## 3. Data Model Schemas

### 3.1 `ShannonDiagnostics` Model Schema ([core/models.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/models.py))

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

---

## 🔗 Related Documentation
- [Integrated Architecture Reference](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/architecture/integrated_pipeline.md)
- [Shannon Estimator Manual](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/telemetry/shannon_estimator.md)
- [Non-Parametric Statistical Testing Framework](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/statistics/statistical_testing_framework.md)
