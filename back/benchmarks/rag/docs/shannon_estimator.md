# Shannon Estimator: Math, Formulas, and Worked Examples

Pure mathematical reference for entropy diagnostics used in the Science Graph RAG benchmark suite. All values are reported in **bits** (\(\log_2\)).

Implementation: `back/benchmarks/rag/core/shannon_estimator.py`  
Unit tests: `back/benchmarks/rag/tests/test_shannon_estimator.py`  
Pipeline wiring / orchestration: [pipeline_orchestration.md](./pipeline_orchestration.md)

This document is **formula- and example-focused**. For how diagnostics are staged in retrieve → generate → evaluate, see the pipeline doc and the short wiring notes at the end.

---

## 1. Metric catalog

| Metric | Symbol | Function | Meaning |
|--------|--------|----------|---------|
| Rank entropy | \(H_{\text{rank}}\) | `compute_rank_entropy` | Uncertainty over candidate relevance scores |
| Lexical entropy | \(H_{\text{lexical}}\) | `compute_lexical_entropy` | Unigram diversity of context text |
| Graph relation entropy | \(H_{\text{graph,rel}}\) | `compute_graph_entropy` → `relation_type_entropy` | Diversity of edge relation types |
| Graph degree entropy | \(H_{\text{graph,deg}}\) | `compute_graph_entropy` → `degree_entropy` | Diversity of node degree mass |
| Generation entropy | \(H_{\text{gen}}\) | `compute_generation_entropy` | Mean per-token predictive entropy |
| Citation entropy | \(H_{\text{citation}}\) | `compute_citation_entropy` | Mean entropy of tokens inside citation spans |
| Entropy reduction | \(\Delta H_{\text{gen}}\) | `compute_entropy_reduction` | \(H_{\text{gen}}^{B0} - H_{\text{gen}}^{RAG}\) |

Assembled retrieval fields (pre/post rank & lexical + graph) are returned by `assemble_retrieval_shannon_fields`.

---

## 2. Formulas

### 2.1 Rank entropy \(H_{\text{rank}}\)

Given candidate scores \(s_1,\ldots,s_N\):

**Empty / singleton:** if \(N \le 1\), \(H_{\text{rank}} = 0\).

**Softmax** (default, temperature \(\tau > 0\); \(\tau \le 0\) is clamped to \(10^{-6}\)):

\[
P(c_i) = \frac{\exp\bigl((s_i - s_{\max}) / \tau\bigr)}{\sum_j \exp\bigl((s_j - s_{\max}) / \tau\bigr)},
\qquad
H_{\text{rank}} = -\sum_{i=1}^{N} P(c_i)\,\log_2 P(c_i)
\]

(The code subtracts \(s_{\max}\) for numerical stability; this does not change \(P\).)

**Minmax** (`method="minmax"`):

\[
\tilde{s}_i =
\begin{cases}
1/N & \text{if }\max s = \min s \\
\dfrac{s_i - \min s}{\max s - \min s} & \text{otherwise}
\end{cases}
\qquad
P(c_i) = \frac{\tilde{s}_i}{\sum_j \tilde{s}_j}
\]

then the same \(H = -\sum P\log_2 P\).

**Sum / L1 / linear** (`method` in `sum`, `l1`, `linear`):

\[
P(c_i) = \frac{s_i}{\sum_j s_j}
\quad\text{(returns 0 if }\sum s \le 0\text{)}
\]

Interpretation: flat scores → high \(H_{\text{rank}}\) (up to \(\log_2 N\)); peaked scores → low \(H_{\text{rank}}\). Pre-rerank vs post-rerank \(\Delta H_{\text{rank}}\) measures ranking sharpening.

---

### 2.2 Lexical entropy \(H_{\text{lexical}}\)

1. Tokenize with `\w+` on lowercased text (`re.findall(r"\w+", text.lower())`).
2. Let \(c(t)\) be unigram counts, \(T = \sum_t c(t)\).
3. \(P(t) = c(t)/T\).

\[
H_{\text{lexical}} = -\sum_{t} P(t)\,\log_2 P(t)
\]

Empty text, whitespace-only, or no `\w+` matches → \(0\).

---

### 2.3 Graph relation and degree entropy

Input: list of edge dicts with flexible keys:

- type: `type` | `relation` | `label` | `predicate` (default `"unknown"`)
- source: `source` | `head` | `subject` | `src` | `from`
- target: `target` | `tail` | `object` | `dst` | `to`

**Relation-type entropy** over \(E\) edges:

\[
P(r) = \frac{\#\{\text{edges of type } r\}}{E},
\qquad
H_{\text{graph,rel}} = -\sum_r P(r)\,\log_2 P(r)
\]

**Degree entropy**: undirected degree mass — each edge increments degree of both endpoints by 1. Let \(d(v)\) be that count, \(D = \sum_v d(v)\) (equals \(2E\) when every edge has both ends):

\[
P(v) = \frac{d(v)}{D},
\qquad
H_{\text{graph,deg}} = -\sum_v P(v)\,\log_2 P(v)
\]

Empty relation list → both zeros. Structured relations are preferred; otherwise `parse_graph_relations_from_text` parses Cypher-style lines from graph context text.

---

### 2.4 Generation entropy \(H_{\text{gen}}\)

For a list of token info dicts of length \(M\):

\[
H_{\text{gen}} = \frac{1}{M}\sum_{k=1}^{M} h_k
\]

where per-token \(h_k\) is extracted in priority order by `_extract_single_token_entropy`:

1. Explicit `entropy` field (bits).
2. Probability mass over `probs` / `top_probs` → \(-\sum p\log_2 p\).
3. `top_logprobs` / `logprobs` → exp, renormalize, then Shannon.
4. Scalar `logprob` → surprisal in bits (\(-\log_2 p\)).
5. Scalar `prob` → \(-\log_2 p\).
6. Else \(0\).

Empty list → \(0\).

---

### 2.5 Citation entropy \(H_{\text{citation}}\)

1. `find_citation_spans(text)` finds character spans for markers such as `[sciq_paper_X]`, `[Block N]`, `[1, 2]`, DOIs, arXiv IDs, author-year, etc. Overlapping/touching spans are merged.
2. Tokens whose character ranges overlap any span are selected.
3. \(H_{\text{citation}} =\) mean entropy of those tokens (same as generation mean); also returns \(n_{\text{citation}}\) token count.

No spans or no overlapping tokens → \((0.0, 0)\).

Prefer measuring on **raw** model text before citation repair so \(H_{\text{citation}}\) reflects model uncertainty, not post-processing.

---

### 2.6 Entropy reduction \(\Delta H_{\text{gen}}\)

\[
\Delta H_{\text{gen}} = H_{\text{gen}}^{B0} - H_{\text{gen}}^{RAG}
= \texttt{compute_entropy_reduction}(h_{b0}, h_{rag})
\]

- Positive \(\Delta H\): RAG context **reduced** predictive uncertainty relative to zero-shot B0.
- Negative \(\Delta H\): RAG made the model **more** uncertain than B0 (noisy/conflicting context).
- Either argument `None` → \(0\).
- By definition B0 stores `delta_h_gen = 0.0` (it is its own baseline).

---

## 3. Worked numerical examples

### Example A — Uniform rank entropy (= 2.0 bits)

**Input:** four equal scores, softmax, \(\tau = 1\):

\[
s = [1.0,\, 1.0,\, 1.0,\, 1.0]
\]

**Calculation:**

\[
P(c_i) = \frac{1}{4} \quad \forall i
\]

\[
H_{\text{rank}}
= -4 \cdot \Bigl(\tfrac{1}{4}\log_2\tfrac{1}{4}\Bigr)
= -4 \cdot \bigl(\tfrac{1}{4}\cdot(-2)\bigr)
= 2.0
\]

Also \(H_{\text{rank}} = \log_2 N = \log_2 4 = 2\) for any uniform distribution on \(N\) candidates.

**Output:** `compute_rank_entropy([1,1,1,1], method="softmax")` → **2.0 bits**.

Matches unit test expectation `math.log2(4)`.

---

### Example B — Peaked rank entropy (sharpening after rerank)

**Input:**

\[
s_{\text{pre}} = [1,1,1,1],\quad
s_{\text{post}} = [10.0,\, 0.1,\, 0.1,\, 0.1],\quad
\tau = 1
\]

**Pre:** from Example A → \(H_{\text{pre}} = 2.0\).

**Post:** let \(s_{\max}=10\),

\[
e_0 = e^{0} = 1,\quad
e_{1,2,3} = e^{-9.9} \approx 5.017\times 10^{-5}
\]

\[
Z \approx 1 + 3\cdot 5.017\times 10^{-5} \approx 1.0001505
\]

\[
P_0 \approx 0.99985,\quad P_{1,2,3} \approx 5.016\times 10^{-5}
\]

\[
H_{\text{post}} \approx -P_0\log_2 P_0 - 3\,P_1\log_2 P_1 \approx 0.0026\ \text{bits}
\]

(order of magnitude; exact value from `compute_rank_entropy` is \(\ll 2\)).

**Output:** \(H_{\text{pre}} \gg H_{\text{post}}\) → large positive rank \(\Delta\): reranker collapsed uncertainty onto one candidate.

---

### Example C — Lexical unigram entropy (= 1.0 bit)

**Input:**

```text
alpha beta alpha beta
```

**Calculation:**

| token | count | \(P(t)\) |
|-------|------:|--------:|
| alpha | 2 | 0.5 |
| beta  | 0.5 | 0.5 |

\[
H_{\text{lexical}}
= -2 \cdot (0.5 \log_2 0.5)
= -2 \cdot (0.5 \cdot (-1))
= 1.0
\]

**Output:** `compute_lexical_entropy("alpha beta alpha beta")` → **1.0 bit**.

Punctuation / case variants (`"Alpha, beta! Alpha; beta."`) tokenize the same way → still **1.0 bit**.

**Contrast:** `"test test test test"` → single type, \(P=1\) → **0.0 bits**.

---

### Example D — Graph relation and degree entropy

**Input relations:**

```python
[
  {"source": "A", "target": "B", "type": "cites"},
  {"source": "B", "target": "C", "type": "cites"},
  {"source": "C", "target": "A", "type": "supports"},
  {"head": "A", "tail": "D", "relation": "supports"},  # alt keys
]
```

**Relation types:** cites×2, supports×2 → \(P=1/2\) each.

\[
H_{\text{graph,rel}} = -2\cdot\bigl(\tfrac12\log_2\tfrac12\bigr) = 1.0\ \text{bit}
\]

**Degrees** (each edge +1 to both ends):

| node | degree |
|------|-------:|
| A | 3 (edges to B, from C, to D) |
| B | 2 |
| C | 2 |
| D | 1 |

\(D = 3+2+2+1 = 8\).

\[
\begin{align*}
H_{\text{graph,deg}}
&= -\Bigl(
  \tfrac{3}{8}\log_2\tfrac{3}{8}
  + 2\cdot\tfrac{2}{8}\log_2\tfrac{2}{8}
  + \tfrac{1}{8}\log_2\tfrac{1}{8}
\Bigr)\\
&= -\Bigl(
  0.375\cdot\log_2 0.375
  + 0.5\cdot\log_2 0.25
  + 0.125\cdot\log_2 0.125
\Bigr)\\
&\approx -(-0.5306 - 1.0 - 0.375)
\approx 1.906\ \text{bits}
\end{align*}
\]

**Output:** `compute_graph_entropy(relations)` →

```python
{"relation_type_entropy": 1.0, "degree_entropy": ≈1.906}
```

---

### Example E — Generation and citation entropy

**Tokens:**

| token text | char range | entropy (bits) |
|------------|------------|---------------:|
| `Results ` | [0, 8) | 0.5 |
| `in ` | [8, 11) | 0.2 |
| `[Block 3]` | [11, 20) | 0.1 |
| ` show ` | [20, 26) | 0.3 |
| `gravity ` | [26, 34) | 0.4 |

**Full text:** `"Results in [Block 3] show gravity "`

\[
H_{\text{gen}} = \frac{0.5+0.2+0.1+0.3+0.4}{5} = \frac{1.5}{5} = 0.3
\]

Citation span for `[Block 3]` is `[11, 20)`. Only that token overlaps →

\[
H_{\text{citation}} = 0.1,\qquad n_{\text{citation}} = 1
\]

**Also:** mean of `[{"entropy": 1.5}, {"entropy": 0.5}]` → **1.0 bit**.  
Fair coin over vocab via `probs: {a:0.5, b:0.5}` → **1.0 bit**.

---

### Example F — \(\Delta H_{\text{gen}}\) (entropy reduction)

**Inputs:**

\[
H_{\text{gen}}^{B0} = 2.40,\qquad H_{\text{gen}}^{B5} = 1.15
\]

**Calculation:**

\[
\Delta H_{\text{gen}} = 2.40 - 1.15 = 1.25\ \text{bits}
\]

**Output:** `compute_entropy_reduction(2.40, 1.15)` → **1.25**.

Interpretation: supplying B5 context removed **1.25 bits** of average per-token uncertainty vs zero-shot.

**Edge cases:**

| \(h_{b0}\) | \(h_{rag}\) | \(\Delta H\) |
|-----------:|------------:|-------------:|
| 2.0 | 2.0 | 0.0 |
| 1.0 | 1.8 | −0.8 (RAG more uncertain) |
| `None` | 1.0 | 0.0 |
| 1.0 | `None` | 0.0 |

On B0 itself the pipeline records `delta_h_gen = 0.0` and caches `h_gen` per query for later RAG rows.

---

### Example G — Assembled pre/post fields (integration)

```python
from core.shannon_estimator import assemble_retrieval_shannon_fields

fields = assemble_retrieval_shannon_fields(
    pre_scores=[1.0, 1.0, 1.0, 1.0],       # Example A → 2.0
    post_scores=[10.0, 0.1, 0.1, 0.1],      # Example B → ≪ 2
    pre_text="alpha beta gamma delta epsilon zeta eta theta",
    post_text="alpha alpha alpha beta",     # lower lexical H
    relations=[
        {"source": "A", "target": "B", "type": "CITES"},
        {"source": "B", "target": "C", "type": "AUTHORED"},
    ],
)
# keys: h_rank_pre_rerank, h_rank_post_rerank,
#       h_lexical_pre_trim, h_lexical_post_trim,
#       h_graph_relation_type, h_graph_degree  (rounded to 4 decimals)
```

**Fallback rule:** if `pre_scores` / `pre_text` are missing, pre metrics **copy post** (honest “stage not captured”, not invented structure).

---

## 4. Public API (`core/shannon_estimator.py`)

| Function | Signature (abridged) | Returns |
|----------|----------------------|---------|
| `compute_rank_entropy` | `(scores, method="softmax", tau=1.0) -> float` | Rank Shannon entropy (bits) |
| `compute_lexical_entropy` | `(text: str) -> float` | Unigram entropy (bits) |
| `compute_graph_entropy` | `(relations: List[Dict]) -> Dict[str, float]` | `relation_type_entropy`, `degree_entropy` |
| `parse_graph_relations_from_text` | `(graph_text: str) -> List[Dict]` | Edge dicts from Cypher-like lines |
| `assemble_retrieval_shannon_fields` | `(*, pre_scores, post_scores, pre_text, post_text, relations, graph_text) -> Dict[str, float]` | Six rounded retrieval fields |
| `empty_retrieval_shannon_fields` | `() -> Dict[str, float]` | All six keys at `0.0` (B0 path) |
| `compute_generation_entropy` | `(tokens_info: List[Dict]) -> float` | Mean token entropy (bits) |
| `compute_citation_entropy` | `(tokens_info, generated_text) -> Tuple[float, int]` | \((H_{\text{citation}}, n_{\text{cit}})\) |
| `find_citation_spans` | `(text: str) -> List[Tuple[int,int]]` | Merged `[start, end)` spans |
| `align_tokens_info` | `(full_text, clean_text, tokens_info) -> List[Dict]` | Re-map char offsets after strip |
| `compute_entropy_reduction` | `(h_b0, h_rag) -> float` | \(\Delta H_{\text{gen}}\) |
| `_extract_single_token_entropy` | `(t_info: Dict) -> float` | Internal per-token helper |

Persisted diagnostic keys on each baseline (`metrics.shannon_diagnostics`):

| Key | Source |
|-----|--------|
| `h_rank_pre_rerank` | pre scores |
| `h_rank_post_rerank` | post / chunk scores |
| `h_lexical_pre_trim` | full context text |
| `h_lexical_post_trim` | trimmed text |
| `h_graph_relation_type` | relations or parsed graph text |
| `h_graph_degree` | same |
| `h_gen` | generation tokens |
| `h_citation` | citation-overlapping tokens |
| `n_citation_tokens` | count |
| `delta_h_gen` | \(H_{B0}-H_{RAG}\) |

---

## 5. Enable / disable

```yaml
rag_components:
  shannon_estimator_enabled: true   # default true; inherited by all baselines
```

When `false`: skip logit streaming; no Shannon fields (zeros on B0-style empty paths only).

---

## 6. Caveats

- Empty or single-candidate score lists → \(H_{\text{rank}} = 0\).
- Softmax \(\tau \le 0\) is clamped to a tiny positive value.
- Graph entropy is zero for empty relation lists (correct for graph-off baselines).
- Offline backfill without stored pre fields falls back to post for both pre and post.
- B0 \(\Delta H_{\text{gen}} \equiv 0\); RAG baselines need B0 \(H_{\text{gen}}\) cached (`_query_b0_h_gen` / disk cache).
- Pre == post is expected when a stage is off; universal equality with reranker/graph **on** is a wiring bug, not a math bug.

---

## 7. Related

- [Pipeline orchestration](./pipeline_orchestration.md) — `run_pipeline.py`, stages, run directories
- `core/generation.py` — live + consume-path diagnostics
- `core/pipelined.py` — pipelined consume-path diagnostics
- `core/retrieval.py` — persists pre-stage fields into contexts YAML
- `core/analytics.py` — aggregation + offline backfill
- `core/reporting.py` — console / markdown Shannon tables
