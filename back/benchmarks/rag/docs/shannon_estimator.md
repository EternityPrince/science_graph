# Shannon Estimator: Component-Level Framework for RAG Development

The Shannon Estimator is a **diagnostic framework** for inspecting how individual RAG components change information uncertainty. Use it when adding, ablating, or tuning retrieval, reranking, graph expansion, context trimming, or generation — not only as a final quality score.

Unlike end-to-end metrics (recall, precision, judge scores), Shannon metrics answer:

> *At this pipeline stage, how uncertain is the ranking / context / graph / model prediction — and what did this component do to that uncertainty?*

All values are in **bits** ($\log_2$).

---

## 1. Metrics

| Metric | Symbol | What it measures | Stage boundary |
|--------|--------|------------------|----------------|
| Rank entropy | $H_{\text{rank}}$ | Uncertainty over candidate relevance scores | pre-rerank → post-rerank |
| Lexical entropy | $H_{\text{lexical}}$ | Unigram diversity of assembled context text | pre-trim → post-trim |
| Graph relation entropy | $H_{\text{graph,rel}}$ | Diversity of edge relation types | graph context for the query |
| Graph degree entropy | $H_{\text{graph,deg}}$ | Diversity of node degree mass | same subgraph |
| Generation entropy | $H_{\text{gen}}$ | Mean per-token predictive entropy of the answer | generation (logits) |
| Citation entropy | $H_{\text{citation}}$ | Mean token entropy inside citation spans | generation |
| Entropy reduction | $\Delta H_{\text{gen}}$ | $H_{\text{gen}}^{B0} - H_{\text{gen}}^{RAG}$ | zero-shot vs RAG |

### Formulas (short)

**Rank** (softmax over scores $s_i$, temperature $\tau$):

$$
P(c_i) = \frac{\exp(s_i / \tau)}{\sum_j \exp(s_j / \tau)}, \quad
H_{\text{rank}} = -\sum_i P(c_i)\log_2 P(c_i)
$$

Also supports `minmax` and `sum` normalization (`compute_rank_entropy`).

**Lexical** (unigram frequencies over `\w+` tokens):

$$
H_{\text{lexical}} = -\sum_t P(t)\log_2 P(t)
$$

**Graph** over structured edges `{source, target, type}`:

$$
H_{\text{rel}} = -\sum_r P(r)\log_2 P(r), \quad
H_{\text{deg}} = -\sum_v \frac{d(v)}{\sum_u d(u)}\log_2\frac{d(v)}{\sum_u d(u)}
$$

**Generation**: average of per-token entropy from logits / top logprobs.

**ΔH_gen**: bits of predictive uncertainty removed by supplying retrieved context vs B0 zero-shot.

---

## 2. Pre / post semantics (important)

| Column | Pre | Post | When equality is expected |
|--------|-----|------|---------------------------|
| `H_rank (pre/post)` | Scores **before** cross-encoder rerank (RRF / hybrid) | Scores **after** rerank (blended or CE raw) | Reranker off, or only one score list captured |
| `H_lexical (pre/post)` | Full `context_text` from `build_context` | `trimmed_text` after `trim_context` | Trimming disabled or context already fits budget |
| `H_graph (rel/deg)` | Structured relations for the assembled subgraph | (single stage — not pre/post) | Graph disabled, no neighbors, or empty relation list |

**Do not interpret pre == post as “Shannon is broken”** when the stage is legitimately off.  
**Do treat universal pre == post across B4/B5/B6 with reranker and graph on as a wiring bug** — that was the old hard-copy bug (pre assigned from post; graph forced to `0.0`).

### Data flow

```
retrieve (Stage 3: RRF scores)
    └─ pre_rerank_scores  ─────────────────────────────┐
rerank (Stage 4)                                       │
    └─ retrieved_chunks[].score (post)                 │
build_context                                          │
    └─ context_text, context_graph, graph_relations    │
trim_context                                           │
    └─ trimmed_text, trimmed_graph                     │
generate                                               │
    └─ assemble_retrieval_shannon_fields(...) ◄────────┘
         + H_gen / H_citation / ΔH_gen from logits
```

Persisted on each baseline in `retrieved_contexts.yaml` (Stage 5):

- `pre_rerank_scores`
- `context_text`, `context_graph`
- `graph_relations` (structured; preferred over regex)
- `trimmed_text`, `trimmed_graph`, `retrieved_chunks`

Live path side-channels on `RAGService`:

- `_last_pre_rerank_scores` — set in `retrieve_relevant_chunks` before CE
- `_last_graph_relations` — set in `_get_scored_graph_lines` / `build_context`

Assembly helper (single source of truth):

```python
from core.shannon_estimator import assemble_retrieval_shannon_fields

fields = assemble_retrieval_shannon_fields(
    pre_scores=pre_rerank_scores,
    post_scores=post_scores,
    pre_text=context_text,
    post_text=trimmed_text,
    relations=graph_relations,
    graph_text=context_graph,  # fallback parser if relations empty
)
```

Used by `generation.py` (live + consume), `pipelined.py`, and offline backfill in `analytics.py`.

---

## 3. Enable / disable

```yaml
rag_components:
  shannon_estimator_enabled: true   # default true
```

When `false`: skip logit streaming; no Shannon fields (or zeros on B0 paths only).

---

## 4. Using Shannon to develop new RAG components

1. **Add a component boundary** where scores, text, or edges change (e.g. a new fusion step).
2. **Capture pre-stage observables** before the component and post-stage after it.
3. **Pass them into** `assemble_retrieval_shannon_fields` (or extend it with a new metric).
4. **Compare baselines**:
   - Reranker: $H_{\text{rank,pre}} - H_{\text{rank,post}}$ — large positive Δ means sharper ranking.
   - Trimming: $H_{\text{lexical,pre}}$ vs post — collapse may mean over-pruning diversity; flat may mean no trim.
   - Graph: non-zero $H_{\text{graph}}$ only when relations exist — use to validate graph expansion actually injects structure.
   - Generation: higher $\Delta H_{\text{gen}}$ means RAG context reduces predictive uncertainty vs B0; pair with faithfulness/citation metrics.

### Baseline reading guide

| Baseline | Expect rank pre≠post? | Lexical pre≠post? | Graph > 0? |
|----------|----------------------|-------------------|------------|
| B0 | no (zeros) | no | no |
| B1 / B2 (no CE) | often equal (honest) | maybe if trim prunes | no if graph off |
| B4 (hybrid + CE) | **yes** when CE reshapes scores | if trim active | no if graph off |
| B5 / B6 (+ graph) | **yes** with CE | if trim active | **yes** when neighbors exist |

---

## 5. Module map

| Path | Role |
|------|------|
| `core/shannon_estimator.py` | Pure math + `assemble_retrieval_shannon_fields` / graph text parser |
| `core/retrieval.py` | Persists pre-stage fields into contexts YAML |
| `core/generation.py` | Live + consume-path diagnostics |
| `core/pipelined.py` | Pipelined consume-path diagnostics |
| `core/analytics.py` | Aggregation + offline backfill |
| `core/reporting.py` | Console / markdown Shannon table |
| `src/services/rag_service.py` | Captures `_last_pre_rerank_scores`, `_last_graph_relations` |

Math unit tests: `tests/test_shannon_estimator.py`.  
Wiring / regression tests: `tests/test_shannon_diagnostics_wiring.py`.

---

## 6. Citation token entropy (generation detail)

During `generate_response_with_logits`, each token stores entropy and character offsets. After the full string is built:

1. `find_citation_spans(text)` finds citation markers (`[sciq_paper_X]`, `[1]`, DOIs, author-year, …).
2. Tokens whose char ranges overlap those spans contribute to $H_{\text{citation}}$.
3. Prefer measuring on **raw** model text before citation repair so uncertainty reflects the model, not post-processing.

---

## 7. Caveats

- Empty or single-candidate score lists → $H_{\text{rank}} = 0$.
- Softmax temperature $\tau \le 0$ is clamped to a tiny positive value.
- Graph entropy is zero for empty relation lists — that is correct for graph-off baselines.
- Offline backfill without stored pre fields falls back to post for both pre and post (honest “unknown pre”, not invented structure).
- B0 $\Delta H_{\text{gen}}$ is defined as 0; RAG baselines need B0 $H_{\text{gen}}$ cached (`_query_b0_h_gen` / disk cache).
