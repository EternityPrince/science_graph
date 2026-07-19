# RAG Benchmark Pipeline Orchestration

End-to-end orchestration for Science Graph RAG quality benchmarking. The pipeline isolates heavy GPU/CPU stages in **separate subprocesses** so retrieval, local generation, and cloud evaluation do not fight for VRAM, and so a crash in one stage does not corrupt others.

Related math/diagnostics: [shannon_estimator.md](./shannon_estimator.md)  
Related baselines detail: [../baselines_description.md](../baselines_description.md)  
Related refactor notes: [refactoring_architecture.md](./refactoring_architecture.md)

---

## Purpose

`run_pipeline.py` is the **top-level orchestrator**. It:

1. Resolves dataset, run directory, and baseline list.
2. Snapshots config + run manifest for reproducibility.
3. Invokes stage CLIs as subprocesses (or an in-process pipelined gen+eval path).
4. Aggregates metrics into markdown/CSV reports under a single run directory.

Stages are deliberately process-isolated:

| Stage | Why isolate |
|-------|-------------|
| Pre-retrieval | Loads embedding / reranker / graph services; can be memory-heavy |
| RAG generation | Loads local LLM (MLX/transformers); needs exclusive VRAM |
| LLM-as-a-Judge | Cloud API only; must not hold local model weights |
| Metrics parse | Pure CPU analytics; no model load |

Subprocess progress is tracked by `core/subprocess_runner.run_command_with_progress`, which parses stage-specific log lines and advances a Rich progress bar.

---

## Entry points

### Top orchestrator

| Path | Role |
|------|------|
| `back/benchmarks/rag/run_pipeline.py` | Full pipeline: retrieve → generate → evaluate → parse |

### Stage CLIs (also runnable alone)

| Path | Stage | Typical inputs | Typical outputs |
|------|-------|----------------|-----------------|
| `run_custom_retrieve.py` | Pre-retrieval | golden dataset YAML | `retrieved_contexts.yaml` |
| `run_benchmarks.py` | Generation | dataset + optional `--consume-contexts` | `evaluation_results.yaml` |
| `run_evaluator.py` | LLM-as-a-Judge | `evaluation_results.yaml` | `result_metrics.yaml` |
| `parse_metrics.py` | Metrics + reports | run directory | `metrics_summary.md`, CSVs, `parsed/` |

### Pipelined core

| Path | Role |
|------|------|
| `core/pipelined.py` | Concurrent generation + evaluation in one process (`--pipelined`) |

Supporting modules used by the orchestrator:

- `core/config.py` — baselines, dataset load, run-dir naming, custom config CLI
- `core/subprocess_runner.py` — progress-aware `subprocess.Popen`
- `core/retrieval.py` — staged retrieval + retrieval comparison tables
- `core/generation.py` — baseline generation (used by `run_benchmarks` and pipelined path)
- `core/evaluator.py` — cloud judge + checkpoints
- `core/analytics.py` / `core/reporting.py` — aggregation and reports

---

## How to run

Run from the repository (or ensure `back/` is on `PYTHONPATH` as the CLIs do via `sys.path` inserts). Examples assume CWD is the repo root or that you invoke via absolute paths.

### Full sequential pipeline (default)

```bash
python back/benchmarks/rag/run_pipeline.py \
  --dataset back/benchmarks/rag/golden_dataset.yaml \
  --baselines all \
  --output-dir graphs \
  --limit 10
```

### Smoke / subset of baselines

```bash
python back/benchmarks/rag/run_pipeline.py \
  -d back/benchmarks/rag/test_smoke.yaml \
  -b B0,B4,B6 \
  -l 3 \
  --output-dir graphs
```

### Cloud generation + judge concurrency controls

```bash
python back/benchmarks/rag/run_pipeline.py \
  --cloud \
  --baselines B0,B5,B6 \
  --concurrency 4 \
  --rpm 30 \
  --retries 3
```

### Pipelined gen+eval (overlap judge with generation)

```bash
python back/benchmarks/rag/run_pipeline.py \
  --pipelined \
  --baselines B0,B4,B6 \
  --limit 20 \
  --concurrency 4
```

Retrieval still runs first as a subprocess; generation and evaluation then share one process with an async queue (`core/pipelined.run_pipelined_stage_async`).

### Skip judge + metrics (generation only)

```bash
python back/benchmarks/rag/run_pipeline.py \
  --skip-eval \
  --baselines B0,B2 \
  --limit 5
```

### Resume into an existing run directory

```bash
python back/benchmarks/rag/run_pipeline.py \
  --output graphs/run_20260629_112933_OCC-RAG-1.7B \
  --baselines all
```

`--output` fixes the run dir. Generation and evaluation reuse successful checkpoints when present; pass `--clear-checkpoint` to force re-evaluation.

### Consume pre-retrieved contexts

If the dataset path is already a contexts YAML (`retrieved_contexts.yaml` / `custom_retrieved_contexts.yaml`, or YAML head contains `baselines:`), retrieval is skipped and outputs write next to that file:

```bash
python back/benchmarks/rag/run_pipeline.py \
  --dataset graphs/some_run/retrieved_contexts.yaml \
  --baselines B0,B4,B6
```

### Custom component / hyperparameter overrides

Pipeline accepts the same custom-config flags as retrieve/benchmarks (`add_custom_config_arguments`). Overrides are written to `temp_custom_config.yaml` in the run dir and passed as `--config-file` to child stages:

```bash
python back/benchmarks/rag/run_pipeline.py \
  --custom \
  --reranker \
  --graph-expansion \
  --score-blend-reranker-weight 0.7 \
  --baselines B0,CUSTOM
```

### Run stages standalone

```bash
# 1a Retrieval only
python back/benchmarks/rag/run_custom_retrieve.py \
  -d back/benchmarks/rag/golden_dataset.yaml \
  -o graphs/my_run/retrieved_contexts.yaml \
  --baselines B4,B6 \
  --no-unique-dir \
  --limit 10

# 1b Generation consuming contexts
python back/benchmarks/rag/run_benchmarks.py \
  -d back/benchmarks/rag/golden_dataset.yaml \
  -o graphs/my_run/evaluation_results.yaml \
  --baselines B0,B4,B6 \
  --consume-contexts graphs/my_run/retrieved_contexts.yaml \
  --no-unique-dir \
  --limit 10

# 2 Judge
python back/benchmarks/rag/run_evaluator.py \
  --input graphs/my_run/evaluation_results.yaml \
  --output graphs/my_run/result_metrics.yaml \
  --baselines B0,B4,B6 \
  --concurrency 4

# 3 Parse / export
python back/benchmarks/rag/parse_metrics.py graphs/my_run
```

---

## Stage diagrams

### Sequential mode (default)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         run_pipeline.py (parent)                         │
│  resolve dataset · create run_dir · config_snapshot · run_manifest       │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
          ┌─────────────────────────▼─────────────────────────┐
          │ STEP 1a  subprocess: run_custom_retrieve.py        │
          │   load dataset → retrieve per baseline (≠ B0)      │
          │   → retrieved_contexts.yaml                        │
          │   progress total = cases × baselines_with_retrieval│
          └─────────────────────────┬─────────────────────────┘
                                    │
          ┌─────────────────────────▼─────────────────────────┐
          │ STEP 1b  subprocess: run_benchmarks.py             │
          │   --consume-contexts retrieved_contexts.yaml       │
          │   generate answers + Shannon fields                │
          │   → evaluation_results.yaml                        │
          │   progress total = cases × baselines               │
          └─────────────────────────┬─────────────────────────┘
                                    │  (unless --skip-eval)
          ┌─────────────────────────▼─────────────────────────┐
          │ STEP 2   subprocess: run_evaluator.py              │
          │   LLM-as-a-Judge + retrieval metrics merge         │
          │   → result_metrics.yaml (+ .eval_checkpoint.json)  │
          │   progress total = cases × baselines               │
          └─────────────────────────┬─────────────────────────┘
                                    │
          ┌─────────────────────────▼─────────────────────────┐
          │ STEP 3   subprocess: parse_metrics.py <run_dir>    │
          │   analytics, stats, markdown + CSV exports         │
          │   → metrics_summary.md/csv, metrics_details.csv,   │
          │     parsed/*                                       │
          └───────────────────────────────────────────────────┘
```

VRAM isolation: after each stage subprocess exits, model weights are released before the next stage starts.

### Pipelined mode (`--pipelined`)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         run_pipeline.py (parent)                         │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
          ┌─────────────────────────▼─────────────────────────┐
          │ STEP 1a  subprocess: run_custom_retrieve.py        │
          │   → retrieved_contexts.yaml  (same as sequential)  │
          └─────────────────────────┬─────────────────────────┘
                                    │
          ┌─────────────────────────▼─────────────────────────┐
          │ In-process: core/pipelined.run_pipelined_stage_async│
          │                                                     │
          │   generator_task (local LLM)                        │
          │        │  each (case, baseline) answer              │
          │        ▼                                            │
          │   asyncio.Queue  ──────────────────────────────┐    │
          │        │                                       │    │
          │        ▼                                       │    │
          │   N evaluator_worker tasks (cloud judge)       │    │
          │        │                                       │    │
          │        ├── append evaluation_results.yaml      │    │
          │        └── append result_metrics.yaml          │    │
          │             (BufferedYAMLWriter, checkpointed) │    │
          └─────────────────────────┬──────────────────────┘───┘
                                    │  (unless --skip-eval)
          ┌─────────────────────────▼─────────────────────────┐
          │ STEP 3   subprocess: parse_metrics.py <run_dir>    │
          └───────────────────────────────────────────────────┘
```

Tradeoff: pipelined mode keeps the generation model loaded while the cloud judge runs (better wall-clock, less VRAM isolation than sequential). Use sequential when local VRAM is tight.

---

## Progress totals

Computed in `run_pipeline.py` after loading the dataset and resolving baselines:

| Quantity | Formula | Used for |
|----------|---------|----------|
| `num_cases` | `len(load_benchmark_dataset(...))` (fallback 50) | all stages |
| `num_baselines` | count of resolved baseline IDs | gen + eval |
| `num_baselines_with_retrieval` | baselines excluding **B0** | retrieval only |
| Retrieval steps | `max(cases × baselines_with_retrieval, 1)` | progress bar 1a |
| Generation steps | `max(cases × baselines, 1)` | progress bar 1b / pipelined gen |
| Evaluation steps | `max(cases × baselines, 1)` | progress bar 2 / pipelined eval |

Examples:

- 20 cases, baselines `B0,B4,B6` → retrieval \(20×2=40\), gen/eval \(20×3=60\)
- 50 cases, `all` (B0–B6 + CUSTOM if selected) with B0–B6 only → retrieval \(50×6=300\), gen/eval \(50×7=350\)

Progress patterns parsed from child stdout (`core/subprocess_runner.py`):

- **retrieval**: lines containing `Query: '`
- **generation**: lines starting with `Running ` (baseline tick)
- **evaluation**: lines `Evaluated case i/N` (absolute completed count)

---

## Baselines B0–B6 overview

Defined in `core/config.BASELINES_INFO` and resolved by `get_baseline_config()`:

| ID | Name | Retrieval / generation behavior |
|----|------|----------------------------------|
| **B0** | Zero-Shot | No retrieval. Parametric-memory prompt only. Shannon retrieval fields zeroed; \(\Delta H_{\text{gen}}=0\) by definition. |
| **B1** | Pure Lexical | FTS5 keyword search only. |
| **B2** | Pure Dense | Dense embedding search only. |
| **B3** | Dense + HyDE | Dense search with hypothetical document embedding. |
| **B4** | Hybrid + Reranker | Lexical + dense, RRF, dynamic alpha, cross-encoder rerank. Graph expansion forced off. |
| **B5** | Hybrid + Graph + Reranker | B4-style hybrid + graph expansion into prompt + context trimming + citation repair. |
| **B6** | Full Pipeline | Inherits live `rag_components` snapshot; forces `hyde=False`, `reranker=True`, `graph_neighbors_in_rrf=True`. |
| **CUSTOM** | User preset / CLI | Components and hyperparameters from `--custom` / `--config-file` / flag overrides. |

When `--baselines all`:

- Normal dataset path → all keys in `BASELINES_INFO` (includes CUSTOM).
- Pre-retrieved contexts path → intersection of `BASELINES_INFO` with baselines present in the YAML (always keeps B0 if requested via presence rules in orchestrator).

For full component matrices and 19-stage RAG service map, see [baselines_description.md](../baselines_description.md).

---

## Data flow and run directory outputs

### Run directory creation

Unless `--output` or pre-retrieved-context detection or `--no-unique-dir`:

```
{output-dir}/run_{YYYYMMDD_HHMMSS}_{safe_model_name}/
```

Created via `core.config.create_graph_run_dir`, which always ensures:

```
run_*/
  traces/
  parsed/
```

### Artifact map

| Artifact | Producer | Description |
|----------|----------|-------------|
| `run_manifest.yaml` | orchestrator | run_id, git commit/branch, baselines, model/embedding/reranker, dataset hash, paths |
| `config_snapshot.yaml` | orchestrator | full `config.data` dump at start |
| `temp_custom_config.yaml` | orchestrator (if overrides) | temporary; deleted at end |
| `retrieved_contexts.yaml` | `run_custom_retrieve.py` | per-query, per-baseline chunks, pre_rerank_scores, context/trimmed text, graph_relations |
| `evaluation_results.yaml` | `run_benchmarks.py` or pipelined | generated answers, latency, chunks, metrics incl. `shannon_diagnostics` |
| `result_metrics.yaml` | `run_evaluator.py` or pipelined | judge metrics + details + rolling summary |
| `evaluation_results_judge.yaml` | judge report helper | flattened judge view |
| `baselines/evaluation_results_judge_b*.yaml` | per-baseline judge splits | optional individual reports |
| `.eval_checkpoint.json` | evaluator | resume cache for judge calls |
| `metrics_summary.md` | `parse_metrics.py` | human-readable summary tables |
| `metrics_summary.csv` | `parse_metrics.py` | wide CSV (Typst-friendly) |
| `metrics_details.csv` | `parse_metrics.py` | long/detailed pandas export |
| `parsed/run_summary.{json,yaml}` | parser/analytics | machine-readable aggregates |
| `parsed/per_query_joined.csv` | parser | joined per-query view |
| `traces/` | RAG service / generation | per-query retrieval/generation traces |

### Conceptual data flow

```
golden_dataset.yaml
        │
        ▼
 retrieved_contexts.yaml
   • id, query, expected_papers
   • baselines[Bn]: retrieved_chunks, pre_rerank_scores,
     context_text, trimmed_text, graph_relations, ...
        │
        ▼
 evaluation_results.yaml
   • baselines[Bn]: generated_answer, metrics,
     metrics.shannon_diagnostics { h_rank_*, h_lexical_*,
       h_graph_*, h_gen, h_citation, delta_h_gen }
        │
        ▼
 result_metrics.yaml
   • eval_metrics: faithfulness, answer_relevance,
     semantic_accuracy, citation_fidelity, ...
        │
        ▼
 metrics_summary.md / CSVs / parsed/*
```

Shannon field assembly and formulas are documented in [shannon_estimator.md](./shannon_estimator.md). Wiring happens in `core/generation.py`, `core/pipelined.py`, and `core/retrieval.py`.

---

## Orchestrator CLI reference

| Flag | Default | Meaning |
|------|---------|---------|
| `--dataset` / `-d` | `golden_dataset.yaml` (fallback `.example`) | Input questions or pre-retrieved contexts |
| `--baselines` / `-b` | `all` | `B0,B2,B6` or `all` |
| `--cloud` | off | Cloud LLM for generation |
| `--concurrency` / `-c` | config | Judge concurrent API calls |
| `--rpm` / `-r` | config | Judge rate limit |
| `--retries` | config | Judge retries |
| `--limit` / `-l` | none | Cap number of cases |
| `--clear-checkpoint` | off | Drop eval checkpoint |
| `--output-dir` | `graphs` | Parent for new run dirs |
| `--output` / `-o` | none | Explicit run directory |
| `--no-unique-dir` | off | Write directly under `--output-dir` |
| `--skip-eval` | off | Stop after generation |
| `--pipelined` | off | Overlap gen + judge |
| custom config flags | — | See `add_custom_config_arguments` in `core/config.py` |

---

## Failure and cleanup behavior

- Child stage non-zero exit → orchestrator logs failure, removes `temp_custom_config.yaml` if present, `sys.exit` with child code (or 1 for pipelined exceptions).
- Retrieval metrics table (`evaluate_and_compare`) is best-effort after step 1a; failures become warnings.
- Pipelined path flushes YAML buffers and forces eval checkpoint save in a `finally` block, then unloads the local LLM.

---

## See also

- [Shannon math & worked examples](./shannon_estimator.md) — rank / lexical / graph / generation / \(\Delta H\) formulas
- [Baselines architecture](../baselines_description.md) — B0–B6 component matrix
- [Core refactoring ADR](./refactoring_architecture.md) — module boundaries after extraction into `core/`
