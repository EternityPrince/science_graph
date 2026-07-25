# RAG Benchmark Pipeline Orchestration

> [!NOTE]
> Detailed specification for process-isolated benchmark execution, VRAM management, CLI flags, sequential vs. pipelined execution modes, baseline configurations, and run directory artifacts.

---

## 1. Overview & Architectural Purpose

`run_pipeline.py` serves as the top-level orchestrator for Science Graph RAG evaluation. It isolates heavy GPU/CPU processing stages into **separate subprocesses** to guarantee that retrieval, local LLM generation, and cloud-based LLM-as-a-Judge evaluation do not contend for VRAM or system memory, and so that an unhandled error in one stage does not corrupt other run artifacts.

```
+-------------------------------------------------------------------------+
|                        run_pipeline.py (parent)                         |
|  resolve dataset · create run_dir · config_snapshot · run_manifest      |
+-----------------------------------┬-------------------------------------+
                                    │
          ┌─────────────────────────▼─────────────────────────┐
          │ STEP 1a  subprocess: run_custom_retrieve.py        │
          │   load dataset → retrieve per baseline (≠ B0)      │
          │   → retrieved_contexts.yaml                        │
          └─────────────────────────┬─────────────────────────┘
                                    │
          ┌─────────────────────────▼─────────────────────────┐
          │ STEP 1b  subprocess: run_benchmarks.py             │
          │   --consume-contexts retrieved_contexts.yaml       │
          │   generate answers + Shannon telemetry fields      │
          │   → evaluation_results.yaml                        │
          └─────────────────────────┬─────────────────────────┘
                                    │ (unless --skip-eval)
          ┌─────────────────────────▼─────────────────────────┐
          │ STEP 2   subprocess: run_evaluator.py              │
          │   LLM-as-a-Judge + retrieval metrics merge         │
          │   → result_metrics.yaml (+ .eval_checkpoint.json)  │
          └─────────────────────────┬─────────────────────────┘
                                    │
          ┌─────────────────────────▼─────────────────────────┐
          │ STEP 3   subprocess: parse_metrics.py <run_dir>    │
          │   analytics, stats, markdown + CSV exports         │
          │   → metrics_summary.md/csv, metrics_details.csv    │
          └───────────────────────────────────────────────────┘
```

### Stage Process Isolation Strategy

| Stage | Process Isolation Rationale | Primary Resource |
| :--- | :--- | :--- |
| **Pre-Retrieval** | Loads embedding models, cross-encoders, and FTS5/graph search indexes | Host RAM & Dense Accelerators |
| **RAG Generation** | Loads local generative LLM (MLX / HuggingFace Transformers); requires exclusive GPU memory | Exclusive VRAM |
| **LLM-as-a-Judge** | Interacts exclusively with external Cloud APIs; must not hold local model weights in VRAM | Network I/O / Concurrency |
| **Metrics Parsing** | Pure CPU analytics, non-parametric statistical hypothesis testing, and report generation | CPU & Host Memory |

Subprocess progress is tracked by `core/subprocess_runner.run_command_with_progress`, which parses stage log output and updates Rich progress bars.

---

## 2. CLI Entry Points & Module Map

### Top-Level Orchestrator
- **`run_pipeline.py`**: Full pipeline entry point (retrieval $\rightarrow$ generation $\rightarrow$ evaluation $\rightarrow$ metrics parsing).

### Stage CLIs (Runnable Standalone)

| CLI Module | Execution Stage | Typical Inputs | Primary Outputs |
| :--- | :--- | :--- | :--- |
| `run_custom_retrieve.py` | Pre-retrieval stage | Golden dataset YAML | `retrieved_contexts.yaml` |
| `run_benchmarks.py` | Generation stage | Dataset + optional `--consume-contexts` | `evaluation_results.yaml` |
| `run_evaluator.py` | LLM-as-a-Judge stage | `evaluation_results.yaml` | `result_metrics.yaml` |
| `parse_metrics.py` | Analytics & exports | Run directory | `metrics_summary.md`, CSV exports, `parsed/` |

### Core Orchestration Modules
- **[core/config.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/config.py)**: Baselines registry, dataset loader, run directory generator, custom configuration CLI parser.
- **[core/subprocess_runner.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/subprocess_runner.py)**: Progress-aware `subprocess.Popen` execution engine.
- **[core/retrieval.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/retrieval.py)**: Staged retrieval execution and comparative evaluation reports.
- **[core/generation.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/generation.py)**: Baseline generation routines (used by `run_benchmarks` and pipelined execution).
- **[core/pipelined.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/pipelined.py)**: Overlapped in-process generation and evaluation engine (`--pipelined`).
- **[core/evaluator.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/evaluator.py)**: Cloud LLM judge integration and resume checkpointing.
- **[core/analytics.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/analytics.py)** / **[core/reporting.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/reporting.py)**: Metrics aggregation, Rich console formatting, and markdown report generation.

---

## 3. Execution Commands & Command-Line Usage

### 3.1 Full Sequential Pipeline (Default)

```bash
python back/benchmarks/rag/run_pipeline.py \
  --dataset back/benchmarks/rag/golden_dataset.yaml \
  --baselines all \
  --output-dir graphs \
  --limit 10
```

### 3.2 Smoke Test / Subset of Baselines

```bash
python back/benchmarks/rag/run_pipeline.py \
  -d back/benchmarks/rag/test_smoke.yaml \
  -b B0,B4,B6 \
  -l 3 \
  --output-dir graphs
```

### 3.3 Cloud Generation & Judge Concurrency Controls

```bash
python back/benchmarks/rag/run_pipeline.py \
  --cloud \
  --baselines B0,B5,B6 \
  --concurrency 4 \
  --rpm 30 \
  --retries 3
```

### 3.4 Pipelined Generation & Evaluation Overlap

```bash
python back/benchmarks/rag/run_pipeline.py \
  --pipelined \
  --baselines B0,B4,B6 \
  --limit 20 \
  --concurrency 4
```

> [!TIP]
> In pipelined mode, retrieval runs first as a subprocess; generation and evaluation then share one process via an async queue (`core/pipelined.run_pipelined_stage_async`), reducing total wall-clock execution time.

### 3.5 Resume into an Existing Run Directory

```bash
python back/benchmarks/rag/run_pipeline.py \
  --output graphs/run_20260629_112933_OCC-RAG-1.7B \
  --baselines all
```

> [!NOTE]
> Passing `--output` fixes the run directory. Generation and evaluation reuse completed checkpoints (`.eval_checkpoint.json`); pass `--clear-checkpoint` to force re-evaluation.

---

## 4. Baselines Catalog (B0–B6 & CUSTOM)

Defined in `core/config.BASELINES_INFO` and resolved via `get_baseline_config()`:

| Baseline ID | Descriptive Name | Architectural Behavior |
| :--- | :--- | :--- |
| **B0** | Zero-Shot Baseline | No retrieval. Direct parametric memory generation. Retrieval Shannon fields zeroed ($\Delta H_{\text{gen}} = 0.0$). |
| **B1** | Pure Lexical | FTS5 keyword BM25 search only. |
| **B2** | Pure Dense | Dense vector embedding similarity search only. |
| **B3** | Dense + HyDE | Dense vector search using Hypothetical Document Embeddings. |
| **B4** | Hybrid + Reranker | Lexical + dense fusion via Reciprocal Rank Fusion (RRF) + Cross-Encoder reranking. Graph expansion disabled. |
| **B5** | Hybrid + Graph + Reranker | B4 hybrid search + graph neighborhood expansion + context trimming + citation repair. |
| **B6** | Full Pipeline | Live `rag_components` snapshot; forces `reranker=True` and `graph_neighbors_in_rrf=True`. |
| **CUSTOM** | User Custom Preset | Component choices and hyperparameters overridden via `--custom` CLI flags or YAML configuration files. |

> [!NOTE]
> For complete component matrix breakdowns, see [baselines_description.md](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/baselines_description.md).

---

## 5. Run Directory Structure & Output Artifacts

```
graphs/run_{YYYYMMDD_HHMMSS}_{model_name}/
├── config_snapshot.yaml         # Complete configuration snapshot at execution start
├── run_manifest.yaml            # Git commit, branch, dataset hash, baselines list, execution timestamp
├── retrieved_contexts.yaml      # Retrieved chunks, pre-rerank scores, and graph relations per query
├── evaluation_results.yaml      # Generated responses, latency, and Shannon diagnostic fields
├── result_metrics.yaml          # LLM-as-a-Judge evaluation metrics and rolling summaries
├── .eval_checkpoint.json        # Evaluation state cache for resume operations
├── metrics_summary.md           # Human-readable markdown comparison tables
├── metrics_summary.csv          # Wide CSV summary
├── metrics_details.csv          # Long detailed query-level pandas export
├── parsed/
│   ├── run_summary.json         # Machine-readable aggregated run summary
│   ├── run_summary.yaml         # YAML aggregated run summary
│   └── per_query_joined.csv     # Aligned per-query wide dataset
└── traces/                      # Retrieval and generation execution traces
```

---

## 6. Orchestrator CLI Parameter Reference

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--dataset` / `-d` | `golden_dataset.yaml` | Path to benchmark question dataset or pre-retrieved context file |
| `--baselines` / `-b` | `all` | Comma-separated list of baselines (`B0,B2,B6` or `all`) |
| `--cloud` | `False` | Use cloud LLM engine for baseline generation |
| `--concurrency` / `-c` | Config setting | Concurrent API calls for cloud judge evaluation |
| `--rpm` / `-r` | Config setting | Rate-limit cap (requests per minute) for judge calls |
| `--retries` | Config setting | Retry count for failed cloud judge requests |
| `--limit` / `-l` | None | Cap the total number of benchmark test cases evaluated |
| `--clear-checkpoint` | `False` | Remove existing `.eval_checkpoint.json` prior to execution |
| `--output-dir` | `graphs` | Target parent directory for auto-named run subdirectories |
| `--output` / `-o` | None | Explicit run directory path |
| `--no-unique-dir` | `False` | Write artifacts directly into `--output-dir` without timestamp subfolder |
| `--skip-eval` | `False` | Terminate pipeline immediately after generation stage |
| `--pipelined` | `False` | Enable overlapped generation and judge execution |

---

## 🔗 Related Documentation
- [Integrated Architecture Reference](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/architecture/integrated_pipeline.md)
- [Logit Telemetry Specifications](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/telemetry/logit_telemetry.md)
- [Shannon Estimator Manual](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/telemetry/shannon_estimator.md)
- [ADR-001: Core Module Refactoring](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/adr/ADR-001-core-module-refactoring.md)
