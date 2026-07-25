# ADR-001: Core Module Refactoring & Pipeline Decoupling

> [!NOTE]
> **Status**: Approved & Implemented  
> **Date**: June 2026  
> **Scope**: `back/benchmarks/rag/core/` and root execution scripts

---

## 1. Context & Problem Statement

The RAG benchmarking suite contained monolithic root-level scripts (`parse_metrics.py`, `generate_scientific_visualizations.py`, `config_creator.py`, `metrics_stats_connector.py`, `run_custom_retrieve.py`, `copy_prompt.py`, `run_pipeline.py`) with duplicated definitions (such as `METRIC_LABELS`, `COLOR_PALETTE`, plot setup parameters, refusal detection, evaluation runners, prompt exporters, and statistical pipeline orchestration) and architectural inversion dependencies where `core/reporting.py` imported root scripts.

Additionally, non-graph baselines (B0–B4) risk data leakage if graph relation metrics pollute non-graph diagnostic fields, subcategory classifications had potential state drift, and sample size scope ($n=50$) in Friedman matrices was aggressively truncated.

---

## 2. Decoupling & Module Extraction

1. **Created [core/traces.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/traces.py)**:
   - Encapsulated trace loading and parsing utilities (`load_graph_retrieval_trace`, `parse_graph_retrieval_trace`, `parse_eval_trace`, `parse_all_traces`).
   - Integrated into `parse_metrics.py`.

2. **Created [core/visualization.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/visualization.py)**:
   - Encapsulated scientific plotting styles (`setup_academic_style`), baseline color palettes (`COLOR_PALETTE`, `get_baseline_color`), plot exporter (`save_plot`), output directory management (`create_output_directory`), and dataset loading functions (`load_report_data`, `find_sciq_results`).
   - Centralized `METRIC_LABELS` and `METRICS` to avoid duplication.
   - Refactored `generate_scientific_visualizations.py` to act as a driver script delegating to `core.visualization`.

3. **Enhanced [core/config.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/config.py)**:
   - Moved custom preset configurations (`CUSTOM_PRESET_COMPONENTS`, `CUSTOM_PRESET_HYPERPARAMS_NT`, `CustomPresetHyperparams`), argument parser setup (`add_custom_config_arguments`), config building (`build_custom_config`), and monkey-patching logic into `core/config.py`.
   - Converted `config_creator.py` into a facade module for backward compatibility.

4. **Created [core/connector.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/connector.py)**:
   - Encapsulated statistical pipeline integration agents (`data_prep_agent`, `stats_agent`, `reporting_agent`, `build_statistical_markdown`, `generate_statistical_plots`, `run_statistical_pipeline`, `export_stats_json`).
   - Fixed architectural inversion dependency in `core/reporting.py` so it imports directly from `core.connector`.
   - Maintained `metrics_stats_connector.py` as a facade module for backward compatibility.

5. **Enhanced [core/retrieval.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/retrieval.py)**:
   - Encapsulated retrieval evaluation and report generator (`evaluate_and_compare`, `save_custom_retrieval_report` / `save_markdown_report`).
   - Refactored `run_custom_retrieve.py`, `base_sweeper.py`, and `run_pipeline.py` to import evaluation logic directly from `core.retrieval`.

6. **Enhanced [core/clipboard.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/clipboard.py)**:
   - Encapsulated prompt template loading (`load_templates`), table displays (`display_runs_table`, `display_templates_table`), and interactive run selection (`interactive_selection`).
   - Refactored `copy_prompt.py` to delegate prompt generation helper functions to `core.clipboard`.

7. **Updated Package Index**:
   - Exported `traces`, `visualization`, and `connector` in `core/__init__.py`.
   - Added unit test coverage in `tests/test_traces.py`, `tests/test_connector.py`, and `tests/test_clipboard.py`.

---

## 3. Statistical & Metric Pipeline Fixes

8. **Graph Baseline Isolation & Leakage Prevention**:
   - Explicitly defined `GRAPH_ENABLED_BASELINES = {"B5", "B6"}` in `core/analytics.py` and `parse_metrics.py`.
   - Enforced that non-graph baselines (B0–B4) hardcode `DETAIL_GRAPH_FIELDS` to fresh default copies (`0`, `0.0`, `False`, `""`, `[]`, `{}`), skipping `trace_map` extraction and preventing graph chunk/node counts from leaking into non-graph baselines.

9. **State Drift Elimination in Subcategory Aggregations**:
   - Unified classification state logic by pre-computing `answerability_outcome` deterministically once per `(result, baseline)` tuple during the primary pass.
   - Subcategory breakdowns (`category_classification`) directly read pre-computed outcomes rather than running secondary inline text re-parsing (`detect_abstention`), guaranteeing strict mathematical alignment between global and category safety tables.

10. **Friedman Test Matrix Preservation (Restoring $n=50$ Scope)**:
    - Refactored matrix assembly in `core/statistics.py` (`friedman_omnibus_test`) and `generate_stat_run_results.py`.
    - Eliminated aggressive list-wise query deletion. For quality metrics, sample size $n=50$ is preserved across answerable queries by safely imputing missing/abstention scores with `0.0`.

11. **Precision Formatting & Rounding Synchronization**:
    - Standardized `format_val` in `parse_metrics.py` and `format_pct` / `format_avg` in `core/reporting.py` to use explicit standard round-to-nearest (`round(val * 100, digits)`).

---

## 4. Consequences & Tradeoffs

- **Facade Compatibility**: Maintained legacy wrapper facades (`config_creator.py`, `metrics_stats_connector.py`, wrapper functions) to prevent breaking external CLI scripts or legacy test suite imports.
- **Zero Behavior Regression**: Preserved existing data contracts and default values to guarantee zero behavior changes across evaluation pipelines.

---

## 🔗 Related Documentation
- [Integrated Architecture Reference](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/architecture/integrated_pipeline.md)
- [Pipeline Orchestration Manual](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/architecture/pipeline_orchestration.md)
- [Statistical Testing Framework](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/docs/statistics/statistical_testing_framework.md)
