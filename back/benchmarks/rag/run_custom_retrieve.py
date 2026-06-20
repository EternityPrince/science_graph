#!/usr/bin/env python3
"""
Science Graph — Custom RAG Staged Retrieval Runner.
Allows running the retrieval pipeline with a custom configuration of components
and hyperparameters, and compares results against baselines.
"""

import sys
import argparse
import copy
import yaml
from pathlib import Path
from typing import NamedTuple

# Set up python path to resolve src and core imports correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.container import container
from src.config import config
from src import console as con
from src.prompts import prompts
from core.retrieval import run_staged_retrieval
from core.metrics import calculate_retrieval_recall, calculate_context_precision
import core.config
import core.retrieval

# Store default config settings for reference and rollback
DEFAULT_COMPONENTS = copy.deepcopy(config.data.get("rag_components", {}))
DEFAULT_HYPERPARAMS = copy.deepcopy(config.data.get("hyperparameters", {}))

# Hardcoded CUSTOM preset configuration (components and hyperparameters).
# These will be applied when the user specifies the --custom flag.
CUSTOM_PRESET_COMPONENTS = {
    "intent_classifier": False,
    "graph_ontology_lookup": True,
    "llm_query_expansion": False,
    "hyde": False,
    "lexical_search": True,
    "dense_search": True,
    "dynamic_alpha_blending": False,
    "rrf": True,
    "graph_expansion": False,
    "reranker": True,
    "score_blending": True,
    "context_trimming": True,
    "citation_repair": True,
}

class RAGPreset(NamedTuple):
    score_blend_reranker_weight: float
    score_blend_rrf_weight: float
    rrf_k: float
    dynamic_alpha_threshold_low: float
    dynamic_alpha_val_low: float
    dynamic_alpha_threshold_mid: float
    dynamic_alpha_val_mid: float
    dynamic_alpha_val_high: float

class GraphPreset(NamedTuple):
    p_base: float
    gamma: float
    crawl_stop_threshold: float
    semantic_score_threshold: float
    semantic_score_top_p: float
    sigmoid_score_threshold: float
    sigmoid_score_top_p: float
    essential_fact_threshold: float
    sigmoid_slope: float
    sigmoid_center: float
    weight_authored: float
    weight_cites: float
    weight_mentions_concept: float
    weight_default: float

class BM25Preset(NamedTuple):
    k1: float
    b: float

class CustomPresetHyperparams(NamedTuple):
    rag: RAGPreset
    graph: GraphPreset
    bm25: BM25Preset

CUSTOM_PRESET_HYPERPARAMS_NT = CustomPresetHyperparams(
    rag=RAGPreset(
        score_blend_reranker_weight=0.75,
        score_blend_rrf_weight=0.25,
        rrf_k=60.0,
        dynamic_alpha_threshold_low=1.2,
        dynamic_alpha_val_low=0.15,
        dynamic_alpha_threshold_mid=3.0,
        dynamic_alpha_val_mid=0.5,
        dynamic_alpha_val_high=1.0,
    ),
    graph=GraphPreset(
        p_base=0.0,
        gamma=0.0,
        crawl_stop_threshold=1.0,
        semantic_score_threshold=0.35,
        semantic_score_top_p=0.9,
        sigmoid_score_threshold=0.4,
        sigmoid_score_top_p=0.9,
        essential_fact_threshold=0.5,
        sigmoid_slope=0.0,
        sigmoid_center=0.5,
        weight_authored=0.8,
        weight_cites=0.7,
        weight_mentions_concept=0.6,
        weight_default=0.5,
    ),
    bm25=BM25Preset(
        k1=1.5,
        b=0.75,
    )
)

CUSTOM_PRESET_HYPERPARAMS = {
    "rag": CUSTOM_PRESET_HYPERPARAMS_NT.rag._asdict(),
    "graph": CUSTOM_PRESET_HYPERPARAMS_NT.graph._asdict(),
    "bm25": CUSTOM_PRESET_HYPERPARAMS_NT.bm25._asdict(),
}


def get_custom_preset_weights(preset_hype: CustomPresetHyperparams) -> dict:
    """Returns the custom preset weights configured for edge-type heuristics from the provided NamedTuple."""
    return {
        "weight_authored": preset_hype.graph.weight_authored,
        "weight_cites": preset_hype.graph.weight_cites,
        "weight_mentions_concept": preset_hype.graph.weight_mentions_concept,
        "weight_default": preset_hype.graph.weight_default,
    }


def build_custom_config(args, file_config=None) -> tuple[dict, dict]:
    """Builds custom components and hyperparameters dictionaries by merging
    defaults, hardcoded preset (if args.custom), file config (if provided), and CLI overrides.
    """
    custom_comp = copy.deepcopy(DEFAULT_COMPONENTS)
    custom_hype = copy.deepcopy(DEFAULT_HYPERPARAMS)

    # Apply hardcoded custom preset if --custom is specified
    if getattr(args, "custom", False):
        custom_comp.update(CUSTOM_PRESET_COMPONENTS)
        for section, params in CUSTOM_PRESET_HYPERPARAMS.items():
            if section not in custom_hype:
                custom_hype[section] = {}
            custom_hype[section].update(params)

    # If file config is provided, merge it
    if file_config:
        if "rag_components" in file_config:
            custom_comp.update(file_config["rag_components"])
        if "hyperparameters" in file_config:
            # Deep merge hyperparameters
            for section, params in file_config["hyperparameters"].items():
                if section not in custom_hype:
                    custom_hype[section] = {}
                custom_hype[section].update(params)

    # Merge CLI arguments for components
    comp_fields = [
        "intent_classifier", "graph_ontology_lookup", "llm_query_expansion", 
        "hyde", "lexical_search", "dense_search", "dynamic_alpha_blending", 
        "rrf", "graph_expansion", "reranker", "score_blending", 
        "context_trimming", "citation_repair"
    ]
    for field in comp_fields:
        val = getattr(args, field, None)
        if val is not None:
            custom_comp[field] = val

    # Merge CLI arguments for hyperparameters
    # RAG hyperparameters
    rag_hype_fields = [
        "score_blend_reranker_weight", "score_blend_rrf_weight", "rrf_k",
        "dynamic_alpha_threshold_low", "dynamic_alpha_val_low",
        "dynamic_alpha_threshold_mid", "dynamic_alpha_val_mid",
        "dynamic_alpha_val_high"
    ]
    for field in rag_hype_fields:
        val = getattr(args, field, None)
        if val is not None:
            if "rag" not in custom_hype:
                custom_hype["rag"] = {}
            custom_hype["rag"][field] = val

    # Graph hyperparameters
    graph_hype_fields = [
        ("graph_p_base", "p_base"),
        ("graph_gamma", "gamma"),
        ("graph_crawl_stop_threshold", "crawl_stop_threshold"),
        ("graph_semantic_score_threshold", "semantic_score_threshold"),
        ("graph_semantic_score_top_p", "semantic_score_top_p"),
        ("graph_sigmoid_score_threshold", "sigmoid_score_threshold"),
        ("graph_sigmoid_score_top_p", "sigmoid_score_top_p"),
        ("graph_essential_fact_threshold", "essential_fact_threshold"),
        ("graph_sigmoid_slope", "sigmoid_slope"),
        ("graph_sigmoid_center", "sigmoid_center"),
        ("graph_weight_authored", "weight_authored"),
        ("graph_weight_cites", "weight_cites"),
        ("graph_weight_mentions_concept", "weight_mentions_concept"),
        ("graph_weight_default", "weight_default")
    ]
    for arg_name, conf_name in graph_hype_fields:
        val = getattr(args, arg_name, None)
        if val is not None:
            if "graph" not in custom_hype:
                custom_hype["graph"] = {}
            custom_hype["graph"][conf_name] = val

    # BM25 hyperparameters
    bm25_hype_fields = [
        ("bm25_k1", "k1"),
        ("bm25_b", "b")
    ]
    for arg_name, conf_name in bm25_hype_fields:
        val = getattr(args, arg_name, None)
        if val is not None:
            if "bm25" not in custom_hype:
                custom_hype["bm25"] = {}
            custom_hype["bm25"][conf_name] = val

    return custom_comp, custom_hype


def patch_retrieval_for_custom(custom_comp: dict, custom_hype: dict):
    """Dynamically patches core config and retrieval functions to support CUSTOM baseline."""
    orig_get_baseline_config = core.config.get_baseline_config

    def custom_get_baseline_config(baseline: str, config_rag_components: dict) -> dict:
        if baseline == "CUSTOM":
            # Apply custom hyperparameters to the active config instance
            config.data["hyperparameters"] = copy.deepcopy(custom_hype)
            return copy.deepcopy(custom_comp)
        else:
            # Restore default/original hyperparameters for other baselines
            config.data["hyperparameters"] = copy.deepcopy(DEFAULT_HYPERPARAMS)
            return orig_get_baseline_config(baseline, config_rag_components)

    # Monkeypatch both references
    core.config.get_baseline_config = custom_get_baseline_config
    core.retrieval.get_baseline_config = custom_get_baseline_config


def evaluate_and_compare(results_file: Path, artifact_dir: Path = None) -> dict:
    """Loads results, computes retrieval metrics (Recall & Precision),
    prints comparison table and saves a Markdown report.
    """
    with open(results_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    if not data:
        con.error("No retrieval data found in results file.")
        return {}

    # Aggregate metrics per baseline
    metrics_summary = {}
    
    for case in data:
        expected = case.get("expected_papers", [])
        baselines = case.get("baselines", {})
        
        for baseline_name, b_data in baselines.items():
            if baseline_name not in metrics_summary:
                metrics_summary[baseline_name] = {
                    "recalls": [],
                    "precisions": [],
                    "latencies": [],
                    "success_count": 0,
                    "total_count": 0
                }
            
            stats = metrics_summary[baseline_name]
            stats["total_count"] += 1
            
            if b_data.get("status") == "success":
                stats["success_count"] += 1
                retrieved_papers = b_data.get("retrieved_papers", [])
                retrieved_chunks = b_data.get("retrieved_chunks", [])
                latency = b_data.get("latency_sec", 0.0)
                
                # Calculate metrics
                recall = calculate_retrieval_recall(expected, retrieved_papers)
                precision = calculate_context_precision(expected, retrieved_chunks)
                
                stats["recalls"].append(recall)
                stats["precisions"].append(precision)
                stats["latencies"].append(latency)
            else:
                stats["recalls"].append(0.0)
                stats["precisions"].append(0.0)
                stats["latencies"].append(0.0)

    # Print summary table to console
    con.info("\n=== RETRIEVAL STAGE BENCHMARK COMPARISON ===")
    
    header_fmt = "| {:<15} | {:<12} | {:<12} | {:<17} | {:<13} |"
    row_fmt = "| {:<15} | {:<12} | {:<12.4f} | {:<17.4f} | {:<12.3f}s |"
    sep = "+" + "-"*17 + "+" + "-"*14 + "+" + "-"*14 + "+" + "-"*19 + "+" + "-"*15 + "+"
    
    print(sep)
    print(header_fmt.format("Baseline", "Success Rate", "Mean Recall", "Mean Precision", "Mean Latency"))
    print(sep)
    
    for b_name in sorted(metrics_summary.keys()):
        stats = metrics_summary[b_name]
        total = stats["total_count"]
        success_rate = (stats["success_count"] / total * 100) if total > 0 else 0.0
        
        mean_recall = sum(stats["recalls"]) / len(stats["recalls"]) if stats["recalls"] else 0.0
        mean_prec = sum(stats["precisions"]) / len(stats["precisions"]) if stats["precisions"] else 0.0
        mean_lat = sum(stats["latencies"]) / len(stats["latencies"]) if stats["latencies"] else 0.0
        
        b_label = b_name
        if b_name == "CUSTOM":
            b_label = "CUSTOM (Ours)"
            
        print(row_fmt.format(b_label, f"{success_rate:.1f}%", mean_recall, mean_prec, mean_lat))
    print(sep)
    print()

    return metrics_summary


def save_markdown_report(metrics_summary: dict, custom_comp: dict, custom_hype: dict, report_path: Path):
    """Saves a rich Markdown report showing configuration diffs and metrics comparison."""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# RAG Custom Retrieval Benchmark Report\n\n")
        f.write("This report displays the retrieval stage performance of your custom configuration compared against active baselines.\n\n")
        
        # Overrides section
        f.write("## ⚙️ Custom Run Configuration Overrides\n\n")
        
        # Component overrides vs B6
        b6_comp = core.config.get_baseline_config("B6", config.rag_components)
        f.write("### Component Settings (vs B6 Full Pipeline)\n\n")
        f.write("| Component | Custom Value | B6 Default | Status |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for k in sorted(custom_comp.keys()):
            custom_val = custom_comp[k]
            b6_val = b6_comp.get(k)
            status = "🟢 **Modified**" if custom_val != b6_val else "Unchanged"
            f.write(f"| `{k}` | `{custom_val}` | `{b6_val}` | {status} |\n")
        f.write("\n")
        
        # Hyperparameter overrides vs Default
        f.write("### Hyperparameter Overrides (vs System Defaults)\n\n")
        f.write("| Parameter | Custom Value | Default Value | Status |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        
        has_hype_overrides = False
        for section in sorted(custom_hype.keys()):
            for k in sorted(custom_hype[section].keys()):
                custom_val = custom_hype[section][k]
                def_val = DEFAULT_HYPERPARAMS.get(section, {}).get(k)
                if custom_val != def_val:
                    has_hype_overrides = True
                    f.write(f"| `{section}.{k}` | `{custom_val}` | `{def_val}` | ⚡ **Overridden** |\n")
        
        if not has_hype_overrides:
            f.write("| *None* | | | | \n")
        f.write("\n\n")
        
        # Results table
        f.write("## 📊 Retrieval Performance Summary\n\n")
        f.write("| Baseline | Success Rate | Mean Recall | Context Precision | Mean Latency |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        
        for b_name in sorted(metrics_summary.keys()):
            stats = metrics_summary[b_name]
            total = stats["total_count"]
            success_rate = (stats["success_count"] / total * 100) if total > 0 else 0.0
            mean_recall = sum(stats["recalls"]) / len(stats["recalls"]) if stats["recalls"] else 0.0
            mean_prec = sum(stats["precisions"]) / len(stats["precisions"]) if stats["precisions"] else 0.0
            mean_lat = sum(stats["latencies"]) / len(stats["latencies"]) if stats["latencies"] else 0.0
            
            b_label = b_name
            if b_name == "CUSTOM":
                b_label = "🏆 **CUSTOM (Ours)**"
                
            f.write(f"| {b_label} | {success_rate:.1f}% | {mean_recall:.4f} | {mean_prec:.4f} | {mean_lat:.3f}s |\n")
            
        f.write("\n\n")
        f.write("> [!NOTE]\n")
        f.write("> - **Retrieval Recall**: proportion of expected papers retrieved.\n")
        f.write("> - **Context Precision**: Mean Average Precision of the retrieved chunks/papers.\n")


def main():
    parser = argparse.ArgumentParser(description="Run RAG retrieval with custom configurations and analyze their impact")
    
    # Standard options
    parser.add_argument(
        "--dataset", "-d", type=str, default=None,
        help="Path to golden dataset YAML file."
    )
    parser.add_argument(
        "--output", "-o", type=str, default="reports/custom_retrieved_contexts.yaml",
        help="Path to save retrieved contexts YAML."
    )
    parser.add_argument(
        "--cloud", action="store_true",
        help="Use cloud LLM engine for retrieval/generation instead of local one."
    )
    parser.add_argument(
        "--baselines", type=str, default="B4,B6",
        help="Comma-separated baselines to compare against (e.g. B4,B6) or 'all'."
    )
    parser.add_argument(
        "--no-unique-dir", action="store_true",
        help="Save output directly to the specified path without timestamp subdirectory."
    )
    parser.add_argument(
        "--config-file", "-c", type=str, default=None,
        help="Path to a custom YAML configuration file containing overrides."
    )
    parser.add_argument(
        "--artifact-dir", type=str, default=None,
        help="Optional path to write a user-facing markdown report artifact."
    )
    parser.add_argument(
        "--custom", action="store_true",
        help="Apply the hardcoded custom preset components and hyperparameters defined in this script."
    )

    # Component overrides (Boolean Optional actions)
    parser.add_argument("--intent-classifier", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--graph-ontology-lookup", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--llm-query-expansion", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--hyde", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--lexical-search", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dense-search", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dynamic-alpha-blending", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rrf", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--graph-expansion", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reranker", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--score-blending", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--context-trimming", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--citation-repair", action=argparse.BooleanOptionalAction, default=None)

    # RAG Hyperparameters overrides
    parser.add_argument("--score-blend-reranker-weight", type=float, default=None)
    parser.add_argument("--score-blend-rrf-weight", type=float, default=None)
    parser.add_argument("--rrf-k", type=float, default=None)
    parser.add_argument("--dynamic-alpha-threshold-low", type=float, default=None)
    parser.add_argument("--dynamic-alpha-val-low", type=float, default=None)
    parser.add_argument("--dynamic-alpha-threshold-mid", type=float, default=None)
    parser.add_argument("--dynamic-alpha-val-mid", type=float, default=None)
    parser.add_argument("--dynamic-alpha-val-high", type=float, default=None)

    # Graph Hyperparameters overrides
    parser.add_argument("--graph-p-base", type=float, default=None)
    parser.add_argument("--graph-gamma", type=float, default=None)
    parser.add_argument("--graph-crawl-stop-threshold", type=float, default=None)
    parser.add_argument("--graph-semantic-score-threshold", type=float, default=None)
    parser.add_argument("--graph-semantic-score-top-p", type=float, default=None)
    parser.add_argument("--graph-sigmoid-score-threshold", type=float, default=None)
    parser.add_argument("--graph-sigmoid-score-top-p", type=float, default=None)
    parser.add_argument("--graph-essential-fact-threshold", type=float, default=None)
    parser.add_argument("--graph-sigmoid-slope", type=float, default=None)
    parser.add_argument("--graph-sigmoid-center", type=float, default=None)
    parser.add_argument("--graph-weight-authored", type=float, default=None)
    parser.add_argument("--graph-weight-cites", type=float, default=None)
    parser.add_argument("--graph-weight-mentions-concept", type=float, default=None)
    parser.add_argument("--graph-weight-default", type=float, default=None)

    # BM25 Hyperparameters overrides
    parser.add_argument("--bm25-k1", type=float, default=None)
    parser.add_argument("--bm25-b", type=float, default=None)

    args = parser.parse_args()

    # Load file config if specified
    file_config = None
    if args.config_file:
        config_path = Path(args.config_file)
        if not config_path.exists():
            con.error(f"Config file not found: {config_path}")
            sys.exit(1)
        with open(config_path, "r", encoding="utf-8") as f:
            file_config = yaml.safe_load(f)

    # Build final custom configuration overrides
    custom_comp, custom_hype = build_custom_config(args, file_config)

    # Apply dynamic patch
    patch_retrieval_for_custom(custom_comp, custom_hype)

    # Determine baselines to run (force inclusion of CUSTOM)
    req_baselines = [b.strip().upper() for b in args.baselines.split(",") if b.strip()] if args.baselines != "all" else ["B4", "B6"]
    if "CUSTOM" not in req_baselines:
        req_baselines.append("CUSTOM")
    args.baselines = ",".join(req_baselines)

    con.info(f"Custom retrieval config constructed.")
    con.info(f"Enabled components: {sorted([k for k, v in custom_comp.items() if v])}")
    con.blank()

    # Run retrieval
    try:
        config.data["rag_components"] = copy.deepcopy(custom_comp)
        run_staged_retrieval(args, config, prompts, container, con)
    except Exception as e:
        con.error(f"Failed to execute retrieval: {e}")
        # Restore configuration before exit
        config.data["hyperparameters"] = copy.deepcopy(DEFAULT_HYPERPARAMS)
        config.data["rag_components"] = copy.deepcopy(DEFAULT_COMPONENTS)
        sys.exit(1)

    # Restore original config hyperparameters and components
    config.data["hyperparameters"] = copy.deepcopy(DEFAULT_HYPERPARAMS)
    config.data["rag_components"] = copy.deepcopy(DEFAULT_COMPONENTS)

    # Evaluate metrics and display comparison
    output_path = Path(args.output)
    if not output_path.exists():
        # Find path if timestamp subdirectory was added
        candidates = list(output_path.parent.glob(f"**/{output_path.name}"))
        if candidates:
            # Sort by modification time to get the newest
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            output_path = candidates[0]
        else:
            con.error(f"Could not locate output file at {output_path}")
            sys.exit(1)

    metrics_summary = evaluate_and_compare(output_path)

    # Save reports
    local_report = Path("reports/retrieval_custom_report.md")
    local_report.parent.mkdir(parents=True, exist_ok=True)
    save_markdown_report(metrics_summary, custom_comp, custom_hype, local_report)
    con.success(f"Detailed Markdown report saved to: {local_report.resolve()}")

    if args.artifact_dir:
        art_report = Path(args.artifact_dir) / "retrieval_comparison_report.md"
        save_markdown_report(metrics_summary, custom_comp, custom_hype, art_report)
        con.success(f"Artifact Markdown report saved to: {art_report.resolve()}")


if __name__ == "__main__":
    main()
