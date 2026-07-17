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

# Set up python path to resolve src and core imports correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.services.container import container
from src.config import config
from src import console as con
from src.prompts import prompts
from core.retrieval import run_staged_retrieval
from core.metrics import calculate_retrieval_recall, calculate_context_precision
import core.config
import core.retrieval

from core.config import (
    DEFAULT_COMPONENTS,
    DEFAULT_HYPERPARAMS,
    build_custom_config,
    patch_config_for_custom as patch_retrieval_for_custom,
    add_custom_config_arguments,
)


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
        "--artifact-dir", type=str, default=None,
        help="Optional path to write a user-facing markdown report artifact."
    )
    add_custom_config_arguments(parser)
    parser.add_argument(
        "--limit", "-l", type=int, default=-1,
        help="Limit the number of questions to evaluate (default: -1 which means no limit/all)."
    )

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

    con.info("Custom retrieval config constructed.")
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
