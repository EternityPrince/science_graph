#!/usr/bin/env python3
"""
Science Graph — RAG Quality Evaluator (LLM-as-a-Judge + Retrieval Metrics).
Iterates over results and scores them using Cloud LLM Judge (delegated to core).
"""

import sys
import argparse
import asyncio
from pathlib import Path

# Set up python path to resolve src and core imports correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import config
from src import console as con
from core.evaluator import run_evaluation


def run_rm_metric(metric_name: str, output_path_str: str):
    import json
    import yaml
    
    ALIASES = {
        "recall": "retrieval_recall",
        "precision": "context_precision",
        "relevance": "answer_relevance",
        "semantic": "semantic_accuracy",
        "citations": "citation_fidelity",
        "fillness": "context_fillness",
    }
    
    if metric_name in ALIASES:
        metric_name = ALIASES[metric_name]
        
    script_dir = Path(__file__).resolve().parent
    output_path = Path(output_path_str)
    if not output_path.is_absolute():
        output_path = script_dir / output_path
        
    checkpoint_path = output_path.parent / ".eval_checkpoint.json"
    
    # 1. Clean checkpoint file
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
        except Exception as e:
            con.error(f"Failed to load checkpoint file: {e}")
            sys.exit(1)
            
        cleared_checkpoints_count = 0
        for key, cached in checkpoint_data.items():
            if not isinstance(cached, dict):
                continue
            if "metrics" in cached:
                metrics_dict = cached["metrics"]
                details_dict = cached.get("details", {})
                if metric_name in metrics_dict:
                    metrics_dict.pop(metric_name)
                    cleared_checkpoints_count += 1
                if metric_name in details_dict:
                    details_dict.pop(metric_name)
            else:
                if metric_name in cached:
                    cached.pop(metric_name)
                    cleared_checkpoints_count += 1
                
        if cleared_checkpoints_count > 0:
            try:
                temp_path = checkpoint_path.with_suffix(".tmp")
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                temp_path.replace(checkpoint_path)
                con.success(f"Cleared metric '{metric_name}' from {cleared_checkpoints_count} checkpoint entries.")
            except Exception as e:
                con.error(f"Failed to write updated checkpoint file: {e}")
                sys.exit(1)
        else:
            con.info(f"Metric '{metric_name}' not found in any checkpoint entries.")
    else:
        con.info(f"Checkpoint file not found: {checkpoint_path}")
        
    # 2. Clean result metrics YAML report file
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                report_data = yaml.safe_load(f)
        except Exception as e:
            con.error(f"Failed to load result metrics YAML: {e}")
            sys.exit(1)
            
        updated_report = False
        
        # Clean summary
        summary = report_data.get("summary", {}) if report_data else {}
        avg_metric_name = f"avg_{metric_name}"
        for baseline, metrics in summary.items():
            if avg_metric_name in metrics:
                metrics.pop(avg_metric_name)
                updated_report = True
                
        # Clean results
        results = report_data.get("results", []) if report_data else []
        for case in results:
            baselines = case.get("baselines", {})
            for baseline, data in baselines.items():
                eval_metrics = data.get("eval_metrics", {})
                if metric_name in eval_metrics:
                    eval_metrics.pop(metric_name)
                    updated_report = True
                    
        if updated_report:
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    yaml.dump(report_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
                con.success(f"Cleared metric '{metric_name}' from report file: {output_path}")
            except Exception as e:
                con.error(f"Failed to write updated result metrics report: {e}")
                sys.exit(1)
        else:
            con.info(f"Metric '{metric_name}' not found in report file: {output_path}")
    else:
        con.info(f"Result metrics report file not found: {output_path}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "rm":
        rm_parser = argparse.ArgumentParser(description="Remove a specific metric from checkpoints and reports to force its re-evaluation")
        rm_parser.add_argument("metric_name", type=str, help="Name of the metric to remove (e.g. faithfulness, citations, recall, etc.)")
        rm_parser.add_argument(
            "--output", "-o", type=str, default="reports/result_metrics.yaml",
            help="Path to the output result metrics YAML. Defaults to reports/result_metrics.yaml"
        )
        args = rm_parser.parse_args(sys.argv[2:])
        run_rm_metric(args.metric_name, args.output)
        return

    parser = argparse.ArgumentParser(description="Science Graph RAG Quality Evaluator (LLM-as-a-Judge)")
    parser.add_argument(
        "--input", "-i", type=str, default="reports/evaluation_results.yaml",
        help="Path to input evaluation results YAML. Defaults to reports/evaluation_results.yaml"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="reports/result_metrics.yaml",
        help="Path to save output result metrics. Defaults to reports/result_metrics.yaml"
    )
    parser.add_argument(
        "--baselines", "-b", type=str, default="all",
        help="Comma-separated baselines to evaluate (e.g. B0,B2,B6) or 'all'."
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=None,
        help="Limit the number of questions to evaluate (for testing)."
    )
    parser.add_argument(
        "--concurrency", "-c", type=int, default=config.llm_evaluation_concurrency,
        help=f"Max concurrent API calls to the cloud provider. Defaults to config ({config.llm_evaluation_concurrency})."
    )
    parser.add_argument(
        "--rpm", "-r", type=int, default=config.llm_evaluation_rpm,
        help=f"Rate limit in requests per minute (RPM). Defaults to config ({config.llm_evaluation_rpm})."
    )
    parser.add_argument(
        "--retries", type=int, default=config.llm_evaluation_retries,
        help=f"Max number of API retries on error. Defaults to config ({config.llm_evaluation_retries})."
    )
    parser.add_argument(
        "--cloud", action="store_true", default=True,
        help="Use cloud LLM engine for evaluation (always required, enabled by default)."
    )
    parser.add_argument(
        "--clear-checkpoint", action="store_true",
        help="Ignore existing evaluation checkpoints and restart from scratch."
    )
    args = parser.parse_args()

    asyncio.run(run_evaluation(args, config, con))


if __name__ == "__main__":
    main()
