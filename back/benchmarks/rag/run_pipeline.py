#!/usr/bin/env python3
"""
Science Graph — End-to-End RAG Quality Benchmarking Pipeline.
Orchestrates:
1. Local/Cloud RAG generation (run_benchmarks.py)
2. Cloud LLM-as-a-Judge evaluation (run_evaluator.py)
3. Quality metrics parsing and CSV/Markdown reports (parse_metrics.py)
"""

import sys
import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Set up python path to resolve src imports correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import config
from src import console as con


def get_safe_model_name(model_name: str) -> str:
    name = Path(model_name).name
    name = name.replace("/", "_").replace(":", "_").replace(" ", "_")
    return name


def main():
    parser = argparse.ArgumentParser(description="End-to-End RAG Benchmarking Pipeline")
    parser.add_argument(
        "--dataset", "-d", type=str, default=None,
        help="Path to golden dataset YAML file."
    )
    parser.add_argument(
        "--baselines", "-b", type=str, default="all",
        help="Comma-separated baselines to run (e.g. B0,B2,B6) or 'all'."
    )
    parser.add_argument(
        "--cloud", action="store_true",
        help="Use cloud LLM engine for generation instead of local one."
    )
    parser.add_argument(
        "--concurrency", "-c", type=int, default=3,
        help="Max concurrent API calls to the cloud provider during evaluation."
    )
    parser.add_argument(
        "--rpm", "-r", type=int, default=60,
        help="Rate limit in requests per minute (RPM) during evaluation."
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=None,
        help="Limit the number of questions to evaluate (for testing)."
    )
    parser.add_argument(
        "--clear-checkpoint", action="store_true",
        help="Ignore existing evaluation checkpoints and restart from scratch."
    )
    parser.add_argument(
        "--output-dir", type=str, default="reports",
        help="Directory to save all run outputs."
    )
    parser.add_argument(
        "--no-unique-dir", action="store_true",
        help="Save results directly into output-dir without creating a unique timestamped subdirectory."
    )
    args = parser.parse_args()

    # Determine paths relative to this script
    script_dir = Path(__file__).resolve().parent
    
    # 1. Resolve dataset
    dataset_path = args.dataset
    if not dataset_path:
        dataset_path = script_dir / "golden_dataset.yaml"
        if not dataset_path.exists():
            dataset_path = script_dir / "golden_dataset.example.yaml"
    
    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.exists():
        con.error(f"Dataset file not found: {dataset_path}")
        sys.exit(1)

    # 2. Determine target output directory
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (script_dir / output_dir).resolve()
        
    if args.no_unique_dir:
        run_dir = output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        llm_provider = config.data["llm"]["provider"]
        if args.cloud:
            llm_model = config.data["llm"]["cloud"]["model_name"]
        else:
            llm_model = config.data["llm"]["local"]["model_path"]
        
        safe_model_name = get_safe_model_name(llm_model)
        run_dir = output_dir / f"run_{timestamp}_{safe_model_name}"

    run_dir.mkdir(parents=True, exist_ok=True)
    con.info(f"All run reports will be saved to: {run_dir}")

    # Output filenames
    eval_results = run_dir / "evaluation_results.yaml"
    metrics_results = run_dir / "result_metrics.yaml"
    summary_md = run_dir / "metrics_summary.md"
    summary_csv = run_dir / "metrics_summary.csv"
    details_csv = run_dir / "metrics_details.csv"
    
    # Subprocesses python interpreter
    python_bin = sys.executable

    # STEP 1: RAG Generation
    con.blank()
    con.info("=== STEP 1: Running RAG Generation ===")
    gen_cmd = [
        python_bin, str(script_dir / "run_benchmarks.py"),
        "--dataset", str(dataset_path),
        "--output", str(eval_results),
        "--baselines", args.baselines,
        "--no-unique-dir"
    ]
    if args.cloud:
        gen_cmd.append("--cloud")

    con.dim(f"Running command: {' '.join(gen_cmd)}")
    t0_gen = time.perf_counter()
    try:
        subprocess.run(gen_cmd, check=True)
        elapsed_gen = time.perf_counter() - t0_gen
        con.success(f"RAG Generation completed in {elapsed_gen:.2f} seconds.")
    except subprocess.CalledProcessError as e:
        con.error(f"RAG Generation failed with exit code {e.returncode}.")
        sys.exit(e.returncode)

    # STEP 2: LLM-as-a-Judge Evaluation
    con.blank()
    con.info("=== STEP 2: Running LLM-as-a-Judge Evaluation ===")
    eval_cmd = [
        python_bin, str(script_dir / "run_evaluator.py"),
        "--input", str(eval_results),
        "--output", str(metrics_results),
        "--baselines", args.baselines,
        "--concurrency", str(args.concurrency),
        "--rpm", str(args.rpm)
    ]
    if args.limit is not None:
        eval_cmd.extend(["--limit", str(args.limit)])
    if args.clear_checkpoint:
        eval_cmd.append("--clear-checkpoint")

    con.dim(f"Running command: {' '.join(eval_cmd)}")
    t0_eval = time.perf_counter()
    try:
        subprocess.run(eval_cmd, check=True)
        elapsed_eval = time.perf_counter() - t0_eval
        con.success(f"LLM-as-a-Judge Evaluation completed in {elapsed_eval:.2f} seconds.")
    except subprocess.CalledProcessError as e:
        con.error(f"LLM-as-a-Judge Evaluation failed with exit code {e.returncode}.")
        sys.exit(e.returncode)

    # STEP 3: Quality Metrics Parsing & Exporting CSVs
    con.blank()
    con.info("=== STEP 3: Parsing Metrics and Exporting Reports ===")
    parse_cmd = [
        python_bin, str(script_dir / "parse_metrics.py"),
        "--file", str(metrics_results),
        "--output-md", str(summary_md),
        "--csv-summary", str(summary_csv),
        "--csv-details", str(details_csv)
    ]
    con.dim(f"Running command: {' '.join(parse_cmd)}")
    try:
        subprocess.run(parse_cmd, check=True)
        con.success("Metrics parsing and CSV exports completed successfully.")
    except subprocess.CalledProcessError as e:
        con.error(f"Metrics parsing/reporting failed with exit code {e.returncode}.")
        sys.exit(e.returncode)

    con.blank()
    con.success("=== PIPELINE RUN COMPLETE ===")
    con.info(f"Markdown Summary: {summary_md}")
    con.info(f"Wide CSV Summary (Typst): {summary_csv}")
    con.info(f"Detailed CSV (Pandas): {details_csv}")


if __name__ == "__main__":
    main()
