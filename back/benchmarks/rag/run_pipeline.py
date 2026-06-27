#!/usr/bin/env python3
"""
Science Graph — End-to-End RAG Quality Benchmarking Pipeline.
Orchestrates the individual stages via subprocesses for optimal VRAM isolation.
"""

import sys
import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Set up python path to resolve src and core imports correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml
from src.config import config
from src import console as con
from core.config import get_safe_model_name

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_creator import (
    build_custom_config,
    add_custom_config_arguments
)


from rich.progress import ProgressColumn
from rich.text import Text

class IterationSpeedColumn(ProgressColumn):
    def render(self, task):
        speed = task.finished_speed or task.speed
        if speed is None or speed == 0:
            return Text("- sec/it", style="progress.data.speed")
        sec_per_it = 1.0 / speed
        return Text(f"{sec_per_it:.2f} sec/it", style="progress.data.speed")


def run_command_with_progress(cmd: list, title: str, total_steps: int, step_pattern: str) -> float:
    from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
    import subprocess
    import time
    import os

    t0 = time.perf_counter()
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, finished_style="green"),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        IterationSpeedColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=con.console
    ) as progress:
        task = progress.add_task(f"[cyan]{title}", total=total_steps)
        
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        
        for line in iter(process.stdout.readline, ""):
            line_str = line.strip()
            
            # Parse progress based on patterns
            if step_pattern == "retrieval":
                if "Query: '" in line_str or "] Query:" in line_str:
                    progress.advance(task, 1)
            elif step_pattern == "generation":
                if line_str.startswith("Running ") and ":" in line_str and not line_str.startswith("Running command:"):
                    progress.advance(task, 1)
            elif step_pattern == "evaluation":
                if "Evaluated case " in line_str:
                    try:
                        parts = line_str.split("Evaluated case ")[1].split("/")
                        current = int(parts[0])
                        progress.update(task, completed=current)
                    except Exception:
                        pass
            
            # Print warnings, errors or success milestones directly above the progress bar
            if "✗" in line_str or "error" in line_str.lower() or "warning" in line_str.lower() or "failed" in line_str.lower():
                progress.console.print(f"  [red]✗[/] {line_str}")
            elif "Loaded" in line_str or "Successfully" in line_str:
                progress.console.print(f"  [green]✓[/] {line_str}")
            elif "Ready" in line_str.lower() or "initializing" in line_str.lower():
                progress.console.print(f"  [dim]{line_str}[/]")
                
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
            
    return time.perf_counter() - t0

from core.pipelined import run_pipelined_stage_async

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
        "--concurrency", "-c", type=int, default=config.llm_evaluation_concurrency,
        help=f"Max concurrent API calls to the cloud provider during evaluation. Defaults to config ({config.llm_evaluation_concurrency})."
    )
    parser.add_argument(
        "--rpm", "-r", type=int, default=config.llm_evaluation_rpm,
        help=f"Rate limit in requests per minute (RPM) during evaluation. Defaults to config ({config.llm_evaluation_rpm})."
    )
    parser.add_argument(
        "--retries", type=int, default=config.llm_evaluation_retries,
        help=f"Max number of API retries on error during evaluation. Defaults to config ({config.llm_evaluation_retries})."
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
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip the LLM-as-a-Judge evaluation stage and metrics parsing."
    )
    parser.add_argument(
        "--pipelined", action="store_true",
        help="Run generation and evaluation concurrently in a pipelined fashion, writing results immediately."
    )
    add_custom_config_arguments(parser)
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

    is_already_retrieved = False
    if dataset_path.name in ["retrieved_contexts.yaml", "custom_retrieved_contexts.yaml"]:
        is_already_retrieved = True
    else:
        # fallback check: read a small part
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                head_lines = [f.readline() for _ in range(50)]
                head_text = "".join(head_lines)
                if "baselines:" in head_text:
                    is_already_retrieved = True
        except Exception:
            pass

    # 2. Determine target output directory
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (script_dir / output_dir).resolve()
        
    if is_already_retrieved:
        run_dir = dataset_path.parent
        con.info(f"Detected pre-retrieved context dataset. Skipping retrieval stage.")
        con.info(f"Outputs will be saved directly to: {run_dir}")
        try:
            con.blank()
            from run_custom_retrieve import evaluate_and_compare
            evaluate_and_compare(dataset_path)
        except Exception as e:
            con.warning(f"Could not generate retrieval metrics table: {e}")
    elif args.no_unique_dir:
        run_dir = output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.cloud:
            llm_model = config.data["llm"]["cloud"]["model_name"]
        else:
            llm_model = config.data["llm"]["local"]["model_path"]
        
        safe_model_name = get_safe_model_name(llm_model)
        run_dir = output_dir / f"run_{timestamp}_{safe_model_name}"

    run_dir.mkdir(parents=True, exist_ok=True)
    con.info(f"All run reports will be saved to: {run_dir}")

    # Build final custom configuration overrides if user requested it or specified a config file or overrides
    temp_config_file = None
    # Check if any component or hyperparameter override is set (or --custom / --config-file)
    has_overrides = args.custom or args.config_file or any(getattr(args, field, None) is not None for field in [
        "intent_classifier", "graph_ontology_lookup", "llm_query_expansion", 
        "hyde", "lexical_search", "dense_search", "dynamic_alpha_blending", 
        "rrf", "graph_expansion", "reranker", "score_blending", 
        "context_trimming", "citation_repair", "score_blend_reranker_weight",
        "score_blend_rrf_weight", "rrf_k", "dynamic_alpha_threshold_low",
        "dynamic_alpha_val_low", "dynamic_alpha_threshold_mid",
        "dynamic_alpha_val_mid", "dynamic_alpha_val_high"
    ])
    if has_overrides:
        file_config = None
        if args.config_file:
            config_path = Path(args.config_file)
            if not config_path.exists():
                con.error(f"Config file not found: {config_path}")
                sys.exit(1)
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f)
        
        custom_comp, custom_hype = build_custom_config(args, file_config)
        temp_config_file = run_dir / "temp_custom_config.yaml"
        with open(temp_config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump({
                "rag_components": custom_comp,
                "hyperparameters": custom_hype
            }, f)
        con.info(f"Generated pipeline custom configuration overrides at: {temp_config_file}")

    # Load dataset to determine total questions and total runs
    try:
        from core.config import load_benchmark_dataset
        test_cases = load_benchmark_dataset(dataset_path, limit=args.limit)
        num_cases = len(test_cases)
    except Exception:
        num_cases = 50

    # Replicate baselines list
    if args.baselines.lower() == "all":
        from core.config import BASELINES_INFO
        baselines_to_run = list(BASELINES_INFO.keys())
        if is_already_retrieved:
            present_baselines = {"B0"}
            try:
                for case in test_cases:
                    for b_name in case.get("baselines", {}).keys():
                        present_baselines.add(b_name.upper())
                baselines_to_run = [b for b in baselines_to_run if b in present_baselines]
            except Exception:
                pass
    else:
        baselines_to_run = [b.strip().upper() for b in args.baselines.split(",") if b.strip()]
        if args.custom and "CUSTOM" not in baselines_to_run:
            baselines_to_run.append("CUSTOM")

    num_baselines = len(baselines_to_run)
    baselines_with_retrieval = [b for b in baselines_to_run if b != "B0"]
    num_baselines_with_retrieval = len(baselines_with_retrieval)

    total_retrieval_steps = max(num_cases * num_baselines_with_retrieval, 1)
    total_generation_steps = max(num_cases * num_baselines, 1)
    total_evaluation_steps = max(num_cases * num_baselines, 1)

    # Output filenames
    eval_results = run_dir / "evaluation_results.yaml"
    metrics_results = run_dir / "result_metrics.yaml"
    summary_md = run_dir / "metrics_summary.md"
    summary_csv = run_dir / "metrics_summary.csv"
    details_csv = run_dir / "metrics_details.csv"
    
    # Subprocesses python interpreter
    python_bin = sys.executable

    retrieved_contexts_file = dataset_path if is_already_retrieved else run_dir / "retrieved_contexts.yaml"

    # STEP 1a: Pre-Retrieval Stage
    if not is_already_retrieved:
        con.blank()
        con.info("=== STEP 1a: Running Pre-Retrieval Stage ===")
        retrieve_cmd = [
            python_bin, str(script_dir / "run_custom_retrieve.py"),
            "--dataset", str(dataset_path),
            "--output", str(retrieved_contexts_file),
            "--baselines", args.baselines,
            "--no-unique-dir"
        ]
        retrieve_cmd.extend(["--limit", str(num_cases)])
        if args.cloud:
            retrieve_cmd.append("--cloud")
        if temp_config_file:
            retrieve_cmd.extend(["--config-file", str(temp_config_file)])

        con.dim(f"Running command: {' '.join(retrieve_cmd)}")
        try:
            elapsed_ret = run_command_with_progress(retrieve_cmd, "Pre-Retrieval Stage", total_retrieval_steps, "retrieval")
            con.success(f"Pre-Retrieval completed in {elapsed_ret:.2f} seconds.")
            try:
                con.blank()
                from run_custom_retrieve import evaluate_and_compare
                evaluate_and_compare(retrieved_contexts_file)
            except Exception as e:
                con.warning(f"Could not generate retrieval metrics table: {e}")
        except subprocess.CalledProcessError as e:
            con.error(f"Pre-Retrieval failed with exit code {e.returncode}.")
            if temp_config_file and temp_config_file.exists():
                try:
                    temp_config_file.unlink()
                except Exception:
                    pass
            sys.exit(e.returncode)

    if args.pipelined:
        import asyncio
        con.blank()
        con.info("=== Running Pipelined RAG Generation and Evaluation ===")
        
        # Apply custom config overrides to this process if we have them
        if has_overrides:
            from config_creator import patch_config_for_custom
            patch_config_for_custom(custom_comp, custom_hype)
            
        t0 = time.perf_counter()
        try:
            asyncio.run(
                run_pipelined_stage_async(
                    args,
                    config,
                    run_dir,
                    dataset_path,
                    baselines_to_run,
                    eval_results,
                    metrics_results,
                    retrieved_contexts_file,
                    total_generation_steps
                )
            )
            con.success(f"Pipelined execution completed in {time.perf_counter() - t0:.2f} seconds.")
        except Exception as e:
            con.error(f"Pipelined run failed: {e}")
            if temp_config_file and temp_config_file.exists():
                try:
                    temp_config_file.unlink()
                except Exception:
                    pass
            sys.exit(1)
            
        if not args.skip_eval:
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
        else:
            con.blank()
            con.info("=== Skipping LLM-as-a-Judge Evaluation & Metrics Parsing (Pipeline Optimized) ===")
    else:
        # STEP 1b: RAG Generation Stage (consuming pre-retrieved contexts)
        con.blank()
        con.info("=== STEP 1b: Running RAG Generation Stage ===")
        gen_cmd = [
            python_bin, str(script_dir / "run_benchmarks.py"),
            "--dataset", str(dataset_path),
            "--output", str(eval_results),
            "--baselines", ",".join(baselines_to_run),
            "--consume-contexts", str(retrieved_contexts_file),
            "--no-unique-dir"
        ]
        gen_cmd.extend(["--limit", str(num_cases)])
        if args.cloud:
            gen_cmd.append("--cloud")
        if temp_config_file:
            gen_cmd.extend(["--config-file", str(temp_config_file)])

        con.dim(f"Running command: {' '.join(gen_cmd)}")
        try:
            elapsed_gen = run_command_with_progress(gen_cmd, "RAG Generation Stage", total_generation_steps, "generation")
            con.success(f"RAG Generation completed in {elapsed_gen:.2f} seconds.")
        except subprocess.CalledProcessError as e:
            con.error(f"RAG Generation failed with exit code {e.returncode}.")
            if temp_config_file and temp_config_file.exists():
                try:
                    temp_config_file.unlink()
                except Exception:
                    pass
            sys.exit(e.returncode)

        if not args.skip_eval:
            # STEP 2: LLM-as-a-Judge Evaluation
            con.blank()
            con.info("=== STEP 2: Running LLM-as-a-Judge Evaluation ===")
            eval_cmd = [
                python_bin, str(script_dir / "run_evaluator.py"),
                "--input", str(eval_results),
                "--output", str(metrics_results),
                "--baselines", ",".join(baselines_to_run),
                "--concurrency", str(args.concurrency),
                "--rpm", str(args.rpm),
                "--retries", str(args.retries)
            ]
            eval_cmd.extend(["--limit", str(num_cases)])
            if args.clear_checkpoint:
                eval_cmd.append("--clear-checkpoint")

            con.dim(f"Running command: {' '.join(eval_cmd)}")
            try:
                elapsed_eval = run_command_with_progress(eval_cmd, "LLM-as-a-Judge Evaluation Stage", total_evaluation_steps, "evaluation")
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
        else:
            con.blank()
            con.info("=== Skipping LLM-as-a-Judge Evaluation & Metrics Parsing (Pipeline Optimized) ===")

    # Clean up temporary configuration file if created
    if temp_config_file and temp_config_file.exists():
        try:
            temp_config_file.unlink()
        except Exception:
            pass

    con.blank()
    con.success("=== PIPELINE RUN COMPLETE ===")
    if not args.skip_eval:
        con.info(f"Markdown Summary: {summary_md}")
        con.info(f"Wide CSV Summary (Typst): {summary_csv}")
        con.info(f"Detailed CSV (Pandas): {details_csv}")
    else:
        con.info(f"Evaluation results yaml saved to: {eval_results}")


if __name__ == "__main__":
    main()
