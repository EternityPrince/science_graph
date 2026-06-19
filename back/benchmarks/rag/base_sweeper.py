#!/usr/bin/env python3
"""
Science Graph — Abstract Hyperparameter Sweeper.
Provides a base class to run multiple retrieval experiments with different configurations
and compare their results.
"""

import abc
import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import yaml

# Set up python path to resolve src and core imports correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.container import container
from src.config import config
from src import console as con
from src.prompts import prompts
from core.retrieval import run_staged_retrieval
from core.metrics import calculate_retrieval_recall, calculate_context_precision

# Import helper functions from run_custom_retrieve to avoid code duplication
import run_custom_retrieve


class BaseHyperparameterSweeper(abc.ABC):
    """
    Abstract base class for running hyperparameter sweeps and component configuration search
    on the custom retrieval pipeline.
    """

    def __init__(
        self,
        dataset_path: str = None,
        output_dir: str = "reports/sweeps",
        cloud: bool = False,
        baselines: str = "CUSTOM",
    ):
        """
        Initialize the sweeper.

        Args:
            dataset_path: Path to the golden dataset YAML file. If None, falls back to default.
            output_dir: Directory where the run results and final reports will be saved.
            cloud: If True, uses the cloud LLM engine instead of the local one.
            baselines: Comma-separated baselines to include in each run (e.g., 'CUSTOM,B6').
        """
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.cloud = cloud
        self.baselines = baselines

    @abc.abstractmethod
    def get_runs(self) -> List[Dict[str, Any]]:
        """
        Abstract method to generate the list of configurations to sweep over.

        Each dictionary in the returned list should have the following structure:
        {
            "name": "experiment_name",
            "components": {
                "reranker": True,
                "score_blending": True,
                # ... other components to override
            },
            "hyperparameters": {
                "rag": {
                    "score_blend_reranker_weight": 0.8,
                    # ... other RAG hyperparameters
                },
                "graph": {
                    "p_base": 0.5,
                    # ... other Graph hyperparameters
                },
                "bm25": {
                    "k1": 1.5,
                    # ... other BM25 hyperparameters
                }
            }
        }

        Returns:
            List[Dict[str, Any]]: A list of run configurations.
        """
        pass

    def run_sweep(self, use_subprocess: bool = False) -> List[Dict[str, Any]]:
        """
        Runs the hyperparameter sweep across all configurations defined in get_runs().

        Args:
            use_subprocess: If True, executes each run in a separate Python subprocess.
                             This is safer for GPU memory clearing but slightly slower.
                             If False, executes programmatically in the current process.

        Returns:
            List[Dict[str, Any]]: The results of all runs including configuration details and performance metrics.
        """
        runs = self.get_runs()
        if not runs:
            con.error("No run configurations returned by get_runs().")
            return []

        con.info(f"Starting hyperparameter sweep with {len(runs)} configurations.")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sweep_results = []

        for idx, run_cfg in enumerate(runs):
            run_name = run_cfg.get("name", f"run_{idx}")
            con.info(f"\n=========================================")
            con.info(f"Running Experiment [{idx + 1}/{len(runs)}]: {run_name}")
            con.info(f"=========================================")

            # Setup directory for this specific run's outputs
            run_output_path = self.output_dir / f"{run_name}_contexts.yaml"

            try:
                if use_subprocess:
                    metrics = self._run_via_subprocess(run_cfg, run_output_path)
                else:
                    metrics = self._run_programmatically(run_cfg, run_output_path)

                # Store result
                run_result = {
                    "name": run_name,
                    "config": run_cfg,
                    "metrics": metrics,
                    "status": "success"
                }
                sweep_results.append(run_result)
                con.success(f"Experiment '{run_name}' completed successfully.")

            except Exception as e:
                con.error(f"Experiment '{run_name}' failed: {e}")
                sweep_results.append({
                    "name": run_name,
                    "config": run_cfg,
                    "status": "failed",
                    "error": str(e)
                })

        # Save summary report
        if sweep_results:
            self._generate_summary_report(sweep_results)

        return sweep_results

    def _run_via_subprocess(self, run_cfg: Dict[str, Any], output_path: Path) -> Dict[str, Any]:
        """Runs the benchmark for a configuration by spawning a python subprocess."""
        # Create a temporary config YAML file
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml.dump(run_cfg, f, default_flow_style=False)
            temp_config_path = Path(f.name)

        try:
            # Build CLI command
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "run_custom_retrieve.py"),
                "--config-file", str(temp_config_path),
                "--output", str(output_path),
                "--baselines", self.baselines,
                "--no-unique-dir",
            ]
            if self.dataset_path:
                cmd.extend(["--dataset", str(self.dataset_path)])
            if self.cloud:
                cmd.append("--cloud")

            con.info(f"Spawning subprocess: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Print output log of subprocess
            print(result.stdout)

            # Evaluate metrics from the output file
            if not output_path.exists():
                raise FileNotFoundError(f"Subprocess output file was not created at {output_path}")

            metrics_summary = run_custom_retrieve.evaluate_and_compare(output_path)
            return self._extract_summary_metrics(metrics_summary)

        finally:
            # Clean up the temporary config file
            if temp_config_path.exists():
                temp_config_path.unlink()

    def _run_programmatically(self, run_cfg: Dict[str, Any], output_path: Path) -> Dict[str, Any]:
        """Runs the benchmark programmatically within the same process."""
        # Setup mock argparse arguments
        args = argparse.Namespace(
            dataset=self.dataset_path,
            output=str(output_path),
            cloud=self.cloud,
            baselines=self.baselines,
            no_unique_dir=True,
            config_file=None,
            custom=True,  # Ensure we set custom flag to merge our configurations
        )

        # Build configurations
        custom_comp, custom_hype = run_custom_retrieve.build_custom_config(args, run_cfg)

        # Dynamically patch retrieval config
        run_custom_retrieve.patch_retrieval_for_custom(custom_comp, custom_hype)

        # Apply settings to the global config data
        orig_comp = copy.deepcopy(config.data.get("rag_components", {}))
        orig_hype = copy.deepcopy(config.data.get("hyperparameters", {}))

        try:
            config.data["rag_components"] = copy.deepcopy(custom_comp)
            # Run staged retrieval
            run_staged_retrieval(args, config, prompts, container, con)
        finally:
            # Clean up and restore original configuration parameters
            config.data["rag_components"] = orig_comp
            config.data["hyperparameters"] = orig_hype

        # Evaluate metrics from the output file
        if not output_path.exists():
            raise FileNotFoundError(f"Output file was not created at {output_path}")

        metrics_summary = run_custom_retrieve.evaluate_and_compare(output_path)
        return self._extract_summary_metrics(metrics_summary)

    def _extract_summary_metrics(self, metrics_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts and averages metrics from raw metrics summary dict."""
        results = {}
        for baseline, stats in metrics_summary.items():
            total = stats["total_count"]
            success_rate = (stats["success_count"] / total * 100) if total > 0 else 0.0
            mean_recall = sum(stats["recalls"]) / len(stats["recalls"]) if stats["recalls"] else 0.0
            mean_precision = sum(stats["precisions"]) / len(stats["precisions"]) if stats["precisions"] else 0.0
            mean_latency = sum(stats["latencies"]) / len(stats["latencies"]) if stats["latencies"] else 0.0

            results[baseline] = {
                "success_rate": success_rate,
                "mean_recall": mean_recall,
                "mean_precision": mean_precision,
                "mean_latency": mean_latency,
            }
        return results

    def _generate_summary_report(self, sweep_results: List[Dict[str, Any]]):
        """Generates a markdown comparison report and prints it to the console."""
        report_path = self.output_dir / "sweep_summary_report.md"
        
        # Sort results: success first, then by custom mean recall descending, then custom mean precision descending
        def sort_key(res):
            if res.get("status") != "success":
                return (-1, 0, 0)
            custom_metrics = res["metrics"].get("CUSTOM", {})
            return (1, custom_metrics.get("mean_recall", 0.0), custom_metrics.get("mean_precision", 0.0))

        sorted_results = sorted(sweep_results, key=sort_key, reverse=True)

        # Generate markdown content
        md_content = []
        md_content.append("# 📊 Hyperparameter Sweep Summary Report\n")
        md_content.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        md_content.append(f"Dataset: `{self.dataset_path or 'Default Golden Dataset'}`\n")
        md_content.append(f"Cloud mode: `{self.cloud}`\n\n")

        md_content.append("## 🏆 Configuration Comparison Table\n\n")
        md_content.append("| Rank | Experiment Name | Status | CUSTOM Recall | CUSTOM Precision | CUSTOM Latency | Success Rate |\n")
        md_content.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")

        for rank, res in enumerate(sorted_results, 1):
            name = res["name"]
            status = res["status"]
            if status == "success":
                custom = res["metrics"].get("CUSTOM", {})
                recall = f"{custom.get('mean_recall', 0.0):.4f}"
                precision = f"{custom.get('mean_precision', 0.0):.4f}"
                latency = f"{custom.get('mean_latency', 0.0):.3f}s"
                success = f"{custom.get('success_rate', 0.0):.1f}%"
                status_str = "🟢 Success"
            else:
                recall = "N/A"
                precision = "N/A"
                latency = "N/A"
                success = "N/A"
                status_str = f"🔴 Failed ({res.get('error', 'unknown error')[:20]}...)"
            
            # Highlight best run
            if rank == 1 and status == "success":
                name = f"⭐ **{name}**"

            md_content.append(f"| {rank} | {name} | {status_str} | {recall} | {precision} | {latency} | {success} |\n")

        md_content.append("\n\n## 🔍 Detailed Configurations\n\n")
        
        for rank, res in enumerate(sorted_results, 1):
            name = res["name"]
            status = res["status"]
            md_content.append(f"### {rank}. {name} ({status.upper()})\n\n")
            
            if status == "success":
                # Print components and hyperparameters that differ from system defaults
                md_content.append("#### Configuration Overrides:\n")
                md_content.append("```yaml\n")
                # Filter out empty or none values to keep it clean
                clean_cfg = {}
                if "components" in res["config"]:
                    clean_cfg["components"] = res["config"]["components"]
                if "hyperparameters" in res["config"]:
                    clean_cfg["hyperparameters"] = res["config"]["hyperparameters"]
                
                md_content.append(yaml.dump(clean_cfg, default_flow_style=False))
                md_content.append("```\n")

                # Show comparison with other baselines run
                md_content.append("#### Metrics:\n")
                md_content.append("| Baseline | Success Rate | Mean Recall | Mean Precision | Mean Latency |\n")
                md_content.append("| :--- | :---: | :---: | :---: | :---: |\n")
                for baseline, m in res["metrics"].items():
                    b_label = f"**CUSTOM (Ours)**" if baseline == "CUSTOM" else baseline
                    md_content.append(
                        f"| {b_label} | {m['success_rate']:.1f}% | {m['mean_recall']:.4f} | {m['mean_precision']:.4f} | {m['mean_latency']:.3f}s |\n"
                    )
                md_content.append("\n")
            else:
                md_content.append(f"Error details: `{res.get('error')}`\n\n")

        # Save to file
        with open(report_path, "w", encoding="utf-8") as f:
            f.writelines(md_content)

        con.info("\n=== SWEEP RUN COMPLETED ===")
        con.success(f"Summary report saved to: {report_path.resolve()}")
        
        # Display the comparison table on console
        print("".join(md_content[:15]))

    @classmethod
    def main(cls):
        """
        Convenience CLI runner method to allow subclasses to be executed directly from terminal.
        """
        parser = argparse.ArgumentParser(description="Run a RAG custom hyperparameter sweep")
        parser.add_argument(
            "--dataset", "-d", type=str, default=None,
            help="Path to golden dataset YAML file."
        )
        parser.add_argument(
            "--output-dir", "-o", type=str, default="reports/sweeps",
            help="Directory to save the sweep reports."
        )
        parser.add_argument(
            "--cloud", action="store_true",
            help="Use cloud LLM engine for retrieval/generation."
        )
        parser.add_argument(
            "--baselines", type=str, default="CUSTOM",
            help="Comma-separated baselines to compare each run against (e.g. 'CUSTOM,B4,B6')."
        )
        parser.add_argument(
            "--subprocess", "-s", action="store_true",
            help="Execute each run in a separate subprocess to avoid CUDA/MPS memory accumulation."
        )

        args = parser.parse_args()

        # Instantiate the subclass
        sweeper = cls(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            cloud=args.cloud,
            baselines=args.baselines,
        )

        # Run sweep
        sweeper.run_sweep(use_subprocess=args.subprocess)
