#!/usr/bin/env python3
"""
Science Graph — RAG Quality Benchmarking Runner.
Runs a golden dataset against baseline configurations (delegated to core).
"""

import sys
import argparse
from pathlib import Path

# Set up python path to resolve src and core imports correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.container import container
from src.config import config
from src import console as con
from src.prompts import prompts

# Backward compatibility exports in case other files import them from here
from core.config import BASELINES_INFO, get_baseline_config
from core.stats import BenchmarkStatsCollector
from core.metrics import calculate_retrieval_recall, calculate_context_precision
from core.generation import run_benchmarking


def main():
    parser = argparse.ArgumentParser(description="Science Graph RAG Baselines Benchmarking Runner")
    parser.add_argument(
        "--dataset", "-d", type=str, default=None,
        help="Path to golden dataset YAML file. Defaults to golden_dataset.yaml or golden_dataset.example.yaml"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="reports/evaluation_results.yaml",
        help="Path to save evaluation output results."
    )
    parser.add_argument(
        "--cloud", action="store_true",
        help="Use cloud LLM engine instead of local one."
    )
    parser.add_argument(
        "--baselines", type=str, default="all",
        help="Comma-separated baselines to run (e.g. B0,B2,B6) or 'all'."
    )
    parser.add_argument(
        "--no-unique-dir", action="store_true",
        help="Save reports directly to the output path without creating a unique timestamped subdirectory (enables merging of results)."
    )
    parser.add_argument(
        "--consume-contexts", type=str, default=None,
        help="Path to pre-retrieved contexts YAML file to consume and bypass retrieval stages."
    )
    args = parser.parse_args()

    run_benchmarking(args, config, prompts, container, con)


if __name__ == "__main__":
    main()
