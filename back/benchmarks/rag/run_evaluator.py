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


def main():
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
