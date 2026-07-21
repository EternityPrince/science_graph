#!/usr/bin/env python3
"""
Science Graph — RAG Staged Retrieval Wrapper.
Runs the retrieval benchmark process in non-overlapping stages (delegated to core).
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
from core.retrieval import run_staged_retrieval


def main():
    parser = argparse.ArgumentParser(description="Staged RAG Benchmarking Retrieval and VRAM Optimizer")
    parser.add_argument(
        "--dataset", "-d", type=str, default=None,
        help="Path to golden dataset YAML file."
    )
    parser.add_argument(
        "--output", "-o", type=str, default="reports/retrieved_contexts.yaml",
        help="Path to save retrieved contexts YAML."
    )
    parser.add_argument(
        "--cloud", action="store_true",
        help="Use cloud LLM engine for retrieval/generation instead of local one."
    )
    parser.add_argument(
        "--baselines", type=str, default="all",
        help="Comma-separated baselines to run (e.g. B0,B2,B6) or 'all'."
    )
    parser.add_argument(
        "--no-unique-dir", action="store_true",
        help="Save output directly to the specified path without timestamp subdirectory."
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=None,
        help="Limit the number of questions to evaluate (for testing/SciQ default)."
    )
    parser.add_argument(
        "--unanswerable-limit", "-u", type=int, default=None,
        help="Limit the number of unanswerable questions (is_answerable: false) to include from the dataset."
    )
    args = parser.parse_args()

    run_staged_retrieval(args, config, prompts, container, con)


if __name__ == "__main__":
    main()
