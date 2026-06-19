#!/usr/bin/env python3
"""
Science Graph — RAG Quality Metrics Parser & Aggregator.
Parses result_metrics.yaml and exports CSV/Markdown reports (delegated to core).
"""

import sys
import argparse
from pathlib import Path

# Set up python path to resolve src and core imports correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.analytics import analyze_metrics
from core.reporting import (
    print_rich_tables,
    generate_markdown_report,
    export_wide_csv,
    export_detailed_csv
)


from core.models import load_report_file


def main():
    parser = argparse.ArgumentParser(description="Parse RAG quality metrics and generate reports")
    parser.add_argument(
        "--file", "-f", type=str, default="reports/result_metrics.yaml",
        help="Path to result_metrics.yaml file."
    )
    parser.add_argument(
        "--output-md", "-m", type=str, default=None,
        help="Path to save summary markdown report."
    )
    parser.add_argument(
        "--csv-summary", type=str, default=None,
        help="Path to save wide-format summary CSV."
    )
    parser.add_argument(
        "--csv-details", type=str, default=None,
        help="Path to save detailed case-by-case CSV."
    )
    args = parser.parse_args()

    input_path = Path(args.file)
    try:
        report = load_report_file(input_path)
        data = report.model_dump()
    except Exception as e:
        print(f"Error loading or validating report file: {e}")
        sys.exit(1)
    
    # Compute all metrics and statistics
    stats = analyze_metrics(data)
    
    # Print tables to stdout
    print_rich_tables(stats)
    
    # Export reports if paths are specified
    if args.output_md:
        generate_markdown_report(stats, Path(args.output_md))
        
    if args.csv_summary:
        export_wide_csv(stats, Path(args.csv_summary))
        
    if args.csv_details:
        export_detailed_csv(data, stats, Path(args.csv_details))


if __name__ == "__main__":
    main()
