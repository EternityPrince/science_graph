"""
Integration layer between parse_metrics.py and core/statistics.py.
Facade module delegating to core.connector.
"""

from __future__ import annotations

from core.connector import (
    data_prep_agent,
    stats_agent,
    reporting_agent,
    build_statistical_markdown,
    generate_statistical_plots,
    run_statistical_pipeline,
    export_stats_json,
)

__all__ = [
    "data_prep_agent",
    "stats_agent",
    "reporting_agent",
    "build_statistical_markdown",
    "generate_statistical_plots",
    "run_statistical_pipeline",
    "export_stats_json",
]