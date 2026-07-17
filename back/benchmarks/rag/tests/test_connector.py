import pytest
from pathlib import Path
from core.connector import (
    data_prep_agent,
    stats_agent,
    reporting_agent,
    build_statistical_markdown,
    export_stats_json,
    run_statistical_pipeline,
)
from core.statistics import StatsConfig


def test_data_prep_agent_structured():
    data = {
        "results": [
            {
                "id": "Q1",
                "category": "single-doc",
                "is_answerable": True,
                "baselines": {
                    "B1": {
                        "status": "success",
                        "latency_sec": 0.1,
                        "eval_metrics": {"semantic_accuracy": 0.9},
                    }
                },
            }
        ]
    }
    records, baselines = data_prep_agent(data)
    assert len(records) > 0
    assert "B1" in baselines


def test_stats_agent_disabled():
    data = {"results": []}
    config = StatsConfig(enable_stats=False)
    res = stats_agent(data, config)
    assert res["enabled"] is False


def test_build_statistical_markdown():
    stats_analysis = {
        "enabled": True,
        "config": {"alpha": 0.05, "ci_method": "percentile", "n_bootstraps": 1000},
        "filtering_note": "Filter note",
        "baselines": ["B1", "B2"],
        "baseline_summary": {
            "B1": {"semantic_accuracy": {"mean": 0.8, "ci_lower": 0.7, "ci_upper": 0.9}},
            "B2": {"semantic_accuracy": {"mean": 0.85, "ci_lower": 0.75, "ci_upper": 0.95}},
        },
    }
    md = build_statistical_markdown(stats_analysis)
    assert "Statistical Analysis" in md
    assert "B1" in md
    assert "B2" in md


def test_export_stats_json(tmp_path):
    stats_analysis = {"enabled": True, "records": [1, 2, 3], "baselines": ["B1"]}
    out_file = tmp_path / "stats.json"
    export_stats_json(stats_analysis, out_file)
    assert out_file.exists()
