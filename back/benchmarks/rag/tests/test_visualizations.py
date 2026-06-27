import pytest
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models import load_report_file
from core.analytics import analyze_metrics
from generate_scientific_visualizations import load_data, METRICS

def test_metrics_aggregation_consistency():
    yaml_path = Path(__file__).resolve().parents[1] / "reports" / "result_metrics.yaml"
    assert yaml_path.exists(), "result_metrics.yaml must exist in reports/ directory to run this test"
    
    # 1. Load using core analytics
    report = load_report_file(yaml_path)
    data = report.model_dump()
    stats = analyze_metrics(data)
    
    # 2. Load using visualization load_data
    df = load_data(yaml_path)
    
    # Check that baseline-wise means match exactly
    for b in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']:
        # Overall summary check
        for m in METRICS:
            core_mean = stats["summary"][b][m]["mean"] * 100
            df_mean = df[df["baseline"] == b][m].mean() * 100
            assert np.isclose(core_mean, df_mean), f"Overall mismatch for {b} {m}: stats={core_mean:.4f}, df={df_mean:.4f}"
            
        # Category breakdown check
        for cat in ['single-document', 'multi-hop']:
            cat_label = 'Single-hop' if cat == 'single-document' else 'Multi-hop'
            core_cat_val = stats["category_stats"][cat][b].get(m, 0.0) * 100
            df_cat_val = df[(df["baseline"] == b) & (df["category"] == cat)][m].mean() * 100
            assert np.isclose(core_cat_val, df_cat_val), f"Category mismatch for {b} {cat} {m}: stats={core_cat_val:.4f}, df={df_cat_val:.4f}"

    print("Success: Visualization dataframe aggregation matches core analytics stats exactly!")
