import pytest
from pathlib import Path
import numpy as np
import shutil
from unittest.mock import patch

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
            core_cat_val = stats["category_stats"][cat][b].get(m, 0.0) * 100
            df_cat_val = df[(df["baseline"] == b) & (df["category"] == cat)][m].mean() * 100
            assert np.isclose(core_cat_val, df_cat_val), f"Category mismatch for {b} {cat} {m}: stats={core_cat_val:.4f}, df={df_cat_val:.4f}"

    print("Success: Visualization dataframe aggregation matches core analytics stats exactly!")

def test_directory_input_resolves_correctly(tmp_path):
    # Setup temporary directory representing a run
    run_dir = tmp_path / "run_test_visualizations"
    run_dir.mkdir()
    
    # Write a mock result_metrics.yaml to this run_dir
    smoke_yaml_path = Path(__file__).resolve().parents[1] / "reports" / "result_metrics_smoke.yaml"
    
    # If the smoke yaml doesn't exist, we skip
    if not smoke_yaml_path.exists():
        pytest.skip("result_metrics_smoke.yaml not found, skipping integration test")
        
    shutil.copy(smoke_yaml_path, run_dir / "result_metrics.yaml")
    
    # Call main in generate_scientific_visualizations with the directory input
    from generate_scientific_visualizations import main
    
    test_args = ["generate_scientific_visualizations.py", "--input", str(run_dir)]
    with patch("sys.argv", test_args):
        main()
        
    # Check that figures folder was created inside run_dir
    figures_dir = run_dir / "figures"
    assert figures_dir.exists()
    
    # Assert at least some figures and captions.md are generated
    assert (figures_dir / "captions.md").exists()
    assert (figures_dir / "fig1_heatmap.png").exists()
    assert (figures_dir / "fig15_multihop_coverage.png").exists()

def test_directory_input_prefers_result_metrics(tmp_path):
    run_dir = tmp_path / "run_test_prefers"
    run_dir.mkdir()
    
    smoke_yaml_path = Path(__file__).resolve().parents[1] / "reports" / "result_metrics_smoke.yaml"
    if not smoke_yaml_path.exists():
        pytest.skip("result_metrics_smoke.yaml not found, skipping integration test")
        
    shutil.copy(smoke_yaml_path, run_dir / "result_metrics.yaml")
    
    # evaluation_results.yaml has invalid YAML to cause a parse crash if it is loaded
    with open(run_dir / "evaluation_results.yaml", "w") as f:
        f.write("invalid: [unclosed bracket")
        
    # If the script incorrectly loads evaluation_results.yaml, it will crash.
    # If it correctly prefers result_metrics.yaml, it will run successfully.
    from generate_scientific_visualizations import main
    test_args = ["generate_scientific_visualizations.py", "--input", str(run_dir)]
    with patch("sys.argv", test_args):
        main()
        
    # Check that figures folder was created inside run_dir
    figures_dir = run_dir / "figures"
    assert figures_dir.exists()
    assert (figures_dir / "captions.md").exists()
