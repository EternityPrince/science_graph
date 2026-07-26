"""Unit and integration tests for core/statistics.py and metrics_stats_connector.py."""

import math
import warnings

import numpy as np
import pytest

from core.statistics import (
    StatsConfig,
    apply_multiple_comparison_correction,
    bootstrap_ci,
    bootstrap_paired_difference_ci,
    compute_classification_metrics,
    compute_mcc,
    compute_statistical_analysis,
    filter_answered_quality_df,
    friedman_omnibus_test,
    holm_correction,
    mcnemar_answerability_test,
    paired_metric_vectors,
    prepare_per_query_records,
    rank_biserial_effect_size,
    significance_stars,
    wilcoxon_signed_rank_test,
)
from metrics_stats_connector import (
    build_statistical_markdown,
    data_prep_agent,
    run_statistical_pipeline,
    stats_agent,
)


def _mock_benchmark_data_two_baselines() -> dict:
    return {
        "results": [
            {
                "id": "Q1",
                "is_answerable": True,
                "baselines": {
                    "B1": {
                        "status": "success",
                        "eval_metrics": {
                            "answerability_outcome": "TP",
                            "semantic_accuracy": 0.9,
                            "faithfulness": 0.85,
                            "answer_relevance": 0.8,
                        },
                    },
                    "B2": {
                        "status": "success",
                        "eval_metrics": {
                            "answerability_outcome": "TP",
                            "semantic_accuracy": 0.7,
                            "faithfulness": 0.75,
                            "answer_relevance": 0.7,
                        },
                    },
                },
            },
            {
                "id": "Q2",
                "is_answerable": True,
                "baselines": {
                    "B1": {
                        "status": "success",
                        "eval_metrics": {
                            "answerability_outcome": "TP",
                            "semantic_accuracy": 0.8,
                            "faithfulness": 0.8,
                            "answer_relevance": 0.75,
                        },
                    },
                    "B2": {
                        "status": "success",
                        "eval_metrics": {
                            "answerability_outcome": "TP",
                            "semantic_accuracy": 0.6,
                            "faithfulness": 0.7,
                            "answer_relevance": 0.65,
                        },
                    },
                },
            },
            {
                "id": "Q3",
                "is_answerable": False,
                "baselines": {
                    "B1": {
                        "status": "success",
                        "eval_metrics": {
                            "answerability_outcome": "TN",
                            "predicted_abstained": True,
                            "semantic_accuracy": None,
                        },
                    },
                    "B2": {
                        "status": "success",
                        "eval_metrics": {
                            "answerability_outcome": "FP",
                            "predicted_abstained": False,
                            "semantic_accuracy": 0.5,
                        },
                    },
                },
            },
            {
                "id": "Q4",
                "is_answerable": True,
                "baselines": {
                    "B1": {
                        "status": "success",
                        "eval_metrics": {
                            "answerability_outcome": "FN",
                            "predicted_abstained": True,
                            "semantic_accuracy": 0.0,
                        },
                    },
                    "B2": {
                        "status": "success",
                        "eval_metrics": {
                            "answerability_outcome": "TP",
                            "semantic_accuracy": 0.95,
                            "faithfulness": 0.9,
                        },
                    },
                },
            },
        ]
    }


def test_compute_mcc_perfect_classifier():
    mcc = compute_mcc(tp=5, fp=0, tn=5, fn=0)
    assert mcc == pytest.approx(1.0)


def test_compute_mcc_all_tn():
    mcc = compute_mcc(tp=0, fp=0, tn=10, fn=0)
    assert mcc is None


def test_quality_nan_for_tn_and_fp():
    records, baselines = prepare_per_query_records(_mock_benchmark_data_two_baselines())
    assert "B1" in baselines and "B2" in baselines

    tn_rows = [r for r in records if r["outcome"] == "TN"]
    fp_rows = [r for r in records if r["outcome"] == "FP"]
    assert len(tn_rows) == 1
    assert len(fp_rows) == 1
    assert math.isnan(tn_rows[0]["semantic_accuracy"])
    assert math.isnan(fp_rows[0]["semantic_accuracy"])


def test_filter_answered_quality_df_excludes_tn_fn():
    records, _ = prepare_per_query_records(_mock_benchmark_data_two_baselines())
    vals = filter_answered_quality_df(records, "semantic_accuracy")
    # TP rows: Q1 B1/B2, Q2 B1/B2, Q4 B2 => 5 values (FN and TN/FP excluded)
    assert len(vals) == 5
    assert np.mean(vals) == pytest.approx(0.79, abs=0.01)


def test_bootstrap_ci_reproducible():
    rng_vals = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    cfg = StatsConfig(n_bootstraps=2000, random_seed=42, alpha=0.05)
    ci1 = bootstrap_ci(rng_vals, cfg)
    ci2 = bootstrap_ci(rng_vals, cfg)
    assert ci1["mean"] == ci2["mean"]
    assert ci1["ci_lower"] == ci2["ci_lower"]
    assert ci1["ci_upper"] == ci2["ci_upper"]
    assert ci1["ci_lower"] < ci1["mean"] < ci1["ci_upper"]


def test_bootstrap_ci_bca_runs():
    vals = np.linspace(0.4, 0.9, 12)
    cfg = StatsConfig(n_bootstraps=500, ci_method="bca", random_seed=42)
    ci = bootstrap_ci(vals, cfg)
    assert ci["ci_lower"] is not None
    assert ci["ci_upper"] is not None


def test_bootstrap_paired_difference_ci():
    a = np.array([0.9, 0.8, 0.85])
    b = np.array([0.7, 0.6, 0.65])
    cfg = StatsConfig(n_bootstraps=1000, random_seed=42)
    diff = bootstrap_paired_difference_ci(a, b, cfg)
    assert diff["delta"] == pytest.approx(0.2, abs=0.01)
    assert diff["ci_lower"] < diff["delta"] < diff["ci_upper"]


def test_wilcoxon_and_rank_biserial():
    a = np.array([0.95, 0.90, 0.92, 0.88, 0.91, 0.89])
    b = np.array([0.55, 0.50, 0.52, 0.48, 0.51, 0.49])
    result = wilcoxon_signed_rank_test(a, b)
    assert result["p_value"] is not None
    assert result["p_value"] < 0.05
    assert result["effect_size"] is not None
    assert result["effect_size"] > 0


def test_wilcoxon_small_n():
    a = np.array([0.9])
    b = np.array([0.7])
    result = wilcoxon_signed_rank_test(a, b)
    assert result["p_value"] is None


def test_mcnemar_answerability():
    records, _ = prepare_per_query_records(_mock_benchmark_data_two_baselines())
    result = mcnemar_answerability_test(records, "B1", "B2")
    assert result["n"] == 4
    assert result["p_value"] is not None
    assert 0.0 <= result["p_value"] <= 1.0


def test_friedman_requires_three_baselines():
    records, _ = prepare_per_query_records(_mock_benchmark_data_two_baselines())
    result = friedman_omnibus_test(records, ["B1", "B2"], "semantic_accuracy")
    assert result["p_value"] is None


def test_friedman_three_baselines():
    data = {
        "results": [
            {
                "id": f"Q{i}",
                "is_answerable": True,
                "baselines": {
                    "B1": {"status": "success", "eval_metrics": {"answerability_outcome": "TP", "semantic_accuracy": 0.5 + i * 0.01}},
                    "B2": {"status": "success", "eval_metrics": {"answerability_outcome": "TP", "semantic_accuracy": 0.6 + i * 0.01}},
                    "B3": {"status": "success", "eval_metrics": {"answerability_outcome": "TP", "semantic_accuracy": 0.7 + i * 0.01}},
                },
            }
            for i in range(5)
        ]
    }
    records, baselines = prepare_per_query_records(data)
    result = friedman_omnibus_test(records, baselines, "semantic_accuracy")
    assert result["p_value"] is not None
    assert result["n"] == 5


def test_friedman_identical_scores_no_warning():
    """Complete within-block ties make scipy's Friedman c=0; should not warn or NaN."""
    data = {
        "results": [
            {
                "id": f"Q{i}",
                "is_answerable": True,
                "baselines": {
                    "B1": {"status": "success", "eval_metrics": {"answerability_outcome": "TP", "semantic_accuracy": 0.8}},
                    "B2": {"status": "success", "eval_metrics": {"answerability_outcome": "TP", "semantic_accuracy": 0.8}},
                    "B3": {"status": "success", "eval_metrics": {"answerability_outcome": "TP", "semantic_accuracy": 0.8}},
                },
            }
            for i in range(5)
        ]
    }
    records, baselines = prepare_per_query_records(data)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = friedman_omnibus_test(records, baselines, "semantic_accuracy")
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not runtime, f"Unexpected RuntimeWarning(s): {[str(w.message) for w in runtime]}"
    assert result["n"] == 5
    assert result["statistic"] == 0.0
    assert result["p_value"] == 1.0


def test_holm_correction():
    flags = holm_correction([0.01, 0.04, 0.03, 0.20], alpha=0.05)
    assert flags[0] is True
    assert sum(flags) >= 1


def test_apply_multiple_comparison_correction_none():
    cfg = StatsConfig(correction_method="none", alpha=0.05)
    flags = apply_multiple_comparison_correction([0.01, 0.10], cfg)
    assert flags == [True, False]


def test_significance_stars():
    assert significance_stars(0.001) == "***"
    assert significance_stars(0.005) == "**"
    assert significance_stars(0.03) == "**"
    assert significance_stars(0.10) == ""


def test_paired_metric_vectors():
    records, _ = prepare_per_query_records(_mock_benchmark_data_two_baselines())
    a, b, ids = paired_metric_vectors(records, "B1", "B2", "semantic_accuracy")
    assert len(ids) == 2  # Q1 and Q2 only (shared TP with non-NaN)
    assert len(a) == len(b) == 2


def test_compute_classification_metrics():
    metrics = compute_classification_metrics(2, 1, 3, 1, total_q=7)
    assert metrics["mcc"] is not None
    assert metrics["hallucination_rate"] == pytest.approx(1 / 4)


def test_integration_compute_statistical_analysis():
    cfg = StatsConfig(n_bootstraps=500, random_seed=42)
    analysis = compute_statistical_analysis(_mock_benchmark_data_two_baselines(), cfg)
    assert analysis["enabled"] is True
    assert "B1" in analysis["baselines"]
    assert analysis["baseline_summary"]["B1"]["semantic_accuracy"]["filter"] == "TP+FP"
    assert len(analysis["pairwise"]) > 0
    assert "filtering_note" in analysis


def test_integration_connector_pipeline():
    cfg = StatsConfig(n_bootstraps=300, random_seed=42)
    analysis = run_statistical_pipeline(_mock_benchmark_data_two_baselines(), config=cfg)
    assert analysis.get("markdown_sections")
    md = build_statistical_markdown(analysis)
    assert "Statistical Analysis" in md
    assert "TP+FP" in md or "TP+FP" in analysis["filtering_note"]


def test_data_prep_agent_from_joined_rows():
    joined = [
        {
            "query_id": "Q1",
            "baseline": "B6",
            "is_answerable": True,
            "answerability_outcome": "TP",
            "semantic_accuracy": 0.88,
            "faithfulness": 0.9,
        }
    ]
    records, baselines = data_prep_agent({}, joined_rows=joined)
    assert len(records) == 1
    assert baselines == ["B6"]


def test_stats_agent_disabled():
    cfg = StatsConfig(enable_stats=False)
    result = stats_agent(_mock_benchmark_data_two_baselines(), cfg)
    assert result["enabled"] is False


def test_all_tn_edge_case():
    data = {
        "results": [
            {
                "id": "Q1",
                "is_answerable": False,
                "baselines": {
                    "B1": {
                        "status": "success",
                        "eval_metrics": {
                            "answerability_outcome": "TN",
                            "predicted_abstained": True,
                        },
                    }
                },
            }
        ]
    }
    records, _ = prepare_per_query_records(data)
    vals = filter_answered_quality_df(records, "semantic_accuracy")
    assert len(vals) == 0
    cfg = StatsConfig(n_bootstraps=100, random_seed=42)
    analysis = compute_statistical_analysis(data, cfg)
    assert analysis["baseline_summary"]["B1"]["semantic_accuracy"]["n"] == 0


def _mock_benchmark_data_with_telemetry() -> dict:
    """Two baselines with shannon_diagnostics / top-level telemetry for stats tests."""
    results = []
    for i in range(6):
        results.append(
            {
                "id": f"Q{i + 1}",
                "is_answerable": True,
                "baselines": {
                    "B1": {
                        "status": "success",
                        "latency_sec": 1.0 + i * 0.1,
                        "eval_metrics": {
                            "answerability_outcome": "TP",
                            "semantic_accuracy": 0.9,
                            "faithfulness": 0.85,
                        },
                        "shannon_diagnostics": {
                            "h_gen": 1.0 + i * 0.05,
                            "h_citation": 0.4,
                            "delta_h_gen": 0.5 + i * 0.02,
                            "avg_msp": 0.70 + i * 0.01,
                            "avg_logit_margin": 1.5 + i * 0.1,
                            "first_token_msp": 0.80,
                            "first_token_margin": 2.0,
                            "ll_rag": -10.0 - i,
                            "ll_base": -12.0 - i,
                            "clr": 2.0 + i * 0.1,
                            "n_citation_tokens": 3,
                        },
                    },
                    "B2": {
                        "status": "success",
                        "latency_sec": 1.5 + i * 0.1,
                        "eval_metrics": {
                            "answerability_outcome": "TP",
                            "semantic_accuracy": 0.7,
                            "faithfulness": 0.7,
                        },
                        # Mix top-level + alias names to exercise extraction paths
                        "avg_msp": 0.55 + i * 0.01,
                        "shannon_diagnostics": {
                            "generation_entropy": 1.5 + i * 0.05,
                            "citation_entropy": 0.6,
                            "entropy_reduction": 0.2 + i * 0.01,
                            "avg_logit_margin": 0.8 + i * 0.05,
                            "first_token_msp": 0.60,
                            "first_token_margin": 1.0,
                            "ll_rag": -14.0 - i,
                            "ll_base": -14.5 - i,
                            "clr": 0.5 + i * 0.05,
                            "citation_token_count": 1,
                        },
                    },
                },
            }
        )
    # One unanswerable TN: telemetry present but excluded from answered filter
    results.append(
        {
            "id": "Q_TN",
            "is_answerable": False,
            "baselines": {
                "B1": {
                    "status": "success",
                    "eval_metrics": {
                        "answerability_outcome": "TN",
                        "predicted_abstained": True,
                    },
                    "shannon_diagnostics": {"avg_msp": 0.99, "h_gen": 0.1},
                },
                "B2": {
                    "status": "success",
                    "eval_metrics": {
                        "answerability_outcome": "TN",
                        "predicted_abstained": True,
                    },
                    "shannon_diagnostics": {"avg_msp": 0.98, "h_gen": 0.2},
                },
            },
        }
    )
    return {"results": results}


def test_telemetry_metrics_in_prepare_summary_pairwise_friedman():
    """Telemetry (MSP, margins, H_gen, CLR, …) appear in records/summary/pairwise/Friedman."""
    data = _mock_benchmark_data_with_telemetry()
    records, baselines = prepare_per_query_records(data)
    assert set(baselines) == {"B1", "B2"}

    tp_b1 = [r for r in records if r["baseline"] == "B1" and r["outcome"] == "TP"]
    assert len(tp_b1) == 6
    assert tp_b1[0]["avg_msp"] == pytest.approx(0.70)
    # Alias path: generation_entropy → h_gen
    tp_b2 = [r for r in records if r["baseline"] == "B2" and r["outcome"] == "TP"]
    assert tp_b2[0]["h_gen"] == pytest.approx(1.5)
    assert tp_b2[0]["n_citation_tokens"] == pytest.approx(1.0)

    # TN rows keep raw telemetry but answered filter excludes them from pairs/means
    tn_rows = [r for r in records if r["outcome"] == "TN"]
    assert len(tn_rows) == 2
    assert tn_rows[0]["avg_msp"] == pytest.approx(0.99) or tn_rows[1]["avg_msp"] == pytest.approx(0.99)

    a, b, ids = paired_metric_vectors(records, "B1", "B2", "avg_msp")
    assert len(ids) == 6  # TP only; TN excluded
    assert float(np.mean(a)) > float(np.mean(b))

    cfg = StatsConfig(n_bootstraps=200, random_seed=42)
    analysis = compute_statistical_analysis(data, cfg)

    sum_b1 = analysis["baseline_summary"]["B1"]
    assert "avg_msp" in sum_b1
    assert sum_b1["avg_msp"]["filter"] == "TP+FP"
    assert sum_b1["avg_msp"]["n"] == 6
    assert sum_b1["avg_msp"]["mean"] == pytest.approx(0.70 + 0.01 * 2.5, abs=1e-6)
    assert "clr" in sum_b1
    assert "h_gen" in sum_b1
    assert sum_b1["clr"]["n"] == 6

    tel_rows = [r for r in analysis["pairwise"] if r.get("family") == "telemetry"]
    qual_rows = [r for r in analysis["pairwise"] if r.get("family") == "quality"]
    safety_rows = [r for r in analysis["pairwise"] if r.get("family") == "safety"]
    assert len(tel_rows) > 0
    assert len(qual_rows) > 0
    assert any(r["metric"] == "answerability_correct" for r in safety_rows)

    msp_pairs = [r for r in tel_rows if r["metric"] == "avg_msp"]
    assert len(msp_pairs) == 1
    assert msp_pairs[0]["baseline_a"] == "B1"
    assert msp_pairs[0]["baseline_b"] == "B2"
    assert msp_pairs[0]["n_pairs"] == 6
    assert msp_pairs[0]["delta"] is not None

    # Three-baseline Friedman includes telemetry metrics
    three = {
        "results": [
            {
                "id": f"Q{i}",
                "is_answerable": True,
                "baselines": {
                    "B1": {
                        "status": "success",
                        "eval_metrics": {"answerability_outcome": "TP", "semantic_accuracy": 0.5},
                        "shannon_diagnostics": {"avg_msp": 0.5 + i * 0.01, "clr": 1.0},
                    },
                    "B2": {
                        "status": "success",
                        "eval_metrics": {"answerability_outcome": "TP", "semantic_accuracy": 0.6},
                        "shannon_diagnostics": {"avg_msp": 0.6 + i * 0.01, "clr": 1.5},
                    },
                    "B3": {
                        "status": "success",
                        "eval_metrics": {"answerability_outcome": "TP", "semantic_accuracy": 0.7},
                        "shannon_diagnostics": {"avg_msp": 0.7 + i * 0.01, "clr": 2.0},
                    },
                },
            }
            for i in range(5)
        ]
    }
    friedman = compute_statistical_analysis(three, StatsConfig(n_bootstraps=50, random_seed=0))["friedman"]
    assert "avg_msp" in friedman
    assert "clr" in friedman
    assert friedman["avg_msp"]["p_value"] is not None
    assert friedman["avg_msp"]["n"] == 5