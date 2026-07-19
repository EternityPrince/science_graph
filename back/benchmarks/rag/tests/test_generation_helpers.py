"""Tests for extracted generation helpers (module-level shipped functions)."""

from unittest.mock import MagicMock, patch

from core.generation import (
    _build_shannon_diag_for_b0,
    _build_shannon_diag_for_rag,
    merge_evaluation_data,
)
from core.shannon_estimator import empty_retrieval_shannon_fields


def test_build_shannon_diag_for_b0_empty_retrieval_and_rounding():
    diag = _build_shannon_diag_for_b0(1.5, 0.5, 2)
    empty = empty_retrieval_shannon_fields()
    for key, val in empty.items():
        assert diag[key] == val
    assert diag["h_gen"] == 1.5
    assert diag["h_citation"] == 0.5
    assert diag["n_citation_tokens"] == 2
    assert diag["delta_h_gen"] == 0.0

    # rounding to 4 decimals
    diag_r = _build_shannon_diag_for_b0(1.23456, 0.98765, 0)
    assert diag_r["h_gen"] == 1.2346
    assert diag_r["h_citation"] == 0.9877
    assert diag_r["n_citation_tokens"] == 0
    assert diag_r["delta_h_gen"] == 0.0


def test_build_shannon_diag_for_b0_includes_all_expected_keys():
    diag = _build_shannon_diag_for_b0(0.0, 0.0, 0)
    expected = set(empty_retrieval_shannon_fields()) | {
        "h_gen",
        "h_citation",
        "n_citation_tokens",
        "delta_h_gen",
    }
    assert set(diag.keys()) == expected


@patch("core.generation._ensure_b0_entropy", return_value=3.5)
def test_build_shannon_diag_for_rag_delta_and_retrieval_fields(mock_b0):
    rag_service = MagicMock()
    config = MagicMock()
    # Peaked post vs flat pre → rank pre entropy higher than post
    pre_scores = [1.0, 1.0, 1.0, 1.0]
    post_scores = [10.0, 0.1, 0.1, 0.1]
    diag = _build_shannon_diag_for_rag(
        rag_service=rag_service,
        query="what is entropy?",
        config=config,
        h_gen=1.2,
        h_cit=0.4,
        n_cit=3,
        pre_scores=pre_scores,
        post_scores=post_scores,
        context_text="alpha beta gamma delta",
        trimmed_text="alpha alpha",
        last_relations=[
            {"source": "A", "target": "B", "type": "CITES"},
            {"source": "B", "target": "C", "type": "AUTHORED"},
        ],
        context_graph="",
        trimmed_graph="",
    )
    mock_b0.assert_called_once_with(rag_service, "what is entropy?", config)
    assert diag["h_gen"] == 1.2
    assert diag["h_citation"] == 0.4
    assert diag["n_citation_tokens"] == 3
    # delta = 3.5 - 1.2 = 2.3
    assert diag["delta_h_gen"] == 2.3
    assert diag["h_rank_pre_rerank"] > diag["h_rank_post_rerank"]
    assert diag["h_graph_relation_type"] > 0.0
    assert "h_lexical_pre_trim" in diag
    assert "h_lexical_post_trim" in diag


@patch("core.generation._ensure_b0_entropy", return_value=2.0)
def test_build_shannon_diag_for_rag_graph_text_fallback(mock_b0):
    rag_service = MagicMock()
    config = MagicMock()
    graph_text = (
        "- (p1:Paper)-[CITES]->(p2:Paper)\n"
        "- (p1:Paper)-[AUTHORED]->(a1:Author)"
    )
    diag = _build_shannon_diag_for_rag(
        rag_service=rag_service,
        query="q",
        config=config,
        h_gen=2.0,
        h_cit=0.0,
        n_cit=0,
        pre_scores=None,
        post_scores=[1.0, 2.0],
        context_text="ctx",
        trimmed_text="ctx",
        last_relations=None,
        context_graph=graph_text,
        trimmed_graph="",
    )
    assert diag["delta_h_gen"] == 0.0  # 2.0 - 2.0
    assert diag["h_graph_relation_type"] > 0.0
    assert diag["h_graph_degree"] > 0.0


def test_merge_evaluation_data_two_baselines_preserve_both():
    existing = {
        "metadata": {
            "baselines_evaluated": ["B0"],
            "model": "m1",
            "run_id": "old",
        },
        "results": [
            {
                "id": "Q1",
                "query": "q1",
                "baselines": {"B0": {"generated_answer": "a0", "score": 1}},
            },
            {
                "id": "Q_only_old",
                "baselines": {"B0": {"generated_answer": "only0"}},
            },
        ],
    }
    new_data = {
        "metadata": {
            "baselines_evaluated": ["B2"],
            "run_id": "new",
        },
        "results": [
            {
                "id": "Q1",
                "baselines": {"B2": {"generated_answer": "a2", "score": 2}},
            },
            {
                "id": "Q_only_new",
                "baselines": {"B2": {"generated_answer": "only2"}},
            },
        ],
    }
    merged = merge_evaluation_data(existing, new_data)
    assert sorted(merged["metadata"]["baselines_evaluated"]) == ["B0", "B2"]
    # new metadata wins on overlapping keys
    assert merged["metadata"]["run_id"] == "new"
    assert merged["metadata"]["model"] == "m1"

    by_id = {r["id"]: r for r in merged["results"]}
    assert set(by_id) == {"Q1", "Q_only_old", "Q_only_new"}
    assert by_id["Q1"]["baselines"]["B0"]["generated_answer"] == "a0"
    assert by_id["Q1"]["baselines"]["B2"]["generated_answer"] == "a2"
    # query field from existing preserved when new item lacks it via merge of dicts —
    # new_item wins on top-level keys, but baselines are deep-merged
    assert by_id["Q_only_old"]["baselines"]["B0"]["generated_answer"] == "only0"
    assert by_id["Q_only_new"]["baselines"]["B2"]["generated_answer"] == "only2"


def test_merge_evaluation_data_overwrites_same_baseline_answer():
    existing = {
        "metadata": {"baselines_evaluated": ["B1"]},
        "results": [
            {"id": "Q1", "baselines": {"B1": {"generated_answer": "old"}}},
        ],
    }
    new_data = {
        "metadata": {"baselines_evaluated": ["B1"]},
        "results": [
            {"id": "Q1", "baselines": {"B1": {"generated_answer": "new"}}},
        ],
    }
    merged = merge_evaluation_data(existing, new_data)
    assert merged["metadata"]["baselines_evaluated"] == ["B1"]
    assert merged["results"][0]["baselines"]["B1"]["generated_answer"] == "new"


def test_merge_evaluation_data_empty_existing_returns_new():
    new_data = {"metadata": {"baselines_evaluated": ["B0"]}, "results": []}
    assert merge_evaluation_data({}, new_data) is new_data
    assert merge_evaluation_data(None, new_data) is new_data
    assert merge_evaluation_data("bad", new_data) is new_data
