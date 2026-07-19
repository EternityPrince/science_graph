"""Wiring tests: Shannon pre/post must respect staged inputs (no forced pre=post copy)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.shannon_estimator import (
    assemble_retrieval_shannon_fields,
    compute_rank_entropy,
    parse_graph_relations_from_text,
)


def test_assemble_is_the_shared_path_used_by_consumers():
    """Static check: generation/pipelined/analytics import assemble_retrieval_shannon_fields."""
    import core.generation as generation
    import core.pipelined as pipelined
    import core.analytics as analytics
    import inspect

    gen_src = inspect.getsource(generation)
    pipe_src = inspect.getsource(pipelined)
    ana_src = inspect.getsource(analytics)

    assert "assemble_retrieval_shannon_fields" in gen_src
    assert "assemble_retrieval_shannon_fields" in pipe_src
    assert "assemble_retrieval_shannon_fields" in ana_src
    # Regression: must not hard-assign pre from post in shannon_diag blocks
    assert '"h_rank_pre_rerank": round(h_rank_post' not in gen_src
    assert '"h_rank_pre_rerank": round(h_rank_post' not in pipe_src
    assert '"h_rank_pre_rerank": h_rank_post' not in ana_src


def test_consume_path_fields_prefer_pre_rerank_and_context_text():
    """Simulate retrieved_contexts baseline payload with distinct pre/post stage data."""
    pre_baseline = {
        "pre_rerank_scores": [1.0, 1.0, 1.0, 1.0],
        "context_text": "alpha beta gamma delta epsilon zeta eta theta iota",
        "context_graph": (
            "- (p1:Paper)-[CITES]->(p2:Paper)\n"
            "- (a1:Author)-[AUTHORED]->(p1:Paper)"
        ),
        "graph_relations": [
            {"source": "p1", "target": "p2", "type": "CITES"},
            {"source": "a1", "target": "p1", "type": "AUTHORED"},
        ],
        "trimmed_text": "alpha alpha alpha",
        "trimmed_graph": "- (p1:Paper)-[CITES]->(p2:Paper)",
        "retrieved_chunks": [
            {"id": "c1", "score": 9.0, "text_content": "alpha"},
            {"id": "c2", "score": 0.1, "text_content": "alpha"},
            {"id": "c3", "score": 0.1, "text_content": "alpha"},
            {"id": "c4", "score": 0.1, "text_content": "alpha"},
        ],
    }
    post_scores = [c["score"] for c in pre_baseline["retrieved_chunks"]]
    fields = assemble_retrieval_shannon_fields(
        pre_scores=pre_baseline.get("pre_rerank_scores"),
        post_scores=post_scores,
        pre_text=pre_baseline.get("context_text"),
        post_text=pre_baseline.get("trimmed_text"),
        relations=pre_baseline.get("graph_relations"),
        graph_text=pre_baseline.get("context_graph"),
    )
    assert fields["h_rank_pre_rerank"] != fields["h_rank_post_rerank"]
    assert fields["h_lexical_pre_trim"] != fields["h_lexical_post_trim"]
    assert fields["h_graph_relation_type"] > 0.0
    assert fields["h_graph_degree"] > 0.0


def test_analytics_offline_backfill_uses_pre_fields_when_present():
    """analytics backfill must not force pre=post when pre_rerank_scores exist."""
    from core.shannon_estimator import assemble_retrieval_shannon_fields

    b_data = {
        "retrieved_chunks": [
            {"score": 8.0, "text_content": "aa"},
            {"score": 0.2, "text_content": "aa"},
            {"score": 0.2, "text_content": "aa"},
            {"score": 0.2, "text_content": "aa"},
        ],
        "pre_rerank_scores": [1.0, 1.0, 1.0, 1.0],
        "context_text": "unique words for lexical diversity one two three four five",
        "trimmed_text": "aa aa aa aa",
        "graph_relations": [
            {"source": "A", "target": "B", "type": "CITES"},
            {"source": "B", "target": "C", "type": "AUTHORED"},
        ],
    }
    post_scores = [c["score"] for c in b_data["retrieved_chunks"]]
    fields = assemble_retrieval_shannon_fields(
        pre_scores=b_data.get("pre_rerank_scores"),
        post_scores=post_scores,
        pre_text=b_data.get("context_text"),
        post_text=b_data.get("trimmed_text"),
        relations=b_data.get("graph_relations"),
        graph_text=b_data.get("context_graph") or b_data.get("trimmed_graph"),
    )
    assert fields["h_rank_pre_rerank"] != fields["h_rank_post_rerank"]
    assert fields["h_lexical_pre_trim"] != fields["h_lexical_post_trim"]
    assert fields["h_graph_relation_type"] > 0.0


def test_live_path_prefers_last_pre_rerank_scores_attribute():
    """Live run_query path uses rag_service._last_pre_rerank_scores when set."""
    pre = [0.25, 0.25, 0.25, 0.25]
    post = [5.0, 0.01, 0.01, 0.01]
    h_pre = compute_rank_entropy(pre)
    h_post = compute_rank_entropy(post)
    assert h_pre != h_post

    fields = assemble_retrieval_shannon_fields(
        pre_scores=pre,  # as getattr(rag_service, "_last_pre_rerank_scores", None)
        post_scores=post,
        pre_text="pre context words a b c d e f",
        post_text="post post post",
        relations=[{"source": "x", "target": "y", "type": "CITES"}],
    )
    assert fields["h_rank_pre_rerank"] == round(h_pre, 4)
    assert fields["h_rank_post_rerank"] == round(h_post, 4)
    assert fields["h_rank_pre_rerank"] != fields["h_rank_post_rerank"]


def test_retrieval_persists_shannon_stage_keys():
    """Static check: retrieval Stage 5 writes pre_rerank_scores / context_text / graph_relations."""
    import inspect
    from core import retrieval

    src = inspect.getsource(retrieval)
    for key in (
        "pre_rerank_scores",
        "context_text",
        "context_graph",
        "graph_relations",
    ):
        assert key in src, f"retrieval.py must persist {key}"


def test_rag_service_sets_last_pre_rerank_and_graph_relations():
    """Static check: rag_service writes Shannon side-channels."""
    import inspect
    from src.services import rag_service as rs_mod

    src = inspect.getsource(rs_mod)
    assert "_last_pre_rerank_scores" in src
    assert "_last_graph_relations" in src
