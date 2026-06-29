import pytest
from unittest.mock import MagicMock, patch
from benchmarks.rag.run_benchmarks import get_baseline_config
from src.config import config

def test_benchmark_config_baseline_defaults():
    """Verify that new graph retrieval flags default to False for baseline configs (B0-B5)."""
    # Simulate default config values
    config_dict = {
        "intent_classifier": False,
        "hyde": True,
        "lexical_search": True,
        "dense_search": True,
        "dynamic_alpha_blending": True,
        "rrf": True,
        "graph_expansion": True,
        "reranker": True,
        "graph_concept_retrieval": False,
        "graph_bridge_retrieval": False,
        "graph_selected_sources_card": False,
        "graph_retrieval_trace": False
    }
    
    # 1. B0: Zero-shot (all False)
    b0 = get_baseline_config("B0", config_dict)
    assert not any(b0.values())
    
    # 2. B4: Standard Hybrid
    b4 = get_baseline_config("B4", config_dict)
    assert b4["dense_search"] is True
    assert b4["lexical_search"] is True
    assert b4["graph_concept_retrieval"] is False
    assert b4["graph_bridge_retrieval"] is False
    assert b4["graph_selected_sources_card"] is False
    
    # 3. B5: Hybrid + Graph
    b5 = get_baseline_config("B5", config_dict)
    assert b5["graph_expansion"] is True
    assert b5["graph_concept_retrieval"] is False
    assert b5["graph_bridge_retrieval"] is False
    assert b5["graph_selected_sources_card"] is False

def test_benchmark_config_b6_and_custom_respects_overrides():
    """Verify that B6 and CUSTOM respect user overrides for the new graph retrieval flags."""
    # User overrides some components to True
    config_dict = {
        "hyde": True,
        "graph_concept_retrieval": True,
        "graph_bridge_retrieval": True,
        "graph_selected_sources_card": True,
        "graph_retrieval_trace": True
    }
    
    # B6
    b6 = get_baseline_config("B6", config_dict)
    assert b6["hyde"] is False # explicitly forced to False in B6
    assert b6["graph_concept_retrieval"] is True
    assert b6["graph_bridge_retrieval"] is True
    assert b6["graph_selected_sources_card"] is True
    
    # CUSTOM
    custom = get_baseline_config("CUSTOM", config_dict)
    assert custom["graph_concept_retrieval"] is True
    assert custom["graph_bridge_retrieval"] is True
    assert custom["graph_selected_sources_card"] is True
