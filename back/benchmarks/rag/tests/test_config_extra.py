import pytest
import yaml
from pathlib import Path
from core.config import get_baseline_config, get_safe_model_name, load_benchmark_dataset

def test_get_baseline_config():
    mock_rag_components = {
        "intent_classifier": True,
        "graph_ontology_lookup": True,
        "llm_query_expansion": True,
        "hyde": True,
        "lexical_search": True,
        "dense_search": True,
        "dynamic_alpha_blending": True,
        "rrf": True,
        "graph_expansion": True,
        "reranker": True,
        "score_blending": True,
        "context_trimming": True,
        "citation_repair": True,
    }
    
    # Test B0
    b0_conf = get_baseline_config("B0", mock_rag_components)
    assert not any(b0_conf.values())
    
    # Test B1
    b1_conf = get_baseline_config("B1", mock_rag_components)
    assert b1_conf["lexical_search"]
    assert not b1_conf["dense_search"]
    
    # Test B2
    b2_conf = get_baseline_config("B2", mock_rag_components)
    assert b2_conf["dense_search"]
    
    # Test B3
    b3_conf = get_baseline_config("B3", mock_rag_components)
    assert b3_conf["dense_search"]
    assert b3_conf["hyde"]
    
    # Test B4
    b4_conf = get_baseline_config("B4", mock_rag_components)
    assert b4_conf["dense_search"]
    assert b4_conf["lexical_search"]
    assert b4_conf["rrf"]
    
    # Test B5
    b5_conf = get_baseline_config("B5", mock_rag_components)
    assert b5_conf["graph_expansion"]
    
    # Test B6
    b6_conf = get_baseline_config("B6", mock_rag_components)
    assert not b6_conf["hyde"]
    assert b6_conf["dense_search"]
    
    # Test CUSTOM
    custom_conf = get_baseline_config("CUSTOM", mock_rag_components)
    assert not custom_conf["hyde"]
    assert custom_conf["dense_search"]

def test_get_safe_model_name():
    assert get_safe_model_name("provider/model-name:v1") == "model-name_v1"
    assert get_safe_model_name("some path/to/model") == "model"

def test_load_benchmark_dataset(tmp_path):
    # Test file missing
    with pytest.raises(FileNotFoundError):
        load_benchmark_dataset(tmp_path / "nonexistent.yaml")
        
    # Test empty dataset
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("", encoding="utf-8")
    assert load_benchmark_dataset(empty_file) == []
    
    # Test standard dataset loading and limiting
    standard_data = [
        {"id": f"Q{i}", "query": f"Query {i}", "golden_answer": f"Ans {i}"}
        for i in range(10)
    ]
    standard_file = tmp_path / "dataset.yaml"
    with open(standard_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(standard_data, f)
        
    loaded = load_benchmark_dataset(standard_file)
    assert len(loaded) == 10
    
    limited = load_benchmark_dataset(standard_file, limit=3)
    assert len(limited) == 3
    
    # Test SciQ format conversion
    sciq_data = [
        {
            "question": {
                "id": i,
                "q": f"SciQ Question {i}",
                "a": f"SciQ Answer {i}",
                "c": f"SciQ Context {i // 2}" # Shared contexts
            }
        }
        for i in range(6)
    ]
    sciq_file = tmp_path / "sciq.yaml"
    with open(sciq_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(sciq_data, f)
        
    sciq_loaded = load_benchmark_dataset(sciq_file)
    # Wait, SciQ conversion will limit to DEFAULT_LIMIT (50). Since len is 6, it stays 6! Wait, rng.sample(data, limit) will only run if len(data) > limit. 6 is not > 50, so it returns all 6!
    assert len(sciq_loaded) == 6
    assert sciq_loaded[0]["id"] == "sciq_0"
    assert sciq_loaded[0]["expected_papers"] == ["sciq_paper_1"]
    
    sciq_limited = load_benchmark_dataset(sciq_file, limit=2)
    assert len(sciq_limited) == 2

    # Test SciQ limit = -1 (line 128)
    sciq_no_limit = load_benchmark_dataset(sciq_file, limit=-1)
    assert len(sciq_no_limit) == 6

    # Test sorting exception (lines 136-137)
    sorting_fail_data = [
        {"id": 1, "query": "Q1"},
        {"id": "two", "query": "Q2"},
        {"id": 3.0, "query": "Q3"}
    ]
    sorting_fail_file = tmp_path / "sorting_fail.yaml"
    with open(sorting_fail_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(sorting_fail_data, f)
    res = load_benchmark_dataset(sorting_fail_file, limit=2)
    assert len(res) == 2
