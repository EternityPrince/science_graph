import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import yaml
from core.retrieval import run_staged_retrieval

def test_run_staged_retrieval(tmp_path):
    # Create a small dataset
    dataset_data = [
        {
            "id": "Q1",
            "query": "Test query?",
            "golden_answer": "Test answer",
            "expected_papers": ["paper_1"]
        }
    ]
    dataset_path = tmp_path / "dataset.yaml"
    with open(dataset_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_data, f)
        
    output_path = tmp_path / "retrieved_contexts.yaml"
    
    # Mock args
    args = MagicMock()
    args.dataset = str(dataset_path)
    args.limit = 1
    args.cloud = False
    args.baselines = "B1"
    args.output = str(output_path)
    
    # Mock config
    config = MagicMock()
    config.rag_components = {
        "intent_classifier": False,
        "graph_ontology_lookup": False,
        "llm_query_expansion": False,
        "hyde": False,
        "lexical_search": True,
        "dense_search": False,
        "dynamic_alpha_blending": False,
        "rrf": False,
        "graph_expansion": False,
        "reranker": False,
        "score_blending": False,
        "context_trimming": False,
        "citation_repair": False,
    }
    config.hyde_enabled = False
    config.data = {
        "llm": {"hyde_enabled": False},
        "rag_components": config.rag_components.copy()
    }
    
    # Mock prompts
    prompts = MagicMock()
    prompts.get_prompt.return_value = "dummy system prompt"
    
    # Mock RAG service
    rag_service = MagicMock()
    
    # Mock chunk
    class MockChunk:
        def __init__(self):
            self.id = "chunk_1"
            self.paper_id = "paper_1"
            self.page_number = 1
            self.text_content = "some text content"
            
    rag_service.retrieve_relevant_chunks.return_value = [
        (MockChunk(), 0.95)
    ]
    rag_service.build_context.return_value = ("some context text", "some context graph")
    rag_service.trim_context.return_value = ("trimmed text", "trimmed graph", [(MockChunk(), 0.95)])
    
    # Mock container
    container = MagicMock()
    container.get_rag_service.return_value = rag_service
    
    # Mock console
    con = MagicMock()
    
    # Run
    run_staged_retrieval(args, config, prompts, container, con)
    
    # Verify
    assert output_path.exists()
    with open(output_path, "r", encoding="utf-8") as f:
        output_data = yaml.safe_load(f)
    assert len(output_data) == 1
    assert output_data[0]["id"] == "Q1"
    assert "B1" in output_data[0]["baselines"]
    assert output_data[0]["baselines"]["B1"]["retrieved_papers"] == ["paper_1"]
