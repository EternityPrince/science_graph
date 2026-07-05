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

def test_run_staged_retrieval_missing_dataset():
    args = MagicMock()
    args.dataset = "nonexistent_dataset.yaml"
    con = MagicMock()
    with pytest.raises(SystemExit):
        run_staged_retrieval(args, MagicMock(), MagicMock(), MagicMock(), con)
    con.error.assert_called_with("Dataset file not found: nonexistent_dataset.yaml")

@patch("core.config.load_benchmark_dataset", side_effect=Exception("parse error"))
def test_run_staged_retrieval_dataset_parse_fail(mock_load):
    args = MagicMock()
    args.dataset = "dataset.yaml"
    with patch("core.retrieval.Path.exists", return_value=True):
        with pytest.raises(SystemExit):
            run_staged_retrieval(args, MagicMock(), MagicMock(), MagicMock(), MagicMock())

@patch("core.config.load_benchmark_dataset", return_value=[])
def test_run_staged_retrieval_dataset_empty(mock_load):
    args = MagicMock()
    args.dataset = "dataset.yaml"
    with patch("core.retrieval.Path.exists", return_value=True):
        with pytest.raises(SystemExit):
            run_staged_retrieval(args, MagicMock(), MagicMock(), MagicMock(), MagicMock())

def test_run_staged_retrieval_rag_service_fail(tmp_path):
    args = MagicMock()
    args.dataset = str(tmp_path / "dataset.yaml")
    (tmp_path / "dataset.yaml").write_text("dataset")
    container = MagicMock()
    container.get_rag_service.side_effect = Exception("init error")
    with patch("core.config.load_benchmark_dataset", return_value=[{"id": "q1"}]):
        with pytest.raises(SystemExit):
            run_staged_retrieval(args, MagicMock(), MagicMock(), container, MagicMock())

def test_run_staged_retrieval_all_features(tmp_path):
    # Runs staged retrieval covering B0, B1, B6 (expander), HyDE, and reranking
    dataset_path = tmp_path / "dataset.yaml"
    dataset_data = [
        {"id": "Q1", "query": "What is staged retrieval?", "expected_papers": ["paper_1"]}
    ]
    with open(dataset_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_data, f)
        
    # Save to "other_name" to trigger reports directory fallback resolver (line 575)
    output_path = tmp_path / "other_name" / "retrieved_contexts.yaml"
    
    args = MagicMock()
    args.dataset = str(dataset_path)
    args.limit = 1
    args.cloud = False
    args.baselines = "B0,B1,B6"
    args.output = str(output_path)
    args.no_unique_dir = False # trigger copy save in reports/run_retrive_*
    
    config = MagicMock()
    config.hyde_enabled = True
    config.hyde_count = 2
    config.hyde_max_tokens = 50
    config.llm_model_max_context = 2048
    config.reranker_model_name = "test_reranker"
    config.rag_components = {
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
        "citation_repair": False,
    }
    config.data = {
        "llm": {
            "hyde_enabled": True,
            "local": {"model_path": "local_model"}
        },
        "embedding": {"model_name": "emb"},
        "rag_components": config.rag_components.copy()
    }
    
    prompts = MagicMock()
    prompts.get_prompt.return_value = "dummy prompt"
    
    rag_service = MagicMock()
    rag_service._expand_query.return_value = ["expanded query 1", "expanded query 2"]
    rag_service.llm_engine.generate_response.return_value = "hypothetical answer"
    rag_service.llm_engine.count_tokens.return_value = 10
    
    class MockChunk:
        def __init__(self, cid, pid, text):
            self.id = cid
            self.paper_id = pid
            self.page_number = 1
            self.text_content = text
            
    mock_chunks = [
        (MockChunk("c1", "paper_1", "text content 1"), 0.9),
        (MockChunk("c2", "paper_2", "text content 2"), 0.8)
    ]
    # Mock retrieve to call inner functions to cover lines 198, 201-204, 207
    def mock_retrieve(q, limit):
        rag_service._expand_query(q)
        # Call generate_response 3 times to exhaust iterator (triggers StopIteration line 203-204)
        rag_service.llm_engine.generate_response("prompt")
        rag_service.llm_engine.generate_response("prompt")
        rag_service.llm_engine.generate_response("prompt")
        rag_service._classify_intent_and_extract_filters(q)
        return mock_chunks
    rag_service.retrieve_relevant_chunks.side_effect = mock_retrieve
    rag_service.build_context.return_value = ("context text", "context graph")
    rag_service.trim_context.return_value = ("trimmed text", "trimmed graph", [mock_chunks[0]])
    
    # Mock reranker predict
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = [0.95, 0.85]
    rag_service._reranker = mock_reranker
    
    container = MagicMock()
    container.get_rag_service.return_value = rag_service
    
    con = MagicMock()
    
    # Run with torch mps disabled and cuda enabled to cover cuda cache clear lines (line 541-542)
    with patch("torch.backends.mps.is_available", return_value=False), \
         patch("torch.cuda.is_available", return_value=True):
        run_staged_retrieval(args, config, prompts, container, con)
        
    assert output_path.exists()
    # Confirm copy run_retrive directory exists in the output path's parent directory
    reports_dir = output_path.parent
    subdirs = [d for d in reports_dir.iterdir() if d.is_dir() and "run_retrive_" in d.name]
    assert len(subdirs) >= 1
    # Clean up the created test reports directories
    import shutil
    for s in subdirs:
        try:
            shutil.rmtree(s)
        except Exception:
            pass

def test_run_staged_retrieval_dataset_fallback_path():
    # Tests dataset default resolution fallback (lines 20-24)
    args = MagicMock()
    args.dataset = None
    args.baselines = "B1"
    
    con = MagicMock()
    
    from pathlib import Path
    orig_exists = Path.exists
    def mock_exists(self):
        if "golden_dataset" in self.name:
            return False
        return orig_exists(self)
        
    with patch("pathlib.Path.exists", side_effect=mock_exists, autospec=True), \
         patch("core.config.load_benchmark_dataset", return_value=[{"id": "Q1", "query": "Query"}]), \
         patch("src.services.container.container.get_rag_service"):
         
        with pytest.raises(SystemExit):
            run_staged_retrieval(args, MagicMock(), MagicMock(), MagicMock(), con)

def test_run_staged_retrieval_all_baselines_option(tmp_path):
    # Tests baselines.lower() == "all" (line 54) and score_blending disabled (line 347)
    dataset_path = tmp_path / "dataset.yaml"
    yaml.dump([{"id": "Q1", "query": "Q"}], open(dataset_path, "w"))
    output_path = tmp_path / "output.yaml"
    
    args = MagicMock()
    args.dataset = str(dataset_path)
    args.limit = 1
    args.cloud = True
    args.baselines = "all"
    args.output = str(output_path)
    args.no_unique_dir = False
    
    config = MagicMock()
    config.hyde_enabled = False
    config.rag_components = {"reranker": False, "score_blending": False}
    config.data = {
        "llm": {"cloud": {"model_name": "model"}},
        "embedding": {"model_name": "emb"},
        "rag_components": config.rag_components.copy()
    }
    
    rag_service = MagicMock()
    rag_service.retrieve_relevant_chunks.return_value = []
    
    container = MagicMock()
    container.get_rag_service.return_value = rag_service
    
    run_staged_retrieval(args, config, MagicMock(), container, MagicMock())
    assert output_path.exists()
    
    # Clean up generated test report directories
    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    if reports_dir.exists():
        import shutil
        for s in reports_dir.iterdir():
            if s.is_dir() and "run_retrive_" in s.name:
                try:
                    shutil.rmtree(s)
                except Exception:
                    pass


def test_run_staged_retrieval_stage_exceptions(tmp_path):
    # Runs staged retrieval and triggers exceptions in different stages to check fallbacks
    dataset_path = tmp_path / "dataset.yaml"
    dataset_data = [{"id": "Q1", "query": "Query", "expected_papers": []}]
    with open(dataset_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_data, f)
    output_path = tmp_path / "retrieved_contexts.yaml"
    
    args = MagicMock()
    args.dataset = str(dataset_path)
    args.limit = 1
    args.cloud = False
    args.baselines = "B1"
    args.output = str(output_path)
    args.no_unique_dir = True
    
    config = MagicMock()
    config.hyde_enabled = True
    config.rag_components = {"llm_query_expansion": True, "hyde": True, "dense_search": True, "reranker": True}
    config.data = {"llm": {"hyde_enabled": True}, "rag_components": config.rag_components.copy()}
    
    prompts = MagicMock()
    rag_service = MagicMock()
    
    # Stage 1 exception
    rag_service._expand_query.side_effect = Exception("expansion failed")
    
    # Stage 3 succeed but empty
    rag_service.retrieve_relevant_chunks.return_value = []
    
    # Stage 5 exception
    rag_service.build_context.side_effect = Exception("build context failed")
    
    container = MagicMock()
    container.get_rag_service.return_value = rag_service
    con = MagicMock()
    
    run_staged_retrieval(args, config, prompts, container, con)
    assert output_path.exists()
    
    with open(output_path, "r", encoding="utf-8") as f:
        res_data = yaml.safe_load(f)
    assert res_data[0]["baselines"]["B1"]["status"] == "error"

def test_run_staged_retrieval_table_printing_error(tmp_path):
    dataset_path = tmp_path / "dataset.yaml"
    dataset_data = [{"id": "Q1", "query": "Query", "expected_papers": []}]
    with open(dataset_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_data, f)
    output_path = tmp_path / "retrieved_contexts.yaml"
    
    args = MagicMock()
    args.dataset = str(dataset_path)
    args.limit = 1
    args.cloud = False
    args.baselines = "B1"
    args.output = str(output_path)
    args.no_unique_dir = True
    
    config = MagicMock()
    config.rag_components = {"reranker": False}
    config.data = {"llm": {"hyde_enabled": False}, "rag_components": config.rag_components.copy()}
    
    rag_service = MagicMock()
    container = MagicMock()
    container.get_rag_service.return_value = rag_service
    con = MagicMock()
    
    # Mock Table to raise exception to cover line 536-543
    with patch("rich.table.Table", side_effect=ValueError("mock table error")):
        run_staged_retrieval(args, config, MagicMock(), container, con)
        
    assert output_path.exists()


def test_run_staged_retrieval_remaining_exceptions(tmp_path):
    dataset_path = tmp_path / "dataset.yaml"
    dataset_data = [{"id": "Q1", "query": "Query", "expected_papers": []}]
    yaml.dump(dataset_data, open(dataset_path, "w"))
    output_path = tmp_path / "retrieved_contexts.yaml"
    
    args = MagicMock()
    args.dataset = str(dataset_path)
    args.limit = 1
    args.cloud = False
    args.baselines = "B6"
    args.output = str(output_path)
    args.no_unique_dir = False
    
    config = MagicMock()
    config.hyde_enabled = True
    config.hyde_count = 1
    config.hyde_max_tokens = 5
    config.rag_components = {"llm_query_expansion": True, "hyde": True, "graph_expansion": True}
    config.data = {
        "llm": {"hyde_enabled": True, "local": {"model_path": "local"}},
        "embedding": {"model_name": "emb"},
        "rag_components": config.rag_components.copy()
    }
    
    rag_service = MagicMock()
    # Trigger Stage 1 exception in generate_response (line 117-120)
    rag_service.llm_engine.generate_response.side_effect = ValueError("Stage 1 LLM failed")
    
    container = MagicMock()
    container.get_rag_service.return_value = rag_service
    con = MagicMock()
    
    from pathlib import Path
    orig_mkdir = Path.mkdir
    def mock_mkdir(self, *args, **kwargs):
        if self == tmp_path:
            raise OSError("mock directory write fail")
        return orig_mkdir(self, *args, **kwargs)
        
    # Mock ExperimentalGraphExpander import/init error (line 422-424) and copy dir error (line 584-585)
    with patch.dict("sys.modules", {"src.services.graph_expander": None}), \
         patch("pathlib.Path.mkdir", side_effect=mock_mkdir, autospec=True):
         
        run_staged_retrieval(args, config, MagicMock(), container, con)
        
    assert Path(args.output).exists()


