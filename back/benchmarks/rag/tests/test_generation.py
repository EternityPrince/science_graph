import pytest
import yaml
from unittest.mock import MagicMock, patch
from core.generation import run_query_on_baseline, merge_evaluation_data, run_benchmarking

class MockChunk:
    def __init__(self, cid, pid, text, page):
        self.id = cid
        self.paper_id = pid
        self.text_content = text
        self.page_number = page

def test_run_query_on_baseline_b0():
    rag_service = MagicMock()
    rag_service.llm_engine.generate_response.return_value = "B0 response"
    rag_service.llm_engine.count_tokens.return_value = 10
    
    config = MagicMock()
    config.rag_components = {"hyde": False, "reranker": False}
    config.is_component_enabled.return_value = False
    config.data = {"llm": {"hyde_enabled": False}, "rag_components": {}}
    
    ans, papers, metrics, chunks = run_query_on_baseline(
        rag_service, "question", "B0", False, config
    )
    assert ans == "B0 response"
    assert papers == []
    assert chunks == []
    assert metrics["components"]["llm_generation"]["calls"] == 1

def test_run_query_on_baseline_b1():
    rag_service = MagicMock()
    chunk = MockChunk("c1", "p1", "some text", 1)
    rag_service.retrieve_relevant_chunks.return_value = [(chunk, 0.9)]
    rag_service.ask.return_value = "ask response"
    rag_service.last_raw_response = "raw response"
    
    config = MagicMock()
    config.rag_components = {"hyde": False, "reranker": False}
    config.is_component_enabled.return_value = False
    config.data = {"llm": {"hyde_enabled": False}, "rag_components": {}}
    
    ans, papers, metrics, chunks = run_query_on_baseline(
        rag_service, "question", "B1", False, config
    )
    assert ans == "raw response"
    assert papers == ["p1"]
    assert len(chunks) == 1
    assert chunks[0]["id"] == "c1"

def test_run_query_on_baseline_b6_expander_fail():
    rag_service = MagicMock()
    rag_service.retrieve_relevant_chunks.return_value = []
    rag_service._get_reranker.side_effect = Exception("No reranker")
    
    config = MagicMock()
    config.rag_components = {"hyde": False, "reranker": True}
    config.is_component_enabled.return_value = False
    config.data = {"llm": {"hyde_enabled": False}, "rag_components": {}}
    
    ans, papers, metrics, chunks = run_query_on_baseline(
        rag_service, "question", "B6", False, config
    )
    assert ans == "Information missing in database."
    assert papers == []
    assert chunks == []

def test_merge_evaluation_data():
    existing = {
        "metadata": {
            "baselines_evaluated": ["B0"]
        },
        "results": [
            {
                "id": "Q1",
                "baselines": {
                    "B0": {"generated_answer": "ans0"}
                }
            }
        ]
    }
    
    new_data = {
        "metadata": {
            "baselines_evaluated": ["B1"]
        },
        "results": [
            {
                "id": "Q1",
                "baselines": {
                    "B1": {"generated_answer": "ans1"}
                }
            },
            {
                "id": "Q2",
                "baselines": {
                    "B1": {"generated_answer": "ans2"}
                }
            }
        ]
    }
    
    merged = merge_evaluation_data(existing, new_data)
    assert merged["metadata"]["baselines_evaluated"] == ["B0", "B1"]
    results = {r["id"]: r for r in merged["results"]}
    assert "Q1" in results
    assert "Q2" in results
    assert "B0" in results["Q1"]["baselines"]
    assert "B1" in results["Q1"]["baselines"]
    
    assert merge_evaluation_data(None, {"test": 1}) == {"test": 1}

def test_run_benchmarking_consume_contexts(tmp_path):
    dataset_path = tmp_path / "dataset.yaml"
    dataset_data = [
        {"id": "Q1", "query": "What is deep learning?", "expected_papers": ["paper_1"]}
    ]
    with open(dataset_path, "w") as f:
        yaml.safe_dump(dataset_data, f)

    pre_contexts_path = tmp_path / "contexts.yaml"
    pre_contexts_data = [
        {
            "id": "Q1",
            "query": "What is deep learning?",
            "expected_papers": ["paper_1"],
            "baselines": {
                "CUSTOM": {
                    "status": "success",
                    "retrieved_papers": ["paper_1"],
                    "retrieved_chunks": [
                        {"id": "c1", "paper_id": "paper_1", "text_content": "chunk content", "page_number": 1}
                    ],
                    "trimmed_text": "chunk content",
                    "trimmed_graph": "",
                    "enrichment_block": "",
                    "metrics": {
                        "components": {
                            "embedding": {"calls": 1, "time_sec": 0.1}
                        },
                        "total_io_calls": 1
                    },
                    "latency_sec": 0.1
                }
            }
        }
    ]
    with open(pre_contexts_path, "w") as f:
        yaml.safe_dump(pre_contexts_data, f)
        
    output_path = tmp_path / "output.yaml"
    
    args = MagicMock()
    args.dataset = str(dataset_path)
    args.consume_contexts = str(pre_contexts_path)
    args.cloud = False
    args.baselines = "CUSTOM"
    args.output = str(output_path)
    args.no_unique_dir = True
    args.clear_checkpoint = False
    args.limit = 1
    
    config = MagicMock()
    config.data = {
        "llm": {
            "provider": "local",
            "local": {"model_path": "test_model"},
            "temp": 0.1,
            "max_tokens": 100
        },
        "embedding": {
            "model_name": "test_emb"
        },
        "rag_components": {
            "reranker": False,
            "citation_repair": True,
            "shannon_estimator_enabled": False,
        }
    }
    config.rag_components = config.data["rag_components"]
    config.llm_model_max_context = 4096
    config.reranker_model_name = "disabled"
    
    prompts = MagicMock()
    prompts.get_prompt.return_value = "dummy system prompt"
    
    rag_service = MagicMock()
    rag_service.llm_engine._ensure_model_loaded = MagicMock()
    rag_service.llm_engine.generate_response.return_value = "generated response"
    rag_service.llm_engine.count_tokens.return_value = 5
    rag_service._validate_and_repair_citations.return_value = "repaired answer"
    
    container = MagicMock()
    container.get_rag_service.return_value = rag_service
    
    con = MagicMock()
    
    run_benchmarking(args, config, prompts, container, con)
    
    assert output_path.exists()
    with open(output_path, "r") as f:
        out_data = yaml.safe_load(f)
    assert len(out_data["results"]) == 1
    assert out_data["results"][0]["baselines"]["CUSTOM"]["status"] == "success"
    assert out_data["results"][0]["baselines"]["CUSTOM"]["generated_answer"] == "generated response"
    rag_service._validate_and_repair_citations.assert_called_once()


def test_run_benchmarking_consume_contexts_attaches_tokens_info(tmp_path):
    """Bug A: consume_contexts path must attach tokens_info when logits generation runs."""
    dataset_path = tmp_path / "dataset.yaml"
    with open(dataset_path, "w") as f:
        yaml.safe_dump(
            [{"id": "Q1", "query": "What is deep learning?", "expected_papers": ["paper_1"]}],
            f,
        )

    pre_contexts_path = tmp_path / "contexts.yaml"
    with open(pre_contexts_path, "w") as f:
        yaml.safe_dump(
            [
                {
                    "id": "Q1",
                    "query": "What is deep learning?",
                    "expected_papers": ["paper_1"],
                    "baselines": {
                        "CUSTOM": {
                            "status": "success",
                            "retrieved_papers": ["paper_1"],
                            "retrieved_chunks": [
                                {
                                    "id": "c1",
                                    "paper_id": "paper_1",
                                    "text_content": "chunk content",
                                    "page_number": 1,
                                    "score": 0.9,
                                }
                            ],
                            "trimmed_text": "chunk content",
                            "trimmed_graph": "",
                            "enrichment_block": "",
                            "metrics": {
                                "components": {
                                    "embedding": {"calls": 1, "time_sec": 0.1}
                                },
                                "total_io_calls": 1,
                            },
                            "latency_sec": 0.1,
                        }
                    },
                }
            ],
            f,
        )

    output_path = tmp_path / "output.yaml"
    mock_tokens = [
        {
            "token": "Deep",
            "logprob": -0.1,
            "top_logprobs": {"Deep": -0.1, "Shallow": -2.0},
        },
        {
            "token": " learning",
            "logprob": -0.2,
            "top_logprobs": {" learning": -0.2, " thinking": -1.5},
        },
    ]

    args = MagicMock()
    args.dataset = str(dataset_path)
    args.consume_contexts = str(pre_contexts_path)
    args.cloud = False
    args.baselines = "CUSTOM"
    args.output = str(output_path)
    args.no_unique_dir = True
    args.clear_checkpoint = False
    args.limit = 1
    args.logit_save = True

    config = MagicMock()
    config.data = {
        "llm": {
            "provider": "local",
            "local": {"model_path": "test_model"},
            "temp": 0.1,
            "max_tokens": 100,
        },
        "embedding": {"model_name": "test_emb"},
        "rag_components": {
            "reranker": False,
            "citation_repair": False,
            "shannon_estimator_enabled": True,
        },
    }
    config.rag_components = config.data["rag_components"]
    config.llm_model_max_context = 4096
    config.reranker_model_name = "disabled"

    prompts = MagicMock()
    prompts.get_prompt.return_value = "dummy system prompt"

    # Use a non-MagicMock engine so _generate_with_logits_safe uses the real logits path
    class StubEngine:
        def _ensure_model_loaded(self):
            return None

        def generate_response_with_logits(self, prompt):
            return "Deep learning", list(mock_tokens)

        def generate_response(self, prompt):
            return "Deep learning"

        def count_tokens(self, text):
            return 5

    rag_service = MagicMock()
    rag_service.llm_engine = StubEngine()
    rag_service._validate_and_repair_citations.side_effect = lambda a, c: a
    rag_service._query_b0_h_gen = {}

    container = MagicMock()
    container.get_rag_service.return_value = rag_service

    with patch("core.generation.score_text_logprobs_base", return_value=list(mock_tokens)):
        with patch("core.generation._ensure_b0_entropy", return_value=1.0):
            run_benchmarking(args, config, prompts, container, MagicMock())

    assert output_path.exists()
    with open(output_path, "r") as f:
        out_data = yaml.safe_load(f)

    baseline = out_data["results"][0]["baselines"]["CUSTOM"]
    assert baseline["status"] == "success"
    metrics_tokens = baseline.get("metrics", {}).get("tokens_info") or baseline.get("tokens_info")
    assert metrics_tokens is not None
    assert len(metrics_tokens) == 2
    assert metrics_tokens[0]["token"] == "Deep"

    sh_diag = baseline.get("metrics", {}).get("shannon_diagnostics") or baseline.get(
        "shannon_diagnostics"
    )
    assert sh_diag is not None
    assert "avg_msp" in sh_diag
    assert "avg_logit_margin" in sh_diag
    assert "first_token_msp" in sh_diag
    assert "msp" in sh_diag
    assert "ll_rag" in sh_diag

def test_run_benchmarking_no_pre_retrieved(tmp_path):
    dataset_path = tmp_path / "dataset.yaml"
    dataset_data = [
        {"id": "Q1", "query": "What is deep learning?", "expected_papers": ["paper_1"]}
    ]
    with open(dataset_path, "w") as f:
        yaml.safe_dump(dataset_data, f)
        
    output_path = tmp_path / "output.yaml"
    
    args = MagicMock()
    args.dataset = str(dataset_path)
    args.consume_contexts = None
    args.cloud = True
    args.baselines = "B0,B1"
    args.output = str(output_path)
    args.no_unique_dir = True
    args.clear_checkpoint = False
    args.limit = 1
    
    config = MagicMock()
    config.data = {
        "llm": {
            "provider": "cloud",
            "cloud": {"model_name": "cloud_model", "provider": "openai"},
            "temp": 0.1,
            "max_tokens": 100
        },
        "embedding": {
            "model_name": "test_emb"
        },
        "rag_components": {
            "reranker": False,
            "citation_repair": False
        }
    }
    config.rag_components = config.data["rag_components"]
    config.llm_model_max_context = 4096
    config.reranker_model_name = "disabled"
    
    prompts = MagicMock()
    
    rag_service = MagicMock()
    chunk = MockChunk("c1", "p1", "some text content", 2)
    rag_service.retrieve_relevant_chunks.return_value = [(chunk, 0.95)]
    rag_service.ask.return_value = "ask response"
    rag_service.last_raw_response = "raw response"
    rag_service.llm_engine.generate_response.return_value = "generated response"
    rag_service.llm_engine.count_tokens.return_value = 5
    
    container = MagicMock()
    container.get_rag_service.return_value = rag_service
    
    con = MagicMock()
    
    run_benchmarking(args, config, prompts, container, con)
    
    assert output_path.exists()


def test_trace_serialization(tmp_path):
    from core.reporting import export_detailed_csv
    
    # Create fake report data
    report_data = {
        "metadata": {"date": "2026-06-29"},
        "results": [
            {
                "id": "Q1",
                "category": "deep_learning",
                "query": "query",
                "golden_answer": "golden",
                "expected_papers": ["paper_1"],
                "baselines": {
                    "B1": {
                        "status": "success",
                        "latency_sec": 1.23,
                        "retrieved_papers": ["paper_1"],
                        "eval_metrics": {"answer_relevance": 0.9},
                        "trace": {
                            "query_id": "Q1",
                            "category": "deep_learning",
                            "seed_chunks_from_lexical_dense": {
                                "lexical": ["c1"],
                                "dense": ["c2"]
                            },
                            "seed_paper_id_list": ["paper_1"],
                            "graph_neighbor_paper_id_list": ["paper_2"],
                            "candidate_count_before_reranker": 10,
                            "candidate_count_after_reranker": 5,
                            "final_context_paper_id_list": ["paper_1"],
                            "final_context_token_count": 200,
                            "whether_graph_neighbor_chunk_survived_into_final_context": True,
                            "answer_token_count": 50
                        }
                    }
                }
            }
        ]
    }
    
    csv_path = tmp_path / "metrics_details.csv"
    stats = {
        "baselines": ["B1"],
        "total_queries": 1
    }
    
    export_detailed_csv(report_data, stats, csv_path)
    
    assert csv_path.exists()
    import csv
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    
    headers = reader[0]
    row = reader[1]
    
    assert "seed_chunks_from_lexical_dense" in headers
    assert "seed_paper_id_list" in headers
    
    # Check that row values are written correctly
    assert "lexical:c1|dense:c2" in row
    assert "paper_1" in row
    assert "paper_2" in row
    assert "10" in row
    assert "5" in row
    assert "200" in row
    assert "True" in row
    assert "50" in row

def test_run_benchmarking_dataset_missing():
    args = MagicMock()
    args.dataset = "nonexistent_dataset.yaml"
    args.consume_contexts = None
    con = MagicMock()
    with pytest.raises(SystemExit):
        run_benchmarking(args, MagicMock(), MagicMock(), MagicMock(), con)
    con.error.assert_called_with("Dataset file not found: nonexistent_dataset.yaml")

@patch("core.config.load_benchmark_dataset", side_effect=Exception("parse error"))
def test_run_benchmarking_dataset_parse_fail(mock_load):
    args = MagicMock()
    args.dataset = "dataset.yaml"
    args.consume_contexts = None
    with patch("core.generation.Path.exists", return_value=True):
        with pytest.raises(SystemExit):
            run_benchmarking(args, MagicMock(), MagicMock(), MagicMock(), MagicMock())

@patch("core.config.load_benchmark_dataset", return_value=[])
def test_run_benchmarking_dataset_empty(mock_load):
    args = MagicMock()
    args.dataset = "dataset.yaml"
    args.consume_contexts = None
    with patch("core.generation.Path.exists", return_value=True):
        with pytest.raises(SystemExit):
            run_benchmarking(args, MagicMock(), MagicMock(), MagicMock(), MagicMock())

def test_run_benchmarking_rag_service_fail(tmp_path):
    args = MagicMock()
    args.dataset = str(tmp_path / "dataset.yaml")
    args.consume_contexts = None
    (tmp_path / "dataset.yaml").write_text("dataset")
    container = MagicMock()
    container.get_rag_service.side_effect = Exception("init error")
    with patch("core.config.load_benchmark_dataset", return_value=[{"id": "q1"}]):
        with pytest.raises(SystemExit):
            run_benchmarking(args, MagicMock(), MagicMock(), container, MagicMock())

def test_run_benchmarking_all_baselines(tmp_path):
    # Tests exit / setup with 'all' baselines
    args = MagicMock()
    args.dataset = str(tmp_path / "dataset.yaml")
    (tmp_path / "dataset.yaml").write_text("dataset")
    args.baselines = "all"
    args.consume_contexts = None
    args.cloud = True
    args.output = str(tmp_path / "output.yaml")
    args.no_unique_dir = True
    args.clear_checkpoint = False
    args.limit = 1

    config = MagicMock()
    config.data = {
        "llm": {
            "provider": "cloud",
            "cloud": {"model_name": "cloud_model"},
            "temp": 0.1,
            "max_tokens": 100
        },
        "embedding": {"model_name": "emb"},
        "rag_components": {"reranker": False}
    }
    config.rag_components = config.data["rag_components"]
    config.is_component_enabled = MagicMock(return_value=False)
    config.llm_model_max_context = 4096
    config.reranker_model_name = "disabled"

    rag_service = MagicMock()
    rag_service.retrieve_relevant_chunks.return_value = []
    rag_service.ask.return_value = "ans"
    rag_service.llm_engine.generate_response.return_value = "gen"
    rag_service.llm_engine.count_tokens.return_value = 5

    container = MagicMock()
    container.get_rag_service.return_value = rag_service

    with patch("core.config.load_benchmark_dataset", return_value=[{"id": "q1", "query": "Q1"}]):
        run_benchmarking(args, config, MagicMock(), container, MagicMock())
    assert (tmp_path / "output.yaml").exists()

def test_run_benchmarking_unique_dir(tmp_path):
    args = MagicMock()
    args.dataset = str(tmp_path / "dataset.yaml")
    (tmp_path / "dataset.yaml").write_text("dataset")
    args.baselines = "B0"
    args.consume_contexts = None
    args.cloud = False
    args.output = str(tmp_path / "reports" / "output.yaml")
    args.no_unique_dir = False # triggers unique directory creation
    args.clear_checkpoint = False
    args.limit = 1

    config = MagicMock()
    config.data = {
        "llm": {
            "provider": "local",
            "local": {"model_path": "local_model"},
            "temp": 0.1,
            "max_tokens": 100
        },
        "embedding": {"model_name": "emb"},
        "rag_components": {"reranker": False}
    }
    config.rag_components = config.data["rag_components"]
    config.is_component_enabled = MagicMock(return_value=False)
    config.llm_model_max_context = 4096
    config.reranker_model_name = "disabled"

    rag_service = MagicMock()
    rag_service.llm_engine.generate_response.return_value = "gen"
    rag_service.llm_engine.count_tokens.return_value = 5

    container = MagicMock()
    container.get_rag_service.return_value = rag_service

    with patch("core.config.load_benchmark_dataset", return_value=[{"id": "q1", "query": "Q1"}]):
        run_benchmarking(args, config, MagicMock(), container, MagicMock())
    # Should create unique directory under tmp_path/reports
    subdirs = [d for d in (tmp_path / "reports").iterdir() if d.is_dir()]
    assert len(subdirs) == 1
    assert (subdirs[0] / "output.yaml").exists()

def test_run_benchmarking_existing_results(tmp_path):
    args = MagicMock()
    args.dataset = str(tmp_path / "dataset.yaml")
    (tmp_path / "dataset.yaml").write_text("dataset")
    args.baselines = "B0"
    args.consume_contexts = None
    args.cloud = True
    args.output = str(tmp_path / "output.yaml")
    args.no_unique_dir = True
    args.clear_checkpoint = False # triggers reuse of existing results
    args.limit = 1

    # Create existing output
    existing_output_data = {
        "metadata": {"baselines_evaluated": ["B0"]},
        "results": [
            {
                "id": "q1",
                "query": "Q1",
                "baselines": {
                    "B0": {
                        "status": "success",
                        "latency_sec": 0.5,
                        "generated_answer": "pre-existing answer"
                    }
                }
            }
        ]
    }
    with open(tmp_path / "output.yaml", "w") as f:
        yaml.safe_dump(existing_output_data, f)

    config = MagicMock()
    config.data = {
        "llm": {
            "provider": "cloud",
            "cloud": {"model_name": "cloud_model"},
            "temp": 0.1,
            "max_tokens": 100
        },
        "embedding": {"model_name": "emb"},
        "rag_components": {"reranker": False}
    }
    config.rag_components = config.data["rag_components"]
    config.is_component_enabled = MagicMock(return_value=False)
    config.llm_model_max_context = 4096
    config.reranker_model_name = "disabled"

    rag_service = MagicMock()
    container = MagicMock()
    container.get_rag_service.return_value = rag_service

    with patch("core.config.load_benchmark_dataset", return_value=[{"id": "q1", "query": "Q1"}]):
        run_benchmarking(args, config, MagicMock(), container, MagicMock())
    
    # Read the output and verify that it reused the pre-existing answer
    with open(tmp_path / "output.yaml", "r") as f:
        res_data = yaml.safe_load(f)
    assert res_data["results"][0]["baselines"]["B0"]["generated_answer"] == "pre-existing answer"

def test_run_benchmarking_consume_contexts_missing(tmp_path):
    args = MagicMock()
    args.dataset = str(tmp_path / "dataset.yaml")
    (tmp_path / "dataset.yaml").write_text("dataset")
    args.baselines = "B1"
    args.consume_contexts = str(tmp_path / "contexts.yaml")
    
    # Empty contexts YAML
    with open(tmp_path / "contexts.yaml", "w") as f:
        yaml.safe_dump([], f)
        
    args.cloud = True
    args.output = str(tmp_path / "output.yaml")
    args.no_unique_dir = True
    args.clear_checkpoint = False
    args.limit = 1

    config = MagicMock()
    config.data = {
        "llm": {
            "provider": "cloud",
            "cloud": {"model_name": "cloud_model"},
            "temp": 0.1,
            "max_tokens": 100
        },
        "embedding": {"model_name": "emb"},
        "rag_components": {"reranker": False}
    }
    config.rag_components = config.data["rag_components"]
    config.is_component_enabled = MagicMock(return_value=False)
    config.llm_model_max_context = 4096
    config.reranker_model_name = "disabled"

    rag_service = MagicMock()
    container = MagicMock()
    container.get_rag_service.return_value = rag_service

    with patch("core.config.load_benchmark_dataset", return_value=[{"id": "q1", "query": "Q1"}]):
        run_benchmarking(args, config, MagicMock(), container, MagicMock())
    
    with open(tmp_path / "output.yaml", "r") as f:
        res_data = yaml.safe_load(f)
    assert res_data["results"][0]["baselines"]["B1"]["status"] == "error"
    assert "Error: No pre-retrieved context" in res_data["results"][0]["baselines"]["B1"]["generated_answer"]

def test_run_benchmarking_consume_contexts_b0_and_tokens_fallback(tmp_path):
    args = MagicMock()
    args.dataset = str(tmp_path / "dataset.yaml")
    (tmp_path / "dataset.yaml").write_text("dataset")
    args.baselines = "B0,B1"
    args.consume_contexts = str(tmp_path / "contexts.yaml")
    
    pre_contexts_data = [
        {
            "id": "q1",
            "query": "Q1",
            "baselines": {
                "B0": {
                    "status": "success",
                    "latency_sec": 0.1,
                },
                "B1": {
                    "status": "success",
                    "latency_sec": 0.5,
                    "retrieved_chunks": [{"id": "c1", "paper_id": "p1", "text_content": "text", "page_number": 1}],
                    "trimmed_text": "text",
                    "trimmed_graph": "graph",
                    "enrichment_block": "No essential knowledge graph enrichment found.",
                    "metrics": {}
                }
            }
        }
    ]
    with open(tmp_path / "contexts.yaml", "w") as f:
        yaml.safe_dump(pre_contexts_data, f)
        
    args.cloud = True
    args.output = str(tmp_path / "output.yaml")
    args.no_unique_dir = True
    args.clear_checkpoint = False
    args.limit = 1

    config = MagicMock()
    config.data = {
        "llm": {
            "provider": "cloud",
            "cloud": {"model_name": "cloud_model"},
            "temp": 0.1,
            "max_tokens": 100
        },
        "embedding": {"model_name": "emb"},
        "rag_components": {"reranker": False, "citation_repair": True}
    }
    config.rag_components = config.data["rag_components"]
    config.is_component_enabled = MagicMock(return_value=False)
    config.llm_model_max_context = 4096
    config.reranker_model_name = "disabled"

    prompts = MagicMock()
    prompts.get_prompt.return_value = "prompt"

    rag_service = MagicMock()
    rag_service.llm_engine.generate_response.return_value = "response"
    # Make count_tokens raise exception to cover line 338-339
    rag_service.llm_engine.count_tokens.side_effect = Exception("token count error")
    # Make citation repair raise exception to cover line 355-357
    rag_service._validate_and_repair_citations.side_effect = Exception("repair error")

    container = MagicMock()
    container.get_rag_service.return_value = rag_service

    with patch("core.config.load_benchmark_dataset", return_value=[{"id": "q1", "query": "Q1"}]):
        run_benchmarking(args, config, prompts, container, MagicMock())
    
    assert (tmp_path / "output.yaml").exists()

def test_run_benchmarking_execution_exceptions(tmp_path):
    args = MagicMock()
    args.dataset = str(tmp_path / "dataset.yaml")
    (tmp_path / "dataset.yaml").write_text("dataset")
    args.baselines = "B0"
    args.consume_contexts = None
    args.cloud = True
    args.output = str(tmp_path / "output.yaml")
    args.no_unique_dir = True
    args.clear_checkpoint = False
    args.limit = 1

    config = MagicMock()
    config.data = {
        "llm": {
            "provider": "cloud",
            "cloud": {"model_name": "cloud_model"},
            "temp": 0.1,
            "max_tokens": 100
        },
        "embedding": {"model_name": "emb"},
        "rag_components": {"reranker": False}
    }
    config.rag_components = config.data["rag_components"]
    config.is_component_enabled = MagicMock(return_value=False)
    config.llm_model_max_context = 4096
    config.reranker_model_name = "disabled"

    rag_service = MagicMock()
    # Trigger exception in run_query_on_baseline to cover lines 420-436
    rag_service.llm_engine.generate_response.side_effect = Exception("generation execution error")

    container = MagicMock()
    container.get_rag_service.return_value = rag_service

    with patch("core.config.load_benchmark_dataset", return_value=[{"id": "q1", "query": "Q1"}]):
        run_benchmarking(args, config, MagicMock(), container, MagicMock())
    
    with open(tmp_path / "output.yaml", "r") as f:
        res_data = yaml.safe_load(f)
    assert res_data["results"][0]["baselines"]["B0"]["status"] == "error"
    assert "Error occurred during generation" in res_data["results"][0]["baselines"]["B0"]["generated_answer"]

def test_run_benchmarking_autosave_fail(tmp_path):
    args = MagicMock()
    args.dataset = str(tmp_path / "dataset.yaml")
    (tmp_path / "dataset.yaml").write_text("dataset")
    args.baselines = "B0"
    args.consume_contexts = None
    args.cloud = True
    args.output = str(tmp_path / "output.yaml")
    args.no_unique_dir = True
    args.clear_checkpoint = False
    args.limit = 1

    config = MagicMock()
    config.data = {
        "llm": {
            "provider": "cloud",
            "cloud": {"model_name": "cloud_model"},
            "temp": 0.1,
            "max_tokens": 100
        },
        "embedding": {"model_name": "emb"},
        "rag_components": {
            "reranker": False
        }
    }
    config.rag_components = config.data["rag_components"]
    config.is_component_enabled = MagicMock(return_value=False)
    config.llm_model_max_context = 4096
    config.reranker_model_name = "disabled"

    rag_service = MagicMock()
    rag_service.llm_engine.generate_response.return_value = "gen"
    rag_service.llm_engine.count_tokens.return_value = 5

    container = MagicMock()
    container.get_rag_service.return_value = rag_service

    # Mock open when saving temp file to raise exception, triggering lines 508-509
    orig_open = open
    def mock_open(file, *args, **kwargs):
        if str(file).endswith(".tmp"):
            raise OSError("mock autosave write error")
        return orig_open(file, *args, **kwargs)

    with patch("core.config.load_benchmark_dataset", return_value=[{"id": "q1", "query": "Q1"}]), \
         patch("builtins.open", side_effect=mock_open):
        run_benchmarking(args, config, MagicMock(), container, MagicMock())
    
    assert (tmp_path / "output.yaml").exists()

def test_run_benchmarking_summary_table_missing_metrics(tmp_path):
    # Runs benchmarking, but mock final results to have recall/precision as None,
    # triggering lines 562-568 and 585
    args = MagicMock()
    args.dataset = str(tmp_path / "dataset.yaml")
    (tmp_path / "dataset.yaml").write_text("dataset")
    args.baselines = "B1"
    args.consume_contexts = None
    args.cloud = True
    args.output = str(tmp_path / "output.yaml")
    args.no_unique_dir = True
    args.clear_checkpoint = False
    args.limit = 1

    config = MagicMock()
    config.data = {
        "llm": {
            "provider": "cloud",
            "cloud": {"model_name": "cloud_model"},
            "temp": 0.1,
            "max_tokens": 100
        },
        "embedding": {"model_name": "emb"},
        "rag_components": {"reranker": False}
    }
    config.rag_components = config.data["rag_components"]
    config.is_component_enabled = MagicMock(return_value=False)
    config.llm_model_max_context = 4096
    config.reranker_model_name = "disabled"

    rag_service = MagicMock()
    chunk = MockChunk("c1", "p1", "some text content", 2)
    rag_service.retrieve_relevant_chunks.return_value = [(chunk, 0.95)]
    rag_service.ask.return_value = "ask response"
    rag_service.last_raw_response = "raw response"
    rag_service.llm_engine.generate_response.return_value = "generated response"
    rag_service.llm_engine.count_tokens.return_value = 5

    container = MagicMock()
    container.get_rag_service.return_value = rag_service

    # Mutate merge_evaluation_data to clear recall/precision metrics to trigger recalculation
    orig_merge = merge_evaluation_data
    def mock_merge(existing, new):
        res = orig_merge(existing, new)
        for item in res["results"]:
            for b in item["baselines"].values():
                b["retrieval_recall"] = None
                b["context_precision"] = None
                # Set status to error to cover line 585
                b["status"] = "error"
        return res

    with patch("core.config.load_benchmark_dataset", return_value=[{"id": "q1", "query": "Q1", "expected_papers": ["p1"]}]), \
         patch("core.generation.merge_evaluation_data", side_effect=mock_merge):
        run_benchmarking(args, config, MagicMock(), container, MagicMock())
    
    assert (tmp_path / "output.yaml").exists()

def test_run_benchmarking_summary_table_error(tmp_path):
    args = MagicMock()
    args.dataset = str(tmp_path / "dataset.yaml")
    (tmp_path / "dataset.yaml").write_text("dataset")
    args.baselines = "B0"
    args.consume_contexts = None
    args.cloud = True
    args.output = str(tmp_path / "output.yaml")
    args.no_unique_dir = True
    args.clear_checkpoint = False
    args.limit = 1

    config = MagicMock()
    config.data = {
        "llm": {
            "provider": "cloud",
            "cloud": {"model_name": "cloud_model"},
            "temp": 0.1,
            "max_tokens": 100
        },
        "embedding": {"model_name": "emb"},
        "rag_components": {"reranker": False}
    }
    config.rag_components = config.data["rag_components"]
    config.is_component_enabled = MagicMock(return_value=False)
    config.llm_model_max_context = 4096
    config.reranker_model_name = "disabled"

    rag_service = MagicMock()
    rag_service.llm_engine.generate_response.return_value = "gen"
    rag_service.llm_engine.count_tokens.return_value = 5

    container = MagicMock()
    container.get_rag_service.return_value = rag_service

    # Mock rich.table.Table creation to raise exception, triggering lines 604-605
    with patch("core.config.load_benchmark_dataset", return_value=[{"id": "q1", "query": "Q1"}]), \
         patch("rich.table.Table", side_effect=ValueError("mock table error")):
        run_benchmarking(args, config, MagicMock(), container, MagicMock())
    
    assert (tmp_path / "output.yaml").exists()

def test_run_benchmarking_unload_model_error(tmp_path):
    args = MagicMock()
    args.dataset = str(tmp_path / "dataset.yaml")
    (tmp_path / "dataset.yaml").write_text("dataset")
    args.baselines = "B0"
    args.consume_contexts = None
    args.cloud = True
    args.output = str(tmp_path / "output.yaml")
    args.no_unique_dir = True
    args.clear_checkpoint = False
    args.limit = 1

    config = MagicMock()
    config.data = {
        "llm": {
            "provider": "cloud",
            "cloud": {"model_name": "cloud_model"},
            "temp": 0.1,
            "max_tokens": 100
        },
        "embedding": {"model_name": "emb"},
        "rag_components": {"reranker": False}
    }
    config.rag_components = config.data["rag_components"]
    config.is_component_enabled = MagicMock(return_value=False)
    config.llm_model_max_context = 4096
    config.reranker_model_name = "disabled"

    rag_service = MagicMock()
    rag_service.llm_engine.generate_response.return_value = "gen"
    rag_service.llm_engine.count_tokens.return_value = 5
    # Trigger exception on unload_model, covering lines 610-611
    rag_service.llm_engine.unload_model.side_effect = Exception("mock unload error")

    container = MagicMock()
    container.get_rag_service.return_value = rag_service

    with patch("core.config.load_benchmark_dataset", return_value=[{"id": "q1", "query": "Q1"}]):
        run_benchmarking(args, config, MagicMock(), container, MagicMock())
    
    assert (tmp_path / "output.yaml").exists()


def test_merge_evaluation_data_type_fallbacks():
    existing = {
        "metadata": {
            "baselines_evaluated": "not_a_list"
        },
        "results": "not_a_list"
    }
    new_data = {
        "metadata": {
            "baselines_evaluated": "not_a_list"
        },
        "results": [
            {
                "id": "Q1",
                "baselines": "not_a_dict"
            }
        ]
    }
    
    res = merge_evaluation_data(existing, new_data)
    assert res["metadata"]["baselines_evaluated"] == []
    assert len(res["results"]) == 1


def test_merge_evaluation_data_sort_exception():
    existing = {
        "metadata": {},
        "results": [
            {"id": 1},
            {"id": "string_id"}
        ]
    }
    new_data = {
        "metadata": {},
        "results": []
    }
    res = merge_evaluation_data(existing, new_data)
    assert len(res["results"]) == 2


def test_mlx_engine_sampler_caching_and_cleanup(tmp_path):
    from unittest.mock import patch
    try:
        from src.llm_engine.mlx_impl import MlxLLMEngine
    except ImportError:
        pytest.skip("MLX not installed")

    dummy_dir = tmp_path / "dummy_mlx_model"
    dummy_dir.mkdir()

    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())):
        engine = MlxLLMEngine(model_path=str(dummy_dir))
        with patch("mlx_lm.sample_utils.make_sampler", side_effect=lambda temp: f"sampler_{temp}") as mock_make:
            s1 = engine._get_sampler(0.1)
            s2 = engine._get_sampler(0.1)
            assert s1 == s2
            assert mock_make.call_count == 1

        engine.unload_model()
        assert len(engine._sampler_cache) == 0


def test_run_benchmarking_checkpoint_reuse_skips_redundant_autosave(tmp_path):
    """Test that when all baselines for test cases are reused from checkpoints, redundant autosave writes are skipped."""
    dataset_path = tmp_path / "dataset.yaml"
    dataset_data = [
        {"id": "q1", "query": "Q1"},
        {"id": "q2", "query": "Q2"},
        {"id": "q3", "query": "Q3"},
        {"id": "q4", "query": "Q4"},
        {"id": "q5", "query": "Q5"},
    ]
    with open(dataset_path, "w") as f:
        yaml.safe_dump(dataset_data, f)

    output_path = tmp_path / "output.yaml"
    existing_output_data = {
        "metadata": {"baselines_evaluated": ["B0"]},
        "results": [
            {"id": f"q{i}", "query": f"Q{i}", "baselines": {"B0": {"status": "success", "latency_sec": 0.1, "generated_answer": f"ans{i}"}}}
            for i in range(1, 6)
        ]
    }
    with open(output_path, "w") as f:
        yaml.safe_dump(existing_output_data, f)

    args = MagicMock()
    args.dataset = str(dataset_path)
    args.baselines = "B0"
    args.consume_contexts = None
    args.cloud = True
    args.output = str(output_path)
    args.no_unique_dir = True
    args.clear_checkpoint = False
    args.limit = 5
    args.logit_save = False

    config = MagicMock()
    config.data = {
        "llm": {"provider": "cloud", "cloud": {"model_name": "cloud_model"}, "temp": 0.1, "max_tokens": 100},
        "embedding": {"model_name": "emb"},
        "rag_components": {"reranker": False}
    }
    config.rag_components = config.data["rag_components"]
    config.is_component_enabled = MagicMock(return_value=False)
    config.llm_model_max_context = 4096
    config.reranker_model_name = "disabled"

    rag_service = MagicMock()
    container = MagicMock()
    container.get_rag_service.return_value = rag_service

    dump_call_count = 0
    orig_dump = yaml.dump
    def mock_dump(*args, **kwargs):
        nonlocal dump_call_count
        dump_call_count += 1
        return orig_dump(*args, **kwargs)

    with patch("core.config.load_benchmark_dataset", return_value=dataset_data), \
         patch("yaml.dump", side_effect=mock_dump), \
         patch("core.generation._clear_gpu_cache") as mock_clear_gpu:
        run_benchmarking(args, config, MagicMock(), container, MagicMock())

    # Verify that autosave dumps were skipped during pure checkpoint reuses
    assert dump_call_count == 0
    # Verify that _clear_gpu_cache was not called during pure checkpoint reuses
    assert mock_clear_gpu.call_count == 0



