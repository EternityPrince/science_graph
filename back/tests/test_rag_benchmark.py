import unittest
from unittest.mock import MagicMock, patch
import yaml
from pathlib import Path

from benchmarks.rag.run_benchmarks import get_baseline_config, run_query_on_baseline
from benchmarks.rag.core.config import BASELINES_INFO
from src.config import config
from src.services.rag_service import RAGService

class TestRagBenchmark(unittest.TestCase):
    def test_get_baseline_config_validation(self):
        """Verifies that RAG component toggles are correct for all baselines."""
        # B0: Zero-shot (No RAG components should be True)
        b0_cfg = get_baseline_config("B0")
        self.assertFalse(any(b0_cfg.values()))
        
        # B1: Pure Lexical (Only lexical_search should be True)
        b1_cfg = get_baseline_config("B1")
        self.assertTrue(b1_cfg["lexical_search"])
        self.assertFalse(any(v for k, v in b1_cfg.items() if k != "lexical_search"))
        
        # B2: Pure Dense (Only dense_search should be True)
        b2_cfg = get_baseline_config("B2")
        self.assertTrue(b2_cfg["dense_search"])
        self.assertFalse(any(v for k, v in b2_cfg.items() if k != "dense_search"))
        
        # B3: Dense + HyDE
        b3_cfg = get_baseline_config("B3")
        self.assertTrue(b3_cfg["dense_search"])
        self.assertTrue(b3_cfg["hyde"])
        
        # B4: Standard Hybrid (Lexical + Dense + RRF)
        b4_cfg = get_baseline_config("B4")
        self.assertTrue(b4_cfg["lexical_search"])
        self.assertTrue(b4_cfg["dense_search"])
        self.assertTrue(b4_cfg["rrf"])
        self.assertFalse(b4_cfg["graph_expansion"])
        self.assertTrue(b4_cfg["reranker"])
        
        # B5: Hybrid + Graph (Lexical + Dense + RRF + Graph)
        b5_cfg = get_baseline_config("B5")
        self.assertTrue(b5_cfg["lexical_search"])
        self.assertTrue(b5_cfg["dense_search"])
        self.assertTrue(b5_cfg["graph_expansion"])
        self.assertTrue(b5_cfg["reranker"])
        
        # B6: Full Pipeline (All components should be True except hyde and intent_classifier)
        b6_cfg = get_baseline_config("B6")
        self.assertFalse(b6_cfg["hyde"])
        self.assertFalse(b6_cfg["intent_classifier"])
        self.assertTrue(all(v for k, v in b6_cfg.items() if k not in ("hyde", "intent_classifier", "graph_concept_retrieval", "graph_bridge_retrieval", "graph_selected_sources_card", "graph_retrieval_trace")))

    def test_example_golden_dataset_structure(self):
        """Validates that the example golden dataset YAML file has a correct schema."""
        example_path = Path(__file__).resolve().parents[1] / "benchmarks" / "rag" / "golden_dataset.example.yaml"
        self.assertTrue(example_path.exists(), f"Example dataset not found at: {example_path}")
        
        with open(example_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        
        for idx, item in enumerate(data):
            self.assertIsInstance(item, dict)
            self.assertIn("id", item, f"Missing 'id' in item {idx}")
            self.assertIn("category", item, f"Missing 'category' in item {idx}")
            self.assertIn("query", item, f"Missing 'query' in item {idx}")
            self.assertIn("golden_answer", item, f"Missing 'golden_answer' in item {idx}")
            self.assertIn("expected_papers", item, f"Missing 'expected_papers' in item {idx}")
            self.assertIsInstance(item["expected_papers"], list, f"'expected_papers' must be a list in item {idx}")

    @patch("src.services.rag_service.RAGService.ask")
    @patch("src.services.rag_service.RAGService.retrieve_relevant_chunks")
    def test_config_backup_and_restore(self, mock_retrieve, mock_ask):
        """Ensures run_query_on_baseline backs up and restores RAG config correctly."""
        # Set up a mock RAG service and engines
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = "Mocked Zero-Shot Answer"
        mock_llm.count_tokens.return_value = 10
        mock_ask.return_value = "Mocked RAG Answer"
        mock_retrieve.return_value = []
        
        mock_rag = MagicMock(spec=RAGService)
        mock_rag.llm_engine = mock_llm
        mock_rag.ask = mock_ask
        mock_rag.retrieve_relevant_chunks = mock_retrieve
        mock_rag.expander = None
        
        # Capture original config values
        original_components = config.rag_components.copy()
        original_hyde = getattr(config, "hyde_enabled", False)
        
        # Run B0 (Zero-Shot) - should call generate_response directly
        ans, retrieved, metrics, chunks = run_query_on_baseline(mock_rag, "Test Question B0", "B0", use_cloud=False)
        self.assertEqual(ans, "Mocked Zero-Shot Answer")
        self.assertEqual(retrieved, [])
        mock_llm.generate_response.assert_called_once()
        
        # Run B1 (Pure Lexical)
        mock_retrieve.return_value = [(MagicMock(paper_id="paper1"), 0.9)]
        ans_b1, retrieved_b1, metrics_b1, chunks_b1 = run_query_on_baseline(mock_rag, "Test Question B1", "B1", use_cloud=False)
        self.assertEqual(ans_b1, "Mocked RAG Answer")
        self.assertEqual(retrieved_b1, ["paper1"])
        
        # Verify that after running baselines, the config is restored back to original
        self.assertEqual(config.rag_components, original_components)
        self.assertEqual(config.hyde_enabled, original_hyde)

    def test_baselines_descriptions(self):
        """Verifies all B0-B6 baselines have descriptions configured."""
        self.assertIn(len(BASELINES_INFO), [7, 8])
        for i in range(7):
            key = f"B{i}"
            self.assertIn(key, BASELINES_INFO)
            self.assertIsInstance(BASELINES_INFO[key], str)
            self.assertTrue(len(BASELINES_INFO[key]) > 0)
        if "CUSTOM" in BASELINES_INFO:
            self.assertIsInstance(BASELINES_INFO["CUSTOM"], str)
            self.assertTrue(len(BASELINES_INFO["CUSTOM"]) > 0)

    def test_merge_evaluation_data(self):
        """Verifies that merge_evaluation_data successfully merges baseline reports."""
        from benchmarks.rag.core.generation import merge_evaluation_data
        
        existing = {
            "metadata": {
                "date": "2026-06-12 12:00:00",
                "llm": {"provider": "local", "model_name": "mock-model"},
                "baselines_evaluated": ["B0"]
            },
            "results": [
                {
                    "id": "Q01",
                    "query": "What is life?",
                    "baselines": {
                        "B0": {
                            "status": "success",
                            "generated_answer": "B0 Answer"
                        }
                    }
                }
            ]
        }
        
        new_data = {
            "metadata": {
                "date": "2026-06-12 13:00:00",
                "llm": {"provider": "local", "model_name": "mock-model"},
                "baselines_evaluated": ["B1"]
            },
            "results": [
                {
                    "id": "Q01",
                    "query": "What is life?",
                    "baselines": {
                        "B1": {
                            "status": "success",
                            "generated_answer": "B1 Answer"
                        }
                    }
                }
            ]
        }
        
        merged = merge_evaluation_data(existing, new_data)
        
        self.assertEqual(merged["metadata"]["baselines_evaluated"], ["B0", "B1"])
        self.assertEqual(len(merged["results"]), 1)
        self.assertIn("B0", merged["results"][0]["baselines"])
        self.assertIn("B1", merged["results"][0]["baselines"])
        self.assertEqual(merged["results"][0]["baselines"]["B0"]["generated_answer"], "B0 Answer")
        self.assertEqual(merged["results"][0]["baselines"]["B1"]["generated_answer"], "B1 Answer")

    def test_stats_collector(self):
        """Ensures BenchmarkStatsCollector records calls and timings correctly."""
        from benchmarks.rag.core.stats import BenchmarkStatsCollector
        
        mock_rag = MagicMock(spec=RAGService)
        mock_rag.emb_engine = MagicMock()
        mock_rag.vector_repo = MagicMock()
        mock_rag.graph_repo = MagicMock()
        mock_rag._reranker = MagicMock()
        mock_rag.llm_engine = MagicMock()
        mock_rag.llm_engine.count_tokens.return_value = 10
        
        collector = BenchmarkStatsCollector(mock_rag)
        collector.start()
        
        # Trigger some calls
        mock_rag.emb_engine.get_embedding("test")
        mock_rag.vector_repo.search_similar_chunks(None)
        mock_rag.graph_repo.get_neighbors("test")
        mock_rag.llm_engine.generate_response("prompt")
        
        # Retrieve stats
        metrics = collector.get_metrics()
        collector.stop()
        
        self.assertEqual(metrics["components"]["embedding"]["calls"], 1)
        self.assertEqual(metrics["components"]["dense_retrieval"]["calls"], 1)
        self.assertEqual(metrics["components"]["graph_neighbors"]["calls"], 1)
        self.assertEqual(metrics["components"]["llm_generation"]["calls"], 1)

    @patch("benchmarks.rag.run_benchmarks.container.get_rag_service")
    def test_run_benchmarks_command_line_incremental(self, mock_get_rag_service):
        """Tests that running the benchmarking runner with different baselines merges the results in the YAML output file."""
        import tempfile
        import sys
        from benchmarks.rag.run_benchmarks import main
        
        # Setup mock RAG service
        mock_rag = MagicMock(spec=RAGService)
        mock_rag.emb_engine = MagicMock()
        mock_rag.vector_repo = MagicMock()
        mock_rag.graph_repo = MagicMock()
        mock_rag._reranker = MagicMock()
        
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = "Mocked Response"
        mock_llm.count_tokens.return_value = 10
        mock_rag.llm_engine = mock_llm
        
        mock_rag.ask.return_value = "Mocked Ask Response"
        mock_rag.retrieve_relevant_chunks.return_value = []
        
        mock_get_rag_service.return_value = mock_rag
        
        # Create a temp directory for reports
        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = Path(tmpdir) / "reports"
            output_yaml = report_dir / "evaluation_results.yaml"
            
            # Create a test smoke dataset with 1 question
            dataset_yaml = Path(tmpdir) / "test_smoke.yaml"
            test_dataset = [
                {
                    "id": "Q01",
                    "category": "general",
                    "query": "What is the speed of light?",
                    "golden_answer": "299792458 m/s",
                    "expected_papers": ["paper1"]
                }
            ]
            with open(dataset_yaml, "w", encoding="utf-8") as f:
                yaml.dump(test_dataset, f)
                
            # First run: run B0 only
            test_args = [
                "run_benchmarks.py",
                "--dataset", str(dataset_yaml),
                "--output", str(output_yaml),
                "--baselines", "B0",
                "--no-unique-dir"
            ]
            
            with patch.object(sys, "argv", test_args):
                main()
                
            # Check report output for B0
            self.assertTrue(output_yaml.exists())
            with open(output_yaml, "r", encoding="utf-8") as f:
                first_run_data = yaml.safe_load(f)
                
            self.assertEqual(first_run_data["metadata"]["baselines_evaluated"], ["B0"])
            self.assertIn("B0", first_run_data["results"][0]["baselines"])
            self.assertNotIn("B1", first_run_data["results"][0]["baselines"])
            
            # Second run: run B1 only on the same file
            test_args_2 = [
                "run_benchmarks.py",
                "--dataset", str(dataset_yaml),
                "--output", str(output_yaml),
                "--baselines", "B1",
                "--no-unique-dir"
            ]
            
            with patch.object(sys, "argv", test_args_2):
                main()
                
            # Check report output has BOTH B0 and B1 (merged)
            with open(output_yaml, "r", encoding="utf-8") as f:
                second_run_data = yaml.safe_load(f)
                
            self.assertEqual(second_run_data["metadata"]["baselines_evaluated"], ["B0", "B1"])
            self.assertIn("B0", second_run_data["results"][0]["baselines"])
            self.assertIn("B1", second_run_data["results"][0]["baselines"])
            self.assertEqual(second_run_data["results"][0]["baselines"]["B0"]["status"], "success")
            self.assertEqual(second_run_data["results"][0]["baselines"]["B1"]["status"], "success")

    def test_csv_export_functions(self):
        """Verifies that export_wide_csv and export_detailed_csv generate correct CSV structures."""
        from benchmarks.rag.parse_metrics import export_wide_csv, export_detailed_csv
        import tempfile
        import csv
        
        # Mock statistics data structure
        stats = {
            "baselines": ["B0", "B1"],
            "summary": {
                "B0": {
                    "success_rate": 100.0,
                    "retrieval_recall": {"mean": 0.0, "count": 0},
                    "context_precision": {"mean": 0.0, "count": 0},
                    "faithfulness": {"mean": 0.0, "count": 0},
                    "answer_relevance": {"mean": 0.8, "count": 1},
                    "citation_fidelity": {"mean": 0.0, "count": 0},
                    "semantic_accuracy": {"mean": 0.75, "count": 1},
                    "latency_sec": {"mean": 5.432, "count": 1},
                    "token_output": {"mean": 0.0, "count": 0},
                    "token_answer": {"mean": 0.0, "count": 0},
                    "token_reasoning": {"mean": 0.0, "count": 0}
                },
                "B1": {
                    "success_rate": 50.0,
                    "retrieval_recall": {"mean": 0.9, "count": 1},
                    "context_precision": {"mean": 0.85, "count": 1},
                    "faithfulness": {"mean": 0.95, "count": 1},
                    "answer_relevance": {"mean": 0.9, "count": 1},
                    "citation_fidelity": {"mean": 1.0, "count": 1},
                    "semantic_accuracy": {"mean": 0.88, "count": 1},
                    "latency_sec": {"mean": 12.345, "count": 1},
                    "token_output": {"mean": 0.0, "count": 0},
                    "token_answer": {"mean": 0.0, "count": 0},
                    "token_reasoning": {"mean": 0.0, "count": 0}
                }
            }
        }
        
        # Mock results data structure
        data = {
            "results": [
                {
                    "id": "Q01",
                    "query": "Test query",
                    "category": "general",
                    "baselines": {
                        "B0": {
                            "status": "success",
                            "latency_sec": 5.432,
                            "eval_metrics": {
                                "answer_relevance": 0.8,
                                "semantic_accuracy": 0.75
                            }
                        },
                        "B1": {
                            "status": "success",
                            "latency_sec": 12.345,
                            "eval_metrics": {
                                "retrieval_recall": 0.9,
                                "context_precision": 0.85,
                                "faithfulness": 0.95,
                                "answer_relevance": 0.9,
                                "citation_fidelity": 1.0,
                                "semantic_accuracy": 0.88
                            }
                        }
                    }
                }
            ]
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_csv = Path(tmpdir) / "summary.csv"
            details_csv = Path(tmpdir) / "details.csv"
            
            # Export wide summary
            export_wide_csv(stats, summary_csv)
            self.assertTrue(summary_csv.exists())
            
            with open(summary_csv, "r", encoding="utf-8") as f:
                reader = list(csv.reader(f))
                
            self.assertEqual(len(reader), 3) # Header + 2 baselines
            self.assertEqual(reader[0][:12], [
                "Baseline", "Success Rate", "Recall", "Precision", 
                "Faithfulness", "Relevance", "Citations", "Semantic Accuracy", "Latency (sec)",
                "Token Output", "Token Answer", "Token Reasoning"
            ])
            # B0 check (N/A for context metrics and token metrics)
            self.assertEqual(reader[1][:12], ["B0", "100.0%", "N/A", "N/A", "N/A", "0.8000", "N/A", "0.7500", "5.43", "N/A", "N/A", "N/A"])
            # B1 check
            self.assertEqual(reader[2][:12], ["B1", "50.0%", "0.9000", "0.8500", "0.9500", "0.9000", "1.0000", "0.8800", "12.35", "N/A", "N/A", "N/A"])
            
            # Export detailed
            export_detailed_csv(data, stats, details_csv)
            self.assertTrue(details_csv.exists())
            
            with open(details_csv, "r", encoding="utf-8") as f:
                reader_det = list(csv.reader(f))
                
            self.assertEqual(len(reader_det), 3) # Header + Q01_B0 + Q01_B1
            self.assertEqual(reader_det[0][:25], [
                "query_id", "category", "baseline", "status", "latency_sec",
                "is_answerable",
                "retrieval_recall", "context_precision", "faithfulness",
                "answer_relevance", "citation_fidelity", "semantic_accuracy",
                "ar_sa_f1",
                "token_output", "token_answer", "token_reasoning",
                "seed_chunks_from_lexical_dense", "seed_paper_id_list",
                "graph_neighbor_paper_id_list", "candidate_count_before_reranker",
                "candidate_count_after_reranker", "final_context_paper_id_list",
                "final_context_token_count", "whether_graph_neighbor_chunk_survived_into_final_context",
                "answer_token_count"
            ]),
            self.assertEqual(reader_det[1][:25], ["Q01", "general", "B0", "success", "5.432", "True", "", "", "", "0.8", "", "0.75", "0.7742", "", "", "", "", "", "", "", "", "", "", "", ""]),
            self.assertEqual(reader_det[2][:25], ["Q01", "general", "B1", "success", "12.345", "True", "0.9", "0.85", "0.95", "0.9", "1.0", "0.88", "0.8899", "", "", "", "", "", "", "", "", "", "", "", ""]),
            self.assertEqual(reader_det[1][:23], ["Q01", "general", "B0", "success", "5.432", "", "", "", "0.8", "", "0.75", "", "", "", "", "", "", "", "", "", "", "", ""]),
            self.assertEqual(reader_det[2][:23], ["Q01", "general", "B1", "success", "12.345", "0.9", "0.85", "0.95", "0.9", "1.0", "0.88", "", "", "", "", "", "", "", "", "", "", "", ""])

    @patch("subprocess.check_output")
    @patch("subprocess.run")

    @patch("benchmarks.rag.run_pipeline.run_command_with_progress")
    def test_run_pipeline_orchestration(self, mock_run_progress, mock_run, mock_check_output):
        """Verifies that run_pipeline.py runs all three stages in sequence with correct arguments."""
        import sys
        import tempfile
        from unittest.mock import patch
        from benchmarks.rag.run_pipeline import main as pipeline_main
        
        mock_run_progress.return_value = 1.0
        
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_yaml = Path(tmpdir) / "test_smoke.yaml"
            test_cases = [{"id": f"Q{i:02d}", "query": f"Query {i}", "golden_answer": "ans", "expected_papers": []} for i in range(1, 11)]
            with open(dataset_yaml, "w", encoding="utf-8") as f:
                yaml.dump(test_cases, f)
                
            test_args = [
                "run_pipeline.py",
                "--dataset", str(dataset_yaml),
                "--baselines", "B0,B1",
                "--concurrency", "5",
                "--rpm", "100",
                "--retries", "5",
                "--limit", "10",
                "--clear-checkpoint",
                "--no-unique-dir",
                "--output-dir", "test_reports"
            ]
            
            with patch.object(sys, "argv", test_args):
                pipeline_main()
                
        self.assertEqual(mock_run_progress.call_count, 3)
        self.assertEqual(mock_run.call_count, 1)
        
        # Verify call 1: run_custom_retrieve.py
        call1_args = mock_run_progress.call_args_list[0][0][0]
        self.assertTrue("run_retrieve.py" in call1_args[1] or "run_custom_retrieve.py" in call1_args[1])
        self.assertIn("--dataset", call1_args)
        self.assertIn(str(dataset_yaml.resolve()), call1_args)
        self.assertIn("--output", call1_args)
        self.assertIn("--baselines", call1_args)
        self.assertIn("B0,B1", call1_args)
        
        # Verify call 2: run_benchmarks.py
        call2_args = mock_run_progress.call_args_list[1][0][0]
        self.assertIn("run_benchmarks.py", call2_args[1])
        self.assertIn("--dataset", call2_args)
        self.assertIn(str(dataset_yaml.resolve()), call2_args)
        self.assertIn("--output", call2_args)
        self.assertIn("--baselines", call2_args)
        self.assertIn("B0,B1", call2_args)
        
        # Verify call 3: run_evaluator.py
        call3_args = mock_run_progress.call_args_list[2][0][0]
        self.assertIn("run_evaluator.py", call3_args[1])
        self.assertIn("--concurrency", call3_args)
        self.assertIn("5", call3_args)
        self.assertIn("--rpm", call3_args)
        self.assertIn("100", call3_args)
        self.assertIn("--retries", call3_args)
        self.assertIn("5", call3_args)
        self.assertIn("--limit", call3_args)
        self.assertIn("10", call3_args)
        self.assertIn("--clear-checkpoint", call3_args)
        
        # Verify call 4: parse_metrics.py
        call4_args = mock_run.call_args_list[0][0][0]
        self.assertIn("parse_metrics.py", call4_args[1])
        self.assertTrue(any("test_reports" in arg for arg in call4_args))

    @patch("benchmarks.rag.run_retrieve.container.get_rag_service")
    def test_run_retrieve_saves_copy_in_run_retrive_directory(self, mock_get_rag_service):
        """Tests that run_retrieve saves results both in the specified output file and in a run_retrive_* directory."""
        import tempfile
        import sys
        from benchmarks.rag.run_retrieve import main

        # Setup mock RAG service
        mock_rag = MagicMock(spec=RAGService)
        mock_rag.emb_engine = MagicMock()
        mock_rag.vector_repo = MagicMock()
        mock_rag.graph_repo = MagicMock()
        mock_rag._reranker = MagicMock()
        mock_rag.llm_engine = MagicMock()
        mock_rag.build_context.return_value = ("", "")
        mock_rag.trim_context.return_value = ("", "", [])

        mock_rag.retrieve_relevant_chunks.return_value = []
        mock_get_rag_service.return_value = mock_rag

        with tempfile.TemporaryDirectory() as tmpdir:
            reports_dir = Path(tmpdir) / "reports"
            reports_dir.mkdir()
            output_yaml = reports_dir / "retrieved_contexts.yaml"

            # Create a test dataset
            dataset_yaml = Path(tmpdir) / "test_dataset.yaml"
            test_dataset = [
                {
                    "id": "Q01",
                    "category": "general",
                    "query": "Is gravity real?",
                    "golden_answer": "Yes",
                    "expected_papers": ["paper1"]
                }
            ]
            with open(dataset_yaml, "w", encoding="utf-8") as f:
                yaml.dump(test_dataset, f)

            test_args = [
                "run_retrieve.py",
                "--dataset", str(dataset_yaml),
                "--output", str(output_yaml),
                "--baselines", "B1"
            ]

            with patch.object(sys, "argv", test_args):
                main()

            # Verify output file exists
            self.assertTrue(output_yaml.exists())

            # Verify that a run_retrive_* directory was created inside reports_dir
            run_dirs = list(reports_dir.glob("run_retrive_*"))
            self.assertEqual(len(run_dirs), 1, "Should have created exactly one run_retrive_* directory")

            run_dir = run_dirs[0]
            copy_yaml = run_dir / "retrieved_contexts.yaml"
            self.assertTrue(copy_yaml.exists(), f"Copy of output file should exist at: {copy_yaml}")

            # Verify contents of the copy are correct
            with open(copy_yaml, "r", encoding="utf-8") as f:
                saved_data = yaml.safe_load(f)
            self.assertEqual(len(saved_data), 1)
            self.assertEqual(saved_data[0]["query"], "Is gravity real?")




