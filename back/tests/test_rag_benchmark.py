import unittest
from unittest.mock import MagicMock, patch
import yaml
from pathlib import Path

from benchmarks.rag.run_benchmarks import get_baseline_config, run_query_on_baseline, BASELINES_INFO
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
        
        # B5: Hybrid + Graph (Lexical + Dense + RRF + Graph)
        b5_cfg = get_baseline_config("B5")
        self.assertTrue(b5_cfg["lexical_search"])
        self.assertTrue(b5_cfg["dense_search"])
        self.assertTrue(b5_cfg["graph_expansion"])
        self.assertFalse(b5_cfg["reranker"])
        
        # B6: Full Pipeline (All components should be True)
        b6_cfg = get_baseline_config("B6")
        self.assertTrue(all(b6_cfg.values()))

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
        ans, retrieved, metrics = run_query_on_baseline(mock_rag, "Test Question B0", "B0", use_cloud=False)
        self.assertEqual(ans, "Mocked Zero-Shot Answer")
        self.assertEqual(retrieved, [])
        mock_llm.generate_response.assert_called_once()
        
        # Run B1 (Pure Lexical)
        mock_retrieve.return_value = [(MagicMock(paper_id="paper1"), 0.9)]
        ans_b1, retrieved_b1, metrics_b1 = run_query_on_baseline(mock_rag, "Test Question B1", "B1", use_cloud=False)
        self.assertEqual(ans_b1, "Mocked RAG Answer")
        self.assertEqual(retrieved_b1, ["paper1"])
        
        # Verify that after running baselines, the config is restored back to original
        self.assertEqual(config.rag_components, original_components)
        self.assertEqual(config.hyde_enabled, original_hyde)

    def test_baselines_descriptions(self):
        """Verifies all B0-B6 baselines have descriptions configured."""
        self.assertEqual(len(BASELINES_INFO), 7)
        for i in range(7):
            key = f"B{i}"
            self.assertIn(key, BASELINES_INFO)
            self.assertIsInstance(BASELINES_INFO[key], str)
            self.assertTrue(len(BASELINES_INFO[key]) > 0)

    def test_merge_evaluation_data(self):
        """Verifies that merge_evaluation_data successfully merges baseline reports."""
        from benchmarks.rag.run_benchmarks import merge_evaluation_data
        
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
        from benchmarks.rag.run_benchmarks import BenchmarkStatsCollector
        
        mock_rag = MagicMock(spec=RAGService)
        mock_rag.emb_engine = MagicMock()
        mock_rag.vector_repo = MagicMock()
        mock_rag.graph_repo = MagicMock()
        mock_rag._reranker = MagicMock()
        mock_rag.llm_engine = MagicMock()
        
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
                "--baselines", "B0"
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
                "--baselines", "B1"
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


