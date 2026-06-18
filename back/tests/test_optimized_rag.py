import unittest
from unittest.mock import MagicMock, patch
import gc
import torch

from src.vector_search import EmbeddingEngine
from src.llm_engine.mlx_impl import MlxLLMEngine
from src.models import Chunk


class TestOptimizedRAG(unittest.TestCase):
    def test_embedding_engine_cache(self):
        engine = EmbeddingEngine(model_name="sentence-transformers/all-MiniLM-L6-v2")
        mock_model = MagicMock()
        # Mock model.encode to return a mock numpy array with a tolist method
        mock_emb = MagicMock()
        mock_emb.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.encode.return_value = mock_emb
        engine.model = mock_model

        # 1. Single query caching
        # First call should call mock_model.encode
        emb1 = engine.get_embedding("test query", is_query=True)
        self.assertEqual(emb1, [0.1, 0.2, 0.3])
        self.assertEqual(mock_model.encode.call_count, 1)

        # Second call should hit the cache and not call encode again
        emb2 = engine.get_embedding("test query", is_query=True)
        self.assertEqual(emb2, [0.1, 0.2, 0.3])
        self.assertEqual(mock_model.encode.call_count, 1)

        # Non-query calls should not be cached
        mock_emb_passage = MagicMock()
        mock_emb_passage.tolist.return_value = [0.4, 0.5, 0.6]
        mock_model.encode.return_value = mock_emb_passage
        emb3 = engine.get_embedding("test passage", is_query=False)
        self.assertEqual(emb3, [0.4, 0.5, 0.6])
        self.assertEqual(mock_model.encode.call_count, 2)

        # 2. Batch query caching
        # Clear cache and reset mock
        engine._query_cache.clear()
        mock_model.reset_mock()
        mock_emb.tolist.return_value = [[0.2]]
        mock_model.encode.return_value = mock_emb

        # Seed the cache with one query
        engine._query_cache["q1"] = [0.1]

        # Call get_embeddings with one cached and one uncached
        res = engine.get_embeddings(["q1", "q2"], is_query=True)
        self.assertEqual(res, [[0.1], [0.2]])
        # Should only encode the uncached "q2"
        mock_model.encode.assert_called_once_with(["q2"], convert_to_numpy=True, show_progress_bar=False)

    def test_embedding_engine_unload(self):
        engine = EmbeddingEngine(model_name="sentence-transformers/all-MiniLM-L6-v2")
        engine.model = MagicMock()
        self.assertIsNotNone(engine.model)

        with patch("torch.mps.empty_cache") as mock_mps_clear:
            engine.unload_model()
            self.assertIsNone(engine.model)
            if torch.backends.mps.is_available():
                mock_mps_clear.assert_called_once()

    @patch("mlx.core.clear_cache")
    def test_mlx_llm_engine_unload(self, mock_mlx_clear):
        # Patch is_dir to pass init path checks
        with patch("os.path.isdir", return_value=True):
            # Pass a dummy path that looks valid
            engine = MlxLLMEngine(model_path="/dummy/path")
            engine.model = MagicMock()
            engine.tokenizer = MagicMock()
            engine._tokenizer_data = MagicMock()

            self.assertIsNotNone(engine.model)
            engine.unload_model()
            self.assertIsNone(engine.model)
            self.assertIsNone(engine.tokenizer)
            self.assertIsNone(engine._tokenizer_data)
            mock_mlx_clear.assert_called_once()

    def test_metric_merging_in_runner(self):
        from benchmarks.rag.run_benchmarks import merge_evaluation_data
        
        existing = {
            "metadata": {
                "baselines_evaluated": ["B1"]
            },
            "results": [
                {
                    "id": "Q01",
                    "baselines": {
                        "B1": {
                            "status": "success",
                            "generated_answer": "Answer 1"
                        }
                    }
                }
            ]
        }

        new_data = {
            "metadata": {
                "baselines_evaluated": ["B2"]
            },
            "results": [
                {
                    "id": "Q01",
                    "baselines": {
                        "B2": {
                            "status": "success",
                            "generated_answer": "Answer 2"
                        }
                    }
                }
            ]
        }

        merged = merge_evaluation_data(existing, new_data)
        self.assertEqual(merged["metadata"]["baselines_evaluated"], ["B1", "B2"])
        self.assertIn("B1", merged["results"][0]["baselines"])
        self.assertIn("B2", merged["results"][0]["baselines"])
        self.assertEqual(merged["results"][0]["baselines"]["B1"]["generated_answer"], "Answer 1")
        self.assertEqual(merged["results"][0]["baselines"]["B2"]["generated_answer"], "Answer 2")


if __name__ == "__main__":
    unittest.main()
