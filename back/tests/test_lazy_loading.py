import pytest
pytestmark = pytest.mark.llm

import unittest
from unittest.mock import MagicMock, patch

from src.vector_search import EmbeddingEngine
from src.llm_engine import MlxLLMEngine

class TestLazyLoading(unittest.TestCase):
    @patch("sentence_transformers.SentenceTransformer")
    @patch.object(EmbeddingEngine, "_get_device", return_value="cpu")
    def test_embedding_engine_lazy_loading(self, mock_get_device, mock_sentence_transformer):
        
        # 1. Instantiate the engine - model should NOT be loaded
        engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
        self.assertIsNone(engine.model)
        mock_sentence_transformer.assert_not_called()

        # Mock encode to return numpy array (with tolist method)
        mock_model_instance = MagicMock()
        mock_sentence_transformer.return_value = mock_model_instance
        mock_emb = MagicMock()
        mock_emb.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model_instance.encode.return_value = mock_emb

        # 2. Call get_embedding - model should be loaded on demand
        emb = engine.get_embedding("hello")
        self.assertEqual(emb, [0.1, 0.2, 0.3])
        mock_sentence_transformer.assert_called_once()
        self.assertIsNotNone(engine.model)

        # 3. Call get_embedding again - model should NOT be re-loaded
        emb2 = engine.get_embedding("world")
        self.assertEqual(emb2, [0.1, 0.2, 0.3])
        mock_sentence_transformer.assert_called_once() # still only called once

    @patch("mlx_lm.load")
    @patch("mlx_lm.generate")
    @patch("os.path.isdir")
    def test_mlx_llm_engine_lazy_loading(self, mock_isdir, mock_generate, mock_load):
        mock_isdir.return_value = True
        mock_tokenizer = MagicMock()
        mock_load.return_value = (MagicMock(), mock_tokenizer)
        mock_generate.return_value = "response text"

        # 1. Instantiate the engine - model and tokenizer should NOT be loaded
        engine = MlxLLMEngine(model_path="/fake/model/path")
        self.assertIsNone(engine.model)
        self.assertIsNone(engine.tokenizer)
        mock_load.assert_not_called()

        # 2. Call count_tokens - model should be loaded on demand
        tokens_count = engine.count_tokens("hello")
        mock_load.assert_called_once()
        self.assertIsNotNone(engine.model)
        self.assertIsNotNone(engine.tokenizer)

        # 3. Call generate_response - model should NOT be re-loaded
        response = engine.generate_response("prompt")
        self.assertEqual(response, "response text")
        mock_load.assert_called_once() # still only called once
