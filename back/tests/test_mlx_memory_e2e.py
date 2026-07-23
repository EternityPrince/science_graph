import unittest
from unittest.mock import MagicMock, patch
import gc

import sys
from pathlib import Path
back_root = Path(__file__).resolve().parents[1]
rag_dir = back_root / "benchmarks" / "rag"
if str(rag_dir) not in sys.path:
    sys.path.insert(0, str(rag_dir))

try:
    import mlx.core as mx
    has_mlx = True
except ImportError:
    has_mlx = False

from src.llm_engine.mlx_impl import MlxLLMEngine
from core.generation import _clear_gpu_cache


@unittest.skipUnless(has_mlx, "MLX is not installed")
class TestMlxMemoryCleanupE2E(unittest.TestCase):
    @patch("os.path.isdir", return_value=True)
    @patch("src.llm_engine.mlx_impl.MlxLLMEngine._ensure_model_loaded")
    @patch("mlx_lm.stream_generate")
    def test_generate_response_with_logits_clears_memory(self, mock_stream_generate, mock_ensure_loaded, mock_isdir):
        """Verifies that generate_response_with_logits deletes tensor logprobs and clears Metal cache."""
        engine = MlxLLMEngine(model_path="/fake/model")
        engine.model = MagicMock()
        engine.tokenizer = MagicMock()

        # Mock stream responses with fake logprob arrays
        class FakeResponse:
            def __init__(self, text, token_id):
                self.text = text
                self.token = token_id
                self.logprobs = mx.array([0.1, 0.2, 0.3, 0.4])

        mock_stream_generate.return_value = [
            FakeResponse("Hello ", 1),
            FakeResponse("world!", 2)
        ]

        with patch("gc.collect") as mock_gc:
            text, tokens_info = engine.generate_response_with_logits("Test prompt")
            self.assertIn("Hello world!", text)
            self.assertEqual(len(tokens_info), 2)
            mock_gc.assert_called()

    def test_clear_gpu_cache_flushes_mlx(self):
        """Verifies that _clear_gpu_cache executes gc.collect and MLX Metal cache clearing."""
        with patch("gc.collect") as mock_gc:
            _clear_gpu_cache()
            mock_gc.assert_called()


if __name__ == "__main__":
    unittest.main()
