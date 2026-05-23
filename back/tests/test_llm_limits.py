import unittest
from unittest.mock import MagicMock, patch
from src.config import config
from src.llm_engine import BaseLLMEngine, OpenAILLMEngine, MlxLLMEngine

class TestLLMLimits(unittest.TestCase):
    def test_truncate_to_context_no_tokenizer(self):
        # BaseLLMEngine without tokenizer should fall back to character slicing: max_input_tokens * 4
        engine = BaseLLMEngine()
        text = "abcdefghijklmnopqrstuvwxyz"
        # with limit of 3 tokens, should slice text to 3 * 4 = 12 characters
        truncated = engine._truncate_to_context(text, 3)
        self.assertEqual(truncated, "abcdefghijkl")

    def test_truncate_to_context_with_tokenizer(self):
        # BaseLLMEngine with mock tokenizer
        engine = BaseLLMEngine()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.decode.side_effect = lambda tokens: f"decoded-{len(tokens)}"
        engine.tokenizer = mock_tokenizer

        # If tokens length <= limit, returns full text
        self.assertEqual(engine._truncate_to_context("full text", 10), "full text")

        # If tokens length > limit, encodes, slices and decodes
        truncated = engine._truncate_to_context("long text", 3)
        self.assertEqual(truncated, "decoded-3")
        mock_tokenizer.encode.assert_called_with("long text")
        mock_tokenizer.decode.assert_called_with([1, 2, 3])

    @patch("openai.OpenAI")
    def test_openai_priority_resolution(self, mock_openai):
        # Mock OpenAI client response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "response"
        mock_client.chat.completions.create.return_value = mock_completion

        # Force config values
        original_data = config.data
        config.data = {
            "llm": {
                "provider": "openai",
                "api_key": "fake-key",
                "model_path": "gpt-4",
                "max_tokens": 999,
                "temp": 0.1,
                "extraction_output_limit": 111,
                "clustering_output_limit": 222,
                "synthesis_output_limit": 333,
            }
        }

        try:
            engine = OpenAILLMEngine()

            # Priority 1: passed_argument
            engine.generate_response("hello", max_tokens=500)
            mock_client.chat.completions.create.assert_called_with(
                model="gpt-4",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=500,
                temperature=0.1
            )

            # Priority 2: task_specific_config for extraction
            engine.generate_response("hello", task="extraction")
            mock_client.chat.completions.create.assert_called_with(
                model="gpt-4",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=111,
                temperature=0.1
            )

            # Priority 2: task_specific_config for clustering
            engine.generate_response("hello", task="clustering")
            mock_client.chat.completions.create.assert_called_with(
                model="gpt-4",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=222,
                temperature=0.1
            )

            # Priority 2: task_specific_config for synthesis
            engine.generate_response("hello", task="synthesis")
            mock_client.chat.completions.create.assert_called_with(
                model="gpt-4",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=333,
                temperature=0.1
            )

            # Priority 3: global_config
            engine.generate_response("hello")
            mock_client.chat.completions.create.assert_called_with(
                model="gpt-4",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=999,
                temperature=0.1
            )

            # Priority 1 over 2: passed_argument overrides task
            engine.generate_response("hello", max_tokens=777, task="synthesis")
            mock_client.chat.completions.create.assert_called_with(
                model="gpt-4",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=777,
                temperature=0.1
            )
        finally:
            config.data = original_data

    @patch("mlx_lm.load")
    @patch("mlx_lm.generate")
    @patch("mlx_lm.sample_utils.make_sampler")
    @patch("os.path.isdir")
    def test_mlx_priority_resolution(self, mock_isdir, mock_make_sampler, mock_generate, mock_load):
        mock_isdir.return_value = True
        mock_tokenizer = MagicMock()
        del mock_tokenizer.apply_chat_template
        mock_load.return_value = (MagicMock(), mock_tokenizer)
        mock_generate.return_value = "response"

        # Force config values
        original_data = config.data
        config.data = {
            "llm": {
                "provider": "mlx",
                "model_path": "/fake/model/path",
                "max_tokens": 999,
                "temp": 0.1,
                "extraction_output_limit": 111,
                "clustering_output_limit": 222,
                "synthesis_output_limit": 333,
            }
        }

        try:
            engine = MlxLLMEngine()

            # Priority 1: passed_argument
            engine.generate_response("hello", max_tokens=500)
            mock_generate.assert_called_with(
                model=engine.model,
                tokenizer=engine.tokenizer,
                prompt="hello",
                max_tokens=500,
                sampler=mock_make_sampler(),
                verbose=False
            )

            # Priority 2: task_specific_config for extraction
            engine.generate_response("hello", task="extraction")
            mock_generate.assert_called_with(
                model=engine.model,
                tokenizer=engine.tokenizer,
                prompt="hello",
                max_tokens=111,
                sampler=mock_make_sampler(),
                verbose=False
            )

            # Priority 3: global_config
            engine.generate_response("hello")
            mock_generate.assert_called_with(
                model=engine.model,
                tokenizer=engine.tokenizer,
                prompt="hello",
                max_tokens=999,
                sampler=mock_make_sampler(),
                verbose=False
            )
        finally:
            config.data = original_data
