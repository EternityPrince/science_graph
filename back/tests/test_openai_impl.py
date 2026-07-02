import unittest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel

from src.config import config
from src.llm_engine.openai_impl import OpenAILLMEngine, AsyncRateLimiter


class MockSchema(BaseModel):
    name: str
    age: int


class TestAsyncRateLimiter(unittest.IsolatedAsyncioTestCase):
    async def test_wait_no_delay(self):
        limiter = AsyncRateLimiter(delay=0.0)
        await limiter.wait()

    async def test_wait_with_delay(self):
        limiter = AsyncRateLimiter(delay=0.1)
        import time
        start = time.monotonic()
        await limiter.wait()
        await limiter.wait()
        elapsed = time.monotonic() - start
        self.assertGreaterEqual(elapsed, 0.09)


class TestOpenAILLMEngine(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.orig_data = config.data
        config.data = {
            "llm": {
                "provider": "openai",
                "api_key": "fake-key",
                "model_path": "gpt-4",
                "cloud": {
                    "provider": "openai",
                    "base_url": "https://api.openai.com/v1",
                    "model_name": "gpt-4"
                },
                "max_tokens": 100,
                "temp": 0.7,
                "request_delay": 0.0,
                "retry_backoff": 0.01,
            }
        }

    def tearDown(self):
        config.data = self.orig_data

    @patch("openai.OpenAI")
    @patch("src.llm_engine.openai_impl.con")
    def test_init_missing_key(self, mock_con, mock_openai):
        config.data["llm"]["api_key"] = ""
        with self.assertRaises(ValueError):
            OpenAILLMEngine()
        mock_con.error.assert_called_once()

    @patch("openai.OpenAI")
    def test_init_success(self, mock_openai):
        mock_tiktoken = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tiktoken.encoding_for_model.return_value = mock_tokenizer
        
        with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
            engine = OpenAILLMEngine()
            self.assertEqual(engine.model_name, "gpt-4")
            self.assertIsNotNone(engine.client)
            self.assertEqual(engine.tokenizer, mock_tokenizer)

    @patch("openai.OpenAI")
    def test_init_tokenizer_failure(self, mock_openai):
        mock_tiktoken = MagicMock()
        mock_tiktoken.encoding_for_model.side_effect = Exception("failed to load tokenizer")
        
        with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
            engine = OpenAILLMEngine()
            self.assertIsNone(engine.tokenizer)

    @patch("openai.OpenAI")
    def test_truncate_to_context_no_tokenizer(self, mock_openai):
        engine = OpenAILLMEngine()
        engine.tokenizer = None
        text = "a" * 100
        truncated = engine._truncate_to_context(text, 10)
        self.assertEqual(truncated, "a" * 40)

    @patch("openai.OpenAI")
    def test_truncate_to_context_with_tokenizer(self, mock_openai):
        engine = OpenAILLMEngine()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(100))
        mock_tokenizer.decode.side_effect = lambda tokens: f"decoded-{len(tokens)}"
        engine.tokenizer = mock_tokenizer

        truncated = engine._truncate_to_context("hello", 10)
        self.assertEqual(truncated, "decoded-10")
        mock_tokenizer.encode.assert_called_once_with("hello")
        mock_tokenizer.decode.assert_called_once_with(list(range(10)))

        mock_tokenizer.encode.return_value = list(range(5))
        truncated = engine._truncate_to_context("hello", 10)
        self.assertEqual(truncated, "hello")

    @patch("openai.OpenAI")
    def test_generate_response_sync(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "  <think>reasoning</think>  Clean Response  "
        mock_client.chat.completions.create.return_value = mock_completion

        engine = OpenAILLMEngine()
        res = engine.generate_response("hello")
        self.assertEqual(res, "Clean Response")
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=config.llm_max_tokens,
            temperature=config.llm_temp
        )

    @patch("openai.OpenAI")
    async def test_generate_response_async_success(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Clean Async Response"
        mock_client.chat.completions.create.return_value = mock_completion

        engine = OpenAILLMEngine()
        res = await engine.generate_response_async("hello", max_tokens=100, temp=0.7)
        self.assertEqual(res, "Clean Async Response")
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=100,
            temperature=0.7
        )

    @patch("openai.OpenAI")
    @patch("asyncio.sleep")
    async def test_generate_response_async_retries(self, mock_sleep, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Success After Retry"
        mock_client.chat.completions.create.side_effect = [
            Exception("Rate limited or server error"),
            Exception("Rate limited or server error"),
            mock_completion
        ]

        engine = OpenAILLMEngine()
        res = await engine.generate_response_async("hello")
        self.assertEqual(res, "Success After Retry")
        self.assertEqual(mock_client.chat.completions.create.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("openai.OpenAI")
    async def test_generate_response_async_failure_max_retries(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Persistent API Error")

        engine = OpenAILLMEngine()
        with self.assertRaises(Exception):
            await engine.generate_response_async("hello")

    @patch("openai.OpenAI")
    def test_generate_json_sync_structured_outputs_success(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = '{"name": "Alice", "age": 30}'
        mock_client.chat.completions.create.return_value = mock_completion

        engine = OpenAILLMEngine()
        res = engine.generate_json("get alice info", schema_class=MockSchema, temp=None)
        self.assertEqual(res, '{"name": "Alice", "age": 30}')
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a strict JSON extractor. Output ONLY valid JSON matching this schema: {MockSchema.model_json_schema()}"
                },
                {"role": "user", "content": "get alice info"}
            ],
            max_tokens=config.llm_max_tokens,
            temperature=config.llm_temp,
            response_format=MockSchema
        )

    @patch("openai.OpenAI")
    def test_generate_json_sync_fallbacks(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = '{"name": "Fallback", "age": 40}'
        mock_client.chat.completions.create.side_effect = [
            Exception("Structured outputs not supported"),
            Exception("JSON mode not supported"),
            mock_completion
        ]

        engine = OpenAILLMEngine()
        res = engine.generate_json("get fallback info", schema_class=MockSchema)
        self.assertEqual(res, '{"name": "Fallback", "age": 40}')
        self.assertEqual(mock_client.chat.completions.create.call_count, 3)

    @patch("openai.OpenAI")
    async def test_generate_json_async_success(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = '{"name": "Bob", "age": 25}'
        mock_client.chat.completions.create.return_value = mock_completion

        engine = OpenAILLMEngine()
        res = await engine.generate_json_async("get bob info", schema_class=MockSchema)
        self.assertEqual(res, '{"name": "Bob", "age": 25}')
        mock_client.chat.completions.create.assert_called_once()

    @patch("openai.OpenAI")
    @patch("subprocess.Popen")
    @patch("time.sleep")
    def test_openai_engine_starts_local_server(self, mock_sleep, mock_popen, mock_openai):
        # Setup mocks
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Make the first list() fail to simulate server offline, and second succeed
        mock_client.models.list.side_effect = [Exception("offline"), MagicMock()]
        
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc
        
        # Temporarily remove PYTEST_CURRENT_TEST to trigger lifecycle logic
        import os
        orig_test = os.environ.get("PYTEST_CURRENT_TEST")
        if "PYTEST_CURRENT_TEST" in os.environ:
            del os.environ["PYTEST_CURRENT_TEST"]
            
        try:
            # We must set provider to openai-compatible, base_url to localhost
            with patch.dict(config.data, {"llm": {"provider": "openai-compatible", "local": {"base_url": "http://localhost:8080/v1", "model_path": "optiq-model"}}}):
                engine = OpenAILLMEngine(base_url="http://localhost:8080/v1")
                
                # Check subprocess was spawned
                mock_popen.assert_called_once()
                self.assertIsNotNone(engine.server_process)
        finally:
            # Restore environment variable
            if orig_test:
                os.environ["PYTEST_CURRENT_TEST"] = orig_test
