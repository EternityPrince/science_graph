import unittest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel

from src.config import config
from src.llm_engine.gguf_impl import GgufLLMEngine


class MockSchema(BaseModel):
    name: str
    age: int


class TestGgufImpl(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.orig_data = config.data
        config.data = {
            "llm": {
                "provider": "gguf",
                "local": {
                    "model_path": "/fake/model/path.gguf",
                },
                "gguf": {
                    "n_gpu_layers": -1,
                    "n_ctx": 2048,
                },
                "max_tokens": 100,
                "temp": 0.7,
                "extraction_output_limit": 50,
                "clustering_output_limit": 60,
                "synthesis_output_limit": 70,
            }
        }

    def tearDown(self):
        config.data = self.orig_data

    @patch("os.path.exists")
    def test_init_missing_file(self, mock_exists):
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            GgufLLMEngine(model_path="/nonexistent.gguf")

    @patch("os.path.exists")
    def test_init_success(self, mock_exists):
        mock_exists.return_value = True
        engine = GgufLLMEngine(model_path="/fake/model/path.gguf")
        self.assertEqual(engine.model_path, "/fake/model/path.gguf")
        self.assertIsNone(engine.model)

    @patch("os.path.exists")
    @patch("llama_cpp.Llama", create=True)
    def test_generate_response_limits_and_templates(self, mock_llama, mock_exists):
        mock_exists.return_value = True
        
        mock_model_instance = MagicMock()
        mock_llama.return_value = mock_model_instance
        
        # Mock completion results
        mock_model_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "<think>thought</think> clean_chat"}}]
        }
        mock_model_instance.return_value = {
            "choices": [{"text": "<think>thought</think> clean_raw"}]
        }
        
        engine = GgufLLMEngine(model_path="/fake/model/path.gguf")
        
        # Test chat completion branch (not formatted)
        res = engine.generate_response("prompt", task="extraction")
        self.assertEqual(res, "clean_chat")
        mock_model_instance.create_chat_completion.assert_called_once()
        args, kwargs = mock_model_instance.create_chat_completion.call_args
        self.assertEqual(kwargs["max_tokens"], 50)
        self.assertEqual(kwargs["temperature"], 0.7)
        
        # Test synthesis task limit
        mock_model_instance.create_chat_completion.reset_mock()
        engine.generate_response("prompt", task="synthesis")
        args, kwargs = mock_model_instance.create_chat_completion.call_args
        self.assertEqual(kwargs["max_tokens"], 70)
        
        # Reset and test formatted branch (uses raw call)
        mock_model_instance.create_chat_completion.reset_mock()
        res = engine.generate_response("<|im_start|>system\nprompt<|im_end|>", task="clustering")
        self.assertEqual(res, "clean_raw")
        mock_model_instance.assert_called_once()
        args, kwargs = mock_model_instance.call_args
        self.assertEqual(kwargs["max_tokens"], 60)
        self.assertEqual(kwargs["temperature"], 0.7)

    @patch("os.path.exists")
    @patch("llama_cpp.Llama", create=True)
    def test_generate_json(self, mock_llama, mock_exists):
        mock_exists.return_value = True
        
        mock_model_instance = MagicMock()
        mock_llama.return_value = mock_model_instance
        
        # Mock structured outputs
        mock_model_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": '{"name": "test", "age": 10}'}}]
        }
        
        engine = GgufLLMEngine(model_path="/fake/model/path.gguf")
        res = engine.generate_json("get json", MockSchema)
        
        self.assertEqual(res, '{"name": "test", "age": 10}')
        mock_model_instance.create_chat_completion.assert_called_once()
        args, kwargs = mock_model_instance.create_chat_completion.call_args
        self.assertIn("response_format", kwargs)
        self.assertEqual(kwargs["response_format"]["type"], "json_object")
        
        # Test count_tokens
        mock_model_instance.tokenize.return_value = [1, 2, 3, 4]
        self.assertEqual(engine.count_tokens("hello"), 4)
        mock_model_instance.tokenize.assert_called_once_with(b"hello", special=True)

    @patch("os.path.exists")
    def test_unload_model(self, mock_exists):
        mock_exists.return_value = True
        engine = GgufLLMEngine(model_path="/fake/model/path.gguf")
        
        # Scenario 1: model is None
        engine.unload_model()
        self.assertIsNone(engine.model)
        
        # Scenario 2: model exists and has close()
        mock_model = MagicMock()
        engine.model = mock_model
        engine.unload_model()
        mock_model.close.assert_called_once()
        self.assertIsNone(engine.model)
        
        # Scenario 3: model exists and close() raises exception
        mock_model = MagicMock()
        mock_model.close.side_effect = Exception("failed to close")
        engine.model = mock_model
        engine.unload_model()  # Should not raise exception
        mock_model.close.assert_called_once()
        self.assertIsNone(engine.model)

    @patch("os.path.exists")
    def test_ensure_model_loaded_import_error(self, mock_exists):
        mock_exists.return_value = True
        engine = GgufLLMEngine(model_path="/fake/model/path.gguf")
        
        with patch.dict("sys.modules", {"llama_cpp": None}):
            with self.assertRaises(ImportError):
                engine._ensure_model_loaded()

    @patch("os.path.exists")
    @patch("llama_cpp.Llama", create=True)
    def test_count_tokens_edge_cases(self, mock_llama, mock_exists):
        mock_exists.return_value = True
        mock_model_instance = MagicMock()
        mock_llama.return_value = mock_model_instance
        engine = GgufLLMEngine(model_path="/fake/model/path.gguf")
        
        # Empty text
        self.assertEqual(engine.count_tokens(""), 0)
        self.assertEqual(engine.count_tokens(None), 0)
        
        # Tokenize exception fallback
        mock_model_instance.tokenize.side_effect = Exception("tokenize error")
        self.assertEqual(engine.count_tokens("hello world"), 2) # len("hello world") is 11 // 4 = 2

    @patch("os.path.exists")
    @patch("llama_cpp.Llama", create=True)
    def test_generate_response_chat_completion_fallback(self, mock_llama, mock_exists):
        mock_exists.return_value = True
        mock_model_instance = MagicMock()
        mock_llama.return_value = mock_model_instance
        
        # create_chat_completion fails, model(...) raw completion succeeds
        mock_model_instance.create_chat_completion.side_effect = Exception("chat completion fails")
        mock_model_instance.return_value = {
            "choices": [{"text": "fallback response"}]
        }
        
        engine = GgufLLMEngine(model_path="/fake/model/path.gguf")
        res = engine.generate_response("prompt")
        self.assertEqual(res, "fallback response")
        mock_model_instance.create_chat_completion.assert_called_once()
        mock_model_instance.assert_called_once_with(
            "prompt",
            max_tokens=config.llm_max_tokens,
            temperature=config.llm_temp
        )

    @patch("os.path.exists")
    @patch("llama_cpp.Llama", create=True)
    def test_generate_json_fallbacks_and_formatted(self, mock_llama, mock_exists):
        mock_exists.return_value = True
        mock_model_instance = MagicMock()
        mock_llama.return_value = mock_model_instance
        engine = GgufLLMEngine(model_path="/fake/model/path.gguf")
        
        # 1. Formatted prompt with json format (uses raw completion self.model)
        mock_model_instance.return_value = {
            "choices": [{"text": '{"name": "inst", "age": 20}'}]
        }
        res = engine.generate_json("[INST] get json [/INST]", MockSchema)
        self.assertEqual(res, '{"name": "inst", "age": 20}')
        
        # 2. create_chat_completion raising exception -> falls back to unconstrained chat completion
        mock_model_instance.create_chat_completion.side_effect = [
            Exception("json_format error"), # first call fails
            {"choices": [{"message": {"content": '{"name": "fallback", "age": 30}'}}]} # fallback succeeds
        ]
        res = engine.generate_json("get json", MockSchema)
        self.assertEqual(res, '{"name": "fallback", "age": 30}')
        
        # 3. create_chat_completion raising exception with formatted prompt -> falls back to unconstrained raw completion
        mock_model_instance.create_chat_completion.side_effect = None
        mock_model_instance.side_effect = [
            Exception("json_format error"), # first call fails
            {"choices": [{"text": '{"name": "raw_fallback", "age": 40}'}]} # fallback succeeds
        ]
        res = engine.generate_json("[INST] get json [/INST]", MockSchema)
        self.assertEqual(res, '{"name": "raw_fallback", "age": 40}')

    @patch("os.path.exists")
    @patch("llama_cpp.Llama", create=True)
    async def test_async_generation(self, mock_llama, mock_exists):
        mock_exists.return_value = True
        mock_model_instance = MagicMock()
        mock_llama.return_value = mock_model_instance
        
        mock_model_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "async response"}}]
        }
        
        engine = GgufLLMEngine(model_path="/fake/model/path.gguf")
        
        res = await engine.generate_response_async("async prompt")
        self.assertEqual(res, "async response")
        
        mock_model_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": '{"name": "async_json", "age": 50}'}}]
        }
        res_json = await engine.generate_json_async("async json prompt", MockSchema)
        self.assertEqual(res_json, '{"name": "async_json", "age": 50}')
