import unittest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel

from src.config import config
from src.llm_engine.gguf_impl import GgufLLMEngine


class MockSchema(BaseModel):
    name: str
    age: int


class TestGgufImpl(unittest.TestCase):
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
