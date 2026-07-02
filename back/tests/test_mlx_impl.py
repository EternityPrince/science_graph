import unittest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel

try:
    import mlx.core as mx
    has_mlx = True
except ImportError:
    has_mlx = False

from src.config import config
from src.llm_engine.mlx_impl import MlxLLMEngine, build_mlx_tokenizer_data, ConstrainedLogitsProcessor


class MockSchema(BaseModel):
    name: str
    age: int


@unittest.skipUnless(has_mlx, "MLX is not available")
class TestMlxImpl(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.orig_data = config.data
        config.data = {
            "llm": {
                "provider": "mlx",
                "local_model_path": "/fake/model/path",
                "max_tokens": 100,
                "temp": 0.7,
                "extraction_output_limit": 50,
                "clustering_output_limit": 60,
                "synthesis_output_limit": 70,
            }
        }

    def tearDown(self):
        config.data = self.orig_data

    @patch("os.path.isdir")
    def test_init_missing_dir(self, mock_isdir):
        mock_isdir.return_value = False
        with self.assertRaises(FileNotFoundError):
            MlxLLMEngine(model_path="/nonexistent")

    @patch("os.path.isdir")
    def test_init_success(self, mock_isdir):
        mock_isdir.value = True
        engine = MlxLLMEngine(model_path="/fake/model/path")
        self.assertEqual(engine.model_path, "/fake/model/path")
        self.assertIsNone(engine.model)
        self.assertIsNone(engine.tokenizer)

    def test_build_mlx_tokenizer_data(self):
        mock_hf_tokenizer = MagicMock()
        mock_hf_tokenizer.__len__.return_value = 5
        mock_hf_tokenizer.all_special_ids = [4]
        mock_hf_tokenizer.eos_token_id = 2
        
        mock_hf_tokenizer.encode.side_effect = lambda text, **kwargs: [10]
        mock_hf_tokenizer.decode.side_effect = lambda tokens: f"decoded-{tokens}"
        
        tokenizer = MagicMock()
        tokenizer._tokenizer = mock_hf_tokenizer
        
        data = build_mlx_tokenizer_data(tokenizer)
        self.assertEqual(data.vocab_size, 5)
        self.assertEqual(data.eos_token_id, [2])
        self.assertEqual(data.decoder([1, 2]), "decoded-[1, 2]")

    def test_constrained_logits_processor(self):
        mock_enforcer = MagicMock()
        mock_allowed_tokens = MagicMock()
        mock_allowed_tokens.allowed_tokens = [1, 3]
        mock_enforcer.get_allowed_tokens.return_value = mock_allowed_tokens
        
        processor = ConstrainedLogitsProcessor(mock_enforcer)
        
        import mlx.core as mx
        mock_tokens = mx.array([0, 10, 11])
        logits = mx.array([0.1, 0.2, 0.3, 0.4, 0.5])
        res = processor(mock_tokens, logits)
        
        mock_enforcer.get_allowed_tokens.assert_called_once_with([10, 11])
        res_list = res.tolist()
        self.assertAlmostEqual(res_list[1], 0.2)
        self.assertAlmostEqual(res_list[3], 0.4)
        self.assertEqual(res_list[0], float('-inf'))
        self.assertEqual(res_list[2], float('-inf'))
        self.assertEqual(res_list[4], float('-inf'))

    def test_constrained_logits_processor_empty_allowed(self):
        mock_enforcer = MagicMock()
        mock_allowed_tokens = MagicMock()
        mock_allowed_tokens.allowed_tokens = []
        mock_enforcer.get_allowed_tokens.return_value = mock_allowed_tokens
        
        processor = ConstrainedLogitsProcessor(mock_enforcer)
        
        import mlx.core as mx
        mock_tokens = mx.array([0, 10])
        logits = mx.array([0.1, 0.2, 0.3])
        res = processor(mock_tokens, logits)
        res_list = res.tolist()
        expected = [0.1, 0.2, 0.3]
        for r, e in zip(res_list, expected):
            self.assertAlmostEqual(r, e)

    @patch("os.path.isdir")
    @patch("mlx_lm.load")
    @patch("mlx_lm.generate")
    def test_generate_response_limits_and_templates(self, mock_generate, mock_load, mock_isdir):
        mock_isdir.return_value = True
        mock_load.return_value = (MagicMock(), MagicMock())
        
        engine = MlxLLMEngine(model_path="/fake/model/path")
        mock_generate.return_value = "<think>thought</think> clean"
        
        res = engine.generate_response("prompt", task="extraction")
        self.assertEqual(res, "clean")
        mock_generate.assert_called_once()
        args, kwargs = mock_generate.call_args
        self.assertEqual(kwargs["max_tokens"], 50)
        
        mock_generate.reset_mock()
        res = engine.generate_response("prompt", task="clustering")
        args, kwargs = mock_generate.call_args
        self.assertEqual(kwargs["max_tokens"], 60)
        
        mock_generate.reset_mock()
        res = engine.generate_response("prompt", task="synthesis")
        args, kwargs = mock_generate.call_args
        self.assertEqual(kwargs["max_tokens"], 70)

        mock_generate.reset_mock()
        res = engine.generate_response("prompt", max_tokens=150)
        args, kwargs = mock_generate.call_args
        self.assertEqual(kwargs["max_tokens"], 150)

    @patch("os.path.isdir")
    @patch("mlx_lm.load")
    @patch("mlx_lm.generate")
    def test_generate_response_chat_template(self, mock_generate, mock_load, mock_isdir):
        mock_isdir.return_value = True
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "templated-prompt"
        mock_load.return_value = (MagicMock(), mock_tokenizer)
        
        engine = MlxLLMEngine(model_path="/fake/model/path")
        mock_generate.return_value = "response"
        
        engine.generate_response("user-prompt")
        mock_tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "user-prompt"}],
            tokenize=False,
            add_generation_prompt=True
        )
        mock_generate.assert_called_once()
        self.assertEqual(mock_generate.call_args[1]["prompt"], "templated-prompt")

        mock_generate.reset_mock()
        mock_tokenizer.apply_chat_template.reset_mock()
        engine.generate_response("[INST] user-prompt [/INST]")
        mock_tokenizer.apply_chat_template.assert_not_called()
        self.assertEqual(mock_generate.call_args[1]["prompt"], "[INST] user-prompt [/INST]")

    @patch("os.path.isdir")
    @patch("mlx_lm.load")
    @patch("mlx_lm.generate")
    @patch("src.llm_engine.mlx_impl.build_mlx_tokenizer_data")
    @patch("lmformatenforcer.TokenEnforcer")
    @patch("lmformatenforcer.JsonSchemaParser")
    def test_generate_json(self, mock_parser, mock_enforcer, mock_build_data, mock_generate, mock_load, mock_isdir):
        mock_isdir.return_value = True
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "templated-prompt"
        mock_load.return_value = (MagicMock(), mock_tokenizer)
        mock_build_data.return_value = MagicMock()
        mock_generate.return_value = '{"name": "test", "age": 1}'
        
        engine = MlxLLMEngine(model_path="/fake/model/path")
        res = engine.generate_json("get json", MockSchema)
        
        self.assertEqual(res, '{"name": "test", "age": 1}')
        mock_build_data.assert_called_once_with(mock_tokenizer)
        mock_parser.assert_called_once_with(MockSchema.model_json_schema())
        mock_enforcer.assert_called_once()
        mock_generate.assert_called_once()
        
        # Test count_tokens method
        mock_tokenizer.encode.return_value = [1, 2, 3]
        self.assertEqual(engine.count_tokens("hello"), 3)

    @patch("os.path.isdir")
    def test_unload_model(self, mock_isdir):
        mock_isdir.return_value = True
        engine = MlxLLMEngine(model_path="/fake/model/path")
        
        # Scenario 1: model is None
        engine.unload_model()
        self.assertIsNone(engine.model)
        
        # Scenario 2: model is loaded, mlx has clear_cache
        mock_model = MagicMock()
        engine.model = mock_model
        engine.tokenizer = MagicMock()
        engine._tokenizer_data = MagicMock()
        
        with patch("mlx.core.clear_cache") as mock_clear_cache:
            engine.unload_model()
            mock_clear_cache.assert_called_once()
            self.assertIsNone(engine.model)
            self.assertIsNone(engine.tokenizer)
            self.assertIsNone(engine._tokenizer_data)
            
        # Scenario 3: mlx has metal.clear_cache instead
        engine.model = mock_model
        with patch("mlx.core") as mock_mx:
            if hasattr(mock_mx, "clear_cache"):
                del mock_mx.clear_cache
            mock_metal_clear_cache = mock_mx.metal.clear_cache
            engine.unload_model()
            mock_metal_clear_cache.assert_called_once()

        # Scenario 4: mlx.core import raises ImportError
        engine.model = mock_model
        with patch.dict("sys.modules", {"mlx.core": None}):
            engine.unload_model()
            self.assertIsNone(engine.model)

    def test_build_mlx_tokenizer_data_edge_cases(self):
        # 1. eos_token_id is None
        mock_hf_tokenizer = MagicMock()
        mock_hf_tokenizer.__len__.return_value = 5
        mock_hf_tokenizer.all_special_ids = [4]
        mock_hf_tokenizer.eos_token_id = None
        mock_hf_tokenizer.encode.side_effect = Exception("encode error")
        
        # fallback encode raises exception, decode throws on some tokens
        def mock_decode(tokens):
            if tokens == [0, 1]:
                raise ValueError("cannot decode")
            return f"decoded-{tokens}"
        mock_hf_tokenizer.decode.side_effect = mock_decode
        
        # Mock token_0 encode fallback
        def mock_encode(text, **kwargs):
            if kwargs.get("add_special_tokens") is False:
                raise ValueError("No special tokens allowed")
            return [0]
        mock_hf_tokenizer.encode.side_effect = mock_encode
        
        tokenizer = MagicMock()
        tokenizer._tokenizer = mock_hf_tokenizer
        
        data = build_mlx_tokenizer_data(tokenizer)
        self.assertEqual(data.eos_token_id, [])
        self.assertEqual(data.vocab_size, 5)

        # 2. eos_token_id is list
        mock_hf_tokenizer.eos_token_id = [1, 2]
        data = build_mlx_tokenizer_data(tokenizer)
        self.assertEqual(data.eos_token_id, [1, 2])

    @patch("os.path.isdir")
    @patch("mlx_lm.load")
    @patch("mlx_lm.generate")
    @patch("src.llm_engine.mlx_impl.build_mlx_tokenizer_data")
    @patch("lmformatenforcer.TokenEnforcer")
    @patch("lmformatenforcer.JsonSchemaParser")
    async def test_async_generation(self, mock_parser, mock_enforcer, mock_build_data, mock_generate, mock_load, mock_isdir):
        mock_isdir.return_value = True
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_generate.return_value = "async response"
        
        engine = MlxLLMEngine(model_path="/fake/model/path")
        res = await engine.generate_response_async("prompt")
        self.assertEqual(res, "async response")
        
        mock_generate.return_value = '{"name": "async", "age": 2}'
        res_json = await engine.generate_json_async("prompt", MockSchema)
        self.assertEqual(res_json, '{"name": "async", "age": 2}')

    @patch("os.path.isdir")
    @patch("mlx_lm.load")
    @patch("mlx_lm.generate")
    @patch("src.llm_engine.mlx_impl.build_mlx_tokenizer_data")
    @patch("lmformatenforcer.TokenEnforcer")
    @patch("lmformatenforcer.JsonSchemaParser")
    async def test_chat_template_exception_handling(self, mock_parser, mock_enforcer, mock_build_data, mock_generate, mock_load, mock_isdir):
        mock_isdir.return_value = True
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.side_effect = Exception("apply_chat_template error")
        mock_load.return_value = (MagicMock(), mock_tokenizer)
        
        mock_generate.return_value = "response with exception template"
        
        engine = MlxLLMEngine(model_path="/fake/model/path")
        res = await engine.generate_response_async("prompt")
        self.assertEqual(res, "response with exception template")
        
        mock_generate.return_value = '{"name": "test", "age": 1}'
        res_json = await engine.generate_json_async("prompt", MockSchema)
        self.assertEqual(res_json, '{"name": "test", "age": 1}')
