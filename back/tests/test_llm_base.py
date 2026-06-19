import unittest
from unittest.mock import MagicMock, patch
import json
from pydantic import BaseModel
from typing import List

from src.config import config
from src.llm_engine.base import (
    BaseLLMEngine,
    ResilientParser,
    strip_thinking_tokens,
    retry_with_temp_decay,
    retry_with_temp_decay_async,
)


class DummySchema(BaseModel):
    items: List[str]
    count: int


class DummyLLMEngine(BaseLLMEngine):
    def __init__(self):
        self.tokenizer = None
        self.generate_response_mock = MagicMock()
        self.generate_json_mock = MagicMock()

    def generate_response(self, prompt: str, max_tokens: int = None, temp: float = None, task: str = None, model: str = None) -> str:
        return self.generate_response_mock(prompt, max_tokens=max_tokens, temp=temp, task=task, model=model)

    def generate_json(self, prompt: str, schema_class: type[BaseModel], temp: float = 0.0, max_tokens: int = None) -> str:
        return self.generate_json_mock(prompt=prompt, schema_class=schema_class, temp=temp, max_tokens=max_tokens)


class TestLlmBase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.orig_data = config.data
        config.data = {
            "llm": {
                "extraction_input_limit": 50,
                "extraction_output_limit": 50,
                "clustering_input_limit": 60,
                "clustering_output_limit": 60,
                "synthesis_input_limit": 70,
                "synthesis_output_limit": 70,
            }
        }

    def tearDown(self):
        config.data = self.orig_data

    def test_strip_thinking_tokens(self):
        self.assertEqual(strip_thinking_tokens(""), "")
        self.assertEqual(strip_thinking_tokens(None), None)
        self.assertEqual(strip_thinking_tokens("hello <think>thought</think> world"), "hello  world")
        self.assertEqual(strip_thinking_tokens("hello <think>unclosed thought"), "hello")
        self.assertEqual(strip_thinking_tokens("hello </think> world"), "hello  world")
        self.assertEqual(strip_thinking_tokens("hello <think> world"), "hello")
        self.assertEqual(strip_thinking_tokens("<|im_start|>assistant\nhello <|im_end|>"), "hello")
        self.assertEqual(strip_thinking_tokens("[INST] hello [/INST] world"), "hello  world")
        self.assertEqual(strip_thinking_tokens("<s>hello</s>"), "hello")

    def test_clean_llm_output_decorator(self):
        class TestEngine(BaseLLMEngine):
            def generate_response(self, prompt: str, **kwargs) -> str:
                return "hello <think>thought</think> world"
            
            async def generate_response_async(self, prompt: str, **kwargs) -> str:
                return "hello <think>thought</think> world"

        engine = TestEngine()
        
        # Calling generate_response should return clean string
        res = engine.generate_response("test prompt")
        self.assertEqual(res, "hello  world")

        # Calling generate_response_async should also return clean string
        import asyncio
        res_async = asyncio.run(engine.generate_response_async("test prompt"))
        self.assertEqual(res_async, "hello  world")

    def test_resilient_json_parser(self):
        # markdown code blocks
        self.assertEqual(ResilientParser.extract_json("```json\n{\"a\": 1}\n```"), "{\"a\": 1}")
        # first open character matching
        self.assertEqual(ResilientParser.extract_json("prefix {\"a\": 1} suffix"), "{\"a\": 1}")
        self.assertEqual(ResilientParser.extract_json("prefix [1, 2, 3] suffix"), "[1, 2, 3]")
        # quote escaping inside strings
        json_with_escapes = '{"text": "he said \\"hello\\" to me"}'
        self.assertEqual(ResilientParser.extract_json("prefix " + json_with_escapes + " suffix"), json_with_escapes)
        # mismatched braces fallbacks
        self.assertEqual(ResilientParser.extract_json("mismatched { \"a\": 1"), "mismatched { \"a\": 1")
        self.assertEqual(ResilientParser.extract_json("mismatched [ 1, 2"), "mismatched [ 1, 2")

    def test_retry_with_temp_decay_decorator(self):
        call_count = 0
        temperatures = []

        @retry_with_temp_decay(max_retries=3)
        def dummy_func(temp=1.0):
            nonlocal call_count
            call_count += 1
            temperatures.append(temp)
            if call_count < 3:
                raise ValueError("temporary error")
            return "success"

        res = dummy_func(temp=1.0)
        self.assertEqual(res, "success")
        self.assertEqual(call_count, 3)
        self.assertEqual(temperatures, [1.0, 1.0 * (1.0 - 1/3), 1.0 * (1.0 - 2/3)])

    async def test_retry_with_temp_decay_async_decorator(self):
        call_count = 0
        temperatures = []

        @retry_with_temp_decay_async(max_retries=2)
        async def dummy_func_async(temp=0.8):
            nonlocal call_count
            call_count += 1
            temperatures.append(temp)
            if call_count < 3:
                raise json.JSONDecodeError("decode error", "", 0)
            return "success_async"

        res = await dummy_func_async(temp=0.8)
        self.assertEqual(res, "success_async")
        self.assertEqual(call_count, 3)
        self.assertEqual(temperatures, [0.8, 0.8 * (1.0 - 1/2), 0.0])

    def test_count_tokens(self):
        engine = DummyLLMEngine()
        self.assertEqual(engine.count_tokens(""), 0)
        self.assertEqual(engine.count_tokens(None), 0)
        
        # no tokenizer
        self.assertEqual(engine.count_tokens("12345678"), 2) # len // 4
        
        # with tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        engine.tokenizer = mock_tokenizer
        self.assertEqual(engine.count_tokens("hello"), 5)
        
        # tokenizer exception fallback
        mock_tokenizer.encode.side_effect = Exception("error")
        self.assertEqual(engine.count_tokens("12345678"), 2)

    def test_generate_and_validate_json(self):
        engine = DummyLLMEngine()
        engine.generate_json_mock.return_value = '{"items": ["a", "b"], "count": 2}'
        
        res = engine.generate_and_validate_json("prompt", DummySchema)
        self.assertEqual(res.items, ["a", "b"])
        self.assertEqual(res.count, 2)
        engine.generate_json_mock.assert_called_once_with(
            prompt="prompt",
            schema_class=DummySchema,
            temp=0.0,
            max_tokens=None
        )

        # test decode fail exception
        engine.generate_json_mock.return_value = 'corrupt json'
        with self.assertRaises(json.JSONDecodeError):
            engine.generate_and_validate_json("prompt", DummySchema)

        # test schema validation exception
        engine.generate_json_mock.return_value = '{"items": ["a"], "count": "not-an-int"}'
        with self.assertRaises(ValueError):
            engine.generate_and_validate_json("prompt", DummySchema)

    async def test_generate_and_validate_json_async(self):
        engine = DummyLLMEngine()
        
        async def mock_gen_json(prompt, schema_class, temp, max_tokens):
            return '{"items": ["c"], "count": 1}'
            
        engine.generate_json_async = mock_gen_json
        res = await engine.generate_and_validate_json_async("prompt", DummySchema)
        self.assertEqual(res.items, ["c"])
        self.assertEqual(res.count, 1)

    @patch("src.llm_schemas.validate_extraction_response")
    def test_extract_concepts_and_metadata(self, mock_validate):
        engine = DummyLLMEngine()
        
        # Mock successful JSON extraction response
        dummy_json = '{"authors": ["Alice"], "concepts": [], "tags": [], "institutions": [], "author_institutions": [], "sponsored_by": [], "datasets": [], "code_repositories": [], "journal_or_conference": "", "citation_intents": [], "concept_relations": []}'
        
        # Mock LLMExtractionResponse validation
        from src.llm_schemas import LLMExtractionResponse
        mock_response_model = LLMExtractionResponse.model_validate(json.loads(dummy_json))
        
        engine.generate_json_mock.return_value = dummy_json
        mock_validate.return_value = (mock_response_model, []) # validated, warnings
        
        res = engine.extract_concepts_and_metadata("text")
        self.assertEqual(res["authors"], ["Alice"])

        # test failure flow
        engine.generate_json_mock.side_effect = Exception("LLM crash")
        res = engine.extract_concepts_and_metadata("text")
        self.assertIsNone(res)

    @patch("src.llm_schemas.validate_extraction_response")
    async def test_extract_concepts_and_metadata_async(self, mock_validate):
        engine = DummyLLMEngine()
        
        dummy_json = '{"authors": ["Bob"], "concepts": [], "tags": [], "institutions": [], "author_institutions": [], "sponsored_by": [], "datasets": [], "code_repositories": [], "journal_or_conference": "", "citation_intents": [], "concept_relations": []}'
        
        from src.llm_schemas import LLMExtractionResponse
        mock_response_model = LLMExtractionResponse.model_validate(json.loads(dummy_json))
        
        async def mock_gen_json_async(prompt, schema_class, temp, max_tokens):
            return dummy_json
            
        engine.generate_json_async = mock_gen_json_async
        mock_validate.return_value = (mock_response_model, ["warning message"])
        
        res = await engine.extract_concepts_and_metadata_async("text")
        self.assertEqual(res["authors"], ["Bob"])

    @patch("src.llm_schemas.validate_clustering_response")
    def test_cluster_chunks_by_topic(self, mock_validate):
        engine = DummyLLMEngine()
        
        dummy_json = '{"Section A": ["chunk1"]}'
        from src.llm_schemas import LLMClusteringResponse
        mock_response_model = LLMClusteringResponse.model_validate(json.loads(dummy_json))
        
        engine.generate_json_mock.return_value = dummy_json
        mock_validate.return_value = (mock_response_model, [])
        
        res = engine.cluster_chunks_by_topic("summary", "topic")
        self.assertEqual(res, {"Section A": ["chunk1"]})

        # failure flow
        engine.generate_json_mock.side_effect = Exception("Clustering error")
        res = engine.cluster_chunks_by_topic("summary", "topic")
        self.assertIsNone(res)

    def test_synthesize_section(self):
        engine = DummyLLMEngine()
        engine.generate_response_mock.return_value = "Synthesized text"
        
        res = engine.synthesize_section("section", "chunks", "topic")
        self.assertEqual(res, "Synthesized text")
        engine.generate_response_mock.assert_called_once_with(
            unittest.mock.ANY,
            max_tokens=None,
            temp=0.2,
            task="synthesis",
            model=None
        )

        # test failure
        engine.generate_response_mock.side_effect = Exception("fail")
        res = engine.synthesize_section("section", "chunks", "topic")
        self.assertIn("Generation failed", res)

    @patch("src.llm_engine.gguf_impl.os.path.exists")
    @patch("src.llm_engine.mlx_impl.os.path.isdir")
    def test_llm_engine_factory(self, mock_isdir, mock_exists):
        mock_isdir.return_value = True
        mock_exists.return_value = True
        
        import os
        with patch.dict(os.environ, {"SCIENCE_GRAPH_USE_CLOUD": "0"}):
            # Test MLX provider
            with patch.dict(config.data, {"llm": {"provider": "mlx", "local": {"model_path": "/fake/path"}}}):
                import src.llm_engine
                src.llm_engine._local_engine_singleton = None
                from src.llm_engine.factory import LLMEngine
                engine = LLMEngine(use_cloud=False)
                from src.llm_engine.mlx_impl import MlxLLMEngine
                self.assertIsInstance(engine, MlxLLMEngine)

            # Test GGUF provider
            with patch.dict(config.data, {"llm": {"provider": "gguf", "local": {"model_path": "/fake/path.gguf"}}}):
                import src.llm_engine
                src.llm_engine._local_engine_singleton = None
                from src.llm_engine.factory import LLMEngine
                engine = LLMEngine(use_cloud=False)
                from src.llm_engine.gguf_impl import GgufLLMEngine
                self.assertIsInstance(engine, GgufLLMEngine)
