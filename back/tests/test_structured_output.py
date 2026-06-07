import unittest
from unittest.mock import MagicMock, AsyncMock
import json
from pydantic import BaseModel
from typing import List

from src.llm_engine.base import BaseLLMEngine, StructuredOutput


class MockSchema(BaseModel):
    items: List[str]
    count: int


class DummyLLMEngine(BaseLLMEngine):
    def __init__(self):
        self.generate_json_mock = MagicMock()
        self.generate_json_async_mock = AsyncMock()

    def generate_response(self, prompt: str, max_tokens: int = None, temp: float = None, task: str = None, model: str = None) -> str:
        return ""

    def generate_json(self, prompt: str, schema_class: type[BaseModel], temp: float = 0.0, max_tokens: int = None) -> str:
        return self.generate_json_mock(prompt, schema_class, temp, max_tokens)

    async def generate_json_async(self, prompt: str, schema_class: type[BaseModel], temp: float = 0.0, max_tokens: int = None) -> str:
        return await self.generate_json_async_mock(prompt, schema_class, temp, max_tokens)


class TestStructuredOutput(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine = DummyLLMEngine()

    def test_sync_happy_path(self):
        self.engine.generate_json_mock.return_value = '{"items": ["apple", "banana"], "count": 2}'
        
        structured = StructuredOutput(MockSchema, max_retries=3)
        res = structured.generate(self.engine, "Give me fruits", temp=0.9)
        
        self.assertEqual(res.items, ["apple", "banana"])
        self.assertEqual(res.count, 2)
        self.engine.generate_json_mock.assert_called_once_with(
            "Give me fruits", MockSchema, 0.9, None
        )

    async def test_async_happy_path(self):
        self.engine.generate_json_async_mock.return_value = '{"items": ["orange"], "count": 1}'
        
        structured = StructuredOutput(MockSchema, max_retries=3)
        res = await structured.generate_async(self.engine, "Give me more fruits", temp=0.6)
        
        self.assertEqual(res.items, ["orange"])
        self.assertEqual(res.count, 1)
        self.engine.generate_json_async_mock.assert_called_once_with(
            "Give me more fruits", MockSchema, 0.6, None
        )

    def test_sync_retries_with_temp_decay_and_success(self):
        # First 2 calls fail (one with bad JSON, one with validation error), 3rd call succeeds.
        calls = []
        def side_effect(prompt, schema_class, temp, max_tokens):
            calls.append(temp)
            if len(calls) == 1:
                return "bad json data"
            elif len(calls) == 2:
                return '{"items": ["pear"], "count": "not-an-int-error"}'
            else:
                return '{"items": ["pear"], "count": 1}'

        self.engine.generate_json_mock.side_effect = side_effect
        
        structured = StructuredOutput(MockSchema, max_retries=3)
        res = structured.generate(self.engine, "Give me a pear", temp=0.9)
        
        self.assertEqual(res.items, ["pear"])
        self.assertEqual(res.count, 1)
        self.assertEqual(len(calls), 3)
        # Check temperature decay steps:
        # 1st try: 0.9
        # 2nd try: 0.9 * (1.0 - 1/3) = 0.6
        # 3rd try: 0.9 * (1.0 - 2/3) = 0.3
        self.assertAlmostEqual(calls[0], 0.9)
        self.assertAlmostEqual(calls[1], 0.6)
        self.assertAlmostEqual(calls[2], 0.3)

    async def test_async_retries_with_temp_decay_and_success(self):
        calls = []
        async def side_effect(prompt, schema_class, temp, max_tokens):
            calls.append(temp)
            if len(calls) == 1:
                return "invalid json"
            elif len(calls) == 2:
                return '{"items": ["grape"], "count": "invalid"}'
            else:
                return '{"items": ["grape"], "count": 5}'

        self.engine.generate_json_async_mock.side_effect = side_effect
        
        structured = StructuredOutput(MockSchema, max_retries=2)
        res = await structured.generate_async(self.engine, "Give grapes", temp=1.0)
        
        self.assertEqual(res.items, ["grape"])
        self.assertEqual(res.count, 5)
        self.assertEqual(len(calls), 3)
        # Check temperature decay steps for max_retries=2:
        # 1st try: 1.0
        # 2nd try: 1.0 * (1.0 - 1/2) = 0.5
        # 3rd try: 0.0 (last attempt is always 0.0)
        self.assertAlmostEqual(calls[0], 1.0)
        self.assertAlmostEqual(calls[1], 0.5)
        self.assertAlmostEqual(calls[2], 0.0)

    def test_sync_persistent_failure(self):
        self.engine.generate_json_mock.return_value = "completely corrupt"
        
        structured = StructuredOutput(MockSchema, max_retries=3)
        with self.assertRaises(json.JSONDecodeError):
            structured.generate(self.engine, "Fail me", temp=0.5)
            
        self.assertEqual(self.engine.generate_json_mock.call_count, 4)

    async def test_async_persistent_failure(self):
        self.engine.generate_json_async_mock.return_value = '{"items": [], "count": "missing_int"}'
        
        structured = StructuredOutput(MockSchema, max_retries=2)
        with self.assertRaises(ValueError):
            await structured.generate_async(self.engine, "Fail async", temp=0.8)
            
        self.assertEqual(self.engine.generate_json_async_mock.call_count, 3)
