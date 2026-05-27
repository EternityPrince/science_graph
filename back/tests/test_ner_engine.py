"""
Unit tests for NER Engine.
"""

import unittest
from unittest.mock import MagicMock, patch
import os

from src.ner_engine import (
    _is_likely_name,
    _is_model_cached,
    NEREngine,
    get_ner_engine,
    extract_persons_from_text,
)

class TestNEREngineFunctions(unittest.TestCase):
    def test_is_likely_name(self):
        # Valid names
        self.assertTrue(_is_likely_name("Ashish Vaswani"))
        self.assertTrue(_is_likely_name("Aidan N. Gomez"))
        self.assertTrue(_is_likely_name("Linus Torvalds"))
        self.assertTrue(_is_likely_name("Guido van Rossum"))

        # Invalid names
        self.assertFalse(_is_likely_name("Guido"))  # Too short
        self.assertFalse(_is_likely_name("A B C D E F"))  # Too long
        self.assertFalse(_is_likely_name("John and Bob"))  # Has conjunction
        self.assertFalse(_is_likely_name("John 3 Doe"))  # Has digit
        self.assertFalse(_is_likely_name("Deep Learning"))  # Stopword
        self.assertFalse(_is_likely_name("Google Brain"))  # Stopword
        self.assertFalse(_is_likely_name("A" * 25 + " " + "B" * 30))  # Too long length (>50 chars, but has 2 words)

    @patch("huggingface_hub.try_to_load_from_cache")
    def test_is_model_cached(self, mock_load):
        mock_load.return_value = "path/to/model"
        self.assertTrue(_is_model_cached("dummy_model"))

        mock_load.return_value = None
        self.assertFalse(_is_model_cached("dummy_model"))

        mock_load.side_effect = Exception("error")
        self.assertFalse(_is_model_cached("dummy_model"))


class TestNEREngine(unittest.TestCase):
    @patch("src.ner_engine.os.environ")
    @patch("src.ner_engine._is_model_cached")
    @patch("transformers.pipeline")
    @patch("src.config.config")
    def test_init_success_cached(self, mock_config, mock_pipeline, mock_cached, mock_environ):
        mock_config.ner_model_name = "cached_model"
        mock_config.hf_token = "some_token"
        mock_cached.return_value = True

        engine = NEREngine()
        self.assertEqual(engine.model_id, "cached_model")
        mock_pipeline.assert_called_once_with(
            "ner",
            model="cached_model",
            aggregation_strategy="first",
            token="some_token",
        )
        self.assertIsNotNone(engine._pipeline)

    @patch("src.ner_engine.os.environ")
    @patch("src.ner_engine._is_model_cached")
    @patch("transformers.pipeline")
    @patch("src.config.config")
    def test_init_success_not_cached(self, mock_config, mock_pipeline, mock_cached, mock_environ):
        mock_config.ner_model_name = "uncached_model"
        mock_config.hf_token = None
        mock_cached.return_value = False

        engine = NEREngine()
        mock_pipeline.assert_called_once_with(
            "ner",
            model="uncached_model",
            aggregation_strategy="first",
            token=None,
        )

    @patch("src.ner_engine._is_model_cached")
    @patch("transformers.pipeline")
    @patch("src.config.config")
    def test_init_pipeline_fails(self, mock_config, mock_pipeline, mock_cached):
        mock_config.ner_model_name = "cached_model"
        mock_config.hf_token = None
        mock_cached.return_value = True
        mock_pipeline.side_effect = Exception("pipeline load error")

        engine = NEREngine()
        self.assertIsNone(engine._pipeline)

    @patch("src.ner_engine.NEREngine._load_model")
    def test_regex_extract(self, mock_load):
        engine = NEREngine()
        engine._pipeline = None  # Ensure regex fallback

        text = "This paper was written by Ashish Vaswani and Aidan N. Gomez. Introduction is here."
        names = engine.extract_persons(text)
        self.assertIn("Ashish Vaswani", names)
        self.assertIn("Aidan N. Gomez", names)
        self.assertNotIn("Introduction", names)

    @patch("src.ner_engine.NEREngine._load_model")
    def test_ner_extract_success(self, mock_load):
        engine = NEREngine()
        mock_pipeline = MagicMock()
        engine._pipeline = mock_pipeline

        # Return entity list
        mock_pipeline.return_value = [
            {"entity_group": "PER", "word": "Ashish Vaswani", "start": 0, "end": 14},
            {"entity_group": "PER", "word": "##ish Vaswani", "start": None, "end": None},  # subword, no start/end
            {"entity_group": "ORG", "word": "Google", "start": 18, "end": 24},
        ]

        text = "Ashish Vaswani and Google"
        results = engine.extract_persons(text)
        self.assertIn("Ashish Vaswani", results)
        self.assertIn("ish Vaswani", results)  # Rebuilt from "##ish Vaswani" using subword replacement

    @patch("src.ner_engine.NEREngine._load_model")
    def test_ner_extract_fails_fallback(self, mock_load):
        engine = NEREngine()
        mock_pipeline = MagicMock()
        engine._pipeline = mock_pipeline
        mock_pipeline.side_effect = Exception("runtime execution error")

        text = "This paper was written by Ashish Vaswani."
        # Should fallback to regex
        results = engine.extract_persons(text)
        self.assertIn("Ashish Vaswani", results)


class TestNERModuleSingleton(unittest.TestCase):
    @patch("src.ner_engine._ner_instance", None)
    @patch("src.ner_engine.NEREngine")
    def test_get_ner_engine(self, mock_ner_cls):
        engine1 = get_ner_engine()
        engine2 = get_ner_engine()
        self.assertIs(engine1, engine2)
        mock_ner_cls.assert_called_once()

    @patch("src.ner_engine.get_ner_engine")
    def test_extract_persons_from_text(self, mock_get_engine):
        mock_instance = MagicMock()
        mock_instance.extract_persons.return_value = ["Alice Smith", "Bob Jones"]
        mock_get_engine.return_value = mock_instance

        res = extract_persons_from_text("some text")
        self.assertEqual(res, ["Alice Smith", "Bob Jones"])
        mock_instance.extract_persons.assert_called_once_with("some text")
