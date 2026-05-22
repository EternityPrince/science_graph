import unittest
from unittest.mock import MagicMock, patch
import json

from src.llm_schemas import (
    validate_extraction_response,
    validate_clustering_response,
    LLMExtractionResponse,
    LLMClusteringResponse
)
from src.llm_engine import BaseLLMEngine


class TestLLMValidation(unittest.TestCase):
    def test_extraction_happy_path(self):
        """Test extraction validation with clean, correct inputs."""
        raw_data = {
            "authors": ["Alice Smith", "Bob Jones"],
            "concepts": [
                {"name": "Self-Attention", "description": "Relates different positions of a single sequence."},
                {"name": "Transformer Model", "description": "Encoder-decoder model based on attention."}
            ],
            "tags": ["deep learning", "nlp"]
        }
        model, warnings = validate_extraction_response(raw_data)
        
        self.assertEqual(warnings, [])
        self.assertEqual(model.authors, ["Alice Smith", "Bob Jones"])
        self.assertEqual(len(model.concepts), 2)
        self.assertEqual(model.concepts[0].name, "Self-Attention")
        self.assertEqual(model.tags, ["deep learning", "nlp"])

    def test_extraction_author_filtering(self):
        """Test that institutional names, citations, emails, and malformed names are filtered from authors."""
        raw_data = {
            "authors": [
                "Jane Doe",
                "University of Toronto",  # institution
                "Department of CS",       # dept
                "John Smith et al.",      # citation/et al
                "bob@example.com",        # email
                "https://example.com",    # URL
                "Vol. 12",                # volume
                "12345",                  # digits only
                "A",                      # too short
                "  Alice Cooper, "        # trailing punctuation/whitespace
            ],
            "concepts": [],
            "tags": []
        }
        model, warnings = validate_extraction_response(raw_data)
        
        self.assertEqual(model.authors, ["Jane Doe", "Alice Cooper"])
        self.assertTrue(any("University of Toronto" in w for w in warnings))
        self.assertTrue(any("Department of CS" in w for w in warnings))
        self.assertTrue(any("John Smith et al." in w for w in warnings))
        self.assertTrue(any("bob@example.com" in w for w in warnings))

    def test_extraction_concept_filtering(self):
        """Test filtering of concepts with names that are too long, contain citations, or are empty."""
        raw_data = {
            "authors": [],
            "concepts": [
                {"name": "Self-Attention", "description": "Good description"},
                {"name": "Transformer Architecture [1]", "description": "Has citation"},
                {"name": "This is a very long concept name with many words", "description": "Too long name"},
                {"name": "", "description": "Empty name"},
                {"name": "Short Name", "description": "No"}  # too short description warning
            ],
            "tags": []
        }
        model, warnings = validate_extraction_response(raw_data)
        
        # Self-Attention is kept.
        # "Transformer Architecture [1]" is kept but cleaned to "Transformer Architecture".
        # "This is a very long concept name..." is filtered out.
        # "" is filtered out.
        # "Short Name" is kept but generates a description length warning.
        concept_names = [c.name for c in model.concepts]
        self.assertIn("Self-Attention", concept_names)
        self.assertIn("Transformer Architecture", concept_names)
        self.assertNotIn("This is a very long concept name with many words", concept_names)
        
        self.assertTrue(any("too long" in w for w in warnings))
        self.assertTrue(any("unusual length" in w for w in warnings))

    def test_extraction_tag_filtering(self):
        """Test tag validation including length limits and invalid characters."""
        raw_data = {
            "authors": [],
            "concepts": [],
            "tags": [
                "deep learning",
                "natural language processing models",  # 4 words - allowed
                "this tag is a whole sentence and is too long",  # 10 words - filtered
                "bad@tag",  # invalid characters - filtered
                "tag[1]"    # citation brackets - filtered
            ]
        }
        model, warnings = validate_extraction_response(raw_data)
        
        self.assertEqual(model.tags, ["deep learning", "natural language processing models"])
        self.assertTrue(any("too long" in w for w in warnings))
        self.assertTrue(any("invalid characters" in w for w in warnings))

    def test_extraction_type_mismatches(self):
        """Test that type mismatches (e.g. list fields not lists) are handled gracefully."""
        raw_data = {
            "authors": "Not A List",
            "concepts": "Not A List",
            "tags": "Not A List"
        }
        model, warnings = validate_extraction_response(raw_data)
        self.assertEqual(model.authors, [])
        self.assertEqual(model.concepts, [])
        self.assertEqual(model.tags, [])
        self.assertTrue(any("Expected 'authors' to be a list" in w for w in warnings))

    def test_clustering_happy_path(self):
        """Test clustering validation with clean correct inputs."""
        raw_data = {
            "Introduction": ["chunk_1", "chunk_2"],
            "Methodology": ["chunk_3"]
        }
        model, warnings = validate_clustering_response(raw_data)
        
        self.assertEqual(warnings, [])
        self.assertEqual(model.root, raw_data)

    def test_clustering_filtering(self):
        """Test filtering of empty sections, non-list chunk IDs, empty lists, etc."""
        raw_data = {
            "Introduction": ["chunk_1", ""],
            "EmptySection": [],
            "  ": ["chunk_2"],
            "NonListSection": "chunk_3",
            123: ["chunk_4"]
        }
        model, warnings = validate_clustering_response(raw_data)
        
        self.assertIn("Introduction", model.root)
        self.assertEqual(model.root["Introduction"], ["chunk_1"])
        self.assertNotIn("EmptySection", model.root)
        self.assertNotIn("  ", model.root)
        self.assertNotIn("NonListSection", model.root)
        self.assertTrue(any("is not a list" in w for w in warnings))
        self.assertTrue(any("has no valid chunk IDs" in w for w in warnings))


class DummyEngine(BaseLLMEngine):
    def __init__(self):
        self.response = ""

    def generate_response(self, prompt: str, max_tokens: int = None, temp: float = None, task: str = None) -> str:
        return self.response


class TestLLMEngineIntegration(unittest.TestCase):
    def setUp(self):
        self.engine = DummyEngine()

    @patch("src.llm_engine.con")
    def test_engine_extract_success(self, mock_con):
        """Test extract_concepts_and_metadata with valid LLM output."""
        self.engine.response = json.dumps({
            "authors": ["John Doe"],
            "concepts": [{"name": "Attention", "description": "Focus on specific parts."}],
            "tags": ["deep learning"]
        })
        
        res = self.engine.extract_concepts_and_metadata("sample text")
        
        self.assertIsNotNone(res)
        self.assertEqual(res["authors"], ["John Doe"])
        self.assertEqual(res["tags"], ["deep learning"])
        mock_con.success.assert_called_with("LLM extraction output validated successfully.")
        mock_con.info.assert_called()

    @patch("src.llm_engine.con")
    def test_engine_extract_invalid_json(self, mock_con):
        """Test extract_concepts_and_metadata handles invalid JSON response gracefully."""
        self.engine.response = "invalid json {..."
        res = self.engine.extract_concepts_and_metadata("sample text")
        
        self.assertIsNone(res)
        mock_con.warning.assert_called()

    @patch("src.llm_engine.con")
    def test_engine_extract_validation_warnings(self, mock_con):
        """Test extraction triggers warning logs on low-quality output, but returns cleaned model."""
        self.engine.response = json.dumps({
            "authors": ["John Doe", "University of Nowhere"],
            "concepts": [
                {"name": "Attention", "description": "Focus on specific parts."},
                {"name": "Very long concept name that is actually a sentence", "description": "Too long"}
            ],
            "tags": ["deep learning"]
        })
        
        res = self.engine.extract_concepts_and_metadata("sample text")
        
        self.assertIsNotNone(res)
        self.assertEqual(res["authors"], ["John Doe"])
        self.assertEqual(len(res["concepts"]), 1)  # long concept is filtered out
        
        # Verify warnings were logged to console
        mock_con.warning.assert_any_call("LLM extraction output validated with warnings:")
        any_warning = any("noisy" in call.args[0] or "too long" in call.args[0] for call in mock_con.warning.call_args_list)
        self.assertTrue(any_warning)

    @patch("src.llm_engine.con")
    def test_engine_cluster_success(self, mock_con):
        """Test cluster_chunks_by_topic happy path validation."""
        self.engine.response = json.dumps({
            "Introduction": ["chunk_1", "chunk_2"]
        })
        
        res = self.engine.cluster_chunks_by_topic("summary", "topic")
        
        self.assertIsNotNone(res)
        self.assertEqual(res, {"Introduction": ["chunk_1", "chunk_2"]})
        mock_con.success.assert_called_with("LLM clustering output validated successfully.")
