import unittest
from unittest.mock import patch, mock_open
from src.config import Config, config
from src.services.extraction_service import ExtractionService

class TestTaxonomy(unittest.TestCase):
    def test_taxonomy_structure(self):
        """Test that the taxonomy property has the expected structure and contents."""
        tax = config.taxonomy
        self.assertIn("concepts", tax)
        self.assertIn("topics", tax)
        self.assertIn("descriptions", tax)
        self.assertIsInstance(tax["concepts"], dict)
        self.assertIsInstance(tax["topics"], dict)
        self.assertIsInstance(tax["descriptions"], dict)

    def test_taxonomy_lazy_loading(self):
        """Test that taxonomy is cached and not re-read from disk every time."""
        cfg = Config()
        self.assertFalse(hasattr(cfg, "_taxonomy"))
        tax1 = cfg.taxonomy
        self.assertTrue(hasattr(cfg, "_taxonomy"))
        tax2 = cfg.taxonomy
        self.assertIs(tax1, tax2)

    def test_taxonomy_missing_file_fallback(self):
        """Test that taxonomy falls back to empty dicts when the yaml file is missing."""
        cfg = Config()
        with patch("src.config.open", side_effect=FileNotFoundError):
            tax = cfg.taxonomy
        self.assertEqual(tax, {"concepts": {}, "topics": {}, "descriptions": {}})

    def test_taxonomy_invalid_yaml_fallback(self):
        """Test that taxonomy falls back to empty dicts when the yaml file is malformed."""
        cfg = Config()
        with patch("src.config.open", mock_open(read_data="invalid: yaml: ["), create=True):
            tax = cfg.taxonomy
        self.assertEqual(tax, {"concepts": {}, "topics": {}, "descriptions": {}})

    def test_taxonomy_keywords_case_insensitivity(self):
        """Test that concept/tag matching handles case-insensitivity correctly."""
        dummy_tax = {
            "concepts": {
                "transformer": "Transformer Architecture",
                "Self-Attention": "Self-Attention Mechanism"
            },
            "topics": {
                "nlp": "Natural Language Processing"
            },
            "descriptions": {}
        }
        
        with patch.object(Config, "taxonomy", new=dummy_tax):
            extractor = ExtractionService(llm_engine=None)
            
            # Match transformer case-insensitively
            res1 = extractor.extract(
                title="A study on TRANSFORMER networks",
                abstract="",
                full_text=""
            )
            concept_names = [c["name"] for c in res1.concepts]
            self.assertIn("Transformer Architecture", concept_names)
            
            # Match self-attention case-insensitively
            res2 = extractor.extract(
                title="About self-attention mechanism",
                abstract="",
                full_text=""
            )
            concept_names2 = [c["name"] for c in res2.concepts]
            self.assertIn("Self-Attention Mechanism", concept_names2)
