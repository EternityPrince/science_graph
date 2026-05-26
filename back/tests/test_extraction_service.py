import os
import unittest
from unittest.mock import MagicMock, patch
from src.services.extraction_service import ExtractionService
from src.models import Paper

class TestExtractionService(unittest.TestCase):
    def setUp(self):
        self.llm_engine = MagicMock()
        self.dummy_tax = {
            "concepts": {
                "transformer": "Transformer Architecture",
                "attention": "Attention Mechanism"
            },
            "topics": {
                "deep learning": "Deep Learning",
                "nlp": "Natural Language Processing"
            },
            "descriptions": {
                "Transformer Architecture": "A model architecture based on self-attention."
            }
        }
        from src.config import Config
        # Patch Config.taxonomy on the class level to return our dummy taxonomy
        self.patcher = patch.object(Config, "taxonomy", new=self.dummy_tax)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_init(self):
        """Test ExtractionService initialization."""
        service = ExtractionService(llm_engine=self.llm_engine)
        self.assertEqual(service.llm_engine, self.llm_engine)
        self.assertEqual(service._tax, self.dummy_tax)

    def test_extract_llm_success(self):
        """Test extraction when LLM succeeds."""
        service = ExtractionService(llm_engine=self.llm_engine)
        self.llm_engine.extract_concepts_and_metadata.return_value = {
            "authors": ["Alice Smith", "Bob Jones"],
            "concepts": [
                {"name": "Transformer Architecture", "description": "Custom LLM description."},
                {"name": "Attention Mechanism"}  # missing description
            ],
            "tags": ["Deep Learning"]
        }

        res = service.extract("Title", "Abstract", "Full text content", use_llm=True)
        
        self.assertTrue(res.via_llm)
        self.assertEqual(res.authors, ["Alice Smith", "Bob Jones"])
        self.assertEqual(res.tags, ["Deep Learning"])
        
        # Verify concepts list and that "Attention Mechanism" description was filled in
        self.assertEqual(len(res.concepts), 2)
        self.assertEqual(res.concepts[0]["name"], "Transformer Architecture")
        self.assertEqual(res.concepts[0]["description"], "Custom LLM description.")
        
        self.assertEqual(res.concepts[1]["name"], "Attention Mechanism")
        # Since it had no description, it should fallback to get_concept_description
        self.assertTrue(len(res.concepts[1]["description"]) > 0)

    def test_extract_llm_returns_empty_fallback_to_regex(self):
        """Test fallback to regex if LLM returns empty/None."""
        service = ExtractionService(llm_engine=self.llm_engine)
        self.llm_engine.extract_concepts_and_metadata.return_value = None

        res = service.extract("Title about transformer models", "Abstract", "nlp is cool", use_llm=True)
        self.assertFalse(res.via_llm)
        concept_names = [c["name"] for c in res.concepts]
        self.assertIn("Transformer Architecture", concept_names)
        self.assertIn("Natural Language Processing", res.tags)

    def test_extract_llm_exception_fallback_to_regex(self):
        """Test fallback to regex if LLM raises an exception."""
        service = ExtractionService(llm_engine=self.llm_engine)
        self.llm_engine.extract_concepts_and_metadata.side_effect = Exception("LLM Timeout")

        res = service.extract("Title about transformer models", "Abstract", "nlp is cool", use_llm=True)
        self.assertFalse(res.via_llm)
        concept_names = [c["name"] for c in res.concepts]
        self.assertIn("Transformer Architecture", concept_names)
        self.assertIn("Natural Language Processing", res.tags)

    def test_extract_use_llm_false(self):
        """Test that extract runs regex only when use_llm is False."""
        service = ExtractionService(llm_engine=self.llm_engine)
        res = service.extract("Title about transformer", "Abstract", "", use_llm=False)
        
        self.assertFalse(res.via_llm)
        self.llm_engine.extract_concepts_and_metadata.assert_not_called()
        concept_names = [c["name"] for c in res.concepts]
        self.assertIn("Transformer Architecture", concept_names)

    def test_extract_no_llm_engine(self):
        """Test that extract runs regex when llm_engine is None."""
        service = ExtractionService(llm_engine=None)
        res = service.extract("Title about transformer", "Abstract", "", use_llm=True)
        
        self.assertFalse(res.via_llm)
        concept_names = [c["name"] for c in res.concepts]
        self.assertIn("Transformer Architecture", concept_names)

    def test_get_concept_description_predefined(self):
        """Test concept description lookup from taxonomy (case insensitive)."""
        service = ExtractionService(llm_engine=None)
        
        # Exact match
        desc1 = service.get_concept_description("Transformer Architecture")
        self.assertEqual(desc1, "A model architecture based on self-attention.")
        
        # Case insensitive match
        desc2 = service.get_concept_description("transformer architecture")
        self.assertEqual(desc2, "A model architecture based on self-attention.")

    def test_get_concept_description_llm_success(self):
        """Test concept description generation via LLM when not in taxonomy."""
        service = ExtractionService(llm_engine=self.llm_engine)
        self.llm_engine.generate_response.return_value = "Definition of new concept."
        
        desc = service.get_concept_description("New Concept")
        self.assertEqual(desc, "Definition of new concept.")
        self.llm_engine.generate_response.assert_called_once()

    def test_get_concept_description_llm_exception_fallback(self):
        """Test fallback description when LLM fails."""
        service = ExtractionService(llm_engine=self.llm_engine)
        self.llm_engine.generate_response.side_effect = Exception("LLM down")
        
        desc = service.get_concept_description("New Concept")
        self.assertEqual(desc, "A key concept representing 'New Concept' within the AI/ML literature.")

    def test_get_concept_description_no_llm_fallback(self):
        """Test fallback description when no LLM is provided."""
        service = ExtractionService(llm_engine=None)
        desc = service.get_concept_description("New Concept")
        self.assertEqual(desc, "A key concept representing 'New Concept' within the AI/ML literature.")

    def test_generate_summary_success(self):
        """Test generating and saving summary for a Paper."""
        service = ExtractionService(llm_engine=self.llm_engine)
        self.llm_engine.generate_response.return_value = "Summary text"
        
        paper = Paper(id="p1", title="My Paper")
        graph_repo = MagicMock()
        
        summary = service.generate_summary(paper, "Paper text content", graph_repo=graph_repo)
        self.assertEqual(summary, "Summary text")
        self.assertEqual(paper.properties["summary"], "Summary text")
        graph_repo.save_paper.assert_called_once_with(paper)

    def test_generate_summary_no_llm(self):
        """Test generate_summary when llm_engine is None."""
        service = ExtractionService(llm_engine=None)
        paper = Paper(id="p1", title="My Paper")
        
        summary = service.generate_summary(paper, "Paper text content")
        self.assertIsNone(summary)
        self.assertNotIn("summary", paper.properties)

    def test_generate_summary_failure_doesnt_crash(self):
        """Test generate_summary returns None and doesn't crash on exception."""
        service = ExtractionService(llm_engine=self.llm_engine)
        self.llm_engine.generate_response.side_effect = Exception("LLM Error")
        paper = Paper(id="p1", title="My Paper")
        
        summary = service.generate_summary(paper, "Paper text")
        self.assertIsNone(summary)
        self.assertNotIn("summary", paper.properties)

    def test_regex_extraction_no_matches(self):
        """Test regex extraction returns empty lists when text doesn't contain keywords."""
        service = ExtractionService(llm_engine=None)
        res = service.extract("Arbitrary title", "Random abstract", "Nothing related to the taxonomy.")
        self.assertEqual(res.concepts, [])
        self.assertEqual(res.tags, [])

    @patch.dict(os.environ, {"SCIENCE_GRAPH_USE_CLOUD": "0"})
    def test_default_semaphore_limit_local(self):
        """Test default semaphore limit is 1 for local models when pool size is not set."""
        class MockMlxEngine:
            use_cloud = False
        
        with patch("src.services.extraction_service.config") as mock_config:
            mock_config.llm_provider = "mlx"
            mock_config.llm_chunk_pool_size = 4
            service = ExtractionService(llm_engine=MockMlxEngine())
            self.assertEqual(service.semaphore._value, 1)

    def test_default_semaphore_limit_cloud(self):
        """Test default semaphore limit is 50 for cloud models when pool size is not set."""
        class OpenAILLMEngine:
            use_cloud = True

        with patch("src.services.extraction_service.config") as mock_config:
            mock_config.llm_provider = "openai"
            mock_config.llm_chunk_pool_size = 4
            service = ExtractionService(llm_engine=OpenAILLMEngine())
            self.assertEqual(service.semaphore._value, 50)

    @patch("src.services.extraction_service.con")
    def test_call_llm_extract_async_message_inside_semaphore(self, mock_con):
        """Test that _call_llm_extract_async prints the message inside the semaphore."""
        import asyncio
        service = ExtractionService(llm_engine=self.llm_engine)
        
        async def mock_extract(text):
            return {"concepts": []}
        self.llm_engine.extract_concepts_and_metadata_async = mock_extract

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(service._call_llm_extract_async("some text", message="Hello World"))
        finally:
            loop.close()

        mock_con.dim.assert_called_once_with("Hello World")
