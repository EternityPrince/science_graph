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

    def test_semaphore_recreated_on_loop_change(self):
        """Test that the semaphore is recreated when the event loop changes."""
        import asyncio
        service = ExtractionService(llm_engine=self.llm_engine, chunk_pool_size=3)
        
        loop1 = asyncio.new_event_loop()
        try:
            async def get_sem1():
                return service.semaphore
            sem1 = loop1.run_until_complete(get_sem1())
        finally:
            loop1.close()

        loop2 = asyncio.new_event_loop()
        try:
            async def get_sem2():
                return service.semaphore
            sem2 = loop2.run_until_complete(get_sem2())
        finally:
            loop2.close()

        self.assertIsNot(sem1, sem2)

    def test_is_chunk_relevant_no_match(self):
        """If no sponsor/promo pattern matches, should return True immediately."""
        service = ExtractionService(llm_engine=None)
        res = service.is_chunk_relevant("This is a purely scientific discussion about transformers.", "Doc Title")
        self.assertTrue(res)

    @patch("src.services.extraction_service.con")
    def test_is_chunk_relevant_match_no_llm(self, mock_con):
        """If pattern matches but LLM is not available, warn and return False."""
        service = ExtractionService(llm_engine=None)
        res = service.is_chunk_relevant("Subscribe to my channel for more content!", "Doc Title")
        self.assertFalse(res)
        mock_con.warning.assert_called_once()

    @patch("src.services.extraction_service.con")
    def test_is_chunk_relevant_match_llm_relevant(self, mock_con):
        """If pattern matches and LLM says it is relevant, return True."""
        llm = MagicMock()
        llm.generate_json.return_value = '{"relevant": true, "reason": "Mentions subscribe in relevant context"}'
        service = ExtractionService(llm_engine=llm)
        res = service.is_chunk_relevant("Subscribe to my newsletter at link.", "Doc Title")
        self.assertTrue(res)

    @patch("src.services.extraction_service.con")
    def test_is_chunk_relevant_match_llm_irrelevant(self, mock_con):
        """If pattern matches and LLM says it is irrelevant, warn and return False."""
        llm = MagicMock()
        llm.generate_json.return_value = '{"relevant": false, "reason": "Sponsor plug"}'
        service = ExtractionService(llm_engine=llm)
        res = service.is_chunk_relevant("This video is sponsored by Squarespace.", "Doc Title")
        self.assertFalse(res)
        mock_con.warning.assert_called_once()

    @patch("src.services.extraction_service.con")
    def test_is_chunk_relevant_llm_exception(self, mock_con):
        """If LLM raises an exception, return True defensively."""
        llm = MagicMock()
        llm.generate_json.side_effect = Exception("LLM Error")
        service = ExtractionService(llm_engine=llm)
        res = service.is_chunk_relevant("Sponsored by Surfshark.", "Doc Title")
        self.assertTrue(res)
        mock_con.warning.assert_called_once()

    def test_extract_from_text_file_markdown(self):
        service = ExtractionService(llm_engine=None)
        content = "# My Markdown Paper\n\nAbstract of the markdown paper.\n\nMore details."
        res = service.extract_from_text_file(content, "my_file")
        self.assertEqual(res.concepts, [])
        # By default, title is extracted from first line starts with "# "
        # Check abstract limit slicing
        
    def test_extract_from_text_file_no_markdown(self):
        service = ExtractionService(llm_engine=None)
        content = "Line one without markdown.\n\nLine two."
        res = service.extract_from_text_file(content, "stem_name")
        self.assertEqual(res.concepts, [])

    def test_split_text_semantically_empty(self):
        service = ExtractionService(llm_engine=None)
        self.assertEqual(service.split_text_semantically("", 100, 10), [])

    def test_split_text_semantically_basic(self):
        service = ExtractionService(llm_engine=None)
        text = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        # Since no llm_engine, it uses len(t) // 4 for tokens.
        # "Paragraph 1" is 11 chars => 2 tokens.
        chunks = service.split_text_semantically(text, max_chunk_tokens=5, overlap_tokens=1)
        self.assertEqual(len(chunks), 2)  # Should split paragraphs

    def test_split_text_semantically_long_paragraph(self):
        service = ExtractionService(llm_engine=None)
        text = "A" * 100  # 25 tokens in one paragraph
        chunks = service.split_text_semantically(text, max_chunk_tokens=10, overlap_tokens=2)
        # Should split on single newlines
        self.assertTrue(len(chunks) > 0)

    @patch("src.services.extraction_service.config")
    def test_extract_map_reduce_sync(self, mock_config):
        mock_config.llm_extraction_input_limit = 100
        mock_config.taxonomy = self.dummy_tax
        mock_config.llm_chunk_pool_size = 50
        mock_config.llm_provider = "openai"
        
        llm = MagicMock()
        llm.count_tokens.return_value = 120
        llm.extract_concepts_and_metadata.return_value = {
            "authors": ["Alice"],
            "concepts": [{"name": "Transformer Architecture", "description": "Desc"}],
            "tags": ["AI"]
        }
        
        service = ExtractionService(llm_engine=llm)
        res = service.extract("Title", "Abstract", "A long full text that exceeds threshold limit", use_llm=True)
        self.assertTrue(res.via_llm)
        self.assertIn("Alice", res.authors)
        self.assertTrue(llm.count_tokens.called)
        self.assertTrue(llm.extract_concepts_and_metadata.called)

    def test_generate_summary_sync_video(self):
        service = ExtractionService(llm_engine=self.llm_engine)
        self.llm_engine.generate_json.return_value = '{"overview": "Sync Video Overview", "themes": ["theme1"], "outline": ["outline1"]}'
        
        paper = Paper(id="p1", title="Video 1")
        paper.properties["source_type"] = "video"
        graph_repo = MagicMock()
        
        summary = service.generate_summary(paper, "Video Transcript", graph_repo=graph_repo)
        self.assertIn("🎥 Обзор ролика", summary)
        self.assertEqual(paper.properties["video_overview"], "Sync Video Overview")
        graph_repo.save_paper.assert_called_once_with(paper)

    def test_generate_summary_sync_video_exception(self):
        service = ExtractionService(llm_engine=self.llm_engine)
        self.llm_engine.generate_json.side_effect = Exception("JSON error")
        self.llm_engine.generate_response.return_value = "Sync Standard Summary"
        
        paper = Paper(id="p1", title="Video 1")
        paper.properties["source_type"] = "video"
        graph_repo = MagicMock()
        
        summary = service.generate_summary(paper, "Video Transcript", graph_repo=graph_repo)
        self.assertEqual(summary, "Sync Standard Summary")


class TestExtractionServiceAsync(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.llm_engine = MagicMock()

    async def test_classify_citation_intent_async_success(self):
        service = ExtractionService(llm_engine=self.llm_engine)
        self.llm_engine.generate_response_async = MagicMock()
        
        async def mock_gen(prompt, **kwargs):
            return "background"
        self.llm_engine.generate_response.return_value = "BACKGROUND"
        
        intent = await service.classify_citation_intent_async("context", "title")
        self.assertEqual(intent, "BACKGROUND")

    async def test_classify_citation_intent_async_no_llm(self):
        service = ExtractionService(llm_engine=None)
        intent = await service.classify_citation_intent_async("context", "title")
        self.assertEqual(intent, "BACKGROUND")

    async def test_classify_citation_intent_async_exception(self):
        service = ExtractionService(llm_engine=self.llm_engine)
        self.llm_engine.generate_response.side_effect = Exception("Timeout")
        intent = await service.classify_citation_intent_async("context", "title")
        self.assertEqual(intent, "BACKGROUND")

    async def test_generate_summary_async_success(self):
        service = ExtractionService(llm_engine=self.llm_engine)
        self.llm_engine.generate_response.return_value = "Summary response"
        
        paper = Paper(id="p1", title="Paper 1")
        graph_repo = MagicMock()
        
        summary = await service.generate_summary_async(paper, "Full text", graph_repo=graph_repo)
        self.assertEqual(summary, "Summary response")
        self.assertEqual(paper.properties["summary"], "Summary response")
        graph_repo.save_paper.assert_called_once_with(paper)

    async def test_generate_summary_async_no_llm(self):
        service = ExtractionService(llm_engine=None)
        paper = Paper(id="p1", title="Paper 1")
        summary = await service.generate_summary_async(paper, "Full text")
        self.assertIsNone(summary)

    async def test_generate_summary_async_exception(self):
        service = ExtractionService(llm_engine=self.llm_engine)
        self.llm_engine.generate_response.side_effect = Exception("Error")
        paper = Paper(id="p1", title="Paper 1")
        summary = await service.generate_summary_async(paper, "Full text")
        self.assertIsNone(summary)

    async def test_call_llm_extract_async_sync_mock(self):
        from unittest.mock import Mock
        sync_func = Mock()
        sync_func.return_value = {"concepts": []}
        
        llm = MagicMock()
        del llm.extract_concepts_and_metadata_async  # Ensure only sync is present
        llm.extract_concepts_and_metadata = sync_func
        
        service = ExtractionService(llm_engine=llm)
        res = await service._call_llm_extract_async("some text")
        self.assertEqual(res, {"concepts": []})

    async def test_call_llm_generate_async_sync_mock(self):
        from unittest.mock import Mock
        sync_func = Mock()
        sync_func.return_value = "Generated text"
        
        llm = MagicMock()
        del llm.generate_response_async
        llm.generate_response = sync_func
        
        service = ExtractionService(llm_engine=llm)
        res = await service._call_llm_generate_async("prompt")
        self.assertEqual(res, "Generated text")

    @patch("src.services.extraction_service.config")
    async def test_extract_map_reduce_async(self, mock_config):
        mock_config.llm_extraction_input_limit = 100
        mock_config.llm_chunk_pool_size = 50
        mock_config.llm_provider = "openai"
        mock_config.taxonomy = {
            "concepts": {"transformer": "Transformer Architecture"},
            "topics": {"nlp": "Natural Language Processing"}
        }
        
        llm = MagicMock()
        llm.count_tokens.return_value = 120
        
        async def mock_extract_async(text, **kwargs):
            return {
                "authors": ["Bob"],
                "concepts": [{"name": "Attention Mechanism", "description": "Desc"}],
                "tags": ["DL"]
            }
        llm.extract_concepts_and_metadata_async = mock_extract_async
        
        service = ExtractionService(llm_engine=llm)
        res = await service.extract_async("Title", "Abstract", "A long full text that exceeds threshold limit", use_llm=True)
        self.assertTrue(res.via_llm)
        self.assertIn("Bob", res.authors)

    async def test_generate_summary_async_video(self):
        service = ExtractionService(llm_engine=self.llm_engine)
        self.llm_engine.generate_and_validate_json_async = MagicMock()
        
        async def mock_json(prompt, schema_class):
            return '{"overview": "Video Overview", "themes": ["theme1"], "outline": ["outline1"]}'
        self.llm_engine.generate_and_validate_json_async.side_effect = mock_json
        
        paper = Paper(id="p1", title="Video 1")
        paper.properties["source_type"] = "video"
        graph_repo = MagicMock()
        
        summary = await service.generate_summary_async(paper, "Video Transcript", graph_repo=graph_repo)
        self.assertIn("🎥 Обзор ролика", summary)
        self.assertEqual(paper.properties["video_overview"], "Video Overview")
        graph_repo.save_paper.assert_called_once_with(paper)

    async def test_generate_summary_async_video_exception(self):
        service = ExtractionService(llm_engine=self.llm_engine)
        self.llm_engine.generate_and_validate_json_async = MagicMock(side_effect=Exception("JSON error"))
        self.llm_engine.generate_response.return_value = "Standard Summary"
        
        paper = Paper(id="p1", title="Video 1")
        paper.properties["source_type"] = "video"
        graph_repo = MagicMock()
        
        summary = await service.generate_summary_async(paper, "Video Transcript", graph_repo=graph_repo)
        self.assertEqual(summary, "Standard Summary")

    async def test_extract_async_single_success(self):
        llm = MagicMock()
        del llm.extract_concepts_and_metadata
        llm.count_tokens.return_value = 10
        
        async def mock_extract_async(text, **kwargs):
            return {
                "authors": ["Charlie"],
                "concepts": [{"name": "Linear Layer", "description": "Desc"}],
                "tags": ["NN"]
            }
        llm.extract_concepts_and_metadata_async = mock_extract_async
        
        service = ExtractionService(llm_engine=llm)
        res = await service.extract_async("Title", "Abstract", "Short text", use_llm=True)
        self.assertTrue(res.via_llm)
        self.assertIn("Charlie", res.authors)

    async def test_extract_async_single_no_data(self):
        llm = MagicMock()
        del llm.extract_concepts_and_metadata
        llm.count_tokens.return_value = 10
        
        async def mock_extract_async(text, **kwargs):
            return None
        llm.extract_concepts_and_metadata_async = mock_extract_async
        
        service = ExtractionService(llm_engine=llm)
        res = await service.extract_async("Title", "Abstract", "Short text", use_llm=True)
        # Should fallback to regex and not crash
        self.assertFalse(res.via_llm)

    async def test_extract_async_single_exception(self):
        llm = MagicMock()
        del llm.extract_concepts_and_metadata
        llm.count_tokens.return_value = 10
        llm.extract_concepts_and_metadata_async = MagicMock(side_effect=Exception("LLM Error"))
        
        service = ExtractionService(llm_engine=llm)
        res = await service.extract_async("Title", "Abstract", "Short text", use_llm=True)
        self.assertFalse(res.via_llm)





