import os
import tempfile
import unittest
import textwrap
import time
import asyncio
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from src.indexer import Indexer, DuplicateDocumentError
from src.config import Config
from src.models import Paper, Chunk, slugify
from src.repository.sqlite_impl import SQLiteGraphRepository, SQLiteVectorRepository

class TestIndexerPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.graph_repo = SQLiteGraphRepository(self.tmp.name)
        self.vector_repo = SQLiteVectorRepository(self.tmp.name)
        
        self.emb_engine = MagicMock()
        self.emb_engine.get_embeddings.side_effect = lambda texts: [[0.1, 0.2, 0.3] for _ in texts]
        
        self.llm_engine = MagicMock()
        self.llm_engine.generate_response.return_value = "Mock LLM Summary Response"
        self.llm_engine.extract_concepts_and_metadata.return_value = None
        
        self.indexer = Indexer(self.graph_repo, self.vector_repo, self.emb_engine, self.llm_engine)
        
        from src.config import config
        # Override config directories to avoid polluting main user config/data dirs
        self.archive_dir = tempfile.mkdtemp()
        self.config_patcher = patch.dict(config.data, {"archive_dir": self.archive_dir})
        self.config_patcher.start()

        # Mock metadata enricher globally for tests to run offline
        self.enrich_patcher = patch("src.services.metadata_enricher.MetadataEnricher.enrich", return_value=None)
        self.enrich_async_patcher = patch("src.services.metadata_enricher.MetadataEnricher.enrich_async", new_callable=AsyncMock)
        self.enrich_patcher.start()
        mock_enrich_async = self.enrich_async_patcher.start()
        mock_enrich_async.return_value = None

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)
        import shutil
        if os.path.exists(self.archive_dir):
            shutil.rmtree(self.archive_dir)
        self.config_patcher.stop()
        self.enrich_patcher.stop()
        self.enrich_async_patcher.stop()

    def _write_file(self, content: str, suffix: str = ".md") -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8")
        f.write(content)
        f.close()
        return f.name

    def test_run_pipeline_basic(self):
        """Test the private _run_pipeline with standard non-enrichment path."""
        paper = Paper(id="p1", title="Test Pipeline Paper", authors=["Alice Jones"], year=2025)
        full_text = "This is some standard text content that is long enough to generate chunks and extract concepts."
        
        # Setup dummy taxonomy to trigger concepts
        dummy_tax = {
            "concepts": {"pipeline": "Pipeline Concept"},
            "topics": {"test": "Test Topic"},
            "descriptions": {"Pipeline Concept": "A concept about pipelines."}
        }
        with patch.object(Config, "taxonomy", new=dummy_tax):
            paper_id = self.indexer._run_pipeline(
                paper=paper,
                full_text=full_text,
                refs_or_links=[],
                is_markdown=False,
                needs_enrichment=False,
                archive_fn=None
            )
            
        self.assertEqual(paper_id, "p1")
        
        # Verify Paper stored
        stored_paper = self.graph_repo.get_paper("p1")
        self.assertIsNotNone(stored_paper)
        self.assertEqual(stored_paper.title, "Test Pipeline Paper")
        
        # Verify Author node and AUTHORED edge
        author_id = slugify("Alice Jones")
        stored_author = self.graph_repo.get_author(author_id)
        self.assertIsNotNone(stored_author)
        self.assertEqual(stored_author.name, "Alice Jones")
        self.assertEqual(len(self.graph_repo.get_all_edges()), 3) # AUTHORED, MENTIONS_CONCEPT, HAS_TAG

    @patch("src.services.metadata_enricher.MetadataEnricher.enrich_async")
    def test_run_pipeline_with_enrichment(self, mock_enrich):
        """Test pipeline when enrichment is requested and succeeds."""
        mock_enrich.return_value = {
            "title": "Enriched Paper Title",
            "authors": ["Enriched Author"],
            "year": 2026,
            "doi": "10.1234/enriched",
            "references": [{"title": "Cited Ref", "doi": "10.9999/ref"}],
            "citations": []
        }
        
        paper = Paper(id="p1", title="Original Title", authors=[], year=None)
        
        # Setup dummy taxonomy
        dummy_tax = {"concepts": {}, "topics": {}, "descriptions": {}}
        with patch.object(Config, "taxonomy", new=dummy_tax):
            self.indexer._run_pipeline(
                paper=paper,
                full_text="Some plain text for chunking.",
                refs_or_links=[],
                is_markdown=False,
                needs_enrichment=True,
                archive_fn=None
            )
            
        # Verify paper updated with enriched info
        stored_paper = self.graph_repo.get_paper("p1")
        self.assertEqual(stored_paper.title, "Enriched Paper Title")
        self.assertEqual(stored_paper.authors, ["Enriched Author"])
        self.assertEqual(stored_paper.year, 2026)
        self.assertEqual(stored_paper.doi, "10.1234/enriched")
        
        # Verify placeholder created for the reference
        ref_id = slugify("10.9999/ref")
        placeholder = self.graph_repo.get_paper(ref_id)
        self.assertIsNotNone(placeholder)
        self.assertTrue(placeholder.properties.get("is_placeholder"))
        self.assertEqual(placeholder.title, "Cited Ref")

    def test_index_markdown_wiki_links_existing(self):
        """Test markdown indexing when wiki-link target already exists as a Paper node."""
        # Pre-create target paper using slugified title as ID
        target_paper = Paper(id="target_paper", title="Target Paper")
        self.graph_repo.save_paper(target_paper)
        
        md_content = textwrap.dedent("""\
            ---
            title: "Source Note"
            ---
            Refer to the [[Target Paper]] for more details.
            """)
        path = self._write_file(md_content)
        
        try:
            self.indexer.index_markdown(path)
            
            # Check edge created between source and target
            edges = self.graph_repo.get_all_edges()
            # Expect: source -> target edge of type RELATED_TO
            related_edges = [e for e in edges if e[2] == "RELATED_TO"]
            self.assertEqual(len(related_edges), 1)
            self.assertEqual(related_edges[0][0], "source_note")
            self.assertEqual(related_edges[0][1], "target_paper")
        finally:
            os.unlink(path)

    def test_index_markdown_wiki_links_missing_placeholder(self):
        """Test markdown indexing when wiki-link target is missing: it creates a concept placeholder."""
        md_content = textwrap.dedent("""\
            ---
            title: "Source Note"
            ---
            Refer to the [[Missing Note Target]] for details.
            """)
        path = self._write_file(md_content)
        
        try:
            self.indexer.index_markdown(path)
            
            # Check placeholder paper created
            target_id = slugify("Missing Note Target")
            placeholder = self.graph_repo.get_paper(target_id)
            self.assertIsNotNone(placeholder)
            self.assertEqual(placeholder.title, "Missing Note Target")
            self.assertTrue(placeholder.properties.get("is_placeholder"))
            
            # Check edge created
            edges = self.graph_repo.get_all_edges()
            related_edges = [e for e in edges if e[2] == "RELATED_TO"]
            self.assertEqual(len(related_edges), 1)
            self.assertEqual(related_edges[0][0], "source_note")
            self.assertEqual(related_edges[0][1], target_id)
        finally:
            os.unlink(path)

    def test_placeholder_upgrade_on_save(self):
        """Test that saving a full paper over a placeholder upgrades it without duplicate nodes."""
        # 1. Create a placeholder
        placeholder = Paper(id="p1", title="Placeholder Title", authors=[], year=None)
        placeholder.properties["is_placeholder"] = True
        self.graph_repo.save_paper(placeholder)
        
        # Verify it is stored
        p = self.graph_repo.get_paper("p1")
        self.assertTrue(p.properties.get("is_placeholder"))
        
        # 2. Save full paper with same ID
        full_paper = Paper(id="p1", title="Full Upgraded Title", authors=["Alice"], year=2024)
        self.graph_repo.save_paper(full_paper)
        
        # Verify it upgraded and placeholder flag is gone
        p2 = self.graph_repo.get_paper("p1")
        self.assertEqual(p2.title, "Full Upgraded Title")
        self.assertEqual(p2.authors, ["Alice"])
        self.assertEqual(p2.year, 2024)
        self.assertFalse(p2.properties.get("is_placeholder", False))

    @patch("src.external_api.fetch_paper_metadata")
    @patch("src.parsers.pdf_parser.PDFParser.parse")
    @patch("src.indexer.split_text_to_chunks")
    def test_index_pdf_pipeline(self, mock_split_text, mock_pdf_extract, mock_fetch):
        """Test full PDF indexing pipeline and archive function."""
        mock_fetch.return_value = None
        # Mock PyMuPDF-based text and reference extraction
        mock_pdf_extract.return_value = (
            Paper(id="pdf_sha", title="Mocked PDF Title", authors=["PDF Author"]),
            ["Raw citation text 1"],
            "This is the PDF full text body content for test."
        )
        mock_split_text.return_value = [
            Chunk(id="pdf_sha#0", paper_id="pdf_sha", text_content="This is the PDF full text body content for test.", page_number=1)
        ]
        
        pdf_path = self._write_file("dummy pdf data", suffix=".pdf")
        
        try:
            # We mock the archive file compression function to prevent actual file writes/compression failures
            with patch.object(self.indexer, "_archive_pdf") as mock_archive:
                paper_id = self.indexer.index_pdf(pdf_path)
                self.assertEqual(paper_id, "pdf_sha")
                
                # Check paper saved
                p = self.graph_repo.get_paper("pdf_sha")
                self.assertEqual(p.title, "Mocked PDF Title")
                
                # Check archive called
                mock_archive.assert_called_once()
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    @patch("src.parsers.epub_parser.EPUBParser.parse")
    def test_index_epub_pipeline(self, mock_epub_parse):
        """Test full EPUB indexing pipeline."""
        mock_epub_parse.return_value = (
            Paper(id="epub_id", title="Mocked EPUB Book", authors=["EPUB Author"]),
            [],
            "EPUB book text content goes here."
        )
        
        epub_path = self._write_file("dummy epub content", suffix=".epub")
        
        try:
            paper_id = self.indexer.index_epub(epub_path)
            self.assertEqual(paper_id, "epub_id")
            
            p = self.graph_repo.get_paper("epub_id")
            self.assertEqual(p.title, "Mocked EPUB Book")
        finally:
            if os.path.exists(epub_path):
                os.unlink(epub_path)

    @patch("src.parsers.url_parser.UrlParser.parse")
    def test_index_url_pipeline(self, mock_url_parse):
        """Test URL indexing pipeline and archiving."""
        mock_url_parse.return_value = (
            Paper(id="url_hash", title="Webpage Title", authors=["Web Author"]),
            [],
            "Webpage body content goes here."
        )
        
        paper_id = self.indexer.index_url("https://example.com/ml-paper")
        self.assertEqual(paper_id, "url_hash")
        
        # Verify archive created
        archive_file = os.path.join(self.archive_dir, "url_hash.md")
        self.assertTrue(os.path.exists(archive_file))
        
        p = self.graph_repo.get_paper("url_hash")
        self.assertEqual(p.title, "Webpage Title")
        self.assertEqual(p.file_path, archive_file)

    def test_reindex_metadata_flow(self):
        """Test reindexing metadata refreshes graph edges/authors without recreating chunks."""
        # 1. Index a paper
        paper = Paper(id="p1", title="Original Title", authors=["Alice"], year=2024)
        self.graph_repo.save_paper(paper)
        
        # Add original authored edge
        self.graph_repo.add_edge("alice", "p1", "AUTHORED")
        
        # Add a mock chunk in vector store
        chunk = Chunk(id="p1#0", paper_id="p1", text_content="Chunk content", page_number=1, embedding=[0.1, 0.2, 0.3])
        self.vector_repo.save_chunks([chunk])
        
        # 2. Mock URL or enrichment
        with patch.object(self.indexer._enricher, "enrich") as mock_enrich:
            mock_enrich.return_value = {
                "title": "New Enriched Title",
                "authors": ["Alice", "Bob"],  # added Bob
                "year": 2025,
                "doi": "10.1234/new"
            }
            
            # Setup dummy taxonomy to trigger concepts
            dummy_tax = {"concepts": {}, "topics": {}, "descriptions": {}}
            with patch.object(Config, "taxonomy", new=dummy_tax):
                success = self.indexer.reindex_metadata("p1")
                self.assertTrue(success)
                
        # 3. Verify graph node updated
        updated_paper = self.graph_repo.get_paper("p1")
        self.assertEqual(updated_paper.title, "New Enriched Title")
        self.assertEqual(updated_paper.authors, ["Alice", "Bob"])
        self.assertEqual(updated_paper.year, 2025)
        
        # Verify AUTHORED edges refreshed (should be 2 now: Alice and Bob)
        edges = self.graph_repo.get_all_edges()
        authored_edges = [e for e in edges if e[2] == "AUTHORED"]
        self.assertEqual(len(authored_edges), 2)
        
        # Verify chunk in vector store was NOT deleted/recreated
        self.assertEqual(len(self.vector_repo.get_chunks_for_paper("p1")), 1)

    @patch("src.indexer.Indexer.index_markdown")
    def test_reindex_full_flow(self, mock_index_markdown):
        """Test reindex_full deletes node and calls parser index methods again."""
        paper = Paper(id="p1", title="Original Title", file_path="my_note.md")
        self.graph_repo.save_paper(paper)
        
        # Verify paper exists
        self.assertIsNotNone(self.graph_repo.get_paper("p1"))
        
        # Call reindex_full
        success = self.indexer.reindex_full("p1")
        self.assertTrue(success)
        
        # Node should be deleted
        self.assertIsNone(self.graph_repo.get_paper("p1"))
        
        # parser/indexer method should have been called
        mock_index_markdown.assert_called_once_with("my_note.md")

    def test_index_batch_basic(self):
        """Test index_batch with multiple markdown files."""
        md1 = self._write_file("# Paper One\nThis is content for paper one.", suffix=".md")
        md2 = self._write_file("# Paper Two\nThis is content for paper two.", suffix=".md")

        # Mock LLM calls
        self.llm_engine.extract_concepts_and_metadata_async = MagicMock()
        self.llm_engine.extract_concepts_and_metadata_async.side_effect = lambda text: {
            "authors": ["Alice"],
            "concepts": [{"name": "Batch Ingestion", "description": "Processing in batches"}],
            "tags": ["batch"]
        }
        self.llm_engine.generate_response = MagicMock(return_value="Mocked batch summary")
        self.llm_engine.generate_response_async = MagicMock()
        self.llm_engine.generate_response_async.side_effect = lambda prompt, **kwargs: "Mocked batch summary"

        # Execute batch indexing
        session_traces = self.indexer.index_batch([md1, md2], use_llm=True)
        self.assertEqual(len(session_traces), 2)
        self.assertTrue(all(trace["success"] for trace in session_traces))

        # Check DB
        p1 = self.graph_repo.get_paper(slugify("Paper One"))
        p2 = self.graph_repo.get_paper(slugify("Paper Two"))
        self.assertIsNotNone(p1)
        self.assertIsNotNone(p2)
        self.assertEqual(p1.properties.get("summary"), "Mocked batch summary")
        self.assertEqual(p2.properties.get("summary"), "Mocked batch summary")

    def test_index_batch_duplicate(self):
        """Test index_batch skips duplicate files early."""
        md1 = self._write_file("# Paper One\nThis is duplicate content.", suffix=".md")
        md2 = self._write_file("# Paper Two\nThis is unique content.", suffix=".md")

        # Mock LLM calls
        self.llm_engine.extract_concepts_and_metadata_async = MagicMock()
        self.llm_engine.extract_concepts_and_metadata_async.side_effect = lambda text: {
            "authors": ["Alice"],
            "concepts": [],
            "tags": []
        }
        self.llm_engine.generate_response = MagicMock(return_value="Mocked summary")
        self.llm_engine.generate_response_async = MagicMock()
        self.llm_engine.generate_response_async.side_effect = lambda prompt, **kwargs: "Mocked summary"

        # Index md1 first
        self.indexer.index_batch([md1], use_llm=True)

        # Index both. md1 should be skipped, md2 should succeed.
        session_traces = self.indexer.index_batch([md1, md2], use_llm=True)
        self.assertEqual(len(session_traces), 2)
        
        trace1 = next(t for t in session_traces if t["name"] == os.path.basename(md1))
        trace2 = next(t for t in session_traces if t["name"] == os.path.basename(md2))

        self.assertFalse(trace1["success"])
        self.assertTrue(trace1.get("skipped_duplicate"))
        self.assertTrue(trace2["success"])

    def test_index_batch_chunk_pool_concurrency(self):
        """Test index_batch sets and respects chunk pool semaphore size."""
        self.indexer.index_batch([], use_llm=False, chunk_pool_size=5)
        self.assertEqual(self.indexer._extractor.semaphore._value, 5)

    def test_duplicate_document_error(self):
        """Test DuplicateDocumentError properties."""
        err = DuplicateDocumentError("error message", "dup_id_123")
        self.assertEqual(str(err), "error message")
        self.assertEqual(err.duplicate_paper_id, "dup_id_123")

    def test_resolve_entity_edge_cases(self):
        """Test resolve_entity for empty, cache init, slug matching, and similarity check branches."""
        # Empty name
        self.assertEqual(self.indexer.resolve_entity("Concept", ""), "")

        # Alias lookup success
        self.graph_repo.get_concept_aliases = MagicMock(return_value={"alias concept": "Canonical Concept"})
        self.assertEqual(self.indexer.resolve_entity("Concept", "alias concept"), slugify("Canonical Concept"))
        
        # Alias lookup failure path (exception)
        self.graph_repo.get_concept_aliases.side_effect = Exception("db error")
        self.indexer._aliases_cache = None
        self.assertEqual(self.indexer.resolve_entity("Concept", "alias concept"), slugify("alias concept"))
        
        # get_nodes_by_label exception path
        self.graph_repo.get_nodes_by_label = MagicMock(side_effect=Exception("DB down"))
        self.assertEqual(self.indexer.resolve_entity("Concept", "Some Concept"), slugify("Some Concept"))

        # Setup nodes for similarity testing
        existing_nodes = [
            ("exact-match", {"name": "Exact Match", "embedding": [0.1, 0.2, 0.3]}),
            ("other-node", {"name": "Other Node", "embedding": [0.5, 0.5, 0.5]}),
            ("no-emb", {"name": "No Emb Node"}),
        ]
        self.graph_repo.get_nodes_by_label = MagicMock(return_value=existing_nodes)
        if hasattr(self.indexer, "_entity_cache"):
            del self.indexer._entity_cache
            
        # 1. Exact slug match
        self.assertEqual(self.indexer.resolve_entity("Concept", "Exact Match"), "exact-match")
        self.assertEqual(self.indexer.resolve_entity("Concept", "exact-match"), "exact-match")

        # 2. Embedding similarity check (sim > 0.95)
        self.emb_engine.get_embedding = MagicMock(return_value=[0.501, 0.501, 0.501])
        del self.indexer._entity_cache
        self.assertEqual(self.indexer.resolve_entity("Concept", "Similar name"), "other-node")

        # Embedding fails / returns invalid shape
        self.emb_engine.get_embedding.side_effect = Exception("embedding failed")
        del self.indexer._entity_cache
        self.assertEqual(self.indexer.resolve_entity("Concept", "Other Node"), "other-node")

        # 3. String similarity fallback (sim > 0.95 ratio)
        del self.indexer._entity_cache
        self.assertEqual(self.indexer.resolve_entity("Concept", "Other Node "), "other-node")

    def test_add_resolved_entity_to_cache(self):
        """Test _add_resolved_entity_to_cache with cache initialization and exception paths."""
        if hasattr(self.indexer, "_entity_cache"):
            del self.indexer._entity_cache
            
        self.graph_repo.get_nodes_by_label = MagicMock(return_value=[])
        self.indexer._add_resolved_entity_to_cache("Concept", "new-id", "New Entity")
        self.assertIn("Concept", self.indexer._entity_cache)
        self.assertEqual(len(self.indexer._entity_cache["Concept"]), 1)
        
        del self.indexer._entity_cache
        self.graph_repo.get_nodes_by_label.side_effect = Exception("DB error")
        self.indexer._add_resolved_entity_to_cache("Concept", "new-id", "New Entity")
        self.assertEqual(len(self.indexer._entity_cache["Concept"]), 1)

    def test_get_citation_context(self):
        """Test citation context extraction helper."""
        text = "This is a sentence. We discuss the paper by Smith (2020) here. That was in 2020."
        
        self.assertEqual(self.indexer._get_citation_context("", "Title"), "")
        self.assertEqual(self.indexer._get_citation_context(text, ""), "")

        ctx = self.indexer._get_citation_context(text, "We discuss the paper")
        self.assertIn("We discuss the paper by Smith", ctx)

        ctx2 = self.indexer._get_citation_context(text, "Random Title", "Smith", 2020)
        self.assertIn("Smith (2020)", ctx2)

        ctx3 = self.indexer._get_citation_context(text, "Random Title", "Smith")
        self.assertIn("Smith (2020)", ctx3)

        self.assertEqual(self.indexer._get_citation_context(text, "Completely different", "Jones", 2021), "")

    def test_trace_stage_context_manager(self):
        """Test measuring trace stages."""
        trace = {}
        with self.indexer._trace_stage("Test Stage", trace):
            time.sleep(0.01)
        self.assertIn("stages", trace)
        self.assertIn("Test Stage", trace["stages"])
        self.assertTrue(trace["stages"]["Test Stage"] > 0)

    def test_index_pdf_file_not_found(self):
        """Test index_pdf raises FileNotFoundError when pdf doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            self.indexer.index_pdf("non_existent_file_path.pdf")

    def test_index_url_archive_write_exception(self):
        """Test index_url handles website archive write exceptions gracefully."""
        self.indexer._extractor.extract_async = AsyncMock(return_value=MagicMock(concepts=[], tags=[], authors=[]))
        self.indexer._extractor.generate_summary_async = AsyncMock(return_value="Summary")
        
        with patch("pathlib.Path.write_text", side_effect=Exception("Disk full")):
            with patch("src.parsers.url_parser.UrlParser.parse") as mock_parse:
                mock_parse.return_value = (
                    Paper(id="url_hash", title="Webpage Title", authors=["Web Author"]),
                    [],
                    "Webpage body content goes here."
                )
                paper_id = self.indexer.index_url("https://example.com/ml-paper")
                self.assertEqual(paper_id, "url_hash")

    def test_run_pipeline_duplicate_document_error(self):
        """Test that detect_duplicate raises DuplicateDocumentError in unified pipeline."""
        paper = Paper(id="p1", title="Duplicate Paper", authors=["Alice"], year=2025)
        self.indexer.detect_duplicate = MagicMock(return_value=("existing_id", "hash_match"))
        
        with self.assertRaises(DuplicateDocumentError) as ctx:
            self.indexer._run_pipeline(paper, "Some body content", [], False, False, None)
        self.assertEqual(ctx.exception.duplicate_paper_id, "existing_id")

    @patch("src.services.metadata_enricher.MetadataEnricher.enrich_async")
    def test_run_pipeline_with_enrichment_failure_and_size_traces(self, mock_enrich):
        """Test metadata enrichment exception/failure path and trace dict tracking."""
        mock_enrich.return_value = None
        
        trace = {}
        tmp_src = self._write_file("dummy source file contents")
        paper = Paper(id="p1", title="Failed Enrichment Paper", authors=[], year=None, file_path=tmp_src)
        try:
            self.indexer._run_pipeline(
                paper=paper,
                full_text="Some plain text for chunking.",
                refs_or_links=[],
                is_markdown=False,
                needs_enrichment=True,
                archive_fn=None,
                source_path=tmp_src,
                trace_info=trace
            )
            self.assertIn("stages", trace)
            self.assertEqual(trace["original_size"], len("dummy source file contents"))
        finally:
            os.unlink(tmp_src)

    def test_reindex_metadata_not_found(self):
        """Test reindex_metadata when paper is not found in database."""
        self.graph_repo.get_paper = MagicMock(return_value=None)
        self.assertFalse(self.indexer.reindex_metadata("non_existent"))

    @patch("src.parsers.url_parser.UrlParser.parse")
    @patch("src.services.metadata_enricher.MetadataEnricher.enrich")
    def test_reindex_metadata_url_reparse_exception(self, mock_enrich, mock_parse):
        """Test reindex_metadata handles URL re-parse exceptions gracefully."""
        mock_enrich.return_value = None
        paper = Paper(id="url_paper", title="Web Title", file_path="https://example.com/web")
        paper.properties["source_type"] = "webpage"
        self.graph_repo.get_paper = MagicMock(return_value=paper)
        mock_parse.side_effect = Exception("Reparse failed")
        
        self.indexer._extractor.extract = MagicMock()
        self.indexer._extractor.extract.return_value = MagicMock(concepts=[], tags=[], authors=[])
        
        self.graph_repo.transaction = MagicMock()
        
        success = self.indexer.reindex_metadata("url_paper")
        self.assertTrue(success)

    def test_reindex_full_error_paths(self):
        """Test reindex_full handles paper/file not found or deletion failures."""
        self.graph_repo.get_paper = MagicMock(return_value=None)
        self.assertFalse(self.indexer.reindex_full("non_existent"))

        paper_no_path = Paper(id="p1", title="No Path Paper")
        self.graph_repo.get_paper = MagicMock(return_value=paper_no_path)
        self.assertFalse(self.indexer.reindex_full("p1"))

        paper_with_path = Paper(id="p1", title="Path Paper", file_path="dummy.pdf")
        self.graph_repo.get_paper = MagicMock(return_value=paper_with_path)
        self.graph_repo.delete_node = MagicMock(side_effect=Exception("delete failed"))
        self.assertFalse(self.indexer.reindex_full("p1"))

    @patch("src.indexer.Indexer.index_url")
    @patch("src.indexer.Indexer.index_pdf")
    @patch("src.indexer.Indexer.index_epub")
    def test_reindex_full_supported_types(self, mock_epub, mock_pdf, mock_url):
        """Test reindex_full routing for different file formats/types."""
        self.graph_repo.delete_node = MagicMock()

        paper_url = Paper(id="p_url", title="URL Paper", file_path="https://example.com")
        self.graph_repo.get_paper = MagicMock(return_value=paper_url)
        self.assertTrue(self.indexer.reindex_full("p_url"))
        mock_url.assert_called_once_with("https://example.com")

        paper_pdf = Paper(id="p_pdf", title="PDF Paper", file_path="dummy.pdf")
        self.graph_repo.get_paper = MagicMock(return_value=paper_pdf)
        self.assertTrue(self.indexer.reindex_full("p_pdf"))
        mock_pdf.assert_called_once_with("dummy.pdf")

        paper_epub = Paper(id="p_epub", title="EPUB Paper", file_path="dummy.epub")
        self.graph_repo.get_paper = MagicMock(return_value=paper_epub)
        self.assertTrue(self.indexer.reindex_full("p_epub"))
        mock_epub.assert_called_once_with("dummy.epub")

        paper_unsupported = Paper(id="p_unsupported", title="Unsupported Paper", file_path="dummy.txt")
        self.graph_repo.get_paper = MagicMock(return_value=paper_unsupported)
        self.assertFalse(self.indexer.reindex_full("p_unsupported"))

    def test_reindex_metadata_batch(self):
        """Test batch re-indexing of metadata with authors/tags filters and limits."""
        p1 = Paper(id="p1", title="Paper 1", authors=[], properties={})
        p2 = Paper(id="p2", title="Paper 2", authors=["Alice"], properties={"tags": []})
        p3 = Paper(id="p3", title="Paper 3", authors=["Alice"], properties={"tags": ["tag"]})
        
        self.graph_repo.get_non_placeholder_paper_ids = MagicMock(return_value=["p1", "p2", "p3"])
        def mock_get_paper(pid):
            if pid == "p1": return p1
            if pid == "p2": return p2
            if pid == "p3": return p3
            return None
        self.graph_repo.get_paper = MagicMock(side_effect=mock_get_paper)
        
        self.indexer.reindex_metadata = MagicMock(return_value=True)

        success, total = self.indexer.reindex_metadata_batch(limit=2)
        self.assertEqual(total, 2)
        self.assertEqual(success, 2)

        success, total = self.indexer.reindex_metadata_batch(missing_authors=True)
        self.assertEqual(total, 1)
        self.indexer.reindex_metadata.assert_any_call("p1", use_llm=False)

        self.indexer.reindex_metadata.reset_mock()
        success, total = self.indexer.reindex_metadata_batch(missing_tags=True)
        self.assertEqual(total, 2)
        self.indexer.reindex_metadata.assert_any_call("p1", use_llm=False)
        self.indexer.reindex_metadata.assert_any_call("p2", use_llm=False)

        self.graph_repo.get_non_placeholder_paper_ids.return_value = []
        success, total = self.indexer.reindex_metadata_batch()
        self.assertEqual(total, 0)
        self.assertEqual(success, 0)

    def test_reindex_full_batch(self):
        """Test batch full re-indexing with paper_id filter and limit."""
        p1 = Paper(id="p1", title="Paper 1", file_path="p1.pdf")
        self.graph_repo.get_paper = MagicMock(return_value=p1)
        self.graph_repo.get_non_placeholder_paper_ids = MagicMock(return_value=["p1", "p2"])
        
        self.indexer.reindex_full = MagicMock(return_value=True)

        success, total = self.indexer.reindex_full_batch(paper_id="p1")
        self.assertEqual(total, 1)
        self.assertEqual(success, 1)
        self.indexer.reindex_full.assert_called_once_with("p1")

        self.indexer.reindex_full.reset_mock()
        success, total = self.indexer.reindex_full_batch(limit=1)
        self.assertEqual(total, 1)
        
        self.graph_repo.get_paper.return_value = None
        with self.assertRaises(ValueError):
            self.indexer.reindex_full_batch(paper_id="non_existent")

        self.graph_repo.get_non_placeholder_paper_ids.return_value = []
        success, total = self.indexer.reindex_full_batch()
        self.assertEqual(total, 0)

    @patch("src.ner_engine.extract_persons_from_text")
    @patch("fitz.open")
    def test_ner_fallback_authors_flow(self, mock_fitz_open, mock_ner_extract):
        """Test _ner_fallback_authors parsing and merging candidates."""
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Author Candidates text context here"
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.__len__.return_value = 1
        mock_fitz_open.return_value = mock_doc
        
        mock_ner_extract.return_value = ["SingleName", "Valid Author Name", "Too Long Name Exceeding Five Words In Total"]
        
        res = self.indexer._ner_fallback_authors(["Existing Author"], "dummy.pdf")
        self.assertIn("Existing Author", res)
        self.assertIn("Valid Author Name", res)
        self.assertNotIn("SingleName", res)

    def test_archive_pdf_compression_flow(self):
        """Test _archive_pdf with compression enabled and fallbacks."""
        from src.config import config
        config.data["pdf_compression_enabled"] = True
        config.data["pdf_compression_dpi_threshold"] = 150
        config.data["pdf_compression_dpi_target"] = 72
        config.data["pdf_compression_quality"] = 50
        
        src = self._write_file("dummy pdf contents", suffix=".pdf")
        archive_dir = tempfile.mkdtemp()
        archive_path = Path(archive_dir) / "archived.pdf"
        
        with patch("src.parsers.pdf_parser.PDFParser.compress_and_save_pdf") as mock_compress:
            def mock_write(input_path, output_path, **kwargs):
                with open(output_path, "w") as f:
                    f.write("compressed contents")
            mock_compress.side_effect = mock_write
            
            self.indexer._archive_pdf(src, archive_path)
            self.assertTrue(archive_path.exists())
            self.assertEqual(archive_path.read_text(), "compressed contents")
            
        if os.path.exists(src): os.unlink(src)
        if archive_path.exists(): os.unlink(archive_path)
            
        src = self._write_file("original contents", suffix=".pdf")
        with patch("src.parsers.pdf_parser.PDFParser.compress_and_save_pdf", side_effect=Exception("Compression error")):
            self.indexer._archive_pdf(src, archive_path)
            self.assertTrue(archive_path.exists())
            self.assertEqual(archive_path.read_text(), "original contents")

        if os.path.exists(src): os.unlink(src)
        if archive_path.exists(): os.unlink(archive_path)
        shutil.rmtree(archive_dir)

    @patch("src.parsers.factory.ParserFactory.get_parser")
    def test_read_local_text_helpers(self, mock_get_parser):
        """Test _read_local_text for PDF, EPUB, and other formats."""
        paper = Paper(id="p1", title="Non Existent PDF", file_path="non_existent.pdf")
        self.assertEqual(self.indexer._read_local_text(paper), "")

        pdf_path = self._write_file("pdf data", suffix=".pdf")
        paper_pdf = Paper(id="p_pdf", title="PDF Paper", file_path=pdf_path)
        mock_parser = MagicMock()
        mock_parser.parse.return_value = (None, None, "Parsed PDF Text")
        mock_get_parser.return_value = mock_parser
        
        try:
            self.assertEqual(self.indexer._read_local_text(paper_pdf), "Parsed PDF Text")
        finally:
            os.unlink(pdf_path)

        epub_path = self._write_file("epub data", suffix=".epub")
        paper_epub = Paper(id="p_epub", title="EPUB Paper", file_path=epub_path)
        mock_parser.parse.return_value = (None, None, "Parsed EPUB Text")
        
        try:
            self.assertEqual(self.indexer._read_local_text(paper_epub), "Parsed EPUB Text")
        finally:
            os.unlink(epub_path)

        txt_path = self._write_file("Parsed Markdown Text", suffix=".md")
        paper_txt = Paper(id="p_txt", title="Markdown Paper", file_path=txt_path)
        try:
            self.assertEqual(self.indexer._read_local_text(paper_txt), "Parsed Markdown Text")
        finally:
            os.unlink(txt_path)

    def test_index_batch_directory_and_unsupported(self):
        """Test index_batch expands directories, checks unsupported types, and URL targets."""
        temp_dir = tempfile.mkdtemp()
        
        pdf_path = os.path.join(temp_dir, "test.pdf")
        md_path = os.path.join(temp_dir, "test.md")
        epub_path = os.path.join(temp_dir, "test.epub")
        unsupported_path = os.path.join(temp_dir, "test.txt")
        
        for p in (pdf_path, md_path, epub_path, unsupported_path):
            with open(p, "w") as f:
                f.write("dummy content")

        self.indexer._extractor.extract_async = AsyncMock(return_value=MagicMock(concepts=[], tags=[], authors=[]))
        self.indexer._extractor.generate_summary_async = AsyncMock(return_value="Summary")
        self.indexer._duplicate_detector.detect_duplicate = MagicMock(return_value=None)
        
        self.emb_engine.get_embeddings = MagicMock(return_value=[[0.1, 0.2, 0.3]] * 3)

        mock_paper = Paper(id="dummy_id", title="Title", authors=["Author"])
        with patch("src.parsers.pdf_parser.PDFParser.parse", return_value=(mock_paper, [], "PDF Body")), \
             patch("src.parsers.epub_parser.EPUBParser.parse", return_value=(mock_paper, [], "EPUB Body")), \
             patch("src.parsers.md_parser.MarkdownParser.parse", return_value=(mock_paper, [], "Markdown Body")), \
             patch("src.parsers.url_parser.UrlParser.parse", return_value=(mock_paper, [], "URL Body")):
                 
            res = self.indexer.index_batch([temp_dir, "https://example.com/some-page"], use_llm=True)
            self.assertEqual(len(res), 4)

        shutil.rmtree(temp_dir)

    def test_build_graph_writes_entities_enrichment(self):
        """Test build_graph_writes_async for all extraction entity kinds (Institution, Dataset, etc.)."""
        paper = Paper(id="paper1", title="Paper Title", authors=["Alice"], properties={"tags": ["tag1"]})
        
        extraction = MagicMock()
        extraction.concepts = [
            {"name": "ConceptA", "description": "DescA"},
            "invalid_concept_format_should_be_skipped",
            {"name": ""}
        ]
        extraction.institutions = ["University A", 123]
        extraction.sponsored_by = ["Sponsor B", 456]
        extraction.author_institutions = [
            {"author": "Alice", "institution": "University A"},
            "invalid_affiliation_format"
        ]
        extraction.datasets = [
            {"name": "Dataset C", "relation": "USED_DATASET"},
            "invalid_dataset"
        ]
        extraction.code_repositories = ["https://github.com/test/repo", 789]
        extraction.journal_or_conference = "Journal of ML"
        extraction.concept_relations = [
            {"source": "ConceptA", "target": "ConceptB", "relation_type": "PREREQUISITE_FOR"},
            "invalid_relation"
        ]

        self.emb_engine.get_embedding = MagicMock()
        def unique_emb(text):
            vec = [0.0] * 128
            # Use character sum or similar stable hash to avoid negative hashes from hash()
            idx = sum(ord(c) for c in text) % 128
            vec[idx] = 1.0
            return vec
        self.emb_engine.get_embedding.side_effect = unique_emb
        self.indexer._extractor.get_concept_description_async = AsyncMock(return_value="concept description")
        self.indexer._classify_cites_edges_async = AsyncMock(return_value=[])

        nodes, edges = asyncio.run(self.indexer._build_graph_writes_async(
            paper=paper,
            extraction=extraction,
            full_text="Body content discussing references.",
            is_markdown=False,
            refs_or_links=["Citation 1"],
            api_references=[{"title": "API Ref", "doi": "10.1234/ref", "authors": ["Author Ref"], "year": 2020}],
            api_citations=[{"title": "API Cit", "doi": "10.1234/cit", "authors": ["Author Cit"], "year": 2021}]
        ))

        node_ids = {n[0] for n in nodes}
        edge_types = {e[2] for e in edges}
        
        self.assertIn("paper1", node_ids)
        self.assertIn("concepta", node_ids)
        self.assertIn("university_a", node_ids)
        self.assertIn("sponsor_b", node_ids)
        self.assertIn("dataset_c", node_ids)
        self.assertIn(slugify("https://github.com/test/repo"), node_ids)
        self.assertIn("journal_of_ml", node_ids)
        
        self.assertIn("AUTHORED", edge_types)
        self.assertIn("MENTIONS_CONCEPT", edge_types)
        self.assertIn("HAS_TAG", edge_types)
        self.assertIn("SPONSORED_BY", edge_types)
        self.assertIn("AFFILIATED_WITH", edge_types)
        self.assertIn("USED_DATASET", edge_types)
        self.assertIn("HAS_CODE", edge_types)
        self.assertIn("PUBLISHED_IN", edge_types)
        self.assertIn("PREREQUISITE_FOR", edge_types)

    def test_build_graph_writes_markdown_notes_relationships(self):
        """Test build_graph_writes_async for markdown note-specific relationships (agrees_with, disagrees_with, comments_on, linked_to)."""
        paper = Paper(
            id="note1",
            title="Note Title",
            properties={
                "source_type": "note",
                "comments_on": ["Target Paper A"],
                "agrees_with": ["Target Paper B"],
                "disagrees_with": ["Target Paper C"],
                "linked_to": ["Concept D"]
            }
        )
        extraction = MagicMock(concepts=[], institutions=[], datasets=[], code_repositories=[], journal_or_conference=None, concept_relations=[])
        self.emb_engine.get_embedding = MagicMock(return_value=[0.1, 0.2, 0.3])
        self.indexer._classify_cites_edges_async = AsyncMock(return_value=[])
        
        nodes, edges = asyncio.run(self.indexer._build_graph_writes_async(
            paper=paper,
            extraction=extraction,
            full_text="Some markdown note body",
            is_markdown=True,
            refs_or_links=["[[Wiki Link Target]]", "https://example.com/external-target"],
            api_references=[],
            api_citations=[]
        ))

        edge_types = {e[2] for e in edges}
        self.assertIn("COMMENTS_ON", edge_types)
        self.assertIn("AGREES_WITH", edge_types)
        self.assertIn("DISAGREES_WITH", edge_types)
        self.assertIn("LINKED_TO", edge_types)
        self.assertIn("RELATED_TO", edge_types)
