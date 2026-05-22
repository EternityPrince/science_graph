import os
import tempfile
import unittest
import textwrap
from unittest.mock import MagicMock, patch
from src.indexer import Indexer
from src.config import Config
from src.models import Paper, Author, Concept, Chunk, slugify
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

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)
        import shutil
        if os.path.exists(self.archive_dir):
            shutil.rmtree(self.archive_dir)
        self.config_patcher.stop()

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

    @patch("src.services.metadata_enricher.MetadataEnricher.enrich")
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

    @patch("src.parsers.pdf_parser.PDFParser.parse")
    @patch("src.indexer.split_text_to_chunks")
    def test_index_pdf_pipeline(self, mock_split_text, mock_pdf_extract):
        """Test full PDF indexing pipeline and archive function."""
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
