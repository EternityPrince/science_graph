import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException

from src.indexer import Indexer, DuplicateDocumentError
from src.models import Paper, Chunk
from src.repository.sqlite_impl import SQLiteGraphRepository, SQLiteVectorRepository
from src.web_app import upload_file

class TestDuplicateDetection(unittest.TestCase):
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

    def test_duplicate_exact_id(self):
        # 1. Save an initial paper
        paper1 = Paper(
            id="test-paper-1",
            title="Initial Title",
            authors=["Alice Smith"],
            doi="10.1000/xyz123",
            abstract="Some abstract text that is unique.",
            properties={"content_hash": "somehash"}
        )
        self.graph_repo.save_paper(paper1)

        # 2. Try to index another paper with the exact same ID
        paper2 = Paper(
            id="test-paper-1",
            title="A completely different title",
            authors=["Bob Jones"],
            doi="10.1000/different",
            abstract="Different abstract text."
        )
        
        with self.assertRaises(DuplicateDocumentError) as ctx:
            self.indexer._run_pipeline(
                paper=paper2,
                full_text="Different body text",
                refs_or_links=[],
                is_markdown=True,
                needs_enrichment=False,
                archive_fn=None
            )
        self.assertEqual(ctx.exception.duplicate_paper_id, "test-paper-1")
        self.assertIn("exact_id", str(ctx.exception))

    def test_duplicate_doi(self):
        # 1. Save initial paper
        paper1 = Paper(
            id="test-paper-1",
            title="Initial Title",
            authors=["Alice Smith"],
            doi="10.1000/xyz123",
            abstract="Some abstract text that is unique.",
            properties={"content_hash": "somehash"}
        )
        self.graph_repo.save_paper(paper1)

        # 2. Try to index a paper with a different ID but same DOI
        paper2 = Paper(
            id="test-paper-2",
            title="A different title",
            authors=["Bob Jones"],
            doi="10.1000/xyz123",
            abstract="Different abstract text."
        )
        
        with self.assertRaises(DuplicateDocumentError) as ctx:
            self.indexer._run_pipeline(
                paper=paper2,
                full_text="Different body text",
                refs_or_links=[],
                is_markdown=True,
                needs_enrichment=False,
                archive_fn=None
            )
        self.assertEqual(ctx.exception.duplicate_paper_id, "test-paper-1")
        self.assertIn("doi", str(ctx.exception))

    def test_duplicate_content_hash(self):
        # 1. Index a paper
        paper1 = Paper(
            id="test-paper-1",
            title="Initial Title",
            authors=["Alice Smith"],
            doi="10.1000/xyz123",
            abstract="Some abstract text.",
        )
        self.indexer._run_pipeline(
            paper=paper1,
            full_text="This is the exact same body content that will match the hash.",
            refs_or_links=[],
            is_markdown=True,
            needs_enrichment=False,
            archive_fn=None
        )

        # 2. Index another paper with a different ID and title, but same body content
        paper2 = Paper(
            id="test-paper-2",
            title="A completely different title",
            authors=["Bob Jones"],
            doi="10.1000/different",
            abstract="Different abstract text."
        )
        
        with self.assertRaises(DuplicateDocumentError) as ctx:
            self.indexer._run_pipeline(
                paper=paper2,
                full_text="This is the exact same body content that will match the hash.",
                refs_or_links=[],
                is_markdown=True,
                needs_enrichment=False,
                archive_fn=None
            )
        self.assertEqual(ctx.exception.duplicate_paper_id, "test-paper-1")
        self.assertIn("content_hash", str(ctx.exception))

    def test_duplicate_title_author_similarity(self):
        # 1. Save a paper
        paper1 = Paper(
            id="test-paper-1",
            title="Deep Learning in Healthcare",
            authors=["Alice Smith", "Bob Jones"],
            doi="10.1000/111",
            abstract="Healthcare ML stuff.",
        )
        self.graph_repo.save_paper(paper1)

        # 2. Index a paper with similar authors and exact title (case-insensitive)
        paper2 = Paper(
            id="test-paper-2",
            title="deep learning in healthcare",
            authors=["Alice Smith", "Charlie Brown"],  # Jaccard = 1/3 = 0.333 > 0.3
            doi="10.1000/222",
            abstract="Different abstract text."
        )
        
        with self.assertRaises(DuplicateDocumentError) as ctx:
            self.indexer._run_pipeline(
                paper=paper2,
                full_text="Totally different body content here.",
                refs_or_links=[],
                is_markdown=True,
                needs_enrichment=False,
                archive_fn=None
            )
        self.assertEqual(ctx.exception.duplicate_paper_id, "test-paper-1")
        self.assertIn("title_author_similarity", str(ctx.exception))

    def test_duplicate_shingles_similarity(self):
        # 1. Index a paper
        paper1 = Paper(
            id="test-paper-1",
            title="Deep Learning in Healthcare",
            authors=["Alice Smith"],
            doi="10.1000/111",
            abstract="Healthcare ML stuff.",
        )
        text1 = "This is a long body text describing machine learning techniques in clinical diagnostics and prediction models."
        self.indexer._run_pipeline(
            paper=paper1,
            full_text=text1,
            refs_or_links=[],
            is_markdown=True,
            needs_enrichment=False,
            archive_fn=None
        )

        # 2. Index a paper with a highly overlapping text (>= 70% 3-word shingle similarity)
        # We share author Alice Smith to make it a candidate, but mock search_similar_chunks to return []
        # to avoid early embedding match.
        paper2 = Paper(
            id="test-paper-2",
            title="A completely different title",
            authors=["Alice Smith"],
            doi="10.1000/222",
            abstract="Different abstract text."
        )
        # Small changes but keeping most of the text (e.g. changing 1-2 words)
        text2 = "This is a long body text describing machine learning techniques in clinical diagnostics and prediction systems."
        
        with patch.object(self.vector_repo, "search_similar_chunks", return_value=[]):
            with self.assertRaises(DuplicateDocumentError) as ctx:
                self.indexer._run_pipeline(
                    paper=paper2,
                    full_text=text2,
                    refs_or_links=[],
                    is_markdown=True,
                    needs_enrichment=False,
                    archive_fn=None
                )
            self.assertEqual(ctx.exception.duplicate_paper_id, "test-paper-1")
            self.assertIn("shingle_similarity", str(ctx.exception))

    def test_duplicate_embedding_similarity(self):
        # 1. Index a paper (chunks get saved in SQLiteVectorRepository)
        paper1 = Paper(
            id="test-paper-1",
            title="Title A",
            authors=["Alice Smith"],
            doi="10.1000/111",
            abstract="Abstract A",
        )
        text1 = "Chunk contents that will be hashed and embedded and are longer than fifty characters to avoid chunk size filters."
        self.indexer._run_pipeline(
            paper=paper1,
            full_text=text1,
            refs_or_links=[],
            is_markdown=True,
            needs_enrichment=False,
            archive_fn=None
        )

        # Now, mock search_similar_chunks to return a chunk from test-paper-1 with similarity 0.99
        # Make sure the full_text of paper2 is slightly different to avoid matching content_hash,
        # but word Jaccard similarity is >= 0.80 (e.g. adding one word)
        mock_chunk = Chunk(
            id="test-paper-1#0",
            paper_id="test-paper-1",
            text_content="Chunk contents that will be hashed and embedded and are longer than fifty characters to avoid chunk size filters.",
            page_number=1,
            embedding=[0.1, 0.2, 0.3]
        )
        
        with patch.object(self.vector_repo, "search_similar_chunks", return_value=[(mock_chunk, 0.99)]):
            paper2 = Paper(
                id="test-paper-2",
                title="Title B",
                authors=["Bob Jones"],
                doi="10.1000/222",
                abstract="Abstract B",
            )
            with self.assertRaises(DuplicateDocumentError) as ctx:
                self.indexer._run_pipeline(
                    paper=paper2,
                    full_text="Chunk contents that will be hashed and embedded and are longer than fifty characters to avoid chunk size filters. Extra word.",
                    refs_or_links=[],
                    is_markdown=True,
                    needs_enrichment=False,
                    archive_fn=None
                )
            self.assertEqual(ctx.exception.duplicate_paper_id, "test-paper-1")
            self.assertIn("embedding_similarity", str(ctx.exception))

    def test_placeholder_not_flagged(self):
        # 1. Save a placeholder paper (placeholder=True)
        paper1 = Paper(
            id="test-paper-1",
            title="Initial Title",
            authors=["Alice Smith"],
            doi="10.1000/xyz123",
            abstract="Some abstract text that is unique.",
            properties={"placeholder": True}
        )
        self.graph_repo.save_paper(paper1)

        # 2. Try to index another paper with the same ID, DOI or content hash.
        # It should succeed because paper1 is a placeholder!
        paper2 = Paper(
            id="test-paper-1",
            title="A completely different title",
            authors=["Bob Jones"],
            doi="10.1000/xyz123",
            abstract="Different abstract text."
        )
        
        paper_id = self.indexer._run_pipeline(
            paper=paper2,
            full_text="Some body text",
            refs_or_links=[],
            is_markdown=True,
            needs_enrichment=False,
            archive_fn=None
        )
        self.assertEqual(paper_id, "test-paper-1")
        # Verify it is no longer a placeholder
        updated = self.graph_repo.get_paper("test-paper-1")
        self.assertFalse(updated.properties.get("placeholder"))

    def test_web_app_duplicate_handling(self):
        # Verify that upload_file and index_url_route endpoints raise HTTPException 409
        # when a DuplicateDocumentError is raised by the indexer.
        from unittest.mock import AsyncMock
        
        # Test upload_file endpoint
        mock_file = MagicMock()
        mock_file.filename = "test.md"
        mock_file.read = AsyncMock(return_value=b"Some content")

        # Mock index_markdown to raise DuplicateDocumentError
        with patch("src.indexer.Indexer.index_markdown", side_effect=DuplicateDocumentError("Duplicate message", "dup-id")):
            with self.assertRaises(HTTPException) as ctx:
                import asyncio
                asyncio.run(upload_file(
                    file=mock_file,
                    graph_repo=self.graph_repo,
                    vector_repo=self.vector_repo,
                    embedding_engine=self.emb_engine,
                    llm_engine=self.llm_engine
                ))
            self.assertEqual(ctx.exception.status_code, 409)
            self.assertIn("Duplicate message", ctx.exception.detail)

    def test_reconstruct_text_non_numeric_chunk_id(self):
        """Test reconstruct_text sort key fallback when chunk id lacks numeric suffix."""
        # 1. Index a paper
        paper1 = Paper(
            id="test-paper-1",
            title="Title A",
            authors=["Alice Smith"],
            doi="10.1000/111",
            abstract="Abstract A",
        )
        self.indexer._run_pipeline(
            paper=paper1,
            full_text="Chunk content that will be matched by shingles or reconstruction.",
            refs_or_links=[],
            is_markdown=True,
            needs_enrichment=False,
            archive_fn=None
        )

        # 2. Mock chunk with non-numeric ID suffix
        mock_chunk = Chunk(
            id="test-paper-1#non_numeric",
            paper_id="test-paper-1",
            text_content="Reconstructed chunk text content that is long enough to pass the length check of fifty characters.",
            page_number=1
        )
        
        with patch.object(self.vector_repo, "get_chunks_for_paper", return_value=[mock_chunk]):
            from src.services.duplicate_detector import DuplicateDetector
            detector = DuplicateDetector(self.graph_repo, self.vector_repo, self.emb_engine)
            # Reconstruct cache should handle the sort fallback and return the chunk text
            paper2 = Paper(id="test-paper-2", title="Title B", authors=["Alice Smith"])
            # Run shingle match which triggers reconstruction
            res = detector.detect_duplicate(paper2, "Reconstructed chunk text content that is long enough to pass the length check of fifty characters.")
            # Should match
            self.assertEqual(res, ("test-paper-1", "shingle_similarity"))

    def test_duplicate_first_chunk_text_fallback(self):
        """Test that first_chunk_text falls back to full_text and title when chunks list is empty."""
        from src.services.duplicate_detector import DuplicateDetector
        detector = DuplicateDetector(self.graph_repo, self.vector_repo, self.emb_engine)
        
        paper = Paper(id="test-paper-1", title="Title A", authors=["Alice Smith"])
        
        # Mock _split_text_to_chunks_raw to return empty list
        with patch("src.services.duplicate_detector._split_text_to_chunks_raw", return_value=[]):
            # Also mock search_similar_chunks to return nothing to ensure it doesn't match early
            with patch.object(self.vector_repo, "search_similar_chunks", return_value=[]):
                # Scenario A: falls back to full_text
                res = detector.detect_duplicate(paper, "Short body text.")
                self.assertIsNone(res)

                # Scenario B: falls back to paper title because full_text is empty
                res2 = detector.detect_duplicate(paper, "")
                self.assertIsNone(res2)

    def test_duplicate_reconstruction_content_hash(self):
        """Test duplicate detection matches on reconstructed content hash when graph properties lack it."""
        # 1. Index a paper
        paper1 = Paper(
            id="test-paper-1",
            title="Title A",
            authors=["Alice Smith"],
            abstract="Abstract A",
        )
        self.indexer._run_pipeline(
            paper=paper1,
            full_text="This is the original text content reconstructed from chunks.",
            refs_or_links=[],
            is_markdown=True,
            needs_enrichment=False,
            archive_fn=None
        )

        # 2. Remove the content_hash from graph repo database node properties to force reconstruction
        props = self.graph_repo.get_paper("test-paper-1").properties
        if "content_hash" in props:
            del props["content_hash"]
        self.graph_repo.update_node_properties("test-paper-1", props)

        # 3. Try to index another paper with the exact same content, sharing author to make it a candidate
        paper2 = Paper(
            id="test-paper-2",
            title="Title B",
            authors=["Alice Smith"],
        )
        
        # Ensure embedding matching is skipped by returning empty
        with patch.object(self.vector_repo, "search_similar_chunks", return_value=[]):
            with self.assertRaises(DuplicateDocumentError) as ctx:
                self.indexer._run_pipeline(
                    paper=paper2,
                    full_text="This is the original text content reconstructed from chunks.",
                    refs_or_links=[],
                    is_markdown=True,
                    needs_enrichment=False,
                    archive_fn=None
                )
            self.assertEqual(ctx.exception.duplicate_paper_id, "test-paper-1")
            self.assertIn("content_hash", str(ctx.exception))
