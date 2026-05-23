"""Tests for Markdown and EPUB parsers, and the Indexer extensions."""

import os
import tempfile
import textwrap
import unittest
from unittest.mock import patch, MagicMock

from src.models import Paper



class TestMarkdownParser(unittest.TestCase):
    def _write_md(self, content: str) -> str:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        )
        f.write(content)
        f.close()
        return f.name

    def tearDown(self):
        pass  # temp files cleaned per test

    def test_frontmatter_title_authors_year(self):
        path = self._write_md(textwrap.dedent("""\
            ---
            title: "My Research Note"
            authors: "Alice Smith, Bob Jones"
            date: 2024-03-15
            tags:
              - transformer
              - attention
            ---
            # My Research Note

            This is an introduction to attention mechanisms.
            """))
        try:
            from src.parsers.md_parser import MarkdownParser
            paper, links, body = MarkdownParser().parse(path)
            self.assertEqual(paper.title, "My Research Note")
            self.assertIn("Alice Smith", paper.authors)
            self.assertIn("Bob Jones", paper.authors)
            self.assertEqual(paper.year, 2024)
            self.assertIn("transformer", paper.properties["tags"])
            self.assertEqual(paper.properties["source_type"], "note")
        finally:
            os.unlink(path)

    def test_wikilinks_extraction(self):
        path = self._write_md(textwrap.dedent("""\
            # Note on Transformers

            See [[Attention Mechanism]] and [[BERT]] for details.
            Also [[GPT|OpenAI GPT]] is relevant.
            """))
        try:
            from src.parsers.md_parser import MarkdownParser
            _, links, _ = MarkdownParser().parse(path)
            self.assertIn("Attention Mechanism", links)
            self.assertIn("BERT", links)
            self.assertIn("GPT", links)
        finally:
            os.unlink(path)

    def test_inline_tags_extraction(self):
        path = self._write_md(textwrap.dedent("""\
            # Notes

            This relates to #deep-learning and #nlp topics.
            """))
        try:
            from src.parsers.md_parser import MarkdownParser
            paper, _, _ = MarkdownParser().parse(path)
            tags = paper.properties.get("tags", [])
            self.assertIn("deep-learning", tags)
            self.assertIn("nlp", tags)
        finally:
            os.unlink(path)

    def test_title_fallback_to_h1(self):
        path = self._write_md(textwrap.dedent("""\
            # Inferred Title

            Some content here.
            """))
        try:
            from src.parsers.md_parser import MarkdownParser
            paper, _, _ = MarkdownParser().parse(path)
            self.assertEqual(paper.title, "Inferred Title")
        finally:
            os.unlink(path)

    def test_stable_id_generation(self):
        """Same title should produce the same ID."""
        path1 = self._write_md("---\ntitle: Stable Title\n---\nContent 1")
        path2 = self._write_md("---\ntitle: Stable Title\n---\nContent 2")
        try:
            from src.parsers.md_parser import MarkdownParser
            p1, _, _ = MarkdownParser().parse(path1)
            p2, _, _ = MarkdownParser().parse(path2)
            self.assertEqual(p1.id, p2.id)
        finally:
            os.unlink(path1)
            os.unlink(path2)


class TestIndexerMarkdownIntegration(unittest.TestCase):
    """Integration test: index a Markdown note end-to-end (no LLM needed)."""

    def setUp(self):
        import tempfile
        from src.repository.sqlite_impl import SQLiteGraphRepository, SQLiteVectorRepository

        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()

        self.graph_repo = SQLiteGraphRepository(self.tmp.name)
        self.vector_repo = SQLiteVectorRepository(self.tmp.name)

        # Mock embedding engine — returns zero vectors
        self.emb_engine = MagicMock()
        self.emb_engine.get_embeddings.side_effect = lambda texts: [[0.0] * 3 for _ in texts]

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_index_markdown_creates_nodes(self):
        from src.indexer import Indexer

        md_content = textwrap.dedent("""\
            ---
            title: "Test Note"
            authors: "Vladimir Kasterin"
            tags:
              - rag
            ---
            # Test Note

            This note discusses [[RAG]] and [[Transformers]].
            """)

        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        )
        f.write(md_content)
        f.close()

        try:
            indexer = Indexer(self.graph_repo, self.vector_repo, self.emb_engine)
            note_id = indexer.index_markdown(f.name)

            # Verify paper node exists
            paper = self.graph_repo.get_paper(note_id)
            self.assertIsNotNone(paper)
            self.assertEqual(paper.title, "Test Note")

            # Verify concept nodes created from wiki-links
            neighbors = self.graph_repo.get_neighbors(note_id, max_depth=1)
            edge_types = {n[2] for n in neighbors}
            self.assertIn("RELATED_TO", edge_types)  # wiki-links
            self.assertIn("AUTHORED", edge_types)

        finally:
            os.unlink(f.name)

    @patch("src.external_api.fetch_paper_metadata")
    def test_reindex_metadata(self, mock_fetch_metadata):
        from src.indexer import Indexer

        # 1. Setup mock metadata return value
        mock_fetch_metadata.return_value = {
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "year": 2017,
            "abstract": "The dominant sequence transduction models...",
            "doi": "10.1145/37565.37566",
            "references": [{"title": "Deep Residual Learning", "doi": "10.1109/CVPR.2016.90"}],
            "citations": [{"title": "BERT: Pre-training", "doi": "10.18653/v1/N19-1423"}]
        }

        # 2. Insert a basic Paper node to the database first
        paper = Paper(
            id="test_paper_id",
            title="Old Title",
            authors=["Old Author"],
            year=2015,
            abstract="Old abstract",
            doi="10.1145/37565.37566",
            file_path="mock_path.pdf"
        )
        self.graph_repo.save_paper(paper)

        # 3. Call reindex_metadata
        indexer = Indexer(self.graph_repo, self.vector_repo, self.emb_engine)
        success = indexer.reindex_metadata("test_paper_id", use_llm=False)
        self.assertTrue(success)

        # 4. Verify updated Paper in database
        updated_paper = self.graph_repo.get_paper("test_paper_id")
        self.assertIsNotNone(updated_paper)
        self.assertEqual(updated_paper.title, "Attention Is All You Need")
        self.assertEqual(updated_paper.year, 2017)
        self.assertEqual(updated_paper.abstract, "The dominant sequence transduction models...")
        self.assertIn("Ashish Vaswani", updated_paper.authors)
        self.assertIn("Noam Shazeer", updated_paper.authors)

        # Verify old author edge is deleted, new authored edges exist
        neighbors = self.graph_repo.get_neighbors("test_paper_id", max_depth=1)
        # Neighbors format: (node_id, label, edge_type, direction)
        edge_types = [n[2] for n in neighbors]
        self.assertIn("AUTHORED", edge_types)

        # Check references/citations were created
        # The reference/citation papers should be created as placeholders
        from src.models import slugify
        ref_id = slugify("10.1109/CVPR.2016.90")
        cit_id = slugify("10.18653/v1/N19-1423")
        
        ref_paper = self.graph_repo.get_paper(ref_id)
        cit_paper = self.graph_repo.get_paper(cit_id)
        self.assertIsNotNone(ref_paper)
        self.assertIsNotNone(cit_paper)
        self.assertEqual(ref_paper.title, "Deep Residual Learning")
        self.assertEqual(cit_paper.title, "BERT: Pre-training")

    @patch("src.external_api.fetch_paper_metadata")
    def test_reindex_retains_chunks(self, mock_fetch_metadata):
        from src.indexer import Indexer
        from src.models import Chunk

        # 1. Setup mock metadata return value
        mock_fetch_metadata.return_value = {
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "year": 2017,
            "abstract": "The dominant sequence transduction models...",
            "doi": "10.1145/37565.37566"
        }

        # 2. Insert paper
        paper = Paper(
            id="test_paper_id",
            title="Old Title",
            authors=["Old Author"],
            year=2015,
            abstract="Old abstract",
            doi="10.1145/37565.37566",
            file_path="mock_path.pdf"
        )
        self.graph_repo.save_paper(paper)

        # 3. Save chunks for this paper in the vector database
        chunk = Chunk(
            id="test_paper_id#0",
            paper_id="test_paper_id",
            text_content="This is some chunk content.",
            page_number=1,
            embedding=[0.1] * 384
        )
        self.vector_repo.save_chunks([chunk])

        # Verify chunks exist
        all_chunks = self.vector_repo.get_all_chunks()
        self.assertEqual(len(all_chunks), 1)
        self.assertEqual(all_chunks[0].id, "test_paper_id#0")

        # 4. Call reindex_metadata
        indexer = Indexer(self.graph_repo, self.vector_repo, self.emb_engine)
        success = indexer.reindex_metadata("test_paper_id", use_llm=False)
        self.assertTrue(success)

        # 5. Verify updated Paper in database
        updated_paper = self.graph_repo.get_paper("test_paper_id")
        self.assertIsNotNone(updated_paper)
        self.assertEqual(updated_paper.title, "Attention Is All You Need")

        # 6. Verify that chunks are STILL present in the database (were not deleted by cascade)
        all_chunks_after = self.vector_repo.get_all_chunks()
        self.assertEqual(len(all_chunks_after), 1, "Chunks were deleted/lost during metadata reindexing!")
        self.assertEqual(all_chunks_after[0].id, "test_paper_id#0")
    @patch("src.parsers.url_parser.UrlParser.parse")
    def test_index_url_saves_local_copy(self, mock_parse):
        from src.indexer import Indexer
        import os

        # 1. Setup mock parse
        mock_paper = Paper(
            id="test_webpage",
            title="Test Webpage Title",
            authors=["Alice Web"],
            year=2024,
            doi="",
            file_path="https://example.com/webpage",
            properties={"source_type": "webpage", "url": "https://example.com/webpage"}
        )
        mock_parse.return_value = (mock_paper, [], "This is mock webpage content in Markdown.")

        # 2. Call index_url
        indexer = Indexer(self.graph_repo, self.vector_repo, self.emb_engine)
        paper_id = indexer.index_url("https://example.com/webpage")

        self.assertEqual(paper_id, "test_webpage")

        # 3. Retrieve paper from database and verify local file path is saved
        paper_in_db = self.graph_repo.get_paper("test_webpage")
        self.assertIsNotNone(paper_in_db)
        local_path = paper_in_db.file_path
        self.assertTrue(local_path.endswith("test_webpage.md"))
        self.assertTrue(os.path.exists(local_path))

        # Check content is saved
        with open(local_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertEqual(content, "This is mock webpage content in Markdown.")

        # Cleanup file
        if os.path.exists(local_path):
            os.remove(local_path)


if __name__ == "__main__":
    unittest.main()

