"""Tests for Markdown and EPUB parsers, and the Indexer extensions."""

import os
import tempfile
import textwrap
import unittest

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
            from src.parsers.md_parser import parse_markdown
            paper, links, body = parse_markdown(path)
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
            from src.parsers.md_parser import parse_markdown
            _, links, _ = parse_markdown(path)
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
            from src.parsers.md_parser import parse_markdown
            paper, _, _ = parse_markdown(path)
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
            from src.parsers.md_parser import parse_markdown
            paper, _, _ = parse_markdown(path)
            self.assertEqual(paper.title, "Inferred Title")
        finally:
            os.unlink(path)

    def test_stable_id_generation(self):
        """Same title should produce the same ID."""
        path1 = self._write_md("---\ntitle: Stable Title\n---\nContent 1")
        path2 = self._write_md("---\ntitle: Stable Title\n---\nContent 2")
        try:
            from src.parsers.md_parser import parse_markdown
            p1, _, _ = parse_markdown(path1)
            p2, _, _ = parse_markdown(path2)
            self.assertEqual(p1.id, p2.id)
        finally:
            os.unlink(path1)
            os.unlink(path2)


class TestIndexerMarkdownIntegration(unittest.TestCase):
    """Integration test: index a Markdown note end-to-end (no LLM needed)."""

    def setUp(self):
        import tempfile
        from src.repository.sqlite_impl import SQLiteGraphRepository, SQLiteVectorRepository
        from unittest.mock import MagicMock

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


if __name__ == "__main__":
    unittest.main()
