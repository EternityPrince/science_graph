"""Tests for new refinements: concept descriptions, pre-generated summaries, and reindex full command."""

import os
import tempfile
import textwrap
import unittest
from unittest.mock import patch, MagicMock

from typer.testing import CliRunner
from src.cli import app
from src.models import Paper
from src.repository.sqlite_impl import SQLiteGraphRepository, SQLiteVectorRepository
from src.indexer import Indexer
from src.config import config

runner = CliRunner()

class TestRefinements(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()

        self.graph_repo = SQLiteGraphRepository(self.tmp.name)
        self.vector_repo = SQLiteVectorRepository(self.tmp.name)

        self.emb_engine = MagicMock()
        self.emb_engine.get_embeddings.side_effect = lambda texts: [[0.1, 0.2, 0.3] for _ in texts]

        self.llm_engine = MagicMock()
        self.llm_engine.generate_response.return_value = "Mock LLM response."
        self.llm_engine.extract_concepts_and_metadata.return_value = None

        self.indexer = Indexer(self.graph_repo, self.vector_repo, self.emb_engine, self.llm_engine)

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)

    def _write_md(self, content: str) -> str:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        )
        f.write(content)
        f.close()
        return f.name

    def test_get_concept_description_predefined(self):
        # "Deep Learning" is in descriptions in taxonomy.yaml
        descriptions = config.taxonomy.get("descriptions", {})
        desc = self.indexer._extractor.get_concept_description("Deep Learning")
        self.assertEqual(desc, descriptions["Deep Learning"])

        # Case insensitivity test
        desc_lower = self.indexer._extractor.get_concept_description("deep learning")
        self.assertEqual(desc_lower, descriptions["Deep Learning"])

    def test_get_concept_description_llm(self):
        self.llm_engine.generate_response.return_value = "'Custom LLM Definition for AI'"
        desc = self.indexer._extractor.get_concept_description("Custom AI term")
        self.assertEqual(desc, "Custom LLM Definition for AI")
        self.llm_engine.generate_response.assert_called_with(
            "Provide a brief, one-sentence definition of the AI/ML concept or term: 'Custom AI term' in English. Do not write anything else. Keep it under 20 words.\n\nEXCEPT: If the concept name 'Custom AI term' is in Russian (contains Cyrillic characters), provide the definition in Russian instead.",
            task="extraction"
        )

    def test_get_concept_description_fallback(self):
        # Indexer without LLM
        indexer_no_llm = Indexer(self.graph_repo, self.vector_repo, self.emb_engine, llm_engine=None)
        desc = indexer_no_llm._extractor.get_concept_description("Random Unseen Concept")
        self.assertEqual(desc, "A key concept representing 'Random Unseen Concept' within the AI/ML literature.")

    def test_generate_and_save_summary(self):
        self.llm_engine.generate_response.return_value = "This is a brief markdown summary of the research note."
        path = self._write_md(textwrap.dedent("""\
            ---
            title: "Supervised Learning Note"
            ---
            Supervised learning is where you have input variables and an output variable.
            """))
        try:
            paper_id = self.indexer.index_markdown(path)
            paper = self.graph_repo.get_paper(paper_id)
            self.assertIn("summary", paper.properties)
            self.assertEqual(paper.properties["summary"], "This is a brief markdown summary of the research note.")
        finally:
            os.unlink(path)

    def test_reindex_full(self):
        # 1. Index a markdown note
        path = self._write_md(textwrap.dedent("""\
            ---
            title: "Initial Title"
            ---
            Initial body text that is long enough to exceed the minimum fifty characters limit so that a chunk is actually generated and stored.
            """))
        
        try:
            paper_id = self.indexer.index_markdown(path)
            paper = self.graph_repo.get_paper(paper_id)
            self.assertEqual(paper.title, "Initial Title")
            
            # Verify we have chunks saved
            chunks = self.vector_repo.search_similar_chunks([0.1, 0.2, 0.3], limit=10)
            self.assertTrue(len(chunks) > 0)

            # 2. Modify the file content and title
            with open(path, "w", encoding="utf-8") as f:
                f.write(textwrap.dedent("""\
                    ---
                    title: "Updated Title"
                    ---
                    Updated body text with some new information that is also long enough to exceed the fifty characters threshold.
                    """))

            # 3. Call reindex_full
            success = self.indexer.reindex_full(paper_id)
            self.assertTrue(success)

            # 4. Assert changes took effect
            # The old paper node should be deleted because of the ID change
            old_paper = self.graph_repo.get_paper(paper_id)
            self.assertIsNone(old_paper)

            # The new paper should be indexed under the new title
            updated_paper = self.graph_repo.find_paper_by_title("Updated Title")
            self.assertIsNotNone(updated_paper)
            self.assertEqual(updated_paper.title, "Updated Title")
            
        finally:
            os.unlink(path)

    @patch("src.cli.get_services")
    def test_cli_reindex_meta(self, mock_get_services):
        from src.indexer import Indexer
        mock_graph_repo = MagicMock()
        mock_get_services.return_value = (mock_graph_repo, self.vector_repo, self.emb_engine, self.llm_engine)
        
        mock_graph_repo.get_non_placeholder_paper_ids.return_value = ["test_p"]
        mock_paper = Paper(id="test_p", title="Test Paper", authors=[])
        mock_graph_repo.get_paper.return_value = mock_paper
        
        real_indexer = Indexer(mock_graph_repo, self.vector_repo, self.emb_engine, self.llm_engine)
        real_indexer.reindex_metadata = MagicMock(return_value=True)
        
        with patch("src.cli.Indexer", return_value=real_indexer):
            result = runner.invoke(app, ["reindex", "meta", "--missing-authors"])
            self.assertEqual(result.exit_code, 0)
            real_indexer.reindex_metadata.assert_called_once_with("test_p", use_llm=False)
            self.assertIn("Re-indexed 1/1 papers successfully.", result.stdout)

    @patch("src.cli.get_services")
    def test_cli_reindex_full(self, mock_get_services):
        from src.indexer import Indexer
        mock_graph_repo = MagicMock()
        mock_get_services.return_value = (mock_graph_repo, self.vector_repo, self.emb_engine, self.llm_engine)
        
        mock_paper = Paper(id="test_p", title="Test Paper", authors=[])
        mock_graph_repo.get_paper.return_value = mock_paper
        
        real_indexer = Indexer(mock_graph_repo, self.vector_repo, self.emb_engine, self.llm_engine)
        real_indexer.reindex_full = MagicMock(return_value=True)
        
        with patch("src.cli.Indexer", return_value=real_indexer):
            result = runner.invoke(app, ["reindex", "full", "--id", "test_p"])
            self.assertEqual(result.exit_code, 0)
            real_indexer.reindex_full.assert_called_once_with("test_p")
            self.assertIn("Fully re-indexed 1/1 papers successfully.", result.stdout)

    def test_slugify_multilingual(self):
        from src.models import slugify
        self.assertEqual(slugify("трансформер"), "трансформер")
        self.assertEqual(slugify("Привет, Мир!"), "привет_мир")
        self.assertEqual(slugify("Attention mechanism (внимание)"), "attention_mechanism_внимание")

    def test_markdown_parser_standard_links(self):
        from src.parsers.md_parser import MarkdownParser
        path = self._write_md(textwrap.dedent("""\
            ---
            title: "Markdown Links Test"
            ---
            Here is a [[WikiLink]].
            And a standard [Normal Link](other_note.md#section).
            And an external [Web Link](https://google.com/path).
            """))
        try:
            paper, links, body = MarkdownParser().parse(path)
            self.assertIn("WikiLink", links)
            self.assertIn("other_note", links)
            self.assertIn("https://google.com/path", links)
            self.assertEqual(len(links), 3)
        finally:
            os.unlink(path)

    @patch("src.parsers.url_parser.requests.get")
    def test_url_parser_links(self, mock_get):
        from src.parsers.url_parser import UrlParser
        class MockResponse:
            def __init__(self, text, status_code=200):
                self.text = text
                self.status_code = status_code
            def raise_for_status(self):
                pass

        html_content = """
        <html>
            <body>
                <main>
                    <p>Some text with <a href="https://example.com/other-page">other page</a> link.</p>
                    <p>Relative <a href="/relative-path">relative link</a>.</p>
                    <p>Self link <a href="https://example.com/blog/1">self</a>.</p>
                    <p>Mailto link <a href="mailto:test@example.com">email</a>.</p>
                </main>
            </body>
        </html>
        """
        mock_get.return_value = MockResponse(html_content)

        paper, links, _ = UrlParser().parse("https://example.com/blog/1")
        self.assertIn("https://example.com/other-page", links)
        self.assertIn("https://example.com/relative-path", links)
        # Should filter out self-links and mailto links
        self.assertNotIn("https://example.com/blog/1", links)
        self.assertNotIn("mailto:test@example.com", links)

    def test_indexer_tag_merging(self):
        path = self._write_md(textwrap.dedent("""\
            ---
            title: "Tag Merging Note"
            tags: ["frontmatter_tag", "shared_tag"]
            ---
            Body text that is long enough to exceed the minimum limit.
            """))
        try:
            self.indexer._extractor.extract = MagicMock(return_value=MagicMock(
                authors=[],
                concepts=[],
                tags=["extracted_tag", "shared_tag"],
                via_llm=False
            ))
            
            paper_id = self.indexer.index_markdown(path)
            paper = self.graph_repo.get_paper(paper_id)
            
            tags = paper.properties.get("tags")
            self.assertIn("frontmatter_tag", tags)
            self.assertIn("shared_tag", tags)
            self.assertIn("extracted_tag", tags)
            self.assertEqual(len(tags), 3)
            
            edges = self.graph_repo.get_all_edges()
            has_tag1 = any(e[0] == paper_id and e[1] == "frontmatter_tag" and e[2] == "HAS_TAG" for e in edges)
            has_tag2 = any(e[0] == paper_id and e[1] == "shared_tag" and e[2] == "HAS_TAG" for e in edges)
            has_tag3 = any(e[0] == paper_id and e[1] == "extracted_tag" and e[2] == "HAS_TAG" for e in edges)
            
            self.assertTrue(has_tag1)
            self.assertTrue(has_tag2)
            self.assertTrue(has_tag3)
        finally:
            os.unlink(path)

    @patch("src.parsers.url_parser.requests.get")
    def test_indexer_webpage_pipeline(self, mock_get):
        class MockResponse:
            def __init__(self, text, status_code=200):
                self.text = text
                self.status_code = status_code
            def raise_for_status(self):
                pass

        html_content = """
        <html>
            <head>
                <title>Web Title</title>
            </head>
            <body>
                <main>
                    <h1>Web Title</h1>
                    <p>Main content with <a href="https://example.com/target-page">target link</a>.</p>
                </main>
            </body>
        </html>
        """
        mock_get.return_value = MockResponse(html_content)
        
        self.indexer._extractor.extract = MagicMock(return_value=MagicMock(
            authors=[],
            concepts=[],
            tags=[],
            via_llm=False
        ))

        paper_id = self.indexer.index_url("https://example.com/blog/2")
        paper = self.graph_repo.get_paper(paper_id)
        self.assertEqual(paper.title, "Web Title")
        
        from src.models import slugify
        expected_target_id = slugify("example.com/target-page")
        
        edges = self.graph_repo.get_all_edges()
        has_edge = any(e[0] == paper_id and e[1] == expected_target_id and e[2] == "RELATED_TO" for e in edges)
        self.assertTrue(has_edge)

