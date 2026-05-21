"""Tests for new refinements: concept descriptions, pre-generated summaries, and reindex full command."""

import os
import tempfile
import textwrap
import unittest
import json
import sqlite3
from unittest.mock import patch, MagicMock

from typer.testing import CliRunner
from src.cli import app
from src.models import Paper, Concept
from src.repository.sqlite_impl import SQLiteGraphRepository, SQLiteVectorRepository
from src.indexer import Indexer, COMMON_CONCEPT_DESCRIPTIONS

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
        # "Deep Learning" is in COMMON_CONCEPT_DESCRIPTIONS
        desc = self.indexer._get_concept_description("Deep Learning")
        self.assertEqual(desc, COMMON_CONCEPT_DESCRIPTIONS["Deep Learning"])

        # Case insensitivity test
        desc_lower = self.indexer._get_concept_description("deep learning")
        self.assertEqual(desc_lower, COMMON_CONCEPT_DESCRIPTIONS["Deep Learning"])

    def test_get_concept_description_llm(self):
        self.llm_engine.generate_response.return_value = "'Custom LLM Definition for AI'"
        desc = self.indexer._get_concept_description("Custom AI term")
        self.assertEqual(desc, "Custom LLM Definition for AI")
        self.llm_engine.generate_response.assert_called_with(
            "Provide a brief, one-sentence definition/description of the AI/ML concept or term: 'Custom AI term'. Do not write anything else. Keep it under 20 words."
        )

    def test_get_concept_description_fallback(self):
        # Indexer without LLM
        indexer_no_llm = Indexer(self.graph_repo, self.vector_repo, self.emb_engine, llm_engine=None)
        desc = indexer_no_llm._get_concept_description("Random Unseen Concept")
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
    @patch("src.cli.sqlite3.connect")
    def test_cli_reindex_meta(self, mock_connect, mock_get_services):
        mock_indexer_instance = MagicMock()
        mock_get_services.return_value = (self.graph_repo, self.vector_repo, self.emb_engine, self.llm_engine)
        
        # Mock database connection to return a paper
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        row = MagicMock()
        row.__getitem__ = lambda s, k: {"id": "test_p", "properties": '{"title": "Test Paper", "authors": []}'}[k]
        mock_conn.execute.return_value.fetchall.return_value = [row]
        
        with patch("src.cli.Indexer", return_value=mock_indexer_instance):
            mock_indexer_instance.reindex_metadata.return_value = True
            result = runner.invoke(app, ["reindex", "meta", "--missing-authors"])
            self.assertEqual(result.exit_code, 0)
            mock_indexer_instance.reindex_metadata.assert_called_once_with("test_p", use_llm=False)
            self.assertIn("Re-indexed 1/1 papers successfully.", result.stdout)

    @patch("src.cli.get_services")
    @patch("src.cli.sqlite3.connect")
    def test_cli_reindex_full(self, mock_connect, mock_get_services):
        mock_indexer_instance = MagicMock()
        mock_get_services.return_value = (self.graph_repo, self.vector_repo, self.emb_engine, self.llm_engine)
        
        # Mock database connection
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        # Mock checks for paper existence and candidate selection
        exists_row = MagicMock()
        exists_row.__getitem__ = lambda s, k: {"id": "test_p"}[k]
        mock_conn.execute.return_value.fetchone.return_value = exists_row
        
        with patch("src.cli.Indexer", return_value=mock_indexer_instance):
            mock_indexer_instance.reindex_full.return_value = True
            result = runner.invoke(app, ["reindex", "full", "--id", "test_p"])
            self.assertEqual(result.exit_code, 0)
            mock_indexer_instance.reindex_full.assert_called_once_with("test_p")
            self.assertIn("Fully re-indexed 1/1 papers successfully.", result.stdout)
