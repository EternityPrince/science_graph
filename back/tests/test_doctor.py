import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from src.models import Concept, Paper, Author, Chunk
from src.repository.sqlite_impl import SQLiteGraphRepository, SQLiteVectorRepository
from src.services.doctor_service import DoctorService, clean_text
from src.cli import app

runner = CliRunner()


class TestDoctor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.graph_repo = SQLiteGraphRepository(self.tmp.name)
        self.vector_repo = SQLiteVectorRepository(self.tmp.name)
        self.doctor_service = DoctorService(self.graph_repo, self.vector_repo)

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_clean_text(self):
        self.assertEqual(clean_text("<think>reasoning</think>Clean Name"), "Clean Name")
        self.assertEqual(clean_text("<thought>reasoning</thought>Clean Name"), "Clean Name")
        self.assertEqual(clean_text("```json\n{\"name\": \"value\"}\n```"), "{\"name\": \"value\"}")
        self.assertEqual(clean_text("\"Wrapping Quotes\""), "Wrapping Quotes")
        self.assertEqual(clean_text("Line 1\n\n\n\nLine 2"), "Line 1\n\nLine 2")
        self.assertEqual(clean_text("   Many   Spaces   "), "Many Spaces")
        self.assertEqual(clean_text("<think>unclosed thought"), "")

    def test_doctor_diagnostics_check_only(self):
        # 1. Save an uncleaned paper
        paper = Paper(
            id="p1",
            title="```json\nTitle with codeblock\n```",
            authors=["\"Author One\"", "<think>x</think>Author Two"],
            abstract="<think>some reasoning</think>This is abstract."
        )
        self.graph_repo.save_paper(paper)

        # 2. Save an uncleaned concept
        concept = Concept(
            id="uncleaned_concept",
            name="\"Machine Learning\"",
            properties={"description": "<think>thought</think>Actual description."}
        )
        self.graph_repo.save_concept(concept)

        # 3. Save an uncleaned chunk
        chunk = Chunk(
            id="p1#0",
            paper_id="p1",
            text_content="```\nChunk content\n```",
            page_number=1,
            embedding=[0.1] * 384
        )
        self.vector_repo.save_chunks([chunk])

        # Run diagnostics in check-only mode
        report = self.doctor_service.run_diagnostics(fix=False)
        
        # Verify stats
        self.assertEqual(report["stats"]["papers_checked"], 1)
        self.assertEqual(report["stats"]["papers_fixed"], 1)
        self.assertEqual(report["stats"]["concepts_checked"], 1)
        self.assertEqual(report["stats"]["concepts_migrated"], 1)
        self.assertEqual(report["stats"]["chunks_checked"], 1)
        self.assertEqual(report["stats"]["chunks_fixed"], 1)

        # Verify database is UNCHANGED (fix=False)
        p = self.graph_repo.get_paper("p1")
        self.assertIn("```json", p.title)
        c = self.graph_repo.get_concept("uncleaned_concept")
        self.assertIsNotNone(c)
        self.assertIn("\"", c.name)

    def test_doctor_diagnostics_fix(self):
        # 1. Create a paper and an author
        paper = Paper(
            id="p1",
            title="```json\nTitle with codeblock\n```",
            authors=["\"Author One\""],
            abstract="<think>some reasoning</think>This is abstract."
        )
        self.graph_repo.save_paper(paper)

        author = Author(
            id="uncleaned_author",
            name="\"Author One\""
        )
        self.graph_repo.save_author(author)
        self.graph_repo.add_edge("uncleaned_author", "p1", "AUTHORED")

        # 2. Save concept and link it
        concept = Concept(
            id="uncleaned_concept",
            name="\"Machine Learning\"",
            properties={"description": "<think>thought</think>Actual description."}
        )
        self.graph_repo.save_concept(concept)
        self.graph_repo.add_edge("p1", "uncleaned_concept", "MENTIONS_CONCEPT")

        # 3. Save chunk
        chunk = Chunk(
            id="p1#0",
            paper_id="p1",
            text_content="```\nChunk content\n```",
            page_number=1,
            embedding=[0.1] * 384
        )
        self.vector_repo.save_chunks([chunk])

        # Run diagnostics in FIX mode
        report = self.doctor_service.run_diagnostics(fix=True)

        # 4. Verify report
        self.assertEqual(report["stats"]["papers_fixed"], 1)
        self.assertEqual(report["stats"]["authors_migrated"], 1)
        self.assertEqual(report["stats"]["concepts_migrated"], 1)
        self.assertEqual(report["stats"]["chunks_fixed"], 1)

        # 5. Verify database corrections
        # Paper
        p = self.graph_repo.get_paper("p1")
        self.assertEqual(p.title, "Title with codeblock")
        self.assertEqual(p.abstract, "This is abstract.")
        self.assertEqual(p.authors, ["Author One"])

        # Author ID should have migrated from 'uncleaned_author' to 'author_one'
        old_author = self.graph_repo.get_author("uncleaned_author")
        self.assertIsNone(old_author)
        new_author = self.graph_repo.get_author("author_one")
        self.assertIsNotNone(new_author)
        self.assertEqual(new_author.name, "Author One")

        # Concept ID should have migrated from 'uncleaned_concept' to 'machine_learning'
        old_concept = self.graph_repo.get_concept("uncleaned_concept")
        self.assertIsNone(old_concept)
        new_concept = self.graph_repo.get_concept("machine_learning")
        self.assertIsNotNone(new_concept)
        self.assertEqual(new_concept.name, "Machine Learning")
        self.assertEqual(new_concept.properties.get("description"), "Actual description.")

        # Edges should be migrated
        # Check author edge
        author_neighbors = self.graph_repo.get_neighbors("author_one")
        self.assertTrue(any(row[3] == "p1" for row in author_neighbors))

        # Check concept edge
        paper_neighbors = self.graph_repo.get_neighbors("p1")
        self.assertTrue(any(row[3] == "machine_learning" for row in paper_neighbors))

        # Chunk text should be corrected
        chunks = self.vector_repo.get_all_chunks()
        self.assertEqual(chunks[0].text_content, "Chunk content")

    @patch("src.cli.get_services")
    def test_cli_doctor_command(self, mock_get_services):
        mock_get_services.return_value = (self.graph_repo, self.vector_repo, MagicMock(), MagicMock())

        # Setup uncleaned concept
        concept = Concept(
            id="uncleaned_concept",
            name="\"Machine Learning\"",
            properties={"description": "<think>thought</think>Actual description."}
        )
        self.graph_repo.save_concept(concept)

        # CLI run: check mode
        result = runner.invoke(app, ["doctor"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Starting Science Graph Database Doctor Diagnostics...", result.stdout)
        self.assertIn("Found 1 anomalies.", result.stdout)

        # CLI run: fix mode
        result_fix = runner.invoke(app, ["doctor", "--fix"])
        self.assertEqual(result_fix.exit_code, 0)
        self.assertIn("Successfully corrected 1 anomalies across all tables!", result_fix.stdout)
