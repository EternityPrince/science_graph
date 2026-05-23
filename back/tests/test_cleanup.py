import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from src.models import Concept, Paper
from src.repository.sqlite_impl import SQLiteGraphRepository
from src.cli import app

runner = CliRunner()

class TestCleanup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.graph_repo = SQLiteGraphRepository(self.tmp.name)

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_repo_cleanup_orphaned_concepts(self):
        # 1. Create a paper node
        paper = Paper(id="p1", title="Paper 1", authors=[], year=2024, abstract="", file_path="path.pdf")
        self.graph_repo.save_paper(paper)

        # 2. Create a connected concept node
        concept1 = Concept(id="concept_connected", name="Connected Concept")
        self.graph_repo.save_concept(concept1)
        self.graph_repo.add_edge("p1", "concept_connected", "RELATED_TO")

        # 3. Create an orphaned concept node
        concept2 = Concept(id="concept_orphaned", name="Orphaned Concept")
        self.graph_repo.save_concept(concept2)

        # 4. Perform cleanup
        deleted = self.graph_repo.cleanup_orphaned_concepts()
        self.assertEqual(deleted, 1)

        # 5. Verify connected concept still exists, orphaned is deleted
        self.assertIsNotNone(self.graph_repo.get_neighbors("p1", max_depth=1))
        # Check nodes left in DB
        with self.graph_repo._get_connection() as conn:
            concept_ids = [row[0] for row in conn.execute("SELECT id FROM nodes WHERE label='Concept'").fetchall()]
        
        self.assertIn("concept_connected", concept_ids)
        self.assertNotIn("concept_orphaned", concept_ids)

    @patch("src.cli.get_services")
    def test_cli_cleanup_command(self, mock_get_services):
        # Mock get_services to return our setup repository
        mock_get_services.return_value = (self.graph_repo, MagicMock(), MagicMock(), MagicMock())

        # 1. Create nodes in setup
        paper = Paper(id="p1", title="Paper 1", authors=[], year=2024, abstract="", file_path="path.pdf")
        self.graph_repo.save_paper(paper)
        concept1 = Concept(id="concept_connected", name="Connected Concept")
        self.graph_repo.save_concept(concept1)
        self.graph_repo.add_edge("p1", "concept_connected", "RELATED_TO")

        concept2 = Concept(id="concept_orphaned", name="Orphaned Concept")
        self.graph_repo.save_concept(concept2)

        # 2. Run CLI command
        result = runner.invoke(app, ["cleanup"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Successfully cleaned up 1 orphaned concept nodes.", result.stdout)

        # 3. Double check DB
        with self.graph_repo._get_connection() as conn:
            concept_ids = [row[0] for row in conn.execute("SELECT id FROM nodes WHERE label='Concept'").fetchall()]
        self.assertIn("concept_connected", concept_ids)
        self.assertNotIn("concept_orphaned", concept_ids)
