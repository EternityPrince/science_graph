import unittest
from unittest.mock import MagicMock
from pathlib import Path
import tempfile
import json
import os

from src.services.visualizer import generate_html_graph
from src.repository.base import GraphRepository

class TestVisualizer(unittest.TestCase):
    def setUp(self):
        self.graph_repo = MagicMock(spec=GraphRepository)
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = Path(self.temp_dir) / "graph.html"

    def tearDown(self):
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        try:
            os.rmdir(self.temp_dir)
        except OSError:
            pass

    def test_empty_graph_raises_value_error(self):
        """Ensures generate_html_graph raises ValueError when the graph has no nodes."""
        self.graph_repo.get_all_nodes.return_value = []
        self.graph_repo.get_all_edges.return_value = []

        with self.assertRaises(ValueError) as context:
            generate_html_graph(self.graph_repo, self.output_path)

        self.assertIn("Knowledge graph is empty", str(context.exception))

    def test_populated_graph_generation(self):
        """Tests generate_html_graph with various node and edge types to cover all color and styling paths."""
        # 1. Setup various nodes rows
        # Format: (node_id, label, props_json_str)
        nodes_rows = [
            # Paper node with note type
            ("p1", "Paper", json.dumps({"title": "A Note Title", "source_type": "note", "year": 2024})),
            # Paper node with book type
            ("p2", "Paper", json.dumps({"title": "A very long book title that exceeds twenty-five characters", "source_type": "book", "year": 2023})),
            # Paper node with webpage type
            ("p3", "Paper", json.dumps({"title": "Web Title", "source_type": "webpage"})),
            # Paper node with default paper type
            ("p4", "Paper", json.dumps({"title": "Paper Title", "source_type": "paper"})),
            # Paper node with unknown source type
            ("p5", "Paper", json.dumps({"title": "Unknown Title", "source_type": "unknown"})),
            # Author node
            ("a1", "Author", json.dumps({"name": "john doe"})),
            # Concept node (not a tag)
            ("c1", "Concept", json.dumps({"name": "neural_networks", "is_tag": False})),
            # Concept node (tag)
            ("c2", "Concept", json.dumps({"name": "artificial_intelligence", "is_tag": True})),
            # Other node type
            ("o1", "OtherLabel", json.dumps({"description": "somethin else"})),
        ]
        
        # 2. Setup various edges rows
        # Format: (src_id, tgt_id, edge_type, timestamp)
        edges_rows = [
            ("a1", "p1", "AUTHORED", "2026-06-17"),
            ("p1", "c1", "MENTIONS_CONCEPT", "2026-06-17"),
            ("p1", "p2", "CITES", "2026-06-17"),
            ("p2", "c2", "HAS_TAG", "2026-06-17"),
            ("o1", "p3", "OTHER_EDGE", "2026-06-17"),
        ]

        self.graph_repo.get_all_nodes.return_value = nodes_rows
        self.graph_repo.get_all_edges.return_value = edges_rows

        # Call the visualizer
        generate_html_graph(self.graph_repo, self.output_path)

        # Assert output file is created
        self.assertTrue(self.output_path.exists())
        self.assertGreater(self.output_path.stat().st_size, 0)

        # Verify output content contains all vis_nodes and vis_edges JSON strings
        content = self.output_path.read_text(encoding="utf-8")
        self.assertIn("A Note Title", content)
        self.assertIn("A very long book title...", content) # truncated title
        self.assertIn("Web Title", content)
        self.assertIn("Neural Networks", content)
        self.assertIn("Artificial Intelligence", content)
        self.assertIn("AUTHORED", content)
        self.assertIn("MENTIONS_CONCEPT", content)
        self.assertIn("CITES", content)
        self.assertIn("HAS_TAG", content)
        self.assertIn("OTHER_EDGE", content)

if __name__ == "__main__":
    unittest.main()
